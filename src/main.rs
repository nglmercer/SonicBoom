// Hide the console window on Windows (no-op on Linux/macOS)
#![windows_subsystem = "windows"]

mod admin;
mod api;
mod auth;
mod config;
mod error;
mod logging;
mod security;
mod web;

use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::RwLock;
use tower_http::{
    timeout::TimeoutLayer,
    trace::{DefaultMakeSpan, DefaultOnFailure, DefaultOnRequest, DefaultOnResponse, TraceLayer},
};
use tower_sessions::{MemoryStore, SessionManagerLayer};

use admin::{handlers::AdminState, lockout::LoginAttemptTracker};
use api::{gate::InferenceGate, rate_limit::RateLimiter};
use auth::store::TokenStore;
use config::AppConfig;
use sonicboom::tts;
#[cfg(feature = "playback")]
use tts::queue::AudioManager;
use tts::{ModelStatus, download, model::ModelHandle};

/// Type alias for the audio manager when feature is disabled.
#[cfg(not(feature = "playback"))]
type AudioManager = ();

#[derive(Clone)]
pub struct AppState {
    pub model_status: Arc<RwLock<ModelStatus>>,
    pub token_store: Arc<TokenStore>,
    pub config: Arc<AppConfig>,
    /// Audio manager for server-side playback. `None` when the `playback` feature is disabled
    /// or when initialization fails.
    pub audio_manager: Arc<Option<AudioManager>>,
    /// Bounded admission control for model inference (see [`InferenceGate`]).
    pub inference_gate: Arc<InferenceGate>,
    /// Per-token sliding-window rate limiter for expensive endpoints.
    pub rate_limiter: Arc<RateLimiter>,
}

#[cfg(all(feature = "gui", not(target_os = "linux")))]
use tao::event_loop::{ControlFlow, EventLoopBuilder};
#[cfg(all(feature = "gui", not(target_os = "linux")))]
use tray_icon::{
    TrayIconBuilder,
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
};

fn main() -> anyhow::Result<()> {
    // Load environment variables from .env file
    dotenvy::dotenv().ok();

    let config = Arc::new(match AppConfig::from_env() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Configuration error: {e}");
            std::process::exit(1);
        }
    });

    // Validate configuration
    if let Err(e) = config.validate() {
        eprintln!("Configuration error: {e}");
        std::process::exit(1);
    }

    // Fail closed when the filesystem queue root is unusable.
    #[cfg(feature = "playback")]
    if let Some(dir) = config.allowed_audio_dir.as_deref() {
        match std::path::Path::new(dir).canonicalize() {
            Ok(canonical) if canonical.is_dir() => {}
            Ok(_) => {
                eprintln!("Configuration error: ALLOWED_AUDIO_DIR is not a directory");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Configuration error: cannot access ALLOWED_AUDIO_DIR: {e}");
                std::process::exit(1);
            }
        }
    }

    // Initialize logging
    logging::init(
        &config.log_dir,
        &config.log_level,
        config.log_to_file,
        config.log_to_stdout,
    );

    // Log startup
    logging::log_startup(config.port, &config.log_dir);

    tracing::info!(admin_id = %config.admin_id, "Admin credentials loaded");

    #[cfg(feature = "gui")]
    {
        run_gui(config)
    }

    #[cfg(not(feature = "gui"))]
    {
        // --- Start the Tokio server on the current thread ---
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(run_server(config))
    }
}

#[cfg(feature = "gui")]
fn run_gui(config: Arc<AppConfig>) -> anyhow::Result<()> {
    // --- Start the Tokio server on a dedicated background thread ---
    // This prevents the tray event loop and the tokio runtime from blocking each other.
    let config_clone = Arc::clone(&config);
    let (server_err_tx, server_err_rx) = std::sync::mpsc::channel::<String>();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
        rt.block_on(async move {
            if let Err(e) = run_server(config_clone).await {
                tracing::error!("Server error: {e}");
                let _ = server_err_tx.send(format!("{e}"));
            }
        });
    });

    // --- Tray icon setup (runs on the main thread) ---
    #[cfg(target_os = "linux")]
    {
        run_linux_tray(server_err_rx)
    }
    #[cfg(not(target_os = "linux"))]
    {
        run_tao_tray(server_err_rx)
    }
}

/// Open the executable's directory (where data/models live) in the OS file
/// browser, falling back to the working directory. Shared by both tray
/// backends' "Open File Directory" menu items.
#[cfg(feature = "gui")]
fn open_file_directory() {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .or_else(|| std::env::current_dir().ok());

    if let Some(dir) = dir {
        if let Err(e) = open::that_detached(&dir) {
            tracing::error!("Failed to open directory {:?}: {e}", dir);
        }
    } else {
        tracing::error!("Could not determine a directory to open");
    }
}

/// Linux tray via StatusNotifierItem (ksni, pure Rust over D-Bus).
///
/// This backend exists so the Linux GUI build compiles no GTK/glib code:
/// the tao/tray-icon Linux backends pull the unmaintained GTK3 bindings
/// (glib 0.18, RUSTSEC-2024-0429). Requires a StatusNotifierWatcher host
/// (KDE, GNOME with appindicator support, etc.); without one the tray is
/// skipped and the HTTP server keeps running headless.
#[cfg(all(feature = "gui", target_os = "linux"))]
fn run_linux_tray(server_err_rx: std::sync::mpsc::Receiver<String>) -> anyhow::Result<()> {
    use ksni::blocking::TrayMethods;

    struct SonicBoomTray {
        /// ARGB32 pixmap (width, height, bytes) decoded from the embedded PNG.
        icon: Option<(i32, i32, Vec<u8>)>,
    }

    impl ksni::Tray for SonicBoomTray {
        fn id(&self) -> String {
            "sonicboom".to_string()
        }

        fn title(&self) -> String {
            "SonicBoom".to_string()
        }

        fn icon_pixmap(&self) -> Vec<ksni::Icon> {
            self.icon
                .as_ref()
                .map(|(width, height, data)| ksni::Icon {
                    width: *width,
                    height: *height,
                    data: data.clone(),
                })
                .into_iter()
                .collect()
        }

        fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
            vec![
                ksni::MenuItem::Standard(ksni::menu::StandardItem {
                    label: "Open File Directory".to_string(),
                    activate: Box::new(|_| open_file_directory()),
                    ..Default::default()
                }),
                ksni::MenuItem::Separator,
                ksni::MenuItem::Standard(ksni::menu::StandardItem {
                    label: "Close Process".to_string(),
                    activate: Box::new(|_| std::process::exit(0)),
                    ..Default::default()
                }),
            ]
        }
    }

    // Embed icon from assets/icon.png at compile time, converted to the
    // ARGB32 pixmap the SNI specification expects.
    const ICON_BYTES: &[u8] = include_bytes!("../assets/icon.png");
    let icon = match image::load_from_memory(ICON_BYTES).map(|i| i.into_rgba8()) {
        Ok(image) => {
            let (width, height) = image.dimensions();
            let mut argb = Vec::with_capacity((width * height * 4) as usize);
            for pixel in image.pixels() {
                let [r, g, b, a] = pixel.0;
                argb.extend_from_slice(&[a, r, g, b]);
            }
            Some((width as i32, height as i32, argb))
        }
        Err(e) => {
            tracing::warn!("Failed to load embedded tray icon: {e}");
            None
        }
    };

    match (SonicBoomTray { icon }).spawn() {
        Ok(handle) => {
            tracing::info!("System tray running (StatusNotifierItem)");
            loop {
                if let Ok(err_msg) = server_err_rx.try_recv() {
                    tracing::error!("Server thread died: {err_msg}");
                }
                if handle.is_closed() {
                    tracing::info!("Tray service closed; shutting down");
                    return Ok(());
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
        Err(e) => {
            // No D-Bus session or SNI host (e.g. headless/SSH): keep serving
            // HTTP on the background thread instead of exiting.
            tracing::error!("System tray unavailable ({e}); running headless");
            loop {
                if let Ok(err_msg) = server_err_rx.try_recv() {
                    tracing::error!("Server thread died: {err_msg}");
                }
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
        }
    }
}

/// macOS/Windows tray via tao + tray-icon (native backends; no GTK there).
#[cfg(all(feature = "gui", not(target_os = "linux")))]
fn run_tao_tray(server_err_rx: std::sync::mpsc::Receiver<String>) -> anyhow::Result<()> {
    let event_loop = EventLoopBuilder::new().build();
    let tray_menu = Menu::new();

    let open_dir_item = MenuItem::new("Open File Directory", true, None);
    let quit_item = MenuItem::new("Close Process", true, None);

    tray_menu.append_items(&[&open_dir_item, &PredefinedMenuItem::separator(), &quit_item])?;

    let mut tray_icon = None;

    // Embed icon from assets/icon.png at compile time
    const ICON_BYTES: &[u8] = include_bytes!("../assets/icon.png");
    let icon = match image::load_from_memory(ICON_BYTES).map(|i| i.into_rgba8()) {
        Ok(image) => {
            let (width, height) = image.dimensions();
            let rgba = image.into_raw();
            tray_icon::Icon::from_rgba(rgba, width, height).ok()
        }
        Err(e) => {
            tracing::warn!("Failed to load embedded tray icon: {e}");
            None
        }
    };

    event_loop.run(move |event, _, control_flow| {
        // Use Poll so OS continuously drives the loop, ensuring menu events
        // are never missed. This is a tray-only app so CPU usage is negligible.
        *control_flow = ControlFlow::Poll;

        // Check if the server thread reported a fatal error
        if let Ok(err_msg) = server_err_rx.try_recv() {
            tracing::error!("Server thread died: {err_msg}");
            // Optionally exit, or just log — here we keep the tray alive
            // so the user can still interact with the app.
        }

        match event {
            tao::event::Event::NewEvents(tao::event::StartCause::Init) => {
                let mut builder = TrayIconBuilder::new()
                    .with_menu(Box::new(tray_menu.clone()))
                    .with_tooltip("SonicBoom");

                if let Some(i) = icon.clone() {
                    builder = builder.with_icon(i);
                }

                tray_icon = Some(builder.build().unwrap());
            }

            tao::event::Event::MainEventsCleared => {
                // Poll menu events every iteration
                while let Ok(menu_event) = MenuEvent::receiver().try_recv() {
                    if menu_event.id == open_dir_item.id() {
                        open_file_directory();
                    } else if menu_event.id == quit_item.id() {
                        let _ = tray_icon.take();
                        *control_flow = ControlFlow::Exit;
                        std::process::exit(0);
                    }
                }
            }
            _ => (),
        }
    });
}

async fn run_server(config: Arc<AppConfig>) -> anyhow::Result<()> {
    // Fail closed on credential-store errors: a malformed or unreadable
    // token file must never silently become an empty store. A missing file
    // is initialized safely by `TokenStore::load`.
    let token_store = Arc::new(
        TokenStore::load(&config.token_store_path)
            .await
            .map_err(|e| {
                tracing::error!("Refusing to start: {e}");
                e
            })?,
    );

    // Resolve model trust early so a bad revision/manifest fails fast,
    // before any download is attempted.
    let (model_revision, expected_trust) =
        tts::download::resolve_trust(&config.model_revision, config.model_hashes_path.as_deref())?;
    let download_limits = tts::download::DownloadLimits::new(
        config.model_download_connect_timeout_secs,
        config.model_download_timeout_secs,
    );

    if config.enable_sample_token {
        tracing::warn!(
            "ENABLE_SAMPLE_TOKEN is on: the development SAMPLE_TOKEN is accepted. \
             Never enable this in production."
        );
    }

    let model_status = Arc::new(RwLock::new(ModelStatus::Idle));

    #[cfg(feature = "playback")]
    let audio_manager = match AudioManager::new(config.max_playback_queue_items) {
        Ok(manager) => Arc::new(Some(manager)),
        Err(e) => {
            tracing::warn!("Failed to initialize audio manager: {}", e);
            Arc::new(None)
        }
    };

    // Prepare the temp playback directory (playback builds): ensure safe
    // ownership/permissions and sweep `<uuid>.wav` leftovers from an
    // unclean shutdown. Non-fatal — request-time checks fail closed.
    #[cfg(feature = "playback")]
    match tts::queue::prepare_temp_dir(std::path::Path::new(&config.temp_audio_dir)) {
        Ok(0) => {}
        Ok(reaped) => {
            tracing::info!("Swept {reaped} leftover temp audio file(s) on startup");
        }
        Err(e) => {
            tracing::warn!("Temp audio dir unavailable: {e}");
        }
    }

    #[cfg(not(feature = "playback"))]
    let audio_manager = Arc::new(None);

    let inference_gate = Arc::new(InferenceGate::new(
        config.max_concurrent_inference,
        config.max_pending_inference,
    ));
    let rate_limiter = Arc::new(RateLimiter::new(
        config.tts_rate_limit_requests,
        config.tts_rate_limit_window_secs,
    ));

    let app_state = AppState {
        model_status: Arc::clone(&model_status),
        token_store: Arc::clone(&token_store),
        config: Arc::clone(&config),
        audio_manager,
        inference_gate,
        rate_limiter,
    };

    let lockout = Arc::new(LoginAttemptTracker::default());
    let admin_state = AdminState {
        token_store: Arc::clone(&token_store),
        lockout: Arc::clone(&lockout),
        config: Arc::clone(&config),
    };

    // Hardened admin session cookie: HttpOnly, SameSite=Strict, optional
    // Secure (enable via COOKIE_SECURE=true when serving HTTPS), and an
    // inactivity expiry. Only the session id is stored client-side.
    let session_store = MemoryStore::default();
    let session_layer = SessionManagerLayer::new(session_store)
        .with_name("sonicboom_admin")
        .with_path("/")
        .with_http_only(true)
        .with_same_site(tower_sessions::cookie::SameSite::Strict)
        .with_secure(config.cookie_secure)
        .with_expiry(tower_sessions::Expiry::OnInactivity(
            time::Duration::seconds(config.admin_session_expiry_secs),
        ));

    let request_timeout = config.request_timeout_secs;

    let app = axum::Router::new()
        .merge(web::router(app_state.clone()))
        .merge(api::router(app_state.clone()))
        .merge(admin::router(admin_state))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&config),
            security::security_headers,
        ))
        .layer(session_layer)
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(request_timeout),
        ))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new())
                .on_request(DefaultOnRequest::new())
                .on_response(DefaultOnResponse::new())
                .on_failure(DefaultOnFailure::new()),
        );

    let port = config.port;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    let model_cache_dir = config.model_cache_dir.clone();
    let hf_token = config.hf_token.clone();
    let model_status_bg = Arc::clone(&model_status);
    tokio::spawn(async move {
        *model_status_bg.write().await = ModelStatus::Downloading { progress: 0.0 };
        let status_for_progress = Arc::clone(&model_status_bg);
        match download::download_models_with_options(
            std::path::Path::new(&model_cache_dir),
            hf_token.as_deref(),
            &model_revision,
            &expected_trust,
            &download_limits,
            move |progress| {
                if let Ok(mut status) = status_for_progress.try_write() {
                    *status = ModelStatus::Downloading { progress };
                }
            },
        )
        .await
        {
            Ok(paths) => {
                *model_status_bg.write().await = ModelStatus::Loading;
                match tokio::task::spawn_blocking(move || ModelHandle::load(&paths)).await {
                    Ok(Ok(handle)) => {
                        tracing::info!("Model ready.");
                        *model_status_bg.write().await = ModelStatus::Ready(Arc::new(handle));
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Model load error: {e}");
                        *model_status_bg.write().await = ModelStatus::Failed(e.to_string());
                    }
                    Err(e) => {
                        tracing::error!("Model load task panic: {e}");
                        *model_status_bg.write().await = ModelStatus::Failed(e.to_string());
                    }
                }
            }
            Err(e) => {
                tracing::error!("Model download error: {e}");
                *model_status_bg.write().await = ModelStatus::Failed(e.to_string());
            }
        }
    });

    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Graceful shutdown on Ctrl+C / SIGTERM
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    tracing::info!("Server shut down gracefully.");
    Ok(())
}

/// Waits for Ctrl+C or SIGTERM signal to trigger graceful shutdown.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("Received Ctrl+C, shutting down..."),
        () = terminate => tracing::info!("Received SIGTERM, shutting down..."),
    }
}
