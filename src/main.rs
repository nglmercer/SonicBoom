// Hide the console window on Windows (no-op on Linux/macOS)
#![windows_subsystem = "windows"]

mod admin;
mod api;
mod auth;
mod config;
mod error;
mod logging;
mod tts;
mod web;

use std::{net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;
use tower_sessions::{MemoryStore, SessionManagerLayer};
use tower_http::trace::{TraceLayer, DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, DefaultOnFailure};

use admin::{handlers::AdminState, lockout::LoginAttemptTracker};
use auth::store::TokenStore;
use config::AppConfig;
use tts::{download, model::ModelHandle, queue::AudioManager, ModelStatus};

#[derive(Clone)]
pub struct AppState {
    pub model_status: Arc<RwLock<ModelStatus>>,
    pub token_store: Arc<TokenStore>,
    pub config: Arc<AppConfig>,
    pub audio_manager: Arc<Option<AudioManager>>,
}

use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
    TrayIconBuilder,
};
use tao::event_loop::{ControlFlow, EventLoopBuilder};

fn main() -> anyhow::Result<()> {
    // Load environment variables from .env file
    dotenvy::dotenv().ok();

    let config = Arc::new(AppConfig::from_env());

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

    // --- Start the Tokio server on a dedicated background thread ---
    // This prevents the tao event loop and the tokio runtime from blocking each other.
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

    // --- Tray icon setup (runs on the main thread, required by most OS windowing systems) ---
    let event_loop = EventLoopBuilder::new().build();
    let tray_menu = Menu::new();

    let open_dir_item = MenuItem::new("Open File Directory", true, None);
    let quit_item = MenuItem::new("Close Process", true, None);

    tray_menu.append_items(&[
        &open_dir_item,
        &PredefinedMenuItem::separator(),
        &quit_item,
    ])?;

    let mut tray_icon = None;

    // Use an icon from assets/icon.png
    let icon_path = std::path::Path::new("assets/icon.png");
    let icon = if icon_path.exists() {
        match image::open(icon_path).map(|i| i.into_rgba8()) {
            Ok(image) => {
                let (width, height) = image.dimensions();
                let rgba = image.into_raw();
                tray_icon::Icon::from_rgba(rgba, width, height).ok()
            }
            Err(e) => {
                tracing::warn!("Failed to load tray icon image: {e}");
                None
            }
        }
    } else {
        None
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
                        // Prefer the executable's directory (where data/models live),
                        // fall back to CWD if unavailable.
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
    let token_store = Arc::new(
        TokenStore::load(&config.token_store_path)
            .await
            .unwrap_or_else(|e| {
                tracing::warn!("Could not load token store: {e}. Starting fresh.");
                panic!("Failed to initialize token store: {e}");
            }),
    );

    let model_status = Arc::new(RwLock::new(ModelStatus::Idle));

    let audio_manager = match AudioManager::new() {
        Ok(manager) => Arc::new(Some(manager)),
        Err(e) => {
            tracing::warn!("Failed to initialize audio manager: {}", e);
            Arc::new(None)
        }
    };

    let app_state = AppState {
        model_status: Arc::clone(&model_status),
        token_store: Arc::clone(&token_store),
        config: Arc::clone(&config),
        audio_manager,
    };

    let lockout = Arc::new(LoginAttemptTracker::default());
    let admin_state = AdminState {
        token_store: Arc::clone(&token_store),
        lockout: Arc::clone(&lockout),
        config: Arc::clone(&config),
    };

    let session_store = MemoryStore::default();
    let session_layer = SessionManagerLayer::new(session_store);

    let app = axum::Router::new()
        .merge(web::router(app_state.clone()))
        .merge(api::router(app_state.clone()))
        .merge(admin::router(admin_state))
        .layer(session_layer)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new())
                .on_request(DefaultOnRequest::new())
                .on_response(DefaultOnResponse::new())
                .on_failure(DefaultOnFailure::new())
        );

    let port = config.port;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    let model_cache_dir = config.model_cache_dir.clone();
    let hf_token = config.hf_token.clone();
    let model_status_bg = Arc::clone(&model_status);
    tokio::spawn(async move {
        *model_status_bg.write().await = ModelStatus::Downloading { progress: 0.0 };
        match download::download_models(&model_cache_dir, hf_token.as_deref(), Arc::clone(&model_status_bg)).await {
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
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}
