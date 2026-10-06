// Hide the console window on Windows for release desktop (`gui`) builds
// only (no-op on Linux/macOS). Headless server builds always keep the
// console so startup errors and logs stay visible; debug `gui` builds keep
// it too so `cargo run` output is visible during development.
#![cfg_attr(
    all(feature = "gui", not(debug_assertions)),
    windows_subsystem = "windows"
)]

mod admin;
mod api;
mod auth;
mod cli;
mod config;
mod error;
mod logging;
mod runtime;
mod security;
mod server;
mod setup;
mod web;

use std::sync::Arc;
use tokio::sync::RwLock;

use admin::{handlers::AdminState, lockout::LoginAttemptTracker};
use api::{gate::InferenceGate, rate_limit::RateLimiter};
use auth::store::TokenStore;
use config::{ConfigManager, paths, secrets};
use runtime::ModelService;
use runtime::reconfigure::reconfiguration_loop;
use runtime::subsystems::{InferenceGateManager, RateLimiterManager};
use server::ApiServerManager;
use sonicboom::tts;
use tts::ModelStatus;
#[cfg(feature = "playback")]
use tts::queue::AudioManager;

/// Type alias for the audio manager when feature is disabled.
#[cfg(not(feature = "playback"))]
type AudioManager = ();

#[derive(Clone)]
pub struct AppState {
    pub model_status: Arc<RwLock<ModelStatus>>,
    pub token_store: Arc<TokenStore>,
    /// Central configuration: every subsystem reads the
    /// live configuration through this manager.
    pub config: Arc<ConfigManager>,
    /// Audio manager for server-side playback. `None` when the `playback` feature is disabled
    /// or when initialization fails.
    pub audio_manager: Arc<Option<AudioManager>>,
    /// Bounded admission control for model inference (see [`InferenceGate`]).
    pub inference_gate: Arc<InferenceGateManager>,
    /// Per-token token-bucket rate limiter for expensive endpoints.
    pub rate_limiter: Arc<RateLimiterManager>,
    /// Model lifecycle owner (download, load, reload).
    pub model_service: Arc<ModelService>,
    /// Secret store (admin password, HuggingFace token).
    pub secrets: Arc<dyn secrets::SecretStore>,
}

#[cfg(all(feature = "gui", not(target_os = "linux")))]
use tao::event_loop::{ControlFlow, EventLoopBuilder};
#[cfg(all(feature = "gui", not(target_os = "linux")))]
use tray_icon::{
    TrayIconBuilder,
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
};

#[cfg(feature = "gui")]
static GUI_URL: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

#[cfg(feature = "gui")]
fn update_gui_url(address: std::net::SocketAddr) {
    let address = if address.ip().is_unspecified() {
        std::net::SocketAddr::new(
            if address.is_ipv6() {
                "::1".parse().unwrap()
            } else {
                "127.0.0.1".parse().unwrap()
            },
            address.port(),
        )
    } else {
        address
    };
    if let Ok(mut url) = GUI_URL.write() {
        *url = Some(format!("http://{address}"));
    }
}

#[cfg(feature = "gui")]
fn open_application() {
    if let Some(url) = GUI_URL.read().ok().and_then(|url| url.clone()) {
        let _ = open::that_detached(url);
    }
}

#[cfg(feature = "gui")]
fn show_startup_error(message: &str) {
    let escape = |value: &str| {
        value
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    };
    let path =
        std::env::temp_dir().join(format!("sonicboom-recovery-{}.html", uuid::Uuid::new_v4()));
    let config_path = paths::resolve_config_path();
    let page = format!(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>SonicBoom configuration error</title><h1>SonicBoom could not start</h1><pre>{}</pre><p>Configuration file: <code>{}</code></p><p>Correct the file and launch SonicBoom again. The existing configuration was preserved.</p></html>",
        escape(message),
        escape(&config_path.display().to_string())
    );
    if std::fs::write(&path, page).is_ok() {
        let _ = open::that(path);
    }
}

/// Outcome of first-run preparation.
enum PrepareOutcome {
    /// An existing (or migrated) configuration is ready.
    Ready,
    /// A bootstrap configuration was created; first-run
    /// setup is pending.
    FirstRun,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // CLI mode: `sonicboom config …` operates on the
    // configuration file through the same
    // `ConfigManager` the server uses, then exits.
    if args.len() > 1 && args[1] == "config" {
        let rt = tokio::runtime::Runtime::new()?;
        let outcome = rt.block_on(cli::run_config(&args[2..]));
        let code = match outcome {
            Ok(()) => 0,
            Err(code) => code,
        };
        std::process::exit(code);
    }

    // First-run detection: migrate a legacy `.env` or
    // bootstrap a fresh `config.toml`.
    let config_path = paths::resolve_config_path();
    match prepare_config(&config_path) {
        Ok(_outcome) => {}
        Err(e) => {
            eprintln!("Configuration error: {e}");
            #[cfg(feature = "gui")]
            show_startup_error(&e);
            std::process::exit(1);
        }
    }

    #[cfg(feature = "gui")]
    {
        run_gui(config_path)
    }

    #[cfg(not(feature = "gui"))]
    {
        // --- Start the Tokio server on the current thread ---
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(run_server(config_path))
    }
}

/// First-run preparation (spec §9/§11):
///
/// - `config.toml` exists → nothing to do.
/// - Legacy `.env` exists → migrate known values into
///   `config.toml` (secrets stay in the environment /
///   secret store).
/// - Neither → bootstrap a fresh `config.toml` with
///   `setup_complete = false` so the first-run wizard
///   runs.
fn prepare_config(config_path: &std::path::Path) -> Result<PrepareOutcome, String> {
    if config_path.is_file() {
        return Ok(PrepareOutcome::Ready);
    }
    // Legacy `.env`: the working-directory file first,
    // then the executable-adjacent sidecar (portable
    // desktop installs).
    let mut env_candidates = vec![std::path::PathBuf::from(".env")];
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        env_candidates.push(exe_dir.join(".env"));
    }
    for env_path in &env_candidates {
        if env_path.is_file() {
            match config::migration::migrate_if_needed(config_path, env_path) {
                Ok(Some(summary)) => {
                    eprintln!("{summary}");
                    return Ok(PrepareOutcome::Ready);
                }
                Ok(None) => continue,
                Err(e) => {
                    return Err(format!("migrating {} failed: {e}", env_path.display()));
                }
            }
        }
    }
    // No configuration anywhere: bootstrap a first-run
    // configuration. The server starts in bootstrap mode
    // (loopback-only, no API bearer auth) and serves the
    // setup wizard until setup completes.
    // The platform data directories the defaults point
    // at may not exist yet; create the common ones.
    let data_dir = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(paths::data_dir);
    for leaf in ["models", "logs", "temp_audio", "audio"] {
        let _ = std::fs::create_dir_all(data_dir.join(leaf));
    }
    // Point the default configuration at the platform
    // data directories (the compiled defaults are
    // relative paths like ./models).
    let mut bootstrap = config::model::AppConfig::default();
    bootstrap.model.cache_dir = data_dir.join("models");
    bootstrap.paths.logs = data_dir.join("logs").to_string_lossy().to_string();
    bootstrap.paths.temp_audio = data_dir.join("temp_audio").to_string_lossy().to_string();
    bootstrap.paths.audio = Some(data_dir.join("audio").to_string_lossy().to_string());
    bootstrap.paths.token_store = data_dir.join("tokens.json").to_string_lossy().to_string();
    // Existing headless deployments with credentials remain unattended.
    if !cfg!(feature = "gui") && std::env::var("SONICBOOM_ADMIN_PW").is_ok() {
        config::validation::validate_admin_password(
            &std::env::var("SONICBOOM_ADMIN_PW").unwrap_or_default(),
        )?;
        bootstrap.setup_complete = true;
        let mut effective = bootstrap.clone();
        config::loader::apply_environment(&mut effective, &mut std::collections::HashMap::new())
            .map_err(|e| e.to_string())?;
        config::validation::validate(&effective)?;
    }
    config::writer::atomic_write_config_blocking(config_path, &bootstrap)
        .map_err(|e| format!("could not create {}: {e}", config_path.display()))?;
    Ok(PrepareOutcome::FirstRun)
}

/// Desktop port handling: avoid the cryptic `Address already in use
/// (os error 98)` crash that is routine on common ports like 3000.
///
/// - A live lock record whose port answers HTTP 200 means *our* server is
///   already up: exit pointing at it (double-clicking twice must not spawn
///   a second server).
/// - Otherwise a taken `PORT` falls through to a nearby free port (gui
///   only; headless builds keep fail-closed binding), recorded together
///   with the pid in `<data>/sonicboom.lock` for discovery.
/// The chosen port is applied as a runtime override (the file is not
/// rewritten) and always reported loudly: the service must never silently
/// move ports.
#[cfg(feature = "gui")]
async fn ensure_gui_port(config: &ConfigManager, data_dir: &std::path::Path) {
    let requested: u16 = config.get().await.server.port;
    let lock_path = data_dir.join(GUI_LOCK_FILE);

    if let Ok(text) = std::fs::read_to_string(&lock_path) {
        if let Some((old_pid, old_port)) = parse_lock_content(&text) {
            if probe_port(old_port) == PortProbe::HttpOk {
                eprintln!(
                    "SonicBoom is already running (pid {old_pid}) at http://127.0.0.1:{old_port} \
                     (see '{}'). Stop it first, or delete that file if it is stale and restart.",
                    lock_path.display()
                );
                std::process::exit(1);
            }
        }
    }

    let Some(port) = select_gui_port(requested) else {
        if probe_port(requested) == PortProbe::HttpOk {
            eprintln!(
                "Port {requested} is already in use by another application (it answers HTTP on /health). \
                 Set a different port with server.port in '{}'.",
                config.path().display()
            );
        } else {
            eprintln!(
                "No free port in {requested}..={} (scanned {GUI_PORT_SCAN_RANGE} past server.port). \
                 Set a different port with server.port in '{}'.",
                requested.saturating_add(GUI_PORT_SCAN_RANGE as u16),
                config.path().display()
            );
        }
        std::process::exit(1);
    };
    if port != requested {
        eprintln!(
            "Port {requested} is in use by another application; using {port} instead. \
             The actual port is recorded in '{}'. To pin a port, set server.port in '{}'.",
            lock_path.display(),
            config.path().display()
        );
        // Runtime-only override: the operator's file is
        // not rewritten by a transient port conflict.
        let update = config.update_runtime(None, |c| c.server.port = port).await;
        if let Err(e) = update {
            eprintln!("Warning: could not apply detected port {port}: {e}");
        }
    }
    // A stale record is harmless: the next start re-probes before trusting it.
    let content = format!("pid={} port={port}\n", std::process::id());
    if std::fs::write(&lock_path, content).is_err() {
        eprintln!(
            "Warning: could not write instance lock to '{}'; \
             double-click detection and port discovery will be unavailable.",
            lock_path.display()
        );
    }
}

/// Instance record so tools and double-clicks can discover the running
/// server. Rewritten on every gui start; a stale record is harmless because
/// the next start re-probes the recorded port before trusting it.
#[cfg(feature = "gui")]
const GUI_LOCK_FILE: &str = "sonicboom.lock";
/// How far past `server.port` to scan for a free port before giving up.
#[cfg(feature = "gui")]
const GUI_PORT_SCAN_RANGE: u32 = 20;

#[cfg(feature = "gui")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortProbe {
    /// Something answers HTTP 200 on `/health` (ours or a foreign app).
    HttpOk,
    /// Nothing answers and a test bind succeeds.
    Free,
    /// Nothing answers, but the port cannot be bound (a non-HTTP owner).
    Taken,
}

/// Parse a `sonicboom.lock` body (`pid=<n> port=<n>`, whitespace-separated,
/// extra tokens ignored). Returns `None` when either field is missing or
/// malformed.
#[cfg(feature = "gui")]
fn parse_lock_content(text: &str) -> Option<(u32, u16)> {
    let mut pid = None;
    let mut port = None;
    for token in text.split_whitespace() {
        if let Some(v) = token.strip_prefix("pid=") {
            pid = v.trim().parse().ok();
        } else if let Some(v) = token.strip_prefix("port=") {
            port = v.trim().parse().ok();
        }
    }
    Some((pid?, port?))
}

/// Classify a port: HTTP-200 responder, bindable, or otherwise taken.
#[cfg(feature = "gui")]
fn probe_port(port: u16) -> PortProbe {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpStream};
    use std::time::Duration;

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    if let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(300)) {
        let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
        let req =
            format!("GET /health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
        if stream.write_all(req.as_bytes()).is_ok() {
            let mut buf = [0u8; 1024];
            let mut head = Vec::new();
            while head.len() < 512 {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => head.extend_from_slice(&buf[..n]),
                    Err(_) => break,
                }
            }
            let text = String::from_utf8_lossy(&head);
            if text.starts_with("HTTP/1.0 200") || text.starts_with("HTTP/1.1 200") {
                return PortProbe::HttpOk;
            }
        }
        return PortProbe::Taken;
    }
    // Nothing listening: confirm the port is actually bindable (a non-TCP
    // owner, permissions, etc. can still refuse it).
    match std::net::TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], port))) {
        Ok(_) => PortProbe::Free,
        Err(_) => PortProbe::Taken,
    }
}

/// First bindable port from `requested` upward (at most `GUI_PORT_SCAN_RANGE`
/// past it). Returns `None` when the whole range is unusable.
#[cfg(feature = "gui")]
fn select_gui_port(requested: u16) -> Option<u16> {
    let end = u32::from(requested) + GUI_PORT_SCAN_RANGE;
    (u32::from(requested)..=end.min(65535))
        .map(|p| p as u16)
        .find(|p| probe_port(*p) == PortProbe::Free)
}

#[cfg(feature = "gui")]
fn run_gui(config_path: std::path::PathBuf) -> anyhow::Result<()> {
    // --- Start the Tokio server on a dedicated background thread ---
    // This prevents the tray event loop and the tokio runtime from blocking each other.
    let (server_err_tx, server_err_rx) = std::sync::mpsc::channel::<String>();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
        rt.block_on(async move {
            if let Err(e) = run_server(config_path).await {
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

/// Open the platform data directory (where config.toml, models, and logs
/// live) in the OS file browser, falling back to the working directory.
/// Shared by both tray backends' "Open File Directory" menu items.
#[cfg(feature = "gui")]
fn open_file_directory() {
    let dir = paths::resolve_config_path()
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(paths::data_dir);

    if let Err(e) = open::that_detached(&dir) {
        tracing::error!("Failed to open directory {:?}: {e}", dir);
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
                    label: "Open SonicBoom".to_string(),
                    activate: Box::new(|_| open_application()),
                    ..Default::default()
                }),
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
                    // Fatal: token store / trust / bind failure. A lingering
                    // tray with a dead server is worse than a visible exit.
                    tracing::error!("Server thread died: {err_msg}");
                    eprintln!("Server error: {err_msg}");
                    show_startup_error(&err_msg);
                    std::process::exit(1);
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
                    eprintln!("Server error: {err_msg}");
                    show_startup_error(&err_msg);
                    std::process::exit(1);
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

    let open_app_item = MenuItem::new("Open SonicBoom", true, None);
    tray_menu.append(&open_app_item)?;
    let open_dir_item = MenuItem::new("Open File Directory", true, None);
    let quit_item = MenuItem::new("Close Process", true, None);

    tray_menu.append_items(&[&open_dir_item, &PredefinedMenuItem::separator(), &quit_item])?;

    let mut tray_icon = None;

    // Embed icon from assets/icon.png at compile time
    const ICON_BYTES: &[u8] = include_bytes!("../assets/icon.png");
    let icon = match image::load_from_memory(ICON_BYTES).map(|i| i.into_rgba8()) {
        Ok(image) => {
            let rgba = image.into_raw();
            tray_icon::Icon::from_rgba(rgba, image.width(), image.height()).ok()
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

        // A server-thread error is fatal (token store / trust / bind
        // failure): exit visibly instead of lingering as a tray with a
        // dead server behind it.
        if let Ok(err_msg) = server_err_rx.try_recv() {
            tracing::error!("Server thread died: {err_msg}");
            eprintln!("Server error: {err_msg}");
            show_startup_error(&err_msg);
            std::process::exit(1);
        }

        match event {
            tao::event::Event::NewEvents(tao::event::StartCause::Init) => {
                let mut builder = TrayIconBuilder::new()
                    .with_menu(Box::new(tray_menu.clone()))
                    .with_tooltip("SonicBoom");

                if let Some(i) = icon.clone() {
                    builder = builder.with_icon(i);
                }

                // Never panic here: tray creation can fail (e.g. no tray
                // host), and a panic is an undebuggable crash for a basic
                // user. Fall back to running the server without an icon.
                match builder.build() {
                    Ok(built) => {
                        tray_icon = Some(built);
                    }
                    Err(e) => {
                        tracing::error!(
                            "System tray unavailable ({e}); running without a tray icon"
                        );
                    }
                }
            }

            tao::event::Event::MainEventsCleared => {
                // Poll menu events every iteration
                while let Ok(menu_event) = MenuEvent::receiver().try_recv() {
                    if menu_event.id == open_app_item.id() {
                        open_application();
                    } else if menu_event.id == open_dir_item.id() {
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

async fn run_server(config_path: std::path::PathBuf) -> anyhow::Result<()> {
    // Load + validate the configuration (bootstrap
    // overrides for first-run are applied inside
    // `ConfigManager::load`).
    let config = Arc::new(
        ConfigManager::load(config_path.clone())
            .await
            .map_err(|e| {
                tracing::error!("Configuration error: {e}");
                anyhow::anyhow!("{e}")
            })?,
    );

    // The platform data directory (config.toml, secrets,
    // models, logs, audio) is the root for lock files and
    // default data locations.
    let data_dir = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(paths::data_dir);

    #[cfg(feature = "gui")]
    ensure_gui_port(&config, &data_dir).await;

    // Secrets live outside config.toml.
    let secrets: Arc<dyn secrets::SecretStore> =
        Arc::new(secrets::FileSecretStore::new(data_dir.join("secrets.json")));

    let current = config.get().await;

    // Fail closed on completed installs without a valid
    // admin password (headless/server compatibility).
    // First-run (bootstrap) mode defers password creation
    // to the setup wizard.
    if current.setup_complete {
        let admin_pw = secrets::load_admin_password(secrets.as_ref())
            .map_err(|e| anyhow::anyhow!("secret store error: {e}"))?
            .unwrap_or_default();
        config::validation::validate_admin_password(&admin_pw).map_err(|e| {
            tracing::error!("Configuration error: {e}");
            anyhow::anyhow!("{e}")
        })?;
    }

    // Initialize logging
    logging::init(
        &current.paths.logs,
        &current.logging.level,
        current.logging.filter.as_deref(),
        current.logging.to_file,
        current.logging.to_stdout,
    );

    // Log startup
    logging::log_startup(current.server.port, &current.paths.logs);

    tracing::info!(
        admin_id = %current.admin.username,
        auth_mode = ?current.server.auth_mode,
        bind = %current.server.bind,
        "Configuration loaded"
    );

    // Fail closed on credential-store errors: a malformed or unreadable
    // token file must never silently become an empty store. A missing file
    // is initialized safely by `TokenStore::load`.
    let token_store = Arc::new(TokenStore::load(&current.paths.token_store).await.map_err(
        |e| {
            tracing::error!("Refusing to start: {e}");
            e
        },
    )?);

    // Resolve model trust early so a bad revision/manifest fails fast,
    // before any download is attempted.
    let hashes_path = current
        .model
        .hashes_path
        .as_deref()
        .map(|p| p.to_string_lossy().to_string());
    let (_model_revision, _expected_trust) =
        tts::download::resolve_trust(&current.model.revision, hashes_path.as_deref())?;
    let download_limits = tts::download::DownloadLimits::new(
        current.model.download_connect_timeout_secs,
        current.model.download_timeout_secs,
    );
    let _ = download_limits;

    if current.admin.enable_sample_token {
        tracing::warn!(
            "admin.enable_sample_token is on: the development SAMPLE_TOKEN is accepted. \
             Never enable this in production."
        );
    }

    let model_status = Arc::new(RwLock::new(ModelStatus::Idle));
    let model_service = Arc::new(ModelService::new(Arc::clone(&model_status)));

    #[cfg(feature = "playback")]
    let audio_manager = match AudioManager::with_output_device(
        current.audio.max_playback_queue_items,
        current.audio.output_device.clone(),
    ) {
        Ok(manager) => Arc::new(Some(manager)),
        Err(e) => {
            tracing::warn!("Failed to initialize audio manager: {}", e);
            Arc::new(None)
        }
    };

    #[cfg(feature = "playback")]
    if let Some(audio) = audio_manager.as_ref() {
        if let Err(error) = audio.set_volume(current.audio.volume).await {
            config
                .record_apply_failure(
                    "audio.volume",
                    serde_json::json!(current.audio.volume),
                    serde_json::json!(1.0),
                    &error.to_string(),
                )
                .await;
        }
    }

    // Prepare the temp playback directory (playback builds): ensure safe
    // ownership/permissions and sweep `<uuid>.wav` leftovers from an
    // unclean shutdown. Non-fatal — request-time checks fail closed.
    #[cfg(feature = "playback")]
    match tts::queue::prepare_temp_dir(std::path::Path::new(&current.paths.temp_audio)) {
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

    let inference_gate = Arc::new(InferenceGateManager::new(InferenceGate::new(
        current.inference.max_concurrent,
        current.inference.max_pending,
    )));
    let rate_limiter = Arc::new(RateLimiterManager::new(RateLimiter::with_burst(
        current.rate_limit.requests,
        current.rate_limit.window_secs,
        current.rate_limit.burst,
    )));

    let app_state = AppState {
        model_status: Arc::clone(&model_status),
        token_store: Arc::clone(&token_store),
        config: Arc::clone(&config),
        audio_manager,
        inference_gate,
        rate_limiter,
        model_service: Arc::clone(&model_service),
        secrets: Arc::clone(&secrets),
    };

    let lockout = Arc::new(LoginAttemptTracker::default());
    let admin_state = AdminState {
        token_store: Arc::clone(&token_store),
        lockout: Arc::clone(&lockout),
        config: Arc::clone(&config),
        secrets: Arc::clone(&secrets),
    };

    // Independently restartable HTTP listener.
    let server = Arc::new(ApiServerManager::new(app_state.clone(), admin_state));
    // Model download + load runs in the background.
    let hf_token = secrets::load_hf_token(secrets.as_ref())
        .map_err(|e| anyhow::anyhow!("secret store error: {e}"))?;
    model_service.start_initial_load(&current.model, hf_token);

    // Runtime reconfiguration: every committed configuration
    // change is applied to the live subsystems here.
    let reconfigurator = Arc::new(runtime::RuntimeReconfigurator::new(
        Arc::clone(&config),
        Arc::clone(&app_state.audio_manager),
        Arc::clone(&server),
        Arc::clone(&model_service),
        Arc::clone(&app_state.rate_limiter),
        Arc::clone(&app_state.inference_gate),
        Arc::clone(&secrets),
    ));
    tokio::spawn(reconfiguration_loop(
        reconfigurator,
        current.clone(),
        config.subscribe(),
    ));

    // Hot reload: watch config.toml for external edits.
    let _watcher = config::watcher::start_watching(Arc::clone(&config));
    server.start().await?;
    #[cfg(feature = "gui")]
    if !current.setup_complete {
        open_application();
    }

    // Wait for Ctrl+C / SIGTERM, then stop the listener.
    shutdown_signal().await;
    server.stop().await?;

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

#[cfg(all(test, feature = "gui"))]
mod gui_tests {
    use super::{GUI_LOCK_FILE, PortProbe, parse_lock_content, probe_port, select_gui_port};

    #[test]
    fn lock_content_round_trips() {
        assert_eq!(GUI_LOCK_FILE, "sonicboom.lock");
        let text = format!("pid={} port={}\n", std::process::id(), 17843);
        assert_eq!(parse_lock_content(&text), Some((std::process::id(), 17843)));
    }

    #[test]
    fn lock_parse_rejects_garbage() {
        for bad in [
            "",
            "pid=abc port=17842\n",
            "pid=123\n",
            "port=17842\n",
            "pid=123 port=not-a-port\n",
            "pid=123 port=70000\n",
        ] {
            assert_eq!(parse_lock_content(bad), None, "accepted {bad:?}");
        }
    }

    #[test]
    fn occupied_port_is_skipped_by_selection() {
        // Hold a port so the scanner must skip it.
        let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let occupied = held.local_addr().unwrap().port();
        assert_ne!(
            probe_port(occupied),
            PortProbe::Free,
            "held port {occupied} looked free"
        );
        let picked = select_gui_port(occupied).expect("a free port must exist nearby");
        assert_ne!(picked, occupied, "scanner did not skip held port");
        assert_eq!(probe_port(picked), PortProbe::Free);
        drop(held);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_config_bootstraps_a_missing_file() {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-prepare-test-{}-{}",
            std::process::id(),
            "bootstrap"
        ));
        let _ = std::fs::create_dir_all(&dir);
        let config_path = dir.join("config.toml");
        let outcome = prepare_config(&config_path).expect("bootstrap must succeed");
        assert!(matches!(outcome, PrepareOutcome::FirstRun));
        let config_path2 = config_path.clone();
        let parsed = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|text| config::loader::parse_toml(&text).ok())
            .expect("bootstrap file must parse");
        assert!(!parsed.setup_complete);
        // Platform data dirs are referenced, not relative defaults.
        assert!(
            !parsed.model.cache_dir.to_string_lossy().starts_with("./"),
            "bootstrap should use platform dirs: {:?}",
            parsed.model.cache_dir
        );
        // Idempotent: a second run sees the existing file.
        assert!(matches!(
            prepare_config(&config_path2).expect("second run must succeed"),
            PrepareOutcome::Ready
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn bootstrap_mode_forces_loopback_local_auth() {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-bootstrap-{}-{}",
            std::process::id(),
            "auth"
        ));
        let _ = std::fs::create_dir_all(&dir);
        let config_path = dir.join("config.toml");
        // A first-run config with an unsafe bind/auth, as if
        // hand-edited: bootstrap must override both in memory.
        // Written raw (not through the validating atomic
        // writer) to simulate an external edit.
        std::fs::write(
            &config_path,
            "setup_complete = false\n\n[server]\nbind = \"0.0.0.0\"\nauth_mode = \"none\"\n",
        )
        .unwrap();
        let manager = ConfigManager::load(config_path.clone()).await.unwrap();
        let config = manager.get().await;
        assert!(config.server.bind.is_loopback());
        assert_eq!(config.server.auth_mode, config::model::AuthMode::Local);
        assert!(!config.setup_complete);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
