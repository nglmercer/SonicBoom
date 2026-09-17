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
    // Desktop first-run bootstrap (gui builds only): sidecar `.env`,
    // exe-adjacent data dirs, generated admin password. Headless server
    // builds keep fail-closed behavior (refuse without explicit password).
    #[cfg(feature = "gui")]
    ensure_gui_env();

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

/// First-run bootstrap for desktop (`gui`) builds so a basic user can
/// double-click the app with no terminal and no exported environment.
///
/// Headless server builds intentionally fail closed when `SONICBOOM_ADMIN_PW`
/// is missing (see `config::AppConfig::validate`). A GUI app has no console
/// (Windows hides it via `#![windows_subsystem]`), so that failure looks
/// like an instant silent crash. For `gui` builds only this function:
/// 1. loads a sidecar `.env` next to the executable (in addition to the
///    CWD `.env`), so a double-clicked app finds its config;
/// 2. defaults data dirs (`TOKEN_STORE_PATH`, `MODEL_CACHE_DIR`, `LOG_DIR`,
///    `TEMP_AUDIO_DIR`, plus `ALLOWED_AUDIO_DIR` with `playback`) to
///    exe-adjacent locations when unset, creating them best-effort;
/// 3. generates a random 256-bit admin password on first run and persists
///    it to that sidecar `.env` (`0600` on Unix) when `SONICBOOM_ADMIN_PW`
///    is still missing. This is a unique per-install secret, never a
///    well-known default, so the password policy still holds. The value
///    itself is never logged; only the file path is reported.
///
/// Explicitly-set environment variables always win: nothing here overrides
/// an existing non-empty value.
#[cfg(feature = "gui")]
fn ensure_gui_env() {
    use std::env;

    // Base directory: executable's parent, falling back to the CWD.
    let base_dir: std::path::PathBuf = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .or_else(|| env::current_dir().ok())
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    // Load configs: CWD `.env` first, then the exe-adjacent sidecar (which
    // only fills in variables the process env / CWD file did not set).
    dotenvy::dotenv().ok();
    let sidecar_env = base_dir.join(".env");
    dotenvy::from_path(&sidecar_env).ok();

    // Set `name` only when currently unset or blank.
    let set_default = |name: &str, value: String| {
        let missing = env::var(name)
            .map(|v| v.trim().is_empty())
            .unwrap_or(true);
        if missing {
            // Safe here: called on the main thread before any other threads
            // are spawned (`ensure_gui_env` runs first in `main`).
            unsafe { env::set_var(name, value) };
        }
    };

    // Exe-adjacent data dirs so double-click runs don't depend on the CWD
    // (which for a GUI launch is unpredictable and may not be writable).
    let dir_default = |name: &str, leaf: &str, create: bool| {
        let path = base_dir.join(leaf);
        if create {
            let _ = std::fs::create_dir_all(&path);
        }
        if let Some(s) = path.to_str() {
            set_default(name, s.to_string());
        }
    };

    dir_default("MODEL_CACHE_DIR", "models", true);
    dir_default("LOG_DIR", "logs", true);
    dir_default("TEMP_AUDIO_DIR", "temp_audio", true);
    // Token store is a file, not a dir: default the path, don't create it
    // here (`TokenStore::load` creates it with restrictive permissions).
    if let Some(s) = base_dir.join("tokens.json").to_str() {
        set_default("TOKEN_STORE_PATH", s.to_string());
    }
    #[cfg(feature = "playback")]
    dir_default("ALLOWED_AUDIO_DIR", "audio", true);

    ensure_gui_port(&base_dir);

    // First run: no admin password anywhere -> generate + persist.
    let pw_missing = env::var("SONICBOOM_ADMIN_PW")
        .map(|v| v.trim().is_empty())
        .unwrap_or(true);
    if !pw_missing {
        return;
    }

    let generated = crate::auth::token::generate_token_value();
    // Persist to the sidecar `.env` (fall back to `./.env`), updating an
    // existing assignment in place so re-runs keep a single entry.
    let mut target = sidecar_env.clone();
    if persist_env_value(&target, "SONICBOOM_ADMIN_PW", &generated).is_err() {
        target = std::path::PathBuf::from(".env");
        if persist_env_value(&target, "SONICBOOM_ADMIN_PW", &generated).is_err() {
            // Last resort: in-memory only for this run (the password changes
            // next restart). Still better than a silent exit for a GUI user.
            eprintln!(
                "Warning: could not write generated SONICBOOM_ADMIN_PW to '{}' or './.env'; \
                 using an in-memory password for this run only.",
                sidecar_env.display()
            );
        }
    }
    // Safe here: still on the main thread before the server thread spawns.
    unsafe { env::set_var("SONICBOOM_ADMIN_PW", &generated) };
    eprintln!(
        "Generated a random admin password and saved it to '{}'. \
         Open that file to sign in to /admin (the tray menu 'Open File Directory' shows the folder). \
         The password itself is never logged.",
        target.display()
    );
}

/// Insert or replace `key=value` in the `.env` file at `path`, creating it
/// (and parents) as needed. Restricts permissions to `0600` on Unix.
#[cfg(feature = "gui")]
fn persist_env_value(
    path: &std::path::Path,
    key: &str,
    value: &str,
) -> std::io::Result<()> {
    use std::io::Write as _;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let assignment = format!("{key}={value}");
    let merged = match std::fs::read_to_string(path) {
        Ok(existing) => {
            let mut replaced = false;
            let mut lines: Vec<String> = Vec::new();
            for line in existing.lines() {
                let trimmed = line.trim_start();
                let key_line = trimmed == key
                    || trimmed.starts_with(&format!("{key}="))
                    || trimmed.starts_with(&format!("{key} "));
                if !replaced && key_line {
                    // Preserve an `export ` prefix style if the user used it.
                    if trimmed.starts_with("export ") {
                        lines.push(format!("export {assignment}"));
                    } else {
                        lines.push(assignment.clone());
                    }
                    replaced = true;
                } else {
                    lines.push(line.to_string());
                }
            }
            if !replaced {
                lines.push(assignment);
            }
            let mut out = lines.join("\n");
            out.push('\n');
            out
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => format!("{assignment}\n"),
        Err(e) => return Err(e),
    };
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(path)?;
    file.write_all(merged.as_bytes())?;
    file.sync_all()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Instance record so tools and double-clicks can discover the running
/// server. Rewritten on every gui start; a stale record is harmless because
/// the next start re-probes the recorded port before trusting it.
#[cfg(feature = "gui")]
const GUI_LOCK_FILE: &str = "sonicboom.lock";
/// How far past `PORT` to scan for a free port before giving up.
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
        let req = format!(
            "GET /health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
        );
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

/// Desktop port handling: avoid the cryptic `Address already in use
/// (os error 98)` crash that is routine on common ports like 3000.
///
/// - A live lock record whose port answers HTTP 200 means *our* server is
///   already up: exit pointing at it (double-clicking twice must not spawn
///   a second server).
/// - Otherwise a taken `PORT` falls through to a nearby free port (gui
///   only; headless builds keep fail-closed binding), recorded together
///   with the pid in `<base>/sonicboom.lock` for discovery.
/// The chosen port is exported back into `PORT` for `AppConfig` and always
/// reported loudly: the service must never silently move ports.
#[cfg(feature = "gui")]
fn ensure_gui_port(base_dir: &std::path::Path) {
    use std::env;

    let requested: u16 = env::var("PORT")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(17842);
    let lock_path = base_dir.join(GUI_LOCK_FILE);
    let env_path = base_dir.join(".env");

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
                 Set a different port with PORT=<free-port> in '{}', or stop the other application.",
                env_path.display()
            );
        } else {
            eprintln!(
                "No free port in {requested}..={} (scanned {GUI_PORT_SCAN_RANGE} past PORT). \
                 Set a different port with PORT=<free-port> in '{}'.",
                requested.saturating_add(GUI_PORT_SCAN_RANGE as u16),
                env_path.display()
            );
        }
        std::process::exit(1);
    };
    if port != requested {
        eprintln!(
            "Port {requested} is in use by another application; using {port} instead. \
             The actual port is recorded in '{}'. To pin a port, set PORT=<port> in '{}'.",
            lock_path.display(),
            env_path.display()
        );
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
    // Safe here: still on the main thread before the server thread spawns.
    unsafe { env::set_var("PORT", port.to_string()) };
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
                    // Fatal: token store / trust / bind failure. A lingering
                    // tray with a dead server is worse than a visible exit.
                    tracing::error!("Server thread died: {err_msg}");
                    eprintln!("Server error: {err_msg}");
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

        // A server-thread error is fatal (token store / trust / bind
        // failure): exit visibly instead of lingering as a tray with a
        // dead server behind it.
        if let Ok(err_msg) = server_err_rx.try_recv() {
            tracing::error!("Server thread died: {err_msg}");
            eprintln!("Server error: {err_msg}");
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
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AddrInUse {
            anyhow::anyhow!(
                "failed to bind {addr}: {e}. \
                 Another SonicBoom instance may be running, or another app uses port {port}. \
                 Set a different port with PORT=<free-port> \
                 (dev: project-root '.env' or `export PORT=...`; \
                 desktop gui: '.env' next to the executable; \
                 docker: `-e PORT=...` with a matching `-p` publish)."
            )
        } else {
            anyhow::anyhow!("failed to bind {addr}: {e}")
        }
    })?;

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

#[cfg(all(test, feature = "gui"))]
mod gui_env_tests {
    use super::persist_env_value;

    fn temp_env_path(tag: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "sonicboom-gui-env-test-{}-{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::create_dir_all(&dir);
        dir.join(".env")
    }

    #[test]
    fn creates_new_env_file_with_assignment() {
        let path = temp_env_path("create");
        let _ = std::fs::remove_file(&path);
        persist_env_value(&path, "SONICBOOM_ADMIN_PW", "secret-value").unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "SONICBOOM_ADMIN_PW=secret-value\n");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replaces_existing_assignment_and_keeps_other_lines() {
        let path = temp_env_path("replace");
        std::fs::write(
            &path,
            "PORT=17842\nSONICBOOM_ADMIN_PW=old-value\nLOG_LEVEL=info\n",
        )
        .unwrap();
        persist_env_value(&path, "SONICBOOM_ADMIN_PW", "new-value").unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            content,
            "PORT=17842\nSONICBOOM_ADMIN_PW=new-value\nLOG_LEVEL=info\n"
        );
        // Re-running keeps a single entry (no duplicates).
        persist_env_value(&path, "SONICBOOM_ADMIN_PW", "newer-value").unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            content.matches("SONICBOOM_ADMIN_PW=").count(),
            1,
            "duplicate entries: {content:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn generated_password_satisfies_policy() {
        let pw = crate::auth::token::generate_token_value();
        assert!(pw.chars().count() >= crate::config::MIN_ADMIN_PASSWORD_LEN);
        assert!(pw.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn lock_content_round_trips() {
        use super::{GUI_LOCK_FILE, parse_lock_content};
        assert_eq!(GUI_LOCK_FILE, "sonicboom.lock");
        let text = format!("pid={} port={}\n", std::process::id(), 17843);
        assert_eq!(parse_lock_content(&text), Some((std::process::id(), 17843)));
    }

    #[test]
    fn lock_parse_rejects_garbage() {
        use super::parse_lock_content;
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
        use super::{PortProbe, probe_port, select_gui_port};
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
