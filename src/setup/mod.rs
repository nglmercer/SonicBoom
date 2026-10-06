//! First-run setup wizard (spec §7/§10–§15).
//!
//! Before `setup_complete` is true the server runs in
//! bootstrap mode (loopback-only, no API bearer auth;
//! see [`crate::config::manager`]). The wizard is a
//! browser flow served at `/setup`:
//!
//! 1. Usage mode (local / LAN / advanced)
//! 2. Audio output device (enumerated from the
//!    running audio subsystem)
//! 3. Model settings
//! 4. Network summary + admin credentials
//!
//! Completing the wizard validates everything, stores
//! the admin password in the secret store (never in
//! `config.toml`), and commits the configuration
//! through the central [`ConfigManager`].

use axum::{
    Json,
    extract::{ConnectInfo, State},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
};
use serde::Deserialize;
use std::net::SocketAddr;
use tower_sessions::Session;

use crate::admin::session as admin_session;
use crate::config::{AuthMode, secrets};
use crate::{AppState, auth::AuthenticatedToken};

/// `GET /setup` — the wizard page.
///
/// Redirects to the admin page once setup is done.
pub async fn get_setup(_token: AuthenticatedToken, State(state): State<AppState>) -> Response {
    if state.config.get().await.setup_complete {
        return axum::response::Redirect::to("/").into_response();
    }
    (
        [(axum::http::header::CACHE_CONTROL, "no-store")],
        Html(WIZARD_HTML),
    )
        .into_response()
}

/// `GET /setup/status` — wizard state.
pub async fn get_setup_status(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let config = state.config.get().await;
    Json(serde_json::json!({
        "setup_complete": config.setup_complete,
        "audio": config.audio,
        "model": {
            "cache_dir": config.model.cache_dir,
            "inference_steps": config.model.inference_steps,
        },
        "server": {
            "bind": config.server.bind.to_string(),
            "port": config.server.port,
            "auth_mode": match config.server.auth_mode {
                AuthMode::Local => "local",
                AuthMode::Token => "token",
                AuthMode::None => "none",
            },
        },
    }))
}

/// `GET /setup/audio-devices` — enumerate output
/// devices through the audio subsystem.
#[cfg(feature = "playback")]
pub async fn get_setup_audio_devices(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    let audio_manager = match state.audio_manager.as_ref().as_ref() {
        Some(manager) => manager,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "success": false,
                    "message": "Audio playback system not initialized",
                })),
            )
                .into_response();
        }
    };
    match audio_manager.output_devices().await {
        Ok(list) => (StatusCode::OK, Json(list)).into_response(),
        Err(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "success": false,
                "message": "Audio device enumeration failed",
            })),
        )
            .into_response(),
    }
}

/// `GET /setup/audio-devices` when playback is
/// disabled: no devices exist.
#[cfg(not(feature = "playback"))]
pub async fn get_setup_audio_devices(
    _token: AuthenticatedToken,
    _state: State<AppState>,
) -> Response {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "success": true,
            "default": "default",
            "devices": [],
        })),
    )
        .into_response()
}

/// `POST /setup/complete` — finish first-run setup.
#[derive(Debug, Deserialize)]
pub struct CompleteSetupRequest {
    /// `local`, `lan`, or `advanced`.
    pub usage_mode: String,
    /// Selected audio output device id.
    pub audio_device: String,
    /// Inference steps for new synthesis.
    #[serde(default = "default_inference_steps")]
    pub inference_steps: usize,
    /// Model cache directory (optional override).
    pub model_cache_dir: Option<String>,
    /// Advanced mode only: bind address.
    pub bind: Option<String>,
    /// Advanced mode only: listen port.
    pub port: Option<u16>,
    /// Advanced mode only: auth mode.
    pub auth_mode: Option<String>,
    /// Admin username.
    #[serde(default = "default_admin_username")]
    pub admin_username: String,
    /// Admin password (stored in the secret store,
    /// never in `config.toml`).
    pub admin_password: String,
}

fn default_inference_steps() -> usize {
    5
}

fn default_admin_username() -> String {
    "admin".to_string()
}

/// Response for a successful setup.
#[derive(Debug, serde::Serialize)]
pub struct SetupCompleteResponse {
    pub success: bool,
    pub api_url: String,
    pub audio_device: String,
    pub auth: String,
    /// Raw API token, shown exactly once when token
    /// authentication was selected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_token: Option<String>,
}

pub async fn post_setup_complete(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    session: Session,
    Json(request): Json<CompleteSetupRequest>,
) -> Response {
    if !addr.ip().is_loopback() {
        return StatusCode::FORBIDDEN.into_response();
    }
    let _setup_guard = state.config.setup_guard().await;
    let current = state.config.get().await;
    if current.setup_complete {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "success": false,
                "message": "setup is already complete",
            })),
        )
            .into_response();
    }

    // Usage mode -> network configuration.
    let (bind, auth_mode) = match request.usage_mode.as_str() {
        "local" => (std::net::IpAddr::from([127, 0, 0, 1]), AuthMode::Local),
        "lan" => (std::net::IpAddr::from([0, 0, 0, 0]), AuthMode::Token),
        "advanced" => {
            let bind = request
                .bind
                .as_deref()
                .unwrap_or("127.0.0.1")
                .parse()
                .map_err(|_| bad_request("bind must be a valid IP address"));
            let bind = match bind {
                Ok(bind) => bind,
                Err(response) => return response,
            };
            let auth_mode = match request.auth_mode.as_deref().unwrap_or("local") {
                "local" => AuthMode::Local,
                "token" => AuthMode::Token,
                "none" => AuthMode::None,
                other => return bad_request(format!("unknown auth mode: {other}")),
            };
            (bind, auth_mode)
        }
        other => return bad_request(format!("unknown usage mode: {other}")),
    };
    let port = request.port.unwrap_or(current.server.port);

    // Admin password policy (fail closed).
    if let Err(e) = crate::config::validation::validate_admin_password(&request.admin_password) {
        return bad_request(format!("admin password: {e}"));
    }
    let admin_username = request.admin_username.trim();
    if admin_username.is_empty() {
        return bad_request("admin username must not be empty");
    }

    // Inference steps are validated by the same
    // range check the configuration uses.
    if request.inference_steps == 0
        || request.inference_steps > crate::config::model::MAX_INFERENCE_STEPS
    {
        return bad_request(format!(
            "inference_steps must be 1..={}",
            crate::config::model::MAX_INFERENCE_STEPS
        ));
    }

    let model_dir = request
        .model_cache_dir
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| current.model.cache_dir.clone());
    if let Err(error) = verify_model_storage(&model_dir) {
        return bad_request(format!("model.cache_dir: {error}"));
    }

    // Store the admin password in the secret store
    // before committing the configuration, so a
    // completed install never starts passwordless.
    if let Err(e) = secrets::store_admin_password(state.secrets.as_ref(), &request.admin_password) {
        tracing::error!("failed to store admin password: {e}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "success": false,
                "message": "failed to store admin credentials",
            })),
        )
            .into_response();
    }

    // Mint an API token when token authentication is
    // selected; the raw value is shown exactly once.
    let api_token = if matches!(auth_mode, AuthMode::Token) {
        match state.token_store.create(None).await {
            Ok((_token, raw)) => Some(raw),
            Err(e) => {
                tracing::error!("failed to create initial API token: {e}");
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"success": false, "message": "failed to create API token"}))).into_response();
            }
        }
    } else {
        None
    };

    // Commit the completed configuration.
    let audio_device = request.audio_device.trim().to_string();
    let model_cache_dir = request
        .model_cache_dir
        .as_deref()
        .map(str::trim)
        .filter(|dir| !dir.is_empty())
        .map(std::path::PathBuf::from);
    let update = state
        .config
        .update_from(
            Some(current.revision),
            crate::config::diff::ChangeSource::FirstRun,
            |config| {
                config.setup_complete = true;
                config.server.bind = bind;
                config.server.port = port;
                config.server.auth_mode = auth_mode;
                config.admin.username = admin_username.to_string();
                config.audio.output_device = audio_device.clone();
                config.model.inference_steps = request.inference_steps;
                if let Some(cache_dir) = &model_cache_dir {
                    config.model.cache_dir = cache_dir.clone();
                }
            },
        )
        .await;
    let revision = match update {
        Ok(update) => update.revision,
        Err(e) => {
            tracing::error!("setup configuration rejected: {e}");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "success": false,
                    "message": e.to_string(),
                })),
            )
                .into_response();
        }
    };
    let _ = revision;

    // Rotate the session so the wizard session cannot
    // be confused with an admin session.
    admin_session::rotate_id(&session).await;
    let _ = headers;

    let api_url = format!("http://{}", SocketAddr::new(bind, port));
    let auth_label = match auth_mode {
        AuthMode::Local => "local (no API token required)".to_string(),
        AuthMode::Token => "bearer token".to_string(),
        AuthMode::None => "none (development only)".to_string(),
    };
    tracing::info!(
        revision = state.config.revision(),
        "first-run setup completed"
    );
    (
        StatusCode::OK,
        [(axum::http::header::CACHE_CONTROL, "no-store")],
        Json(SetupCompleteResponse {
            success: true,
            api_url,
            audio_device,
            auth: auth_label,
            api_token,
        }),
    )
        .into_response()
}

fn verify_model_storage(path: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    let probe = path.join(format!(".sonicboom-write-test-{}", uuid::Uuid::new_v4()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(&probe)?;
    let result = file.sync_all();
    drop(file);
    let cleanup = std::fs::remove_file(probe);
    result.and(cleanup)
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "success": false,
            "message": message.into(),
        })),
    )
        .into_response()
}

/// The wizard page. A single self-contained document
/// with step navigation; no external assets so it
/// works before any configuration exists.
const WIZARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SonicBoom — First-Run Setup</title>
<link rel="stylesheet" href="/static/setup.css">
</head>
<body>
<h1>Welcome to SonicBoom</h1>
<p id="progress">Step <span id="step-no">1</span> of <span id="step-count">4</span></p>

<form id="wizard">
  <!-- Step 1: usage mode -->
  <fieldset class="step active" data-step="1">
    <legend>How will SonicBoom be used?</legend>
    <label><input type="radio" name="usage_mode" value="local" checked>
      <strong>Local application</strong><br>
      <span class="hint">Only applications on this computer. No API token required. Recommended.</span></label>
    <label><input type="radio" name="usage_mode" value="lan">
      <strong>LAN / API server</strong><br>
      <span class="hint">Allow other devices or applications. Bearer token authentication required.</span></label>
    <label><input type="radio" name="usage_mode" value="advanced">
      <strong>Advanced</strong><br>
      <span class="hint">Configure networking and security manually.</span></label>
  </fieldset>

  <!-- Step 1b: advanced network options -->
  <fieldset class="step" data-step="1b">
    <legend>Network &amp; security</legend>
    <label>Bind address <input name="bind" value="127.0.0.1"></label>
    <label>Port <input name="port" type="number" min="1" max="65535" value="17842"></label>
    <label>Authentication mode
      <select name="auth_mode">
        <option value="local">local — no API token (loopback only)</option>
        <option value="token">token — bearer token required</option>
        <option value="none">none — development only</option>
      </select></label>
    <p class="hint">Local mode always requires loopback. Insecure remote mode
    requires an explicit security.allow_insecure_remote option.</p>
  </fieldset>

  <!-- Step 2: audio -->
  <fieldset class="step" data-step="2">
    <legend>Audio output</legend>
    <label>Output device <select name="audio_device" id="audio-device">
      <option value="" selected disabled>Choose an output device</option>
      <option value="default">System Default</option>
    </select></label>
    <button type="button" id="test-audio" disabled>Test Audio</button>
    <p class="hint">Devices are enumerated from this machine. Select
    "System Default" to follow the OS default.</p>
  </fieldset>

  <!-- Step 3: model -->
  <fieldset class="step" data-step="3">
    <legend>Model</legend>
    <p>Model: Supertonic 3</p>
    <label>Model directory <input name="model_cache_dir" placeholder="(default platform directory)"></label>
    <label>Inference steps <input name="inference_steps" type="number" min="1" max="50" value="5"></label>
    <p class="hint">Missing models are downloaded automatically on first use.</p>
  </fieldset>

  <!-- Step 4: credentials + summary -->
  <fieldset class="step" data-step="4">
    <legend>Finish setup</legend>
    <label>Admin username <input name="admin_username" value="admin"></label>
    <label>Admin password <input name="admin_password" type="password"
      placeholder="12+ characters, not a common password" required></label>
    <p class="hint">The password is stored in the OS-restricted secret store,
    never in config.toml.</p>
    <h2>Summary</h2>
    <div id="summary">…</div>
  </fieldset>

  <div class="nav">
    <button type="button" id="back">Back</button>
    <button type="button" id="next">Continue</button>
    <button type="submit" id="finish" hidden>Complete Setup</button>
  </div>
  <p id="error" role="alert"></p>
</form>

<script src="/static/setup.js" defer></script>
</body>
</html>
"#;
