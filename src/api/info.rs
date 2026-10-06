//! Public server info endpoint (spec §31).
//!
//! `GET /api/info` describes the server without
//! exposing secrets: authentication mode, feature
//! flags, configuration revision, and model
//! status. The browser GUI uses it to discover
//! whether an API token is required (spec §32).

use axum::{Json, extract::State};
use serde::Serialize;

use crate::{AppState, config::AuthMode, tts::ModelStatus};

/// Response for `GET /api/info`.
#[derive(Serialize)]
pub struct InfoResponse {
    pub version: String,
    /// `desktop` for GUI builds, `server` for
    /// headless builds.
    pub mode: &'static str,
    pub setup_complete: bool,
    pub auth: AuthInfo,
    pub features: Features,
    pub config_revision: u64,
    pub model: ModelInfo,
}

#[derive(Serialize)]
pub struct AuthInfo {
    pub mode: String,
    /// Whether API requests must present a
    /// bearer token.
    pub required: bool,
}

#[derive(Serialize)]
pub struct Features {
    pub playback: bool,
    pub audio_devices: bool,
    pub hot_reload: bool,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub status: String,
}

/// `GET /api/info`
pub async fn get_info(State(state): State<AppState>) -> Json<InfoResponse> {
    let config = state.config.get().await;
    let model_status = state.model_status.read().await;
    let model_status = match &*model_status {
        ModelStatus::Idle => "idle",
        ModelStatus::Downloading { .. } => "downloading",
        ModelStatus::Loading => "loading",
        ModelStatus::Ready(_) => "ready",
        ModelStatus::Failed(_) => "failed",
    };
    Json(InfoResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        mode: if cfg!(feature = "gui") {
            "desktop"
        } else {
            "server"
        },
        setup_complete: config.setup_complete,
        auth: AuthInfo {
            mode: match config.server.auth_mode {
                AuthMode::Local => "local",
                AuthMode::Token => "token",
                AuthMode::None => "none",
            }
            .to_string(),
            required: matches!(config.server.auth_mode, AuthMode::Token),
        },
        features: Features {
            playback: cfg!(feature = "playback"),
            audio_devices: cfg!(feature = "playback"),
            hot_reload: true,
        },
        config_revision: config.revision,
        model: ModelInfo {
            status: model_status.to_string(),
        },
    })
}
