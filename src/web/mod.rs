pub mod index;
pub mod static_files;

use crate::AppState;
use axum::{
    Router,
    routing::{get, post},
};

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index::get_index))
        .route("/health", get(index::get_health))
        .route("/ready", get(index::get_ready))
        .route("/static/{file}", get(static_files::get_static))
        // First-run setup wizard (bootstrap mode:
        // loopback-only, no API bearer auth).
        .route("/setup", get(crate::setup::get_setup))
        .route("/setup/status", get(crate::setup::get_setup_status))
        .route(
            "/setup/audio-devices",
            get(crate::setup::get_setup_audio_devices),
        )
        .route("/setup/complete", post(crate::setup::post_setup_complete))
        .with_state(state)
}
