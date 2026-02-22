pub mod tts;

use axum::{routing::post, Router};
use crate::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/tts", post(tts::post_tts))
        .with_state(state)
}
