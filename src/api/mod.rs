pub mod openai;
pub mod tts;

use axum::{routing::{post, get}, Router};
use crate::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        // Original TTS API
        .route("/api/tts", post(tts::post_tts))
        // OpenAI-compatible endpoints
        .route("/v1/audio/speech", post(openai::post_speech))
        .route("/v1/models", get(openai::get_models))
        .route("/v1/models/list", get(openai::get_models))
        .route("/v1/voices", get(openai::get_voices))
        .with_state(state)
}
