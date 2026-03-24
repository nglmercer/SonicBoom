pub mod openai;
pub mod queue;
pub mod tts;

use axum::{routing::{post, get}, Router};
use crate::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        // Original TTS API
        .route("/api/tts", post(tts::post_tts))
        .route("/api/tts/play", post(tts::post_tts_and_play))
        .route("/api/status", get(tts::get_status))
        // Audio queue endpoints
        .route("/api/queue", post(queue::queue_audio))
        .route("/api/queue/next", post(queue::play_next))
        .route("/api/queue/pause", post(queue::pause_audio))
        .route("/api/queue/resume", post(queue::resume_audio))
        .route("/api/queue/stop", post(queue::stop_audio))
        .route("/api/queue/volume", post(queue::set_volume))
        .route("/api/queue/status", get(queue::get_queue_status))
        // OpenAI-compatible endpoints
        .route("/v1/audio/speech", post(openai::post_speech))
        .route("/v1/models", get(openai::get_models))
        .route("/v1/models/list", get(openai::get_models))
        .route("/v1/voices", get(openai::get_voices))
        .with_state(state)
}
