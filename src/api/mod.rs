pub mod openai;
#[cfg(feature = "playback")]
pub mod queue;
pub mod tts;

use crate::AppState;
use axum::{
    Router,
    routing::{get, post},
};

pub fn router(state: AppState) -> Router {
    #[allow(unused_mut)]
    let mut router = Router::new()
        // Original TTS API
        .route("/api/tts", post(tts::post_tts))
        .route("/api/status", get(tts::get_status))
        // OpenAI-compatible endpoints
        .route("/v1/audio/speech", post(openai::post_speech))
        .route("/v1/models", get(openai::get_models))
        .route("/v1/models/list", get(openai::get_models))
        .route("/v1/voices", get(openai::get_voices));

    #[cfg(feature = "playback")]
    {
        // Audio playback and queue endpoints (only available when playback feature is enabled)
        router = router
            .route("/api/tts/play", post(tts::post_tts_and_play))
            .route("/api/queue", post(queue::queue_audio))
            .route("/api/queue/next", post(queue::play_next))
            .route("/api/queue/pause", post(queue::pause_audio))
            .route("/api/queue/resume", post(queue::resume_audio))
            .route("/api/queue/stop", post(queue::stop_audio))
            .route("/api/queue/volume", post(queue::set_volume))
            .route("/api/queue/status", get(queue::get_queue_status));
    }

    router.with_state(state)
}
