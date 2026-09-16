#[cfg(feature = "playback")]
use axum::Json;
use axum::{
    body::Body,
    extract::{Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    auth::AuthenticatedToken,
    error::AppError,
    tts::{ModelStatus, audio, inference},
};

#[derive(Deserialize)]
pub struct TtsQuery {
    pub voice: Option<String>,
    pub lang: Option<String>,
    /// Output format: "opus", "wav", "mp3", or "flac"
    #[serde(default)]
    pub format: Option<String>,
    /// Play immediately?
    #[cfg(feature = "playback")]
    pub play_now: Option<bool>,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub status: String,
    pub progress: Option<f32>,
    pub error: Option<String>,
}

pub async fn get_status(State(state): State<crate::AppState>) -> impl IntoResponse {
    let status = state.model_status.read().await;
    let response = match &*status {
        ModelStatus::Idle => StatusResponse {
            status: "idle".to_string(),
            progress: None,
            error: None,
        },
        ModelStatus::Downloading { progress } => StatusResponse {
            status: "downloading".to_string(),
            progress: Some(*progress * 100.0),
            error: None,
        },
        ModelStatus::Loading => StatusResponse {
            status: "loading".to_string(),
            progress: None,
            error: None,
        },
        ModelStatus::Ready(_) => StatusResponse {
            status: "ready".to_string(),
            progress: None,
            error: None,
        },
        ModelStatus::Failed(reason) => {
            // Do not expose internal load failures; they are logged server-side.
            tracing::error!("model status requested while failed: {reason}");
            StatusResponse {
                status: "failed".to_string(),
                progress: None,
                error: Some("Model failed to load.".to_string()),
            }
        }
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&response).unwrap(),
    )
}

/// Shared pre-inference pipeline: rate limit -> validate -> admission control.
fn check_rate_limit(state: &crate::AppState, token: &AuthenticatedToken) -> Result<(), AppError> {
    if !state.rate_limiter.allow(&token.rate_limit_key()) {
        return Err(AppError::TooManyRequests(
            "Rate limit exceeded. Try again later.".to_string(),
        ));
    }
    Ok(())
}

pub async fn post_tts(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<TtsQuery>,
    body: String,
) -> Result<Response, AppError> {
    check_rate_limit(&state, &token)?;
    let lang = query.lang.clone().unwrap_or_else(|| "en".to_string());
    let voice = query.voice.clone().unwrap_or_else(|| "M1".to_string());
    sonicboom::engine::validate_tts_input(
        &body,
        &lang,
        &voice,
        state.config.inference_steps,
        state.config.max_text_length,
    )
    .map_err(|e| AppError::BadRequest(e.to_string()))?;

    let model_handle = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Ready(handle) => Arc::clone(handle),
            ModelStatus::Downloading { progress } => {
                return Err(AppError::ServiceUnavailable(format!(
                    "Model is downloading ({:.0}% complete).",
                    progress * 100.0
                )));
            }
            ModelStatus::Loading => {
                return Err(AppError::ServiceUnavailable(
                    "Model is loading.".to_string(),
                ));
            }
            ModelStatus::Idle => {
                return Err(AppError::ServiceUnavailable(
                    "Model has not started loading yet.".to_string(),
                ));
            }
            ModelStatus::Failed(reason) => {
                tracing::error!("tts request while model failed: {reason}");
                return Err(AppError::internal("model unavailable"));
            }
        }
    };

    let text = body.trim().to_string();
    let voice_name = match query.voice {
        Some(ref v) if model_handle.voice_styles.contains_key(v.as_str()) => v.clone(),
        _ => model_handle
            .default_voice()
            .ok_or_else(|| AppError::internal("no voice styles available"))?
            .to_string(),
    };
    let sample_rate = model_handle.sample_rate();
    let inference_steps = state.config.inference_steps;
    let max_chunk_chars = state.config.max_chunk_chars;

    // Determine output format
    let format = query
        .format
        .as_deref()
        .map(audio::AudioFormat::parse)
        .unwrap_or(audio::AudioFormat::Opus);

    // Bounded admission control before spawning blocking work.
    let _permit = state.inference_gate.admit().await.ok_or_else(|| {
        AppError::TooManyRequests(
            "Server is busy. Too many pending inference requests.".to_string(),
        )
    })?;

    let samples = tokio::task::spawn_blocking(move || {
        inference::synthesize(
            &model_handle,
            &text,
            &lang,
            &voice_name,
            inference_steps,
            max_chunk_chars,
        )
    })
    .await
    .map_err(|e| AppError::internal(format!("inference task failed: {e}")))?
    .map_err(|e| AppError::internal(e.to_string()))?;

    let audio_bytes = audio::encode_audio(&samples, sample_rate, format)
        .map_err(|e| AppError::internal(e.to_string()))?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        Body::from(audio_bytes),
    )
        .into_response())
}

#[cfg(feature = "playback")]
pub async fn post_tts_and_play(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<TtsQuery>,
    body: String,
) -> Result<Response, AppError> {
    check_rate_limit(&state, &token)?;
    let lang = query.lang.clone().unwrap_or_else(|| "en".to_string());
    let voice = query.voice.clone().unwrap_or_else(|| "M1".to_string());
    sonicboom::engine::validate_tts_input(
        &body,
        &lang,
        &voice,
        state.config.inference_steps,
        state.config.max_text_length,
    )
    .map_err(|e| AppError::BadRequest(e.to_string()))?;

    let model_handle = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Ready(handle) => Arc::clone(handle),
            _ => {
                return Err(AppError::ServiceUnavailable(
                    "Model is not ready for synthesis.".to_string(),
                ));
            }
        }
    };

    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return Err(AppError::ServiceUnavailable(
                "Audio playback system not initialized.".to_string(),
            ));
        }
    };

    let text = body.trim().to_string();
    let voice_name = match query.voice {
        Some(ref v) if model_handle.voice_styles.contains_key(v.as_str()) => v.clone(),
        _ => model_handle
            .default_voice()
            .ok_or_else(|| AppError::internal("no voice styles available"))?
            .to_string(),
    };

    let sample_rate = model_handle.sample_rate();
    let inference_steps = state.config.inference_steps;
    let max_chunk_chars = state.config.max_chunk_chars;
    let play_now = query.play_now.unwrap_or(false);

    let _permit = state.inference_gate.admit().await.ok_or_else(|| {
        AppError::TooManyRequests(
            "Server is busy. Too many pending inference requests.".to_string(),
        )
    })?;

    // Synthesis to WAV for local playback (Rodio likes WAV/Decoder compatibility)
    let samples = tokio::task::spawn_blocking(move || {
        inference::synthesize(
            &model_handle,
            &text,
            &lang,
            &voice_name,
            inference_steps,
            max_chunk_chars,
        )
    })
    .await
    .map_err(|e| AppError::internal(format!("inference task failed: {e}")))?
    .map_err(|e| AppError::internal(e.to_string()))?;

    // We use WAV for internal queue to ensure maximum compatibility with rodio
    let audio_bytes = audio::encode_audio(&samples, sample_rate, audio::AudioFormat::Wav)
        .map_err(|e| AppError::internal(e.to_string()))?;

    // Ensure temp directory exists
    let temp_dir = std::path::PathBuf::from(&state.config.temp_audio_dir);
    if !temp_dir.exists() {
        tokio::fs::create_dir_all(&temp_dir)
            .await
            .map_err(|e| AppError::internal(e.to_string()))?;
    }

    // Save to temp file
    let id = uuid::Uuid::new_v4().to_string();
    let filename = format!("{id}.wav");
    let path = temp_dir.join(&filename);

    tokio::fs::write(&path, audio_bytes)
        .await
        .map_err(|e| AppError::internal(e.to_string()))?;

    // Register temp file for cleanup after playback, then add to queue
    audio_manager.register_temp(path.clone()).await;
    if play_now {
        audio_manager.play_now(id.clone(), path).await;
    } else {
        audio_manager.add_to_queue(id.clone(), path).await;
    }

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "success": true,
            "message": if play_now { "Playing immediately" } else { "Added to queue" },
            "id": id
        })),
    )
        .into_response())
}
