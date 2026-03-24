use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    auth::AuthenticatedToken,
    error::AppError,
    tts::{audio, inference, ModelStatus},
};

#[derive(Deserialize)]
pub struct TtsQuery {
    pub voice: Option<String>,
    pub lang: Option<String>,
    /// Output format: "opus", "wav", "mp3", or "flac"
    #[serde(default)]
    pub format: Option<String>,
    /// Play immediately?
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
        ModelStatus::Failed(reason) => StatusResponse {
            status: "failed".to_string(),
            progress: None,
            error: Some(reason.clone()),
        },
    };

    (StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], serde_json::to_string(&response).unwrap())
}

pub async fn post_tts(
    _token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<TtsQuery>,
    body: String,
) -> Result<Response, AppError> {
    if body.trim().is_empty() {
        return Err(AppError::BadRequest("Text cannot be empty.".to_string()));
    }

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
                return Err(AppError::Internal(format!("Model failed to load: {reason}")));
            }
        }
    };

    let text = body.trim().to_string();
    let voice_name = match query.voice {
        Some(ref v) if model_handle.voice_styles.contains_key(v.as_str()) => v.clone(),
        _ => model_handle
            .default_voice()
            .ok_or_else(|| AppError::Internal("No voice styles available.".to_string()))?
            .to_string(),
    };
    let lang = query.lang.unwrap_or_else(|| "en".to_string());
    let sample_rate = model_handle.sample_rate();
    let inference_steps = state.config.inference_steps;

    // Determine output format
    let format = query
        .format
        .as_deref()
        .map(audio::AudioFormat::from_str)
        .unwrap_or(audio::AudioFormat::Opus);

    let samples = tokio::task::spawn_blocking(move || {
        inference::synthesize(&model_handle, &text, &lang, &voice_name, inference_steps)
    })
    .await
    .map_err(|e| AppError::Internal(e.to_string()))?
    .map_err(|e| AppError::Internal(e.to_string()))?;

    let audio_bytes = audio::encode_audio(&samples, sample_rate, format)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        Body::from(audio_bytes),
    )
        .into_response())
}

pub async fn post_tts_and_play(
    _token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<TtsQuery>,
    body: String,
) -> Result<Response, AppError> {
    if body.trim().is_empty() {
        return Err(AppError::BadRequest("Text cannot be empty.".to_string()));
    }

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
            .ok_or_else(|| AppError::Internal("No voice styles available.".to_string()))?
            .to_string(),
    };
    
    let lang = query.lang.unwrap_or_else(|| "en".to_string());
    let sample_rate = model_handle.sample_rate();
    let inference_steps = state.config.inference_steps;
    let play_now = query.play_now.unwrap_or(false);

    // Synthesis to WAV for local playback (Rodio likes WAV/Decoder compatibility)
    let samples = tokio::task::spawn_blocking(move || {
        inference::synthesize(&model_handle, &text, &lang, &voice_name, inference_steps)
    })
    .await
    .map_err(|e| AppError::Internal(e.to_string()))?
    .map_err(|e| AppError::Internal(e.to_string()))?;

    // We use WAV for internal queue to ensure maximum compatibility with rodio
    let audio_bytes = audio::encode_audio(&samples, sample_rate, audio::AudioFormat::Wav)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    // Ensure temp directory exists
    let temp_dir = std::path::PathBuf::from("temp_audio");
    if !temp_dir.exists() {
        std::fs::create_dir_all(&temp_dir).map_err(|e| AppError::Internal(e.to_string()))?;
    }

    // Save to temp file
    let id = uuid::Uuid::new_v4().to_string();
    let filename = format!("{}.wav", id);
    let path = temp_dir.join(&filename);
    
    std::fs::write(&path, audio_bytes).map_err(|e| AppError::Internal(e.to_string()))?;

    // Add to queue or play now
    if play_now {
        audio_manager.play_now(id.clone(), path).await;
    } else {
        audio_manager.add_to_queue(id.clone(), path.clone()).await;
        
        // Auto-play if nothing is currently playing
        let status = audio_manager.status().await;
        if !status.is_playing {
            audio_manager.play_next().await;
        }
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
