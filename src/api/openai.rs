#![allow(dead_code)]

//! OpenAI-compatible TTS API endpoints
//!
//! This module provides endpoints that mimic the OpenAI TTS API format,
//! allowing SonicBoom to be used as a drop-in replacement for OpenAI's TTS service.

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
};
use serde::Deserialize;
use std::sync::Arc;

use crate::{
    auth::AuthenticatedToken,
    error::AppError,
    tts::{audio, inference, ModelStatus},
};

/// OpenAI-compatible TTS request body
#[derive(Deserialize)]
pub struct SpeechRequest {
    /// The model to use (ignored, we use Supertonic 2)
    #[serde(default = "default_model")]
    pub model: String,
    /// The text to synthesize
    pub input: String,
    /// The voice to use
    #[serde(default = "default_voice")]
    pub voice: String,
    /// The output format
    #[serde(default = "default_format")]
    pub response_format: String,
    /// Speech speed (0.25 to 4.0)
    #[serde(default = "default_speed")]
    pub speed: f32,
}

fn default_model() -> String {
    "tts-1".to_string()
}

fn default_voice() -> String {
    "alloy".to_string()
}

fn default_format() -> String {
    "opus".to_string()
}

fn default_speed() -> f32 {
    1.0
}

/// Map OpenAI voice names to Supertonic 2 voice styles
fn map_voice(openai_voice: &str) -> String {
    match openai_voice {
        // OpenAI standard voices -> Supertonic 2 voices
        "alloy" => "M1".to_string(),
        "echo" => "M2".to_string(),
        "fable" => "M3".to_string(),
        "onyx" => "M4".to_string(),
        "nova" => "F1".to_string(),
        "shimmer" => "F2".to_string(),
        // Direct Supertonic 2 voice names (F1-F5, M1-M5)
        _ => {
            // Check if it's a valid voice name (M1-M5 or F1-F5)
            if (openai_voice.starts_with('M') || openai_voice.starts_with('F'))
                && openai_voice.len() == 2
            {
                if let Some(num) = openai_voice[1..].parse::<u32>().ok() {
                    if (1..=5).contains(&num) {
                        return openai_voice.to_string();
                    }
                }
            }
            "M1".to_string()
        }
    }
}

/// Synthesize speech using OpenAI-compatible API
#[allow(clippy::unused_async)]
pub async fn post_speech(
    _token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    body: String,
) -> Result<impl IntoResponse, AppError> {
    let request: SpeechRequest = serde_json::from_str(&body).map_err(|e| {
        AppError::BadRequest(format!("Invalid JSON in request body: {e}"))
    })?;

    if request.input.trim().is_empty() {
        return Err(AppError::BadRequest("Input text cannot be empty.".to_string()));
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
                return Err(AppError::ServiceUnavailable("Model is loading.".to_string()));
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

    let text = request.input.trim().to_string();
    let voice_name = map_voice(&request.voice);

    // Verify voice exists, fallback to default
    let voice_name = if model_handle.voice_styles.contains_key(&voice_name) {
        voice_name
    } else {
        model_handle
            .default_voice()
            .ok_or_else(|| AppError::Internal("No voice styles available.".to_string()))?
            .to_string()
    };
    
    let lang = "en".to_string(); // Default language
    let sample_rate = model_handle.sample_rate();
    let inference_steps = state.config.inference_steps;

    let samples = tokio::task::spawn_blocking(move || {
        inference::synthesize(&model_handle, &text, &lang, &voice_name, inference_steps)
    })
    .await
    .map_err(|e| AppError::Internal(e.to_string()))?
    .map_err(|e| AppError::Internal(e.to_string()))?;

    let opus_bytes = audio::encode_opus(&samples, sample_rate)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let content_type = match request.response_format.as_str() {
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        _ => "audio/opus", // Default to opus
    };

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, content_type)],
        Body::from(opus_bytes),
    ))
}

/// List available models (OpenAI-compatible endpoint)
pub async fn get_models() -> impl IntoResponse {
    let body = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "supertonic-2",
                "object": "model",
                "created": 1704067200,
                "owned_by": "local",
                "permission": [],
                "root": "supertonic-2",
                "parent": null,
            }
        ]
    });
    let body = body.to_string();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
}

/// List available voices (OpenAI-compatible endpoint)
pub async fn get_voices(State(state): State<crate::AppState>) -> impl IntoResponse {
    let voices = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Ready(handle) => {
                handle
                    .voice_styles
                    .keys()
                    .map(|name| {
                        serde_json::json!({
                            "id": name,
                            "name": name,
                            "object": "voice"
                        })
                    })
                    .collect::<Vec<_>>()
            }
            _ => vec![],
        }
    };

    let body = serde_json::json!({
        "object": "list",
        "data": voices
    });
    let body = body.to_string();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
}
