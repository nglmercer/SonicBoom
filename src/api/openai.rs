//! OpenAI-compatible TTS API endpoints
//!
//! This module provides endpoints that mimic the OpenAI TTS API format,
//! allowing SonicBoom to be used as a drop-in replacement for OpenAI's TTS service.

use axum::{
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::IntoResponse,
};
use serde::Deserialize;
use std::sync::Arc;

use super::tts::synthesize_bounded;
use crate::{
    auth::AuthenticatedToken,
    error::AppError,
    tts::{ModelStatus, audio},
};

/// Model identifiers accepted by `/v1/audio/speech`. All currently select
/// the local Supertonic 3 model; unknown identifiers are rejected with
/// `400` instead of silently using another model.
pub const SUPPORTED_OPENAI_MODELS: &[&str] = &["tts-1", "tts-1-hd", "supertonic-3"];

/// OpenAI-compatible TTS request body
#[derive(Deserialize)]
pub struct SpeechRequest {
    /// The model to use (must be one of [`SUPPORTED_OPENAI_MODELS`]).
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
    /// Speech speed. Only `1.0` is currently supported; anything else is
    /// rejected explicitly instead of silently ignored.
    #[serde(default)]
    pub speed: Option<f32>,
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

/// Validate the requested model identifier.
pub fn validate_model(model: &str) -> Result<(), AppError> {
    if SUPPORTED_OPENAI_MODELS.contains(&model) {
        Ok(())
    } else {
        Err(AppError::BadRequest(format!(
            "unsupported model '{model}' (expected one of: {})",
            SUPPORTED_OPENAI_MODELS.join(", ")
        )))
    }
}

/// Validate the requested speech speed.
///
/// OpenAI documents `0.25`–`4.0`; SonicBoom does not implement speed
/// adjustment yet, so only `1.0` (or an omitted value) is accepted and
/// anything else fails explicitly.
pub fn validate_speed(speed: Option<f32>) -> Result<(), AppError> {
    let speed = speed.unwrap_or(1.0);
    if !speed.is_finite() {
        return Err(AppError::BadRequest(
            "speed must be a finite number".to_string(),
        ));
    }
    if !(0.25..=4.0).contains(&speed) {
        return Err(AppError::BadRequest(format!(
            "speed {speed} is outside the supported range 0.25–4.0"
        )));
    }
    if speed != 1.0 {
        return Err(AppError::BadRequest(format!(
            "unsupported speed {speed}: only 1.0 is currently supported"
        )));
    }
    Ok(())
}

/// Map OpenAI voice names to Supertonic 3 voice styles.
///
/// Unknown names are rejected with `400` so application bugs cannot hide
/// behind a silent fallback voice.
pub fn map_voice(openai_voice: &str) -> Result<String, AppError> {
    match openai_voice {
        // OpenAI standard voices -> Supertonic 3 voices
        "alloy" => Ok("M1".to_string()),
        "echo" => Ok("M2".to_string()),
        "fable" => Ok("M3".to_string()),
        "onyx" => Ok("M4".to_string()),
        "nova" => Ok("F1".to_string()),
        "shimmer" => Ok("F2".to_string()),
        // Direct Supertonic 3 voice names (F1-F5, M1-M5)
        _ => {
            if (openai_voice.starts_with('M') || openai_voice.starts_with('F'))
                && openai_voice.len() == 2
                && let Ok(num) = openai_voice[1..].parse::<u32>()
                && (1..=5).contains(&num)
            {
                return Ok(openai_voice.to_string());
            }
            Err(AppError::BadRequest(format!(
                "unsupported voice '{openai_voice}' (expected one of: alloy, echo, fable, onyx, nova, shimmer, M1-M5, F1-F5)"
            )))
        }
    }
}

/// Synthesize speech using OpenAI-compatible API
#[allow(clippy::unused_async)]
pub async fn post_speech(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    body: String,
) -> Result<impl IntoResponse, AppError> {
    if !state.rate_limiter.allow(&token.rate_limit_key()) {
        return Err(AppError::TooManyRequests(
            "Rate limit exceeded. Try again later.".to_string(),
        ));
    }

    let request: SpeechRequest = serde_json::from_str(&body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON in request body: {e}")))?;

    validate_model(&request.model)?;
    validate_speed(request.speed)?;
    let voice_name = map_voice(&request.voice)?;
    let format = audio::AudioFormat::parse(&request.response_format)
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    sonicboom::engine::validate_tts_input(
        &request.input,
        "en",
        &request.voice,
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
                tracing::error!("openai tts request while model failed: {reason}");
                return Err(AppError::internal("model unavailable"));
            }
        }
    };

    let text = request.input.trim().to_string();

    // Verify voice exists, fallback to default
    let voice_name = if model_handle.voice_styles.contains_key(&voice_name) {
        voice_name
    } else {
        model_handle
            .default_voice()
            .ok_or_else(|| AppError::internal("no voice styles available"))?
            .to_string()
    };

    let lang = "en".to_string(); // Default language
    let sample_rate = model_handle.sample_rate();

    let samples = synthesize_bounded(&state, model_handle, text, lang, voice_name).await?;

    let audio_bytes = audio::encode_audio(&samples, sample_rate, format)
        .map_err(|e| AppError::internal(e.to_string()))?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        Body::from(audio_bytes),
    ))
}

/// List available models (OpenAI-compatible endpoint)
///
/// Public metadata endpoint: exposes only non-sensitive model identifiers
/// (see docs). Authentication is intentionally not required.
pub async fn get_models() -> impl IntoResponse {
    let body = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "supertonic-3",
                "object": "model",
                "created": 1704067200,
                "owned_by": "local",
                "permission": [],
                "root": "supertonic-3",
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
///
/// Public metadata endpoint: exposes only non-sensitive voice names.
pub async fn get_voices(State(state): State<crate::AppState>) -> impl IntoResponse {
    let voices = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Ready(handle) => handle
                .voice_styles
                .keys()
                .map(|name| {
                    serde_json::json!({
                        "id": name,
                        "name": name,
                        "object": "voice"
                    })
                })
                .collect::<Vec<_>>(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_models_are_accepted() {
        for model in ["tts-1", "tts-1-hd", "supertonic-3"] {
            assert!(validate_model(model).is_ok(), "model {model} rejected");
        }
    }

    #[test]
    fn unknown_models_are_rejected() {
        for model in ["", "gpt-4o", "tts-2", "supertonic-2", "TTS-1", "tts-1 "] {
            assert!(
                validate_model(model).is_err(),
                "model {model:?} silently accepted"
            );
        }
    }

    #[test]
    fn speed_validation() {
        // Omitted and 1.0 are accepted.
        assert!(validate_speed(None).is_ok());
        assert!(validate_speed(Some(1.0)).is_ok());
        // Anything else is rejected explicitly.
        for speed in [
            Some(0.5),
            Some(2.0),
            Some(0.24),
            Some(4.01),
            Some(0.0),
            Some(-1.0),
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
        ] {
            assert!(
                validate_speed(speed).is_err(),
                "speed {speed:?} silently accepted"
            );
        }
    }

    #[test]
    fn known_voices_map_correctly() {
        for (openai, expected) in [
            ("alloy", "M1"),
            ("echo", "M2"),
            ("fable", "M3"),
            ("onyx", "M4"),
            ("nova", "F1"),
            ("shimmer", "F2"),
            ("M1", "M1"),
            ("M5", "M5"),
            ("F1", "F1"),
            ("F5", "F5"),
        ] {
            assert_eq!(map_voice(openai).unwrap(), expected);
        }
    }

    #[test]
    fn unknown_voices_are_rejected_not_defaulted() {
        for voice in [
            "", "banana", "alloy2", "M0", "M6", "F0", "F6", "m1", "X1", " M1",
        ] {
            assert!(
                map_voice(voice).is_err(),
                "voice {voice:?} silently mapped to M1"
            );
        }
    }

    #[test]
    fn speech_request_defaults_match_documented_behavior() {
        let request: SpeechRequest = serde_json::from_str(r#"{"input":"hi"}"#).unwrap();
        assert_eq!(request.model, "tts-1");
        assert_eq!(request.voice, "alloy");
        assert_eq!(request.response_format, "opus");
        assert_eq!(request.speed, None);
        assert!(validate_speed(request.speed).is_ok());
    }
}
