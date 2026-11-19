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

use super::speech_prep::{SpeechOptions, SpeechText};
use super::tts::{check_rate_limit, synthesize_bounded};
use crate::{
    auth::AuthenticatedToken,
    error::AppError,
    tts::{ModelStatus, audio},
};

/// Model identifiers accepted by `/v1/audio/speech`. Legacy aliases select
/// the local Supertonic 3 model; cloning models route to their native
/// provider. Unknown identifiers are rejected with `400` instead of
/// silently using another model.
pub use sonicboom::tts::providers::catalog::SUPPORTED_OPENAI_MODELS;
use sonicboom::tts::providers::catalog::{ModelProvider, model_provider};

/// Whether `model` selects a voice-cloning provider (voice must be an
/// authorized profile ID) rather than Supertonic built-in voices.
/// The removed 0.6B Candle model is no longer a cloning model.
/// Model IDs of the retired GPT-SoVITS provider. They are rejected with
/// an explicit retired-model error, never silently rerouted to Pocket TTS
/// or Supertonic.
pub const RETIRED_GPT_SOVITS_MODELS: &[&str] = &[
    "gpt-sovits-v2-onnx",
    "gpt-sovits-v2pro-onnx",
    "gpt-sovits-v2proplus-onnx",
];

pub fn is_cloning_model(model: &str) -> bool {
    matches!(
        model_provider(model),
        Some(ModelProvider::QwenGguf | ModelProvider::PocketTts | ModelProvider::QwenCandle)
    )
}

/// Models with built-in named voices that route through the provider registry
/// without a voice profile (Kyutai TTS).
pub fn is_named_voice_model(model: &str) -> bool {
    matches!(model_provider(model), Some(ModelProvider::KyutaiCandle))
}

/// Recognized identifiers remain stable across feature combinations. Check
/// compiled availability before profiles or synthesis; runtime enablement and
/// installed model readiness are checked by the owning provider afterward.
fn require_compiled_provider(model: &str) -> Result<(), AppError> {
    if let Some(provider) = model_provider(model)
        && !provider.compiled()
    {
        return Err(AppError::ServiceUnavailable(format!(
            "provider '{}' for model '{model}' is not compiled into this binary",
            provider.id()
        )));
    }
    Ok(())
}

/// OpenAI-compatible TTS request body
#[derive(Deserialize)]
pub struct SpeechRequest {
    /// The model to use (must be one of [`SUPPORTED_OPENAI_MODELS`]).
    #[serde(default = "default_model")]
    pub model: String,
    /// The text to synthesize
    pub input: String,
    /// The voice to use: a Supertonic voice name/alias, or (for cloning
    /// models) an authorized voice-profile ID from `POST /api/voices/clones`.
    #[serde(default = "default_voice")]
    pub voice: String,
    /// The output format
    #[serde(default = "default_format")]
    pub response_format: String,
    /// Speech speed. Only `1.0` is currently supported; anything else is
    /// rejected explicitly instead of silently ignored.
    #[serde(default)]
    pub speed: Option<f32>,
    /// Synthesis language (BCP-47 tag). Legacy Supertonic requests ignore
    /// this and synthesize English; cloning providers validate it against
    /// their supported languages without changing old defaults.
    #[serde(default)]
    pub language: Option<String>,
    /// Deterministic seed for cloning providers, when supported.
    #[serde(default)]
    pub seed: Option<u64>,
    /// SonicBoom extension: text normalization mode, `auto` or `off`.
    /// Absent uses the server's configured default.
    #[serde(default)]
    pub normalization: Option<String>,
    /// SonicBoom extension: pronunciation locale (e.g. `es-PE`). Absent or
    /// `auto` selects the speech language's documented default.
    #[serde(default)]
    pub locale: Option<String>,
    /// SonicBoom extension: reading style for written numbers and times
    /// (`precise`, `natural` or `digits`). Absent uses the server default.
    #[serde(default)]
    pub reading_style: Option<String>,
}

impl SpeechRequest {
    /// The request's normalization options.
    fn speech_options(&self) -> SpeechOptions {
        SpeechOptions {
            normalization: self.normalization.clone(),
            locale: self.locale.clone(),
            reading_style: self.reading_style.clone(),
        }
    }

    /// Prepare `input` for `language` (legacy trimming first), once.
    pub(crate) fn speech(
        &self,
        config: &crate::config::AppConfig,
        language: &str,
    ) -> Result<SpeechText, AppError> {
        self.speech_options()
            .prepare(config, language, self.input.trim())
    }
}

/// Provider request for a cloning profile. Only the target text is
/// prepared; the stored reference transcript and audio are passed through
/// byte-identical.
fn cloning_provider_request(
    profile: &sonicboom::tts::voices::VoiceProfile,
    reference: sonicboom::tts::providers::types::ReferenceAudio,
    speech: SpeechText,
    language: String,
    seed: Option<u64>,
) -> Result<sonicboom::tts::providers::ProviderRequest, AppError> {
    use sonicboom::tts::providers::types::DevicePreference;
    let mut provider_request =
        super::speech_prep::provider_request(&profile.model, &profile.id, speech, language);
    provider_request.device = DevicePreference::Auto;
    provider_request.clone_mode =
        Some(super::voices::parse_clone_mode_public(&profile.clone_mode)?);
    provider_request.reference_transcript = profile.transcript.clone();
    provider_request.reference_audio = Some(reference);
    provider_request.seed = seed;
    Ok(provider_request)
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

/// Validate the requested model identifier. Retired GPT-SoVITS IDs get
/// an explicit retired-model error instead of the generic unknown-model
/// message, and are never rerouted to another provider.
pub fn validate_model(model: &str) -> Result<(), AppError> {
    if SUPPORTED_OPENAI_MODELS.contains(&model) {
        Ok(())
    } else if RETIRED_GPT_SOVITS_MODELS.contains(&model) {
        Err(AppError::BadRequest(format!(
            "model '{model}' is retired (GPT-SoVITS was replaced by Pocket TTS); \
             use one of: {}",
            SUPPORTED_OPENAI_MODELS.join(", ")
        )))
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

/// Synthesize speech with a cloning provider for `POST /v1/audio/speech`
/// when `model` selects Qwen3-TTS or Pocket TTS. `voice` must be an
/// authorized voice-profile ID owned by the calling token; the stored
/// reference audio and transcript travel with the request.
async fn post_speech_cloning(
    token: &AuthenticatedToken,
    state: &crate::AppState,
    request: &SpeechRequest,
    format: sonicboom::tts::audio::AudioFormat,
) -> Result<axum::response::Response, AppError> {
    use super::providers::{build_registry_for_state, provider_error_to_app};
    use super::tts::synthesize_provider_bounded;

    let config = state.config.get().await;
    sonicboom::engine::validate_tts_input(
        &request.input,
        request.language.as_deref().unwrap_or("en"),
        "clone",
        config.model.inference_steps,
        config.tts.max_text_length,
    )
    .map_err(|e| AppError::BadRequest(e.to_string()))?;
    let profile_id = request.voice.trim();
    sonicboom::tts::voices::validate_profile_id(profile_id).map_err(|_| {
        AppError::BadRequest(format!(
            "model '{}' requires an authorized voice-profile ID as voice \
             (create one with POST /api/voices/clones); '{}' is not a profile ID",
            request.model, request.voice
        ))
    })?;
    let (profile, reference) =
        super::voices::owned_reference(state, profile_id.to_string(), token.rate_limit_key())
            .await?;
    if profile.provider == "gpt-sovits" {
        return Err(AppError::BadRequest(
            "voice profile uses the retired GPT-SoVITS provider and is unavailable; recreate it \
             as a pocket-tts profile (mode audio_prompt) from its stored reference audio"
                .into(),
        ));
    }
    if profile.model_revision != super::voices::config_revision_for(&config, &profile.provider) {
        return Err(AppError::BadRequest(
            "voice profile belongs to an incompatible model revision; create a new profile".into(),
        ));
    }
    if profile.model != request.model {
        return Err(AppError::BadRequest(format!(
            "voice profile '{}' is bound to model '{}' and cannot be used with '{}'",
            profile.id, profile.model, request.model
        )));
    }
    let language = request.language.clone().unwrap_or(profile.language.clone());
    let speech = request.speech(&config, &language)?;
    let provider_request =
        cloning_provider_request(&profile, reference, speech, language, request.seed)?;
    let registry = build_registry_for_state(state).await;
    let provider = registry
        .resolve_model(&request.model)
        .map_err(provider_error_to_app)?;
    if provider.id() != profile.provider {
        return Err(AppError::BadRequest(format!(
            "voice profile '{}' belongs to provider '{}' and cannot be used as '{}'",
            profile.id,
            profile.provider,
            provider.id()
        )));
    }
    let audio = synthesize_provider_bounded(state, provider, provider_request)
        .await
        .map_err(provider_error_to_app)?;
    let audio_bytes = super::tts::encode_output(audio.samples, audio.sample_rate, format).await?;
    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        Body::from(audio_bytes),
    )
        .into_response())
}

/// Named-voice providers (Kyutai TTS): the voice is a file stem in the provider's
/// voices directory; the provider validates it. No voice profile is involved.
async fn post_speech_named_voice(
    state: &crate::AppState,
    request: &SpeechRequest,
    format: audio::AudioFormat,
) -> Result<axum::response::Response, AppError> {
    use super::providers::{build_registry_for_state, provider_error_to_app};
    use super::tts::synthesize_provider_bounded;
    let config = state.config.get().await;
    let language = request.language.clone().unwrap_or_else(|| "en".to_string());
    let speech = request.speech(&config, &language)?;
    let mut provider_request =
        super::speech_prep::provider_request(&request.model, &request.voice, speech, language);
    provider_request.device = sonicboom::tts::providers::types::DevicePreference::Auto;
    provider_request.seed = request.seed;
    let registry = build_registry_for_state(state).await;
    let provider = registry
        .resolve_model(&request.model)
        .map_err(provider_error_to_app)?;
    let audio = synthesize_provider_bounded(state, provider, provider_request)
        .await
        .map_err(provider_error_to_app)?;
    let audio_bytes = super::tts::encode_output(audio.samples, audio.sample_rate, format).await?;
    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        Body::from(audio_bytes),
    )
        .into_response())
}

/// Synthesize speech using OpenAI-compatible API
pub async fn post_speech(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    body: String,
) -> Result<axum::response::Response, AppError> {
    super::jobs::with_admission(&state, post_speech_inner(token, State(state.clone()), body)).await
}

async fn post_speech_inner(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    body: String,
) -> Result<axum::response::Response, AppError> {
    check_rate_limit(&state, &token)?;

    let request: SpeechRequest = serde_json::from_str(&body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON in request body: {e}")))?;

    validate_model(&request.model)?;
    require_compiled_provider(&request.model)?;
    validate_speed(request.speed)?;
    let format = audio::AudioFormat::parse(&request.response_format)
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    // Cloning models route to their native provider with an authorized
    // voice profile; legacy aliases keep the exact Supertonic path below.
    if is_cloning_model(&request.model) {
        check_rate_limit(&state, &token)?;
        return super::jobs::with_reference_memory(
            &state,
            post_speech_cloning(&token, &state, &request, format),
        )
        .await;
    }
    if is_named_voice_model(&request.model) {
        return post_speech_named_voice(&state, &request, format).await;
    }
    let voice_name = map_voice(&request.voice)?;

    // TTS limits are read per request (spec §46).
    let config = state.config.get().await;
    sonicboom::engine::validate_tts_input(
        &request.input,
        "en",
        &request.voice,
        config.model.inference_steps,
        config.tts.max_text_length,
    )
    .map_err(|e| AppError::BadRequest(e.to_string()))?;
    // Legacy Supertonic aliases synthesize English; the text is prepared
    // once for that language before the model check.
    let lang = "en".to_string(); // Default language
    let speech = request.speech(&config, &lang)?;

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

    // Verify voice exists, fallback to default
    let voice_name = if model_handle.voice_styles.contains_key(&voice_name) {
        voice_name
    } else {
        model_handle
            .default_voice()
            .ok_or_else(|| AppError::internal("no voice styles available"))?
            .to_string()
    };

    let sample_rate = model_handle.sample_rate();

    let samples = synthesize_bounded(&state, model_handle, speech, lang, voice_name).await?;

    let audio_bytes = super::tts::encode_output(samples, sample_rate, format).await?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        Body::from(audio_bytes),
    )
        .into_response())
}

/// List available models (OpenAI-compatible endpoint)
///
/// Public metadata endpoint: exposes only non-sensitive model identifiers
/// (see docs). Authentication is intentionally not required. Built from
/// stable identity metadata: recognized IDs are listed even when a backend
/// is not compiled or ready. This endpoint is not a readiness guarantee.
/// Synthesis returns 503 for feature-disabled providers.
pub async fn get_models() -> impl IntoResponse {
    let data: Vec<serde_json::Value> = SUPPORTED_OPENAI_MODELS
        .iter()
        .map(|id| {
            let owned_by = match model_provider(id) {
                Some(ModelProvider::QwenGguf) => "local:qwen3-gguf",
                Some(ModelProvider::PocketTts) => "local:pocket-tts",
                Some(ModelProvider::QwenCandle) => "local:qwen3-candle",
                Some(ModelProvider::KyutaiCandle) => "local:kyutai-candle",
                _ => "local",
            };
            serde_json::json!({
                "id": id,
                "object": "model",
                "created": 1704067200,
                "owned_by": owned_by,
                "permission": [],
                "root": id,
                "parent": null,
            })
        })
        .collect();
    let body = serde_json::json!({ "object": "list", "data": data }).to_string();
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
        for model in SUPPORTED_OPENAI_MODELS {
            assert!(validate_model(model).is_ok(), "model {model} rejected");
        }
        // Legacy aliases stay Supertonic; new IDs route to cloning providers.
        for model in ["tts-1", "tts-1-hd", "supertonic-3"] {
            assert!(!is_cloning_model(model), "{model} must stay supertonic");
        }
        for model in ["qwen3-tts-1.7b-base", "pocket-tts-en", "pocket-tts-es"] {
            assert!(
                is_cloning_model(model),
                "{model} must route to a cloning provider"
            );
        }
        // Retired GPT-SoVITS IDs are rejected, never rerouted.
        for model in RETIRED_GPT_SOVITS_MODELS {
            assert!(!is_cloning_model(model), "{model} must stay retired");
            assert!(validate_model(model).is_err(), "{model} must be rejected");
        }
        // The removed 0.6B Candle model is neither listed nor cloning.
        assert!(validate_model("qwen3-tts-0.6b-base").is_err());
        assert!(!is_cloning_model("qwen3-tts-0.6b-base"));
    }

    #[tokio::test]
    async fn model_ownership_and_feature_disabled_errors() {
        for (model, provider) in [
            ("tts-1", ModelProvider::Supertonic),
            ("qwen3-tts-1.7b-base", ModelProvider::QwenGguf),
            ("pocket-tts-en", ModelProvider::PocketTts),
            ("pocket-tts-es", ModelProvider::PocketTts),
        ] {
            assert_eq!(model_provider(model), Some(provider));
            let result = require_compiled_provider(model);
            if provider.compiled() {
                assert!(result.is_ok());
            } else {
                let response = result.unwrap_err().into_response();
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let body = String::from_utf8(body.to_vec()).unwrap();
                assert!(body.contains(provider.id()));
                assert!(body.contains("not compiled"));
            }
        }
        assert_eq!(model_provider("unknown"), None);
        for model in RETIRED_GPT_SOVITS_MODELS {
            assert_eq!(model_provider(model), None);
        }
        let response = get_models().await.into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let qwen = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == "qwen3-tts-1.7b-base")
            .unwrap();
        assert_eq!(qwen["owned_by"], "local:qwen3-gguf");
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
        // New optional fields default to absent without changing old defaults.
        assert_eq!(request.language, None);
        assert_eq!(request.seed, None);
        assert_eq!(request.normalization, None);
        assert_eq!(request.locale, None);
        assert_eq!(request.reading_style, None);
        // Standard OpenAI clients may send fields SonicBoom does not know.
        let request: SpeechRequest =
            serde_json::from_str(r#"{"input":"hi","instructions":"calm","stream_format":"audio"}"#)
                .unwrap();
        assert_eq!(request.input, "hi");
    }

    #[test]
    fn legacy_and_cloning_paths_prepare_text_once_with_request_options() {
        let config = crate::api::tests::test_config();
        let request: SpeechRequest = serde_json::from_str(
            r#"{"input":"  Pago 26/10/2026  ","normalization":"auto","locale":"es-PE"}"#,
        )
        .unwrap();
        let speech = request.speech(&config, "es").unwrap();
        assert_eq!(
            speech.as_str(),
            "Pago veintiséis de octubre de dos mil veintiséis"
        );
        assert_eq!(speech.identity(), "tn-v2.1|auto|es|es-PE|precise");
        // Legacy aliases synthesize English: a Spanish locale is a 400.
        assert!(matches!(
            request.speech(&config, "en"),
            Err(AppError::BadRequest(_))
        ));
        let off: SpeechRequest =
            serde_json::from_str(r#"{"input":"  Pago 26/10/2026  ","normalization":"OFF"}"#)
                .unwrap();
        assert_eq!(
            off.speech(&config, "es").unwrap().as_str(),
            "Pago 26/10/2026"
        );
        let invalid: SpeechRequest =
            serde_json::from_str(r#"{"input":"hi","normalization":"strict"}"#).unwrap();
        assert!(matches!(
            invalid.speech(&config, "en"),
            Err(AppError::BadRequest(_))
        ));
        let natural: SpeechRequest =
            serde_json::from_str(r#"{"input":"12:30","language":"es","reading_style":"natural"}"#)
                .unwrap();
        let speech = natural.speech(&config, "es").unwrap();
        assert_eq!(speech.as_str(), "doce y media");
        assert_eq!(speech.identity(), "tn-v2.1|auto|es|es-ES|natural");
        let invalid: SpeechRequest =
            serde_json::from_str(r#"{"input":"hi","reading_style":"loud"}"#).unwrap();
        assert!(matches!(
            invalid.speech(&config, "en"),
            Err(AppError::BadRequest(_))
        ));
    }

    #[test]
    fn cloning_requests_keep_the_reference_transcript_byte_identical() {
        let config = crate::api::tests::test_config();
        let now = chrono::Utc::now();
        let transcript = "Reference: 12 words, 3/4 of them; ¿sí? 26/10/2026".to_string();
        let profile = sonicboom::tts::voices::VoiceProfile {
            id: uuid::Uuid::new_v4().to_string(),
            name: "Synthetic authorized fixture".into(),
            provider: "qwen3-gguf".into(),
            model: "qwen3-tts-1.7b-base".into(),
            model_revision: "test".into(),
            language: "es".into(),
            clone_mode: "icl".into(),
            reference_hash: "synthetic-fixture".into(),
            reference_sample_rate: 24_000,
            reference_duration_secs: 3.0,
            transcript: Some(transcript.clone()),
            owner_token_hash: Some("test-owner".into()),
            consent: sonicboom::tts::voices::VoiceConsent {
                basis: "owner".into(),
                note: String::new(),
                created_at: now,
            },
            created_at: now,
            updated_at: now,
        };
        let reference = sonicboom::tts::providers::types::ReferenceAudio {
            pcm: vec![0.25; 2400],
            sample_rate: 24_000,
        };
        let request: SpeechRequest = serde_json::from_str(
            r#"{"model":"qwen3-tts-1.7b-base","input":"Tengo 3 gatos el 26/10/2026","seed":7}"#,
        )
        .unwrap();
        let speech = request.speech(&config, &profile.language).unwrap();
        let identity = speech.identity().to_string();
        let built = cloning_provider_request(
            &profile,
            reference.clone(),
            speech,
            profile.language.clone(),
            request.seed,
        )
        .unwrap();
        assert!(built.text.contains("veintiséis de octubre"));
        assert_ne!(built.text, request.input);
        assert_eq!(built.speech_identity.as_deref(), Some(identity.as_str()));
        assert_eq!(
            built.reference_transcript.as_deref().map(str::as_bytes),
            Some(transcript.as_bytes())
        );
        assert_eq!(built.reference_audio.as_ref().unwrap().pcm, reference.pcm);
        assert_eq!(built.seed, Some(7));
        assert_eq!(built.clone_mode.unwrap().as_str(), "icl");
        assert_eq!(built.language, "es");
    }
}
