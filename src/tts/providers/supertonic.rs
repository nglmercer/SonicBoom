//! Supertonic 3 provider adapter.
//!
//! Wraps the existing `ort`-based Supertonic 3 pipeline behind the native
//! provider contract. All ten style voices, `alloy` aliases, output formats,
//! language defaults and inference behavior are preserved exactly.

use std::sync::{Arc, RwLock};

use super::TtsProvider;
use super::error::ProviderError;
use super::types::{
    DevicePreference, EffectiveDevice, PcmAudio, ProviderCapabilities, ProviderRequest,
    ProviderStatus, ProviderStatusState,
};

/// Canonical Supertonic model identifier.
pub const SUPERTONIC_MODEL_ID: &str = "supertonic-3";
/// Legacy OpenAI aliases that select Supertonic (preserved behavior).
pub const SUPERTONIC_ALIASES: &[&str] = &["tts-1", "tts-1-hd", "supertonic-3"];
/// Existing studio language choices, advertised as metadata. Native language
/// tag acceptance and inference preprocessing remain unchanged.
pub const SUPERTONIC_LANGUAGE_CHOICES: &[&str] = &[
    "en", "ko", "ja", "zh", "es", "pt", "fr", "de", "it", "ru", "ar", "hi", "nl", "pl", "tr", "vi",
    "id", "th", "sv", "da", "fi", "el", "cs", "sk", "sl", "hr", "ro", "hu", "bg", "uk", "lt", "lv",
    "et",
];
/// All ten built-in style voices.
pub const SUPERTONIC_VOICES: &[&str] =
    &["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"];

/// Map legacy/OpenAI voice names to Supertonic style voices.
/// Unknown names are rejected explicitly, never defaulted silently.
pub fn map_supertonic_voice(name: &str) -> Result<String, ProviderError> {
    match name {
        "alloy" => Ok("M1".to_string()),
        "echo" => Ok("M2".to_string()),
        "fable" => Ok("M3".to_string()),
        "onyx" => Ok("M4".to_string()),
        "nova" => Ok("F1".to_string()),
        "shimmer" => Ok("F2".to_string()),
        _ => {
            if SUPERTONIC_VOICES.contains(&name) {
                return Ok(name.to_string());
            }
            Err(ProviderError::invalid_voice(format!(
                "unsupported voice '{name}' for provider 'supertonic' \
                 (expected one of: alloy, echo, fable, onyx, nova, shimmer, F1-F5, M1-M5)"
            )))
        }
    }
}

/// Supertonic provider state.
struct SupertonicState {
    handle: Option<Arc<crate::tts::model::ModelHandle>>,
    active_model: Option<String>,
    load_error: Option<String>,
}

/// Native provider adapter around the existing Supertonic 3 model handle.
pub struct SupertonicProvider {
    state: RwLock<SupertonicState>,
}

impl SupertonicProvider {
    pub fn new(_model_dir: std::path::PathBuf) -> Self {
        Self {
            state: RwLock::new(SupertonicState {
                handle: None,
                active_model: None,
                load_error: None,
            }),
        }
    }

    /// Attach an already-loaded model handle (startup / reload path stages
    /// candidates and atomically swaps ready instances; on reload failure
    /// the last working handle stays live).
    pub fn attach_ready(&self, handle: Arc<crate::tts::model::ModelHandle>, model_id: &str) {
        if let Ok(mut state) = self.state.write() {
            state.handle = Some(handle);
            state.active_model = Some(model_id.to_string());
            state.load_error = None;
        }
    }

    pub fn attach_error(&self, detail: String) {
        if let Ok(mut state) = self.state.write() {
            state.load_error = Some(detail);
        }
    }

    fn handle(&self) -> Result<Arc<crate::tts::model::ModelHandle>, ProviderError> {
        self.state
            .read()
            .ok()
            .and_then(|state| state.handle.clone())
            .ok_or_else(|| {
                if let Some(detail) = self
                    .state
                    .read()
                    .ok()
                    .and_then(|state| state.load_error.clone())
                {
                    ProviderError::inference_failed(format!(
                        "supertonic model unavailable: {detail}"
                    ))
                } else {
                    ProviderError::loading("supertonic model is not loaded yet")
                }
            })
    }
}

impl TtsProvider for SupertonicProvider {
    fn id(&self) -> &'static str {
        "supertonic"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            provider_id: "supertonic".to_string(),
            models: SUPERTONIC_ALIASES.iter().map(|s| s.to_string()).collect(),
            languages: SUPERTONIC_LANGUAGE_CHOICES
                .iter()
                .map(|language| language.to_string())
                .collect(),
            clone_modes: vec![],
            devices: vec!["cpu".to_string(), "auto".to_string()],
            requires_reference_audio: false,
            requires_reference_transcript: false,
            voices: vec![],
            sample_rate: None,
        }
    }

    fn status(&self) -> ProviderStatus {
        let (ready, active_model, detail) = self
            .state
            .read()
            .map(|state| {
                (
                    state.handle.is_some(),
                    state.active_model.clone(),
                    state.load_error.clone(),
                )
            })
            .unwrap_or((false, None, None));
        let backend = self
            .state
            .read()
            .ok()
            .and_then(|state| state.handle.as_ref().map(|h| h.backend_name()));
        let device = if backend == Some("candle-vulkan") {
            "vulkan"
        } else {
            EffectiveDevice::Cpu.as_str()
        };
        ProviderStatus {
            provider_id: "supertonic".to_string(),
            enabled: true,
            state: if ready {
                ProviderStatusState::Ready
            } else if detail.is_some() {
                ProviderStatusState::Error
            } else {
                ProviderStatusState::Loading
            },
            active_model,
            effective_device: Some(device.to_string()),
            model_revision: None,
            backend_diagnostics: backend.map(|b| serde_json::json!({ "engine": b })),
            detail,
        }
    }

    fn load(&self, model_id: &str) -> Result<(), ProviderError> {
        if !SUPERTONIC_ALIASES.contains(&model_id) {
            return Err(ProviderError::unsupported_model(
                "supertonic",
                model_id,
                format!("expected one of: {}", SUPERTONIC_ALIASES.join(", ")),
            ));
        }
        // Actual download + ONNX load stays on the existing ModelService
        // path, which calls `attach_ready` on success. Here we only validate
        // the model identifier so misconfiguration fails fast.
        Ok(())
    }

    fn unload(&self) {
        if let Ok(mut state) = self.state.write() {
            state.handle = None;
            state.active_model = None;
        }
    }

    fn synthesize(&self, request: &ProviderRequest) -> Result<PcmAudio, ProviderError> {
        if !SUPERTONIC_ALIASES.contains(&request.model.as_str()) && !request.model.trim().is_empty()
        {
            return Err(ProviderError::unsupported_model(
                "supertonic",
                &request.model,
                format!("expected one of: {}", SUPERTONIC_ALIASES.join(", ")),
            ));
        }
        if !matches!(
            request.device,
            DevicePreference::Cpu | DevicePreference::Auto
        ) {
            return Err(ProviderError::device_unavailable(
                "provider 'supertonic' supports devices 'cpu' and 'auto' on this build; \
                 GPU execution providers are opt-in ORT build features",
            ));
        }
        let voice = map_supertonic_voice(&request.voice)?;
        let text = request.text.trim();
        if text.is_empty() {
            return Err(ProviderError::invalid_voice("text cannot be empty"));
        }
        let handle = self.handle()?;
        if !handle.voice_styles.contains_key(&voice) {
            return Err(ProviderError::invalid_voice(format!(
                "voice '{voice}' is not installed in the loaded Supertonic model"
            )));
        }
        let samples = crate::tts::inference::synthesize(
            &handle,
            text,
            &request.language,
            &voice,
            5,
            crate::tts::text::DEFAULT_MAX_CHUNK_CHARS,
        )
        .map_err(|e| {
            ProviderError::inference_failed(format!("supertonic inference failed: {e}"))
        })?;
        Ok(PcmAudio::mono(samples, handle.sample_rate()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_voice_aliases_map_without_silent_defaults() {
        for (input, expected) in [
            ("alloy", "M1"),
            ("echo", "M2"),
            ("fable", "M3"),
            ("onyx", "M4"),
            ("nova", "F1"),
            ("shimmer", "F2"),
            ("M1", "M1"),
            ("F5", "F5"),
        ] {
            assert_eq!(map_supertonic_voice(input).unwrap(), expected);
        }
        assert!(map_supertonic_voice("banana").is_err());
        assert!(map_supertonic_voice("M0").is_err());
        assert!(map_supertonic_voice("m1").is_err());
    }

    #[test]
    fn unloaded_supertonic_reports_loading_not_silent_audio() {
        let provider = SupertonicProvider::new(std::path::PathBuf::from("./models"));
        let request = ProviderRequest::new("supertonic-3", "F1", "hello", "en");
        let error = provider.synthesize(&request).unwrap_err();
        assert_eq!(error.code(), "loading");
        assert_eq!(error.http_status(), 503);
    }

    #[test]
    fn unknown_supertonic_model_is_rejected() {
        let provider = SupertonicProvider::new(std::path::PathBuf::from("./models"));
        let error = provider.load("qwen3-tts-0.6b-base").unwrap_err();
        assert_eq!(error.code(), "unsupported_model");
    }
}
