//! Provider registry keyed by provider ID + model ID.
//!
//! No hardcoded `match` on model names at call sites: routing, capability
//! validation, readiness and effective-device reporting all go through the
//! registry. One failing optional provider can never take down Supertonic.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::TtsProvider;
use super::error::{ProviderError, ProviderErrorKind};
use super::types::{ProviderRequest, ProviderStatus};

/// Registry of native TTS providers.
#[derive(Default)]
pub struct TtsProviderRegistry {
    providers: RwLock<HashMap<String, Arc<dyn TtsProvider>>>,
    model_index: RwLock<HashMap<String, String>>,
}

impl TtsProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            model_index: RwLock::new(HashMap::new()),
        }
    }

    /// Register a provider and index every model ID it advertises.
    pub fn register(&self, provider: Arc<dyn TtsProvider>) {
        let id = provider.id().to_string();
        for model in provider.capabilities().models.clone() {
            if let Ok(mut index) = self.model_index.write() {
                index.insert(model, id.clone());
            }
        }
        if let Ok(mut providers) = self.providers.write() {
            providers.insert(id, provider);
        }
    }

    pub fn get(&self, provider_id: &str) -> Option<Arc<dyn TtsProvider>> {
        self.providers.read().ok()?.get(provider_id).cloned()
    }

    /// Resolve which provider owns `model_id`. Unknown models are an
    /// explicit error, never a silent fallback to another provider.
    pub fn resolve_model(&self, model_id: &str) -> Result<Arc<dyn TtsProvider>, ProviderError> {
        let owner = self
            .model_index
            .read()
            .ok()
            .and_then(|index| index.get(model_id).cloned());
        match owner {
            Some(id) => self.get(&id).ok_or_else(|| {
                ProviderError::new(
                    ProviderErrorKind::ModelMissing,
                    format!("model '{model_id}' is indexed but provider '{id}' is unavailable"),
                )
            }),
            None => {
                let known: Vec<String> = self
                    .model_index
                    .read()
                    .map(|index| {
                        let mut models: Vec<String> = index.keys().cloned().collect();
                        models.sort();
                        models
                    })
                    .unwrap_or_default();
                Err(ProviderError::unsupported_model(
                    "<registry>",
                    model_id,
                    format!("unknown model (known models: {})", known.join(", ")),
                ))
            }
        }
    }

    /// Blocking synthesis routed by `request.model` (preferred) or by an
    /// explicit provider-keyed model prefix. Empty model selects the
    /// default Supertonic provider for backward compatibility.
    pub fn synthesize(
        &self,
        request: &ProviderRequest,
    ) -> Result<super::types::PcmAudio, ProviderError> {
        if request.model.trim().is_empty() {
            let provider = self.get("supertonic").ok_or_else(|| {
                ProviderError::model_missing("default supertonic provider is not registered")
            })?;
            return provider.synthesize(request);
        }
        let provider = self.resolve_model(&request.model)?;
        provider.synthesize(request)
    }

    pub fn statuses(&self) -> Vec<ProviderStatus> {
        self.providers
            .read()
            .map(|providers| {
                let mut out: Vec<ProviderStatus> = providers.values().map(|p| p.status()).collect();
                out.sort_by_key(|status| status.provider_id.clone());
                out
            })
            .unwrap_or_default()
    }

    pub fn provider_ids(&self) -> Vec<String> {
        self.providers
            .read()
            .map(|p| {
                let mut ids: Vec<String> = p.keys().cloned().collect();
                ids.sort();
                ids
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::providers::types::{
        CloneMode, DevicePreference, PcmAudio, ProviderCapabilities, ProviderStatusState,
    };

    struct FakeProvider {
        id: &'static str,
        models: Vec<String>,
        fail: Option<ProviderError>,
    }

    impl TtsProvider for FakeProvider {
        fn id(&self) -> &'static str {
            self.id
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                provider_id: self.id.to_string(),
                models: self.models.clone(),
                languages: vec!["en".to_string()],
                clone_modes: vec![],
                devices: vec!["cpu".to_string(), "auto".to_string()],
                requires_reference_audio: false,
                requires_reference_transcript: false,
                voices: vec![],
                sample_rate: Some(24000),
            }
        }

        fn status(&self) -> ProviderStatus {
            ProviderStatus {
                provider_id: self.id.to_string(),
                enabled: true,
                state: if self.fail.is_some() {
                    ProviderStatusState::Error
                } else {
                    ProviderStatusState::Ready
                },
                active_model: self.models.first().cloned(),
                effective_device: Some("cpu".to_string()),
                model_revision: None,
                backend_diagnostics: None,
                detail: self.fail.as_ref().map(|e| e.message().to_string()),
            }
        }

        fn load(&self, _model_id: &str) -> Result<(), ProviderError> {
            Ok(())
        }

        fn unload(&self) {}

        fn synthesize(&self, request: &ProviderRequest) -> Result<PcmAudio, ProviderError> {
            if let Some(error) = &self.fail {
                return Err(error.clone());
            }
            if request.text.trim().is_empty() {
                return Err(ProviderError::invalid_voice("text cannot be empty"));
            }
            Ok(PcmAudio::mono(vec![0.25; 240], 24000))
        }
    }

    fn registry() -> TtsProviderRegistry {
        let registry = TtsProviderRegistry::new();
        registry.register(Arc::new(FakeProvider {
            id: "supertonic",
            models: vec!["supertonic-3".to_string()],
            fail: None,
        }));
        registry.register(Arc::new(FakeProvider {
            id: "qwen3",
            models: vec!["qwen3-tts-0.6b-base".to_string()],
            fail: None,
        }));
        registry
    }

    #[test]
    fn unknown_models_are_rejected_with_known_model_list() {
        let registry = registry();
        let mut request = ProviderRequest::new("no-such-model", "voice", "hello", "en");
        request.device = DevicePreference::Auto;
        let error = registry.synthesize(&request).unwrap_err();
        assert_eq!(error.code(), "unsupported_model");
        assert_eq!(error.http_status(), 400);
        assert!(error.message().contains("supertonic-3"));
        assert!(error.message().contains("qwen3-tts-0.6b-base"));
    }

    #[test]
    fn routing_selects_the_owning_provider() {
        let registry = registry();
        let request = ProviderRequest::new("qwen3-tts-0.6b-base", "voice", "hello", "en");
        assert!(registry.synthesize(&request).is_ok());
    }

    #[test]
    fn provider_failure_does_not_remove_sibling_providers() {
        let registry = TtsProviderRegistry::new();
        registry.register(Arc::new(FakeProvider {
            id: "supertonic",
            models: vec!["supertonic-3".to_string()],
            fail: None,
        }));
        registry.register(Arc::new(FakeProvider {
            id: "qwen3",
            models: vec!["qwen3-tts-0.6b-base".to_string()],
            fail: Some(ProviderError::model_missing("weights absent")),
        }));
        let failing = ProviderRequest::new("qwen3-tts-0.6b-base", "v", "hi", "en");
        let error = registry.synthesize(&failing).unwrap_err();
        assert_eq!(error.code(), "model_missing");
        let healthy = ProviderRequest::new("supertonic-3", "F1", "hi", "en");
        assert!(registry.synthesize(&healthy).is_ok());
        assert_eq!(registry.statuses().len(), 2);
    }

    #[test]
    fn empty_model_selects_supertonic_for_backward_compatibility() {
        let registry = registry();
        let request = ProviderRequest::new("", "F1", "hello", "en");
        assert!(registry.synthesize(&request).is_ok());
        let _ = CloneMode::Icl;
    }
}
