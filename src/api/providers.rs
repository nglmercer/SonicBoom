//! Provider-aware HTTP surface: registry wiring, status and discovery.
//!
//! Built per request from live configuration plus the current Supertonic
//! handle snapshot, so config hot reload applies without restarts and a
//! failed optional provider can never take down Supertonic.

use std::sync::Arc;

use sonicboom::tts::model::ModelHandle;
use sonicboom::tts::providers::TtsProviderRegistry;
use sonicboom::tts::providers::supertonic::SupertonicProvider;
use sonicboom::tts::providers::types::DevicePreference;

#[cfg(test)]
use crate::config::model::AppConfig;

fn parse_device(raw: &str) -> DevicePreference {
    DevicePreference::parse(raw).unwrap_or(DevicePreference::Auto)
}

/// Map `[providers.qwen3_candle]` / `[providers.kyutai_candle]` to the runtime config.
#[cfg(any(feature = "candle-qwen", feature = "candle-kyutai"))]
pub fn map_candle_config(
    c: &crate::config::model::CandleProviderConfig,
) -> sonicboom::tts::providers::candle_models::CandleProviderConfig {
    sonicboom::tts::providers::candle_models::CandleProviderConfig {
        enabled: c.enabled,
        model_dir: crate::config::paths::resolve_model_dir(&c.model_dir),
        voices_dir: crate::config::paths::resolve_model_dir(&c.voices_dir),
        default_voice: c.default_voice.clone(),
        device: c.device.trim().to_ascii_lowercase(),
        download: c.download,
    }
}

/// Sync both candle providers from the live configuration.
pub fn sync_candle_providers(
    pool: &sonicboom::tts::providers::ProviderPool,
    providers: &crate::config::model::ProvidersConfig,
) {
    #[cfg(feature = "candle-qwen")]
    pool.sync_qwen_candle_config(map_candle_config(&providers.qwen3_candle));
    #[cfg(feature = "candle-kyutai")]
    pool.sync_kyutai_candle_config(map_candle_config(&providers.kyutai_candle));
    #[cfg(not(any(feature = "candle-qwen", feature = "candle-kyutai")))]
    let _ = (pool, providers);
}

#[cfg(feature = "qwen-gguf")]
pub fn map_qwen_gguf_config(
    gguf: &crate::config::model::Qwen3GgufConfig,
) -> sonicboom::tts::providers::qwen_gguf::provider::QwenGgufConfig {
    use sonicboom::tts::providers::qwen_gguf::worker::Device;
    sonicboom::tts::providers::qwen_gguf::provider::QwenGgufConfig {
        enabled: gguf.enabled,
        model_dir: crate::config::paths::resolve_model_dir(&gguf.model_dir),
        device: match gguf.device.trim().to_ascii_lowercase().as_str() {
            "cpu" => Device::Cpu,
            "vulkan" => Device::Vulkan,
            "cuda" => Device::Cuda,
            _ => Device::Auto,
        },
        gpu_layers: gguf.gpu_layers,
        threads: gguf.threads.max(1),
        max_tokens: gguf.max_tokens.max(1),
        cache_capacity: gguf.cache_entries.max(1),
        allow_cpu_fallback: gguf.allow_cpu_fallback,
        sampling: gguf.sampling.clone(),
        quality: {
            use sonicboom::tts::quality::{DetectionMode, QualityPolicy};
            let mode = if gguf.quality.enabled {
                DetectionMode::parse(&gguf.quality.policy).unwrap_or(DetectionMode::Report)
            } else {
                DetectionMode::Off
            };
            QualityPolicy::new(mode, gguf.quality.max_retries)
        },
    }
}

pub fn map_pocket_config(
    pocket: &crate::config::model::PocketTtsConfig,
) -> sonicboom::tts::providers::pocket_tts::PocketTtsConfig {
    sonicboom::tts::providers::pocket_tts::PocketTtsConfig {
        enabled: pocket.enabled,
        model_dir: pocket.model_dir.clone(),
        model_id: pocket.model_id.clone(),
        revision: pocket.revision.clone(),
        device: parse_device(&pocket.device),
        cache_capacity: pocket.cache_entries.max(1),
        temperature: pocket.temperature,
        lsd_steps: pocket.lsd_steps.max(1),
        cpu_threads: pocket.cpu_threads.clamp(1, 32),
    }
}

/// Build a request-scoped registry from live config, the resident provider
/// pool, and the current Supertonic handle (if any). Singleton configs sync
/// from the live configuration first (hot reload without restarts), and an
/// enabled-but-unloaded provider kicks its background download + load.
/// Optional providers are always registered; when disabled or uninstalled
/// they report that state explicitly and reject synthesis with actionable
/// errors. One failing optional provider can never take down Supertonic.
#[cfg(test)]
pub fn build_registry(
    config: &AppConfig,
    supertonic_handle: Option<Arc<ModelHandle>>,
    pool: &sonicboom::tts::providers::ProviderPool,
) -> TtsProviderRegistry {
    pool.sync_pocket_config(map_pocket_config(&config.providers.pocket_tts));
    #[cfg(feature = "qwen-gguf")]
    pool.sync_gguf_config(map_qwen_gguf_config(&config.providers.qwen3_gguf));
    sync_candle_providers(pool, &config.providers);
    // Hot-reload path: no HF token at request time (public bundles need
    // none; gated bundles report an explicit auth error in status).
    pool.kick_ensure_if_needed(None);

    let supertonic = Arc::new(SupertonicProvider::new(config.model.cache_dir.clone()));
    if let Some(handle) = supertonic_handle {
        supertonic.attach_ready(handle, &config.model.revision);
    }
    pool.registry(supertonic)
}

/// Versioned request registry: stale request snapshots cannot roll back a
/// resident provider after another request adopted a newer configuration.
pub async fn build_registry_for_state(state: &crate::AppState) -> TtsProviderRegistry {
    let (generation, config) = state.config.versioned_snapshot().await;
    state
        .providers
        .sync_versioned(generation, map_pocket_config(&config.providers.pocket_tts));
    #[cfg(feature = "qwen-gguf")]
    state
        .providers
        .sync_gguf_config(map_qwen_gguf_config(&config.providers.qwen3_gguf));
    sync_candle_providers(&state.providers, &config.providers);
    state.providers.kick_ensure_if_needed(None);
    let supertonic = Arc::new(SupertonicProvider::new(config.model.cache_dir.clone()));
    if let Some(handle) = supertonic_handle(state).await {
        supertonic.attach_ready(handle, &config.model.revision);
    }
    state.providers.registry(supertonic)
}

/// Snapshot the live Supertonic handle (if the model is ready).
pub async fn supertonic_handle(state: &crate::AppState) -> Option<Arc<ModelHandle>> {
    let status = state.model_status.read().await;
    match &*status {
        sonicboom::tts::ModelStatus::Ready(handle) => Some(Arc::clone(handle)),
        _ => None,
    }
}

/// `GET /api/providers`: authenticated capability + status discovery.
pub async fn get_providers(
    _token: crate::auth::AuthenticatedToken,
    axum::extract::State(state): axum::extract::State<crate::AppState>,
) -> impl axum::response::IntoResponse {
    use axum::http::{StatusCode, header};
    let registry = build_registry_for_state(&state).await;
    let capabilities = [
        registry.get("supertonic").map(|p| p.capabilities()),
        registry.get("pocket-tts").map(|p| p.capabilities()),
    ]
    .into_iter()
    .chain(
        cfg!(feature = "qwen-gguf").then(|| registry.get("qwen3-gguf").map(|p| p.capabilities())),
    )
    .chain(
        cfg!(feature = "candle-qwen")
            .then(|| registry.get("qwen3-candle").map(|p| p.capabilities())),
    )
    .chain(
        cfg!(feature = "candle-kyutai")
            .then(|| registry.get("kyutai-candle").map(|p| p.capabilities())),
    )
    .collect::<Vec<_>>();
    let body = serde_json::json!({
        "providers": registry.statuses(),
        "capabilities": capabilities,
    })
    .to_string();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
}

/// `GET /api/providers/{id}/status`: one provider's lifecycle status.
pub async fn get_provider_status(
    _token: crate::auth::AuthenticatedToken,
    axum::extract::State(state): axum::extract::State<crate::AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl axum::response::IntoResponse, crate::error::AppError> {
    use axum::http::{StatusCode, header};
    let registry = build_registry_for_state(&state).await;
    match registry.get(id.trim()) {
        Some(provider) => {
            let body = serde_json::to_string(&provider.status())
                .map_err(|e| crate::error::AppError::internal(e.to_string()))?;
            Ok((
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                body,
            ))
        }
        None => Err(crate::error::AppError::BadRequest(format!(
            "unknown provider '{id}' (expected one of: supertonic, pocket-tts{})",
            if cfg!(feature = "qwen-gguf") {
                ", qwen3-gguf"
            } else {
                ""
            }
        ))),
    }
}

/// Map a typed provider error onto the existing HTTP error policy.
pub fn provider_error_to_app(
    error: sonicboom::tts::providers::ProviderError,
) -> crate::error::AppError {
    use sonicboom::tts::providers::ProviderErrorKind;
    match error.kind() {
        ProviderErrorKind::UnsupportedModel
        | ProviderErrorKind::UnsupportedLanguage
        | ProviderErrorKind::InvalidVoice => {
            crate::error::AppError::BadRequest(error.message().to_string())
        }
        ProviderErrorKind::Busy => {
            crate::error::AppError::TooManyRequests(error.message().to_string())
        }
        ProviderErrorKind::ModelMissing
        | ProviderErrorKind::Loading
        | ProviderErrorKind::DeviceUnavailable => {
            crate::error::AppError::ServiceUnavailable(error.message().to_string())
        }
        ProviderErrorKind::InferenceFailed if error.message().contains("maximum frame limit without EOS") => {
            crate::error::AppError::ServiceUnavailable("Generation reached the configured frame limit without EOS; audio was withheld. Use shorter text or increase providers.qwen3_gguf.max_tokens within its validated range.".into())
        }
        ProviderErrorKind::InferenceFailed => {
            crate::error::AppError::internal(error.message().to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lists_providers_with_explicit_devices() {
        let config = AppConfig::default();
        let pool = sonicboom::tts::providers::ProviderPool::new();
        let registry = build_registry(&config, None, &pool);
        // The Candle provider is gone: no "qwen3" id may appear.
        assert!(!registry.provider_ids().contains(&"qwen3".to_string()));
        // GPT-SoVITS is retired: it must never appear as a provider id.
        assert!(!registry.provider_ids().contains(&"gpt-sovits".to_string()));
        let mut expected = vec!["pocket-tts".to_string(), "supertonic".to_string()];
        #[cfg(feature = "qwen-gguf")]
        expected.push("qwen3-gguf".to_string());
        #[cfg(feature = "candle-qwen")]
        expected.push("qwen3-candle".to_string());
        #[cfg(feature = "candle-kyutai")]
        expected.push("kyutai-candle".to_string());
        expected.sort();
        assert_eq!(registry.provider_ids(), expected);
        for status in registry.statuses() {
            if status.provider_id == "supertonic" {
                // No model loaded in this test: the engine device is still the CPU default.
                assert_eq!(status.effective_device.as_deref(), Some("cpu"));
            } else {
                // Optional providers are disabled by default.
                assert!(!status.enabled);
            }
        }
    }

    #[test]
    fn unknown_provider_status_is_rejected() {
        let config = AppConfig::default();
        let pool = sonicboom::tts::providers::ProviderPool::new();
        let registry = build_registry(&config, None, &pool);
        assert!(registry.get("nope").is_none());
    }

    #[test]
    fn enabling_a_provider_without_weights_reports_not_installed() {
        // Hot reload path: flipping `enabled` takes effect on the next
        // request-scoped registry build; without installed weights the
        // provider reports `not_installed` (never a silent fallback) and
        // Supertonic keeps serving.
        // Uses an isolated empty dir: the workspace `./models` may hold
        // real downloaded weights, which must not flip this expectation.
        let dir = std::env::temp_dir().join(format!("pocket-no-weights-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut config = AppConfig::default();
        config.providers.pocket_tts.enabled = true;
        config.providers.pocket_tts.model_dir = dir.clone();
        // Built without the background kick on purpose: kicking would
        // start a real download in this test.
        let pool = sonicboom::tts::providers::ProviderPool::new();
        pool.sync_pocket_config(map_pocket_config(&config.providers.pocket_tts));
        let supertonic =
            std::sync::Arc::new(SupertonicProvider::new(config.model.cache_dir.clone()));
        let registry = pool.registry(supertonic);
        let status = registry.get("pocket-tts").unwrap().status();
        assert!(status.enabled);
        assert_eq!(
            status.state,
            sonicboom::tts::providers::types::ProviderStatusState::NotInstalled
        );
        assert_eq!(status.effective_device.as_deref(), Some("cpu"));
        assert!(registry.get("supertonic").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "qwen-gguf")]
    #[test]
    fn enabling_gguf_without_weights_reports_not_installed() {
        // Same hot-reload contract as the other native providers: enabled
        // but no verified bundle means `not_installed`, and the 1.7B model
        // stays routable to its owning provider while Supertonic keeps serving.
        let dir =
            std::env::temp_dir().join(format!("qwen-gguf-no-weights-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut config = AppConfig::default();
        config.providers.qwen3_gguf.enabled = true;
        config.providers.qwen3_gguf.model_dir = dir.clone();
        let pool = sonicboom::tts::providers::ProviderPool::new();
        pool.sync_gguf_config(map_qwen_gguf_config(&config.providers.qwen3_gguf));
        let supertonic =
            std::sync::Arc::new(SupertonicProvider::new(config.model.cache_dir.clone()));
        let registry = pool.registry(supertonic);
        let status = registry.get("qwen3-gguf").unwrap().status();
        assert!(status.enabled);
        assert_eq!(
            status.state,
            sonicboom::tts::providers::types::ProviderStatusState::NotInstalled
        );
        assert_eq!(
            registry.resolve_model("qwen3-tts-1.7b-base").unwrap().id(),
            "qwen3-gguf"
        );
        assert!(registry.get("supertonic").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
