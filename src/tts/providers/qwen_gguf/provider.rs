//! `qwen3-gguf` provider: native 1.7B Base GGUF/ONNX inference behind the
//! shared [`TtsProvider`] contract.
//!
//! Backend identity is explicit: this provider owns `qwen3-tts-1.7b-base`
//! and nothing else. When the `qwen-gguf` Cargo feature is enabled the
//! Candle provider stops advertising the 1.7B identifier (see `QWEN_MODELS`),
//! so the registry can never index one model under two providers.
//!
//! Only validated ICL cloning is exposed. X-vector mode is rejected with
//! an explicit error until it is validated against the reference
//! implementation. Per-request device selection stays within
//! `cpu`/`auto`/`cuda` for API compatibility; `vulkan`/`cuda` execution
//! is selected in `[providers.qwen3_gguf]` and always reported back as the
//! effective device — never switched silently mid-request.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::super::assets;
use super::super::error::ProviderError;
use super::super::types::{
    CloneMode, DevicePreference, PcmAudio, ProviderCapabilities, ProviderRequest, ProviderStatus,
    ProviderStatusState, SamplingRequest,
};
use super::models;
use super::worker::{Device, Settings, Worker};
use crate::tts::quality::QualityPolicy;

/// Stable provider identifier for the native GGUF/ONNX backend.
pub const QWEN_GGUF_PROVIDER_ID: &str = "qwen3-gguf";
/// Model identifiers owned by this provider (exactly one: no collisions).
pub use crate::tts::providers::catalog::QWEN_GGUF_MODELS;
/// Languages with validated prompt token IDs (`conditioning::prompt`).
pub const QWEN_GGUF_LANGUAGES: &[&str] =
    &["en", "de", "es", "zh", "ja", "fr", "ko", "ru", "it", "pt"];
/// Output sample rate of the GGUF/ONNX codec path.
pub const QWEN_GGUF_SAMPLE_RATE: u32 = 24_000;
/// Minimum/maximum reference duration (seconds, matches worker bounds).
pub const QWEN_GGUF_REF_MIN_SECS: f64 = crate::tts::voices::REFERENCE_MIN_SECS;
pub const QWEN_GGUF_REF_MAX_SECS: f64 = crate::tts::voices::REFERENCE_MAX_SECS;

pub fn is_gguf_model(model: &str) -> bool {
    QWEN_GGUF_MODELS.contains(&model)
}

/// Runtime configuration (mirrors `[providers.qwen3_gguf]` in config.toml).
#[derive(Debug, Clone, PartialEq)]
pub struct QwenGgufConfig {
    pub enabled: bool,
    pub model_dir: PathBuf,
    pub device: Device,
    pub gpu_layers: i32,
    pub threads: usize,
    pub max_tokens: usize,
    pub cache_capacity: usize,
    /// Permit a CPU fallback when an explicit `device` cannot initialize.
    /// `Device::Auto` always permits it; explicit devices only when set.
    pub allow_cpu_fallback: bool,
    /// Default sampling (the `balanced` preset is the pinned upstream default).
    pub sampling: SamplingRequest,
    /// Quality detection/regeneration policy (default: detection off).
    pub quality: QualityPolicy,
}

impl Default for QwenGgufConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: PathBuf::from("./models/qwen3-tts-1.7b-gguf"),
            device: Device::Auto,
            gpu_layers: -1,
            threads: 4,
            max_tokens: 512,
            cache_capacity: 16,
            allow_cpu_fallback: false,
            sampling: SamplingRequest::default(),
            quality: QualityPolicy::default(),
        }
    }
}

struct ProviderState {
    loaded_model: Option<String>,
    effective_device: Option<String>,
    fallback_reason: Option<String>,
    load_error: Option<String>,
    ensuring: bool,
    download_progress: f32,
    /// A real network fetch started during this ensure pass. Hash
    /// verification alone advances `download_progress` too, so status
    /// lines consult this flag to say "verifying" instead of lying
    /// about "downloading" on warm boots.
    downloading: bool,
}

/// Resident singleton owning at most one native worker at a time.
///
/// The [`Worker`] is `Send` but not `Sync`; only the channel handle lives
/// here behind a mutex, so this provider stays `Sync`. Inference is
/// serialized by the worker's bounded queue plus `inference_lock`, which
/// also keeps `unload` from releasing the model mid-synthesis. The worker
/// handle lives in its own mutex so `status` never blocks behind a
/// running synthesis.
pub struct QwenGgufProvider {
    config: Mutex<QwenGgufConfig>,
    state: Mutex<ProviderState>,
    worker: Mutex<Option<Worker>>,
    inference_lock: Mutex<()>,
}

impl QwenGgufProvider {
    pub fn new(config: QwenGgufConfig) -> Arc<Self> {
        Arc::new(Self {
            config: Mutex::new(config),
            state: Mutex::new(ProviderState {
                loaded_model: None,
                effective_device: None,
                fallback_reason: None,
                load_error: None,
                ensuring: false,
                download_progress: 0.0,
                downloading: false,
            }),
            worker: Mutex::new(None),
            inference_lock: Mutex::new(()),
        })
    }

    fn backend_diagnostics(&self) -> serde_json::Value {
        let config = self.config_snapshot();
        let discovery = sonicboom_qwen_llama::runtime_backends();
        let gpu = discovery
            .lines()
            .find_map(|line| line.strip_prefix("vulkan: "));
        let mut compiled = vec!["cpu"];
        if cfg!(feature = "qwen-gguf-vulkan") {
            compiled.push("vulkan");
        }
        if cfg!(feature = "qwen-gguf-cuda") {
            compiled.push("cuda");
        }
        let state = self.state.lock().ok();
        let effective = state.as_ref().and_then(|s| s.effective_device.as_deref());
        let fallback = state.as_ref().and_then(|s| s.fallback_reason.as_deref());
        let failed = state.as_ref().is_some_and(|s| s.load_error.is_some());
        let loading = state.as_ref().is_some_and(|s| s.ensuring);
        let initialization_error = discovery.lines().find(|line| {
            line.contains("missing") || line.contains("No usable") || line.contains("failed")
        });
        serde_json::json!({
            "compiled_backends": compiled,
            "available_runtime_backends": if gpu.is_some() { vec!["cpu", "vulkan"] } else { vec!["cpu"] },
            "vulkan_gpu_name": gpu,
            "requested_device": format!("{:?}", config.device).to_ascii_lowercase(),
            "effective_device": effective,
            "model_loading_status": if failed { "failed" } else if loading { "loading" } else if effective.is_some() { "loaded" } else { "unloaded" },
            "vulkan_initialization_error": initialization_error,
            // Detailed model errors can include file paths. Keep those in local logs.
            "cpu_fallback_reason": fallback.map(|_| initialization_error.unwrap_or("Vulkan model initialization failed; CPU fallback active")),
            "status": if effective == Some("vulkan") { "Vulkan running" } else if fallback.is_some() { "CPU fallback active" } else if failed && config.device == Device::Vulkan { "Vulkan model initialization failed" } else if gpu.is_some() { "Vulkan available" } else { initialization_error.unwrap_or("Vulkan not compiled") }
        })
    }

    fn config_snapshot(&self) -> QwenGgufConfig {
        self.config.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// Adopt `next` only when it differs: reconfiguration drops the loaded
    /// worker (and its model/sessions), so spurious resets would interrupt
    /// a serving provider.
    pub fn sync_config(&self, next: QwenGgufConfig) {
        let changed = self
            .config
            .lock()
            .map(|current| *current != next)
            .unwrap_or(true);
        if !changed {
            return;
        }
        if let Ok(mut config) = self.config.lock() {
            *config = next;
        }
        self.unload_worker();
    }

    /// Drop the resident worker without touching configuration (joins the
    /// owner thread; can block while native state unwinds).
    fn unload_worker(&self) {
        let Ok(_lifecycle) = self.inference_lock.lock() else {
            return;
        };
        // Drop outside every provider lock: joining the owner thread can
        // block, and the worker never touches provider locks while exiting.
        let worker = self.worker.lock().ok().and_then(|mut slot| slot.take());
        drop(worker);
        if let Ok(mut state) = self.state.lock() {
            state.loaded_model = None;
            state.effective_device = None;
            state.fallback_reason = None;
        }
    }

    fn check_model(model: &str) -> Result<(), ProviderError> {
        if QWEN_GGUF_MODELS.contains(&model) {
            Ok(())
        } else {
            Err(ProviderError::unsupported_model(
                QWEN_GGUF_PROVIDER_ID,
                model,
                format!(
                    "GGUF/ONNX supports only {}; CustomVoice/VoiceDesign and the \
                     0.6B Candle bundle are rejected here",
                    QWEN_GGUF_MODELS.join(", ")
                ),
            ))
        }
    }

    fn check_language(language: &str) -> Result<(), ProviderError> {
        if QWEN_GGUF_LANGUAGES.contains(&language) {
            Ok(())
        } else {
            Err(ProviderError::unsupported_language(
                QWEN_GGUF_PROVIDER_ID,
                language,
                QWEN_GGUF_LANGUAGES,
            ))
        }
    }

    /// Validate a decoded reference buffer (duration, sample count).
    /// Same bounds the worker enforces at synthesis, so uploads fail fast
    /// instead of at inference time.
    pub fn validate_reference(pcm: &[f32], sample_rate: u32) -> Result<(), ProviderError> {
        if pcm.iter().any(|s| !s.is_finite()) {
            return Err(ProviderError::invalid_voice(
                "reference audio must contain finite samples",
            ));
        }
        if pcm.is_empty() {
            return Err(ProviderError::invalid_voice(
                "reference audio is empty; provide a 1-30 s authorized recording",
            ));
        }
        if !(8000..=192000).contains(&sample_rate) {
            return Err(ProviderError::invalid_voice(
                "reference sample rate must be 8000..=192000 Hz",
            ));
        }
        let secs = pcm.len() as f64 / f64::from(sample_rate);
        if secs < QWEN_GGUF_REF_MIN_SECS {
            return Err(ProviderError::invalid_voice(format!(
                "reference audio is too short ({secs:.1}s); provide at least {:.0}s",
                QWEN_GGUF_REF_MIN_SECS
            )));
        }
        if secs > QWEN_GGUF_REF_MAX_SECS {
            return Err(ProviderError::invalid_voice(format!(
                "reference audio is too long ({secs:.1}s); keep it under {:.0}s",
                QWEN_GGUF_REF_MAX_SECS
            )));
        }
        Ok(())
    }

    /// Fast existence probe for status reporting (no hashing; `load`
    /// performs full SHA-256 verification before spawning the worker).
    fn assets_present(model_dir: &std::path::Path) -> bool {
        models::manifest()
            .iter()
            .all(|file| model_dir.join(file.path).is_file())
    }

    fn check_device_support(device: Device) -> Result<(), ProviderError> {
        #[cfg(not(feature = "qwen-gguf-vulkan"))]
        if device == Device::Vulkan {
            return Err(ProviderError::device_unavailable(
                "device=vulkan requested but this build omits llama.cpp Vulkan; \
                 rebuild with `--features qwen-gguf-vulkan` or use cpu/auto",
            ));
        }
        #[cfg(not(feature = "qwen-gguf-cuda"))]
        if device == Device::Cuda {
            return Err(ProviderError::device_unavailable(
                "device=cuda requested but this build omits llama.cpp CUDA; \
                 rebuild with `--features qwen-gguf-cuda` or use cpu/auto",
            ));
        }
        Ok(())
    }

    fn not_loaded_error(&self) -> ProviderError {
        let config = self.config_snapshot();
        let state = self.state.lock().ok();
        match state {
            Some(state) if state.ensuring && state.download_progress >= 1.0 => {
                return ProviderError::loading(
                    "qwen3-tts-1.7b-base assets verified; native model is loading",
                );
            }
            Some(state) if state.ensuring && state.downloading => {
                return ProviderError::loading(format!(
                    "qwen3-gguf bundle is downloading ({:.0}% complete)",
                    state.download_progress * 100.0
                ));
            }
            Some(state) if state.ensuring => {
                return ProviderError::loading(format!(
                    "qwen3-tts-1.7b-base assets verifying cache ({:.0}% complete)",
                    state.download_progress * 100.0
                ));
            }
            _ => {}
        }
        if !Self::assets_present(&config.model_dir) {
            return ProviderError::model_missing(format!(
                "qwen3-tts-1.7b-base GGUF/ONNX bundle not installed in '{}' \
                 (see docs/providers.md)",
                config.model_dir.display()
            ));
        }
        ProviderError::model_missing(
            "qwen3-tts-1.7b-base is not loaded; call load (or enable \
             [providers.qwen3_gguf] so startup loads it)",
        )
    }
}

impl super::super::TtsProvider for QwenGgufProvider {
    fn id(&self) -> &'static str {
        QWEN_GGUF_PROVIDER_ID
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let devices = ["cpu", "auto"]
            .into_iter()
            .map(str::to_string)
            .chain(cfg!(feature = "qwen-gguf-vulkan").then_some("vulkan".to_string()))
            .chain(cfg!(feature = "qwen-gguf-cuda").then_some("cuda".to_string()))
            .collect();
        ProviderCapabilities {
            provider_id: QWEN_GGUF_PROVIDER_ID.to_string(),
            models: QWEN_GGUF_MODELS.iter().map(|s| s.to_string()).collect(),
            languages: QWEN_GGUF_LANGUAGES.iter().map(|s| s.to_string()).collect(),
            // ICL only: x-vector is unvalidated against the reference
            // implementation and is rejected explicitly in synthesize.
            clone_modes: vec!["icl".to_string()],
            devices,
            requires_reference_audio: true,
            requires_reference_transcript: true,
            voices: vec![],
            sample_rate: Some(QWEN_GGUF_SAMPLE_RATE),
        }
    }

    fn status(&self) -> ProviderStatus {
        let config = self.config_snapshot();
        if !config.enabled {
            return ProviderStatus {
                provider_id: QWEN_GGUF_PROVIDER_ID.to_string(),
                enabled: false,
                state: ProviderStatusState::Disabled,
                active_model: None,
                effective_device: None,
                model_revision: Some(models::REVISION.to_string()),
                backend_diagnostics: Some(self.backend_diagnostics()),
                detail: Some(
                    "provider disabled in configuration \
                     ([providers.qwen3_gguf] enabled=false)"
                        .to_string(),
                ),
            };
        }
        let state = self.state.lock().ok();
        let loaded = state.as_ref().and_then(|s| s.loaded_model.clone());
        let effective = state.as_ref().and_then(|s| s.effective_device.clone());
        let fallback = state.as_ref().and_then(|s| s.fallback_reason.clone());
        let error = state.as_ref().and_then(|s| s.load_error.clone());
        let ensuring = state.as_ref().is_some_and(|s| s.ensuring);
        let downloading = state.as_ref().is_some_and(|s| s.downloading);
        let progress = state.as_ref().map(|s| s.download_progress).unwrap_or(0.0);
        drop(state);
        let installed = Self::assets_present(&config.model_dir);
        let ready = loaded.is_some();
        let state = if ready {
            ProviderStatusState::Ready
        } else if ensuring {
            ProviderStatusState::Loading
        } else if error.is_some() {
            ProviderStatusState::Error
        } else if installed {
            ProviderStatusState::Loading
        } else {
            ProviderStatusState::NotInstalled
        };
        // A GPU fallback is reported, never silent: the effective device
        // plus the reason that forced CPU appear in status detail.
        let detail = if ensuring {
            Some(if downloading {
                format!(
                    "qwen3-gguf bundle downloading ({:.0}% complete)",
                    progress * 100.0
                )
            } else if progress >= 1.0 {
                "qwen3-tts-1.7b-base assets verified; loading native model".to_string()
            } else {
                format!(
                    "qwen3-tts-1.7b-base assets verifying cache ({:.0}% complete)",
                    progress * 100.0
                )
            })
        } else {
            error.or_else(|| {
                if ready {
                    fallback.map(|reason| format!("ready ({reason})"))
                } else if installed {
                    Some("qwen3-tts-1.7b-base bundle present; loading on first use".to_string())
                } else {
                    Some(format!(
                        "qwen3-tts-1.7b-base GGUF/ONNX bundle not installed in '{}' \
                         (see docs/providers.md)",
                        config.model_dir.display()
                    ))
                }
            })
        };
        ProviderStatus {
            provider_id: QWEN_GGUF_PROVIDER_ID.to_string(),
            enabled: true,
            state,
            active_model: loaded.or(Some(QWEN_GGUF_MODELS[0].to_string())),
            effective_device: effective,
            model_revision: Some(models::REVISION.to_string()),
            backend_diagnostics: Some(self.backend_diagnostics()),
            detail,
        }
    }

    fn load(&self, model_id: &str) -> Result<(), ProviderError> {
        Self::check_model(model_id)?;
        let config = self.config_snapshot();
        self.load_with_config(model_id, &config)
    }

    fn unload(&self) {
        self.unload_worker();
    }

    fn invalidate_voice(&self, _voice: &str) {
        // Best effort: a busy worker keeps serving rather than failing the
        // invalidation caller; the next synthesis reconditions on miss.
        if let Ok(slot) = self.worker.lock()
            && let Some(worker) = slot.as_ref()
        {
            let _ = worker.invalidate();
        }
    }

    fn synthesize(&self, request: &ProviderRequest) -> Result<PcmAudio, ProviderError> {
        Self::check_model(&request.model)?;
        Self::check_language(&request.language)?;
        if request.text.trim().is_empty() {
            return Err(ProviderError::invalid_voice("text cannot be empty"));
        }
        let mode = request.clone_mode.unwrap_or(CloneMode::Icl);
        if mode != CloneMode::Icl {
            return Err(ProviderError::invalid_voice(
                "qwen3-gguf supports validated ICL cloning only; x-vector \
                 cloning is not validated against the reference implementation",
            ));
        }
        if request
            .reference_transcript
            .as_deref()
            .map(str::trim)
            .is_none_or(|t| t.is_empty())
        {
            return Err(ProviderError::invalid_voice(
                "qwen3-gguf icl mode requires the exact reference transcript",
            ));
        }
        // Fail fast on invalid sampling (NaN/Inf, out-of-range, bad seed)
        // so callers get an explicit 400 rather than an inference failure.
        if let Err(detail) = super::sampling::resolve(request.sampling.as_ref(), request.seed) {
            return Err(ProviderError::invalid_voice(detail));
        }
        // Reference audio travels with the request: concurrent voices can
        // never observe each other's data through a shared mutable slot.
        // Validated here so callers fail fast; the worker revalidates the
        // exact frames it conditions on.
        match request.reference_audio.as_ref() {
            Some(reference) => {
                if reference.pcm.iter().any(|s| !s.is_finite()) {
                    return Err(ProviderError::invalid_voice(
                        "reference audio must contain finite samples",
                    ));
                }
                if reference.pcm.is_empty() || reference.sample_rate == 0 {
                    return Err(ProviderError::invalid_voice(
                        "reference audio is empty; provide a 1-30 s authorized recording",
                    ));
                }
            }
            None => {
                return Err(ProviderError::invalid_voice(
                    "qwen3-gguf cloning requires reference audio; synthesize through \
                     an authorized voice profile (POST /api/voices/clones)",
                ));
            }
        }
        let config = self.config_snapshot();
        if !config.enabled {
            return Err(ProviderError::model_missing(
                "qwen3-gguf provider is disabled ([providers.qwen3_gguf] enabled=false)",
            ));
        }
        // Serializes synthesis against unload/reconfigure; the worker queue
        // additionally serializes concurrent syntheses on the owner thread.
        // Note: blocks the calling thread; async callers must use
        // `spawn_blocking` (all HTTP synthesis paths already do).
        let _guard = crate::tts::control::lock_cancellable(&self.inference_lock)
            .map_err(ProviderError::inference_failed)?;
        let resident = self
            .state
            .lock()
            .ok()
            .and_then(|s| s.effective_device.clone());
        let Some(device) = resident else {
            return Err(self.not_loaded_error());
        };
        if request.device != DevicePreference::Auto {
            let expected = match request.device {
                DevicePreference::Cpu => "cpu",
                DevicePreference::Cuda => "cuda",
                DevicePreference::Auto => device.as_str(),
            };
            if device != expected {
                return Err(ProviderError::device_unavailable(
                    "requested device differs from the resident model; change \
                     providers.qwen3_gguf.device and wait for reload",
                ));
            }
        }
        if let Some(control) = crate::tts::control::current() {
            control.set_context(
                QWEN_GGUF_PROVIDER_ID,
                &request.model,
                &request.voice,
                &device,
            );
        }
        let mut worker_request = request.clone();
        worker_request.model = QWEN_GGUF_MODELS[0].to_string();
        let synthesis = self
            .worker
            .lock()
            .map_err(|_| ProviderError::inference_failed("qwen3-gguf worker lock poisoned"))?
            .as_ref()
            .ok_or_else(|| self.not_loaded_error())?
            .synthesize(worker_request)
            .map_err(|error| {
                let message = format!("{error:#}");
                if message.contains("queue is full") || message.contains("queue unavailable") {
                    ProviderError::busy(message)
                } else {
                    ProviderError::inference_failed(message)
                }
            })?;
        let audio = synthesis.audio;
        if audio.samples.is_empty()
            || audio.sample_rate == 0
            || audio.samples.iter().any(|s| !s.is_finite())
            || !audio.is_audible()
        {
            return Err(ProviderError::inference_failed(
                "qwen3-gguf synthesis produced no audio",
            ));
        }
        Ok(audio)
    }
}

impl QwenGgufProvider {
    /// Blocking download-verified native load under an explicit config
    /// snapshot. Background ensure calls this after the bundle is fetched;
    /// a snapshot that no longer matches live config is discarded as
    /// obsolete so a hot-reload can never resurrect a retired backend.
    fn load_with_config(
        &self,
        model_id: &str,
        config: &QwenGgufConfig,
    ) -> Result<(), ProviderError> {
        Self::check_model(model_id)?;
        if !config.enabled {
            return Err(ProviderError::model_missing(
                "qwen3-gguf provider is disabled \
                 ([providers.qwen3_gguf] enabled=false)",
            ));
        }
        if self.config_snapshot() != *config {
            return Err(ProviderError::loading("obsolete qwen3-gguf load discarded"));
        }
        Self::check_device_support(config.device)?;
        // Full pinned integrity verification before any native load claim.
        assets::verify_local_files(&config.model_dir, &models::manifest())?;
        if self
            .state
            .lock()
            .ok()
            .and_then(|s| s.loaded_model.clone())
            .is_some_and(|loaded| loaded == model_id)
        {
            return Ok(());
        }
        let worker = Worker::spawn(Settings {
            dir: config.model_dir.clone(),
            device: config.device,
            gpu_layers: config.gpu_layers,
            threads: config.threads,
            max_tokens: config.max_tokens,
            cache_capacity: config.cache_capacity,
            allow_cpu_fallback: config.allow_cpu_fallback,
            sampling: config.sampling.clone(),
            quality: config.quality,
        })
        .map_err(|error| ProviderError::inference_failed(format!("{error:#}")))?;
        let ready = worker.ready.clone();
        if let Ok(mut slot) = self.worker.lock() {
            *slot = Some(worker);
        }
        if let Ok(mut state) = self.state.lock() {
            state.loaded_model = Some(model_id.to_string());
            state.effective_device = Some(ready.effective_device);
            state.fallback_reason = ready.fallback_reason;
            state.load_error = None;
        }
        Ok(())
    }

    /// Start download + load in the background (deduplicated). Returns
    /// immediately; progress surfaces through [`TtsProvider::status`].
    /// Used at startup and when hot-reload enables the provider.
    pub fn ensure_background(self: &Arc<Self>, hf_token: Option<String>) {
        {
            let mut state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            if state.ensuring {
                return;
            }
            if state.loaded_model.is_some() {
                return;
            }
            state.ensuring = true;
            state.download_progress = 0.0;
            state.downloading = false;
        }
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let result = this.ensure_async(hf_token).await;
            if let Ok(mut state) = this.state.lock() {
                state.ensuring = false;
                state.downloading = false;
                match &result {
                    Ok(()) => state.load_error = None,
                    Err(error) if error.message().contains("obsolete") => {
                        state.load_error = None;
                    }
                    Err(error) => state.load_error = Some(error.message().to_string()),
                }
            }
            if let Err(error) = &result {
                tracing::error!("qwen3-gguf background prepare failed: {error}");
            }
        });
    }

    async fn ensure_async(self: &Arc<Self>, hf_token: Option<String>) -> Result<(), ProviderError> {
        let config = self.config_snapshot();
        if !config.enabled {
            return Err(ProviderError::model_missing(
                "qwen3-gguf provider is disabled ([providers.qwen3_gguf] enabled=false)",
            ));
        }
        let manifest = models::manifest();
        let progress_self = Arc::clone(self);
        let download_self = Arc::clone(self);
        assets::ensure_pinned_files(
            "qwen3-gguf",
            &config.model_dir,
            &manifest,
            hf_token.as_deref(),
            move |p| {
                if let Ok(mut state) = progress_self.state.lock() {
                    state.download_progress = p;
                }
            },
            move || {
                if let Ok(mut state) = download_self.state.lock() {
                    state.downloading = true;
                }
            },
            || false,
        )
        .await?;
        if let Ok(mut state) = self.state.lock() {
            state.downloading = false;
        }
        // Heavy native load runs on a blocking thread; the async task only
        // records the outcome. Reload failure keeps the last working model.
        let loader = Arc::clone(self);
        tokio::task::spawn_blocking(move || loader.load_with_config(QWEN_GGUF_MODELS[0], &config))
            .await
            .map_err(|error| {
                ProviderError::inference_failed(format!("qwen3-gguf load task failed: {error}"))
            })?
    }

    /// Whether a background ensure is currently running.
    pub fn is_ensuring(&self) -> bool {
        self.state.lock().map(|s| s.ensuring).unwrap_or(false)
    }

    /// Trigger a background ensure when the provider is enabled but has no
    /// loaded model and no error yet (hot-reload path). Called during
    /// request-scoped registry sync; never blocks.
    pub fn kick_ensure_if_needed(self: &Arc<Self>, hf_token: Option<String>) {
        let resident = self
            .state
            .lock()
            .ok()
            .and_then(|s| s.loaded_model.clone())
            .is_some();
        let needs = {
            let config = self.config_snapshot();
            let state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            config.enabled && !resident && state.load_error.is_none() && !state.ensuring
        };
        if needs {
            self.ensure_background(hf_token);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::TtsProvider;
    use super::super::super::registry::TtsProviderRegistry;
    use super::*;
    use crate::tts::providers::types::ReferenceAudio;

    fn enabled_missing_dir() -> Arc<QwenGgufProvider> {
        QwenGgufProvider::new(QwenGgufConfig {
            enabled: true,
            model_dir: PathBuf::from("/nonexistent-qwen-gguf-bundle"),
            ..QwenGgufConfig::default()
        })
    }

    fn cloning_request() -> ProviderRequest {
        let mut request = ProviderRequest::new(
            QWEN_GGUF_MODELS[0],
            "test-owned-reference",
            "Hello, this is a native voice cloning test.",
            "en",
        );
        request.clone_mode = Some(CloneMode::Icl);
        request.reference_transcript = Some("exact reference transcript".to_string());
        request.reference_audio = Some(ReferenceAudio {
            pcm: vec![0.1; 24_000],
            sample_rate: 24_000,
        });
        request
    }

    #[test]
    fn provider_is_disabled_by_default() {
        let provider = QwenGgufProvider::new(QwenGgufConfig::default());
        let status = provider.status();
        assert!(!status.enabled);
        assert_eq!(status.state, ProviderStatusState::Disabled);
        assert_eq!(provider.id(), "qwen3-gguf");
    }

    #[test]
    fn capabilities_advertise_validated_icl_only() {
        let capabilities = QwenGgufProvider::new(QwenGgufConfig::default()).capabilities();
        assert_eq!(capabilities.models, vec!["qwen3-tts-1.7b-base".to_string()]);
        assert_eq!(capabilities.clone_modes, vec!["icl".to_string()]);
        assert!(capabilities.requires_reference_audio);
        assert!(capabilities.requires_reference_transcript);
        assert_eq!(capabilities.sample_rate, Some(24_000));
        assert!(capabilities.devices.contains(&"cpu".to_string()));
        assert!(capabilities.devices.contains(&"auto".to_string()));
    }

    #[test]
    fn missing_bundle_reports_not_installed_without_hashing() {
        let provider = enabled_missing_dir();
        let status = provider.status();
        assert_eq!(status.state, ProviderStatusState::NotInstalled);
        assert!(status.detail.is_some_and(|d| d.contains("not installed")));
        let error = provider.load(QWEN_GGUF_MODELS[0]).unwrap_err();
        assert_eq!(error.code(), "model_missing");
        let error = provider.synthesize(&cloning_request()).unwrap_err();
        assert_eq!(error.code(), "model_missing");
    }

    #[test]
    fn unknown_models_and_languages_are_rejected() {
        let provider = enabled_missing_dir();
        let error = provider.load("qwen3-tts-0.6b-base").unwrap_err();
        assert_eq!(error.code(), "unsupported_model");
        let mut request = cloning_request();
        request.model = "supertonic-3".to_string();
        assert_eq!(
            provider.synthesize(&request).unwrap_err().code(),
            "unsupported_model"
        );
        let mut request = cloning_request();
        request.language = "xx".to_string();
        assert_eq!(
            provider.synthesize(&request).unwrap_err().code(),
            "unsupported_language"
        );
    }

    #[test]
    fn xvector_mode_is_rejected_as_unvalidated() {
        let provider = enabled_missing_dir();
        let mut request = cloning_request();
        request.clone_mode = Some(CloneMode::XVector);
        let error = provider.synthesize(&request).unwrap_err();
        assert_eq!(error.code(), "invalid_voice");
        assert!(error.message().contains("x-vector"));
    }

    #[test]
    fn empty_text_and_missing_conditioning_are_rejected() {
        let provider = enabled_missing_dir();
        let mut request = cloning_request();
        request.text = "   ".to_string();
        assert_eq!(
            provider.synthesize(&request).unwrap_err().code(),
            "invalid_voice"
        );
        let mut request = cloning_request();
        request.reference_transcript = None;
        let error = provider.synthesize(&request).unwrap_err();
        assert_eq!(error.code(), "invalid_voice");
        assert!(error.message().contains("transcript"));
        let mut request = cloning_request();
        request.reference_audio = None;
        assert_eq!(
            provider.synthesize(&request).unwrap_err().code(),
            "invalid_voice"
        );
    }

    #[tokio::test]
    async fn disabled_provider_never_starts_a_download() {
        // Only Supertonic weights download by default: a disabled GGUF
        // provider must not start background work when kicked.
        let provider = QwenGgufProvider::new(QwenGgufConfig::default());
        provider.kick_ensure_if_needed(None);
        assert!(!provider.is_ensuring());
        let status = provider.status();
        assert!(!status.enabled);
        assert_eq!(status.state, ProviderStatusState::Disabled);
    }

    #[tokio::test]
    #[ignore = "requires pinned GGUF/ONNX bundle in QWEN_GGUF_MODEL_DIR; loads native CPU model"]
    async fn background_ensure_loads_verified_bundle_without_download() {
        let dir = std::env::var_os("QWEN_GGUF_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("set QWEN_GGUF_MODEL_DIR to the downloaded model bundle");
        let provider = QwenGgufProvider::new(QwenGgufConfig {
            enabled: true,
            model_dir: dir,
            ..QwenGgufConfig::default()
        });
        provider.kick_ensure_if_needed(None);
        assert!(provider.is_ensuring());
        // Kick is deduplicated: a second kick while ensuring starts nothing new.
        provider.kick_ensure_if_needed(None);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        let status = loop {
            let status = provider.status();
            if status.state == ProviderStatusState::Ready
                || status.state == ProviderStatusState::Error
            {
                break status;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "ensure timed out: {status:?}"
            );
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        };
        assert!(!provider.is_ensuring());
        assert_eq!(status.state, ProviderStatusState::Ready, "{status:?}");
        assert_eq!(status.effective_device.as_deref(), Some("cpu"));
        // The bundle was already verified: no network fetch may start.
        assert!(
            status.detail.is_none(),
            "warm boot must not report a download: {status:?}"
        );
        provider.unload();
        assert!(
            provider.status().state.eq(&ProviderStatusState::Loading),
            "unloaded provider with a present bundle reports loading, not silence"
        );
    }

    #[test]
    fn registry_routes_qwen_models_without_candle() {
        // The Candle provider is gone: 1.7B resolves to the GGUF backend
        // and the retired 0.6B identifier is explicitly unknown (never a
        // silent fallback to another engine).
        let registry = TtsProviderRegistry::new();
        registry.register(enabled_missing_dir() as Arc<dyn TtsProvider>);
        assert_eq!(
            registry.resolve_model("qwen3-tts-1.7b-base").unwrap().id(),
            "qwen3-gguf"
        );
        let error = match registry.resolve_model("qwen3-tts-0.6b-base") {
            Ok(_) => panic!("retired Candle model must not resolve"),
            Err(error) => error,
        };
        assert_eq!(error.code(), "unsupported_model");
        assert_eq!(registry.provider_ids(), vec!["qwen3-gguf".to_string()]);
    }
}
