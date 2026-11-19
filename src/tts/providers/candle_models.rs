//! Candle-backed providers (the candle migration, see docs/candle-migration.md).
//!
//! * `qwen3-candle`  — Qwen3-TTS Base (0.6B / 1.7B safetensors) with ICL / x-vector
//!   voice cloning, executed on candle (CPU or Vulkan).
//! * `kyutai-candle` — Kyutai DSM-TTS `tts-1.6b-en_fr` with pre-computed voice
//!   embeddings (`kyutai/tts-voices`), executed on candle (CPU or Vulkan).
//!
//! Lifecycle matches the other native providers: configuration is synced from the
//! live `AppConfig` (hot reload), an enabled provider with a model directory loads
//! in a background thread, and until then it reports `not_installed` / `loading`.
//! The (non-`Sync`) model sits behind a mutex, so concurrent callers serialize.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use super::TtsProvider;
use super::assets::{PinnedFile, ensure_pinned_files};
use super::error::{ProviderError, ProviderErrorKind};
use super::types::{
    DevicePreference, PcmAudio, ProviderCapabilities, ProviderRequest, ProviderStatus,
    ProviderStatusState,
};

/// Runtime settings (mapped from `[providers.qwen3_candle]` / `[providers.kyutai_candle]`).
#[derive(Debug, Clone, PartialEq)]
pub struct CandleProviderConfig {
    pub enabled: bool,
    pub model_dir: PathBuf,
    /// Kyutai only.
    pub voices_dir: PathBuf,
    /// Kyutai only.
    pub default_voice: String,
    /// `auto` | `cpu` | `vulkan`.
    pub device: String,
    /// Fetch and verify the pinned bundle into `model_dir` (and the default voice into
    /// `voices_dir`) when enabled. `false` uses whatever is in the directories.
    pub download: bool,
}

impl Default for CandleProviderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: PathBuf::from("models/candle"),
            voices_dir: PathBuf::from("models/candle/voices"),
            default_voice: "calm".into(),
            device: "auto".into(),
            download: true,
        }
    }
}

/// Pinned bundles (Hugging Face revisions, SHA-256 and sizes verified on download and on reuse).
pub mod manifests {
    use super::PinnedFile;

    fn hf(repo: &str, rev: &str, path: &'static str) -> String {
        PinnedFile::huggingface(repo, rev, path)
    }

    pub const QWEN_REPO: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base";
    pub const QWEN_REVISION: &str = "fd4b254389122332181a7c3db7f27e918eec64e3";
    pub const KYUTAI_REPO: &str = "kyutai/tts-1.6b-en_fr";
    pub const KYUTAI_REVISION: &str = "f65439609986c392cb12df63938abcc550c3fb15";
    pub const KYUTAI_VOICES_REPO: &str = "kyutai/tts-voices";
    pub const KYUTAI_VOICES_REVISION: &str = "323332d33f997de8394f24a193e1a76df720e01a";
    /// File name of the bundled default voice inside `voices_dir`.
    pub const KYUTAI_DEFAULT_VOICE_FILE: &str = "calm.safetensors";

    fn pin(repo: &str, rev: &str, path: &'static str, sha256: &'static str, size: u64) -> PinnedFile {
        PinnedFile { path, url: hf(repo, rev, path), sha256, size }
    }

    /// Qwen3-TTS 1.7B Base (Apache-2.0).
    pub fn qwen_files() -> Vec<PinnedFile> {
        let (r, v) = (QWEN_REPO, QWEN_REVISION);
        vec![
            pin(r, v, "config.json", "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9", 4494),
            pin(r, v, "merges.txt", "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3", 1_671_839),
            pin(r, v, "vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", 2_776_833),
            pin(r, v, "speech_tokenizer/model.safetensors", "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258", 682_293_092),
            pin(r, v, "model.safetensors", "38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c", 3_857_413_744),
        ]
    }

    /// Kyutai TTS 1.6B en/fr (weights CC-BY 4.0).
    pub fn kyutai_files() -> Vec<PinnedFile> {
        let (r, v) = (KYUTAI_REPO, KYUTAI_REVISION);
        vec![
            pin(r, v, "config.json", "fa750c09900257f65350444bc6ff4ddbdfe8a7023f4c6428b99567e1618ffdb1", 2423),
            pin(r, v, "tokenizer_spm_8k_en_fr_audio.model", "cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69", 120_378),
            pin(r, v, "tokenizer-e351c8d8-checkpoint125.safetensors", "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50", 384_644_900),
            pin(r, v, "dsm_tts_1e68beda@240.safetensors", "726ddadd90a080c89cbc6b217745296ef32d8e25666d30f81a09e8ae5c9e0f0c", 3_683_719_712),
        ]
    }

    /// The default `calm` speaker embedding (installed as `calm.safetensors`).
    pub fn kyutai_default_voice() -> Vec<PinnedFile> {
        vec![PinnedFile {
            path: KYUTAI_DEFAULT_VOICE_FILE,
            url: hf(
                KYUTAI_VOICES_REPO,
                KYUTAI_VOICES_REVISION,
                "expresso/ex03-ex01_calm_001_channel1_1143s.wav.1e68beda@240.safetensors",
            ),
            sha256: "94fdd01c5a713a5e9b91619bd700b10e1d97bcbbf55feec46bb2596456bd0688",
            size: 256_136,
        }]
    }
}

/// Device to use for a configured `device` string and a per-request preference.
fn resolve_device(configured: &str, request: DevicePreference) -> &'static str {
    let gpu = cfg!(feature = "candle-vulkan");
    match (configured, request) {
        (_, DevicePreference::Cpu) | ("cpu", _) => "cpu",
        ("vulkan", _) => "vulkan",
        _ if gpu => "vulkan",
        _ => "cpu",
    }
}

/// State shared by both providers: configuration, the loaded model and load progress.
struct Lifecycle<M> {
    config: Mutex<CandleProviderConfig>,
    model: Mutex<Option<(M, &'static str)>>,
    loading: AtomicBool,
    /// Download progress in permille while `downloading`.
    progress_permille: AtomicU32,
    downloading: AtomicBool,
    error: Mutex<Option<String>>,
}

impl<M> Lifecycle<M> {
    fn new(config: CandleProviderConfig) -> Self {
        Self {
            config: Mutex::new(config),
            model: Mutex::new(None),
            loading: AtomicBool::new(false),
            progress_permille: AtomicU32::new(0),
            downloading: AtomicBool::new(false),
            error: Mutex::new(None),
        }
    }

    fn config(&self) -> CandleProviderConfig {
        self.config.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// Replace the configuration; a change that affects the loaded model drops it.
    fn sync(&self, new: CandleProviderConfig) {
        let Ok(mut current) = self.config.lock() else { return };
        if *current == new {
            return;
        }
        let affects_model = current.enabled != new.enabled
            || current.model_dir != new.model_dir
            || current.voices_dir != new.voices_dir
            || current.default_voice != new.default_voice
            || current.download != new.download
            || current.device != new.device;
        *current = new;
        drop(current);
        if affects_model {
            if let Ok(mut m) = self.model.lock() {
                *m = None;
            }
            if let Ok(mut e) = self.error.lock() {
                *e = None;
            }
        }
    }

    fn status(&self, id: &str, model_id: &str, installed: bool) -> ProviderStatus {
        let cfg = self.config();
        let guard = self.model.lock().ok();
        let loaded = guard.as_ref().and_then(|g| g.as_ref());
        let error = self.error.lock().ok().and_then(|e| e.clone());
        let state = if !cfg.enabled {
            ProviderStatusState::Disabled
        } else if loaded.is_some() {
            ProviderStatusState::Ready
        } else if self.loading.load(Ordering::Acquire) {
            ProviderStatusState::Loading
        } else if error.is_some() {
            ProviderStatusState::Error
        } else if !installed {
            ProviderStatusState::NotInstalled
        } else {
            ProviderStatusState::Loading
        };
        ProviderStatus {
            provider_id: id.into(),
            enabled: cfg.enabled,
            state,
            active_model: loaded.map(|_| model_id.to_string()),
            effective_device: loaded.map(|(_, d)| d.to_string()),
            model_revision: None,
            backend_diagnostics: None,
            detail: error.or_else(|| {
                self.downloading.load(Ordering::Acquire).then(|| {
                    format!(
                        "downloading model files ({:.0}%)",
                        f64::from(self.progress_permille.load(Ordering::Acquire)) / 10.0
                    )
                })
            }),
        }
    }

    /// Run `fetch` (pinned downloads, when enabled) and `load` on a background thread
    /// unless one is running or the model is already loaded or failed.
    fn kick<F>(
        self: &Arc<Self>,
        provider: &'static str,
        installed: bool,
        manifest: Vec<(PathBuf, Vec<PinnedFile>)>,
        hf_token: Option<String>,
        load: F,
    ) where
        M: Send + 'static,
        F: FnOnce(&CandleProviderConfig) -> Result<(M, &'static str), String> + Send + 'static,
    {
        let cfg = self.config();
        if !cfg.enabled || !(cfg.download || installed) {
            return;
        }
        if self.model.lock().map(|m| m.is_some()).unwrap_or(true)
            || self.error.lock().map(|e| e.is_some()).unwrap_or(true)
        {
            return;
        }
        if self.loading.swap(true, Ordering::AcqRel) {
            return;
        }
        let this = Arc::clone(self);
        std::thread::spawn(move || {
            let result = (|| {
                if cfg.download {
                    this.downloading.store(true, Ordering::Release);
                    this.progress_permille.store(0, Ordering::Release);
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| format!("download runtime failed: {e}"))?;
                    let total = manifest.len().max(1) as f32;
                    for (index, (dir, files)) in manifest.iter().enumerate() {
                        let this_ref = &this;
                        let expected = &cfg;
                        let outcome = runtime.block_on(ensure_pinned_files(
                            provider,
                            dir,
                            files,
                            hf_token.as_deref(),
                            |p| {
                                let overall = (index as f32 + p.clamp(0.0, 1.0)) / total;
                                this_ref
                                    .progress_permille
                                    .store((overall * 1000.0) as u32, Ordering::Release);
                            },
                            || {},
                            || this_ref.config() != *expected,
                        ));
                        if let Err(error) = outcome {
                            this.downloading.store(false, Ordering::Release);
                            return Err(error.message().to_string());
                        }
                    }
                    this.downloading.store(false, Ordering::Release);
                }
                load(&cfg)
            })();
            match result {
                Ok(loaded) => {
                    // A reconfiguration while loading wins: drop a stale result.
                    if this.config() == cfg {
                        if let Ok(mut m) = this.model.lock() {
                            *m = Some(loaded);
                        }
                    }
                }
                Err(message) => {
                    tracing::error!(%message, "candle provider load failed");
                    if let Ok(mut e) = this.error.lock() {
                        *e = Some(message);
                    }
                }
            }
            this.loading.store(false, Ordering::Release);
        });
    }
}

// ----------------------------------------------------------------------------
// Qwen3-TTS
// ----------------------------------------------------------------------------

#[cfg(feature = "candle-qwen")]
pub mod qwen {
    use super::*;
    use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions};

    pub const PROVIDER_ID: &str = "qwen3-candle";
    pub const MODEL_ID: &str = "qwen3-tts-candle";

    pub struct CandleQwenProvider {
        life: Arc<Lifecycle<Qwen3TTS>>,
    }

    fn installed(dir: &std::path::Path) -> bool {
        dir.join("config.json").is_file() && dir.join("model.safetensors").is_file()
    }

    impl CandleQwenProvider {
        pub fn new(config: CandleProviderConfig) -> Arc<Self> {
            Arc::new(Self { life: Arc::new(Lifecycle::new(config)) })
        }

        pub fn sync_config(&self, config: CandleProviderConfig) {
            self.life.sync(config);
        }

        /// Start the background download + load when enabled and not yet loaded.
        pub fn kick_ensure_if_needed(&self, hf_token: Option<String>) {
            let dir = self.life.config().model_dir;
            let manifest = vec![(dir.clone(), manifests::qwen_files())];
            self.life.kick(PROVIDER_ID, installed(&dir), manifest, hf_token, |cfg| {
                let device = resolve_device(&cfg.device, DevicePreference::Auto);
                let parsed = qwen3_tts::parse_device(device).map_err(|e| format!("{e}"))?;
                Qwen3TTS::from_pretrained(&cfg.model_dir.to_string_lossy(), parsed)
                    .map(|m| (m, device))
                    .map_err(|e| format!("qwen3 candle load failed: {e}"))
            });
        }
    }

    impl TtsProvider for CandleQwenProvider {
        fn id(&self) -> &'static str {
            PROVIDER_ID
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                provider_id: PROVIDER_ID.into(),
                models: vec![MODEL_ID.into()],
                languages: ["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                clone_modes: vec!["icl".into(), "x_vector".into()],
                devices: vec!["cpu".into(), "auto".into(), "vulkan".into()],
                requires_reference_audio: true,
                requires_reference_transcript: false,
                voices: vec![],
                sample_rate: Some(24_000),
            }
        }

        fn status(&self) -> ProviderStatus {
            let installed = installed(&self.life.config().model_dir);
            self.life.status(PROVIDER_ID, MODEL_ID, installed)
        }

        fn load(&self, model_id: &str) -> Result<(), ProviderError> {
            if model_id != MODEL_ID {
                return Err(ProviderError::unsupported_model(PROVIDER_ID, model_id, format!("expected {MODEL_ID}")));
            }
            self.kick_ensure_if_needed(None);
            Ok(())
        }

        fn unload(&self) {
            if let Ok(mut m) = self.life.model.lock() {
                *m = None;
            }
        }

        fn synthesize(&self, request: &ProviderRequest) -> Result<PcmAudio, ProviderError> {
            let text = request.text.trim();
            if text.is_empty() {
                return Err(ProviderError::invalid_voice("text cannot be empty"));
            }
            let reference = request.reference_audio.as_ref().ok_or_else(|| {
                ProviderError::invalid_voice("qwen3-candle requires reference audio (voice cloning)")
            })?;
            let language: Language = request.language.parse().map_err(|_| {
                ProviderError::new(
                    ProviderErrorKind::UnsupportedLanguage,
                    format!("unsupported language '{}'", request.language),
                )
            })?;
            let guard = self.life.model.lock().map_err(|_| ProviderError::inference_failed("qwen state poisoned"))?;
            let (model, _) = guard.as_ref().ok_or_else(|| ProviderError::loading("qwen3-candle is not loaded"))?;
            let transcript = match request.clone_mode {
                Some(super::super::types::CloneMode::XVector) => None,
                _ => request.reference_transcript.as_deref(),
            };
            let buffer = AudioBuffer { samples: reference.pcm.clone(), sample_rate: reference.sample_rate };
            let prompt = model
                .create_voice_clone_prompt(&buffer, transcript)
                .map_err(|e| ProviderError::inference_failed(format!("reference conditioning failed: {e}")))?;
            let options = SynthesisOptions {
                max_length: (text.chars().count() * 2 + 25).max(50),
                seed: request.seed,
                ..SynthesisOptions::default()
            };
            let audio = model
                .synthesize_voice_clone(text, &prompt, language, Some(options))
                .map_err(|e| ProviderError::inference_failed(format!("qwen3 candle inference failed: {e}")))?;
            Ok(PcmAudio::mono(audio.samples, audio.sample_rate))
        }
    }
}

// ----------------------------------------------------------------------------
// Kyutai TTS
// ----------------------------------------------------------------------------

#[cfg(feature = "candle-kyutai")]
pub mod kyutai {
    use super::*;
    use candle_core::{DType, Device};
    use kyutai_tts::tts::{KyutaiTts, Options};

    pub const PROVIDER_ID: &str = "kyutai-candle";
    pub const MODEL_ID: &str = "kyutai-tts-1.6b-en-fr";

    pub struct CandleKyutaiProvider {
        life: Arc<Lifecycle<KyutaiTts>>,
    }

    fn installed(dir: &std::path::Path) -> bool {
        dir.join("config.json").is_file()
    }

    /// Voice names (file stems) in `dir`; the default voice is always offered so a
    /// fresh install lists it before its file has been downloaded.
    fn list_voices(dir: &std::path::Path, default_voice: &str) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                (p.extension().is_some_and(|x| x == "safetensors"))
                    .then(|| p.file_stem()?.to_str().map(str::to_string))
                    .flatten()
            })
            .collect();
        if !default_voice.is_empty() && !names.iter().any(|n| n == default_voice) {
            names.push(default_voice.to_string());
        }
        names.sort();
        names.dedup();
        names
    }

    fn voice_path(cfg: &CandleProviderConfig, voice: &str) -> Result<PathBuf, ProviderError> {
        let name = if voice.is_empty() { cfg.default_voice.as_str() } else { voice };
        // Voice names are file stems inside voices_dir: no separators allowed.
        if name.contains(['/', '\\']) || name.contains("..") {
            return Err(ProviderError::invalid_voice(format!("invalid voice name '{name}'")));
        }
        let p = cfg.voices_dir.join(format!("{name}.safetensors"));
        if p.is_file() {
            Ok(p)
        } else {
            Err(ProviderError::invalid_voice(format!("voice '{name}' not found in the voices directory")))
        }
    }

    impl CandleKyutaiProvider {
        pub fn new(config: CandleProviderConfig) -> Arc<Self> {
            Arc::new(Self { life: Arc::new(Lifecycle::new(config)) })
        }

        pub fn sync_config(&self, config: CandleProviderConfig) {
            self.life.sync(config);
        }

        pub fn kick_ensure_if_needed(&self, hf_token: Option<String>) {
            let cfg = self.life.config();
            let dir = cfg.model_dir.clone();
            let mut manifest = vec![(dir.clone(), manifests::kyutai_files())];
            if cfg.default_voice == "calm" {
                manifest.push((cfg.voices_dir.clone(), manifests::kyutai_default_voice()));
            }
            self.life.kick(PROVIDER_ID, installed(&dir), manifest, hf_token, |cfg| {
                let name = resolve_device(&cfg.device, DevicePreference::Auto);
                let (device, dtype) = match name {
                    "cpu" => (Device::Cpu, DType::F32),
                    #[cfg(feature = "candle-vulkan")]
                    "vulkan" => (
                        Device::new_vulkan(0).map_err(|e| format!("vulkan unavailable: {e}"))?,
                        // BF16 is the only dtype that is both correct and fits in GPU memory.
                        DType::BF16,
                    ),
                    other => return Err(format!("device '{other}' is not available in this build")),
                };
                let voice = voice_path(cfg, "").map_err(|e| e.message().to_string())?;
                KyutaiTts::load(&cfg.model_dir, &voice, &device, dtype, &Options::default().cfg_value)
                    .map(|m| (m, name))
                    .map_err(|e| format!("kyutai candle load failed: {e}"))
            });
        }
    }

    impl TtsProvider for CandleKyutaiProvider {
        fn id(&self) -> &'static str {
            PROVIDER_ID
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                provider_id: PROVIDER_ID.into(),
                models: vec![MODEL_ID.into()],
                languages: vec!["en".into(), "fr".into()],
                clone_modes: vec![],
                devices: vec!["cpu".into(), "auto".into(), "vulkan".into()],
                requires_reference_audio: false,
                requires_reference_transcript: false,
                voices: list_voices(&self.life.config().voices_dir, &self.life.config().default_voice),
                sample_rate: Some(24_000),
            }
        }

        fn status(&self) -> ProviderStatus {
            let installed = installed(&self.life.config().model_dir);
            self.life.status(PROVIDER_ID, MODEL_ID, installed)
        }

        fn load(&self, model_id: &str) -> Result<(), ProviderError> {
            if model_id != MODEL_ID {
                return Err(ProviderError::unsupported_model(PROVIDER_ID, model_id, format!("expected {MODEL_ID}")));
            }
            self.kick_ensure_if_needed(None);
            Ok(())
        }

        fn unload(&self) {
            if let Ok(mut m) = self.life.model.lock() {
                *m = None;
            }
        }

        fn synthesize(&self, request: &ProviderRequest) -> Result<PcmAudio, ProviderError> {
            let text = request.text.trim();
            if text.is_empty() {
                return Err(ProviderError::invalid_voice("text cannot be empty"));
            }
            if !matches!(request.language.as_str(), "en" | "fr") {
                return Err(ProviderError::new(
                    ProviderErrorKind::UnsupportedLanguage,
                    format!("kyutai-candle supports 'en' and 'fr', got '{}'", request.language),
                ));
            }
            let cfg = self.life.config();
            let voice = voice_path(&cfg, &request.voice)?;
            let mut guard = self.life.model.lock().map_err(|_| ProviderError::inference_failed("kyutai state poisoned"))?;
            let (tts, _) = guard.as_mut().ok_or_else(|| ProviderError::loading("kyutai-candle is not loaded"))?;
            let mut opts = Options::default();
            if let Some(seed) = request.seed {
                opts.seed = seed;
            }
            tts.set_voice(&voice, &opts.cfg_value)
                .map_err(|e| ProviderError::inference_failed(format!("voice conditioning failed: {e}")))?;
            let (pcm, _timings) = tts
                .synthesize(text, &opts, |_| {})
                .map_err(|e| ProviderError::inference_failed(format!("kyutai candle inference failed: {e}")))?;
            Ok(PcmAudio::mono(pcm, tts.sample_rate as u32))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifests_are_well_formed() {
        for (name, files) in [
            ("qwen", manifests::qwen_files()),
            ("kyutai", manifests::kyutai_files()),
            ("voice", manifests::kyutai_default_voice()),
        ] {
            assert!(!files.is_empty(), "{name}");
            for f in files {
                assert_eq!(f.sha256.len(), 64, "{name}/{} sha length", f.path);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()), "{name}/{}", f.path);
                assert!(f.size > 0 && f.url.starts_with("https://huggingface.co/"), "{name}/{}", f.path);
                assert!(!f.path.starts_with('/') && !f.path.contains(".."), "{name}/{}", f.path);
            }
        }
    }

    #[test]
    fn download_false_without_files_reports_not_installed() {
        let dir = std::env::temp_dir().join(format!("candle-empty-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let life: Arc<Lifecycle<()>> = Arc::new(Lifecycle::new(CandleProviderConfig {
            enabled: true,
            model_dir: dir.clone(),
            download: false,
            ..Default::default()
        }));
        // Nothing to download and nothing installed: no thread is started.
        life.kick("test", false, vec![], None, |_| Err("must not run".into()));
        assert!(!life.loading.load(Ordering::Acquire));
        assert_eq!(life.status("test", "m", false).state, ProviderStatusState::NotInstalled);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Network: downloads the pinned 256 KB default voice into an empty directory.
    #[tokio::test]
    #[ignore = "downloads from huggingface.co"]
    async fn default_voice_downloads_and_verifies() {
        let dir = std::env::temp_dir().join(format!("candle-voice-{}", uuid::Uuid::new_v4()));
        let files = manifests::kyutai_default_voice();
        ensure_pinned_files("kyutai-candle", &dir, &files, None, |_| {}, || {}, || false)
            .await
            .expect("download");
        let installed = dir.join(manifests::KYUTAI_DEFAULT_VOICE_FILE);
        assert_eq!(std::fs::metadata(&installed).unwrap().len(), files[0].size);
        // Second call reuses the verified file.
        ensure_pinned_files("kyutai-candle", &dir, &files, None, |_| {}, || {}, || false)
            .await
            .expect("reuse");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
