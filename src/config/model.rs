//! Structured, sectioned SonicBoom configuration model.
//!
//! The configuration is split into logical sections so each subsystem
//! owns its settings, and the whole tree serializes to/from
//! `config.toml`. Secrets (admin password, HuggingFace token) are
//! deliberately **not** part of this model — they live in the
//! [`crate::config::secrets::SecretStore`].

use std::net::IpAddr;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use sonicboom::tts::providers::types::SamplingRequest;

// ---------------------------------------------------------------------------
// Public constants (referenced across the binary)
// ---------------------------------------------------------------------------

/// Minimum admin password length. Length-only check: no composition rules.
pub const MIN_ADMIN_PASSWORD_LEN: usize = 12;
/// Absolute maximum for inference steps (matches the reusable engine).
pub const MAX_INFERENCE_STEPS: usize = 50;
/// Default pinned HuggingFace revision for Supertonic-3 model downloads.
pub const DEFAULT_MODEL_REVISION: &str = "3cadd1ee6394adea1bd021217a0e650ede09a323";

/// Default cap on waiting + playing items in the server-side playback queue.
pub const DEFAULT_MAX_PLAYBACK_QUEUE_ITEMS: usize = 100;
/// Default TCP connect timeout for model downloads.
pub const DEFAULT_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS: u64 = 10;
/// Default total timeout per model file download.
pub const DEFAULT_MODEL_DOWNLOAD_TIMEOUT_SECS: u64 = 1800;
/// Default per-token TTS request budget: 300 requests per 60s window.
/// Sized for high-frequency event-driven workloads (chat, gifts, automation);
/// sustained abuse is still bounded by the token-bucket refill rate.
pub const DEFAULT_TTS_RATE_LIMIT_REQUESTS: u32 = 300;
pub const DEFAULT_TTS_RATE_LIMIT_WINDOW_SECS: u64 = 60;
/// Default audio output selection: follow the OS default output device.
pub const DEFAULT_AUDIO_OUTPUT_DEVICE: &str = "default";
/// Maximum audio output device selection length (Unicode characters).
pub const MAX_AUDIO_OUTPUT_DEVICE_LEN: usize = 256;

/// Current `config.toml` schema version.
pub const CONFIG_VERSION: u32 = 1;

/// Upper bounds that keep operator input from overflowing internal
/// synchronization primitives or requesting absurd allocations.
pub const MAX_CONCURRENT_INFERENCE_LIMIT: usize = 64;
pub const MAX_PENDING_INFERENCE_LIMIT: usize = 10_000;
pub const MAX_TEXT_LENGTH_LIMIT: usize = 1_000_000;
pub const MAX_CHUNK_CHARS_LIMIT: usize = 10_000;
pub const MAX_REQUEST_TIMEOUT_SECS: u64 = 3600;
pub const MAX_RATE_LIMIT_WINDOW_SECS: u64 = 86_400;
pub const MIN_ADMIN_SESSION_EXPIRY_SECS: i64 = 60;
pub const MAX_ADMIN_SESSION_EXPIRY_SECS: i64 = 2_592_000;
pub const MIN_BODY_BYTES: usize = 1024;
pub const MAX_BODY_BYTES: usize = 16_777_216;
pub const MAX_PLAYBACK_QUEUE_ITEMS_LIMIT: usize = 10_000;
pub const MAX_DOWNLOAD_CONNECT_TIMEOUT_SECS: u64 = 300;
pub const MAX_DOWNLOAD_TIMEOUT_SECS: u64 = 86_400;
pub const MAX_RATE_LIMIT_REQUESTS: u32 = 1_000_000;

/// Default HTTP port.
pub const DEFAULT_PORT: u16 = 17842;

/// Passwords that are never accepted, even if they met the length rule.
pub const REJECTED_PASSWORDS: &[&str] = &[
    "1234",
    "password",
    "admin",
    "change-me",
    "changeme",
    "change-me-to-a-strong-password",
    "example-password",
    "example_password",
    "your-password",
    "your_password",
    "your-secure-password",
    "your_secure_password",
    "your_strong_password_12_plus_chars",
    "default-password",
    "letmein",
    "qwerty",
    "sonicboom",
];

// ---------------------------------------------------------------------------
// Authentication modes
// ---------------------------------------------------------------------------

/// How API requests are authenticated.
///
/// - `local`: no API authentication; the server must bind loopback only.
/// - `token`: bearer-token authentication required; may bind anywhere.
/// - `none`: development-only insecure mode; non-loopback binding requires
///   an explicit `security.allow_insecure_remote` opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AuthMode {
    /// Loopback-only, no API authentication (desktop default).
    Local,
    /// Bearer token required (server/headless default).
    #[default]
    Token,
    /// No authentication (development only).
    None,
}

impl AuthMode {
    /// Whether requests to the API require a bearer token.
    pub fn requires_token(self) -> bool {
        matches!(self, AuthMode::Token)
    }
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

/// HTTP listener and request timeouts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Address to bind. `local` auth mode requires a loopback address.
    #[serde(default = "default_bind")]
    pub bind: IpAddr,
    /// TCP port.
    #[serde(default = "default_port")]
    pub port: u16,
    /// API authentication mode.
    #[serde(default = "default_auth_mode")]
    pub auth_mode: AuthMode,
    /// Per-request timeout in seconds.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
}

/// Admin panel credentials and sessions. The admin **password** is never
/// stored here — it lives in the secret store.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdminConfig {
    /// Admin username (constant-time compared at login).
    #[serde(default = "default_admin_username")]
    pub username: String,
    /// Admin session inactivity expiry in seconds.
    #[serde(default = "default_admin_session_expiry_secs")]
    pub session_expiry_secs: i64,
    /// Accept the development `SAMPLE_TOKEN` credential (never in production).
    #[serde(default)]
    pub enable_sample_token: bool,
}

/// Audio playback configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioConfig {
    /// Output device selection (`default` follows the OS default output).
    #[serde(default = "default_audio_output_device")]
    pub output_device: String,
    /// Playback volume (0.0–1.0).
    #[serde(default = "default_volume")]
    pub volume: f32,
    /// Hard cap on waiting + playing items in the playback queue.
    #[serde(default = "default_max_playback_queue_items")]
    pub max_playback_queue_items: usize,
}

/// Model download, verification, and inference settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Directory for downloaded ONNX model files.
    #[serde(default = "default_model_cache_dir")]
    pub cache_dir: PathBuf,
    /// Pinned HuggingFace revision (40-char commit SHA).
    #[serde(default = "default_model_revision")]
    pub revision: String,
    /// Optional JSON manifest mapping filenames to expected SHA-256 digests.
    #[serde(default)]
    pub hashes_path: Option<PathBuf>,
    /// Number of inference steps per synthesis.
    #[serde(default = "default_inference_steps")]
    pub inference_steps: usize,
    /// Supertonic inference engine: `auto` (candle when compiled in, otherwise
    /// ONNX Runtime), `ort`, `candle-cpu` or `candle-vulkan`.
    #[serde(default = "default_model_backend")]
    pub backend: String,
    /// TCP connect timeout for model downloads (seconds).
    #[serde(default = "default_model_download_connect_timeout_secs")]
    pub download_connect_timeout_secs: u64,
    /// Total timeout per model file download (seconds).
    #[serde(default = "default_model_download_timeout_secs")]
    pub download_timeout_secs: u64,
}

/// TTS request limits and HTTP body size caps.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TtsConfig {
    /// Maximum text length for TTS requests (Unicode characters).
    #[serde(default = "default_max_text_length")]
    pub max_text_length: usize,
    /// Hard maximum characters per synthesis chunk.
    #[serde(default = "default_max_chunk_chars")]
    pub max_chunk_chars: usize,
    /// Body limit for `/api/tts` (bytes).
    #[serde(default = "default_tts_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Body limit for OpenAI-compatible endpoints (bytes).
    #[serde(default = "default_openai_max_body_bytes")]
    pub openai_max_body_bytes: usize,
    /// Body limit for queue endpoints (bytes).
    #[serde(default = "default_queue_max_body_bytes")]
    pub queue_max_body_bytes: usize,
    /// Body limit for admin endpoints (bytes).
    #[serde(default = "default_admin_max_body_bytes")]
    pub admin_max_body_bytes: usize,
    /// Text normalization (pronunciation) settings.
    #[serde(default)]
    pub normalization: NormalizationConfig,
}

/// Text normalization applied before synthesis (numbers, dates,
/// currency, ...). Additive section: older files without
/// `[tts.normalization]` load with these defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NormalizationConfig {
    /// `false` forces mode `off` regardless of `mode`.
    pub enabled: bool,
    /// `auto` or `off` (case-insensitive; stored lowercase).
    #[serde(deserialize_with = "deserialize_normalization_mode")]
    pub mode: String,
    /// `auto` (per-language default) or a supported locale tag.
    pub default_locale: String,
    /// How written numbers and times are read: `precise`, `natural` or
    /// `digits` (case-insensitive; stored lowercase).
    #[serde(deserialize_with = "deserialize_normalization_mode")]
    pub reading_style: String,
    /// Report ambiguous expressions as warnings.
    pub report_ambiguities: bool,
    /// Maximum spoken/original length ratio (1.0..=16.0).
    pub max_expansion_ratio: f32,
    /// Maximum spoken text length in characters (1..=200000).
    pub max_normalized_chars: usize,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: "auto".to_string(),
            default_locale: "auto".to_string(),
            reading_style: "precise".to_string(),
            report_ambiguities: true,
            max_expansion_ratio: 8.0,
            max_normalized_chars: 20_000,
        }
    }
}

impl NormalizationConfig {
    /// Effective default mode: `enabled = false` is always `Off`;
    /// otherwise the parsed `mode` (falls back to `Auto` for a value
    /// validation would already have rejected).
    pub fn effective_mode(&self) -> crate::tts::normalization::NormalizationMode {
        use crate::tts::normalization::NormalizationMode;
        if !self.enabled {
            return NormalizationMode::Off;
        }
        NormalizationMode::parse(&self.mode).unwrap_or(NormalizationMode::Auto)
    }

    /// Effective reading style (falls back to [`ReadingStyle::Precise`] for a
    /// value validation would already have rejected).
    pub fn effective_reading_style(&self) -> crate::tts::normalization::ReadingStyle {
        crate::tts::normalization::ReadingStyle::parse(&self.reading_style).unwrap_or_default()
    }

    /// Normalizer limits; the input bound comes from `tts.max_text_length`.
    pub fn limits(&self, max_input_chars: usize) -> crate::tts::normalization::NormalizeLimits {
        crate::tts::normalization::NormalizeLimits {
            max_input_chars,
            max_output_chars: self.max_normalized_chars,
            max_expansion_ratio: self.max_expansion_ratio,
        }
    }
}

/// Trim and lowercase the mode or reading style so `"OFF"`/`"Natural"` are
/// stored as `off`/`natural`. Validation rejects other values.
fn deserialize_normalization_mode<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    Ok(raw.trim().to_ascii_lowercase())
}

/// Inference admission control.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceConfig {
    /// Deadline for queued and executing jobs, independent of HTTP response timeout.
    #[serde(default = "default_generation_deadline_secs")]
    pub generation_deadline_secs: u64,
    #[serde(default = "default_job_retention_secs")]
    pub job_retention_secs: u64,
    #[serde(default = "default_max_retained_jobs")]
    pub max_retained_jobs: usize,
    /// Maximum concurrent model inferences.
    #[serde(default = "default_max_concurrent_inference")]
    pub max_concurrent: usize,
    /// Maximum queued (waiting) inference requests.
    #[serde(default = "default_max_pending_inference")]
    pub max_pending: usize,
}

/// Per-token token-bucket rate limiting for expensive endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateLimitConfig {
    /// Sustained request budget per window (`0` disables limiting).
    #[serde(default = "default_rate_limit_requests")]
    pub requests: u32,
    /// Rate-limit window in seconds.
    #[serde(default = "default_rate_limit_window_secs")]
    pub window_secs: u64,
    /// Burst capacity (defaults to `requests`).
    #[serde(default = "default_rate_limit_burst")]
    pub burst: u32,
}

/// Structured logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// Optional tracing directives, including the legacy RUST_LOG override.
    #[serde(default)]
    pub filter: Option<String>,
    /// Log level filter (e.g. `info`, `debug`).
    #[serde(default = "default_log_level")]
    pub level: String,
    /// Write logs to the log directory.
    #[serde(default = "default_log_to_file")]
    pub to_file: bool,
    /// Write logs to stdout.
    #[serde(default = "default_log_to_stdout")]
    pub to_stdout: bool,
}

/// Filesystem paths for data directories.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathsConfig {
    /// Log file directory.
    #[serde(default = "default_logs_dir")]
    pub logs: String,
    /// Allowed audio directory for the playback queue (prevents path
    /// traversal). Mandatory when the `playback` feature is enabled.
    #[serde(default)]
    pub audio: Option<String>,
    /// Directory for temporary synthesized playback files.
    #[serde(default = "default_temp_audio_dir")]
    pub temp_audio: String,
    /// API token store file.
    #[serde(default = "default_token_store_path")]
    pub token_store: String,
}

/// Removed Candle `[providers.qwen3]` section, kept only so old
/// configuration files still boot. Every field is accepted and ignored;
/// the server logs a warning pointing at `[providers.qwen3_gguf]`.
/// Do not add new fields here and do not read this anywhere: it carries
/// no runtime meaning since the Candle provider was deleted.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeprecatedQwen3Section {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub model_dir: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub max_concurrent: Option<usize>,
    #[serde(default)]
    pub cache_entries: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

/// Removed GPT-SoVITS `[providers.gpt_sovits]` section, kept only so old
/// configuration files still boot. Every field is accepted and ignored;
/// the server logs a warning pointing at `[providers.pocket_tts]`.
/// Do not add new fields here and do not read this anywhere: it carries
/// no runtime meaning since the GPT-SoVITS provider was retired. Stored
/// voice profiles that reference GPT-SoVITS models are never reinterpreted
/// as Pocket TTS voices; their reference WAVs stay on disk for explicit
/// user recreation as Pocket profiles.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeprecatedGptSovitsSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub model_dir: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub max_concurrent: Option<usize>,
    #[serde(default)]
    pub cache_entries: Option<usize>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

/// Native Qwen3-TTS 1.7B GGUF/ONNX provider settings
/// (`[providers.qwen3_gguf]`). Additive: older configuration files without
/// this section keep working via `#[serde(default)]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3GgufConfig {
    /// Enable the GGUF/ONNX provider (default `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Directory holding the verified GGUF/ONNX bundle.
    #[serde(default = "default_qwen_gguf_model_dir")]
    pub model_dir: PathBuf,
    /// Requested device: `auto` (CPU unless a GPU backend is compiled in),
    /// `cpu`, `vulkan` (requires `qwen-gguf-vulkan`), or `cuda`
    /// (requires `qwen-gguf-cuda`; rejected otherwise).
    #[serde(default = "default_provider_device")]
    pub device: String,
    /// llama.cpp GPU layer offload (`-1` offloads all layers, `0` is CPU).
    #[serde(default = "default_qwen_gguf_gpu_layers")]
    pub gpu_layers: i32,
    /// Worker threads for GGUF inference.
    #[serde(default = "default_qwen_gguf_threads")]
    pub threads: usize,
    /// Maximum concurrent syntheses (serialized per model by default).
    #[serde(default = "default_provider_concurrency")]
    pub max_concurrent: usize,
    /// Voice-conditioning cache entries.
    #[serde(default = "default_provider_cache")]
    pub cache_entries: usize,
    /// Maximum acoustic code frames per synthesis.
    #[serde(default = "default_qwen_gguf_max_tokens")]
    pub max_tokens: usize,
    /// Explicit `device = "vulkan"`/`"cuda"` that cannot initialize normally
    /// fails the load with an actionable error. Set `true` to allow a
    /// reported fallback to CPU instead; `device = "auto"` always permits it.
    #[serde(default)]
    pub allow_cpu_fallback: bool,
    /// Default sampling for Qwen3-TTS (`[providers.qwen3_gguf.sampling]`).
    /// Omitted means the pinned upstream `balanced` behavior. Per-request
    /// overrides take precedence over this default.
    #[serde(default)]
    pub sampling: SamplingRequest,
    /// Quality detection/regeneration policy (`[providers.qwen3_gguf.quality]`).
    #[serde(default)]
    pub quality: QwenQualitySection,
}

/// Qwen3-TTS quality detection configuration
/// (`[providers.qwen3_gguf.quality]`). Disabled by default: detection adds
/// deterministic CPU work but no regeneration unless `policy = "retry"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenQualitySection {
    /// Enable detection/reporting (default `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Handling mode: `off`, `report`, `retry`, or `experimental_trim`.
    #[serde(default = "default_qwen_quality_policy")]
    pub policy: String,
    /// Automatic regeneration attempts (hard ceiling of 1).
    #[serde(default = "default_qwen_quality_retries")]
    pub max_retries: usize,
}

impl Default for QwenQualitySection {
    fn default() -> Self {
        Self {
            enabled: false,
            policy: default_qwen_quality_policy(),
            max_retries: default_qwen_quality_retries(),
        }
    }
}

/// Native Pocket TTS ONNX provider settings (`[providers.pocket_tts]`).
/// Additive: older configuration files without this section keep working
/// via `#[serde(default)]`. Disabled by default; weights download only
/// after `enabled = true`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PocketTtsConfig {
    /// Enable the Pocket TTS provider (default `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Directory holding the verified per-language ONNX bundles
    /// (English, Spanish, Portuguese, German, Italian, French 24-layer preview).
    #[serde(default = "default_pocket_model_dir")]
    pub model_dir: PathBuf,
    /// Model identifier (`pocket-tts-en/es/pt/de/it`, `pocket-tts-fr-24l`).
    #[serde(default = "default_pocket_model_id")]
    pub model_id: String,
    /// Pinned export revision.
    #[serde(default = "default_pocket_revision")]
    pub revision: String,
    /// CPU only: `cpu` or `auto`. GPU values are rejected explicitly.
    #[serde(default = "default_provider_device")]
    pub device: String,
    /// Maximum concurrent Pocket TTS syntheses (serialized per model).
    #[serde(default = "default_provider_concurrency")]
    pub max_concurrent: usize,
    /// Voice-embedding cache entries.
    #[serde(default = "default_provider_cache")]
    pub cache_entries: usize,
    /// Sampling temperature (`0.0` disables noise, upstream default `0.7`).
    #[serde(default = "default_pocket_temperature")]
    pub temperature: f32,
    /// Flow-matching integration steps (upstream default `1`).
    #[serde(default = "default_pocket_lsd_steps")]
    pub lsd_steps: usize,
    /// ORT intra-op threads (clamped to 1..=32).
    #[serde(default = "default_pocket_cpu_threads")]
    pub cpu_threads: usize,
}

impl Default for ProvidersConfig {
    fn default() -> Self {
        Self {
            qwen3_gguf: Qwen3GgufConfig::default(),
            pocket_tts: PocketTtsConfig::default(),
            qwen3_candle: CandleProviderConfig::default(),
            kyutai_candle: default_kyutai_candle(),
            qwen3: DeprecatedQwen3Section::default(),
            gpt_sovits: DeprecatedGptSovitsSection::default(),
        }
    }
}

/// Settings shared by the candle voice providers (`[providers.qwen3_candle]`,
/// `[providers.kyutai_candle]`). Weights are read from `model_dir`; with
/// `download = true` the pinned bundle is fetched and SHA-256 verified there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandleProviderConfig {
    /// Enable the provider (default `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Directory holding the model files (`config.json`, safetensors, tokenizer).
    #[serde(default = "default_qwen_candle_model_dir")]
    pub model_dir: PathBuf,
    /// Kyutai only: directory of `kyutai/tts-voices` embeddings (`<voice>.safetensors`).
    #[serde(default = "default_kyutai_voices_dir")]
    pub voices_dir: PathBuf,
    /// Kyutai only: voice used when a request does not name one.
    #[serde(default = "default_kyutai_voice")]
    pub default_voice: String,
    /// `auto` (Vulkan when compiled in, else CPU), `cpu` or `vulkan`.
    #[serde(default = "default_provider_device")]
    pub device: String,
    /// Download and verify the pinned bundle (and Kyutai's default voice) into the
    /// directories above when enabled. Set `false` to use files you manage yourself.
    #[serde(default = "default_candle_download")]
    pub download: bool,
}

impl Default for CandleProviderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: default_qwen_candle_model_dir(),
            voices_dir: default_kyutai_voices_dir(),
            default_voice: default_kyutai_voice(),
            device: default_provider_device(),
            download: default_candle_download(),
        }
    }
}

fn default_qwen_candle_model_dir() -> PathBuf {
    PathBuf::from("models/qwen3-tts-candle")
}
fn default_kyutai_voices_dir() -> PathBuf {
    PathBuf::from("models/kyutai-tts/voices")
}
fn default_kyutai_voice() -> String {
    "calm".to_string()
}
fn default_candle_download() -> bool {
    true
}
fn default_kyutai_candle() -> CandleProviderConfig {
    CandleProviderConfig {
        model_dir: PathBuf::from("models/kyutai-tts"),
        ..CandleProviderConfig::default()
    }
}

/// Provider section grouping (`[providers.qwen3_gguf]`,
/// `[providers.pocket_tts]`). Leftover `[providers.qwen3]` Candle and
/// `[providers.gpt_sovits]` sections are tolerated for boot compatibility
/// but ignored (see [`DeprecatedQwen3Section`] and
/// [`DeprecatedGptSovitsSection`]): delete them from old configuration
/// files.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub qwen3_gguf: Qwen3GgufConfig,
    #[serde(default)]
    pub pocket_tts: PocketTtsConfig,
    /// Qwen3-TTS (0.6B/1.7B Base safetensors) on candle (`[providers.qwen3_candle]`).
    #[serde(default)]
    pub qwen3_candle: CandleProviderConfig,
    /// Kyutai TTS 1.6B en/fr on candle (`[providers.kyutai_candle]`).
    #[serde(default = "default_kyutai_candle")]
    pub kyutai_candle: CandleProviderConfig,
    #[serde(default, skip_serializing)]
    pub qwen3: DeprecatedQwen3Section,
    #[serde(default, skip_serializing)]
    pub gpt_sovits: DeprecatedGptSovitsSection,
}

/// Voice-profile store settings (`[voices]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoicesConfig {
    #[serde(default = "default_max_voice_profiles")]
    pub max_profiles: usize,
    /// Directory for `profiles.json` plus `<uuid>.wav` references.
    #[serde(default = "default_voices_dir")]
    pub store_dir: String,
    /// Maximum reference upload size in bytes.
    #[serde(default = "default_voices_max_upload_bytes")]
    pub max_upload_bytes: usize,
}

/// Security-relevant settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[derive(Default)]
pub struct SecurityConfig {
    /// Honor forwarded headers (`X-Forwarded-For`) from trusted proxies.
    #[serde(default)]
    pub trust_proxy: bool,
    /// Proxy addresses/CIDRs allowed to forward client IPs.
    #[serde(default)]
    pub trusted_proxies: Vec<String>,
    /// Set the `Secure` attribute on admin session cookies (HTTPS only).
    #[serde(default)]
    pub cookie_secure: bool,
    /// Enable HSTS (known-HTTPS deployments only).
    #[serde(default)]
    pub enable_hsts: bool,
    /// Explicit opt-in for `auth_mode = "none"` on non-loopback binds.
    #[serde(default)]
    pub allow_insecure_remote: bool,
}

/// The complete SonicBoom configuration, split into logical sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    /// Schema version of the persisted file.
    #[serde(default = "default_config_version")]
    pub version: u32,
    /// Monotonic revision of the persisted file.
    /// Every successful change increments it; used
    /// for optimistic concurrency.
    #[serde(default = "default_revision")]
    pub revision: u64,
    /// Whether first-run setup has been completed.
    #[serde(default)]
    pub setup_complete: bool,

    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub admin: AdminConfig,
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub tts: TtsConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub rate_limit: RateLimitConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub paths: PathsConfig,
    #[serde(default)]
    pub security: SecurityConfig,
    #[serde(default)]
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub voices: VoicesConfig,
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

fn default_config_version() -> u32 {
    CONFIG_VERSION
}

fn default_revision() -> u64 {
    1
}
fn default_bind() -> IpAddr {
    "127.0.0.1".parse().expect("loopback bind default")
}
fn default_port() -> u16 {
    DEFAULT_PORT
}
fn default_auth_mode() -> AuthMode {
    AuthMode::Token
}
fn default_request_timeout_secs() -> u64 {
    120
}
fn default_admin_username() -> String {
    "admin".to_string()
}
fn default_admin_session_expiry_secs() -> i64 {
    8 * 3600
}
fn default_audio_output_device() -> String {
    DEFAULT_AUDIO_OUTPUT_DEVICE.to_string()
}
fn default_volume() -> f32 {
    1.0
}
fn default_max_playback_queue_items() -> usize {
    DEFAULT_MAX_PLAYBACK_QUEUE_ITEMS
}
fn default_model_cache_dir() -> PathBuf {
    PathBuf::from("./models")
}
fn default_model_revision() -> String {
    DEFAULT_MODEL_REVISION.to_string()
}
fn default_model_backend() -> String {
    "auto".to_string()
}

fn default_inference_steps() -> usize {
    5
}
fn default_model_download_connect_timeout_secs() -> u64 {
    DEFAULT_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS
}
fn default_model_download_timeout_secs() -> u64 {
    DEFAULT_MODEL_DOWNLOAD_TIMEOUT_SECS
}
fn default_max_text_length() -> usize {
    10_000
}
fn default_max_chunk_chars() -> usize {
    200
}
fn default_tts_max_body_bytes() -> usize {
    65_536
}
fn default_openai_max_body_bytes() -> usize {
    65_536
}
fn default_queue_max_body_bytes() -> usize {
    16_384
}
fn default_admin_max_body_bytes() -> usize {
    16_384
}
fn default_generation_deadline_secs() -> u64 {
    1800
}
fn default_job_retention_secs() -> u64 {
    600
}
fn default_max_retained_jobs() -> usize {
    64
}
fn default_max_concurrent_inference() -> usize {
    1
}
fn default_max_pending_inference() -> usize {
    8
}
fn default_rate_limit_requests() -> u32 {
    DEFAULT_TTS_RATE_LIMIT_REQUESTS
}
fn default_rate_limit_window_secs() -> u64 {
    DEFAULT_TTS_RATE_LIMIT_WINDOW_SECS
}
fn default_rate_limit_burst() -> u32 {
    DEFAULT_TTS_RATE_LIMIT_REQUESTS
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_log_to_file() -> bool {
    true
}
fn default_log_to_stdout() -> bool {
    true
}
fn default_logs_dir() -> String {
    "./logs".to_string()
}
fn default_temp_audio_dir() -> String {
    "./temp_audio".to_string()
}
fn default_token_store_path() -> String {
    "./tokens.json".to_string()
}
fn default_provider_device() -> String {
    "auto".to_string()
}
fn default_provider_concurrency() -> usize {
    1
}
fn default_provider_cache() -> usize {
    16
}
fn default_qwen_gguf_model_dir() -> PathBuf {
    PathBuf::from("./models/qwen3-tts-1.7b-gguf")
}
fn default_qwen_gguf_gpu_layers() -> i32 {
    -1
}
fn default_qwen_gguf_threads() -> usize {
    4
}
fn default_qwen_gguf_max_tokens() -> usize {
    512
}
fn default_qwen_quality_policy() -> String {
    "report".to_string()
}
fn default_qwen_quality_retries() -> usize {
    1
}
fn default_pocket_model_dir() -> PathBuf {
    PathBuf::from("./models/pocket-tts")
}
fn default_pocket_model_id() -> String {
    "pocket-tts-en".to_string()
}
fn default_pocket_revision() -> String {
    "58a6d00cf13d239b6748cb0769f35c580a8f606c".to_string()
}
fn default_pocket_temperature() -> f32 {
    0.7
}
fn default_pocket_lsd_steps() -> usize {
    1
}
fn default_pocket_cpu_threads() -> usize {
    4
}

impl Default for PocketTtsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: default_pocket_model_dir(),
            model_id: default_pocket_model_id(),
            revision: default_pocket_revision(),
            device: default_provider_device(),
            max_concurrent: default_provider_concurrency(),
            cache_entries: default_provider_cache(),
            temperature: default_pocket_temperature(),
            lsd_steps: default_pocket_lsd_steps(),
            cpu_threads: default_pocket_cpu_threads(),
        }
    }
}

impl Default for Qwen3GgufConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: default_qwen_gguf_model_dir(),
            device: default_provider_device(),
            gpu_layers: default_qwen_gguf_gpu_layers(),
            threads: default_qwen_gguf_threads(),
            max_concurrent: default_provider_concurrency(),
            cache_entries: default_provider_cache(),
            max_tokens: default_qwen_gguf_max_tokens(),
            allow_cpu_fallback: false,
            sampling: SamplingRequest::default(),
            quality: QwenQualitySection::default(),
        }
    }
}
fn default_max_voice_profiles() -> usize {
    256
}
fn default_voices_dir() -> String {
    "./voices".to_string()
}
fn default_voices_max_upload_bytes() -> usize {
    8_388_608
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            revision: default_revision(),
            setup_complete: false,
            server: ServerConfig {
                bind: default_bind(),
                port: default_port(),
                auth_mode: default_auth_mode(),
                request_timeout_secs: default_request_timeout_secs(),
            },
            admin: AdminConfig {
                username: default_admin_username(),
                session_expiry_secs: default_admin_session_expiry_secs(),
                enable_sample_token: false,
            },
            audio: AudioConfig {
                output_device: default_audio_output_device(),
                volume: default_volume(),
                max_playback_queue_items: default_max_playback_queue_items(),
            },
            model: ModelConfig {
                cache_dir: default_model_cache_dir(),
                revision: default_model_revision(),
                hashes_path: None,
                inference_steps: default_inference_steps(),
                backend: default_model_backend(),
                download_connect_timeout_secs: default_model_download_connect_timeout_secs(),
                download_timeout_secs: default_model_download_timeout_secs(),
            },
            tts: TtsConfig {
                max_text_length: default_max_text_length(),
                max_chunk_chars: default_max_chunk_chars(),
                max_body_bytes: default_tts_max_body_bytes(),
                openai_max_body_bytes: default_openai_max_body_bytes(),
                queue_max_body_bytes: default_queue_max_body_bytes(),
                admin_max_body_bytes: default_admin_max_body_bytes(),
                normalization: NormalizationConfig::default(),
            },
            inference: InferenceConfig {
                max_concurrent: default_max_concurrent_inference(),
                max_pending: default_max_pending_inference(),
                generation_deadline_secs: default_generation_deadline_secs(),
                job_retention_secs: default_job_retention_secs(),
                max_retained_jobs: default_max_retained_jobs(),
            },
            rate_limit: RateLimitConfig {
                requests: default_rate_limit_requests(),
                window_secs: default_rate_limit_window_secs(),
                burst: default_rate_limit_burst(),
            },
            logging: LoggingConfig {
                filter: None,
                level: default_log_level(),
                to_file: default_log_to_file(),
                to_stdout: default_log_to_stdout(),
            },
            paths: PathsConfig::default(),
            security: SecurityConfig {
                trust_proxy: false,
                trusted_proxies: Vec::new(),
                cookie_secure: false,
                enable_hsts: false,
                allow_insecure_remote: false,
            },
            providers: ProvidersConfig::default(),
            voices: VoicesConfig::default(),
        }
    }
}

impl Default for VoicesConfig {
    fn default() -> Self {
        Self {
            max_profiles: default_max_voice_profiles(),
            store_dir: default_voices_dir(),
            max_upload_bytes: default_voices_max_upload_bytes(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            port: default_port(),
            auth_mode: default_auth_mode(),
            request_timeout_secs: default_request_timeout_secs(),
        }
    }
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            username: default_admin_username(),
            session_expiry_secs: default_admin_session_expiry_secs(),
            enable_sample_token: false,
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            output_device: default_audio_output_device(),
            volume: default_volume(),
            max_playback_queue_items: default_max_playback_queue_items(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_model_cache_dir(),
            revision: default_model_revision(),
            hashes_path: None,
            inference_steps: default_inference_steps(),
            backend: default_model_backend(),
            download_connect_timeout_secs: default_model_download_connect_timeout_secs(),
            download_timeout_secs: default_model_download_timeout_secs(),
        }
    }
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            max_text_length: default_max_text_length(),
            max_chunk_chars: default_max_chunk_chars(),
            max_body_bytes: default_tts_max_body_bytes(),
            openai_max_body_bytes: default_openai_max_body_bytes(),
            queue_max_body_bytes: default_queue_max_body_bytes(),
            admin_max_body_bytes: default_admin_max_body_bytes(),
            normalization: NormalizationConfig::default(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_concurrent: default_max_concurrent_inference(),
            max_pending: default_max_pending_inference(),
            generation_deadline_secs: default_generation_deadline_secs(),
            job_retention_secs: default_job_retention_secs(),
            max_retained_jobs: default_max_retained_jobs(),
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests: default_rate_limit_requests(),
            window_secs: default_rate_limit_window_secs(),
            burst: default_rate_limit_burst(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            filter: None,
            level: default_log_level(),
            to_file: default_log_to_file(),
            to_stdout: default_log_to_stdout(),
        }
    }
}

impl Default for PathsConfig {
    fn default() -> Self {
        // Desktop/playback builds default the queue root
        // to the platform config directory so a fresh
        // installation validates out of the box.
        let audio = {
            #[cfg(feature = "playback")]
            {
                Some(
                    super::paths::config_dir()
                        .join("audio")
                        .to_string_lossy()
                        .to_string(),
                )
            }
            #[cfg(not(feature = "playback"))]
            {
                None
            }
        };
        Self {
            logs: default_logs_dir(),
            audio,
            temp_audio: default_temp_audio_dir(),
            token_store: default_token_store_path(),
        }
    }
}

/// A malformed configuration value. Absent values fall back to defaults;
/// present-but-malformed values are a startup failure, never a silent
/// default — especially for security-sensitive options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    pub variable: String,
    pub message: String,
}

impl ConfigError {
    pub fn invalid(variable: &str, message: impl Into<String>) -> Self {
        Self {
            variable: variable.to_string(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid value for {}: {}", self.variable, self.message)
    }
}

impl std::error::Error for ConfigError {}
