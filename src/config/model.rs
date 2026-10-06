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
}

/// Inference admission control.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceConfig {
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

/// Security-relevant settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
            },
            inference: InferenceConfig {
                max_concurrent: default_max_concurrent_inference(),
                max_pending: default_max_pending_inference(),
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
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_concurrent: default_max_concurrent_inference(),
            max_pending: default_max_pending_inference(),
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

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            trust_proxy: false,
            trusted_proxies: Vec::new(),
            cookie_secure: false,
            enable_hsts: false,
            allow_insecure_remote: false,
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
