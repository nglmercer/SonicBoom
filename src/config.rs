use std::env;
use std::fmt;

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

/// Passwords that are never accepted, even if they met the length rule.
const REJECTED_PASSWORDS: &[&str] = &[
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

/// A malformed environment value. Absent variables fall back to defaults;
/// present-but-malformed values are a startup failure, never a silent
/// default — especially for security-sensitive options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    pub variable: String,
    pub message: String,
}

impl ConfigError {
    fn invalid(variable: &str, message: impl Into<String>) -> Self {
        Self {
            variable: variable.to_string(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid value for {}: {}", self.variable, self.message)
    }
}

impl std::error::Error for ConfigError {}

pub struct AppConfig {
    pub admin_id: String,
    pub admin_pw: String,
    pub enable_sample_token: bool,
    pub token_store_path: String,
    pub model_cache_dir: String,
    pub model_revision: String,
    /// Optional JSON file mapping model filenames to expected SHA-256 hex
    /// digests. When set, downloads and cached files are verified.
    pub model_hashes_path: Option<String>,
    pub hf_token: Option<String>,
    pub inference_steps: usize,
    pub port: u16,
    // Logging configuration
    pub log_dir: String,
    pub log_level: String,
    pub log_to_file: bool,
    pub log_to_stdout: bool,
    // Authentication configuration
    pub auth_required: bool,
    // Security: allowed audio directory for queue (prevents path traversal).
    // Mandatory when the `playback` feature (filesystem queue) is enabled.
    pub allowed_audio_dir: Option<String>,
    // Maximum text length for TTS requests (Unicode characters)
    pub max_text_length: usize,
    // Request timeout in seconds
    pub request_timeout_secs: u64,
    // Inference admission control
    pub max_concurrent_inference: usize,
    pub max_pending_inference: usize,
    // Text chunking (Unicode characters, hard maximum per chunk)
    pub max_chunk_chars: usize,
    // Rate limiting for expensive TTS endpoints (per token, sliding window)
    pub tts_rate_limit_requests: u32,
    pub tts_rate_limit_window_secs: u64,
    // HTTP body limits (bytes, enforced before full allocation)
    pub tts_max_body_bytes: usize,
    pub openai_max_body_bytes: usize,
    pub queue_max_body_bytes: usize,
    pub admin_max_body_bytes: usize,
    // Reverse proxy handling: never trust forwarded headers unless explicit.
    pub trust_proxy: bool,
    pub trusted_proxies: Vec<String>,
    // Admin session cookie: set true when serving HTTPS in production.
    pub cookie_secure: bool,
    pub admin_session_expiry_secs: i64,
    // Directory for temporary TTS playback files.
    pub temp_audio_dir: String,
    // Opt-in Strict-Transport-Security header (only for known-HTTPS deployments).
    pub enable_hsts: bool,
    // Maximum items waiting in the server-side playback queue.
    pub max_playback_queue_items: usize,
    // Model download network limits.
    pub model_download_connect_timeout_secs: u64,
    pub model_download_timeout_secs: u64,
}

/// Accepted boolean spellings: `true`/`false` (any ASCII case) or `1`/`0`.
/// Anything else — `yes`, `no`, typos, empty strings — is rejected so
/// operator mistakes fail loudly instead of silently becoming `false`.
fn parse_bool_str(value: &str) -> Option<bool> {
    if value.eq_ignore_ascii_case("true") || value == "1" {
        Some(true)
    } else if value.eq_ignore_ascii_case("false") || value == "0" {
        Some(false)
    } else {
        None
    }
}

fn env_bool(
    get: &impl Fn(&str) -> Option<String>,
    name: &str,
    default: bool,
) -> Result<bool, ConfigError> {
    match get(name) {
        None => Ok(default),
        Some(raw) => parse_bool_str(raw.trim()).ok_or_else(|| {
            ConfigError::invalid(
                name,
                format!("expected one of: true, false, 1, 0 (case-insensitive), got '{raw}'"),
            )
        }),
    }
}

fn env_parse<T>(
    get: &impl Fn(&str) -> Option<String>,
    name: &str,
    default: T,
) -> Result<T, ConfigError>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    match get(name) {
        None => Ok(default),
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Err(ConfigError::invalid(
                    name,
                    "value is present but empty; unset it to use the default",
                ));
            }
            trimmed
                .parse::<T>()
                .map_err(|e| ConfigError::invalid(name, format!("could not parse '{raw}': {e}")))
        }
    }
}

fn env_usize(
    get: &impl Fn(&str) -> Option<String>,
    name: &str,
    default: usize,
) -> Result<usize, ConfigError> {
    env_parse(get, name, default)
}

fn env_u64(
    get: &impl Fn(&str) -> Option<String>,
    name: &str,
    default: u64,
) -> Result<u64, ConfigError> {
    env_parse(get, name, default)
}

impl AppConfig {
    /// Load configuration from the process environment.
    ///
    /// Absent variables use documented defaults, but a variable that is
    /// present and malformed is a hard error: the server refuses to start
    /// rather than silently running with a different security posture than
    /// the operator intended (e.g. `COOKIE_SECURE=treu` must not become
    /// `false`, and `REQUEST_TIMEOUT_SECS=banana` must not become `120`).
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_env_with(|name| env::var(name).ok())
    }

    fn from_env_with(get: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        Ok(Self {
            admin_id: get("SONICBOOM_ADMIN_ID").unwrap_or_else(|| "admin".to_string()),
            // No default password: missing/weak values fail validation.
            admin_pw: get("SONICBOOM_ADMIN_PW").unwrap_or_default(),
            enable_sample_token: env_bool(&get, "ENABLE_SAMPLE_TOKEN", false)?,
            token_store_path: get("TOKEN_STORE_PATH")
                .unwrap_or_else(|| "./tokens.json".to_string()),
            model_cache_dir: get("MODEL_CACHE_DIR").unwrap_or_else(|| "./models".to_string()),
            // Normalized to lowercase; validated as a 40-hex commit SHA.
            model_revision: get("MODEL_REVISION")
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| DEFAULT_MODEL_REVISION.to_string()),
            model_hashes_path: get("MODEL_SHA256_JSON_PATH"),
            hf_token: get("HF_TOKEN"),
            inference_steps: env_usize(&get, "INFERENCE_STEPS", 5)?,
            port: env_parse(&get, "PORT", 17842)?,
            // Logging settings
            log_dir: get("LOG_DIR").unwrap_or_else(|| "./logs".to_string()),
            log_level: get("LOG_LEVEL").unwrap_or_else(|| "info".to_string()),
            log_to_file: env_bool(&get, "LOG_TO_FILE", true)?,
            log_to_stdout: env_bool(&get, "LOG_TO_STDOUT", true)?,
            // Authentication settings (strict boolean; default requires auth).
            auth_required: env_bool(&get, "SONICBOOM_AUTH_REQUIRED", true)?,
            // Security: allowed audio directory for queue (prevents path traversal)
            allowed_audio_dir: get("ALLOWED_AUDIO_DIR"),
            // Maximum text length for TTS requests (default: 10000 chars)
            max_text_length: env_usize(&get, "MAX_TEXT_LENGTH", 10_000)?,
            // Request timeout in seconds (default: 120s)
            request_timeout_secs: env_u64(&get, "REQUEST_TIMEOUT_SECS", 120)?,
            max_concurrent_inference: env_usize(&get, "MAX_CONCURRENT_INFERENCE", 1)?,
            max_pending_inference: env_usize(&get, "MAX_PENDING_INFERENCE", 8)?,
            max_chunk_chars: env_usize(&get, "MAX_CHUNK_CHARS", 200)?,
            tts_rate_limit_requests: env_parse(&get, "TTS_RATE_LIMIT_REQUESTS", 20)?,
            tts_rate_limit_window_secs: env_u64(&get, "TTS_RATE_LIMIT_WINDOW_SECS", 60)?,
            tts_max_body_bytes: env_usize(&get, "TTS_MAX_BODY_BYTES", 65_536)?,
            openai_max_body_bytes: env_usize(&get, "OPENAI_MAX_BODY_BYTES", 65_536)?,
            queue_max_body_bytes: env_usize(&get, "QUEUE_MAX_BODY_BYTES", 16_384)?,
            admin_max_body_bytes: env_usize(&get, "ADMIN_MAX_BODY_BYTES", 16_384)?,
            trust_proxy: env_bool(&get, "TRUST_PROXY", false)?,
            trusted_proxies: get("TRUSTED_PROXIES")
                .map(|v| {
                    v.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            cookie_secure: env_bool(&get, "COOKIE_SECURE", false)?,
            admin_session_expiry_secs: env_parse(&get, "ADMIN_SESSION_EXPIRY_SECS", 8 * 3600)?,
            temp_audio_dir: get("TEMP_AUDIO_DIR").unwrap_or_else(|| "./temp_audio".to_string()),
            enable_hsts: env_bool(&get, "ENABLE_HSTS", false)?,
            max_playback_queue_items: env_usize(
                &get,
                "MAX_PLAYBACK_QUEUE_ITEMS",
                DEFAULT_MAX_PLAYBACK_QUEUE_ITEMS,
            )?,
            model_download_connect_timeout_secs: env_u64(
                &get,
                "MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS",
                DEFAULT_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS,
            )?,
            model_download_timeout_secs: env_u64(
                &get,
                "MODEL_DOWNLOAD_TIMEOUT_SECS",
                DEFAULT_MODEL_DOWNLOAD_TIMEOUT_SECS,
            )?,
        })
    }

    /// Validate configuration values. Security-sensitive misconfiguration
    /// fails closed: the server refuses to start.
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=MAX_INFERENCE_STEPS).contains(&self.inference_steps) {
            return Err(format!(
                "INFERENCE_STEPS must be between 1 and {MAX_INFERENCE_STEPS}"
            ));
        }
        if self.port == 0 {
            return Err("PORT must be greater than 0".to_string());
        }
        if !(1..=MAX_TEXT_LENGTH_LIMIT).contains(&self.max_text_length) {
            return Err(format!(
                "MAX_TEXT_LENGTH must be between 1 and {MAX_TEXT_LENGTH_LIMIT}"
            ));
        }
        if !(1..=MAX_CHUNK_CHARS_LIMIT).contains(&self.max_chunk_chars) {
            return Err(format!(
                "MAX_CHUNK_CHARS must be between 1 and {MAX_CHUNK_CHARS_LIMIT}"
            ));
        }
        if !(1..=MAX_CONCURRENT_INFERENCE_LIMIT).contains(&self.max_concurrent_inference) {
            return Err(format!(
                "MAX_CONCURRENT_INFERENCE must be between 1 and {MAX_CONCURRENT_INFERENCE_LIMIT}"
            ));
        }
        if !(0..=MAX_PENDING_INFERENCE_LIMIT).contains(&self.max_pending_inference) {
            return Err(format!(
                "MAX_PENDING_INFERENCE must be between 0 and {MAX_PENDING_INFERENCE_LIMIT}"
            ));
        }
        if !(1..=MAX_REQUEST_TIMEOUT_SECS).contains(&self.request_timeout_secs) {
            return Err(format!(
                "REQUEST_TIMEOUT_SECS must be between 1 and {MAX_REQUEST_TIMEOUT_SECS}"
            ));
        }
        if self.tts_rate_limit_requests > MAX_RATE_LIMIT_REQUESTS {
            return Err(format!(
                "TTS_RATE_LIMIT_REQUESTS must be between 0 and {MAX_RATE_LIMIT_REQUESTS} (0 disables limiting)"
            ));
        }
        if self.admin_id.trim().is_empty() {
            return Err("SONICBOOM_ADMIN_ID must not be empty".to_string());
        }
        self.validate_admin_password()?;
        #[cfg(feature = "playback")]
        {
            match &self.allowed_audio_dir {
                Some(dir) if !dir.trim().is_empty() => {}
                _ => {
                    return Err(
                        "ALLOWED_AUDIO_DIR must be set when the playback/filesystem queue is enabled. \
                         Set it in '.env' (e.g. ALLOWED_AUDIO_DIR=./audio) and create the directory \
                         ('mkdir -p audio'), or run headless with \
                         'cargo run --no-default-features --features server'."
                            .to_string(),
                    );
                }
            }
        }
        if !sonicboom::tts::download::is_valid_commit_sha(&self.model_revision) {
            return Err(
                "MODEL_REVISION must be a 40-character commit SHA (mutable names like 'main' are rejected)"
                    .to_string(),
            );
        }
        if self.model_revision != DEFAULT_MODEL_REVISION && self.model_hashes_path.is_none() {
            return Err(
                "custom MODEL_REVISION requires MODEL_SHA256_JSON_PATH with trusted hashes for that exact revision"
                    .to_string(),
            );
        }
        if !(1..=MAX_RATE_LIMIT_WINDOW_SECS).contains(&self.tts_rate_limit_window_secs) {
            return Err(format!(
                "TTS_RATE_LIMIT_WINDOW_SECS must be between 1 and {MAX_RATE_LIMIT_WINDOW_SECS}"
            ));
        }
        if !(MIN_ADMIN_SESSION_EXPIRY_SECS..=MAX_ADMIN_SESSION_EXPIRY_SECS)
            .contains(&self.admin_session_expiry_secs)
        {
            return Err(format!(
                "ADMIN_SESSION_EXPIRY_SECS must be between {MIN_ADMIN_SESSION_EXPIRY_SECS} and {MAX_ADMIN_SESSION_EXPIRY_SECS}"
            ));
        }
        for (name, value) in [
            ("TTS_MAX_BODY_BYTES", self.tts_max_body_bytes),
            ("OPENAI_MAX_BODY_BYTES", self.openai_max_body_bytes),
            ("QUEUE_MAX_BODY_BYTES", self.queue_max_body_bytes),
            ("ADMIN_MAX_BODY_BYTES", self.admin_max_body_bytes),
        ] {
            if !(MIN_BODY_BYTES..=MAX_BODY_BYTES).contains(&value) {
                return Err(format!(
                    "{name} must be between {MIN_BODY_BYTES} and {MAX_BODY_BYTES}"
                ));
            }
        }
        if !(1..=MAX_PLAYBACK_QUEUE_ITEMS_LIMIT).contains(&self.max_playback_queue_items) {
            return Err(format!(
                "MAX_PLAYBACK_QUEUE_ITEMS must be between 1 and {MAX_PLAYBACK_QUEUE_ITEMS_LIMIT}"
            ));
        }
        if !(1..=MAX_DOWNLOAD_CONNECT_TIMEOUT_SECS)
            .contains(&self.model_download_connect_timeout_secs)
        {
            return Err(format!(
                "MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS must be between 1 and {MAX_DOWNLOAD_CONNECT_TIMEOUT_SECS}"
            ));
        }
        if !(1..=MAX_DOWNLOAD_TIMEOUT_SECS).contains(&self.model_download_timeout_secs) {
            return Err(format!(
                "MODEL_DOWNLOAD_TIMEOUT_SECS must be between 1 and {MAX_DOWNLOAD_TIMEOUT_SECS}"
            ));
        }
        if self.trust_proxy && self.trusted_proxies.is_empty() {
            return Err(
                "TRUSTED_PROXIES must list at least one proxy address when TRUST_PROXY is enabled"
                    .to_string(),
            );
        }
        for entry in &self.trusted_proxies {
            if !crate::admin::client_ip::is_valid_proxy_entry(entry) {
                return Err(format!(
                    "TRUSTED_PROXIES entry '{entry}' is not a valid IP address or CIDR prefix"
                ));
            }
        }
        Ok(())
    }

    fn validate_admin_password(&self) -> Result<(), String> {
        if self.admin_pw.is_empty() {
            return Err(
                "SONICBOOM_ADMIN_PW is not set. Refusing to start with no admin password. \
                 Set SONICBOOM_ADMIN_PW to a strong password (minimum 12 characters). \
                 Config file: create '.env' in the project root (copy from '.env.example'), \
                 set SONICBOOM_ADMIN_PW there, run 'chmod 600 .env'. \
                 Generate with: python3 -c 'import secrets; print(secrets.token_urlsafe(24))'. \
                 Or export it: export SONICBOOM_ADMIN_PW='<generated>'. \
                 There is intentionally no default password (see SECURITY.md)."
                    .to_string(),
            );
        }
        // Length is measured in Unicode characters, matching the documented policy.
        if self.admin_pw.chars().count() < MIN_ADMIN_PASSWORD_LEN {
            return Err(format!(
                "SONICBOOM_ADMIN_PW must be at least {MIN_ADMIN_PASSWORD_LEN} characters"
            ));
        }
        if REJECTED_PASSWORDS
            .iter()
            .any(|bad| self.admin_pw.eq_ignore_ascii_case(bad))
        {
            return Err("SONICBOOM_ADMIN_PW must not be a well-known default password".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn valid_config() -> AppConfig {
        AppConfig {
            admin_id: "admin".to_string(),
            admin_pw: "correct-horse-battery-staple".to_string(),
            enable_sample_token: false,
            token_store_path: "./tokens.json".to_string(),
            model_cache_dir: "./models".to_string(),
            model_revision: DEFAULT_MODEL_REVISION.to_string(),
            model_hashes_path: None,
            hf_token: None,
            inference_steps: 5,
            port: 17842,
            log_dir: "./logs".to_string(),
            log_level: "info".to_string(),
            log_to_file: true,
            log_to_stdout: true,
            auth_required: true,
            #[cfg(feature = "playback")]
            allowed_audio_dir: Some("./audio".to_string()),
            #[cfg(not(feature = "playback"))]
            allowed_audio_dir: None,
            max_text_length: 10_000,
            request_timeout_secs: 120,
            max_concurrent_inference: 1,
            max_pending_inference: 8,
            max_chunk_chars: 200,
            tts_rate_limit_requests: 20,
            tts_rate_limit_window_secs: 60,
            tts_max_body_bytes: 65_536,
            openai_max_body_bytes: 65_536,
            queue_max_body_bytes: 16_384,
            admin_max_body_bytes: 16_384,
            trust_proxy: false,
            trusted_proxies: vec![],
            cookie_secure: false,
            admin_session_expiry_secs: 8 * 3600,
            temp_audio_dir: "./temp_audio".to_string(),
            enable_hsts: false,
            max_playback_queue_items: DEFAULT_MAX_PLAYBACK_QUEUE_ITEMS,
            model_download_connect_timeout_secs: DEFAULT_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS,
            model_download_timeout_secs: DEFAULT_MODEL_DOWNLOAD_TIMEOUT_SECS,
        }
    }

    /// Build a config from an explicit variable map instead of the process
    /// environment so parsing tests are deterministic under parallel runs.
    fn config_from(pairs: &[(&str, &str)]) -> Result<AppConfig, ConfigError> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        AppConfig::from_env_with(|name| map.get(name).cloned())
    }

    #[test]
    fn valid_config_passes() {
        assert!(valid_config().validate().is_ok());
    }

    #[test]
    fn missing_admin_password_fails() {
        let mut config = valid_config();
        config.admin_pw.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn default_and_short_passwords_fail() {
        for pw in ["", "1234", "password", "admin", "short-pass", "changeme"] {
            let mut config = valid_config();
            config.admin_pw = pw.to_string();
            assert!(config.validate().is_err(), "password {pw:?} was accepted");
        }
    }

    #[test]
    fn placeholder_passwords_fail_despite_length() {
        for pw in [
            "change-me-to-a-strong-password",
            "CHANGE-ME-TO-A-STRONG-PASSWORD",
            "example-password",
            "example_password",
            "your-secure-password",
            "your_secure_password",
            "your_strong_password_12_plus_chars",
            "default-password",
        ] {
            let mut config = valid_config();
            config.admin_pw = pw.to_string();
            assert!(config.validate().is_err(), "password {pw:?} was accepted");
        }
    }

    #[test]
    fn password_length_counts_unicode_characters() {
        // 11 chars (multibyte) must fail the 12-character policy ...
        let mut config = valid_config();
        config.admin_pw = "pässwörd-ün".to_string();
        assert_eq!(config.admin_pw.chars().count(), 11);
        assert!(config.validate().is_err());
        // ... while 12 Unicode chars pass.
        let mut config = valid_config();
        config.admin_pw = "pässwörd-üni".to_string();
        assert_eq!(config.admin_pw.chars().count(), 12);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn model_revision_must_be_immutable_sha() {
        for rev in [
            "",
            "main",
            "master",
            "latest",
            "refs/heads/main",
            "v3",
            "abc123",
            "3cadd1ee6394adea1bd021217a0e650ede09a32",
            "3cadd1ee6394adea1bd021217a0e650ede09a323f",
            "zcadd1ee6394adea1bd021217a0e650ede09a323",
        ] {
            let mut config = valid_config();
            config.model_revision = rev.to_string();
            assert!(config.validate().is_err(), "revision {rev:?} was accepted");
        }
        let mut config = valid_config();
        config.model_revision = DEFAULT_MODEL_REVISION.to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn custom_revision_requires_custom_manifest() {
        let mut config = valid_config();
        config.model_revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        assert!(config.validate().is_err());
        config.model_hashes_path = Some("/tmp/custom-manifest.json".to_string());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn inference_steps_are_bounded() {
        for steps in [0, MAX_INFERENCE_STEPS + 1, 999_999] {
            let mut config = valid_config();
            config.inference_steps = steps;
            assert!(config.validate().is_err(), "steps={steps} was accepted");
        }
        let mut config = valid_config();
        config.inference_steps = MAX_INFERENCE_STEPS;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn zero_limits_fail() {
        let mut config = valid_config();
        config.max_text_length = 0;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.max_concurrent_inference = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn trust_proxy_requires_trusted_proxies() {
        let mut config = valid_config();
        config.trust_proxy = true;
        assert!(config.validate().is_err());
        config.trusted_proxies = vec!["10.0.0.1".to_string()];
        assert!(config.validate().is_ok());
    }

    #[test]
    fn malformed_trusted_proxy_entries_fail() {
        let mut config = valid_config();
        config.trusted_proxies = vec!["not-an-ip".to_string()];
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.trusted_proxies = vec!["10.0.0.0/33".to_string()];
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.trusted_proxies = vec!["10.0.0.0/24".to_string(), "::1".to_string()];
        assert!(config.validate().is_ok());
    }

    #[cfg(feature = "playback")]
    #[test]
    fn playback_requires_allowed_audio_dir() {
        let mut config = valid_config();
        config.allowed_audio_dir = None;
        assert!(config.validate().is_err());
    }

    #[test]
    fn malformed_integers_fail_parsing_instead_of_defaulting() {
        for (name, bad) in [
            ("PORT", "banana"),
            ("PORT", ""),
            ("PORT", "3.5"),
            ("INFERENCE_STEPS", "banana"),
            ("MAX_TEXT_LENGTH", "ten-thousand"),
            ("REQUEST_TIMEOUT_SECS", "banana"),
            ("MAX_CONCURRENT_INFERENCE", "-1"),
            ("MAX_PENDING_INFERENCE", "lots"),
            ("MAX_CHUNK_CHARS", ""),
            ("TTS_RATE_LIMIT_REQUESTS", "unlimited"),
            ("TTS_RATE_LIMIT_WINDOW_SECS", "minute"),
            ("TTS_MAX_BODY_BYTES", "64k"),
            ("OPENAI_MAX_BODY_BYTES", "64k"),
            ("QUEUE_MAX_BODY_BYTES", "16k"),
            ("ADMIN_MAX_BODY_BYTES", "16k"),
            ("ADMIN_SESSION_EXPIRY_SECS", "eight-hours"),
            ("MAX_PLAYBACK_QUEUE_ITEMS", "many"),
            ("MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS", "soon"),
            ("MODEL_DOWNLOAD_TIMEOUT_SECS", "eventually"),
        ] {
            let err = match config_from(&[(name, bad)]) {
                Ok(_) => panic!("{name}={bad:?} should fail strict parsing"),
                Err(err) => err,
            };
            assert_eq!(err.variable, name);
        }
    }

    #[test]
    fn malformed_booleans_fail_parsing_instead_of_defaulting() {
        for name in [
            "ENABLE_SAMPLE_TOKEN",
            "SONICBOOM_AUTH_REQUIRED",
            "TRUST_PROXY",
            "COOKIE_SECURE",
            "ENABLE_HSTS",
            "LOG_TO_FILE",
            "LOG_TO_STDOUT",
        ] {
            for bad in ["treu", "yes", "no", "banana", "", "2", "on"] {
                let err = match config_from(&[(name, bad)]) {
                    Ok(_) => panic!("{name}={bad:?} should fail strict boolean parsing"),
                    Err(err) => err,
                };
                assert_eq!(err.variable, name);
            }
        }
    }

    #[test]
    fn accepted_boolean_spellings_parse() {
        for (raw, expected) in [
            ("true", true),
            ("True", true),
            ("TRUE", true),
            ("tRuE", true),
            ("1", true),
            ("false", false),
            ("False", false),
            ("FALSE", false),
            ("0", false),
        ] {
            let config = config_from(&[("COOKIE_SECURE", raw)]).expect("valid bool");
            assert_eq!(config.cookie_secure, expected, "input {raw:?}");
        }
        // Absence still uses defaults.
        let config = config_from(&[]).expect("empty env uses defaults");
        assert!(!config.cookie_secure);
        assert!(config.auth_required);
        assert_eq!(config.request_timeout_secs, 120);
        assert_eq!(
            config.max_playback_queue_items,
            DEFAULT_MAX_PLAYBACK_QUEUE_ITEMS
        );
    }

    #[test]
    fn auth_required_defaults_to_true_and_parses_strictly() {
        assert!(config_from(&[]).unwrap().auth_required);
        assert!(
            !config_from(&[("SONICBOOM_AUTH_REQUIRED", "0")])
                .unwrap()
                .auth_required
        );
        assert!(
            !config_from(&[("SONICBOOM_AUTH_REQUIRED", "false")])
                .unwrap()
                .auth_required
        );
        assert!(
            config_from(&[("SONICBOOM_AUTH_REQUIRED", "1")])
                .unwrap()
                .auth_required
        );
        assert!(config_from(&[("SONICBOOM_AUTH_REQUIRED", "yes")]).is_err());
    }

    #[test]
    fn resource_limits_have_upper_bounds() {
        let mut config = valid_config();
        config.max_concurrent_inference = MAX_CONCURRENT_INFERENCE_LIMIT + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.max_pending_inference = MAX_PENDING_INFERENCE_LIMIT + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.max_text_length = MAX_TEXT_LENGTH_LIMIT + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.max_chunk_chars = MAX_CHUNK_CHARS_LIMIT + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.request_timeout_secs = MAX_REQUEST_TIMEOUT_SECS + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.request_timeout_secs = 0;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.tts_rate_limit_window_secs = MAX_RATE_LIMIT_WINDOW_SECS + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.admin_session_expiry_secs = MIN_ADMIN_SESSION_EXPIRY_SECS - 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.admin_session_expiry_secs = MAX_ADMIN_SESSION_EXPIRY_SECS + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.tts_max_body_bytes = MIN_BODY_BYTES - 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.openai_max_body_bytes = MAX_BODY_BYTES + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.max_playback_queue_items = 0;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.max_playback_queue_items = MAX_PLAYBACK_QUEUE_ITEMS_LIMIT + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.model_download_connect_timeout_secs = MAX_DOWNLOAD_CONNECT_TIMEOUT_SECS + 1;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.model_download_timeout_secs = MAX_DOWNLOAD_TIMEOUT_SECS + 1;
        assert!(config.validate().is_err());
        // Boundary values themselves are accepted.
        let mut config = valid_config();
        config.max_concurrent_inference = MAX_CONCURRENT_INFERENCE_LIMIT;
        config.max_pending_inference = MAX_PENDING_INFERENCE_LIMIT;
        config.max_playback_queue_items = MAX_PLAYBACK_QUEUE_ITEMS_LIMIT;
        assert!(config.validate().is_ok());
    }
}
