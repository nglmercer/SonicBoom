use std::env;

/// Minimum admin password length. Length-only check: no composition rules.
pub const MIN_ADMIN_PASSWORD_LEN: usize = 12;
/// Absolute maximum for inference steps (matches the reusable engine).
pub const MAX_INFERENCE_STEPS: usize = 50;
/// Default pinned HuggingFace revision for Supertonic-3 model downloads.
pub const DEFAULT_MODEL_REVISION: &str = "3cadd1ee6394adea1bd021217a0e650ede09a323";

/// Passwords that are never accepted, even if they met the length rule.
const REJECTED_PASSWORDS: &[&str] = &[
    "1234",
    "password",
    "admin",
    "changeme",
    "letmein",
    "qwerty",
    "sonicboom",
];

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
}

fn parse_bool(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name).map(|v| parse_bool(&v)).unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            admin_id: env::var("SONICBOOM_ADMIN_ID").unwrap_or_else(|_| "admin".to_string()),
            // No default password: missing/weak values fail validation.
            admin_pw: env::var("SONICBOOM_ADMIN_PW").unwrap_or_default(),
            enable_sample_token: env_bool("ENABLE_SAMPLE_TOKEN", false),
            token_store_path: env::var("TOKEN_STORE_PATH")
                .unwrap_or_else(|_| "./tokens.json".to_string()),
            model_cache_dir: env::var("MODEL_CACHE_DIR").unwrap_or_else(|_| "./models".to_string()),
            model_revision: env::var("MODEL_REVISION")
                .unwrap_or_else(|_| DEFAULT_MODEL_REVISION.to_string()),
            model_hashes_path: env::var("MODEL_SHA256_JSON_PATH").ok(),
            hf_token: env::var("HF_TOKEN").ok(),
            inference_steps: env_usize("INFERENCE_STEPS", 5),
            port: env::var("PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3000),
            // Logging settings
            log_dir: env::var("LOG_DIR").unwrap_or_else(|_| "./logs".to_string()),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            log_to_file: env_bool("LOG_TO_FILE", true),
            log_to_stdout: env_bool("LOG_TO_STDOUT", true),
            // Authentication settings
            // Set to false to allow API access without authentication
            auth_required: env::var("SONICBOOM_AUTH_REQUIRED")
                .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE" | "False"))
                .unwrap_or(true),
            // Security: allowed audio directory for queue (prevents path traversal)
            allowed_audio_dir: env::var("ALLOWED_AUDIO_DIR").ok(),
            // Maximum text length for TTS requests (default: 10000 chars)
            max_text_length: env_usize("MAX_TEXT_LENGTH", 10_000),
            // Request timeout in seconds (default: 120s)
            request_timeout_secs: env_u64("REQUEST_TIMEOUT_SECS", 120),
            max_concurrent_inference: env_usize("MAX_CONCURRENT_INFERENCE", 1),
            max_pending_inference: env_usize("MAX_PENDING_INFERENCE", 8),
            max_chunk_chars: env_usize("MAX_CHUNK_CHARS", 200),
            tts_rate_limit_requests: env::var("TTS_RATE_LIMIT_REQUESTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            tts_rate_limit_window_secs: env_u64("TTS_RATE_LIMIT_WINDOW_SECS", 60),
            tts_max_body_bytes: env_usize("TTS_MAX_BODY_BYTES", 65_536),
            openai_max_body_bytes: env_usize("OPENAI_MAX_BODY_BYTES", 65_536),
            queue_max_body_bytes: env_usize("QUEUE_MAX_BODY_BYTES", 16_384),
            admin_max_body_bytes: env_usize("ADMIN_MAX_BODY_BYTES", 16_384),
            trust_proxy: env_bool("TRUST_PROXY", false),
            trusted_proxies: env::var("TRUSTED_PROXIES")
                .map(|v| {
                    v.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            cookie_secure: env_bool("COOKIE_SECURE", false),
            admin_session_expiry_secs: env::var("ADMIN_SESSION_EXPIRY_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8 * 3600),
            temp_audio_dir: env::var("TEMP_AUDIO_DIR")
                .unwrap_or_else(|_| "./temp_audio".to_string()),
        }
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
        if self.max_text_length == 0 {
            return Err("MAX_TEXT_LENGTH must be greater than 0".to_string());
        }
        if self.max_chunk_chars == 0 {
            return Err("MAX_CHUNK_CHARS must be greater than 0".to_string());
        }
        if self.max_concurrent_inference == 0 {
            return Err("MAX_CONCURRENT_INFERENCE must be greater than 0".to_string());
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
                        "ALLOWED_AUDIO_DIR must be set when the playback/filesystem queue is enabled"
                            .to_string(),
                    );
                }
            }
        }
        if self.model_revision.trim().is_empty() {
            return Err("MODEL_REVISION must not be empty".to_string());
        }
        if self.tts_rate_limit_window_secs == 0 {
            return Err("TTS_RATE_LIMIT_WINDOW_SECS must be greater than 0".to_string());
        }
        if self.admin_session_expiry_secs <= 0 {
            return Err("ADMIN_SESSION_EXPIRY_SECS must be greater than 0".to_string());
        }
        for (name, value) in [
            ("TTS_MAX_BODY_BYTES", self.tts_max_body_bytes),
            ("OPENAI_MAX_BODY_BYTES", self.openai_max_body_bytes),
            ("QUEUE_MAX_BODY_BYTES", self.queue_max_body_bytes),
            ("ADMIN_MAX_BODY_BYTES", self.admin_max_body_bytes),
        ] {
            if value == 0 {
                return Err(format!("{name} must be greater than 0"));
            }
        }
        if self.trust_proxy && self.trusted_proxies.is_empty() {
            return Err(
                "TRUSTED_PROXIES must list at least one proxy address when TRUST_PROXY is enabled"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn validate_admin_password(&self) -> Result<(), String> {
        if self.admin_pw.is_empty() {
            return Err(
                "SONICBOOM_ADMIN_PW is not set. Refusing to start with no admin password. \
                 Set SONICBOOM_ADMIN_PW to a strong password (minimum 12 characters)."
                    .to_string(),
            );
        }
        if self.admin_pw.len() < MIN_ADMIN_PASSWORD_LEN {
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
            port: 3000,
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
        }
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

    #[cfg(feature = "playback")]
    #[test]
    fn playback_requires_allowed_audio_dir() {
        let mut config = valid_config();
        config.allowed_audio_dir = None;
        assert!(config.validate().is_err());
    }
}
