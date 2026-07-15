use std::env;

pub struct AppConfig {
    pub admin_id: String,
    pub admin_pw: String,
    pub enable_sample_token: bool,
    pub token_store_path: String,
    pub model_cache_dir: String,
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
    // Security: allowed audio directory for queue (prevents path traversal)
    pub allowed_audio_dir: Option<String>,
    // Maximum text length for TTS requests
    pub max_text_length: usize,
    // Request timeout in seconds
    pub request_timeout_secs: u64,
}

impl AppConfig {
    pub fn from_env() -> Self {
        // Default password is "1234" per specification. Set SONICBOOM_ADMIN_PW
        // to override (recommended for production deployments).
        let admin_pw =
            env::var("SONICBOOM_ADMIN_PW").unwrap_or_else(|_| "1234".to_string());

        Self {
            admin_id: env::var("SONICBOOM_ADMIN_ID").unwrap_or_else(|_| "admin".to_string()),
            admin_pw,
            enable_sample_token: env::var("ENABLE_SAMPLE_TOKEN")
                .map(|v| v == "1")
                .unwrap_or(false),
            token_store_path: env::var("TOKEN_STORE_PATH")
                .unwrap_or_else(|_| "./tokens.json".to_string()),
            model_cache_dir: env::var("MODEL_CACHE_DIR").unwrap_or_else(|_| "./models".to_string()),
            hf_token: env::var("HF_TOKEN").ok(),
            inference_steps: env::var("INFERENCE_STEPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            port: env::var("PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3000),
            // Logging settings
            log_dir: env::var("LOG_DIR").unwrap_or_else(|_| "./logs".to_string()),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            log_to_file: env::var("LOG_TO_FILE")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),
            log_to_stdout: env::var("LOG_TO_STDOUT")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),
            // Authentication settings
            // Set to false to allow API access without authentication
            // Can be toggled via SONICBOOM_AUTH_REQUIRED=0 or SONICBOOM_AUTH_REQUIRED=false
            auth_required: env::var("SONICBOOM_AUTH_REQUIRED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            // Security: allowed audio directory for queue (prevents path traversal)
            allowed_audio_dir: env::var("ALLOWED_AUDIO_DIR").ok(),
            // Maximum text length for TTS requests (default: 10000 chars)
            max_text_length: env::var("MAX_TEXT_LENGTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10_000),
            // Request timeout in seconds (default: 120s)
            request_timeout_secs: env::var("REQUEST_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(120),
        }
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), String> {
        if self.inference_steps == 0 {
            return Err("INFERENCE_STEPS must be greater than 0".to_string());
        }
        if self.port == 0 {
            return Err("PORT must be greater than 0".to_string());
        }
        if self.max_text_length == 0 {
            return Err("MAX_TEXT_LENGTH must be greater than 0".to_string());
        }
        Ok(())
    }
}
