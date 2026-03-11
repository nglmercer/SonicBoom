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
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            admin_id: env::var("SONICBOOM_ADMIN_ID").unwrap_or_else(|_| "admin".to_string()),
            admin_pw: env::var("SONICBOOM_ADMIN_PW").unwrap_or_else(|_| "1234".to_string()),
            enable_sample_token: env::var("ENABLE_SAMPLE_TOKEN")
                .map(|v| v == "1")
                .unwrap_or(false),
            token_store_path: env::var("TOKEN_STORE_PATH")
                .unwrap_or_else(|_| "./tokens.json".to_string()),
            model_cache_dir: env::var("MODEL_CACHE_DIR")
                .unwrap_or_else(|_| "./models".to_string()),
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
                .map(|v| v == "1")
                .unwrap_or(true),
            log_to_stdout: env::var("LOG_TO_STDOUT")
                .map(|v| v == "1")
                .unwrap_or(true),
            // Authentication settings
            // Set to false to allow API access without authentication
            // Can be toggled via SONICBOOM_AUTH_REQUIRED=0 or SONICBOOM_AUTH_REQUIRED=false
            auth_required: env::var("SONICBOOM_AUTH_REQUIRED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
        }
    }
}
