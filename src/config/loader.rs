//! Configuration loading: `config.toml` + environment overrides
//! with per-value source tracking.
//!
//! Precedence (highest first):
//! 1. runtime override (applied via [`ConfigManager::update`])
//! 2. environment variables (Docker/systemd/CI compatibility)
//! 3. `config.toml`
//! 4. compiled defaults

use std::collections::HashMap;
use std::env;
use std::fmt;
use std::path::Path;

use super::effective::{ConfigSource, EffectiveConfig};
use super::model::{AppConfig, AuthMode, ConfigError};

/// Load the configuration from a `config.toml` file, then apply
/// environment-variable overrides.
///
/// A missing file yields the default configuration (all values
/// sourced from `Default`); a present-but-malformed file or a
/// malformed environment value is a hard error.
pub fn load_from_file(path: &Path) -> Result<EffectiveConfig, ConfigError> {
    let mut effective = if path.exists() {
        let text = std::fs::read_to_string(path).map_err(|e| {
            ConfigError::invalid(
                &path.display().to_string(),
                format!("could not read config file: {e}"),
            )
        })?;
        from_toml(&text)?
    } else {
        let mut sources = HashMap::new();
        record_default_sources(&mut sources);
        EffectiveConfig {
            config: AppConfig::default(),
            sources,
        }
    };
    apply_environment(&mut effective.config, &mut effective.sources)?;
    Ok(effective)
}

/// Parse persisted values and track only explicitly supplied TOML fields.
pub fn from_toml(text: &str) -> Result<EffectiveConfig, ConfigError> {
    let config = parse_toml(text)?;
    let document: toml::Value =
        toml::from_str(text).map_err(|e| ConfigError::invalid("config.toml", e.to_string()))?;
    let mut sources = HashMap::new();
    record_default_sources(&mut sources);
    for path in ALL_CONFIG_PATHS {
        let mut value = Some(&document);
        for part in path.split('.') {
            value = value.and_then(|v| v.get(part));
        }
        if value.is_some() {
            sources.insert((*path).to_string(), ConfigSource::Toml);
        }
    }
    Ok(EffectiveConfig { config, sources })
}

/// Parse a `config.toml` document. Unknown keys are rejected so
/// typos fail loudly instead of silently doing nothing.
pub fn parse_toml(text: &str) -> Result<AppConfig, ConfigError> {
    let parsed: toml::Value = toml::from_str(text).map_err(|e| {
        ConfigError::invalid(
            "config.toml",
            format!("could not parse TOML: {}", e.message()),
        )
    })?;
    // Reject unknown top-level keys with a helpful message.
    if let toml::Value::Table(table) = &parsed {
        for key in table.keys() {
            let known = matches!(
                key.as_str(),
                "version"
                    | "setup_complete"
                    | "revision"
                    | "server"
                    | "admin"
                    | "audio"
                    | "model"
                    | "tts"
                    | "inference"
                    | "rate_limit"
                    | "logging"
                    | "paths"
                    | "security"
            );
            if !known {
                return Err(ConfigError::invalid(
                    "config.toml",
                    format!("unknown top-level key '{key}'"),
                ));
            }
        }
    }
    let config: AppConfig = toml::from_str(text).map_err(|e| {
        ConfigError::invalid(
            "config.toml",
            format!("invalid configuration: {}", e.message()),
        )
    })?;
    Ok(config)
}

fn record_default_sources(sources: &mut HashMap<String, ConfigSource>) {
    for path in ALL_CONFIG_PATHS {
        sources.insert(path.to_string(), ConfigSource::Default);
    }
}

/// Every dotted configuration path known to the loader.
pub const ALL_CONFIG_PATHS: &[&str] = &[
    "server.bind",
    "server.port",
    "server.auth_mode",
    "server.request_timeout_secs",
    "admin.username",
    "admin.session_expiry_secs",
    "admin.enable_sample_token",
    "audio.output_device",
    "audio.volume",
    "audio.max_playback_queue_items",
    "model.cache_dir",
    "model.revision",
    "model.hashes_path",
    "model.inference_steps",
    "model.download_connect_timeout_secs",
    "model.download_timeout_secs",
    "tts.max_text_length",
    "tts.max_chunk_chars",
    "tts.max_body_bytes",
    "tts.openai_max_body_bytes",
    "tts.queue_max_body_bytes",
    "tts.admin_max_body_bytes",
    "inference.max_concurrent",
    "inference.max_pending",
    "rate_limit.requests",
    "rate_limit.window_secs",
    "rate_limit.burst",
    "logging.level",
    "logging.filter",
    "logging.to_file",
    "logging.to_stdout",
    "paths.logs",
    "paths.audio",
    "paths.temp_audio",
    "paths.token_store",
    "security.trust_proxy",
    "security.trusted_proxies",
    "security.cookie_secure",
    "security.enable_hsts",
    "security.allow_insecure_remote",
];

// ---------------------------------------------------------------------------
// Environment overrides
// ---------------------------------------------------------------------------

/// Accepted boolean spellings: `true`/`false` (any ASCII case) or
/// `1`/`0`. Anything else — `yes`, `no`, typos, empty strings — is
/// rejected so operator mistakes fail loudly instead of silently
/// becoming `false`.
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

fn env_string(
    get: &impl Fn(&str) -> Option<String>,
    name: &str,
    default: &str,
) -> Result<String, ConfigError> {
    match get(name) {
        None => Ok(default.to_string()),
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Err(ConfigError::invalid(
                    name,
                    "value is present but empty; unset it to use the default",
                ));
            }
            Ok(trimmed.to_string())
        }
    }
}

/// Apply environment-variable overrides to a configuration,
/// recording the source of every overridden value.
pub fn apply_environment(
    config: &mut AppConfig,
    sources: &mut HashMap<String, ConfigSource>,
) -> Result<(), ConfigError> {
    let get = |name: &str| env::var(name).ok();
    apply_environment_with(&get, config, sources)
}

/// Test-friendly variant that takes an explicit variable map.
pub fn apply_environment_with(
    get: &impl Fn(&str) -> Option<String>,
    config: &mut AppConfig,
    sources: &mut HashMap<String, ConfigSource>,
) -> Result<(), ConfigError> {
    let override_value = |sources: &mut HashMap<String, ConfigSource>, path: &str| {
        sources.insert(path.to_string(), ConfigSource::Environment);
    };

    // --- server ---
    if let Ok(port) = env_parse(&get, "PORT", config.server.port) {
        if get("PORT").is_some() {
            config.server.port = port;
            override_value(sources, "server.port");
        }
    }
    if get("PORT").is_some() {
        // re-run to surface parse errors even when the value matches
        config.server.port = env_parse(&get, "PORT", config.server.port)?;
    }
    config.server.bind = match get("BIND") {
        Some(raw) => {
            let trimmed = raw.trim();
            let addr: std::net::IpAddr = trimmed.parse().map_err(|e| {
                ConfigError::invalid("BIND", format!("could not parse '{trimmed}': {e}"))
            })?;
            override_value(sources, "server.bind");
            addr
        }
        None => config.server.bind,
    };
    config.server.request_timeout_secs = {
        let v = env_u64(
            &get,
            "REQUEST_TIMEOUT_SECS",
            config.server.request_timeout_secs,
        )?;
        if get("REQUEST_TIMEOUT_SECS").is_some() {
            override_value(sources, "server.request_timeout_secs");
        }
        v
    };
    // SONICBOOM_AUTH_REQUIRED=true -> token, false -> none (legacy mapping).
    if get("SONICBOOM_AUTH_REQUIRED").is_some() {
        let required = env_bool(&get, "SONICBOOM_AUTH_REQUIRED", true)?;
        config.server.auth_mode = if required {
            AuthMode::Token
        } else {
            AuthMode::None
        };
        override_value(sources, "server.auth_mode");
    }

    // --- admin ---
    config.admin.username = {
        let v = env_string(&get, "SONICBOOM_ADMIN_ID", &config.admin.username)?;
        if get("SONICBOOM_ADMIN_ID").is_some() {
            override_value(sources, "admin.username");
        }
        v
    };
    config.admin.session_expiry_secs = {
        let v = env_parse(
            &get,
            "ADMIN_SESSION_EXPIRY_SECS",
            config.admin.session_expiry_secs,
        )?;
        if get("ADMIN_SESSION_EXPIRY_SECS").is_some() {
            override_value(sources, "admin.session_expiry_secs");
        }
        v
    };
    config.admin.enable_sample_token = {
        let v = env_bool(
            &get,
            "ENABLE_SAMPLE_TOKEN",
            config.admin.enable_sample_token,
        )?;
        if get("ENABLE_SAMPLE_TOKEN").is_some() {
            override_value(sources, "admin.enable_sample_token");
        }
        v
    };

    // --- audio ---
    config.audio.output_device = {
        let raw = get("AUDIO_OUTPUT_DEVICE");
        let present = raw.is_some();
        let v = raw
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| config.audio.output_device.clone());
        if present {
            override_value(sources, "audio.output_device");
        }
        v
    };
    config.audio.max_playback_queue_items = {
        let v = env_usize(
            &get,
            "MAX_PLAYBACK_QUEUE_ITEMS",
            config.audio.max_playback_queue_items,
        )?;
        if get("MAX_PLAYBACK_QUEUE_ITEMS").is_some() {
            override_value(sources, "audio.max_playback_queue_items");
        }
        v
    };

    // --- model ---
    config.model.cache_dir = {
        let v = env_string(
            &get,
            "MODEL_CACHE_DIR",
            &config.model.cache_dir.to_string_lossy(),
        )?;
        if get("MODEL_CACHE_DIR").is_some() {
            override_value(sources, "model.cache_dir");
        }
        std::path::PathBuf::from(v)
    };
    config.model.revision = {
        let raw = get("MODEL_REVISION");
        let present = raw.is_some();
        let v = raw
            .map(|v| v.trim().to_lowercase())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| config.model.revision.clone());
        if present {
            override_value(sources, "model.revision");
        }
        v
    };
    config.model.hashes_path = match get("MODEL_SHA256_JSON_PATH") {
        Some(raw) => {
            let trimmed = raw.trim();
            let path = if trimmed.is_empty() {
                return Err(ConfigError::invalid(
                    "MODEL_SHA256_JSON_PATH",
                    "value is present but empty; unset it to disable",
                ));
            } else {
                Some(std::path::PathBuf::from(trimmed))
            };
            override_value(sources, "model.hashes_path");
            path
        }
        None => config.model.hashes_path.clone(),
    };
    config.model.inference_steps = {
        let v = env_usize(&get, "INFERENCE_STEPS", config.model.inference_steps)?;
        if get("INFERENCE_STEPS").is_some() {
            override_value(sources, "model.inference_steps");
        }
        v
    };
    config.model.download_connect_timeout_secs = {
        let v = env_u64(
            &get,
            "MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS",
            config.model.download_connect_timeout_secs,
        )?;
        if get("MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS").is_some() {
            override_value(sources, "model.download_connect_timeout_secs");
        }
        v
    };
    config.model.download_timeout_secs = {
        let v = env_u64(
            &get,
            "MODEL_DOWNLOAD_TIMEOUT_SECS",
            config.model.download_timeout_secs,
        )?;
        if get("MODEL_DOWNLOAD_TIMEOUT_SECS").is_some() {
            override_value(sources, "model.download_timeout_secs");
        }
        v
    };

    // --- tts ---
    config.tts.max_text_length = {
        let v = env_usize(&get, "MAX_TEXT_LENGTH", config.tts.max_text_length)?;
        if get("MAX_TEXT_LENGTH").is_some() {
            override_value(sources, "tts.max_text_length");
        }
        v
    };
    config.tts.max_chunk_chars = {
        let v = env_usize(&get, "MAX_CHUNK_CHARS", config.tts.max_chunk_chars)?;
        if get("MAX_CHUNK_CHARS").is_some() {
            override_value(sources, "tts.max_chunk_chars");
        }
        v
    };
    config.tts.max_body_bytes = {
        let v = env_usize(&get, "TTS_MAX_BODY_BYTES", config.tts.max_body_bytes)?;
        if get("TTS_MAX_BODY_BYTES").is_some() {
            override_value(sources, "tts.max_body_bytes");
        }
        v
    };
    config.tts.openai_max_body_bytes = {
        let v = env_usize(
            &get,
            "OPENAI_MAX_BODY_BYTES",
            config.tts.openai_max_body_bytes,
        )?;
        if get("OPENAI_MAX_BODY_BYTES").is_some() {
            override_value(sources, "tts.openai_max_body_bytes");
        }
        v
    };
    config.tts.queue_max_body_bytes = {
        let v = env_usize(
            &get,
            "QUEUE_MAX_BODY_BYTES",
            config.tts.queue_max_body_bytes,
        )?;
        if get("QUEUE_MAX_BODY_BYTES").is_some() {
            override_value(sources, "tts.queue_max_body_bytes");
        }
        v
    };
    config.tts.admin_max_body_bytes = {
        let v = env_usize(
            &get,
            "ADMIN_MAX_BODY_BYTES",
            config.tts.admin_max_body_bytes,
        )?;
        if get("ADMIN_MAX_BODY_BYTES").is_some() {
            override_value(sources, "tts.admin_max_body_bytes");
        }
        v
    };

    // --- inference ---
    config.inference.max_concurrent = {
        let v = env_usize(
            &get,
            "MAX_CONCURRENT_INFERENCE",
            config.inference.max_concurrent,
        )?;
        if get("MAX_CONCURRENT_INFERENCE").is_some() {
            override_value(sources, "inference.max_concurrent");
        }
        v
    };
    config.inference.max_pending = {
        let v = env_usize(&get, "MAX_PENDING_INFERENCE", config.inference.max_pending)?;
        if get("MAX_PENDING_INFERENCE").is_some() {
            override_value(sources, "inference.max_pending");
        }
        v
    };

    // --- rate_limit ---
    let requests = env_parse(&get, "TTS_RATE_LIMIT_REQUESTS", config.rate_limit.requests)?;
    if get("TTS_RATE_LIMIT_REQUESTS").is_some() {
        config.rate_limit.requests = requests;
        override_value(sources, "rate_limit.requests");
    }
    let window = env_u64(
        &get,
        "TTS_RATE_LIMIT_WINDOW_SECS",
        config.rate_limit.window_secs,
    )?;
    if get("TTS_RATE_LIMIT_WINDOW_SECS").is_some() {
        config.rate_limit.window_secs = window;
        override_value(sources, "rate_limit.window_secs");
    }
    // Burst defaults to the sustained budget when omitted.
    let default_burst = if config.rate_limit.requests == 0 {
        super::model::DEFAULT_TTS_RATE_LIMIT_REQUESTS
    } else {
        config.rate_limit.requests
    };
    let burst = env_parse(&get, "TTS_RATE_LIMIT_BURST", default_burst)?;
    if get("TTS_RATE_LIMIT_BURST").is_some() {
        config.rate_limit.burst = burst;
        override_value(sources, "rate_limit.burst");
    }

    // --- logging ---
    if let Some(filter) = get("RUST_LOG") {
        config.logging.filter = Some(filter);
        override_value(sources, "logging.filter");
    }
    config.logging.level = {
        let v = env_string(&get, "LOG_LEVEL", &config.logging.level)?;
        if get("LOG_LEVEL").is_some() {
            override_value(sources, "logging.level");
        }
        v
    };
    config.logging.to_file = {
        let v = env_bool(&get, "LOG_TO_FILE", config.logging.to_file)?;
        if get("LOG_TO_FILE").is_some() {
            override_value(sources, "logging.to_file");
        }
        v
    };
    config.logging.to_stdout = {
        let v = env_bool(&get, "LOG_TO_STDOUT", config.logging.to_stdout)?;
        if get("LOG_TO_STDOUT").is_some() {
            override_value(sources, "logging.to_stdout");
        }
        v
    };

    // --- paths ---
    config.paths.logs = {
        let v = env_string(&get, "LOG_DIR", &config.paths.logs)?;
        if get("LOG_DIR").is_some() {
            override_value(sources, "paths.logs");
        }
        v
    };
    config.paths.audio = match get("ALLOWED_AUDIO_DIR") {
        Some(raw) => {
            let trimmed = raw.trim();
            let dir = if trimmed.is_empty() {
                return Err(ConfigError::invalid(
                    "ALLOWED_AUDIO_DIR",
                    "value is present but empty; unset it to disable",
                ));
            } else {
                Some(trimmed.to_string())
            };
            override_value(sources, "paths.audio");
            dir
        }
        None => config.paths.audio.clone(),
    };
    config.paths.temp_audio = {
        let v = env_string(&get, "TEMP_AUDIO_DIR", &config.paths.temp_audio)?;
        if get("TEMP_AUDIO_DIR").is_some() {
            override_value(sources, "paths.temp_audio");
        }
        v
    };
    config.paths.token_store = {
        let v = env_string(&get, "TOKEN_STORE_PATH", &config.paths.token_store)?;
        if get("TOKEN_STORE_PATH").is_some() {
            override_value(sources, "paths.token_store");
        }
        v
    };

    // --- security ---
    config.security.trust_proxy = {
        let v = env_bool(&get, "TRUST_PROXY", config.security.trust_proxy)?;
        if get("TRUST_PROXY").is_some() {
            override_value(sources, "security.trust_proxy");
        }
        v
    };
    config.security.trusted_proxies = match get("TRUSTED_PROXIES") {
        Some(raw) => {
            let list: Vec<String> = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            override_value(sources, "security.trusted_proxies");
            list
        }
        None => config.security.trusted_proxies.clone(),
    };
    config.security.cookie_secure = {
        let v = env_bool(&get, "COOKIE_SECURE", config.security.cookie_secure)?;
        if get("COOKIE_SECURE").is_some() {
            override_value(sources, "security.cookie_secure");
        }
        v
    };
    config.security.enable_hsts = {
        let v = env_bool(&get, "ENABLE_HSTS", config.security.enable_hsts)?;
        if get("ENABLE_HSTS").is_some() {
            override_value(sources, "security.enable_hsts");
        }
        v
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn config_from(
        pairs: &[(&str, &str)],
    ) -> Result<(AppConfig, HashMap<String, ConfigSource>), ConfigError> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let mut config = AppConfig::default();
        let mut sources = HashMap::new();
        record_default_sources(&mut sources);
        apply_environment_with(&|name| map.get(name).cloned(), &mut config, &mut sources)?;
        Ok((config, sources))
    }

    #[test]
    fn partial_toml_tracks_defaults_and_rejects_nested_typos() {
        let effective = from_toml("[server]\nport = 19000\n").unwrap();
        assert_eq!(effective.source_of("server.port"), Some(ConfigSource::Toml));
        assert_eq!(
            effective.source_of("logging.level"),
            Some(ConfigSource::Default)
        );
        assert!(parse_toml("[server]\nprort = 19000\n").is_err());
        assert!(parse_toml("[admin]\npassword = 'secret'\n").is_err());
    }

    #[test]
    fn empty_environment_uses_defaults() {
        let (config, sources) = config_from(&[]).unwrap();
        assert_eq!(config.server.port, 17842);
        assert_eq!(config.server.auth_mode, AuthMode::Token);
        assert_eq!(config.model.inference_steps, 5);
        assert_eq!(sources["server.port"], ConfigSource::Default);
    }

    #[test]
    fn environment_overrides_are_tracked() {
        let (config, sources) = config_from(&[
            ("PORT", "19000"),
            ("LOG_LEVEL", "debug"),
            ("AUDIO_OUTPUT_DEVICE", "CABLE Input"),
        ])
        .unwrap();
        assert_eq!(config.server.port, 19000);
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.audio.output_device, "CABLE Input");
        assert_eq!(sources["server.port"], ConfigSource::Environment);
        assert_eq!(sources["logging.level"], ConfigSource::Environment);
        assert_eq!(sources["audio.output_device"], ConfigSource::Environment);
        assert_eq!(sources["server.bind"], ConfigSource::Default);
    }

    #[test]
    fn legacy_auth_required_maps_to_auth_mode() {
        let (config, _) = config_from(&[("SONICBOOM_AUTH_REQUIRED", "true")]).unwrap();
        assert_eq!(config.server.auth_mode, AuthMode::Token);
        let (config, _) = config_from(&[("SONICBOOM_AUTH_REQUIRED", "false")]).unwrap();
        assert_eq!(config.server.auth_mode, AuthMode::None);
        assert!(config_from(&[("SONICBOOM_AUTH_REQUIRED", "yes")]).is_err());
    }

    #[test]
    fn malformed_integers_fail_instead_of_defaulting() {
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
            ("TTS_RATE_LIMIT_BURST", "many"),
            ("TTS_MAX_BODY_BYTES", "64k"),
            ("OPENAI_MAX_BODY_BYTES", "64k"),
            ("QUEUE_MAX_BODY_BYTES", "16k"),
            ("ADMIN_MAX_BODY_BYTES", "16k"),
            ("ADMIN_SESSION_EXPIRY_SECS", "eight-hours"),
            ("MAX_PLAYBACK_QUEUE_ITEMS", "many"),
            ("MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS", "soon"),
            ("MODEL_DOWNLOAD_TIMEOUT_SECS", "eventually"),
        ] {
            let err = config_from(&[(name, bad)]).expect_err("should fail strict parsing");
            assert_eq!(err.variable, name, "input {name}={bad:?}");
        }
    }

    #[test]
    fn malformed_booleans_fail_instead_of_defaulting() {
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
                let err =
                    config_from(&[(name, bad)]).expect_err("should fail strict boolean parsing");
                assert_eq!(err.variable, name, "input {name}={bad:?}");
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
            let (config, _) = config_from(&[("COOKIE_SECURE", raw)]).unwrap();
            assert_eq!(config.security.cookie_secure, expected, "input {raw:?}");
        }
    }

    #[test]
    fn audio_output_device_handling() {
        let (config, _) = config_from(&[]).unwrap();
        assert_eq!(config.audio.output_device, "default");
        let (config, _) = config_from(&[("AUDIO_OUTPUT_DEVICE", "   ")]).unwrap();
        assert_eq!(config.audio.output_device, "default");
        let (config, _) = config_from(&[("AUDIO_OUTPUT_DEVICE", "  Speakers (USB)  ")]).unwrap();
        assert_eq!(config.audio.output_device, "Speakers (USB)");
    }

    #[test]
    fn parse_toml_rejects_unknown_keys() {
        let err = parse_toml("[nonsense]\nfoo = 1\n").expect_err("unknown key");
        assert!(err.message.contains("unknown top-level key"));
    }

    #[test]
    fn parse_toml_rejects_bad_types() {
        let err = parse_toml("[server]\nport = \"hello\"\n").expect_err("bad type");
        assert!(err.message.contains("invalid configuration"));
    }

    #[test]
    fn parse_toml_partial_file_fills_defaults() {
        let config =
            parse_toml("[server]\nport = 19000\n\n[audio]\noutput_device = \"CABLE Input\"\n")
                .unwrap();
        assert_eq!(config.server.port, 19000);
        assert_eq!(config.audio.output_device, "CABLE Input");
        // Unset sections keep compiled defaults.
        assert_eq!(config.model.inference_steps, 5);
        assert_eq!(config.server.bind.to_string(), "127.0.0.1");
    }
}
