//! Configuration validation. Security-sensitive misconfiguration
//! fails closed: the server refuses to start (startup) or the
//! candidate is rejected (hot reload).

use super::model::{
    AppConfig, AuthMode, MAX_ADMIN_SESSION_EXPIRY_SECS, MAX_AUDIO_OUTPUT_DEVICE_LEN,
    MAX_BODY_BYTES, MAX_CHUNK_CHARS_LIMIT, MAX_CONCURRENT_INFERENCE_LIMIT,
    MAX_DOWNLOAD_CONNECT_TIMEOUT_SECS, MAX_DOWNLOAD_TIMEOUT_SECS, MAX_INFERENCE_STEPS,
    MAX_PENDING_INFERENCE_LIMIT, MAX_PLAYBACK_QUEUE_ITEMS_LIMIT, MAX_RATE_LIMIT_REQUESTS,
    MAX_RATE_LIMIT_WINDOW_SECS, MAX_REQUEST_TIMEOUT_SECS, MAX_TEXT_LENGTH_LIMIT,
    MIN_ADMIN_PASSWORD_LEN, MIN_ADMIN_SESSION_EXPIRY_SECS, MIN_BODY_BYTES, REJECTED_PASSWORDS,
};

/// Validate a candidate configuration. Every rule fails closed.
pub fn validate(config: &AppConfig) -> Result<(), String> {
    if config.version != super::model::CONFIG_VERSION {
        return Err(format!(
            "unsupported configuration version {}",
            config.version
        ));
    }
    if config.revision == 0 {
        return Err("revision must be positive".into());
    }
    validate_server(config)?;
    validate_admin(config)?;
    validate_audio(config)?;
    validate_model(config)?;
    validate_tts(config)?;
    validate_inference(config)?;
    validate_rate_limit(config)?;
    validate_logging(config)?;
    validate_paths(config)?;
    validate_security(config)?;
    validate_providers(config)?;
    validate_voices(config)?;
    Ok(())
}

fn validate_server(config: &AppConfig) -> Result<(), String> {
    let server = &config.server;
    if server.port == 0 {
        return Err("server.port must be greater than 0".to_string());
    }
    if !(1..=MAX_REQUEST_TIMEOUT_SECS).contains(&server.request_timeout_secs) {
        return Err(format!(
            "server.request_timeout_secs must be between 1 and {MAX_REQUEST_TIMEOUT_SECS}"
        ));
    }
    // Security combinations (spec §4/§55): fail closed.
    match server.auth_mode {
        AuthMode::Local if !server.bind.is_loopback() => {
            return Err(format!(
                "auth_mode = local requires a loopback bind address, got {}",
                server.bind
            ));
        }
        AuthMode::None if !server.bind.is_loopback() && !config.security.allow_insecure_remote => {
            return Err(
                "auth_mode = none may only bind non-loopback addresses when \
                 [security] allow_insecure_remote = true is set explicitly"
                    .to_string(),
            );
        }
        _ => {}
    }
    Ok(())
}

fn validate_admin(config: &AppConfig) -> Result<(), String> {
    let admin = &config.admin;
    if admin.username.trim().is_empty() {
        return Err("admin.username must not be empty".to_string());
    }
    if !(MIN_ADMIN_SESSION_EXPIRY_SECS..=MAX_ADMIN_SESSION_EXPIRY_SECS)
        .contains(&admin.session_expiry_secs)
    {
        return Err(format!(
            "admin.session_expiry_secs must be between {MIN_ADMIN_SESSION_EXPIRY_SECS} and {MAX_ADMIN_SESSION_EXPIRY_SECS}"
        ));
    }
    Ok(())
}

/// Validate an admin password against the documented policy.
/// The password itself is stored in the secret store, never in
/// `config.toml`; this runs whenever a password is set or loaded
/// from the environment/secret store.
pub fn validate_admin_password(password: &str) -> Result<(), String> {
    if password.is_empty() {
        return Err("No admin password configured. Set one with \
             `sonicboom admin password <value>` or the SONICBOOM_ADMIN_PW \
             environment variable (minimum 12 characters). There is \
             intentionally no default password (see SECURITY.md)."
            .to_string());
    }
    // Length is measured in Unicode characters, matching the documented policy.
    if password.chars().count() < MIN_ADMIN_PASSWORD_LEN {
        return Err(format!(
            "admin password must be at least {MIN_ADMIN_PASSWORD_LEN} characters"
        ));
    }
    if REJECTED_PASSWORDS
        .iter()
        .any(|bad| password.eq_ignore_ascii_case(bad))
    {
        return Err("admin password must not be a well-known default password".to_string());
    }
    Ok(())
}

fn validate_audio(config: &AppConfig) -> Result<(), String> {
    let audio = &config.audio;
    let device_len = audio.output_device.chars().count();
    if device_len == 0 || device_len > MAX_AUDIO_OUTPUT_DEVICE_LEN {
        return Err(format!(
            "audio.output_device must be between 1 and {MAX_AUDIO_OUTPUT_DEVICE_LEN} characters"
        ));
    }
    if audio.output_device.chars().any(|c| c.is_control()) {
        return Err("audio.output_device must not contain control characters".to_string());
    }
    if !(0.0..=1.0).contains(&audio.volume) || !audio.volume.is_finite() {
        return Err("audio.volume must be a finite number between 0.0 and 1.0".to_string());
    }
    if !(1..=MAX_PLAYBACK_QUEUE_ITEMS_LIMIT).contains(&audio.max_playback_queue_items) {
        return Err(format!(
            "audio.max_playback_queue_items must be between 1 and {MAX_PLAYBACK_QUEUE_ITEMS_LIMIT}"
        ));
    }
    Ok(())
}

fn validate_model(config: &AppConfig) -> Result<(), String> {
    let model = &config.model;
    if !(1..=MAX_INFERENCE_STEPS).contains(&model.inference_steps) {
        return Err(format!(
            "model.inference_steps must be between 1 and {MAX_INFERENCE_STEPS}"
        ));
    }
    if let Err(message) = crate::tts::backend::BackendChoice::resolve(&model.backend) {
        return Err(message);
    }
    if !crate::tts::download::is_valid_commit_sha(&model.revision) {
        return Err(
            "model.revision must be a 40-character commit SHA (mutable names like 'main' are rejected)"
                .to_string(),
        );
    }
    if model.revision != super::model::DEFAULT_MODEL_REVISION && model.hashes_path.is_none() {
        return Err(
            "a custom model.revision requires model.hashes_path with trusted hashes for that exact revision"
                .to_string(),
        );
    }
    if !(1..=MAX_DOWNLOAD_CONNECT_TIMEOUT_SECS).contains(&model.download_connect_timeout_secs) {
        return Err(format!(
            "model.download_connect_timeout_secs must be between 1 and {MAX_DOWNLOAD_CONNECT_TIMEOUT_SECS}"
        ));
    }
    if !(1..=MAX_DOWNLOAD_TIMEOUT_SECS).contains(&model.download_timeout_secs) {
        return Err(format!(
            "model.download_timeout_secs must be between 1 and {MAX_DOWNLOAD_TIMEOUT_SECS}"
        ));
    }
    Ok(())
}

fn validate_tts(config: &AppConfig) -> Result<(), String> {
    let tts = &config.tts;
    if !(1..=MAX_TEXT_LENGTH_LIMIT).contains(&tts.max_text_length) {
        return Err(format!(
            "tts.max_text_length must be between 1 and {MAX_TEXT_LENGTH_LIMIT}"
        ));
    }
    if !(1..=MAX_CHUNK_CHARS_LIMIT).contains(&tts.max_chunk_chars) {
        return Err(format!(
            "tts.max_chunk_chars must be between 1 and {MAX_CHUNK_CHARS_LIMIT}"
        ));
    }
    for (name, value) in [
        ("tts.max_body_bytes", tts.max_body_bytes),
        ("tts.openai_max_body_bytes", tts.openai_max_body_bytes),
        ("tts.queue_max_body_bytes", tts.queue_max_body_bytes),
        ("tts.admin_max_body_bytes", tts.admin_max_body_bytes),
    ] {
        if !(MIN_BODY_BYTES..=MAX_BODY_BYTES).contains(&value) {
            return Err(format!(
                "{name} must be between {MIN_BODY_BYTES} and {MAX_BODY_BYTES}"
            ));
        }
    }
    validate_normalization(&tts.normalization)
}

/// Upper bound for `tts.normalization.max_normalized_chars`.
const MAX_NORMALIZED_CHARS_LIMIT: usize = 200_000;
/// Bounds for `tts.normalization.max_expansion_ratio`.
const MIN_EXPANSION_RATIO: f32 = 1.0;
const MAX_EXPANSION_RATIO: f32 = 16.0;

fn validate_normalization(n: &super::model::NormalizationConfig) -> Result<(), String> {
    use crate::tts::normalization::{NormalizationMode, ReadingStyle, is_supported_locale};
    if NormalizationMode::parse(&n.mode).is_none() {
        return Err("tts.normalization.mode must be one of: auto, off".to_string());
    }
    if ReadingStyle::parse(&n.reading_style).is_none() {
        return Err(
            "tts.normalization.reading_style must be one of: precise, natural, digits".to_string(),
        );
    }
    let locale = n.default_locale.trim();
    if !locale.eq_ignore_ascii_case("auto") && !is_supported_locale(locale) {
        return Err(
            "tts.normalization.default_locale must be \"auto\" or a supported locale tag              (for example es-ES, es-MX, es-PE, en-US, en-GB, pt-BR, pt-PT, fr-FR, de-DE,              it-IT, ru-RU, zh-CN, zh-TW, ja-JP, ko-KR)"
                .to_string(),
        );
    }
    if !n.max_expansion_ratio.is_finite()
        || !(MIN_EXPANSION_RATIO..=MAX_EXPANSION_RATIO).contains(&n.max_expansion_ratio)
    {
        return Err(format!(
            "tts.normalization.max_expansion_ratio must be a finite number between              {MIN_EXPANSION_RATIO} and {MAX_EXPANSION_RATIO}"
        ));
    }
    if !(1..=MAX_NORMALIZED_CHARS_LIMIT).contains(&n.max_normalized_chars) {
        return Err(format!(
            "tts.normalization.max_normalized_chars must be between 1 and {MAX_NORMALIZED_CHARS_LIMIT}"
        ));
    }
    Ok(())
}

fn validate_inference(config: &AppConfig) -> Result<(), String> {
    let inference = &config.inference;
    if !(1..=86400).contains(&inference.generation_deadline_secs)
        || !(1..=86400).contains(&inference.job_retention_secs)
        || !(1..=1024).contains(&inference.max_retained_jobs)
    {
        return Err(
            "inference deadline/retention must be 1..=86400 seconds and max_retained_jobs 1..=1024"
                .into(),
        );
    }
    if !(1..=MAX_CONCURRENT_INFERENCE_LIMIT).contains(&inference.max_concurrent) {
        return Err(format!(
            "inference.max_concurrent must be between 1 and {MAX_CONCURRENT_INFERENCE_LIMIT}"
        ));
    }
    if !(0..=MAX_PENDING_INFERENCE_LIMIT).contains(&inference.max_pending) {
        return Err(format!(
            "inference.max_pending must be between 0 and {MAX_PENDING_INFERENCE_LIMIT}"
        ));
    }
    Ok(())
}

fn validate_rate_limit(config: &AppConfig) -> Result<(), String> {
    let rl = &config.rate_limit;
    if rl.requests > MAX_RATE_LIMIT_REQUESTS {
        return Err(format!(
            "rate_limit.requests must be between 0 and {MAX_RATE_LIMIT_REQUESTS} (0 disables limiting)"
        ));
    }
    if !(1..=MAX_RATE_LIMIT_WINDOW_SECS).contains(&rl.window_secs) {
        return Err(format!(
            "rate_limit.window_secs must be between 1 and {MAX_RATE_LIMIT_WINDOW_SECS}"
        ));
    }
    if rl.burst > MAX_RATE_LIMIT_REQUESTS {
        return Err(format!(
            "rate_limit.burst must be between 0 and {MAX_RATE_LIMIT_REQUESTS}"
        ));
    }
    if rl.requests > 0 && rl.burst == 0 {
        return Err(format!(
            "rate_limit.burst must be between 1 and {MAX_RATE_LIMIT_REQUESTS} when rate limiting is enabled"
        ));
    }
    Ok(())
}

fn validate_logging(config: &AppConfig) -> Result<(), String> {
    if let Some(filter) = &config.logging.filter {
        if filter.trim().is_empty() {
            return Err("logging.filter must not be empty".into());
        }
        tracing_subscriber::EnvFilter::try_new(filter)
            .map_err(|e| format!("invalid logging.filter: {e}"))?;
    }
    let level = config.logging.level.trim();
    if level.is_empty() {
        return Err("logging.level must not be empty".to_string());
    }
    // Accept the standard tracing levels (case-insensitive).
    if !["trace", "debug", "info", "warn", "error"].contains(&level.to_ascii_lowercase().as_str()) {
        return Err("logging.level must be one of: trace, debug, info, warn, error".to_string());
    }
    Ok(())
}

fn validate_paths(config: &AppConfig) -> Result<(), String> {
    let paths = &config.paths;
    if paths.logs.trim().is_empty() {
        return Err("paths.logs must not be empty".to_string());
    }
    if paths.temp_audio.trim().is_empty() {
        return Err("paths.temp_audio must not be empty".to_string());
    }
    if paths.token_store.trim().is_empty() {
        return Err("paths.token_store must not be empty".to_string());
    }
    if config.model.cache_dir.as_os_str().is_empty() {
        return Err("model.cache_dir must not be empty".to_string());
    }
    #[cfg(feature = "playback")]
    {
        match &paths.audio {
            Some(dir) if !dir.trim().is_empty() => {}
            _ => {
                return Err(
                    "paths.audio must be set when the playback/filesystem queue is enabled. \
                     Set it in config.toml (e.g. paths.audio = \"./audio\") and create the \
                     directory, or run headless with \
                     'cargo run --no-default-features --features server'."
                        .to_string(),
                );
            }
        }
    }
    Ok(())
}

fn validate_security(config: &AppConfig) -> Result<(), String> {
    let security = &config.security;
    if security.trust_proxy && security.trusted_proxies.is_empty() {
        return Err(
            "security.trusted_proxies must list at least one proxy address when \
             security.trust_proxy is enabled"
                .to_string(),
        );
    }
    for entry in &security.trusted_proxies {
        if !crate::admin::client_ip::is_valid_proxy_entry(entry) {
            return Err(format!(
                "security.trusted_proxies entry '{entry}' is not a valid IP address or CIDR prefix"
            ));
        }
    }
    Ok(())
}

fn validate_providers(config: &AppConfig) -> Result<(), String> {
    if !(1..=4096).contains(&config.voices.max_profiles) {
        return Err("voices.max_profiles must be 1..=4096".into());
    }
    for (name, candle) in [
        ("qwen3_candle", &config.providers.qwen3_candle),
        ("kyutai_candle", &config.providers.kyutai_candle),
    ] {
        if !["auto", "cpu", "vulkan"].contains(&candle.device.as_str()) {
            return Err(format!(
                "providers.{name}.device must be one of: auto, cpu, vulkan"
            ));
        }
        if candle.default_voice.is_empty()
            || candle.default_voice.contains(['/', '\\'])
            || candle.default_voice.contains("..")
        {
            return Err(format!(
                "providers.{name}.default_voice must be a plain voice name"
            ));
        }
    }
    let gguf = &config.providers.qwen3_gguf;
    if !["cpu", "auto", "vulkan", "cuda"].contains(&gguf.device.as_str()) {
        return Err(
            "providers.qwen3_gguf.device must be one of: cpu, auto, vulkan, cuda".to_string(),
        );
    }
    if gguf.max_concurrent != 1 {
        return Err("providers.qwen3_gguf.max_concurrent must be 1: one model-owning worker serializes native inference".to_string());
    }
    if !(1..=1024).contains(&gguf.cache_entries) {
        return Err("providers.qwen3_gguf.cache_entries must be between 1 and 1024".to_string());
    }
    if !(1..=1024).contains(&gguf.max_tokens) {
        return Err("providers.qwen3_gguf.max_tokens must be between 1 and 1024".to_string());
    }
    if !(1..=256).contains(&gguf.threads) {
        return Err("providers.qwen3_gguf.threads must be between 1 and 256".to_string());
    }
    if gguf.gpu_layers < -1 {
        return Err("providers.qwen3_gguf.gpu_layers must be -1 (all) or >= 0".to_string());
    }
    if let Some(params) = &gguf.sampling.talker {
        params.validate("providers.qwen3_gguf.sampling.talker")?;
    }
    if let Some(params) = &gguf.sampling.predictor {
        params.validate("providers.qwen3_gguf.sampling.predictor")?;
    }
    if !["off", "report", "retry", "experimental_trim"].contains(&gguf.quality.policy.as_str()) {
        return Err(
            "providers.qwen3_gguf.quality.policy must be one of: off, report, retry, \
             experimental_trim"
                .to_string(),
        );
    }
    if gguf.quality.max_retries > sonicboom::tts::quality::MAX_RETRIES_CEILING {
        return Err(format!(
            "providers.qwen3_gguf.quality.max_retries must be 0 or {} (unbounded regeneration is \
             rejected)",
            sonicboom::tts::quality::MAX_RETRIES_CEILING
        ));
    }
    if gguf.model_dir.as_os_str().is_empty() {
        return Err("providers.qwen3_gguf.model_dir must not be empty".to_string());
    }
    let pocket = &config.providers.pocket_tts;
    if !["cpu", "auto"].contains(&pocket.device.as_str()) {
        return Err("providers.pocket_tts.device must be one of: cpu, auto \
             (GPU execution providers are unsupported for Pocket TTS)"
            .to_string());
    }
    if pocket.model_id.trim().is_empty() || pocket.model_id.len() > 64 {
        return Err("providers.pocket_tts.model_id must be 1-64 characters".to_string());
    }
    if pocket.revision.trim().is_empty() || pocket.revision.len() > 128 {
        return Err("providers.pocket_tts.revision must be 1-128 characters".to_string());
    }
    if pocket.max_concurrent != 1 {
        return Err(
            "providers.pocket_tts.max_concurrent must be 1: mutable model state is serialized"
                .to_string(),
        );
    }
    if !(1..=1024).contains(&pocket.cache_entries) {
        return Err("providers.pocket_tts.cache_entries must be between 1 and 1024".to_string());
    }
    if !(0.0..=2.0).contains(&pocket.temperature) || !pocket.temperature.is_finite() {
        return Err("providers.pocket_tts.temperature must be between 0.0 and 2.0".to_string());
    }
    if !(1..=8).contains(&pocket.lsd_steps) {
        return Err("providers.pocket_tts.lsd_steps must be between 1 and 8".to_string());
    }
    if !(1..=32).contains(&pocket.cpu_threads) {
        return Err("providers.pocket_tts.cpu_threads must be between 1 and 32".to_string());
    }
    if pocket.model_dir.as_os_str().is_empty() {
        return Err("providers.pocket_tts.model_dir must not be empty".to_string());
    }
    Ok(())
}

fn validate_voices(config: &AppConfig) -> Result<(), String> {
    if config.voices.store_dir.trim().is_empty() {
        return Err("voices.store_dir must not be empty".to_string());
    }
    if !(1024..=64 * 1024 * 1024).contains(&config.voices.max_upload_bytes) {
        return Err("voices.max_upload_bytes must be between 1024 and 67108864".to_string());
    }
    Ok(())
}

impl AppConfig {
    /// Whether `paths.audio` (the queue root) is configured.
    pub fn requires_audio_dir(&self) -> bool {
        self.paths
            .audio
            .as_deref()
            .map(str::trim)
            .is_some_and(|d| !d.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> AppConfig {
        let mut config = AppConfig::default();
        config.admin.username = "admin".to_string();
        #[cfg(feature = "playback")]
        {
            config.paths.audio = Some("./audio".to_string());
        }
        config
    }

    #[test]
    fn normalization_defaults_are_valid_and_bad_values_rejected() {
        let mut config = valid_config();
        config.tts.normalization.mode = "OFF".to_string();
        config.tts.normalization.default_locale = "es-pe".to_string();
        config.tts.normalization.reading_style = "NATURAL".to_string();
        assert!(validate(&config).is_ok());

        type Mutation = fn(&mut AppConfig);
        let cases: [(&str, Mutation); 9] = [
            ("tts.normalization.mode", |c| {
                c.tts.normalization.mode = "strict".into()
            }),
            ("tts.normalization.reading_style", |c| {
                c.tts.normalization.reading_style = "loud".into()
            }),
            ("tts.normalization.default_locale", |c| {
                c.tts.normalization.default_locale = "xx-YY".into()
            }),
            ("tts.normalization.max_expansion_ratio", |c| {
                c.tts.normalization.max_expansion_ratio = 0.5
            }),
            ("tts.normalization.max_expansion_ratio", |c| {
                c.tts.normalization.max_expansion_ratio = 16.5
            }),
            ("tts.normalization.max_expansion_ratio", |c| {
                c.tts.normalization.max_expansion_ratio = f32::NAN
            }),
            ("tts.normalization.max_expansion_ratio", |c| {
                c.tts.normalization.max_expansion_ratio = f32::INFINITY
            }),
            ("tts.normalization.max_normalized_chars", |c| {
                c.tts.normalization.max_normalized_chars = 0
            }),
            ("tts.normalization.max_normalized_chars", |c| {
                c.tts.normalization.max_normalized_chars = 200_001
            }),
        ];
        for (field, mutate) in cases {
            let mut config = valid_config();
            mutate(&mut config);
            let error = validate(&config).unwrap_err();
            assert!(error.contains(field), "{error}");
        }
    }

    #[test]
    fn default_config_is_valid() {
        assert!(validate(&valid_config()).is_ok());
    }

    #[test]
    fn local_auth_with_loopback_is_valid() {
        for bind in ["127.0.0.1", "::1"] {
            let mut config = valid_config();
            config.server.auth_mode = AuthMode::Local;
            config.server.bind = bind.parse().unwrap();
            assert!(validate(&config).is_ok(), "bind {bind}");
        }
    }

    #[test]
    fn local_auth_with_non_loopback_is_rejected() {
        let mut config = valid_config();
        config.server.auth_mode = AuthMode::Local;
        config.server.bind = "0.0.0.0".parse().unwrap();
        assert!(validate(&config).is_err());
    }

    #[test]
    fn token_auth_may_bind_anywhere() {
        let mut config = valid_config();
        config.server.auth_mode = AuthMode::Token;
        config.server.bind = "0.0.0.0".parse().unwrap();
        assert!(validate(&config).is_ok());
    }

    #[test]
    fn none_auth_requires_explicit_remote_opt_in() {
        let mut config = valid_config();
        config.server.auth_mode = AuthMode::None;
        config.server.bind = "0.0.0.0".parse().unwrap();
        assert!(validate(&config).is_err());
        config.security.allow_insecure_remote = true;
        assert!(validate(&config).is_ok());
    }

    #[test]
    fn admin_password_policy_enforced() {
        assert!(validate_admin_password("").is_err());
        assert!(validate_admin_password("short").is_err());
        assert!(validate_admin_password("password").is_err());
        assert!(validate_admin_password("correct-horse-battery-staple").is_ok());
        // Length counts Unicode characters.
        assert!(validate_admin_password("pässwörd-ün").is_err()); // 11 chars
        assert!(validate_admin_password("pässwörd-üni").is_ok()); // 12 chars
    }

    #[test]
    fn bounded_values_are_enforced() {
        let mut config = valid_config();
        config.model.inference_steps = MAX_INFERENCE_STEPS + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.inference.max_concurrent = MAX_CONCURRENT_INFERENCE_LIMIT + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.inference.max_pending = MAX_PENDING_INFERENCE_LIMIT + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.tts.max_text_length = MAX_TEXT_LENGTH_LIMIT + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.tts.max_chunk_chars = MAX_CHUNK_CHARS_LIMIT + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.server.request_timeout_secs = MAX_REQUEST_TIMEOUT_SECS + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.rate_limit.window_secs = MAX_RATE_LIMIT_WINDOW_SECS + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.rate_limit.requests = MAX_RATE_LIMIT_REQUESTS + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.audio.max_playback_queue_items = MAX_PLAYBACK_QUEUE_ITEMS_LIMIT + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.tts.max_body_bytes = MAX_BODY_BYTES + 1;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.admin.session_expiry_secs = MAX_ADMIN_SESSION_EXPIRY_SECS + 1;
        assert!(validate(&config).is_err());
        // Boundary values are accepted.
        let mut config = valid_config();
        config.model.inference_steps = MAX_INFERENCE_STEPS;
        config.inference.max_concurrent = MAX_CONCURRENT_INFERENCE_LIMIT;
        config.inference.max_pending = MAX_PENDING_INFERENCE_LIMIT;
        config.tts.max_text_length = MAX_TEXT_LENGTH_LIMIT;
        config.tts.max_chunk_chars = MAX_CHUNK_CHARS_LIMIT;
        config.server.request_timeout_secs = MAX_REQUEST_TIMEOUT_SECS;
        config.rate_limit.window_secs = MAX_RATE_LIMIT_WINDOW_SECS;
        config.rate_limit.requests = MAX_RATE_LIMIT_REQUESTS;
        config.audio.max_playback_queue_items = MAX_PLAYBACK_QUEUE_ITEMS_LIMIT;
        config.tts.max_body_bytes = MAX_BODY_BYTES;
        config.admin.session_expiry_secs = MAX_ADMIN_SESSION_EXPIRY_SECS;
        assert!(validate(&config).is_ok());
    }

    #[test]
    fn invalid_log_level_is_rejected() {
        let mut config = valid_config();
        config.logging.level = "verbose".to_string();
        assert!(validate(&config).is_err());
        config.logging.level = "debug".to_string();
        assert!(validate(&config).is_ok());
    }

    #[test]
    fn zero_port_is_rejected() {
        let mut config = valid_config();
        config.server.port = 0;
        assert!(validate(&config).is_err());
    }

    #[test]
    fn optional_providers_default_to_disabled() {
        let config = valid_config();
        assert!(!config.providers.qwen3_gguf.enabled);
        assert!(!config.providers.gpt_sovits.enabled);
        assert!(validate(&config).is_ok());
    }

    #[test]
    fn gguf_section_defaults_to_disabled_and_validates() {
        let config = valid_config();
        assert!(!config.providers.qwen3_gguf.enabled);
        assert!(validate(&config).is_ok());
        let mut config = valid_config();
        config.providers.qwen3_gguf.device = "rocm".to_string();
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.providers.qwen3_gguf.device = "vulkan".to_string();
        assert!(validate(&config).is_ok());
        let mut config = valid_config();
        config.providers.qwen3_gguf.max_concurrent = 2;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.providers.qwen3_gguf.max_tokens = 0;
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.providers.qwen3_gguf.threads = 257;
        assert!(validate(&config).is_err());
    }

    #[test]
    fn provider_devices_are_constrained() {
        // Pocket TTS is CPU-only: GPU values are rejected, not reinterpreted.
        let mut config = valid_config();
        config.providers.pocket_tts.device = "cuda".to_string();
        assert!(validate(&config).is_err());
        let mut config = valid_config();
        config.providers.pocket_tts.device = "rocm".to_string();
        assert!(validate(&config).is_err());
        // A leftover [providers.gpt_sovits] section is tolerated and ignored.
        let mut config = valid_config();
        config.providers.gpt_sovits.enabled = true;
        config.providers.gpt_sovits.device = Some("cuda".to_string());
        assert!(validate(&config).is_ok());
    }
}
