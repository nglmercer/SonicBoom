//! Migration from the legacy `.env` configuration to
//! `config.toml`.
//!
//! When `config.toml` is missing but a legacy `.env` exists,
//! known values are converted and persisted; secrets
//! (admin password, HuggingFace token) are kept in the
//! secret store / environment and are **never** written
//! into `config.toml`.

use std::collections::HashMap;
use std::path::Path;

use super::model::{AppConfig, AuthMode};

/// Keys that carry secrets and must never be written to
/// `config.toml`.
pub const SECRET_KEYS: &[&str] = &["SONICBOOM_ADMIN_PW", "HF_TOKEN"];

/// Parse a `.env` file into a key/value map (best-effort:
/// comments and malformed lines are skipped).
pub fn parse_env_file(path: &Path) -> Option<HashMap<String, String>> {
    if let Ok(values) = dotenvy::from_path_iter(path)
        .ok()?
        .collect::<Result<HashMap<_, _>, _>>()
    {
        return Some(values);
    }
    // Legacy device names were accepted without quotes. Preserve that shape
    // while still rejecting malformed assignments and unterminated quotes.
    let text = std::fs::read_to_string(path).ok()?;
    for line in text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let (key, value) = line
            .strip_prefix("export ")
            .unwrap_or(line)
            .split_once('=')?;
        if !key
            .trim()
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return None;
        }
        let value = value.trim();
        if (value.starts_with('"') && !value.ends_with('"'))
            || (value.starts_with('\'') && !value.ends_with('\''))
        {
            return None;
        }
    }
    Some(parse_env_text(&text))
}

fn parse_env_text(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let mut value = value.trim();
        // Strip surrounding quotes.
        if value.len() >= 2
            && ((value.starts_with('"') && value.ends_with('"'))
                || (value.starts_with('\'') && value.ends_with('\'')))
        {
            value = &value[1..value.len() - 1];
        }
        if !key.is_empty() {
            map.insert(key.to_string(), value.to_string());
        }
    }
    map
}

/// Migrate a legacy `.env` at `env_path` into the configuration,
/// returning the migrated configuration plus a report. The
/// result is **not** persisted here — the caller validates and
/// writes it through the `ConfigManager` so there is exactly
/// one write path.
pub fn migrate_env(
    env_path: &Path,
    fallback: &AppConfig,
) -> Result<Option<(AppConfig, Vec<&'static str>, Vec<&'static str>)>, String> {
    let env = parse_env_file(env_path)
        .ok_or_else(|| "could not read legacy environment file".to_string())?;
    if env.is_empty() {
        return Ok(None);
    }
    let mut config = fallback.clone();
    let mut converted: Vec<&'static str> = Vec::new();
    let mut secrets: Vec<&'static str> = Vec::new();

    let take = |key: &str| -> Option<String> {
        env.get(key)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };

    if let Some(v) = take("PORT").and_then(|v| v.parse::<u16>().ok()) {
        config.server.port = v;
        converted.push("PORT -> server.port");
    }
    if let Some(v) = take("BIND").and_then(|v| v.parse::<std::net::IpAddr>().ok()) {
        config.server.bind = v;
        converted.push("BIND -> server.bind");
    }
    if let Some(v) = take("SONICBOOM_AUTH_REQUIRED") {
        match v.to_ascii_lowercase().as_str() {
            "true" | "1" => {
                config.server.auth_mode = AuthMode::Token;
                converted.push("SONICBOOM_AUTH_REQUIRED -> server.auth_mode");
            }
            "false" | "0" => {
                config.server.auth_mode = AuthMode::None;
                converted.push("SONICBOOM_AUTH_REQUIRED -> server.auth_mode");
            }
            _ => {}
        }
    }
    if let Some(v) = take("REQUEST_TIMEOUT_SECS").and_then(|v| v.parse::<u64>().ok()) {
        config.server.request_timeout_secs = v;
        converted.push("REQUEST_TIMEOUT_SECS -> server.request_timeout_secs");
    }
    if let Some(v) = take("SONICBOOM_ADMIN_ID") {
        config.admin.username = v;
        converted.push("SONICBOOM_ADMIN_ID -> admin.username");
    }
    if let Some(v) = take("ADMIN_SESSION_EXPIRY_SECS").and_then(|v| v.parse::<i64>().ok()) {
        config.admin.session_expiry_secs = v;
        converted.push("ADMIN_SESSION_EXPIRY_SECS -> admin.session_expiry_secs");
    }
    if let Some(v) = take("ENABLE_SAMPLE_TOKEN") {
        if matches!(v.to_ascii_lowercase().as_str(), "true" | "1") {
            config.admin.enable_sample_token = true;
            converted.push("ENABLE_SAMPLE_TOKEN -> admin.enable_sample_token");
        }
    }
    if let Some(v) = take("AUDIO_OUTPUT_DEVICE") {
        config.audio.output_device = v;
        converted.push("AUDIO_OUTPUT_DEVICE -> audio.output_device");
    }
    if let Some(v) = take("MAX_PLAYBACK_QUEUE_ITEMS").and_then(|v| v.parse::<usize>().ok()) {
        config.audio.max_playback_queue_items = v;
        converted.push("MAX_PLAYBACK_QUEUE_ITEMS -> audio.max_playback_queue_items");
    }
    if let Some(v) = take("MODEL_CACHE_DIR") {
        config.model.cache_dir = std::path::PathBuf::from(v);
        converted.push("MODEL_CACHE_DIR -> model.cache_dir");
    }
    if let Some(v) = take("MODEL_REVISION") {
        let v = v.to_lowercase();
        config.model.revision = v;
        converted.push("MODEL_REVISION -> model.revision");
    }
    if let Some(v) = take("MODEL_SHA256_JSON_PATH") {
        config.model.hashes_path = Some(std::path::PathBuf::from(v));
        converted.push("MODEL_SHA256_JSON_PATH -> model.hashes_path");
    }
    if let Some(v) = take("INFERENCE_STEPS").and_then(|v| v.parse::<usize>().ok()) {
        config.model.inference_steps = v;
        converted.push("INFERENCE_STEPS -> model.inference_steps");
    }
    if let Some(v) = take("MAX_TEXT_LENGTH").and_then(|v| v.parse::<usize>().ok()) {
        config.tts.max_text_length = v;
        converted.push("MAX_TEXT_LENGTH -> tts.max_text_length");
    }
    if let Some(v) = take("MAX_CHUNK_CHARS").and_then(|v| v.parse::<usize>().ok()) {
        config.tts.max_chunk_chars = v;
        converted.push("MAX_CHUNK_CHARS -> tts.max_chunk_chars");
    }
    if let Some(v) = take("TTS_MAX_BODY_BYTES").and_then(|v| v.parse::<usize>().ok()) {
        config.tts.max_body_bytes = v;
        converted.push("TTS_MAX_BODY_BYTES -> tts.max_body_bytes");
    }
    if let Some(v) = take("OPENAI_MAX_BODY_BYTES").and_then(|v| v.parse::<usize>().ok()) {
        config.tts.openai_max_body_bytes = v;
        converted.push("OPENAI_MAX_BODY_BYTES -> tts.openai_max_body_bytes");
    }
    if let Some(v) = take("QUEUE_MAX_BODY_BYTES").and_then(|v| v.parse::<usize>().ok()) {
        config.tts.queue_max_body_bytes = v;
        converted.push("QUEUE_MAX_BODY_BYTES -> tts.queue_max_body_bytes");
    }
    if let Some(v) = take("ADMIN_MAX_BODY_BYTES").and_then(|v| v.parse::<usize>().ok()) {
        config.tts.admin_max_body_bytes = v;
        converted.push("ADMIN_MAX_BODY_BYTES -> tts.admin_max_body_bytes");
    }
    if let Some(v) = take("MAX_CONCURRENT_INFERENCE").and_then(|v| v.parse::<usize>().ok()) {
        config.inference.max_concurrent = v;
        converted.push("MAX_CONCURRENT_INFERENCE -> inference.max_concurrent");
    }
    if let Some(v) = take("MAX_PENDING_INFERENCE").and_then(|v| v.parse::<usize>().ok()) {
        config.inference.max_pending = v;
        converted.push("MAX_PENDING_INFERENCE -> inference.max_pending");
    }
    if let Some(v) = take("TTS_RATE_LIMIT_REQUESTS").and_then(|v| v.parse::<u32>().ok()) {
        config.rate_limit.requests = v;
        converted.push("TTS_RATE_LIMIT_REQUESTS -> rate_limit.requests");
    }
    if let Some(v) = take("TTS_RATE_LIMIT_WINDOW_SECS").and_then(|v| v.parse::<u64>().ok()) {
        config.rate_limit.window_secs = v;
        converted.push("TTS_RATE_LIMIT_WINDOW_SECS -> rate_limit.window_secs");
    }
    if let Some(v) = take("TTS_RATE_LIMIT_BURST").and_then(|v| v.parse::<u32>().ok()) {
        config.rate_limit.burst = v;
        converted.push("TTS_RATE_LIMIT_BURST -> rate_limit.burst");
    }
    if let Some(v) = take("LOG_LEVEL") {
        config.logging.level = v;
        converted.push("LOG_LEVEL -> logging.level");
    }
    if let Some(v) = take("LOG_TO_FILE") {
        if matches!(v.to_ascii_lowercase().as_str(), "true" | "1") {
            config.logging.to_file = true;
            converted.push("LOG_TO_FILE -> logging.to_file");
        }
    }
    if let Some(v) = take("LOG_TO_STDOUT") {
        if matches!(v.to_ascii_lowercase().as_str(), "true" | "1") {
            config.logging.to_stdout = true;
            converted.push("LOG_TO_STDOUT -> logging.to_stdout");
        }
    }
    if let Some(v) = take("LOG_DIR") {
        config.paths.logs = v;
        converted.push("LOG_DIR -> paths.logs");
    }
    if let Some(v) = take("ALLOWED_AUDIO_DIR") {
        config.paths.audio = Some(v);
        converted.push("ALLOWED_AUDIO_DIR -> paths.audio");
    }
    if let Some(v) = take("TEMP_AUDIO_DIR") {
        config.paths.temp_audio = v;
        converted.push("TEMP_AUDIO_DIR -> paths.temp_audio");
    }
    if let Some(v) = take("TOKEN_STORE_PATH") {
        config.paths.token_store = v;
        converted.push("TOKEN_STORE_PATH -> paths.token_store");
    }
    if let Some(v) = take("TRUST_PROXY") {
        if matches!(v.to_ascii_lowercase().as_str(), "true" | "1") {
            config.security.trust_proxy = true;
            converted.push("TRUST_PROXY -> security.trust_proxy");
        }
    }
    if let Some(v) = take("TRUSTED_PROXIES") {
        config.security.trusted_proxies = v
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        converted.push("TRUSTED_PROXIES -> security.trusted_proxies");
    }
    if let Some(v) = take("COOKIE_SECURE") {
        if matches!(v.to_ascii_lowercase().as_str(), "true" | "1") {
            config.security.cookie_secure = true;
            converted.push("COOKIE_SECURE -> security.cookie_secure");
        }
    }
    if let Some(v) = take("ENABLE_HSTS") {
        if matches!(v.to_ascii_lowercase().as_str(), "true" | "1") {
            config.security.enable_hsts = true;
            converted.push("ENABLE_HSTS -> security.enable_hsts");
        }
    }

    // Secrets are reported but never written into the TOML.
    for key in SECRET_KEYS {
        if env.contains_key(*key) {
            secrets.push(*key);
        }
    }

    let get = |key: &str| env.get(key).cloned();
    super::loader::apply_environment_with(&get, &mut config, &mut HashMap::new())
        .map_err(|error| error.to_string())?;
    if let Some(password) = env.get("SONICBOOM_ADMIN_PW") {
        super::validation::validate_admin_password(password)?;
        config.setup_complete = true;
    }
    Ok(Some((config, converted, secrets)))
}

/// Perform first-run migration: when `config.toml` does not
/// exist but a legacy `.env` does, convert it, validate the
/// result, and atomically write `config.toml` (without
/// secrets). Returns a human-readable summary when a
/// migration happened.
pub fn migrate_if_needed(config_path: &Path, env_path: &Path) -> Result<Option<String>, String> {
    if config_path.is_file() || !env_path.exists() {
        return Ok(None);
    }
    let Some((config, converted, secrets)) = migrate_env(env_path, &AppConfig::default())? else {
        return Ok(None);
    };
    super::validation::validate(&config)
        .map_err(|e| format!("migrated configuration failed validation: {e}"))?;
    let legacy =
        parse_env_file(env_path).ok_or_else(|| "could not read legacy secrets".to_string())?;
    let store = super::secrets::FileSecretStore::new(config_path.with_file_name("secrets.json"));
    use super::secrets::SecretStore;
    for (old_key, new_key) in [
        ("SONICBOOM_ADMIN_PW", super::secrets::keys::ADMIN_PASSWORD),
        ("HF_TOKEN", super::secrets::keys::HF_TOKEN),
    ] {
        if let Some(value) = legacy.get(old_key) {
            store.set(new_key, value).map_err(|e| e.to_string())?;
        }
    }
    // Persist through the atomic writer. The `ConfigManager`
    // remains the single logical write path at runtime; this
    // bootstrap path only runs when no config exists yet.
    super::writer::atomic_write_config_blocking(config_path, &config).map_err(|e| e.to_string())?;
    let mut summary = format!(
        "Migrated {} setting(s) from {} to {}",
        converted.len(),
        env_path.display(),
        config_path.display()
    );
    if !secrets.is_empty() {
        summary.push_str(". Secrets kept in the environment/secret store: ");
        summary.push_str(&secrets.join(", "));
    }
    Ok(Some(summary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-migrate-test-{}-{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn parse_env_text_handles_quotes_and_comments() {
        let env =
            parse_env_text("# comment\nPORT=17842\nLOG_LEVEL=\"debug\"\nEMPTY=\nexport FOO=bar\n");
        assert_eq!(env["PORT"], "17842");
        assert_eq!(env["LOG_LEVEL"], "debug");
        assert_eq!(env.get("EMPTY"), Some(&"".to_string()));
        assert_eq!(env["FOO"], "bar");
    }

    #[test]
    fn migrate_converts_known_keys() {
        let dir = temp_dir("convert");
        let env_path = dir.join(".env");
        std::fs::write(
            &env_path,
            "PORT=19000\nAUDIO_OUTPUT_DEVICE=CABLE Input\nINFERENCE_STEPS=10\nLOG_LEVEL=debug\nTTS_RATE_LIMIT_REQUESTS=500\n",
        )
        .unwrap();
        let (config, converted, secrets) = migrate_env(&env_path, &AppConfig::default())
            .unwrap()
            .unwrap();
        assert_eq!(config.server.port, 19000);
        assert_eq!(config.audio.output_device, "CABLE Input");
        assert_eq!(config.model.inference_steps, 10);
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.rate_limit.requests, 500);
        assert!(converted.contains(&"PORT -> server.port"));
        assert!(converted.contains(&"AUDIO_OUTPUT_DEVICE -> audio.output_device"));
        assert!(converted.contains(&"INFERENCE_STEPS -> model.inference_steps"));
        assert!(converted.contains(&"LOG_LEVEL -> logging.level"));
        assert!(converted.contains(&"TTS_RATE_LIMIT_REQUESTS -> rate_limit.requests"));
        assert!(secrets.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_never_writes_secrets_into_toml() {
        let dir = temp_dir("secrets");
        let env_path = dir.join(".env");
        std::fs::write(
            &env_path,
            "SONICBOOM_ADMIN_PW=super-secret-password\nHF_TOKEN=hf_secret_token\nPORT=17842\n",
        )
        .unwrap();
        let (config, _converted, secrets) = migrate_env(&env_path, &AppConfig::default())
            .unwrap()
            .unwrap();
        assert_eq!(secrets.len(), 2);
        // The serialized TOML must not contain either secret.
        let text = super::super::writer::to_toml_string(&config).unwrap();
        assert!(!text.contains("super-secret-password"));
        assert!(!text.contains("hf_secret_token"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migration_of_invalid_values_is_rejected() {
        let dir = temp_dir("invalid-values");
        let env_path = dir.join(".env");
        std::fs::write(&env_path, "PORT=hello\nCOOKIE_SECURE=treu\n").unwrap();
        assert!(migrate_env(&env_path, &AppConfig::default()).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn migrate_if_needed_writes_valid_toml() {
        let dir = temp_dir("needed");
        let env_path = dir.join(".env");
        let config_path = dir.join("config.toml");
        std::fs::write(&env_path, "PORT=19000\nLOG_LEVEL=debug\n").unwrap();
        let summary = migrate_if_needed(&config_path, &env_path)
            .unwrap()
            .expect("migration should happen");
        assert!(summary.contains("Migrated"));
        assert!(config_path.is_file());
        // The written file parses and reflects the migration.
        let text = std::fs::read_to_string(&config_path).unwrap();
        let parsed = super::super::loader::parse_toml(&text).unwrap();
        assert_eq!(parsed.server.port, 19000);
        assert_eq!(parsed.logging.level, "debug");
        // Second run: config exists, no migration.
        assert!(
            migrate_if_needed(&config_path, &env_path)
                .unwrap()
                .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_if_needed_without_env_is_a_noop() {
        let dir = temp_dir("noop");
        let env_path = dir.join(".env");
        let config_path = dir.join("config.toml");
        assert!(
            migrate_if_needed(&config_path, &env_path)
                .unwrap()
                .is_none()
        );
        assert!(!config_path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
