//! `sonicboom config …` subcommands (spec §38).
//!
//! Every command operates on the same [`ConfigManager`]
//! the server, GUI, API, and MCP use — there is no
//! second configuration engine.

use std::sync::Arc;

use crate::config::effective::ConfigSource;
use crate::config::{self, AuthMode, ConfigManager};

/// Exit code for a handled CLI invocation.
pub type CliResult = Result<(), i32>;

/// Run `sonicboom config <subcommand> [args…]`.
///
/// Returns `Ok(())` on success and `Err(exit_code)`
/// on failure (the caller exits with that code).
pub async fn run_config(args: &[String]) -> CliResult {
    let (subcommand, rest) = match args.split_first() {
        Some(parts) => parts,
        None => {
            eprintln!(
                "usage: sonicboom config <get|set|list|effective|schema|validate|reload|path|status> [args]"
            );
            return Err(2);
        }
    };
    match subcommand.as_str() {
        "get" => cmd_get(rest).await,
        "set" => cmd_set(rest).await,
        "list" => cmd_list().await,
        "effective" => cmd_effective().await,
        "schema" => cmd_schema(),
        "validate" => cmd_validate().await,
        "reload" => cmd_reload().await,
        "path" => cmd_path(),
        "status" => cmd_status().await,
        other => {
            eprintln!("unknown config subcommand: {other}");
            Err(2)
        }
    }
}

/// Resolve the managed config path.
fn config_path() -> std::path::PathBuf {
    config::paths::resolve_config_path()
}

/// Load the configuration manager (startup-style
/// validation, bootstrap overrides applied).
async fn load_manager() -> Result<Arc<ConfigManager>, i32> {
    let path = config_path();
    if !path.is_file() {
        eprintln!(
            "config file not found: {}\n\
             run sonicboom to complete first-run setup",
            path.display()
        );
        return Err(1);
    }
    ConfigManager::load(path).await.map_err(|e| {
        eprintln!("failed to load configuration: {e}");
        1
    })
}

/// Prefer status from a local running application; offline operations still
/// use the same manager and are observed by its filesystem watcher.
async fn live_request(
    path: &str,
    method: reqwest::Method,
) -> Result<Option<serde_json::Value>, i32> {
    let cfg = config::loader::load_from_file(&config_path())
        .ok()
        .map(|e| e.config)
        .unwrap_or_default();
    let bind = if cfg.server.bind.is_unspecified() {
        if cfg.server.bind.is_ipv6() {
            "::1".parse().unwrap()
        } else {
            "127.0.0.1".parse().unwrap()
        }
    } else {
        cfg.server.bind
    };
    let address = std::net::SocketAddr::new(bind, cfg.server.port);
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .map_err(|_| 1)?;
    let info = match client
        .get(format!("http://{address}/api/info"))
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => {
            match response
                .bytes()
                .await
                .ok()
                .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            {
                Some(info) if info["features"]["hot_reload"] == true => info,
                _ => return Ok(None),
            }
        }
        _ => return Ok(None),
    };
    let mut request = client.request(method, format!("http://{address}{path}"));
    if info["auth"]["required"] == true {
        match std::env::var("SONICBOOM_API_TOKEN") {
            Ok(token) => request = request.bearer_auth(token),
            Err(_) => {
                eprintln!(
                    "running server requires authentication; set SONICBOOM_API_TOKEN for live configuration status"
                );
                return Err(1);
            }
        }
    }
    let response = request.send().await.map_err(|e| {
        eprintln!("configuration request failed: {e}");
        1
    })?;
    let status = response.status();
    let bytes = response.bytes().await.map_err(|_| 1)?;
    let body: serde_json::Value = serde_json::from_slice(&bytes).map_err(|_| 1)?;
    if !status.is_success() {
        eprintln!(
            "configuration request rejected: {}",
            body["message"]
                .as_str()
                .or_else(|| body["error"].as_str())
                .unwrap_or("request rejected")
        );
        return Err(1);
    }
    Ok(Some(body))
}

/// `config get <path>` — print the value at a dotted path.
async fn cmd_get(args: &[String]) -> CliResult {
    let path = match args.first() {
        Some(p) => p.as_str(),
        None => {
            eprintln!("usage: sonicboom config get <path>");
            return Err(2);
        }
    };
    let manager = load_manager().await?;
    let config = manager.get().await;
    match get_by_path(&config, path) {
        Some(value) => {
            println!("{value}");
            Ok(())
        }
        None => {
            eprintln!("unknown configuration path: {path}");
            Err(1)
        }
    }
}

/// `config set <path> <value> [--expected-revision N]`.
async fn cmd_set(args: &[String]) -> CliResult {
    let mut positional = Vec::new();
    let mut expected_revision: Option<u64> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--expected-revision" {
            let raw = match iter.next() {
                Some(v) => v,
                None => {
                    eprintln!("--expected-revision requires a value");
                    return Err(2);
                }
            };
            expected_revision = Some(raw.parse::<u64>().map_err(|_| {
                eprintln!("invalid revision: {raw}");
                2
            })?);
        } else {
            positional.push(arg.as_str());
        }
    }
    if positional.len() != 2 {
        eprintln!("usage: sonicboom config set <path> <value> [--expected-revision N]");
        return Err(2);
    }
    let (path, raw_value) = (positional[0], positional[1]);

    let manager = load_manager().await?;
    let revision = manager.revision();

    // Parse and validate the candidate before it
    // reaches the manager: a malformed value must
    // fail closed, never silently succeed.
    let schema = config::schema();
    let Some(setting) = schema.get(path) else {
        eprintln!("unknown or managed configuration path: {path}");
        return Err(2);
    };
    if setting.secret {
        eprintln!("secrets cannot be written to config.toml");
        return Err(2);
    }
    let type_name = setting.type_name;

    let mut candidate = manager.get_persisted().await;
    if let Err(e) = set_by_path(&mut candidate, path, raw_value, type_name) {
        eprintln!("configuration change failed: {e}");
        return Err(1);
    }
    if let Err(e) = config::validation::validate(&candidate) {
        eprintln!("configuration change failed: {e}");
        return Err(1);
    }

    let update = manager
        .update_from(
            expected_revision.or(Some(revision)),
            crate::config::diff::ChangeSource::Cli,
            |config| {
                *config = candidate.clone();
            },
        )
        .await;

    match update {
        Ok(update) => {
            println!("configuration revision {} -> {}", revision, update.revision);
            for change in &update.changes {
                println!("  {} ({})", change.path(), change.strategy().as_str());
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("configuration change failed: {e}");
            Err(1)
        }
    }
}

/// `config list` — every setting with its current value.
async fn cmd_list() -> CliResult {
    let manager = load_manager().await?;
    let config = manager.get().await;
    let schema = config::schema();
    for (path, setting) in &schema {
        if setting.secret {
            continue;
        }
        let value = get_by_path(&config, path).unwrap_or_else(|| "<unknown>".to_string());
        println!("{path} = {value}");
    }
    Ok(())
}

/// `config effective` — values with their sources.
async fn cmd_effective() -> CliResult {
    if let Some(effective) = live_request("/api/config/effective", reqwest::Method::GET).await? {
        println!("{}", serde_json::to_string_pretty(&effective).unwrap());
        return Ok(());
    }
    let manager = load_manager().await?;
    let effective = manager.get_effective().await;
    let schema = config::schema();
    for path in schema.keys() {
        if schema[path].secret {
            continue;
        }
        let value = get_by_path(&effective.config, path).unwrap_or_else(|| "<unknown>".to_string());
        let source = effective
            .source_of(path)
            .map(source_label)
            .unwrap_or("unknown")
            .to_string();
        println!("{path} = {value}  # source: {source}");
    }
    Ok(())
}

/// `config schema` — the agent/GUI-facing schema document.
fn cmd_schema() -> CliResult {
    let document = config::schema::schema_json();
    println!("{}", serde_json::to_string_pretty(&document).unwrap());
    Ok(())
}

/// `config validate` — validate the on-disk file.
async fn cmd_validate() -> CliResult {
    let manager = load_manager().await?;
    let config = manager.get().await;
    match config::validation::validate(&config) {
        Ok(()) => {
            println!("configuration is valid");
            Ok(())
        }
        Err(e) => {
            eprintln!("configuration is invalid: {e}");
            Err(1)
        }
    }
}

/// `config reload` — re-read the file through the manager.
///
/// In a running server the filesystem watcher applies
/// external edits automatically; this command proves the
/// on-disk file parses and validates right now.
async fn cmd_reload() -> CliResult {
    if let Some(response) = live_request("/api/config/reload", reqwest::Method::POST).await? {
        println!(
            "reloaded running application (revision {})",
            response["revision"]
        );
        return Ok(());
    }
    let manager = load_manager().await?;
    match manager.reload_from_disk().await {
        Ok(update) => {
            println!("reloaded configuration (revision {})", update.revision);
            Ok(())
        }
        Err(e) => {
            eprintln!("reload rejected: {e}");
            eprintln!("previous configuration remains active");
            Err(1)
        }
    }
}

/// `config path` — print the managed config file path.
fn cmd_path() -> CliResult {
    println!("{}", config_path().display());
    Ok(())
}

/// `config status` — watcher/validation status.
async fn cmd_status() -> CliResult {
    if let Some(status) = live_request("/api/config/status", reqwest::Method::GET).await? {
        println!("{}", serde_json::to_string_pretty(&status).unwrap());
        return Ok(());
    }
    let manager = load_manager().await?;
    let status = manager.status().await;
    println!("{}", serde_json::to_string_pretty(&status).unwrap());
    Ok(())
}

fn source_label(source: ConfigSource) -> &'static str {
    match source {
        ConfigSource::Default => "compiled default",
        ConfigSource::Toml => "config.toml",
        ConfigSource::Environment => "environment",
        ConfigSource::Runtime => "runtime",
    }
}

/// Read the value at a dotted path, formatted as a string.
pub fn get_by_path(config: &config::AppConfig, path: &str) -> Option<String> {
    match path {
        "version" => Some(config.version.to_string()),
        "setup_complete" => Some(config.setup_complete.to_string()),

        "server.bind" => Some(config.server.bind.to_string()),
        "server.port" => Some(config.server.port.to_string()),
        "server.auth_mode" => Some(auth_mode_str(config.server.auth_mode).to_string()),
        "server.request_timeout_secs" => Some(config.server.request_timeout_secs.to_string()),

        "admin.username" => Some(config.admin.username.clone()),
        "admin.session_expiry_secs" => Some(config.admin.session_expiry_secs.to_string()),
        "admin.enable_sample_token" => Some(config.admin.enable_sample_token.to_string()),

        "audio.output_device" => Some(config.audio.output_device.clone()),
        "audio.volume" => Some(config.audio.volume.to_string()),
        "audio.max_playback_queue_items" => Some(config.audio.max_playback_queue_items.to_string()),

        "model.cache_dir" => Some(config.model.cache_dir.display().to_string()),
        "model.revision" => Some(config.model.revision.clone()),
        "model.hashes_path" => Some(
            config
                .model
                .hashes_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "unset".to_string()),
        ),
        "model.inference_steps" => Some(config.model.inference_steps.to_string()),
        "model.download_connect_timeout_secs" => {
            Some(config.model.download_connect_timeout_secs.to_string())
        }
        "model.download_timeout_secs" => Some(config.model.download_timeout_secs.to_string()),

        "tts.max_text_length" => Some(config.tts.max_text_length.to_string()),
        "tts.max_chunk_chars" => Some(config.tts.max_chunk_chars.to_string()),
        "tts.max_body_bytes" => Some(config.tts.max_body_bytes.to_string()),
        "tts.openai_max_body_bytes" => Some(config.tts.openai_max_body_bytes.to_string()),
        "tts.queue_max_body_bytes" => Some(config.tts.queue_max_body_bytes.to_string()),
        "tts.admin_max_body_bytes" => Some(config.tts.admin_max_body_bytes.to_string()),

        "inference.max_concurrent" => Some(config.inference.max_concurrent.to_string()),
        "inference.max_pending" => Some(config.inference.max_pending.to_string()),

        "rate_limit.requests" => Some(config.rate_limit.requests.to_string()),
        "rate_limit.window_secs" => Some(config.rate_limit.window_secs.to_string()),
        "rate_limit.burst" => Some(config.rate_limit.burst.to_string()),

        "logging.filter" => Some(
            config
                .logging
                .filter
                .clone()
                .unwrap_or_else(|| "null".into()),
        ),
        "logging.level" => Some(config.logging.level.clone()),
        "logging.to_file" => Some(config.logging.to_file.to_string()),
        "logging.to_stdout" => Some(config.logging.to_stdout.to_string()),

        "paths.logs" => Some(config.paths.logs.clone()),
        "paths.audio" => Some(
            config
                .paths
                .audio
                .clone()
                .unwrap_or_else(|| "unset".to_string()),
        ),
        "paths.temp_audio" => Some(config.paths.temp_audio.clone()),
        "paths.token_store" => Some(config.paths.token_store.clone()),

        "security.trust_proxy" => Some(config.security.trust_proxy.to_string()),
        "security.trusted_proxies" => Some(config.security.trusted_proxies.join(", ")),
        "security.cookie_secure" => Some(config.security.cookie_secure.to_string()),
        "security.enable_hsts" => Some(config.security.enable_hsts.to_string()),
        "security.allow_insecure_remote" => Some(config.security.allow_insecure_remote.to_string()),

        _ => None,
    }
}

/// Write the value at a dotted path. `type_name` comes
/// from the schema and controls how the raw string is
/// parsed.
pub fn set_by_path(
    config: &mut config::AppConfig,
    path: &str,
    raw: &str,
    type_name: &str,
) -> Result<(), String> {
    match path {
        "setup_complete" => {
            config.setup_complete = parse_bool(raw)?;
        }

        "server.bind" => {
            config.server.bind = raw
                .parse()
                .map_err(|_| format!("server.bind: invalid IP address '{raw}'"))?;
        }
        "server.port" => {
            config.server.port = parse_typed(raw, type_name, "server.port")?;
        }
        "server.auth_mode" => {
            config.server.auth_mode = match raw {
                "local" => AuthMode::Local,
                "token" => AuthMode::Token,
                "none" => AuthMode::None,
                other => {
                    return Err(format!(
                        "server.auth_mode: unknown mode '{other}' \
                         (expected local, token, or none)"
                    ));
                }
            };
        }
        "server.request_timeout_secs" => {
            config.server.request_timeout_secs = parse_typed(raw, type_name, path)?;
        }

        "admin.username" => config.admin.username = raw.to_string(),
        "admin.session_expiry_secs" => {
            config.admin.session_expiry_secs = parse_typed(raw, type_name, path)?;
        }
        "admin.enable_sample_token" => {
            config.admin.enable_sample_token = parse_bool(raw)?;
        }

        "audio.output_device" => config.audio.output_device = raw.to_string(),
        "audio.volume" => {
            config.audio.volume = parse_typed(raw, type_name, path)?;
        }
        "audio.max_playback_queue_items" => {
            config.audio.max_playback_queue_items = parse_typed(raw, type_name, path)?;
        }

        "model.cache_dir" => {
            config.model.cache_dir = std::path::PathBuf::from(raw);
        }
        "model.revision" => config.model.revision = raw.to_string(),
        "model.hashes_path" => {
            if raw == "unset" || raw.is_empty() {
                config.model.hashes_path = None;
            } else {
                config.model.hashes_path = Some(std::path::PathBuf::from(raw));
            }
        }
        "model.inference_steps" => {
            config.model.inference_steps = parse_typed(raw, type_name, path)?;
        }
        "model.download_connect_timeout_secs" => {
            config.model.download_connect_timeout_secs = parse_typed(raw, type_name, path)?;
        }
        "model.download_timeout_secs" => {
            config.model.download_timeout_secs = parse_typed(raw, type_name, path)?;
        }

        "tts.max_text_length" => {
            config.tts.max_text_length = parse_typed(raw, type_name, path)?;
        }
        "tts.max_chunk_chars" => {
            config.tts.max_chunk_chars = parse_typed(raw, type_name, path)?;
        }
        "tts.max_body_bytes" => {
            config.tts.max_body_bytes = parse_typed(raw, type_name, path)?;
        }
        "tts.openai_max_body_bytes" => {
            config.tts.openai_max_body_bytes = parse_typed(raw, type_name, path)?;
        }
        "tts.queue_max_body_bytes" => {
            config.tts.queue_max_body_bytes = parse_typed(raw, type_name, path)?;
        }
        "tts.admin_max_body_bytes" => {
            config.tts.admin_max_body_bytes = parse_typed(raw, type_name, path)?;
        }

        "inference.max_concurrent" => {
            config.inference.max_concurrent = parse_typed(raw, type_name, path)?;
        }
        "inference.max_pending" => {
            config.inference.max_pending = parse_typed(raw, type_name, path)?;
        }

        "rate_limit.requests" => {
            config.rate_limit.requests = parse_typed(raw, type_name, path)?;
        }
        "rate_limit.window_secs" => {
            config.rate_limit.window_secs = parse_typed(raw, type_name, path)?;
        }
        "rate_limit.burst" => {
            config.rate_limit.burst = parse_typed(raw, type_name, path)?;
        }

        "logging.filter" => {
            config.logging.filter = if raw == "null" {
                None
            } else {
                Some(raw.to_string())
            }
        }
        "logging.level" => config.logging.level = raw.to_string(),
        "logging.to_file" => config.logging.to_file = parse_bool(raw)?,
        "logging.to_stdout" => config.logging.to_stdout = parse_bool(raw)?,

        "paths.logs" => config.paths.logs = raw.to_string(),
        "paths.audio" => {
            if raw == "unset" || raw.is_empty() {
                config.paths.audio = None;
            } else {
                config.paths.audio = Some(raw.to_string());
            }
        }
        "paths.temp_audio" => config.paths.temp_audio = raw.to_string(),
        "paths.token_store" => config.paths.token_store = raw.to_string(),

        "security.trust_proxy" => config.security.trust_proxy = parse_bool(raw)?,
        "security.trusted_proxies" => {
            config.security.trusted_proxies = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
        }
        "security.cookie_secure" => config.security.cookie_secure = parse_bool(raw)?,
        "security.enable_hsts" => config.security.enable_hsts = parse_bool(raw)?,
        "security.allow_insecure_remote" => {
            config.security.allow_insecure_remote = parse_bool(raw)?;
        }

        // Top-level `version` and every unknown path are
        // rejected: the schema is the source of truth for
        // what may be changed.
        _ => return Err(format!("{path}: not a settable configuration path")),
    }
    Ok(())
}

fn auth_mode_str(mode: AuthMode) -> &'static str {
    match mode {
        AuthMode::Local => "local",
        AuthMode::Token => "token",
        AuthMode::None => "none",
    }
}

fn parse_bool(raw: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        other => Err(format!("expected a boolean, got '{other}'")),
    }
}

/// Parse `raw` into `T`, rejecting non-numeric input so
/// malformed CLI values fail closed instead of defaulting.
fn parse_typed<T>(raw: &str, type_name: &str, path: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match type_name {
        "integer" | "number" => raw
            .trim()
            .parse::<T>()
            .map_err(|e| format!("{path}: invalid value '{raw}': {e}")),
        _ => Err(format!("{path}: cannot parse '{raw}' as {type_name}")),
    }
}
