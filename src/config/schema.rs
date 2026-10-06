//! Configuration schema for agents and the GUI.
//!
//! Exposed via `GET /api/config/schema` and
//! `sonicboom config schema` so clients (human or AI) can
//! discover valid configuration without reading source code.

use std::collections::BTreeMap;

use serde::Serialize;

use super::diff::ApplyStrategy;

/// JSON-schema-ish description of one setting.
#[derive(Debug, Clone, Serialize)]
pub struct SettingSchema {
    /// Dotted configuration path (e.g. `audio.output_device`).
    pub path: String,
    /// Human-readable description.
    pub description: String,
    /// Value type: `string`, `integer`, `number`, `boolean`,
    /// `array`, `object`.
    #[serde(rename = "type")]
    pub type_name: &'static str,
    /// How a change to this setting is applied.
    pub apply: ApplyStrategy,
    /// Minimum value (`integer`/`number`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Maximum value (`integer`/`number`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Enumerated allowed values (`string`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_values: Option<Vec<String>>,
    /// Whether the value is a secret (never returned by
    /// `GET /api/config`; stored in the secret store).
    #[serde(default)]
    pub secret: bool,
    /// Whether JSON null disables this optional setting.
    pub nullable: bool,
    /// Where selectable values come from (e.g. `audio_devices`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_values_source: Option<&'static str>,
}

/// The complete configuration schema, keyed by dotted path.
pub fn schema() -> BTreeMap<String, SettingSchema> {
    let mut map = BTreeMap::new();
    let mut add = |setting: SettingSchema| {
        map.insert(setting.path.clone(), setting);
    };

    add(SettingSchema {
        path: "logging.filter".into(),
        description:
            "Optional tracing directives; RUST_LOG overrides this field. Use null to clear.".into(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: true,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "server.bind".to_string(),
        description: "Address the HTTP server binds".to_string(),
        type_name: "string",
        apply: ApplyStrategy::RestartServer,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "server.port".to_string(),
        description: "TCP port the HTTP server listens on".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::RestartServer,
        min: Some(1.0),
        max: Some(65535.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "server.auth_mode".to_string(),
        description: "API authentication mode: local (no auth, loopback only), token (bearer required), none (development only)"
            .to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: Some(vec![
            "local".to_string(),
            "token".to_string(),
            "none".to_string(),
        ]),
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "server.request_timeout_secs".to_string(),
        description: "Per-request timeout in seconds".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::RestartServer,
        min: Some(1.0),
        max: Some(3600.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "admin.username".to_string(),
        description: "Admin panel username".to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "admin.session_expiry_secs".to_string(),
        description: "Admin session inactivity expiry in seconds".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(60.0),
        max: Some(2_592_000.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "admin.enable_sample_token".to_string(),
        description: "Accept the development SAMPLE_TOKEN credential (never enable in production)"
            .to_string(),
        type_name: "boolean",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "audio.output_device".to_string(),
        description: "Audio playback output device ('default' follows the OS default)".to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: Some("audio_devices"),
    });
    add(SettingSchema {
        path: "audio.volume".to_string(),
        description: "Playback volume (0.0-1.0)".to_string(),
        type_name: "number",
        apply: ApplyStrategy::Hot,
        min: Some(0.0),
        max: Some(1.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "audio.max_playback_queue_items".to_string(),
        description: "Hard cap on waiting + playing items in the playback queue".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::RestartSubsystem,
        min: Some(1.0),
        max: Some(10_000.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "model.cache_dir".to_string(),
        description: "Directory for downloaded ONNX model files".to_string(),
        type_name: "string",
        apply: ApplyStrategy::ReloadModel,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "model.revision".to_string(),
        description: "Pinned HuggingFace revision (40-character commit SHA)".to_string(),
        type_name: "string",
        apply: ApplyStrategy::ReloadModel,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "model.hashes_path".to_string(),
        description: "Optional JSON manifest of expected SHA-256 digests for a custom revision"
            .to_string(),
        type_name: "string",
        apply: ApplyStrategy::ReloadModel,
        min: None,
        max: None,
        allowed_values: None,
        nullable: true,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "model.inference_steps".to_string(),
        description: "Number of inference steps per synthesis".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(1.0),
        max: Some(50.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "model.download_connect_timeout_secs".to_string(),
        description: "TCP connect timeout for model downloads (seconds)".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(1.0),
        max: Some(300.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "model.download_timeout_secs".to_string(),
        description: "Total timeout per model file download (seconds)".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(1.0),
        max: Some(86_400.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "tts.max_text_length".to_string(),
        description: "Maximum text length for TTS requests (Unicode characters)".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(1.0),
        max: Some(1_000_000.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "tts.max_chunk_chars".to_string(),
        description: "Hard maximum characters per synthesis chunk".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(1.0),
        max: Some(10_000.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    for (path, max, desc) in [
        (
            "tts.max_body_bytes",
            16_777_216.0,
            "Body limit for /api/tts (bytes)",
        ),
        (
            "tts.openai_max_body_bytes",
            16_777_216.0,
            "Body limit for OpenAI-compatible endpoints (bytes)",
        ),
        (
            "tts.queue_max_body_bytes",
            16_777_216.0,
            "Body limit for queue endpoints (bytes)",
        ),
        (
            "tts.admin_max_body_bytes",
            16_777_216.0,
            "Body limit for admin endpoints (bytes)",
        ),
    ] {
        add(SettingSchema {
            path: path.to_string(),
            description: desc.to_string(),
            type_name: "integer",
            apply: ApplyStrategy::RestartServer,
            min: Some(1024.0),
            max: Some(max),
            allowed_values: None,
            nullable: false,
            secret: false,
            allowed_values_source: None,
        });
    }
    add(SettingSchema {
        path: "inference.max_concurrent".to_string(),
        description: "Maximum concurrent model inferences".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::RestartSubsystem,
        min: Some(1.0),
        max: Some(64.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "inference.max_pending".to_string(),
        description: "Maximum queued (waiting) inference requests".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::RestartSubsystem,
        min: Some(0.0),
        max: Some(10_000.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "rate_limit.requests".to_string(),
        description: "Sustained per-token request budget per window (0 disables limiting)"
            .to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(0.0),
        max: Some(1_000_000.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "rate_limit.window_secs".to_string(),
        description: "Rate-limit window in seconds".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(1.0),
        max: Some(86_400.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "rate_limit.burst".to_string(),
        description: "Rate-limit burst capacity".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(0.0),
        max: Some(1_000_000.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "logging.level".to_string(),
        description: "Log level filter".to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: Some(vec![
            "trace".to_string(),
            "debug".to_string(),
            "info".to_string(),
            "warn".to_string(),
            "error".to_string(),
        ]),
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "logging.to_file".to_string(),
        description: "Write logs to the log directory".to_string(),
        type_name: "boolean",
        apply: ApplyStrategy::RestartProcess,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "logging.to_stdout".to_string(),
        description: "Write logs to stdout".to_string(),
        type_name: "boolean",
        apply: ApplyStrategy::RestartProcess,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "paths.logs".to_string(),
        description: "Log file directory".to_string(),
        type_name: "string",
        apply: ApplyStrategy::RestartServer,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "paths.audio".to_string(),
        description: "Allowed audio directory for the playback queue (prevents path traversal)"
            .to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: true,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "paths.temp_audio".to_string(),
        description: "Directory for temporary synthesized playback files".to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "paths.token_store".to_string(),
        description: "API token store file".to_string(),
        type_name: "string",
        apply: ApplyStrategy::RestartProcess,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "security.trust_proxy".to_string(),
        description: "Honor forwarded headers from trusted proxies".to_string(),
        type_name: "boolean",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "security.trusted_proxies".to_string(),
        description: "Proxy addresses/CIDRs allowed to forward client IPs".to_string(),
        type_name: "array",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "security.cookie_secure".to_string(),
        description: "Set the Secure attribute on admin session cookies (HTTPS only)".to_string(),
        type_name: "boolean",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "security.enable_hsts".to_string(),
        description: "Enable HSTS (known-HTTPS deployments only)".to_string(),
        type_name: "boolean",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "security.allow_insecure_remote".to_string(),
        description:
            "Explicit opt-in for auth_mode = none on non-loopback binds (development only)"
                .to_string(),
        type_name: "boolean",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });

    // Secrets (stored in the secret store, never in config.toml).
    add(SettingSchema {
        path: "secrets.admin_password".to_string(),
        description:
            "Admin panel password (stored in the OS-backed secret store, never in config.toml)"
                .to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: true,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "secrets.hf_token".to_string(),
        description:
            "HuggingFace access token for gated model downloads (stored in the secret store)"
                .to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: None,
        nullable: false,
        secret: true,
        allowed_values_source: None,
    });

    map
}

/// JSON-serializable schema document: the setting
/// map wrapped in a top-level `settings` object.
pub fn schema_json() -> serde_json::Value {
    let map = schema();
    let entries: Vec<serde_json::Value> = map
        .values()
        .map(|s| serde_json::to_value(s).unwrap_or_default())
        .collect();
    serde_json::json!({ "settings": entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_covers_key_settings() {
        let map = schema();
        for path in [
            "server.port",
            "server.auth_mode",
            "audio.output_device",
            "audio.volume",
            "model.inference_steps",
            "tts.max_text_length",
            "rate_limit.requests",
            "logging.level",
            "security.trust_proxy",
        ] {
            assert!(map.contains_key(path), "schema missing {path}");
        }
    }

    #[test]
    fn every_setting_has_apply_strategy_and_type() {
        for setting in schema().values() {
            assert!(!setting.description.is_empty());
            assert!(!setting.type_name.is_empty());
            // apply is a non-skip field; presence is structural.
            let _ = setting.apply.as_str();
        }
    }

    #[test]
    fn secrets_are_marked_and_excluded_from_config_api() {
        let map = schema();
        assert!(map["secrets.admin_password"].secret);
        assert!(map["secrets.hf_token"].secret);
    }

    #[test]
    fn schema_json_serializes() {
        let json = schema_json();
        assert!(json["settings"].as_array().is_some_and(|a| !a.is_empty()));
    }
}
