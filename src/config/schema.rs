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
    add(SettingSchema {
        path: "tts.normalization.enabled".to_string(),
        description: "Enable text normalization before synthesis; false forces mode off"
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
        path: "tts.normalization.mode".to_string(),
        description: "Default normalization mode: auto (conservative number/date/currency rewriting) or off (text spoken exactly as sent)".to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: Some(vec!["auto".to_string(), "off".to_string()]),
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "tts.normalization.default_locale".to_string(),
        description:
            "Default pronunciation locale: auto (per-language default) or a supported locale tag"
                .to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: Some(vec![
            "auto".to_string(),
            "es-ES".to_string(),
            "es-MX".to_string(),
            "es-PE".to_string(),
            "en-US".to_string(),
            "en-GB".to_string(),
            "pt-BR".to_string(),
            "pt-PT".to_string(),
            "fr-FR".to_string(),
            "de-DE".to_string(),
            "it-IT".to_string(),
            "ru-RU".to_string(),
            "zh-CN".to_string(),
            "zh-TW".to_string(),
            "ja-JP".to_string(),
            "ko-KR".to_string(),
        ]),
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "tts.normalization.reading_style".to_string(),
        description:
            "How numbers and clock times are read: precise (compatibility), natural (conversational) or digits (digit by digit)"
                .to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: Some(vec![
            "precise".to_string(),
            "natural".to_string(),
            "digits".to_string(),
        ]),
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "tts.normalization.report_ambiguities".to_string(),
        description:
            "Report ambiguous expressions (for example 01/02/2026) as warnings instead of guessing"
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
        path: "tts.normalization.max_expansion_ratio".to_string(),
        description: "Maximum spoken-to-original length ratio".to_string(),
        type_name: "number",
        apply: ApplyStrategy::Hot,
        min: Some(1.0),
        max: Some(16.0),
        allowed_values: None,
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "tts.normalization.max_normalized_chars".to_string(),
        description: "Maximum spoken text length after normalization (characters)".to_string(),
        type_name: "integer",
        apply: ApplyStrategy::Hot,
        min: Some(1.0),
        max: Some(200000.0),
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
    for (path, description, max) in [
        (
            "voices.max_profiles",
            "Maximum stored voice profiles (also bounded by manifest byte limit)",
            4096.0,
        ),
        (
            "inference.generation_deadline_secs",
            "Cooperative inference deadline, independent of HTTP timeout",
            86400.0,
        ),
        (
            "inference.job_retention_secs",
            "Completed job and private audio retention in seconds",
            86400.0,
        ),
        (
            "inference.max_retained_jobs",
            "Bound on retained synthesis job metadata",
            1024.0,
        ),
    ] {
        add(SettingSchema {
            path: path.into(),
            description: description.into(),
            type_name: "integer",
            apply: ApplyStrategy::Hot,
            min: Some(1.0),
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

    add(SettingSchema {
        path: "providers.pocket_tts.enabled".to_string(),
        description: "Enable the Pocket TTS voice-cloning provider (default false)".to_string(),
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
        path: "model.backend".to_string(),
        description: "Supertonic inference engine: auto (candle when compiled in, else ort), ort, candle-cpu or candle-vulkan"
            .to_string(),
        type_name: "string",
        apply: ApplyStrategy::ReloadModel,
        min: None,
        max: None,
        allowed_values: Some(vec![
            "auto".to_string(),
            "ort".to_string(),
            "candle-cpu".to_string(),
            "candle-vulkan".to_string(),
        ]),
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    for provider in ["qwen3_candle", "kyutai_candle"] {
        add(SettingSchema {
            path: format!("providers.{provider}.download"),
            description: format!(
                "Download and SHA-256 verify the pinned {provider} bundle into model_dir (false: use files you manage)"
            ),
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
            path: format!("providers.{provider}.enabled"),
            description: format!("Enable the {provider} voice provider (default false)"),
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
            path: format!("providers.{provider}.device"),
            description: format!("{provider} device: auto (Vulkan when compiled in), cpu or vulkan"),
            type_name: "string",
            apply: ApplyStrategy::Hot,
            min: None,
            max: None,
            allowed_values: Some(vec!["auto".to_string(), "cpu".to_string(), "vulkan".to_string()]),
            nullable: false,
            secret: false,
            allowed_values_source: None,
        });
    }
    for (path, desc) in [
        ("providers.qwen3_candle.model_dir", "Directory with the Qwen3-TTS Base safetensors bundle (config.json, model.safetensors, speech_tokenizer/)"),
        ("providers.kyutai_candle.model_dir", "Directory with the Kyutai TTS 1.6B en/fr bundle"),
        ("providers.kyutai_candle.voices_dir", "Directory with kyutai/tts-voices embeddings (<voice>.safetensors)"),
        ("providers.kyutai_candle.default_voice", "Kyutai voice used when a request does not name one"),
    ] {
        add(SettingSchema {
            path: path.to_string(),
            description: desc.to_string(),
            type_name: "string",
            apply: ApplyStrategy::Hot,
            min: None,
            max: None,
            allowed_values: None,
            nullable: false,
            secret: false,
            allowed_values_source: None,
        });
    }
    add(SettingSchema {
        path: "providers.qwen3_gguf.enabled".to_string(),
        description: "Enable the Qwen3-TTS GGUF/ONNX voice-cloning provider (default false)"
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
        path: "providers.qwen3_gguf.allow_cpu_fallback".to_string(),
        description:
            "Allow an explicit device=vulkan/cuda that cannot initialize to fall back to CPU \
             with a reported reason (device=auto always may)"
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
    for (path, desc) in [
        (
            "providers.qwen3_gguf.model_dir",
            "Directory holding the verified Qwen3-TTS 1.7B GGUF/ONNX bundle",
        ),
        (
            "providers.pocket_tts.model_dir",
            "Directory holding the verified Pocket TTS ONNX bundles",
        ),
        (
            "providers.pocket_tts.model_id",
            "Pocket TTS model identifier (pocket-tts-en/es/pt/de/it, pocket-tts-fr-24l)",
        ),
        (
            "providers.pocket_tts.revision",
            "Pinned Pocket TTS export revision",
        ),
        (
            "voices.store_dir",
            "Directory for voice profiles and reference audio",
        ),
    ] {
        add(SettingSchema {
            path: path.to_string(),
            description: desc.to_string(),
            type_name: "string",
            apply: ApplyStrategy::Hot,
            min: None,
            max: None,
            allowed_values: if path == "providers.pocket_tts.model_id" {
                Some(
                    sonicboom::tts::providers::pocket_tts::POCKET_TTS_MODELS
                        .iter()
                        .map(|s| (*s).to_string())
                        .collect(),
                )
            } else {
                None
            },
            nullable: false,
            secret: false,
            allowed_values_source: None,
        });
    }
    add(SettingSchema {
        path: "providers.qwen3_gguf.device".to_string(),
        description:
            "Qwen3-TTS GGUF/ONNX device: cpu, auto, vulkan (qwen-gguf-vulkan build), or cuda (qwen-gguf-cuda build)"
                .to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: Some(vec![
            "cpu".to_string(),
            "auto".to_string(),
            "vulkan".to_string(),
            "cuda".to_string(),
        ]),
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });
    add(SettingSchema {
        path: "providers.pocket_tts.device".to_string(),
        description: "Pocket TTS device: cpu or auto (GPU is unsupported for this provider)"
            .to_string(),
        type_name: "string",
        apply: ApplyStrategy::Hot,
        min: None,
        max: None,
        allowed_values: Some(vec!["cpu".to_string(), "auto".to_string()]),
        nullable: false,
        secret: false,
        allowed_values_source: None,
    });

    for (path, desc, min, max) in [
        (
            "providers.qwen3_gguf.max_concurrent",
            "Qwen3-TTS GGUF/ONNX uses one serialized model-owning worker",
            1.0,
            1.0,
        ),
        (
            "providers.qwen3_gguf.cache_entries",
            "Qwen3-TTS GGUF/ONNX voice-conditioning cache entries",
            1.0,
            1024.0,
        ),
        (
            "providers.qwen3_gguf.max_tokens",
            "Maximum Qwen3-TTS GGUF/ONNX acoustic frames per synthesis",
            1.0,
            1024.0,
        ),
        (
            "providers.qwen3_gguf.threads",
            "Qwen3-TTS GGUF/ONNX worker threads",
            1.0,
            256.0,
        ),
        (
            "providers.qwen3_gguf.gpu_layers",
            "Qwen3-TTS GGUF/ONNX GPU layer offload (-1 offloads all layers)",
            -1.0,
            1024.0,
        ),
        (
            "providers.pocket_tts.max_concurrent",
            "Pocket TTS uses one serialized mutable model",
            1.0,
            1.0,
        ),
        (
            "providers.pocket_tts.cache_entries",
            "Pocket TTS voice-embedding cache entries",
            1.0,
            1024.0,
        ),
        (
            "providers.pocket_tts.lsd_steps",
            "Pocket TTS flow-matching integration steps",
            1.0,
            8.0,
        ),
        (
            "providers.pocket_tts.cpu_threads",
            "Pocket TTS ORT intra-op threads",
            1.0,
            32.0,
        ),
        (
            "voices.max_upload_bytes",
            "Maximum voice reference upload size in bytes",
            1024.0,
            67_108_864.0,
        ),
    ] {
        add(SettingSchema {
            path: path.to_string(),
            description: desc.to_string(),
            type_name: "integer",
            apply: ApplyStrategy::Hot,
            min: Some(min),
            max: Some(max),
            allowed_values: None,
            nullable: false,
            secret: false,
            allowed_values_source: None,
        });
    }
    add(SettingSchema {
        path: "providers.pocket_tts.temperature".to_string(),
        description: "Pocket TTS sampling temperature (0.0 disables noise)".to_string(),
        type_name: "number",
        apply: ApplyStrategy::Hot,
        min: Some(0.0),
        max: Some(2.0),
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
    fn schema_covers_every_reloadable_config_path() {
        // The studio UI renders its enable toggles and setting fields from
        // this schema: any reloadable key missing here is silently
        // uneditable (e.g. a provider with no Enable button).
        let map = schema();
        for path in super::super::loader::ALL_CONFIG_PATHS {
            assert!(map.contains_key(*path), "schema missing {path}");
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
