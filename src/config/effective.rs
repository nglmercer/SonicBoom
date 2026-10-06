//! Effective configuration: the merged view of defaults,
//! `config.toml`, environment variables, and runtime overrides,
//! with the source of every value.

use std::collections::HashMap;

use super::model::AppConfig;

/// Where a configuration value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSource {
    /// Compiled-in default.
    Default,
    /// `config.toml`.
    Toml,
    /// Environment variable.
    Environment,
    /// Runtime override (GUI/CLI/API/MCP).
    Runtime,
}

impl ConfigSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ConfigSource::Default => "default",
            ConfigSource::Toml => "config.toml",
            ConfigSource::Environment => "environment",
            ConfigSource::Runtime => "runtime",
        }
    }
}

/// A single effective value plus its source.
///
/// Part of the effective-configuration API (spec §21);
/// the manager exposes sources through
/// [`EffectiveConfig::source_of`] and this typed wrapper
/// is available to consumers that need per-value
/// provenance.
#[allow(dead_code)]
#[derive(Debug, Clone, serde::Serialize)]
pub struct EffectiveValue<T> {
    pub value: T,
    pub source: ConfigSource,
}

/// The effective configuration: the live values plus a map from
/// dotted path to the source that provided each value.
#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    pub config: AppConfig,
    pub sources: HashMap<String, ConfigSource>,
}

impl EffectiveConfig {
    /// Source of a dotted configuration path (`None` when the
    /// path is unknown).
    pub fn source_of(&self, path: &str) -> Option<ConfigSource> {
        self.sources.get(path).copied()
    }

    /// JSON-serializable view: `{ "path": { "value": ..., "source": ... } }`.
    pub fn describe(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for path in super::loader::ALL_CONFIG_PATHS {
            let source = self
                .sources
                .get(*path)
                .copied()
                .unwrap_or(ConfigSource::Default);
            let value = json_value_at(&self.config, path);
            map.insert(
                (*path).to_string(),
                serde_json::json!({
                    "value": value,
                    "source": source.as_str(),
                }),
            );
        }
        serde_json::Value::Object(map)
    }
}

/// Extract the JSON value at a dotted path (best-effort; unknown
/// paths yield `null`).
fn json_value_at(config: &AppConfig, path: &str) -> serde_json::Value {
    let full = serde_json::to_value(config).unwrap_or_default();
    let mut cursor = &full;
    for part in path.split('.') {
        cursor = match cursor.get(part) {
            Some(next) => next,
            None => return serde_json::Value::Null,
        };
    }
    cursor.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_covers_all_known_paths() {
        let mut sources = HashMap::new();
        sources.insert("logging.level".to_string(), ConfigSource::Environment);
        let effective = EffectiveConfig {
            config: AppConfig::default(),
            sources,
        };
        let described = effective.describe();
        assert_eq!(
            described["logging.level"]["value"],
            serde_json::json!("info")
        );
        assert_eq!(
            described["logging.level"]["source"],
            serde_json::json!("environment")
        );
        assert_eq!(described["server.port"]["value"], serde_json::json!(17842));
        assert_eq!(
            described["server.port"]["source"],
            serde_json::json!("default")
        );
    }

    #[test]
    fn json_value_at_navigates_sections() {
        let mut config = AppConfig::default();
        config.audio.output_device = "CABLE Input".to_string();
        assert_eq!(
            json_value_at(&config, "audio.output_device"),
            serde_json::json!("CABLE Input")
        );
        assert_eq!(
            json_value_at(&config, "audio.missing.key"),
            serde_json::Value::Null
        );
    }
}
