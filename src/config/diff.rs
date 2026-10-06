//! Configuration diffs, change classification, and events.
//!
//! Every configuration change is expressed as a set of
//! [`ConfigChange`] values, each carrying the smallest
//! component that must react to it (see [`ApplyStrategy`]).

use std::net::SocketAddr;
use std::path::PathBuf;

use serde::Serialize;

use super::model::{AppConfig, AuthMode, RateLimitConfig};

/// Where a configuration change originated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeSource {
    Gui,
    Api,
    Cli,
    Mcp,
    Filesystem,
    Migration,
    FirstRun,
}

impl ChangeSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ChangeSource::Gui => "gui",
            ChangeSource::Api => "api",
            ChangeSource::Cli => "cli",
            ChangeSource::Mcp => "mcp",
            ChangeSource::Filesystem => "filesystem",
            ChangeSource::Migration => "migration",
            ChangeSource::FirstRun => "first_run",
        }
    }
}

/// How a configuration change is applied at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplyStrategy {
    /// Applied immediately to the live subsystem.
    Hot,
    /// The owning subsystem is replaced (e.g. inference gate).
    RestartSubsystem,
    /// Only the HTTP listener is restarted.
    RestartServer,
    /// The model is re-downloaded/validated and swapped.
    ReloadModel,
    /// A full process restart is required (avoid; currently unused).
    RestartProcess,
}

impl ApplyStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            ApplyStrategy::Hot => "hot",
            ApplyStrategy::RestartSubsystem => "restart_subsystem",
            ApplyStrategy::RestartServer => "restart_server",
            ApplyStrategy::ReloadModel => "reload_model",
            ApplyStrategy::RestartProcess => "restart_process",
        }
    }
}

/// A single semantic configuration difference.
#[derive(Debug, Clone, Serialize)]
pub enum ConfigChange {
    AudioOutputDevice {
        old: String,
        new: String,
    },
    Volume {
        old: f32,
        new: f32,
    },
    MaxPlaybackQueueItems {
        old: usize,
        new: usize,
    },
    LogLevel {
        old: String,
        new: String,
    },
    LogToFile {
        old: bool,
        new: bool,
    },
    LogToStdout {
        old: bool,
        new: bool,
    },
    InferenceSteps {
        old: usize,
        new: usize,
    },
    InferenceLimits {
        old: (usize, usize),
        new: (usize, usize),
    },
    RateLimit {
        old: RateLimitConfig,
        new: RateLimitConfig,
    },
    ServerAddress {
        old: SocketAddr,
        new: SocketAddr,
    },
    ServerRequestTimeout {
        old: u64,
        new: u64,
    },
    AuthMode {
        old: AuthMode,
        new: AuthMode,
    },
    TtsLimits {
        old: TtsLimitSnapshot,
        new: TtsLimitSnapshot,
    },
    ModelRevision {
        old: String,
        new: String,
    },
    ModelCacheDir {
        old: PathBuf,
        new: PathBuf,
    },
    ModelHashesPath {
        old: Option<PathBuf>,
        new: Option<PathBuf>,
    },
    /// A change that does not map to a dedicated variant yet
    /// (e.g. paths, admin, security settings that are read at
    /// request time or applied on restart).
    Other {
        path: String,
        old: serde_json::Value,
        new: serde_json::Value,
    },
}

/// Snapshot of the TTS limits that handlers read per request.
#[derive(Debug, Clone, Serialize)]
pub struct TtsLimitSnapshot {
    pub max_text_length: usize,
    pub max_chunk_chars: usize,
    pub max_body_bytes: usize,
    pub openai_max_body_bytes: usize,
    pub queue_max_body_bytes: usize,
    pub admin_max_body_bytes: usize,
}

impl TtsLimitSnapshot {
    pub fn from(config: &AppConfig) -> Self {
        Self {
            max_text_length: config.tts.max_text_length,
            max_chunk_chars: config.tts.max_chunk_chars,
            max_body_bytes: config.tts.max_body_bytes,
            openai_max_body_bytes: config.tts.openai_max_body_bytes,
            queue_max_body_bytes: config.tts.queue_max_body_bytes,
            admin_max_body_bytes: config.tts.admin_max_body_bytes,
        }
    }
}

impl ConfigChange {
    /// Dotted configuration path this change affects.
    pub fn path(&self) -> String {
        match self {
            ConfigChange::AudioOutputDevice { .. } => "audio.output_device".to_string(),
            ConfigChange::Volume { .. } => "audio.volume".to_string(),
            ConfigChange::MaxPlaybackQueueItems { .. } => {
                "audio.max_playback_queue_items".to_string()
            }
            ConfigChange::LogLevel { .. } => "logging.level".to_string(),
            ConfigChange::LogToFile { .. } => "logging.to_file".to_string(),
            ConfigChange::LogToStdout { .. } => "logging.to_stdout".to_string(),
            ConfigChange::InferenceSteps { .. } => "model.inference_steps".to_string(),
            ConfigChange::InferenceLimits { .. } => "inference".to_string(),
            ConfigChange::RateLimit { .. } => "rate_limit".to_string(),
            ConfigChange::ServerAddress { .. } => "server".to_string(),
            ConfigChange::ServerRequestTimeout { .. } => "server.request_timeout_secs".to_string(),
            ConfigChange::AuthMode { .. } => "server.auth_mode".to_string(),
            ConfigChange::TtsLimits { .. } => "tts".to_string(),
            ConfigChange::ModelRevision { .. } => "model.revision".to_string(),
            ConfigChange::ModelCacheDir { .. } => "model.cache_dir".to_string(),
            ConfigChange::ModelHashesPath { .. } => "model.hashes_path".to_string(),
            ConfigChange::Other { path, .. } => path.clone(),
        }
    }

    /// How this change is applied at runtime.
    pub fn strategy(&self) -> ApplyStrategy {
        match self {
            ConfigChange::AudioOutputDevice { .. }
            | ConfigChange::Volume { .. }
            | ConfigChange::LogLevel { .. }
            | ConfigChange::InferenceSteps { .. }
            | ConfigChange::RateLimit { .. }
            | ConfigChange::AuthMode { .. }
            | ConfigChange::InferenceLimits { .. } => ApplyStrategy::Hot,
            ConfigChange::TtsLimits { old, new }
                if old.max_body_bytes == new.max_body_bytes
                    && old.openai_max_body_bytes == new.openai_max_body_bytes
                    && old.queue_max_body_bytes == new.queue_max_body_bytes
                    && old.admin_max_body_bytes == new.admin_max_body_bytes =>
            {
                ApplyStrategy::Hot
            }
            ConfigChange::ServerRequestTimeout { .. } | ConfigChange::TtsLimits { .. } => {
                // Request timeout and body limits are
                // router layers: a listener restart
                // rebuilds the router and applies them.
                ApplyStrategy::RestartServer
            }
            ConfigChange::ServerAddress { .. } => ApplyStrategy::RestartServer,
            // The global tracing subscriber is initialized
            // once per process; changing log sinks requires
            // a restart (the level itself is hot-reloadable).
            ConfigChange::LogToFile { .. } | ConfigChange::LogToStdout { .. } => {
                ApplyStrategy::RestartProcess
            }
            ConfigChange::ModelRevision { .. }
            | ConfigChange::ModelCacheDir { .. }
            | ConfigChange::ModelHashesPath { .. } => ApplyStrategy::ReloadModel,
            ConfigChange::MaxPlaybackQueueItems { .. } => ApplyStrategy::RestartSubsystem,
            ConfigChange::Other { path, .. } => match path.as_str() {
                "security.cookie_secure" | "admin.session_expiry_secs" => {
                    ApplyStrategy::RestartServer
                }
                "paths.logs" | "paths.token_store" => ApplyStrategy::RestartProcess,
                "model.download_connect_timeout_secs" | "model.download_timeout_secs" => {
                    ApplyStrategy::ReloadModel
                }
                _ => ApplyStrategy::Hot,
            },
        }
    }
}

/// Compute the semantic differences between two configurations.
pub fn diff(old: &AppConfig, new: &AppConfig) -> Vec<ConfigChange> {
    let mut changes = Vec::new();

    if old.audio.output_device != new.audio.output_device {
        changes.push(ConfigChange::AudioOutputDevice {
            old: old.audio.output_device.clone(),
            new: new.audio.output_device.clone(),
        });
    }
    if old.audio.volume != new.audio.volume {
        changes.push(ConfigChange::Volume {
            old: old.audio.volume,
            new: new.audio.volume,
        });
    }
    if old.audio.max_playback_queue_items != new.audio.max_playback_queue_items {
        changes.push(ConfigChange::MaxPlaybackQueueItems {
            old: old.audio.max_playback_queue_items,
            new: new.audio.max_playback_queue_items,
        });
    }
    if old.logging.level != new.logging.level {
        changes.push(ConfigChange::LogLevel {
            old: old.logging.level.clone(),
            new: new.logging.level.clone(),
        });
    }
    if old.logging.to_file != new.logging.to_file {
        changes.push(ConfigChange::LogToFile {
            old: old.logging.to_file,
            new: new.logging.to_file,
        });
    }
    if old.logging.to_stdout != new.logging.to_stdout {
        changes.push(ConfigChange::LogToStdout {
            old: old.logging.to_stdout,
            new: new.logging.to_stdout,
        });
    }
    if old.model.inference_steps != new.model.inference_steps {
        changes.push(ConfigChange::InferenceSteps {
            old: old.model.inference_steps,
            new: new.model.inference_steps,
        });
    }
    if old.inference.max_concurrent != new.inference.max_concurrent
        || old.inference.max_pending != new.inference.max_pending
    {
        changes.push(ConfigChange::InferenceLimits {
            old: (old.inference.max_concurrent, old.inference.max_pending),
            new: (new.inference.max_concurrent, new.inference.max_pending),
        });
    }
    if old.rate_limit.requests != new.rate_limit.requests
        || old.rate_limit.window_secs != new.rate_limit.window_secs
        || old.rate_limit.burst != new.rate_limit.burst
    {
        changes.push(ConfigChange::RateLimit {
            old: old.rate_limit.clone(),
            new: new.rate_limit.clone(),
        });
    }
    let old_addr = SocketAddr::new(old.server.bind, old.server.port);
    let new_addr = SocketAddr::new(new.server.bind, new.server.port);
    if old_addr != new_addr {
        changes.push(ConfigChange::ServerAddress {
            old: old_addr,
            new: new_addr,
        });
    }
    if old.server.request_timeout_secs != new.server.request_timeout_secs {
        changes.push(ConfigChange::ServerRequestTimeout {
            old: old.server.request_timeout_secs,
            new: new.server.request_timeout_secs,
        });
    }
    if old.server.auth_mode != new.server.auth_mode {
        changes.push(ConfigChange::AuthMode {
            old: old.server.auth_mode,
            new: new.server.auth_mode,
        });
    }
    let old_tts = TtsLimitSnapshot::from(old);
    let new_tts = TtsLimitSnapshot::from(new);
    if old_tts.max_text_length != new_tts.max_text_length
        || old_tts.max_chunk_chars != new_tts.max_chunk_chars
        || old_tts.max_body_bytes != new_tts.max_body_bytes
        || old_tts.openai_max_body_bytes != new_tts.openai_max_body_bytes
        || old_tts.queue_max_body_bytes != new_tts.queue_max_body_bytes
        || old_tts.admin_max_body_bytes != new_tts.admin_max_body_bytes
    {
        changes.push(ConfigChange::TtsLimits {
            old: old_tts,
            new: new_tts,
        });
    }
    if old.model.revision != new.model.revision {
        changes.push(ConfigChange::ModelRevision {
            old: old.model.revision.clone(),
            new: new.model.revision.clone(),
        });
    }
    if old.model.cache_dir != new.model.cache_dir {
        changes.push(ConfigChange::ModelCacheDir {
            old: old.model.cache_dir.clone(),
            new: new.model.cache_dir.clone(),
        });
    }
    if old.model.hashes_path != new.model.hashes_path {
        changes.push(ConfigChange::ModelHashesPath {
            old: old.model.hashes_path.clone(),
            new: new.model.hashes_path.clone(),
        });
    }

    // Generic comparison for everything not covered above, so
    // API/CLI consumers still see the full set of differences.
    let old_json = serde_json::to_value(old).unwrap_or_default();
    let new_json = serde_json::to_value(new).unwrap_or_default();
    let mut covered = std::collections::HashSet::new();
    for change in &changes {
        let paths: Vec<String> = match change {
            ConfigChange::ServerAddress { .. } => vec!["server.bind".into(), "server.port".into()],
            ConfigChange::InferenceLimits { .. } => vec![
                "inference.max_concurrent".into(),
                "inference.max_pending".into(),
            ],
            ConfigChange::RateLimit { .. } => vec![
                "rate_limit.requests".into(),
                "rate_limit.window_secs".into(),
                "rate_limit.burst".into(),
            ],
            ConfigChange::TtsLimits { .. } => super::loader::ALL_CONFIG_PATHS
                .iter()
                .filter(|p| p.starts_with("tts."))
                .map(|p| (*p).to_string())
                .collect(),
            _ => vec![change.path()],
        };
        covered.extend(paths);
    }
    // `revision` is an internal write counter, not a
    // user setting: it changes on every write and must
    // not surface as a configuration change.
    covered.insert("revision".to_string());
    collect_uncovered(&old_json, &new_json, "", &covered, &mut changes);

    changes
}

fn collect_uncovered(
    old: &serde_json::Value,
    new: &serde_json::Value,
    prefix: &str,
    covered: &std::collections::HashSet<String>,
    changes: &mut Vec<ConfigChange>,
) {
    match (old, new) {
        (serde_json::Value::Object(old_map), serde_json::Value::Object(new_map)) => {
            let keys: std::collections::BTreeSet<_> =
                old_map.keys().chain(new_map.keys()).collect();
            for key in keys {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                if covered
                    .iter()
                    .any(|c| c == &path || path.starts_with(&format!("{c}.")))
                {
                    continue;
                }
                let (old_v, new_v) = (old_map.get(key), new_map.get(key));
                match (old_v, new_v) {
                    (Some(a), Some(b)) if a != b => {
                        collect_uncovered(a, b, &path, covered, changes);
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        changes.push(ConfigChange::Other {
                            path,
                            old: old_v.cloned().unwrap_or(serde_json::Value::Null),
                            new: new_v.cloned().unwrap_or(serde_json::Value::Null),
                        });
                    }
                    _ => {}
                }
            }
        }
        _ => {
            if old != new && !prefix.is_empty() {
                changes.push(ConfigChange::Other {
                    path: prefix.to_string(),
                    old: old.clone(),
                    new: new.clone(),
                });
            }
        }
    }
}

/// Event published by the [`ConfigManager`](super::manager::ConfigManager)
/// on every committed configuration change.
#[derive(Debug, Clone, Serialize)]
pub enum ConfigEvent {
    /// A validated configuration was installed.
    Updated {
        old_revision: u64,
        new_revision: u64,
        changes: Vec<ConfigChange>,
        config: AppConfig,
        source: ChangeSource,
    },
    /// A candidate configuration (usually from the filesystem
    /// watcher) was rejected; the previous configuration
    /// remains active.
    ReloadRejected { error: String, source: ChangeSource },
    /// A configuration change was committed but the runtime
    /// subsystem failed to apply it; the configured value
    /// differs from the active runtime value.
    ApplyFailed { path: String, error: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::IpAddr;

    fn config() -> AppConfig {
        AppConfig::default()
    }

    #[test]
    fn grouped_changes_do_not_hide_other_fields_or_duplicate_them() {
        let old = config();
        let mut new = old.clone();
        new.server.port = 19000;
        new.server.request_timeout_secs = 300;
        new.model.inference_steps = 10;
        new.model.download_timeout_secs = 300;
        new.admin.username = "operator".into();
        let changes = diff(&old, &new);
        let paths: Vec<_> = changes.iter().map(ConfigChange::path).collect();
        assert_eq!(paths.len(), 5, "{paths:?}");
        assert!(paths.contains(&"server.request_timeout_secs".to_string()));
        assert!(paths.contains(&"model.download_timeout_secs".to_string()));
        assert!(paths.contains(&"admin.username".to_string()));
    }

    #[test]
    fn identical_configs_produce_no_changes() {
        assert!(diff(&config(), &config()).is_empty());
    }

    #[test]
    fn audio_device_change_is_hot() {
        let mut new = config();
        new.audio.output_device = "CABLE Input".to_string();
        let changes = diff(&config(), &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path(), "audio.output_device");
        assert_eq!(changes[0].strategy(), ApplyStrategy::Hot);
    }

    #[test]
    fn port_change_restarts_server() {
        let mut new = config();
        new.server.port = 19000;
        let changes = diff(&config(), &new);
        assert!(
            changes
                .iter()
                .any(|c| matches!(c, ConfigChange::ServerAddress { .. }))
        );
        let addr_change = changes
            .iter()
            .find(|c| matches!(c, ConfigChange::ServerAddress { .. }))
            .unwrap();
        assert_eq!(addr_change.strategy(), ApplyStrategy::RestartServer);
    }

    #[test]
    fn rate_limit_change_is_hot() {
        let mut new = config();
        new.rate_limit.requests = 500;
        let changes = diff(&config(), &new);
        assert!(
            changes
                .iter()
                .any(|c| matches!(c, ConfigChange::RateLimit { .. }))
        );
    }

    #[test]
    fn model_revision_change_reloads_model() {
        let mut new = config();
        new.model.revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        new.model.hashes_path = Some(PathBuf::from("hashes.json"));
        let changes = diff(&config(), &new);
        let rev = changes
            .iter()
            .find(|c| matches!(c, ConfigChange::ModelRevision { .. }))
            .unwrap();
        assert_eq!(rev.strategy(), ApplyStrategy::ReloadModel);
    }

    #[test]
    fn inference_steps_change_is_hot() {
        let mut new = config();
        new.model.inference_steps = 10;
        let changes = diff(&config(), &new);
        assert!(
            changes
                .iter()
                .any(|c| matches!(c, ConfigChange::InferenceSteps { .. }))
        );
    }

    #[test]
    fn tts_limit_change_is_hot() {
        let mut new = config();
        new.tts.max_text_length = 20_000;
        let changes = diff(&config(), &new);
        assert!(
            changes
                .iter()
                .any(|c| matches!(c, ConfigChange::TtsLimits { .. }))
        );
    }

    #[test]
    fn auth_mode_change_is_hot() {
        let mut new = config();
        new.server.auth_mode = AuthMode::Local;
        new.server.bind = IpAddr::from([127, 0, 0, 1]);
        let changes = diff(&config(), &new);
        assert!(
            changes
                .iter()
                .any(|c| matches!(c, ConfigChange::AuthMode { .. }))
        );
    }

    #[test]
    fn untracked_section_change_reports_other() {
        let mut new = config();
        new.admin.username = "operator".to_string();
        let changes = diff(&config(), &new);
        assert!(
            changes
                .iter()
                .any(|c| matches!(c, ConfigChange::Other { .. }))
        );
    }
}
