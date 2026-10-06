//! The central `ConfigManager`.
//!
//! All configuration changes — GUI, REST, CLI, MCP, first-run
//! wizard, and the filesystem watcher — flow through this
//! component. It owns validation, revisioning, atomic
//! persistence, diff computation, and change events.

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tokio::sync::{Mutex, RwLock, broadcast};

use super::diff::{ChangeSource, ConfigChange, ConfigEvent, diff};
use super::effective::{ConfigSource, EffectiveConfig};
use super::loader;
use super::model::{AppConfig, AuthMode, ConfigError};
use super::validation;
use super::writer;

/// The result of a configuration update.
#[derive(Debug, Clone)]
pub struct ConfigUpdate {
    /// Revision after the change.
    pub revision: u64,
    /// Semantic changes applied.
    pub changes: Vec<ConfigChange>,
}

/// Outcome of a filesystem reload attempt.
#[derive(Debug, Clone)]
pub enum ReloadOutcome {
    /// The file changed and a new revision was installed.
    Updated {
        revision: u64,
        changes: Vec<ConfigChange>,
    },
    /// The file was unchanged (duplicate watcher event).
    Unchanged,
    /// The file is invalid; the previous configuration
    /// remains active.
    Rejected { error: String },
}

/// Runtime status of the configuration system, exposed via
/// `GET /api/config/status`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConfigStatus {
    pub config_path: String,
    pub revision: u64,
    /// Whether the filesystem watcher is active.
    pub watcher: &'static str,
    /// Whether the active configuration is valid.
    pub valid: bool,
    /// Last successful reload timestamp (ISO 8601).
    pub last_reload: Option<String>,
    /// Last reload error, if any.
    pub last_error: Option<String>,
    /// Whether a subsystem restart is pending.
    pub pending_restart: bool,
    pub pending_applies: Vec<String>,
    /// Runtime values that failed to apply (configured
    /// value differs from the active value).
    pub apply_failures: Vec<ApplyFailure>,
}

/// A configured value whose runtime application failed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ApplyFailure {
    pub path: String,
    pub configured: serde_json::Value,
    pub active: serde_json::Value,
    pub error: String,
}

/// Feedback-loop protection state: the content hash of
/// the most recent internal write.
#[derive(Debug, Clone)]
struct WriteRecord {
    content_hash: u64,
}

/// Central configuration manager (see module docs).
pub struct ConfigManager {
    setup_lock: Mutex<()>,
    transaction: Mutex<()>,
    current: RwLock<AppConfig>,
    effective: RwLock<EffectiveConfig>,
    runtime_overrides: RwLock<HashMap<String, serde_json::Value>>,
    path: PathBuf,
    revision: AtomicU64,
    /// Whether the filesystem watcher is running.
    watching: AtomicBool,
    events: broadcast::Sender<ConfigEvent>,
    /// Hash of the file content written by the last internal
    /// write; watcher events matching it are ignored.
    last_internal_write: RwLock<Option<WriteRecord>>,
    status: RwLock<StatusState>,
}

#[derive(Debug, Default)]
struct StatusState {
    last_reload: Option<String>,
    last_error: Option<String>,
    pending_restart: bool,
    pending_applies: HashMap<String, u64>,
    apply_failures: Vec<ApplyFailure>,
}

impl ConfigManager {
    /// Load the configuration from `path` (TOML + environment
    /// overrides), validating before use.
    pub async fn load(path: PathBuf) -> Result<Arc<Self>, ConfigError> {
        let path = if path.is_absolute() {
            path
        } else {
            std::env::current_dir()
                .map_err(|e| ConfigError::invalid("config path", e.to_string()))?
                .join(path)
        };
        let persisted = if path.exists() {
            loader::parse_toml(
                &std::fs::read_to_string(&path)
                    .map_err(|e| ConfigError::invalid("config.toml", e.to_string()))?,
            )?
        } else {
            AppConfig::default()
        };
        let effective = loader::load_from_file(&path)?;
        let mut config = effective.config;
        let mut sources = effective.sources;
        // First-run bootstrap: until setup completes, the
        // server must only listen on loopback without API
        // bearer auth, so an unfinished setup can never
        // expose the server — even if the on-disk file
        // was hand-edited to something unsafe (spec §10).
        if !config.setup_complete {
            config.server.bind = IpAddr::from([127, 0, 0, 1]);
            config.server.auth_mode = AuthMode::Local;
            sources.insert("server.bind".to_string(), ConfigSource::Runtime);
            sources.insert("server.auth_mode".to_string(), ConfigSource::Runtime);
        }
        validation::validate(&config)
            .map_err(|e| ConfigError::invalid(path.display().to_string().as_str(), e))?;
        let file_revision = config.revision.max(1);
        let (tx, _) = broadcast::channel(64);
        let manager = Arc::new(Self {
            setup_lock: Mutex::new(()),
            transaction: Mutex::new(()),
            current: RwLock::new(persisted),
            runtime_overrides: RwLock::new(HashMap::new()),
            effective: RwLock::new(EffectiveConfig { config, sources }),
            path,
            revision: AtomicU64::new(file_revision),
            watching: AtomicBool::new(false),
            events: tx,
            last_internal_write: RwLock::new(None),
            status: RwLock::new(StatusState::default()),
        });
        Ok(manager)
    }

    /// Construct a manager around an in-memory configuration.
    ///
    /// Used by tests and embedding contexts that do
    /// not read a `config.toml` from disk.
    pub fn in_memory(config: AppConfig) -> Arc<Self> {
        let (events, _) = broadcast::channel(64);
        let effective = EffectiveConfig {
            config: config.clone(),
            sources: loader::ALL_CONFIG_PATHS
                .iter()
                .map(|path| ((*path).to_string(), ConfigSource::Runtime))
                .collect(),
        };
        let file_revision = config.revision.max(1);
        Arc::new(Self {
            setup_lock: Mutex::new(()),
            transaction: Mutex::new(()),
            current: RwLock::new(config),
            runtime_overrides: RwLock::new(HashMap::new()),
            effective: RwLock::new(effective),
            path: PathBuf::new(),
            revision: AtomicU64::new(file_revision),
            watching: AtomicBool::new(false),
            events,
            last_internal_write: RwLock::new(None),
            status: RwLock::new(StatusState::default()),
        })
    }

    /// Serialize first-run credential creation and completion across wizard requests.
    pub async fn setup_guard(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.setup_lock.lock().await
    }

    /// The path of the managed `config.toml`.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Snapshot the current configuration.
    pub async fn get(&self) -> AppConfig {
        self.effective.read().await.config.clone()
    }

    /// Snapshot the effective configuration (with sources).
    pub async fn get_effective(&self) -> EffectiveConfig {
        self.effective.read().await.clone()
    }

    /// Current configuration revision.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::SeqCst)
    }

    /// Subscribe to configuration change events.
    pub fn subscribe(&self) -> broadcast::Receiver<ConfigEvent> {
        self.events.subscribe()
    }

    /// Validate a candidate configuration.
    pub async fn validate(&self, config: &AppConfig) -> Result<(), String> {
        validation::validate(config)
    }

    /// Apply a transactional update.
    ///
    /// The updater mutates a clone of the current
    /// configuration; the result is validated, atomically
    /// persisted, installed, and broadcast. When
    /// `expected_revision` is `Some` and does not match the
    /// current revision, the update is rejected with a
    /// conflict (optimistic concurrency).
    pub async fn update<F>(
        &self,
        expected_revision: Option<u64>,
        updater: F,
    ) -> Result<ConfigUpdate, UpdateError>
    where
        F: FnOnce(&mut AppConfig),
    {
        self.update_from(expected_revision, ChangeSource::Api, updater)
            .await
    }

    /// Commit a persistent change with its originating surface recorded.
    pub async fn update_from<F>(
        &self,
        expected_revision: Option<u64>,
        source: ChangeSource,
        updater: F,
    ) -> Result<ConfigUpdate, UpdateError>
    where
        F: FnOnce(&mut AppConfig),
    {
        let _transaction = self.transaction.lock().await;
        if let Some(expected) = expected_revision
            && expected != self.revision()
        {
            return Err(UpdateError::Conflict {
                expected,
                current: self.revision(),
            });
        }
        let _file_lock = self.lock_file().await?;
        let stored = self.current.read().await.clone();
        if self.path.exists() {
            let disk = loader::parse_toml(
                &std::fs::read_to_string(&self.path).map_err(writer::AtomicWriteError::Io)?,
            )
            .map_err(|e| UpdateError::Validation(e.to_string()))?;
            if !diff(&stored, &disk).is_empty() || disk.revision != stored.revision {
                return Err(UpdateError::Conflict {
                    expected: expected_revision.unwrap_or(self.revision()),
                    current: disk.revision.max(self.revision().saturating_add(1)),
                });
            }
        }
        let old = self.get().await;
        let mut candidate = stored;
        updater(&mut candidate);
        validation::validate(&candidate).map_err(UpdateError::Validation)?;
        let new_revision = self
            .revision()
            .checked_add(1)
            .ok_or_else(|| UpdateError::Validation("revision exhausted".into()))?;
        candidate.revision = new_revision;
        let mut effective = loader::from_toml(
            &writer::to_toml_string(&candidate).map_err(writer::AtomicWriteError::Serialize)?,
        )
        .map_err(|e| UpdateError::Validation(e.to_string()))?;
        loader::apply_environment(&mut effective.config, &mut effective.sources)
            .map_err(|e| UpdateError::Validation(e.to_string()))?;
        self.apply_overrides(&mut effective).await?;
        validation::validate(&effective.config).map_err(UpdateError::Validation)?;
        if !self.path.as_os_str().is_empty() {
            writer::atomic_write_config(&self.path, &candidate).await?;
        }
        let changes = diff(&old, &effective.config);
        *self.current.write().await = candidate.clone();
        self.install(effective, changes.clone(), new_revision, source)
            .await;
        *self.last_internal_write.write().await = Some(WriteRecord {
            content_hash: hash_config(&candidate),
        });
        self.touch_reload_time().await;
        Ok(ConfigUpdate {
            revision: new_revision,
            changes,
        })
    }

    async fn lock_file(&self) -> Result<Option<std::fs::File>, UpdateError> {
        if self.path.as_os_str().is_empty() {
            return Ok(None);
        }
        let path = self.path.with_extension("toml.lock");
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)?;
            file.lock()?;
            Ok::<_, std::io::Error>(Some(file))
        })
        .await
        .map_err(|e| UpdateError::Validation(e.to_string()))?
        .map_err(|e| UpdateError::Write(writer::AtomicWriteError::Io(e)))
    }

    /// Read a revision and persisted document from the same transaction.
    pub async fn snapshot(&self) -> (u64, AppConfig) {
        let _transaction = self.transaction.lock().await;
        (self.revision(), self.current.read().await.clone())
    }

    /// Snapshot the persisted configuration without environment or runtime overlays.
    pub async fn get_persisted(&self) -> AppConfig {
        self.current.read().await.clone()
    }

    async fn apply_overrides(&self, effective: &mut EffectiveConfig) -> Result<(), UpdateError> {
        let overrides = self.runtime_overrides.read().await;
        let mut document = serde_json::to_value(&effective.config)
            .map_err(|e| UpdateError::Validation(e.to_string()))?;
        for (path, value) in overrides.iter() {
            let mut cursor = &mut document;
            let parts: Vec<_> = path.split('.').collect();
            for part in &parts[..parts.len() - 1] {
                cursor = &mut cursor[*part];
            }
            cursor[parts[parts.len() - 1]] = value.clone();
            effective
                .sources
                .insert(path.clone(), ConfigSource::Runtime);
        }
        effective.config =
            serde_json::from_value(document).map_err(|e| UpdateError::Validation(e.to_string()))?;
        if !effective.config.setup_complete {
            effective.config.server.bind = IpAddr::from([127, 0, 0, 1]);
            effective.config.server.auth_mode = AuthMode::Local;
            effective
                .sources
                .insert("server.bind".into(), ConfigSource::Runtime);
            effective
                .sources
                .insert("server.auth_mode".into(), ConfigSource::Runtime);
        }
        Ok(())
    }

    /// Apply an in-memory-only update (never persisted).
    ///
    /// Used for transient runtime overrides such as
    /// detected free ports: the operator's file is
    /// not rewritten by a transient condition.
    pub async fn update_runtime<F>(
        &self,
        expected_revision: Option<u64>,
        updater: F,
    ) -> Result<ConfigUpdate, UpdateError>
    where
        F: FnOnce(&mut AppConfig),
    {
        let _transaction = self.transaction.lock().await;
        if let Some(expected) = expected_revision
            && expected != self.revision()
        {
            return Err(UpdateError::Conflict {
                expected,
                current: self.revision(),
            });
        }
        let old = self.get().await;
        let mut candidate = old.clone();
        updater(&mut candidate);
        validation::validate(&candidate).map_err(UpdateError::Validation)?;
        let changes = diff(&old, &candidate);
        let old_json = serde_json::to_value(&old).unwrap();
        let new_json = serde_json::to_value(&candidate).unwrap();
        let mut overrides = self.runtime_overrides.write().await;
        for path in loader::ALL_CONFIG_PATHS {
            let pointer = format!("/{}", path.replace('.', "/"));
            if old_json.pointer(&pointer) != new_json.pointer(&pointer) {
                overrides.insert(
                    (*path).to_string(),
                    new_json.pointer(&pointer).cloned().unwrap_or_default(),
                );
            }
        }
        drop(overrides);
        let old_revision = self.revision();
        let new_revision = old_revision + 1;
        candidate.revision = new_revision;
        let mut effective = self.get_effective().await;
        effective.config = candidate;
        self.install(effective, changes.clone(), new_revision, ChangeSource::Api)
            .await;
        self.record_sources_from_runtime(&changes).await;
        Ok(ConfigUpdate {
            revision: new_revision,
            changes,
        })
    }

    /// Reload the configuration from disk (explicit reload).
    pub async fn reload_from_disk(&self) -> Result<ConfigUpdate, UpdateError> {
        let outcome = self.reload_from_disk_internal(ChangeSource::Api).await;
        match outcome {
            ReloadOutcome::Updated { revision, changes } => Ok(ConfigUpdate { revision, changes }),
            ReloadOutcome::Unchanged => Ok(ConfigUpdate {
                revision: self.revision(),
                changes: Vec::new(),
            }),
            ReloadOutcome::Rejected { error } => Err(UpdateError::Validation(error)),
        }
    }

    /// Filesystem-watcher entry point. Debouncing happens in
    /// the watcher; this performs the parse/validate/install
    /// flow and guards against feedback loops from internal
    /// writes.
    pub(crate) async fn reload_from_filesystem(&self) -> ReloadOutcome {
        // Feedback-loop guard: if the file content exactly
        // matches our last internal write, this event is our
        // own echo — ignore it.
        if let Some(record) = self.last_internal_write.read().await.as_ref() {
            if let Ok(current) = std::fs::read_to_string(&self.path) {
                if hash_str(&current) == record.content_hash {
                    self.touch_reload_time().await;
                    return ReloadOutcome::Unchanged;
                }
            }
        }
        self.reload_from_disk_internal(ChangeSource::Filesystem)
            .await
    }

    async fn reload_from_disk_internal(&self, source: ChangeSource) -> ReloadOutcome {
        let _transaction = self.transaction.lock().await;
        let _file_lock = match self.lock_file().await {
            Ok(lock) => lock,
            Err(e) => {
                let error = e.to_string();
                self.record_reload_error(&error).await;
                return ReloadOutcome::Rejected { error };
            }
        };
        // Read + parse + validate the candidate without
        // touching the live configuration.
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) => {
                let error = format!("could not read config file: {e}");
                self.record_reload_error(&error).await;
                self.events
                    .send(ConfigEvent::ReloadRejected {
                        error: error.clone(),
                        source,
                    })
                    .ok();
                return ReloadOutcome::Rejected { error };
            }
        };
        let candidate = match loader::from_toml(&text) {
            Ok(candidate) => candidate,
            Err(e) => {
                let error = e.to_string();
                tracing::warn!("Configuration reload rejected: {error}");
                self.record_reload_error(&error).await;
                self.events
                    .send(ConfigEvent::ReloadRejected {
                        error: error.clone(),
                        source,
                    })
                    .ok();
                return ReloadOutcome::Rejected { error };
            }
        };
        let persisted = candidate.config.clone();
        let mut effective = candidate;
        let resolved = loader::apply_environment(&mut effective.config, &mut effective.sources)
            .map_err(|e| e.to_string());
        let resolved = match resolved {
            Ok(()) => self
                .apply_overrides(&mut effective)
                .await
                .map_err(|e| e.to_string()),
            Err(e) => Err(e),
        };
        if let Err(error) = resolved {
            self.record_reload_error(&error).await;
            let _ = self.events.send(ConfigEvent::ReloadRejected {
                error: error.clone(),
                source,
            });
            return ReloadOutcome::Rejected { error };
        }
        let sources = effective.sources;
        let candidate = effective.config;
        if let Err(e) = validation::validate(&candidate) {
            tracing::warn!("Configuration reload rejected: {e}");
            self.record_reload_error(&e).await;
            self.events
                .send(ConfigEvent::ReloadRejected {
                    error: e.clone(),
                    source,
                })
                .ok();
            return ReloadOutcome::Rejected { error: e };
        }
        let old = self.get().await;
        let changes = diff(&old, &candidate);
        let persisted_changes = diff(&*self.current.read().await, &persisted);
        if changes.is_empty()
            && persisted_changes.is_empty()
            && old.version == candidate.version
            && old.setup_complete == candidate.setup_complete
        {
            // Nothing semantically changed: a duplicate
            // filesystem event. Update the reload timestamp
            // only.
            let mut persisted = persisted;
            persisted.revision = self.revision();
            *self.current.write().await = persisted;
            self.effective.write().await.sources = sources;
            self.touch_reload_time().await;
            return ReloadOutcome::Unchanged;
        }
        let old_revision = self.revision();
        let mut new_revision = old_revision + 1;
        // An external edit may carry its own (higher)
        // revision: never move backwards, and keep the
        // installed view consistent with the file.
        new_revision = new_revision.max(candidate.revision);
        let mut candidate = candidate;
        candidate.revision = new_revision;
        let mut persisted = persisted;
        persisted.revision = new_revision;
        if let Err(error) = writer::atomic_write_config(&self.path, &persisted).await {
            let error = error.to_string();
            self.record_reload_error(&error).await;
            let _ = self.events.send(ConfigEvent::ReloadRejected {
                error: error.clone(),
                source,
            });
            return ReloadOutcome::Rejected { error };
        }
        *self.last_internal_write.write().await = Some(WriteRecord {
            content_hash: hash_config(&persisted),
        });
        *self.current.write().await = persisted;
        self.install(
            EffectiveConfig {
                config: candidate,
                sources,
            },
            changes.clone(),
            new_revision,
            source,
        )
        .await;
        self.touch_reload_time().await;
        tracing::info!(
            old_revision,
            new_revision,
            changes = changes.len(),
            source = source.as_str(),
            "Configuration reloaded from disk"
        );
        ReloadOutcome::Updated {
            revision: new_revision,
            changes,
        }
    }

    /// Install a new configuration: swap the live config and
    /// effective view, bump the revision, and broadcast.
    async fn install(
        &self,
        effective: EffectiveConfig,
        changes: Vec<ConfigChange>,
        new_revision: u64,
        source: ChangeSource,
    ) {
        let old_revision = self.revision();
        let config = effective.config.clone();
        *self.effective.write().await = effective;
        self.revision.store(new_revision, Ordering::SeqCst);
        self.events
            .send(ConfigEvent::Updated {
                old_revision,
                new_revision,
                changes,
                config,
                source,
            })
            .ok();
    }

    async fn record_sources_from_runtime(&self, _changes: &[ConfigChange]) {
        let overrides = self.runtime_overrides.read().await;
        let mut effective = self.effective.write().await;
        for path in overrides.keys() {
            effective
                .sources
                .insert(path.clone(), ConfigSource::Runtime);
        }
    }

    /// Persist the current configuration (used after
    /// first-run setup completes).
    pub async fn save(&self) -> Result<(), writer::AtomicWriteError> {
        let _transaction = self.transaction.lock().await;
        let _file_lock = self
            .lock_file()
            .await
            .map_err(|e| writer::AtomicWriteError::Validation(e.to_string()))?;
        let current = self.current.read().await.clone();
        writer::atomic_write_config(&self.path, &current).await?;
        *self.last_internal_write.write().await = Some(WriteRecord {
            content_hash: hash_config(&current),
        });
        Ok(())
    }

    /// Record a runtime apply failure (configured value
    /// differs from the active runtime value).
    pub async fn record_apply_failure(
        &self,
        path: &str,
        configured: serde_json::Value,
        active: serde_json::Value,
        error: &str,
    ) {
        let failure = ApplyFailure {
            path: path.to_string(),
            configured,
            active,
            error: error.to_string(),
        };
        tracing::warn!(
            path,
            configured = failure.configured.to_string(),
            active = failure.active.to_string(),
            error,
            "configuration apply failed; previous value remains active"
        );
        let mut status = self.status.write().await;
        status.apply_failures.retain(|f| f.path != path);
        status.apply_failures.push(failure);
        self.events
            .send(ConfigEvent::ApplyFailed {
                path: path.to_string(),
                error: error.to_string(),
            })
            .ok();
    }

    /// Clear a recorded apply failure (the value applied
    /// successfully after all).
    pub async fn clear_apply_failure(&self, path: &str) {
        let mut status = self.status.write().await;
        status.apply_failures.retain(|f| f.path != path);
    }

    pub async fn mark_pending_apply(&self, path: &str, revision: u64) {
        self.status
            .write()
            .await
            .pending_applies
            .insert(path.to_string(), revision);
    }

    pub async fn finish_pending_apply(&self, path: &str, revision: u64) {
        let mut status = self.status.write().await;
        if status.pending_applies.get(path) == Some(&revision) {
            status.pending_applies.remove(path);
        }
    }

    /// Mark that a subsystem restart is pending.
    pub async fn set_pending_restart(&self, pending: bool) {
        self.status.write().await.pending_restart = pending;
    }

    /// Current configuration-system status.
    pub async fn status(&self) -> ConfigStatus {
        let state = self.status.read().await;
        ConfigStatus {
            config_path: self.path.display().to_string(),
            revision: self.revision(),
            watcher: if self.watching.load(Ordering::SeqCst) {
                "active"
            } else {
                "inactive"
            },
            valid: true,
            last_reload: state.last_reload.clone(),
            last_error: state.last_error.clone(),
            pending_restart: state.pending_restart,
            pending_applies: state.pending_applies.keys().cloned().collect(),
            apply_failures: state.apply_failures.clone(),
        }
    }

    /// Record whether the filesystem watcher is running.
    pub fn set_watching(&self, watching: bool) {
        self.watching.store(watching, Ordering::SeqCst);
    }

    async fn touch_reload_time(&self) {
        let mut status = self.status.write().await;
        status.last_reload = Some(chrono::Utc::now().to_rfc3339());
        status.last_error = None;
    }

    async fn record_reload_error(&self, error: &str) {
        let error = error.to_string();
        let mut status = self.status.write().await;
        status.last_error = Some(error);
    }
}

/// Failure to apply a configuration update.
#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    #[error("configuration conflict: expected revision {expected}, current {current}")]
    Conflict { expected: u64, current: u64 },
    #[error("invalid configuration: {0}")]
    Validation(String),
    #[error("failed to persist configuration: {0}")]
    Write(#[from] writer::AtomicWriteError),
}

/// Stable content hash for feedback-loop detection.
fn hash_config(config: &AppConfig) -> u64 {
    let text = writer::to_toml_string(config).unwrap_or_default();
    hash_str(&text)
}

fn hash_str(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    async fn temp_manager(tag: &str) -> Arc<ConfigManager> {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-manager-test-{}-{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("config.toml");
        let _ = std::fs::remove_file(&path);
        let mut config = AppConfig::default();
        {
            config.paths.audio = Some(dir.join("audio").to_string_lossy().to_string());
        }
        writer::atomic_write_config_blocking(&path, &config).unwrap();
        ConfigManager::load(path).await.unwrap()
    }

    fn temp_dir_of(manager: &ConfigManager) -> PathBuf {
        manager.path().parent().unwrap().to_path_buf()
    }

    #[tokio::test]
    async fn restoring_the_last_committed_file_clears_reload_error() {
        let manager = temp_manager("restore-valid").await;
        manager
            .update(None, |c| c.audio.volume = 0.5)
            .await
            .unwrap();
        let valid = std::fs::read(manager.path()).unwrap();
        std::fs::write(manager.path(), "[server]\nport = 'invalid'\n").unwrap();
        assert!(matches!(
            manager.reload_from_filesystem().await,
            ReloadOutcome::Rejected { .. }
        ));
        assert!(manager.status().await.last_error.is_some());
        std::fs::write(manager.path(), valid).unwrap();
        assert!(matches!(
            manager.reload_from_filesystem().await,
            ReloadOutcome::Unchanged
        ));
        assert!(manager.status().await.last_error.is_none());
        assert_eq!(manager.revision(), 2);
        std::fs::remove_dir_all(temp_dir_of(&manager)).unwrap();
    }

    #[tokio::test]
    async fn runtime_overrides_survive_reload_without_being_persisted() {
        let manager = temp_manager("runtime-overlay").await;
        manager
            .update_runtime(None, |c| c.audio.volume = 0.25)
            .await
            .unwrap();
        manager
            .update(None, |c| c.model.inference_steps = 10)
            .await
            .unwrap();
        assert_eq!(manager.get().await.audio.volume, 0.25);
        assert_eq!(manager.get_persisted().await.audio.volume, 1.0);
        let mut external = manager.get_persisted().await;
        external.audio.volume = 0.75;
        writer::atomic_write_config_blocking(manager.path(), &external).unwrap();
        manager.reload_from_disk().await.unwrap();
        assert_eq!(manager.get().await.audio.volume, 0.25);
        assert_eq!(manager.get_persisted().await.audio.volume, 0.75);
        assert_eq!(
            manager.get_effective().await.source_of("audio.volume"),
            Some(ConfigSource::Runtime)
        );
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn separate_managers_cannot_overwrite_a_newer_file() {
        let first = temp_manager("separate-writers").await;
        let second = ConfigManager::load(first.path().to_path_buf())
            .await
            .unwrap();
        first.update(None, |c| c.audio.volume = 0.5).await.unwrap();
        assert!(matches!(
            second.update(Some(1), |c| c.server.port = 19000).await,
            Err(UpdateError::Conflict { .. })
        ));
        second.reload_from_disk().await.unwrap();
        second
            .update(None, |c| c.server.port = 19000)
            .await
            .unwrap();
        let stored = loader::parse_toml(&std::fs::read_to_string(first.path()).unwrap()).unwrap();
        assert_eq!(stored.audio.volume, 0.5);
        assert_eq!(stored.server.port, 19000);
        let _ = std::fs::remove_dir_all(temp_dir_of(&first));
    }

    #[tokio::test]
    async fn concurrent_updates_preserve_both_changes() {
        let manager = temp_manager("concurrent").await;
        let (first, second) = tokio::join!(
            manager.update(None, |c| c.server.port = 19000),
            manager.update(None, |c| c.audio.volume = 0.5),
        );
        first.unwrap();
        second.unwrap();
        let current = manager.get().await;
        assert_eq!(current.server.port, 19000);
        assert_eq!(current.audio.volume, 0.5);
        assert_eq!(current.revision, 3);
        let disk = loader::parse_toml(&std::fs::read_to_string(manager.path()).unwrap()).unwrap();
        assert_eq!(disk.server.port, current.server.port);
        assert_eq!(disk.audio.volume, current.audio.volume);
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn concurrent_expected_revision_has_only_one_winner() {
        let manager = temp_manager("concurrent-conflict").await;
        let (first, second) = tokio::join!(
            manager.update(Some(1), |c| c.server.port = 19000),
            manager.update(Some(1), |c| c.server.port = 19001),
        );
        assert_ne!(first.is_ok(), second.is_ok());
        assert_eq!(manager.revision(), 2);
        let error = first.err().or_else(|| second.err()).unwrap();
        assert!(matches!(
            error,
            UpdateError::Conflict {
                expected: 1,
                current: 2
            }
        ));
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn reload_events_identify_filesystem_source() {
        let manager = temp_manager("reload-source").await;
        let mut events = manager.subscribe();
        let mut candidate = manager.get().await;
        candidate.audio.volume = 0.5;
        writer::atomic_write_config_blocking(manager.path(), &candidate).unwrap();
        assert!(matches!(
            manager.reload_from_filesystem().await,
            ReloadOutcome::Updated { .. }
        ));
        assert!(matches!(
            events.recv().await.unwrap(),
            ConfigEvent::Updated {
                source: ChangeSource::Filesystem,
                ..
            }
        ));
        let status = manager.status().await;
        chrono::DateTime::parse_from_rfc3339(status.last_reload.as_ref().unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn load_and_get_round_trip() {
        let manager = temp_manager("basic").await;
        let config = manager.get().await;
        assert_eq!(config.server.port, 17842);
        assert_eq!(manager.revision(), 1);
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn update_increments_revision_and_persists() {
        let manager = temp_manager("update").await;
        let update = manager
            .update(None, |c| c.server.port = 19000)
            .await
            .unwrap();
        assert_eq!(update.revision, 2);
        assert_eq!(manager.revision(), 2);
        assert_eq!(manager.get().await.server.port, 19000);
        // Persisted to disk.
        let text = std::fs::read_to_string(manager.path()).unwrap();
        assert!(text.contains("19000"));
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn update_with_wrong_expected_revision_conflicts() {
        let manager = temp_manager("conflict").await;
        manager
            .update(None, |c| c.server.port = 19000)
            .await
            .unwrap();
        let err = manager
            .update(Some(1), |c| c.server.port = 19001)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            UpdateError::Conflict {
                expected: 1,
                current: 2
            }
        ));
        // The failed change did not increment the revision.
        assert_eq!(manager.revision(), 2);
        assert_eq!(manager.get().await.server.port, 19000);
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn update_with_correct_expected_revision_succeeds() {
        let manager = temp_manager("expected").await;
        let update = manager
            .update(Some(1), |c| c.server.port = 19000)
            .await
            .unwrap();
        assert_eq!(update.revision, 2);
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn invalid_update_is_rejected_and_not_persisted() {
        let manager = temp_manager("invalid").await;
        let err = manager
            .update(None, |c| c.server.port = 0)
            .await
            .unwrap_err();
        assert!(matches!(err, UpdateError::Validation(_)));
        assert_eq!(manager.revision(), 1);
        assert_eq!(manager.get().await.server.port, 17842);
        // Disk still holds the previous valid file.
        let text = std::fs::read_to_string(manager.path()).unwrap();
        assert!(!text.contains("port = 0"));
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn local_auth_cannot_bind_non_loopback() {
        let manager = temp_manager("security").await;
        let err = manager
            .update(None, |c| {
                c.server.auth_mode = super::super::model::AuthMode::Local;
                c.server.bind = "0.0.0.0".parse().unwrap();
            })
            .await
            .unwrap_err();
        assert!(matches!(err, UpdateError::Validation(_)));
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn reload_from_disk_installs_external_changes() {
        let manager = temp_manager("reload").await;
        // Simulate an external edit.
        let mut new_config = manager.get().await;
        new_config.server.port = 19001;
        writer::atomic_write_config_blocking(manager.path(), &new_config).unwrap();
        // Clear the internal-write guard so the external
        // change is treated as foreign.
        *manager.last_internal_write.write().await = None;
        let outcome = manager.reload_from_filesystem().await;
        match outcome {
            ReloadOutcome::Updated { revision, changes } => {
                assert_eq!(revision, 2);
                assert!(!changes.is_empty());
            }
            other => panic!("expected update, got {other:?}"),
        }
        assert_eq!(manager.get().await.server.port, 19001);
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn invalid_external_edit_is_rejected_and_keeps_runtime() {
        let manager = temp_manager("badedit").await;
        let before = manager.get().await;
        std::fs::write(manager.path(), "[server]\nport = \"hello\"\n").unwrap();
        *manager.last_internal_write.write().await = None;
        let outcome = manager.reload_from_filesystem().await;
        match &outcome {
            ReloadOutcome::Rejected { error } => {
                assert!(
                    error.contains("invalid configuration") || error.contains("could not parse")
                );
            }
            other => panic!("expected rejection, got {other:?}"),
        }
        // Runtime configuration is unchanged.
        assert_eq!(manager.get().await.server.port, before.server.port);
        assert_eq!(manager.revision(), 1);
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn duplicate_watcher_events_are_ignored() {
        let manager = temp_manager("dup").await;
        let mut new_config = manager.get().await;
        new_config.logging.level = "debug".to_string();
        writer::atomic_write_config_blocking(manager.path(), &new_config).unwrap();
        *manager.last_internal_write.write().await = None;
        let first = manager.reload_from_filesystem().await;
        assert!(matches!(first, ReloadOutcome::Updated { .. }));
        // Second event with identical content: unchanged.
        let second = manager.reload_from_filesystem().await;
        assert!(matches!(second, ReloadOutcome::Unchanged));
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn internal_write_does_not_reload_from_own_event() {
        let manager = temp_manager("feedback").await;
        manager
            .update(None, |c| c.audio.output_device = "Speakers".to_string())
            .await
            .unwrap();
        // The watcher sees the file we just wrote: it must be
        // treated as our own echo, not an external change.
        let outcome = manager.reload_from_filesystem().await;
        assert!(matches!(outcome, ReloadOutcome::Unchanged));
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn events_are_broadcast_to_subscribers() {
        let manager = temp_manager("events").await;
        let mut rx = manager.subscribe();
        manager
            .update(None, |c| c.server.port = 19002)
            .await
            .unwrap();
        let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("event must arrive")
            .expect("channel open");
        match event {
            ConfigEvent::Updated {
                old_revision,
                new_revision,
                changes,
                ..
            } => {
                assert_eq!(old_revision, 1);
                assert_eq!(new_revision, 2);
                assert!(!changes.is_empty());
            }
            other => panic!("expected Updated, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn apply_failures_are_tracked_and_cleared() {
        let manager = temp_manager("applyfail").await;
        manager
            .record_apply_failure(
                "audio.output_device",
                serde_json::json!("Missing Device"),
                serde_json::json!("default"),
                "device unavailable",
            )
            .await;
        let status = manager.status().await;
        assert_eq!(status.apply_failures.len(), 1);
        assert_eq!(status.apply_failures[0].path, "audio.output_device");
        manager.clear_apply_failure("audio.output_device").await;
        assert!(manager.status().await.apply_failures.is_empty());
        let _ = std::fs::remove_dir_all(temp_dir_of(&manager));
    }

    #[tokio::test]
    async fn effective_config_tracks_sources() {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-manager-effective-{}-{}",
            std::process::id(),
            "eff"
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("config.toml");
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, "[server]\nport = 19000\n").unwrap();
        let manager = ConfigManager::load(path).await.unwrap();
        let effective = manager.get_effective().await;
        assert_eq!(effective.source_of("server.port"), Some(ConfigSource::Toml));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
