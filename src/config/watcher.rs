//! Filesystem watcher for `config.toml`.
//!
//! Debounces the burst of low-level events a text
//! editor produces (300–500 ms), then routes the
//! change through the [`ConfigManager`] so validation,
//! diffing, and runtime reconfiguration stay in one
//! place.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use notify_debouncer_full::{DebounceEventResult, new_debouncer};

use super::manager::ConfigManager;

/// Debounce window for filesystem events (spec §43).
pub const DEBOUNCE_MILLIS: u64 = 400;

/// Handle for the background watcher task.
pub struct ConfigWatcher {
    manager: Arc<ConfigManager>,
    poll_task: tokio::task::JoinHandle<()>,
    _debouncer: Option<
        notify_debouncer_full::Debouncer<
            notify::RecommendedWatcher,
            notify_debouncer_full::RecommendedCache,
        >,
    >,
}

impl Drop for ConfigWatcher {
    fn drop(&mut self) {
        self.poll_task.abort();
        self.manager.set_watching(false);
    }
}

/// Start watching `config_path` for external edits.
///
/// Native notifications are supplemented by content polling, including
/// platforms or filesystems where native notification setup fails.
pub fn start_watching(manager: Arc<ConfigManager>) -> Option<ConfigWatcher> {
    let config_path = manager.path().to_path_buf();
    let manager_clone = Arc::clone(&manager);
    let closure_path = config_path.clone();
    // Captured from the runtime thread `start_watching`
    // is called on; notify delivers events on its own
    // watcher thread, where `Handle::current()` would
    // panic.
    let handle = tokio::runtime::Handle::current();

    let mut debouncer = new_debouncer(
        Duration::from_millis(DEBOUNCE_MILLIS),
        None,
        move |result: DebounceEventResult| {
            if let Err(e) = &result {
                tracing::warn!("Config watcher error: {e:?}");
                return;
            }
            let events = result.as_ref().unwrap();

            // Only react to file-content changes on the
            // config file itself (create/write/remove/
            // rename), not metadata-only notices.
            let relevant = events.iter().any(|debounced| {
                let event = &debounced.event;
                event.paths.iter().any(|p| p == &closure_path)
                    && matches!(
                        event.kind,
                        notify::EventKind::Modify(_)
                            | notify::EventKind::Create(_)
                            | notify::EventKind::Remove(_)
                    )
            });
            if !relevant {
                return;
            }
            // Notify events are delivered on a dedicated
            // watcher thread; hop onto the Tokio runtime
            // (via the handle captured above) to perform
            // the async reload without blocking the
            // watcher thread.
            let manager = Arc::clone(&manager_clone);
            let path = closure_path.clone();
            handle.spawn(async move {
                match manager.reload_from_filesystem().await {
                    super::manager::ReloadOutcome::Updated { revision, changes } => {
                        tracing::info!(
                            revision,
                            changes = changes.len(),
                            "Applied external configuration change from {path:?}"
                        );
                    }
                    super::manager::ReloadOutcome::Rejected { error } => {
                        tracing::warn!(
                            "Configuration reload rejected: {error} \
                             (previous configuration remains active)"
                        );
                    }
                    super::manager::ReloadOutcome::Unchanged => {}
                }
            });
        },
    )
    .map_err(|error| {
        tracing::warn!("Native config notifications unavailable; using polling: {error}")
    })
    .ok();

    // Watch the config file's parent directory so
    // editor rename-over-write patterns are caught,
    // and the file itself.
    let watch_path = config_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    if let Some(watcher) = debouncer.as_mut() {
        if let Err(error) = watcher.watch(&watch_path, notify::RecursiveMode::NonRecursive) {
            tracing::warn!("Native config watch unavailable; using polling: {error}");
            debouncer = None;
        }
    }

    manager.set_watching(true);

    tracing::info!(
        path = ?config_path,
        debounce_ms = DEBOUNCE_MILLIS,
        "Watching configuration file for changes"
    );

    // Some network/virtual filesystems silently omit native notifications.
    // Content polling supplements notify; the manager deduplicates both paths.
    let poll_manager = Arc::clone(&manager);
    let poll_path = config_path;
    let mut previous = std::fs::read(&poll_path).ok();
    let poll_task = tokio::spawn(async move {
        let _ = poll_manager.reload_from_filesystem().await;
        let mut interval = tokio::time::interval(Duration::from_millis(DEBOUNCE_MILLIS));
        loop {
            interval.tick().await;
            let content = tokio::fs::read(&poll_path).await.ok();
            if content != previous {
                previous = content;
                let _ = poll_manager.reload_from_filesystem().await;
            }
        }
    });
    Some(ConfigWatcher {
        manager,
        poll_task,
        _debouncer: debouncer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debounce_window_is_in_spec_range() {
        assert!(
            (300..=500).contains(&DEBOUNCE_MILLIS),
            "debounce must be 300-500ms"
        );
    }

    #[tokio::test]
    async fn real_watcher_applies_edits_and_rejects_invalid_file() {
        let dir = std::env::temp_dir().join(format!("sonicboom-watch-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let mut config = super::super::model::AppConfig::default();
        config.paths.audio = Some(dir.join("audio").display().to_string());
        super::super::writer::atomic_write_config_blocking(&path, &config).unwrap();
        let manager = ConfigManager::load(path.clone()).await.unwrap();
        let watcher = start_watching(manager.clone()).unwrap();
        let mut events = manager.subscribe();
        config.audio.volume = 0.5;
        super::super::writer::atomic_write_config_blocking(&path, &config).unwrap();
        let event = tokio::time::timeout(Duration::from_secs(5), events.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            event,
            super::super::diff::ConfigEvent::Updated { .. }
        ));
        assert_eq!(manager.get().await.audio.volume, 0.5);
        let revision = manager.revision();
        std::fs::write(&path, "[server]\nport = 'hello'\n").unwrap();
        let event = tokio::time::timeout(Duration::from_secs(5), events.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            event,
            super::super::diff::ConfigEvent::ReloadRejected { .. }
        ));
        assert_eq!(manager.revision(), revision);
        assert_eq!(manager.get().await.audio.volume, 0.5);
        drop(watcher);
        assert_eq!(manager.status().await.watcher, "inactive");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn watching_a_temp_config_file_works() {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-watcher-test-{}-{}",
            std::process::id(),
            "watch"
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("config.toml");
        let _ = std::fs::remove_file(&path);
        let mut config = super::super::model::AppConfig::default();
        {
            config.paths.audio = Some(dir.join("audio").to_string_lossy().to_string());
        }
        super::super::writer::atomic_write_config_blocking(&path, &config).unwrap();
        let manager = ConfigManager::load(path).await.unwrap();
        // Watching may be unavailable on some platforms;
        // the contract is "Some on supported platforms".
        let watcher = start_watching(manager);
        assert!(watcher.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
