//! Runtime reconfiguration coordinator.
//!
//! Applies committed configuration changes to the
//! live subsystems with the smallest possible blast
//! radius: hot apply first, subsystem replacement
//! next, HTTP listener restart, then model reload.
//! A full process restart is never triggered here.

use std::sync::Arc;

use crate::config::diff::{ApplyStrategy, ConfigChange};
use crate::config::manager::ConfigManager;
use crate::config::model::AppConfig;
use crate::config::secrets::SecretStore;
use crate::runtime::model::ModelService;
use crate::runtime::subsystems::{InferenceGateManager, RateLimiterManager};
use crate::server::ApiServerManager;
#[cfg(feature = "playback")]
use crate::tts::queue::AudioManager;

/// Placeholder audio manager type when the
/// playback feature is disabled: the audio
/// field is always `None` in that case.
#[cfg(not(feature = "playback"))]
type AudioManager = ();

/// Outcome of applying one configuration change.
#[derive(Debug, Clone)]
pub struct ApplyResult {
    pub path: String,
    pub strategy: ApplyStrategy,
    pub status: ApplyStatus,
}

/// Application outcome.
#[derive(Debug, Clone)]
pub enum ApplyStatus {
    /// Applied to the live subsystem.
    Applied,
    /// Applied by reading the new value at request
    /// time (no subsystem action needed).
    PerRequest,
    /// The runtime subsystem failed to apply the
    /// configured value; the previous working value
    /// stays active and the failure is recorded in
    /// the configuration status.
    Failed(String),
    /// Requires a process restart (e.g. log sink
    /// re-initialization).
    PendingRestart,
    PendingApply,
}

/// Applies configuration changes to live subsystems.
pub struct RuntimeReconfigurator {
    pub config: Arc<ConfigManager>,
    #[cfg_attr(not(feature = "playback"), allow(dead_code))]
    pub audio: Arc<Option<AudioManager>>,
    pub server: Arc<ApiServerManager>,
    pub model: Arc<ModelService>,
    pub rate_limit: Arc<RateLimiterManager>,
    pub inference: Arc<InferenceGateManager>,
    pub secrets: Arc<dyn SecretStore>,
}

impl RuntimeReconfigurator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Arc<ConfigManager>,
        audio: Arc<Option<AudioManager>>,
        server: Arc<ApiServerManager>,
        model: Arc<ModelService>,
        rate_limit: Arc<RateLimiterManager>,
        inference: Arc<InferenceGateManager>,
        secrets: Arc<dyn SecretStore>,
    ) -> Self {
        Self {
            config,
            audio,
            server,
            model,
            rate_limit,
            inference,
            secrets,
        }
    }

    /// Apply a set of committed changes to the live
    /// subsystems. Each change is applied
    /// independently: one failure does not prevent
    /// the others from applying.
    pub async fn apply(&self, new: &AppConfig, changes: &[ConfigChange]) -> Vec<ApplyResult> {
        let mut results = Vec::new();
        let model_paths: Vec<_> = changes
            .iter()
            .filter(|change| change.strategy() == ApplyStrategy::ReloadModel)
            .map(ConfigChange::path)
            .collect();
        if !model_paths.is_empty() {
            self.start_model_reload(new.clone(), model_paths).await;
        }
        let mut restarted_server: Option<ApplyStatus> = None;

        for change in changes {
            let result = match change.strategy() {
                ApplyStrategy::RestartServer => {
                    let status = match &restarted_server {
                        Some(status) => status.clone(),
                        None => {
                            let status = self.apply_server_restart().await;
                            restarted_server = Some(status.clone());
                            status
                        }
                    };
                    ApplyResult {
                        path: change.path(),
                        strategy: change.strategy(),
                        status,
                    }
                }
                ApplyStrategy::ReloadModel => ApplyResult {
                    path: change.path(),
                    strategy: change.strategy(),
                    status: ApplyStatus::PendingApply,
                },
                ApplyStrategy::RestartProcess => {
                    self.config.set_pending_restart(true).await;
                    ApplyResult {
                        path: change.path(),
                        strategy: change.strategy(),
                        status: ApplyStatus::PendingRestart,
                    }
                }
                _ => self.apply_change(new, change).await,
            };
            results.push(result);
        }
        results
    }

    async fn apply_change(&self, new: &AppConfig, change: &ConfigChange) -> ApplyResult {
        let path = change.path();
        let strategy = change.strategy();
        let status =
            match change {
                ConfigChange::AudioOutputDevice { new: device, .. } => {
                    self.apply_audio_device(device).await
                }
                ConfigChange::Volume { new: volume, .. } => self.apply_volume(*volume).await,
                ConfigChange::LogLevel { .. } => self
                    .apply_log_level(new.logging.filter.as_deref().unwrap_or(&new.logging.level)),
                ConfigChange::RateLimit { new: rl, .. } => {
                    self.rate_limit
                        .swap(crate::api::rate_limit::RateLimiter::with_burst(
                            rl.requests,
                            rl.window_secs,
                            rl.burst,
                        ));
                    ApplyStatus::Applied
                }
                ConfigChange::InferenceLimits {
                    new: (concurrent, pending),
                    ..
                } => {
                    self.inference
                        .swap(crate::api::gate::InferenceGate::new(*concurrent, *pending));
                    ApplyStatus::Applied
                }
                ConfigChange::MaxPlaybackQueueItems { new: max_items, .. } => {
                    self.apply_queue_items(*max_items).await
                }
                ConfigChange::ServerAddress { .. } | ConfigChange::ServerRequestTimeout { .. } => {
                    // Body limits and the request timeout are
                    // router layers: rebuilding the router
                    // (part of the listener restart) applies
                    // them.
                    self.apply_server_restart().await
                }
                ConfigChange::ModelRevision { .. }
                | ConfigChange::ModelCacheDir { .. }
                | ConfigChange::ModelHashesPath { .. } => self.apply_model_reload(new).await,
                ConfigChange::LogToFile { .. } | ConfigChange::LogToStdout { .. } => {
                    self.config.set_pending_restart(true).await;
                    ApplyStatus::PendingRestart
                }
                ConfigChange::Other { path, .. } if path == "logging.filter" => self
                    .apply_log_level(new.logging.filter.as_deref().unwrap_or(&new.logging.level)),
                // Read at request time through the
                // ConfigManager: no subsystem action.
                ConfigChange::AuthMode { .. }
                | ConfigChange::InferenceSteps { .. }
                | ConfigChange::TtsLimits { .. }
                | ConfigChange::Other { .. } => ApplyStatus::PerRequest,
            };
        if let ApplyStatus::Failed(error) = &status {
            tracing::warn!(path = %path, error = %error, "Configuration change failed to apply; previous value remains active");
        }
        ApplyResult {
            path,
            strategy,
            status,
        }
    }

    async fn apply_audio_device(&self, device: &str) -> ApplyStatus {
        #[cfg(feature = "playback")]
        {
            let Some(manager) = self.audio.as_ref().as_ref() else {
                return ApplyStatus::Failed("audio playback system not initialized".to_string());
            };
            match manager.set_output_device(device.to_string()).await {
                Ok(active) => {
                    self.config.clear_apply_failure("audio.output_device").await;
                    tracing::info!(
                        configured = %device,
                        active = %active.device,
                        "Audio output device applied"
                    );
                    ApplyStatus::Applied
                }
                Err(e) => {
                    // Keep the configured value; record the
                    // divergence between configured and
                    // active for the status API.
                    let active = manager.output_device().await.ok();
                    let active_name = active
                        .map(|a| a.device)
                        .unwrap_or_else(|| "unknown".to_string());
                    self.config
                        .record_apply_failure(
                            "audio.output_device",
                            serde_json::json!(device),
                            serde_json::json!(active_name),
                            &e.to_string(),
                        )
                        .await;
                    ApplyStatus::Failed(e.to_string())
                }
            }
        }
        #[cfg(not(feature = "playback"))]
        {
            let _ = device;
            ApplyStatus::Failed("audio playback system not initialized".to_string())
        }
    }

    async fn apply_volume(&self, volume: f32) -> ApplyStatus {
        #[cfg(feature = "playback")]
        {
            let Some(manager) = self.audio.as_ref().as_ref() else {
                return ApplyStatus::Failed("audio playback system not initialized".to_string());
            };
            match manager.set_volume(volume).await {
                Ok(()) => ApplyStatus::Applied,
                Err(e) => ApplyStatus::Failed(e.to_string()),
            }
        }
        #[cfg(not(feature = "playback"))]
        {
            let _ = volume;
            ApplyStatus::Failed("audio playback system not initialized".to_string())
        }
    }

    fn apply_log_level(&self, level: &str) -> ApplyStatus {
        match crate::logging::set_level(level) {
            Ok(()) => {
                tracing::info!(level = %level, "Logging level applied");
                ApplyStatus::Applied
            }
            Err(e) => ApplyStatus::Failed(e),
        }
    }

    async fn apply_queue_items(&self, max_items: usize) -> ApplyStatus {
        #[cfg(feature = "playback")]
        {
            let Some(manager) = self.audio.as_ref().as_ref() else {
                return ApplyStatus::Failed("audio playback system not initialized".to_string());
            };
            match manager.set_max_queue_items(max_items).await {
                Ok(()) => ApplyStatus::Applied,
                Err(e) => ApplyStatus::Failed(e.to_string()),
            }
        }
        #[cfg(not(feature = "playback"))]
        {
            let _ = max_items;
            ApplyStatus::Failed("audio playback system not initialized".to_string())
        }
    }

    async fn apply_server_restart(&self) -> ApplyStatus {
        match self.server.restart().await {
            Ok(()) => ApplyStatus::Applied,
            Err(e) => {
                tracing::error!("HTTP listener restart failed: {e}");
                ApplyStatus::Failed(e.to_string())
            }
        }
    }

    async fn start_model_reload(&self, new: AppConfig, paths: Vec<String>) {
        let revision = new.revision;
        for path in &paths {
            self.config.mark_pending_apply(path, revision).await;
        }
        let model = self.model.clone();
        let config = self.config.clone();
        let secrets = self.secrets.clone();
        tokio::spawn(async move {
            let active = serde_json::to_value(model.active_config().await).unwrap_or_default();
            let result = match crate::config::secrets::load_hf_token(secrets.as_ref()) {
                Ok(token) => model
                    .reload(&new.model, token)
                    .await
                    .map_err(|e| e.to_string()),
                Err(error) => Err(format!("secret store error: {error}")),
            };
            let still_current = serde_json::to_value(&config.get().await.model).ok()
                == serde_json::to_value(&new.model).ok();
            let document = serde_json::to_value(&new).unwrap_or_default();
            for path in &paths {
                config.finish_pending_apply(path, revision).await;
                if still_current {
                    match &result {
                        Ok(()) => config.clear_apply_failure(path).await,
                        Err(error) => {
                            let pointer = format!("/{}", path.replace('.', "/"));
                            config
                                .record_apply_failure(
                                    path,
                                    document.pointer(&pointer).cloned().unwrap_or_default(),
                                    active
                                        .pointer(&format!(
                                            "/{}",
                                            path.strip_prefix("model.").unwrap_or(path)
                                        ))
                                        .cloned()
                                        .unwrap_or_default(),
                                    error,
                                )
                                .await;
                        }
                    }
                }
            }
        });
    }

    async fn apply_model_reload(&self, new: &AppConfig) -> ApplyStatus {
        let hf_token = match crate::config::secrets::load_hf_token(self.secrets.as_ref()) {
            Ok(token) => token,
            Err(e) => return ApplyStatus::Failed(format!("secret store error: {e}")),
        };
        match self.model.reload(&new.model, hf_token).await {
            Ok(()) => ApplyStatus::Applied,
            Err(e) => ApplyStatus::Failed(e.to_string()),
        }
    }
}

/// Background task: apply every committed configuration
/// change to the live subsystems.
pub async fn reconfiguration_loop(
    reconfigurator: Arc<RuntimeReconfigurator>,
    initial_config: AppConfig,
    mut events: tokio::sync::broadcast::Receiver<crate::config::diff::ConfigEvent>,
) {
    let mut last_config = initial_config;
    let mut active = serde_json::to_value(&last_config).unwrap_or_default();
    loop {
        let event = match events.recv().await {
            Ok(event) => event,
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            Err(tokio::sync::broadcast::error::RecvError::Lagged(count)) => {
                tracing::error!(count, "runtime configuration event receiver lagged");
                continue;
            }
        };
        if let crate::config::diff::ConfigEvent::Updated { config: new, .. } = event {
            let changes = crate::config::diff::diff(&last_config, &new);
            last_config = new.clone();
            if changes.is_empty() {
                continue;
            }
            let results = reconfigurator.apply(&new, &changes).await;
            let configured = serde_json::to_value(&new).unwrap_or_default();
            for result in results {
                let pointer = format!("/{}", result.path.replace('.', "/"));
                match &result.status {
                    ApplyStatus::Failed(error) if result.path != "audio.output_device" => {
                        reconfigurator
                            .config
                            .record_apply_failure(
                                &result.path,
                                configured.pointer(&pointer).cloned().unwrap_or_default(),
                                active.pointer(&pointer).cloned().unwrap_or_default(),
                                error,
                            )
                            .await;
                    }
                    ApplyStatus::Applied | ApplyStatus::PerRequest => {
                        reconfigurator
                            .config
                            .clear_apply_failure(&result.path)
                            .await;
                        if let (Some(target), Some(value)) =
                            (active.pointer_mut(&pointer), configured.pointer(&pointer))
                        {
                            *target = value.clone();
                        }
                    }
                    _ => {}
                }
                match &result.status {
                    ApplyStatus::Applied => {
                        tracing::debug!(
                            path = %result.path,
                            strategy = %result.strategy.as_str(),
                            "Configuration change applied"
                        );
                    }
                    ApplyStatus::PerRequest => {
                        tracing::debug!(
                            path = %result.path,
                            "Configuration change takes effect for new requests"
                        );
                    }
                    ApplyStatus::Failed(error) => {
                        tracing::warn!(
                            path = %result.path,
                            error = %error,
                            "Configuration change could not be applied; \
                             the previous runtime value remains active"
                        );
                    }
                    ApplyStatus::PendingApply => {
                        tracing::info!(path = %result.path, "Subsystem reload started in background");
                    }
                    ApplyStatus::PendingRestart => {
                        tracing::info!(
                            path = %result.path,
                            "Configuration change requires a process \
                             restart to take effect"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "playback")]
    #[tokio::test]
    async fn failed_audio_apply_preserves_configured_and_active_values() {
        use crate::tts::devices::ActiveOutputDevice;
        use crate::tts::queue::{AudioCommand, AudioManagerError};
        let mut initial = AppConfig::default();
        initial.paths.audio = Some("/tmp".into());
        let config = ConfigManager::in_memory(initial);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let audio = Arc::new(Some(AudioManager::for_test(tx)));
        let status = Arc::new(tokio::sync::RwLock::new(crate::tts::ModelStatus::Idle));
        let model = Arc::new(ModelService::new(status.clone()));
        let rate_limit = Arc::new(RateLimiterManager::new(
            crate::api::rate_limit::RateLimiter::new(0, 60),
        ));
        let inference = Arc::new(InferenceGateManager::new(
            crate::api::gate::InferenceGate::new(1, 8),
        ));
        let secrets: Arc<dyn SecretStore> =
            Arc::new(crate::config::secrets::MemorySecretStore::default());
        let tokens = Arc::new(crate::auth::store::TokenStore::empty());
        let state = crate::AppState {
            config: config.clone(),
            audio_manager: audio.clone(),
            model_status: status,
            token_store: tokens.clone(),
            model_service: model.clone(),
            secrets: secrets.clone(),
            rate_limiter: rate_limit.clone(),
            inference_gate: inference.clone(),
        };
        let admin = crate::admin::handlers::AdminState {
            config: config.clone(),
            secrets: secrets.clone(),
            token_store: tokens,
            lockout: Arc::new(crate::admin::lockout::LoginAttemptTracker::default()),
        };
        let server = Arc::new(ApiServerManager::new(state, admin));
        let coordinator = RuntimeReconfigurator::new(
            config.clone(),
            audio,
            server,
            model,
            rate_limit,
            inference,
            secrets,
        );
        let fake_hardware = tokio::spawn(async move {
            match rx.recv().await.unwrap() {
                AudioCommand::SetOutputDevice { device, reply } => {
                    assert_eq!(device, "Missing Device");
                    reply.send(Err(AudioManagerError::UnknownDevice)).unwrap();
                }
                _ => panic!("expected device switch"),
            }
            match rx.recv().await.unwrap() {
                AudioCommand::GetOutputDevice { reply } => {
                    reply
                        .send(ActiveOutputDevice {
                            device: "default".into(),
                            resolved_name: Some("Speakers".into()),
                            available: true,
                        })
                        .unwrap();
                }
                _ => panic!("expected active device query"),
            }
        });
        let update = config
            .update(None, |c| c.audio.output_device = "Missing Device".into())
            .await
            .unwrap();
        let results = coordinator
            .apply(&config.get().await, &update.changes)
            .await;
        assert!(matches!(results[0].status, ApplyStatus::Failed(_)));
        let status = config.status().await;
        assert_eq!(config.get().await.audio.output_device, "Missing Device");
        assert_eq!(status.apply_failures[0].configured, "Missing Device");
        assert_eq!(status.apply_failures[0].active, "default");
        fake_hardware.await.unwrap();
    }

    #[test]
    fn apply_result_paths_match_changes() {
        let result = ApplyResult {
            path: "audio.output_device".to_string(),
            strategy: ApplyStrategy::Hot,
            status: ApplyStatus::Applied,
        };
        assert_eq!(result.path, "audio.output_device");
        assert_eq!(result.strategy.as_str(), "hot");
    }
}
