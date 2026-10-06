//! Model lifecycle service.
//!
//! Owns the model status (idle/downloading/loading/
//! ready/failed) and the initial download+load flow.
//! A configuration change to `model.revision`,
//! `model.cache_dir`, or `model.hashes_path` triggers
//! [`ModelService::reload`], which downloads and
//! validates a candidate **before** swapping it into
//! the live status — the working model is never taken
//! offline for a failed candidate.

use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use crate::config::model::ModelConfig;
use crate::tts::{ModelStatus, download, model::ModelHandle};

/// Failure to prepare a model candidate.
#[derive(Debug, thiserror::Error)]
pub enum ModelServiceError {
    #[error("trust resolution failed: {0}")]
    Trust(#[from] anyhow::Error),
    #[error("download failed: {0}")]
    Download(String),
    #[error("model load failed: {0}")]
    Load(String),
    #[error("model reload superseded by a newer configuration")]
    Superseded,
}

/// Model lifecycle owner.
#[derive(Clone)]
pub struct ModelService {
    lifecycle: Arc<Mutex<()>>,
    requested_generation: Arc<std::sync::atomic::AtomicU64>,
    active_config: Arc<RwLock<Option<ModelConfig>>>,
    status: Arc<RwLock<ModelStatus>>,
}

impl ModelService {
    pub fn new(status: Arc<RwLock<ModelStatus>>) -> Self {
        Self {
            status,
            lifecycle: Arc::new(Mutex::new(())),
            requested_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            active_config: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn active_config(&self) -> Option<ModelConfig> {
        self.active_config.read().await.clone()
    }

    /// Shared status handle (for `AppState`).
    pub fn status_handle(&self) -> Arc<RwLock<ModelStatus>> {
        Arc::clone(&self.status)
    }

    /// Snapshot the current status.
    pub async fn status(&self) -> ModelStatus {
        self.status.read().await.clone()
    }

    /// Resolve trust for a model configuration.
    fn resolve_trust(
        model: &ModelConfig,
    ) -> Result<(String, download::ExpectedTrust), anyhow::Error> {
        let hashes_path = model
            .hashes_path
            .as_deref()
            .map(|p| p.to_string_lossy().to_string());
        download::resolve_trust(&model.revision, hashes_path.as_deref())
    }

    /// Start the initial download+load in the background
    /// (startup path). Progress is reported through the
    /// shared status. `hf_token` is used for gated or
    /// private model repositories.
    pub fn start_initial_load(&self, model: &ModelConfig, hf_token: Option<String>) {
        let service = self.clone();
        let model = model.clone();
        tokio::spawn(async move {
            if let Err(error) = service.reload(&model, hf_token).await {
                tracing::error!("Initial model preparation failed: {error}");
                let mut status = service.status.write().await;
                if !matches!(*status, ModelStatus::Ready(_)) {
                    *status = ModelStatus::Failed(error.to_string());
                }
            }
        });
    }

    /// Reload the model from a new configuration.
    ///
    /// Downloads and validates the candidate; only when
    /// the candidate is fully loaded does it replace the
    /// live model. A failure leaves the previous model
    /// active and returns the error.
    pub async fn reload(
        &self,
        model: &ModelConfig,
        hf_token: Option<String>,
    ) -> Result<(), ModelServiceError> {
        let generation = self
            .requested_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        let _lifecycle = self.lifecycle.lock().await;
        let (model_revision, expected_trust) =
            Self::resolve_trust(model).map_err(ModelServiceError::Trust)?;
        let download_limits = download::DownloadLimits::new(
            model.download_connect_timeout_secs,
            model.download_timeout_secs,
        );
        let cache_dir = model.cache_dir.clone();
        let status = Arc::clone(&self.status);

        // Download the candidate, reporting progress
        // without clobbering a Ready status into a
        // misleading state: keep the old status visible
        // until the candidate is loaded.
        let previous = self.status().await;
        let keep_ready = matches!(previous, ModelStatus::Ready(_));
        if !keep_ready {
            *status.write().await = ModelStatus::Downloading { progress: 0.0 };
        }
        let status_for_progress = Arc::clone(&status);
        let paths = match download::download_models_with_options(
            cache_dir.as_path(),
            hf_token.as_deref(),
            &model_revision,
            &expected_trust,
            &download_limits,
            move |progress| {
                if !keep_ready && let Ok(mut status) = status_for_progress.try_write() {
                    *status = ModelStatus::Downloading { progress };
                }
            },
        )
        .await
        {
            Ok(paths) => paths,
            Err(e) => {
                tracing::error!("Model reload download failed: {e}");
                // Restore the previous status: the old
                // model is still the live one.
                *status.write().await = previous.clone();
                return Err(ModelServiceError::Download(e.to_string()));
            }
        };

        if !keep_ready {
            *status.write().await = ModelStatus::Loading;
        }
        let handle = match tokio::task::spawn_blocking(move || ModelHandle::load(&paths)).await {
            Ok(Ok(handle)) => handle,
            Ok(Err(e)) => {
                tracing::error!("Model reload load failed: {e}");
                *status.write().await = previous.clone();
                return Err(ModelServiceError::Load(e.to_string()));
            }
            Err(e) => {
                tracing::error!("Model reload load task panic: {e}");
                *status.write().await = previous.clone();
                return Err(ModelServiceError::Load(e.to_string()));
            }
        };

        if self
            .requested_generation
            .load(std::sync::atomic::Ordering::SeqCst)
            != generation
        {
            *status.write().await = previous;
            return Err(ModelServiceError::Superseded);
        }
        *self.active_config.write().await = Some(model.clone());
        // Candidate is fully loaded: atomic swap.
        *status.write().await = ModelStatus::Ready(Arc::new(handle));
        tracing::info!(revision = %model_revision, "Model reloaded");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{
        DEFAULT_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS, DEFAULT_MODEL_DOWNLOAD_TIMEOUT_SECS,
        DEFAULT_MODEL_REVISION,
    };

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            cache_dir: std::path::PathBuf::from("./models"),
            revision: DEFAULT_MODEL_REVISION.to_string(),
            hashes_path: None,
            inference_steps: 5,
            download_connect_timeout_secs: DEFAULT_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS,
            download_timeout_secs: DEFAULT_MODEL_DOWNLOAD_TIMEOUT_SECS,
        }
    }

    #[tokio::test]
    async fn status_round_trips() {
        let status = Arc::new(RwLock::new(ModelStatus::Idle));
        let service = ModelService::new(Arc::clone(&status));
        assert!(matches!(service.status().await, ModelStatus::Idle));
        *status.write().await = ModelStatus::Loading;
        assert!(matches!(service.status().await, ModelStatus::Loading));
        assert!(Arc::ptr_eq(&service.status_handle(), &status));
    }

    #[tokio::test]
    async fn reload_failure_restores_previous_status() {
        // Any non-Downloading state works as the "previous"
        // state: the point is that a failed reload restores it.
        let status = Arc::new(RwLock::new(ModelStatus::Failed("previous".to_string())));
        let service = ModelService::new(Arc::clone(&status));
        // Point at a bogus cache dir so the download fails.
        let mut model = test_model_config();
        model.cache_dir = std::path::PathBuf::from("/nonexistent-sonicboom-models");
        let result = service.reload(&model, None).await;
        assert!(result.is_err());
        // The previous status is restored (not Downloading).
        let current = service.status().await;
        assert!(
            !matches!(current, ModelStatus::Downloading { .. }),
            "failed reload must not leave Downloading status"
        );
    }
}
