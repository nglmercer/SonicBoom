//! Swappable runtime subsystems.
//!
//! Wrappers around the rate limiter and inference gate
//! so configuration changes can atomically replace the
//! live instance without touching request paths.

use std::sync::{Arc, RwLock};

use crate::api::gate::{InferenceGate, InferencePermit, RunBoundedError};
use crate::api::rate_limit::{RateLimitDecision, RateLimiter};

/// Hot-swappable rate limiter.
///
/// The current limiter is read under a short-lived
/// `std::sync` lock on every check; configuration
/// changes swap the whole limiter (the token buckets
/// reset, which is the documented behavior of a
/// rate-limit change).
#[derive(Debug)]
pub struct RateLimiterManager {
    current: RwLock<Arc<RateLimiter>>,
}

impl RateLimiterManager {
    pub fn new(limiter: RateLimiter) -> Self {
        Self {
            current: RwLock::new(Arc::new(limiter)),
        }
    }

    /// Check the budget for `key`, recording the request
    /// when allowed.
    pub fn check(&self, key: &str) -> RateLimitDecision {
        // A read lock cannot be held across the check
        // (it is synchronous and non-blocking), so this
        // is a plain critical section.
        let limiter = self.current.read().unwrap_or_else(|e| e.into_inner());
        limiter.check(key)
    }

    /// Whether the request is allowed (and recorded).
    pub fn allow(&self, key: &str) -> bool {
        self.check(key).allowed
    }

    /// Replace the live limiter.
    pub fn swap(&self, limiter: RateLimiter) {
        let mut current = self.current.write().unwrap_or_else(|e| e.into_inner());
        *current = Arc::new(limiter);
    }

    /// The live limiter (for tests).
    pub fn current(&self) -> Arc<RateLimiter> {
        self.current
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// Hot-swappable inference admission gate.
#[derive(Debug, Clone)]
pub struct InferenceGateManager {
    current: Arc<RwLock<Arc<InferenceGate>>>,
}

impl InferenceGateManager {
    pub fn new(gate: InferenceGate) -> Self {
        Self {
            current: Arc::new(RwLock::new(Arc::new(gate))),
        }
        // Note: std RwLock would suffice, but the gate is
        // awaited across .await points, so the tokio-free
        // std lock is only held for the clone below.
    }

    /// Try to admit one request.
    pub async fn admit(&self) -> Option<InferencePermit> {
        let gate = self
            .current
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        gate.admit().await
    }

    /// Admit one request and run blocking work with the
    /// permit owned by the blocking task itself.
    pub async fn run_bounded<F, T>(&self, work: F) -> Result<T, RunBoundedError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let gate = self
            .current
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        gate.run_bounded(work).await
    }

    /// Replace the live gate (waiting slots reset).
    pub fn swap(&self, gate: InferenceGate) {
        let mut current = self.current.write().unwrap_or_else(|e| e.into_inner());
        *current = Arc::new(gate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_manager_swaps() {
        let manager = RateLimiterManager::new(RateLimiter::new(2, 60));
        assert!(manager.allow("k"));
        assert!(manager.allow("k"));
        assert!(!manager.allow("k"));
        // Swap resets budgets with the new configuration.
        manager.swap(RateLimiter::new(5, 60));
        assert!(manager.allow("k"));
        assert_eq!(manager.current().limit(), 5);
    }

    #[tokio::test]
    async fn inference_gate_manager_swaps() {
        let manager = InferenceGateManager::new(InferenceGate::new(1, 0));
        let _held = manager.admit().await.expect("admitted");
        assert!(manager.admit().await.is_none());
        manager.swap(InferenceGate::new(2, 0));
        assert!(manager.admit().await.is_some());
    }

    #[tokio::test]
    async fn inference_gate_manager_runs_bounded() {
        let manager = InferenceGateManager::new(InferenceGate::new(1, 0));
        let result = manager.run_bounded(|| 42).await.unwrap();
        assert_eq!(result, 42);
    }
}
