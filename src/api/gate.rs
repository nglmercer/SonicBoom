use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Bounded admission control for expensive model inference.
///
/// Two semaphores bound the two resources independently:
/// - `running`: at most `max_concurrent` inferences execute at once
///   (model inference is effectively serialized; default 1).
/// - `admitted`: at most `max_concurrent + max_pending` requests may hold a
///   slot (running or waiting). Admission uses `try_acquire`: when the queue
///   is full the request is rejected immediately with `429/503` instead of
///   piling up unbounded `spawn_blocking` tasks.
///
/// Usage: call [`InferenceGate::admit`] after authentication and request
/// validation, hold the returned permit across `spawn_blocking` + inference,
/// then drop it.
#[derive(Clone, Debug)]
pub struct InferenceGate {
    running: Arc<Semaphore>,
    admitted: Arc<Semaphore>,
}

pub struct InferencePermit {
    _running: OwnedSemaphorePermit,
    _admitted: OwnedSemaphorePermit,
}

impl InferenceGate {
    pub fn new(max_concurrent: usize, max_pending: usize) -> Self {
        let max_concurrent = max_concurrent.max(1);
        Self {
            running: Arc::new(Semaphore::new(max_concurrent)),
            admitted: Arc::new(Semaphore::new(max_concurrent + max_pending)),
        }
    }

    /// Try to admit one request. Returns `None` when saturated (caller
    /// should respond `429 Too Many Requests`).
    pub async fn admit(&self) -> Option<InferencePermit> {
        let admitted = self.admitted.clone().try_acquire_owned().ok()?;
        // At most `max_pending` waiters can be here, so this wait is bounded.
        let running = self.running.clone().acquire_owned().await.ok()?;
        Some(InferencePermit {
            _running: running,
            _admitted: admitted,
        })
    }

    #[cfg(test)]
    pub fn available_admissions(&self) -> usize {
        self.admitted.available_permits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn admits_up_to_capacity_then_rejects() {
        let gate = InferenceGate::new(1, 1);
        let _first = gate.admit().await.expect("first admitted");
        // Second admission occupies the running slot waiter (pending=1).
        let second = tokio::spawn({
            let gate = gate.clone();
            async move { gate.admit().await.is_some() }
        });
        // Give the waiter a moment to block on `running`.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // Capacity is 1 running + 1 pending: a third request must be rejected.
        assert!(
            gate.admit().await.is_none(),
            "saturated gate must reject admission"
        );
        drop(_first);
        assert!(second.await.unwrap(), "waiter proceeds after release");
    }

    #[tokio::test]
    async fn permits_are_released_on_drop() {
        let gate = InferenceGate::new(1, 0);
        assert_eq!(gate.available_admissions(), 1);
        {
            let _permit = gate.admit().await.expect("admitted");
            assert_eq!(gate.available_admissions(), 0);
            assert!(gate.admit().await.is_none());
        }
        assert_eq!(gate.available_admissions(), 1);
        assert!(gate.admit().await.is_some());
    }
}
