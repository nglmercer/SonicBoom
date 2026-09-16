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
/// Cancellation safety: the permit must be owned by the blocking inference
/// task itself (see [`InferenceGate::run_bounded`]), never by the async HTTP
/// request future. If the request future is cancelled by the timeout layer
/// while `spawn_blocking` work is still running, a request-owned permit
/// would be dropped early and the concurrency bound would leak. Moving the
/// permit into the blocking closure ties its lifetime to actual inference.
#[derive(Clone, Debug)]
pub struct InferenceGate {
    running: Arc<Semaphore>,
    admitted: Arc<Semaphore>,
}

pub struct InferencePermit {
    _running: OwnedSemaphorePermit,
    _admitted: OwnedSemaphorePermit,
}

/// Failure to run bounded inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunBoundedError {
    /// The gate is saturated; the caller should respond `429`.
    Saturated,
    /// The blocking task panicked or was otherwise lost.
    JoinFailed(String),
}

impl InferenceGate {
    pub fn new(max_concurrent: usize, max_pending: usize) -> Self {
        let max_concurrent = max_concurrent.max(1);
        // Checked arithmetic: operator input must never overflow or panic
        // semaphore construction (config validation bounds these far below
        // any overflow, but the gate stays total regardless).
        let total = max_concurrent
            .saturating_add(max_pending)
            .clamp(1, Semaphore::MAX_PERMITS);
        Self {
            running: Arc::new(Semaphore::new(max_concurrent.min(Semaphore::MAX_PERMITS))),
            admitted: Arc::new(Semaphore::new(total)),
        }
    }

    /// Try to admit one request. Returns `None` when saturated (caller
    /// should respond `429 Too Many Requests`).
    ///
    /// Prefer [`InferenceGate::run_bounded`]: a permit held across `.await`
    /// points in the request future is released early if the request is
    /// cancelled. Only use `admit` directly when the permit is immediately
    /// moved into the blocking task.
    pub async fn admit(&self) -> Option<InferencePermit> {
        let admitted = self.admitted.clone().try_acquire_owned().ok()?;
        // At most `max_pending` waiters can be here, so this wait is bounded.
        let running = self.running.clone().acquire_owned().await.ok()?;
        Some(InferencePermit {
            _running: running,
            _admitted: admitted,
        })
    }

    /// Admit one request and run blocking work with the permit owned by the
    /// blocking task itself.
    ///
    /// The permit is moved into the `spawn_blocking` closure, so cancelling
    /// the returned future (e.g. via the HTTP request timeout) does **not**
    /// release the slot early: the slot is freed only when the blocking
    /// work actually finishes. All HTTP inference paths must use this
    /// helper instead of holding a permit in the request future.
    pub async fn run_bounded<F, T>(&self, work: F) -> Result<T, RunBoundedError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let permit = self.admit().await.ok_or(RunBoundedError::Saturated)?;
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            work()
        })
        .await
        .map_err(|e| RunBoundedError::JoinFailed(e.to_string()))
    }

    #[cfg(test)]
    pub fn available_admissions(&self) -> usize {
        self.admitted.available_permits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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
        tokio::time::sleep(Duration::from_millis(50)).await;
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

    #[tokio::test]
    async fn semaphore_arithmetic_cannot_overflow_or_panic() {
        // Absurd operator input must not panic semaphore construction.
        let gate = InferenceGate::new(usize::MAX, usize::MAX);
        assert!(gate.admit().await.is_some());
        let gate = InferenceGate::new(0, usize::MAX);
        assert!(gate.admit().await.is_some());
    }

    /// Simulate an HTTP request timeout: the awaiting future is aborted
    /// while blocking work still runs. The running slot must stay occupied
    /// until the blocking task finishes.
    #[tokio::test]
    async fn cancellation_does_not_release_running_permit() {
        let gate = InferenceGate::new(1, 0);
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

        let request = tokio::spawn({
            let gate = gate.clone();
            async move {
                gate.run_bounded(move || {
                    let _ = started_tx.send(());
                    // Block until the test allows completion.
                    let _ = release_rx.recv();
                    42
                })
                .await
            }
        });

        // Wait until the blocking task owns the permit.
        tokio::task::spawn_blocking(move || {
            started_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("blocking work started");
        })
        .await
        .unwrap();
        assert!(
            gate.admit().await.is_none(),
            "slot must be occupied while blocking work runs"
        );

        // Simulate the request timeout: drop/abort the request-side future.
        request.abort();
        let _ = request.await;
        // Give the abort a moment to settle; the blocking thread still runs.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            gate.admit().await.is_none(),
            "HTTP cancellation must not release a running inference permit"
        );

        // Let the blocking task finish; only now is the slot freed.
        drop(release_tx);
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if gate.admit().await.is_some() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("permit released after blocking work finished");
    }

    /// Same cancellation property with a pending queue configured: while
    /// one inference runs and one waits, saturation rejects newcomers, and
    /// aborting the waiter frees only the pending slot.
    #[tokio::test]
    async fn cancellation_with_pending_queue_keeps_running_slot() {
        let gate = InferenceGate::new(1, 1);
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

        let running = tokio::spawn({
            let gate = gate.clone();
            async move {
                gate.run_bounded(move || {
                    let _ = started_tx.send(());
                    let _ = release_rx.recv();
                    1
                })
                .await
            }
        });
        tokio::task::spawn_blocking(move || {
            started_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("blocking work started");
        })
        .await
        .unwrap();

        // A second request occupies the single pending slot.
        let waiter = tokio::spawn({
            let gate = gate.clone();
            async move { gate.run_bounded(|| 2).await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        // 1 running + 1 pending: newcomers are rejected.
        assert!(gate.admit().await.is_none());

        // Abort the pending waiter (its request timed out). Its admitted
        // slot is freed, but the running inference still holds `running`.
        waiter.abort();
        let _ = waiter.await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        // A new waiter may now queue (pending slot free) but must block
        // until the running inference finishes.
        let probe = tokio::spawn({
            let gate = gate.clone();
            async move { gate.run_bounded(|| 3).await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!probe.is_finished(), "running slot still held");

        drop(release_tx);
        let first = tokio::time::timeout(Duration::from_secs(10), running)
            .await
            .expect("running inference finished")
            .expect("join ok");
        assert_eq!(first, Ok(1));
        let third = tokio::time::timeout(Duration::from_secs(10), probe)
            .await
            .expect("queued inference proceeded")
            .expect("join ok");
        assert_eq!(third, Ok(3));
    }

    #[tokio::test]
    async fn saturated_run_bounded_rejects_immediately() {
        let gate = InferenceGate::new(1, 0);
        let _held = gate.admit().await.expect("admitted");
        assert_eq!(
            gate.run_bounded(|| 1).await,
            Err(RunBoundedError::Saturated)
        );
    }
}
