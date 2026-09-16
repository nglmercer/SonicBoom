use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
    time::{Duration, Instant},
};

/// Sliding-window rate limiter keyed by caller identity (bearer-token hash).
///
/// This complements — not replaces — the bounded [`crate::api::gate::InferenceGate`].
#[derive(Debug)]
pub struct RateLimiter {
    max_requests: u32,
    window: Duration,
    /// Token identity -> timestamps of recent requests. Bounded: entries are
    /// evicted when their window expires and the map is capped.
    hits: Mutex<HashMap<String, VecDeque<Instant>>>,
    max_keys: usize,
    /// Last full-map expiry sweep. Sweeps are throttled so cleanup under
    /// large key churn cannot create latency spikes on the global mutex.
    last_sweep: Mutex<Instant>,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self {
            max_requests,
            window: Duration::from_secs(window_secs.max(1)),
            hits: Mutex::new(HashMap::new()),
            max_keys: 100_000,
            last_sweep: Mutex::new(Instant::now()),
        }
    }

    /// Returns `true` when the request is allowed (and records it),
    /// `false` when the caller exhausted its window budget.
    pub fn allow(&self, key: &str) -> bool {
        if self.max_requests == 0 {
            return true; // 0 disables rate limiting.
        }
        let now = Instant::now();
        let mut hits = self.hits.lock().unwrap_or_else(|e| e.into_inner());
        // Incremental expiry: once the map is half full, sweep expired keys
        // at most once per window so no single request pays a full retain
        // over 100k keys while the mutex is held.
        if hits.len() >= self.max_keys / 2
            && let Ok(mut last_sweep) = self.last_sweep.try_lock()
            && now.duration_since(*last_sweep) >= self.window.min(Duration::from_secs(60))
        {
            hits.retain(|_, times| {
                times.retain(|t| now.duration_since(*t) < self.window);
                !times.is_empty()
            });
            *last_sweep = now;
        }
        // Opportunistic eviction of fully-expired keys when at capacity.
        if hits.len() >= self.max_keys {
            hits.retain(|_, times| {
                times.retain(|t| now.duration_since(*t) < self.window);
                !times.is_empty()
            });
            if hits.len() >= self.max_keys {
                // Still full under pressure: fail closed for new keys.
                if !hits.contains_key(key) {
                    return false;
                }
            }
        }
        let times = hits.entry(key.to_string()).or_default();
        while let Some(front) = times.front() {
            if now.duration_since(*front) < self.window {
                break;
            }
            times.pop_front();
        }
        if times.len() >= self.max_requests as usize {
            return false;
        }
        times.push_back(now);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_up_to_limit_then_blocks() {
        let limiter = RateLimiter::new(3, 60);
        assert!(limiter.allow("token-a"));
        assert!(limiter.allow("token-a"));
        assert!(limiter.allow("token-a"));
        assert!(!limiter.allow("token-a"));
        // Other tokens have their own budget.
        assert!(limiter.allow("token-b"));
    }

    #[test]
    fn window_expiry_restores_budget() {
        let limiter = RateLimiter::new(1, 60);
        // Push an expired timestamp directly.
        {
            let mut hits = limiter.hits.lock().unwrap();
            let mut queue = VecDeque::new();
            queue.push_back(Instant::now() - Duration::from_secs(61));
            hits.insert("token-a".to_string(), queue);
        }
        assert!(limiter.allow("token-a"));
    }

    #[test]
    fn periodic_sweep_reclaims_expired_keys_before_capacity() {
        let limiter = RateLimiter::new(5, 60);
        {
            let mut hits = limiter.hits.lock().unwrap();
            for i in 0..60_000 {
                let mut queue = VecDeque::new();
                queue.push_back(Instant::now() - Duration::from_secs(61));
                hits.insert(format!("stale-{i}"), queue);
            }
            *limiter.last_sweep.lock().unwrap() = Instant::now() - Duration::from_secs(61);
        }
        assert!(limiter.allow("fresh"));
        assert!(
            limiter.hits.lock().unwrap().len() < 100,
            "throttled sweep should have reclaimed expired keys"
        );
    }

    #[test]
    fn zero_disables_limiting() {
        let limiter = RateLimiter::new(0, 60);
        for _ in 0..100 {
            assert!(limiter.allow("token-a"));
        }
    }
}
