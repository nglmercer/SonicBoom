use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Duration, Instant},
};

/// Outcome of a rate-limit check, carrying the retry metadata served to the
/// caller on `429` (`Retry-After`, `X-RateLimit-*`). All values derive from
/// the actual bucket state — never hardcoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitDecision {
    /// Whether the request is allowed (and was recorded).
    pub allowed: bool,
    /// Bucket capacity: the maximum burst for this key.
    pub limit: u32,
    /// Whole tokens remaining after this decision (`0` when rejected).
    pub remaining: u32,
    /// Seconds until at least one token is available (`0` when allowed).
    /// Always `>= 1` when rejected so `Retry-After` is meaningful.
    pub retry_after_secs: u64,
    /// Seconds until the bucket refills completely. Served as
    /// `X-RateLimit-Reset` (seconds-from-now, not a timestamp).
    pub reset_secs: u64,
}

/// Per-key token bucket state.
#[derive(Debug)]
struct Bucket {
    /// Currently available tokens (fractional; capped at capacity).
    tokens: f64,
    /// Last time the bucket was refilled.
    last_refill: Instant,
}

/// Burst-friendly token-bucket rate limiter keyed by caller identity
/// (bearer-token hash).
///
/// - `capacity` (burst) bounds how many requests an event burst may fire at
///   once; it defaults to the sustained budget.
/// - `refill_rate` (`max_requests` per `window`) bounds the sustained request
///   volume that refills over time.
///
/// This complements — not replaces — the bounded [`crate::api::gate::InferenceGate`].
#[derive(Debug)]
pub struct RateLimiter {
    max_requests: u32,
    window: Duration,
    /// Sustained refill rate in tokens per second.
    refill_per_sec: f64,
    /// Bucket capacity (maximum burst).
    capacity: f64,
    /// Token identity -> bucket. Bounded: fully-refilled idle buckets are
    /// evicted on sweep and the map is capped.
    buckets: Mutex<HashMap<String, Bucket>>,
    max_keys: usize,
    /// Last full-map expiry sweep. Sweeps are throttled so cleanup under
    /// large key churn cannot create latency spikes on the global mutex.
    last_sweep: Mutex<Instant>,
}

impl RateLimiter {
    /// Sustained budget of `max_requests` per `window_secs`, with the burst
    /// capacity derived from `max_requests` (one full window may arrive
    /// at once). `max_requests == 0` disables limiting.
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self::with_burst(max_requests, window_secs, max_requests)
    }

    /// Sustained budget of `max_requests` per `window_secs` with an explicit
    /// burst capacity. A `burst` of `0` falls back to `max_requests`
    /// (or `1` when the limiter is disabled, where the value is unused).
    pub fn with_burst(max_requests: u32, window_secs: u64, burst: u32) -> Self {
        let window = Duration::from_secs(window_secs.max(1));
        let capacity = if max_requests == 0 {
            burst.max(1) as f64
        } else if burst == 0 {
            max_requests as f64
        } else {
            burst as f64
        };
        let refill_per_sec = if max_requests == 0 {
            0.0
        } else {
            max_requests as f64 / window.as_secs_f64()
        };
        Self {
            max_requests,
            window,
            refill_per_sec,
            capacity,
            buckets: Mutex::new(HashMap::new()),
            max_keys: 100_000,
            last_sweep: Mutex::new(Instant::now()),
        }
    }

    /// Bucket capacity served as `X-RateLimit-Limit`.
    pub fn limit(&self) -> u32 {
        if self.max_requests == 0 {
            0
        } else {
            self.capacity as u32
        }
    }

    fn disabled_decision(&self) -> RateLimitDecision {
        RateLimitDecision {
            allowed: true,
            limit: 0,
            remaining: u32::MAX,
            retry_after_secs: 0,
            reset_secs: 0,
        }
    }

    /// Check the budget for `key`, recording the request when allowed.
    pub fn check(&self, key: &str) -> RateLimitDecision {
        if self.max_requests == 0 {
            return self.disabled_decision(); // 0 disables rate limiting.
        }
        let now = Instant::now();
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        // Incremental expiry: once the map is half full, sweep fully-refilled
        // idle buckets at most once per window so no single request pays a
        // full retain over 100k keys while the mutex is held. A full bucket
        // is indistinguishable from a missing one, so eviction is lossless.
        if buckets.len() >= self.max_keys / 2
            && let Ok(mut last_sweep) = self.last_sweep.try_lock()
            && now.duration_since(*last_sweep) >= self.window.min(Duration::from_secs(60))
        {
            let capacity = self.capacity;
            let rate = self.refill_per_sec;
            buckets.retain(|_, bucket| refill(bucket, capacity, rate, now) < capacity);
            *last_sweep = now;
        }
        // Opportunistic eviction of fully-refilled buckets when at capacity.
        if buckets.len() >= self.max_keys {
            let capacity = self.capacity;
            let rate = self.refill_per_sec;
            buckets.retain(|_, bucket| refill(bucket, capacity, rate, now) < capacity);
            if buckets.len() >= self.max_keys && !buckets.contains_key(key) {
                // Still full under pressure: fail closed for new keys.
                return RateLimitDecision {
                    allowed: false,
                    limit: self.limit(),
                    remaining: 0,
                    retry_after_secs: 1,
                    reset_secs: self.window.as_secs().max(1),
                };
            }
        }
        let bucket = buckets.entry(key.to_string()).or_insert(Bucket {
            tokens: self.capacity,
            last_refill: now,
        });
        refill(bucket, self.capacity, self.refill_per_sec, now);
        let limit = self.limit();
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            let remaining = bucket.tokens.floor().max(0.0) as u32;
            RateLimitDecision {
                allowed: true,
                limit,
                remaining,
                retry_after_secs: 0,
                reset_secs: secs_until(bucket.tokens, self.capacity, self.refill_per_sec),
            }
        } else {
            RateLimitDecision {
                allowed: false,
                limit,
                remaining: 0,
                retry_after_secs: secs_until(bucket.tokens, 1.0, self.refill_per_sec).max(1),
                reset_secs: secs_until(bucket.tokens, self.capacity, self.refill_per_sec).max(1),
            }
        }
    }

    /// Returns `true` when the request is allowed (and records it),
    /// `false` when the caller exhausted its budget. Prefer [`RateLimiter::check`]
    /// when retry metadata is needed.
    pub fn allow(&self, key: &str) -> bool {
        self.check(key).allowed
    }
}

/// Refill `bucket` to `now`, capped at `capacity`. Returns the new balance.
fn refill(bucket: &mut Bucket, capacity: f64, rate_per_sec: f64, now: Instant) -> f64 {
    let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
    if elapsed > 0.0 {
        bucket.tokens = (bucket.tokens + elapsed * rate_per_sec).min(capacity);
        bucket.last_refill = now;
    }
    bucket.tokens
}

/// Whole seconds until `tokens` reaches `target` at `rate_per_sec`.
fn secs_until(tokens: f64, target: f64, rate_per_sec: f64) -> u64 {
    if tokens >= target || rate_per_sec <= 0.0 {
        return 0;
    }
    ((target - tokens) / rate_per_sec).ceil().max(1.0) as u64
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
    fn rejection_carries_retry_metadata_from_bucket_state() {
        let limiter = RateLimiter::new(2, 60);
        let first = limiter.check("token-a");
        assert!(first.allowed);
        assert_eq!(first.limit, 2);
        assert_eq!(first.remaining, 1);
        let second = limiter.check("token-a");
        assert!(second.allowed);
        assert_eq!(second.remaining, 0);
        let rejected = limiter.check("token-a");
        assert!(!rejected.allowed);
        assert_eq!(rejected.limit, 2);
        assert_eq!(rejected.remaining, 0);
        // 2 per 60s refills one token every 30s.
        assert_eq!(rejected.retry_after_secs, 30);
        assert!(rejected.reset_secs >= rejected.retry_after_secs);
    }

    #[test]
    fn refill_restores_budget_over_time() {
        let limiter = RateLimiter::new(60, 60); // 1 token/sec, burst 60
        for _ in 0..60 {
            assert!(limiter.allow("token-a"));
        }
        assert!(!limiter.allow("token-a"));
        // Age the bucket 2 seconds: ~2 tokens refill.
        {
            let mut buckets = limiter.buckets.lock().unwrap();
            let bucket = buckets.get_mut("token-a").unwrap();
            bucket.last_refill -= Duration::from_secs(2);
        }
        assert!(limiter.allow("token-a"));
        assert!(limiter.allow("token-a"));
        assert!(!limiter.allow("token-a"));
    }

    #[test]
    fn explicit_burst_allows_event_bursts_above_sustained_rate() {
        // Sustained 2/min but bursts of 5 at once.
        let limiter = RateLimiter::with_burst(2, 60, 5);
        for _ in 0..5 {
            assert!(limiter.allow("token-a"));
        }
        let rejected = limiter.check("token-a");
        assert!(!rejected.allowed);
        assert_eq!(rejected.limit, 5);
        // Sustained rate is 2/min: one token every 30s.
        assert_eq!(rejected.retry_after_secs, 30);
    }

    #[test]
    fn zero_burst_falls_back_to_request_budget() {
        let limiter = RateLimiter::with_burst(3, 60, 0);
        assert_eq!(limiter.limit(), 3);
        for _ in 0..3 {
            assert!(limiter.allow("token-a"));
        }
        assert!(!limiter.allow("token-a"));
    }

    #[test]
    fn periodic_sweep_reclaims_idle_buckets_before_capacity() {
        let limiter = RateLimiter::new(5, 60);
        {
            let mut buckets = limiter.buckets.lock().unwrap();
            for i in 0..60_000 {
                // Idle long enough to be fully refilled: evictable.
                buckets.insert(
                    format!("stale-{i}"),
                    Bucket {
                        tokens: 0.0,
                        last_refill: Instant::now() - Duration::from_secs(3600),
                    },
                );
            }
            *limiter.last_sweep.lock().unwrap() = Instant::now() - Duration::from_secs(61);
        }
        assert!(limiter.allow("fresh"));
        assert!(
            limiter.buckets.lock().unwrap().len() < 100,
            "throttled sweep should have reclaimed idle buckets"
        );
    }

    #[test]
    fn zero_disables_limiting() {
        let limiter = RateLimiter::new(0, 60);
        for _ in 0..100 {
            assert!(limiter.allow("token-a"));
        }
        let decision = limiter.check("token-a");
        assert!(decision.allowed);
        assert_eq!(decision.retry_after_secs, 0);
    }
}
