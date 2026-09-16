use std::{
    collections::HashMap,
    net::IpAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const MAX_ATTEMPTS: u32 = 5;
const WINDOW: Duration = Duration::from_secs(10 * 60);
const LOCK_DURATION: Duration = Duration::from_secs(15 * 60);
/// Upper bound on tracked IPs to prevent unbounded memory growth.
const MAX_TRACKED_IPS: usize = 10_000;

struct Attempts {
    count: u32,
    window_start: Instant,
    locked_until: Option<Instant>,
}

#[derive(Clone, Default)]
pub struct LoginAttemptTracker {
    map: Arc<Mutex<HashMap<IpAddr, Attempts>>>,
}

impl LoginAttemptTracker {
    pub fn is_locked(&self, ip: IpAddr) -> bool {
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        match map.get_mut(&ip) {
            Some(entry) => {
                // Expire stale window state.
                if entry.locked_until.is_none() && now.duration_since(entry.window_start) >= WINDOW
                {
                    map.remove(&ip);
                    return false;
                }
                match entry.locked_until {
                    Some(until) if now < until => true,
                    Some(_) => {
                        map.remove(&ip);
                        false
                    }
                    None => false,
                }
            }
            None => false,
        }
    }

    pub fn record_failure(&self, ip: IpAddr) {
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        self.evict_expired_locked(&mut map);
        if map.len() >= MAX_TRACKED_IPS && !map.contains_key(&ip) {
            // Under pressure: evict the oldest window entry instead of growing.
            if let Some(oldest) = map
                .iter()
                .min_by_key(|(_, a)| a.window_start)
                .map(|(ip, _)| *ip)
            {
                map.remove(&oldest);
            }
        }
        let now = Instant::now();
        let entry = map.entry(ip).or_insert(Attempts {
            count: 0,
            window_start: now,
            locked_until: None,
        });
        if now.duration_since(entry.window_start) >= WINDOW {
            entry.count = 0;
            entry.window_start = now;
            entry.locked_until = None;
        }
        entry.count += 1;
        if entry.count >= MAX_ATTEMPTS {
            entry.locked_until = Some(now + LOCK_DURATION);
        }
    }

    pub fn record_success(&self, ip: IpAddr) {
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        map.remove(&ip);
    }

    fn evict_expired_locked(&self, map: &mut HashMap<IpAddr, Attempts>) {
        let now = Instant::now();
        map.retain(|_, entry| match entry.locked_until {
            Some(until) => now < until,
            None => now.duration_since(entry.window_start) < WINDOW,
        });
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn ip(last: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(192, 0, 2, last))
    }

    #[test]
    fn locks_after_five_failures() {
        let tracker = LoginAttemptTracker::default();
        for _ in 0..4 {
            tracker.record_failure(ip(1));
            assert!(!tracker.is_locked(ip(1)));
        }
        tracker.record_failure(ip(1));
        assert!(tracker.is_locked(ip(1)));
        // Other IPs are unaffected.
        assert!(!tracker.is_locked(ip(2)));
    }

    #[test]
    fn success_clears_failures() {
        let tracker = LoginAttemptTracker::default();
        for _ in 0..4 {
            tracker.record_failure(ip(1));
        }
        tracker.record_success(ip(1));
        assert!(!tracker.is_locked(ip(1)));
        tracker.record_failure(ip(1));
        assert!(!tracker.is_locked(ip(1)));
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn lockout_expires() {
        let tracker = LoginAttemptTracker::default();
        for _ in 0..MAX_ATTEMPTS {
            tracker.record_failure(ip(1));
        }
        assert!(tracker.is_locked(ip(1)));
        // Simulate expiry by rewinding the timestamps.
        {
            let mut map = tracker.map.lock().unwrap();
            let entry = map.get_mut(&ip(1)).unwrap();
            entry.locked_until = Some(Instant::now() - Duration::from_secs(1));
        }
        assert!(!tracker.is_locked(ip(1)));
    }

    #[test]
    fn window_expiry_resets_count() {
        let tracker = LoginAttemptTracker::default();
        for _ in 0..4 {
            tracker.record_failure(ip(1));
        }
        {
            let mut map = tracker.map.lock().unwrap();
            let entry = map.get_mut(&ip(1)).unwrap();
            entry.window_start = Instant::now() - WINDOW - Duration::from_secs(1);
        }
        assert!(!tracker.is_locked(ip(1)));
        tracker.record_failure(ip(1));
        assert!(!tracker.is_locked(ip(1)));
    }

    #[test]
    fn tracker_capacity_is_bounded() {
        let tracker = LoginAttemptTracker::default();
        for i in 0..(MAX_TRACKED_IPS + 500) {
            let addr = IpAddr::V4(Ipv4Addr::new(10, (i >> 16) as u8, (i >> 8) as u8, i as u8));
            tracker.record_failure(addr);
        }
        assert!(tracker.len() <= MAX_TRACKED_IPS);
    }
}
