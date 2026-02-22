use std::{
    collections::HashMap,
    net::IpAddr,
    sync::{Arc, Mutex},
};

const MAX_ATTEMPTS: u32 = 5;

#[derive(Default)]
struct Attempts {
    count: u32,
    locked: bool,
}

#[derive(Clone, Default)]
pub struct LoginAttemptTracker {
    map: Arc<Mutex<HashMap<IpAddr, Attempts>>>,
}

impl LoginAttemptTracker {
    pub fn is_locked(&self, ip: IpAddr) -> bool {
        let map = self.map.lock().unwrap();
        map.get(&ip).map(|a| a.locked).unwrap_or(false)
    }

    pub fn record_failure(&self, ip: IpAddr) {
        let mut map = self.map.lock().unwrap();
        let entry = map.entry(ip).or_default();
        entry.count += 1;
        if entry.count >= MAX_ATTEMPTS {
            entry.locked = true;
        }
    }

    pub fn record_success(&self, ip: IpAddr) {
        let mut map = self.map.lock().unwrap();
        map.remove(&ip);
    }
}
