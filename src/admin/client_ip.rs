use std::net::{IpAddr, SocketAddr};

use axum::http::HeaderMap;

use crate::config::AppConfig;

/// Resolve the client IP for login-attempt tracking.
///
/// Forwarded headers (`X-Forwarded-For`, `X-Real-IP`) are only honored when
/// `TRUST_PROXY` is enabled **and** the socket peer is a configured trusted
/// proxy. Otherwise the socket peer address is used. The resulting IP is only
/// ever used for rate/lockout accounting — never for authentication.
pub fn client_ip(headers: &HeaderMap, peer: SocketAddr, config: &AppConfig) -> IpAddr {
    if config.trust_proxy
        && is_trusted_proxy(peer.ip(), config)
        && let Some(forwarded) = first_forwarded_ip(headers)
    {
        return forwarded;
    }
    peer.ip()
}

fn is_trusted_proxy(peer: IpAddr, config: &AppConfig) -> bool {
    config.trusted_proxies.iter().any(|entry| {
        entry
            .parse::<IpAddr>()
            .map(|addr| addr == peer)
            .unwrap_or_else(|_| {
                // Also allow CIDR prefixes in `a.b.c.d/n` form.
                parse_cidr(entry).is_some_and(|(net, bits)| cidr_contains(net, bits, peer))
            })
    })
}

fn first_forwarded_ip(headers: &HeaderMap) -> Option<IpAddr> {
    // X-Forwarded-For: client, proxy1, proxy2 — the leftmost entry is the
    // original client as claimed by the trusted proxy chain.
    if let Some(value) = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        && let Ok(ip) = value.trim().trim_matches('"').parse::<IpAddr>()
    {
        return Some(ip);
    }
    if let Some(value) = headers.get("x-real-ip").and_then(|v| v.to_str().ok())
        && let Ok(ip) = value.trim().parse::<IpAddr>()
    {
        return Some(ip);
    }
    None
}

fn parse_cidr(entry: &str) -> Option<(IpAddr, u8)> {
    let (net, bits) = entry.split_once('/')?;
    let net: IpAddr = net.trim().parse().ok()?;
    let bits: u8 = bits.trim().parse().ok()?;
    Some((net, bits))
}

fn cidr_contains(net: IpAddr, bits: u8, peer: IpAddr) -> bool {
    match (net, peer) {
        (IpAddr::V4(net), IpAddr::V4(peer)) => {
            if bits > 32 {
                return false;
            }
            let mask = if bits == 0 {
                0
            } else {
                u32::MAX << (32 - bits)
            };
            (u32::from(net) & mask) == (u32::from(peer) & mask)
        }
        (IpAddr::V6(net), IpAddr::V6(peer)) => {
            if bits > 128 {
                return false;
            }
            let mask = if bits == 0 {
                0
            } else {
                u128::MAX << (128 - bits)
            };
            (u128::from(net) & mask) == (u128::from(peer) & mask)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn config_with(trust_proxy: bool, trusted: &[&str]) -> AppConfig {
        AppConfig {
            admin_id: "admin".to_string(),
            admin_pw: "long-enough-password".to_string(),
            enable_sample_token: false,
            token_store_path: String::new(),
            model_cache_dir: String::new(),
            model_revision: String::new(),
            model_hashes_path: None,
            hf_token: None,
            inference_steps: 5,
            port: 3000,
            log_dir: String::new(),
            log_level: "info".to_string(),
            log_to_file: false,
            log_to_stdout: true,
            auth_required: true,
            allowed_audio_dir: None,
            max_text_length: 100,
            request_timeout_secs: 1,
            max_concurrent_inference: 1,
            max_pending_inference: 1,
            max_chunk_chars: 10,
            tts_rate_limit_requests: 1,
            tts_rate_limit_window_secs: 1,
            tts_max_body_bytes: 1,
            openai_max_body_bytes: 1,
            queue_max_body_bytes: 1,
            admin_max_body_bytes: 1,
            trust_proxy,
            trusted_proxies: trusted.iter().map(|s| s.to_string()).collect(),
            cookie_secure: false,
            admin_session_expiry_secs: 60,
            temp_audio_dir: "./temp_audio".to_string(),
        }
    }

    fn peer() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 7)), 1234)
    }

    #[test]
    fn forwarded_headers_ignored_by_default() {
        let config = config_with(false, &[]);
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "198.51.100.9".parse().unwrap());
        assert_eq!(client_ip(&headers, peer(), &config), peer().ip());
    }

    #[test]
    fn forwarded_headers_ignored_from_untrusted_peer() {
        let config = config_with(true, &["10.0.0.1"]);
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "198.51.100.9".parse().unwrap());
        assert_eq!(client_ip(&headers, peer(), &config), peer().ip());
    }

    #[test]
    fn trusted_proxy_forwarded_ip_used() {
        let config = config_with(true, &["203.0.113.7"]);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "198.51.100.9, 203.0.113.7".parse().unwrap(),
        );
        assert_eq!(
            client_ip(&headers, peer(), &config),
            "198.51.100.9".parse::<IpAddr>().unwrap()
        );
    }

    #[test]
    fn trusted_proxy_cidr_used() {
        let config = config_with(true, &["203.0.113.0/24"]);
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "198.51.100.9".parse().unwrap());
        assert_eq!(
            client_ip(&headers, peer(), &config),
            "198.51.100.9".parse::<IpAddr>().unwrap()
        );
    }
}
