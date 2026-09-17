use std::net::{IpAddr, SocketAddr};

use axum::http::HeaderMap;

use crate::config::AppConfig;

/// Resolve the client IP for login-attempt tracking.
///
/// Forwarded headers (`X-Forwarded-For`, `X-Real-IP`) are only honored when
/// `TRUST_PROXY` is enabled **and** the socket peer is a configured trusted
/// proxy. Otherwise the socket peer address is used. The resulting IP is only
/// ever used for rate/lockout accounting — never for authentication.
///
/// Proxy chains are parsed from the trusted (right) side: addresses are
/// walked right-to-left, configured trusted proxies are skipped, and the
/// first untrusted valid IP is the client. An attacker-supplied leftmost
/// entry can therefore never be selected while a real chain exists.
pub fn client_ip(headers: &HeaderMap, peer: SocketAddr, config: &AppConfig) -> IpAddr {
    if config.trust_proxy
        && is_trusted_proxy(peer.ip(), config)
        && let Some(forwarded) = forwarded_client_ip(headers, config)
    {
        return forwarded;
    }
    peer.ip()
}

fn is_trusted_proxy(peer: IpAddr, config: &AppConfig) -> bool {
    config
        .trusted_proxies
        .iter()
        .any(|entry| proxy_entry_contains(entry, peer))
}

fn proxy_entry_contains(entry: &str, peer: IpAddr) -> bool {
    if let Ok(addr) = entry.parse::<IpAddr>() {
        return addr == peer;
    }
    // Also allow CIDR prefixes in `a.b.c.d/n` form.
    parse_cidr(entry).is_some_and(|(net, bits)| cidr_contains(net, bits, peer))
}

/// Validate a `TRUSTED_PROXIES` entry: a plain IP address or a CIDR prefix
/// with in-range bits. Used by config validation to fail closed on typos.
pub fn is_valid_proxy_entry(entry: &str) -> bool {
    let entry = entry.trim();
    if entry.is_empty() {
        return false;
    }
    if entry.parse::<IpAddr>().is_ok() {
        return true;
    }
    match parse_cidr(entry) {
        Some((IpAddr::V4(_), bits)) => bits <= 32,
        Some((IpAddr::V6(_), bits)) => bits <= 128,
        None => false,
    }
}

/// Extract the client IP from forwarded headers, parsing from the trusted
/// (right) side.
///
/// - All `X-Forwarded-For` header values are collected (a request may carry
///   several) and split on commas.
/// - Entries are normalized (trimmed); quoted or otherwise malformed values
///   are ignored, never trusted.
/// - Walking right-to-left, configured trusted proxies are skipped; the
///   first untrusted valid IP is the client.
/// - If nothing usable remains, `None` is returned and the caller falls
///   back to the socket peer.
///
/// `X-Real-IP` is honored only when no `X-Forwarded-For` is present at all
/// and it contains exactly one valid IP — the only unambiguous case.
fn forwarded_client_ip(headers: &HeaderMap, config: &AppConfig) -> Option<IpAddr> {
    let mut chain: Vec<IpAddr> = Vec::new();
    let mut saw_xff = false;
    for value in headers.get_all("x-forwarded-for") {
        saw_xff = true;
        let Ok(text) = value.to_str() else {
            continue;
        };
        for entry in text.split(',') {
            if let Some(ip) = parse_forwarded_entry(entry) {
                chain.push(ip);
            }
        }
    }
    if saw_xff {
        // Walk from the trusted side: skip configured proxies, take the
        // first untrusted address. Spoofed leftmost entries are unreachable
        // while the real chain is intact.
        for ip in chain.iter().rev() {
            if !config.trusted_proxies.iter().any(|entry| {
                entry
                    .parse::<IpAddr>()
                    .map(|addr| addr == *ip)
                    .unwrap_or_else(|_| {
                        parse_cidr(entry).is_some_and(|(net, bits)| cidr_contains(net, bits, *ip))
                    })
            }) {
                return Some(*ip);
            }
        }
        return None;
    }
    // No XFF at all: a single unambiguous X-Real-IP may be used.
    let mut real_ips = headers
        .get_all("x-real-ip")
        .into_iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|text| text.split(','))
        .filter_map(parse_forwarded_entry);
    let first = real_ips.next()?;
    if real_ips.next().is_some() {
        // Multiple values are ambiguous; fail safe to the socket peer.
        return None;
    }
    Some(first)
}

/// Parse one forwarded entry strictly: surrounding whitespace is ignored,
/// but quoted strings and anything that is not exactly an IP address are
/// rejected (returning `None`) rather than trusted.
fn parse_forwarded_entry(entry: &str) -> Option<IpAddr> {
    let entry = entry.trim();
    if entry.is_empty() {
        return None;
    }
    // Quoted values are malformed for our purposes; never strip quotes to
    // manufacture trust in an unexpected string.
    if entry.starts_with('"') || entry.starts_with('\'') {
        return None;
    }
    entry.parse::<IpAddr>().ok()
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
            port: 17842,
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
            tts_max_body_bytes: 1024,
            openai_max_body_bytes: 1024,
            queue_max_body_bytes: 1024,
            admin_max_body_bytes: 1024,
            trust_proxy,
            trusted_proxies: trusted.iter().map(|s| s.to_string()).collect(),
            cookie_secure: false,
            admin_session_expiry_secs: 60,
            temp_audio_dir: "./temp_audio".to_string(),
            enable_hsts: false,
            max_playback_queue_items: 100,
            model_download_connect_timeout_secs: 10,
            model_download_timeout_secs: 1800,
        }
    }

    fn peer() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 7)), 1234)
    }

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
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
        assert_eq!(client_ip(&headers, peer(), &config), ip("198.51.100.9"));
    }

    #[test]
    fn trusted_proxy_cidr_used() {
        let config = config_with(true, &["203.0.113.0/24"]);
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "198.51.100.9".parse().unwrap());
        assert_eq!(client_ip(&headers, peer(), &config), ip("198.51.100.9"));
    }

    #[test]
    fn spoofed_leftmost_entry_cannot_be_selected() {
        // Attacker supplies `X-Forwarded-For: 1.2.3.4`; the proxy appends
        // the real client and itself. Right-to-left parsing must resolve
        // the real client, never the spoofed leftmost value.
        let config = config_with(true, &["203.0.113.7"]);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "1.2.3.4, 198.51.100.9, 203.0.113.7".parse().unwrap(),
        );
        assert_eq!(client_ip(&headers, peer(), &config), ip("198.51.100.9"));
    }

    #[test]
    fn multiple_trusted_proxies_skipped_right_to_left() {
        let config = config_with(true, &["203.0.113.7", "10.0.0.1", "10.0.0.2"]);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "1.2.3.4, 198.51.100.9, 10.0.0.2, 10.0.0.1".parse().unwrap(),
        );
        assert_eq!(client_ip(&headers, peer(), &config), ip("198.51.100.9"));
    }

    #[test]
    fn ipv4_cidr_chain_parsing() {
        let config = config_with(true, &["203.0.113.0/24", "10.0.0.0/8"]);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "192.0.2.55, 10.1.2.3, 203.0.113.7".parse().unwrap(),
        );
        assert_eq!(client_ip(&headers, peer(), &config), ip("192.0.2.55"));
    }

    #[test]
    fn ipv6_cidr_chain_parsing() {
        let peer6 = SocketAddr::new("2001:db8::1".parse().unwrap(), 443);
        let config = config_with(true, &["2001:db8::/32"]);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "2001:db8:ffff::99, 2001:db8::1".parse().unwrap(),
        );
        // Both are inside the trusted /32: nothing untrusted remains, so
        // fail safe to the socket peer.
        assert_eq!(client_ip(&headers, peer6, &config), peer6.ip());

        let config = config_with(true, &["2001:db8::1"]);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            " 2001:db8:1::5 , 2001:db8::1 ".parse().unwrap(),
        );
        assert_eq!(
            client_ip(&headers, peer6, &config),
            ip("2001:db8:1::5"),
            "whitespace around IPv6 entries must be tolerated"
        );
    }

    #[test]
    fn malformed_chain_fails_safe() {
        let config = config_with(true, &["203.0.113.7"]);
        for xff in [
            "not-an-ip",
            "\"198.51.100.9\"",
            "198.51.100.9/24",
            "999.999.999.999",
            "",
            "   ",
            "1.2.3.4.5",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert("x-forwarded-for", xff.parse().unwrap());
            assert_eq!(
                client_ip(&headers, peer(), &config),
                peer().ip(),
                "malformed XFF {xff:?} must fall back to socket peer"
            );
        }
        // Malformed entries mixed into a real chain are ignored, not trusted.
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "garbage, 198.51.100.9, 203.0.113.7".parse().unwrap(),
        );
        assert_eq!(client_ip(&headers, peer(), &config), ip("198.51.100.9"));
    }

    #[test]
    fn multiple_xff_headers_do_not_allow_spoofing() {
        let config = config_with(true, &["203.0.113.7"]);
        let mut headers = HeaderMap::new();
        headers.append("x-forwarded-for", "1.2.3.4".parse().unwrap());
        headers.append(
            "x-forwarded-for",
            "198.51.100.9, 203.0.113.7".parse().unwrap(),
        );
        assert_eq!(client_ip(&headers, peer(), &config), ip("198.51.100.9"));
    }

    #[test]
    fn all_trusted_chain_falls_back_to_peer() {
        let config = config_with(true, &["203.0.113.7", "198.51.100.9"]);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "198.51.100.9, 203.0.113.7".parse().unwrap(),
        );
        assert_eq!(client_ip(&headers, peer(), &config), peer().ip());
    }

    #[test]
    fn xff_takes_precedence_over_x_real_ip() {
        let config = config_with(true, &["203.0.113.7"]);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "198.51.100.9, 203.0.113.7".parse().unwrap(),
        );
        headers.insert("x-real-ip", "1.2.3.4".parse().unwrap());
        assert_eq!(client_ip(&headers, peer(), &config), ip("198.51.100.9"));
    }

    #[test]
    fn ambiguous_x_real_ip_falls_back_to_peer() {
        let config = config_with(true, &["203.0.113.7"]);
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "1.2.3.4, 5.6.7.8".parse().unwrap());
        assert_eq!(client_ip(&headers, peer(), &config), peer().ip());
    }

    #[test]
    fn proxy_entry_validation() {
        assert!(is_valid_proxy_entry("10.0.0.1"));
        assert!(is_valid_proxy_entry("::1"));
        assert!(is_valid_proxy_entry("10.0.0.0/24"));
        assert!(is_valid_proxy_entry("2001:db8::/32"));
        assert!(is_valid_proxy_entry("0.0.0.0/0"));
        assert!(!is_valid_proxy_entry(""));
        assert!(!is_valid_proxy_entry("   "));
        assert!(!is_valid_proxy_entry("not-an-ip"));
        assert!(!is_valid_proxy_entry("10.0.0.0/33"));
        assert!(!is_valid_proxy_entry("2001:db8::/129"));
        assert!(!is_valid_proxy_entry("10.0.0.0/-1"));
        assert!(!is_valid_proxy_entry("10.0.0.0/"));
    }
}
