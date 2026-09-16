use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A bearer API token record.
///
/// Only the SHA-256 hash of the token is ever persisted. The raw token is
/// high-entropy (256 bits from a CSPRNG), so a fast hash is appropriate; the
/// hash exists to make a stolen `tokens.json` useless for authentication.
/// The raw value is shown exactly once at creation time and is never stored,
/// logged, or returned by list endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub id: String,
    pub token_hash: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked: bool,
}

impl Token {
    pub fn new(token_hash: String, expires_at: Option<DateTime<Utc>>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            token_hash,
            created_at: Utc::now(),
            expires_at,
            revoked: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        if self.revoked {
            return false;
        }
        if let Some(exp) = self.expires_at {
            return Utc::now() < exp;
        }
        true
    }

    /// Short non-sensitive fingerprint for admin display, e.g. `abcd1234…`.
    /// Derived from the stored hash; it cannot be used to authenticate.
    pub fn fingerprint(&self) -> String {
        const PREFIX: usize = 8;
        if self.token_hash.len() >= PREFIX {
            format!("{}…", &self.token_hash[..PREFIX])
        } else {
            "invalid…".to_string()
        }
    }
}

/// Hash a raw bearer token with SHA-256 and return lowercase hex.
pub fn hash_token_value(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    hex::encode(hasher.finalize())
}

/// Generate a new 256-bit random bearer token, hex-encoded (64 chars).
pub fn generate_token_value() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic_hex() {
        let a = hash_token_value("hello");
        let b = hash_token_value("hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert_eq!(
            a,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn different_tokens_have_different_hashes() {
        assert_ne!(hash_token_value("a"), hash_token_value("b"));
    }

    #[test]
    fn generated_tokens_are_unique_and_hex() {
        let a = generate_token_value();
        let b = generate_token_value();
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn fingerprint_does_not_contain_raw_token() {
        let raw = generate_token_value();
        let token = Token::new(hash_token_value(&raw), None);
        let fp = token.fingerprint();
        assert!(fp.len() < raw.len());
        assert!(!fp.contains(&raw));
        assert!(fp.ends_with('…'));
    }

    #[test]
    fn expired_and_revoked_tokens_are_invalid() {
        let mut token = Token::new(hash_token_value("x"), None);
        assert!(token.is_valid());
        token.expires_at = Some(Utc::now() - chrono::Duration::seconds(1));
        assert!(!token.is_valid());
        token.expires_at = None;
        token.revoked = true;
        assert!(!token.is_valid());
    }
}
