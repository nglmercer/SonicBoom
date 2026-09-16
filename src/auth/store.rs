use crate::auth::token::{Token, generate_token_value, hash_token_value};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Persistent bearer-token store.
///
/// Keys are SHA-256 hex digests of the raw tokens; raw values never touch
/// disk. See [`Token`] for the threat model.
pub struct TokenStore {
    tokens: Arc<RwLock<HashMap<String, Token>>>,
    path: String,
}

impl TokenStore {
    /// Load the store from `path`.
    ///
    /// A missing file is initialized as an empty store (and created on disk
    /// with restrictive permissions). Any other failure — unreadable file,
    /// malformed JSON, invalid records — is a hard error: we never silently
    /// fall back to an empty credential store.
    pub async fn load(path: &str) -> Result<Self> {
        match tokio::fs::read_to_string(path).await {
            Ok(data) => {
                let token_list: Vec<Token> = serde_json::from_str(&data)
                    .with_context(|| format!("failed to parse token store '{path}'"))?;
                let mut tokens = HashMap::with_capacity(token_list.len());
                for token in token_list {
                    validate_stored_token(&token)
                        .with_context(|| format!("invalid token record '{}'", token.id))?;
                    tokens.insert(token.token_hash.clone(), token);
                }
                Ok(Self {
                    tokens: Arc::new(RwLock::new(tokens)),
                    path: path.to_string(),
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let store = Self {
                    tokens: Arc::new(RwLock::new(HashMap::new())),
                    path: path.to_string(),
                };
                store.save_new_file().await?;
                Ok(store)
            }
            Err(e) => Err(anyhow!("cannot read token store '{path}': {e}")),
        }
    }

    /// Create an empty token store (used as fallback when loading fails)
    pub fn empty() -> Self {
        Self {
            tokens: Arc::new(RwLock::new(HashMap::new())),
            path: String::new(),
        }
    }

    /// Validate a raw bearer token against the stored hashes using
    /// constant-time comparison.
    pub async fn validate(&self, value: &str) -> bool {
        let candidate_hash = hash_token_value(value);
        let candidate_bytes = candidate_hash.as_bytes();
        let tokens = self.tokens.read().await;
        // Compare against every stored hash in constant time so the outcome
        // does not leak which hash (if any) matched via timing.
        let mut matched_valid = false;
        for token in tokens.values() {
            if constant_time_eq::constant_time_eq(token.token_hash.as_bytes(), candidate_bytes)
                && token.is_valid()
            {
                matched_valid = true;
            }
        }
        matched_valid
    }

    pub async fn list(&self) -> Vec<Token> {
        self.tokens.read().await.values().cloned().collect()
    }

    /// Generate a fresh token, persist only its hash, and return the raw
    /// value exactly once. The caller must display it to the operator now:
    /// it can never be recovered afterwards.
    pub async fn create(&self, expires_at: Option<DateTime<Utc>>) -> Result<(Token, String)> {
        let raw = generate_token_value();
        let token = Token::new(hash_token_value(&raw), expires_at);
        let mut tokens = self.tokens.write().await;
        tokens.insert(token.token_hash.clone(), token.clone());
        self.save_locked(&tokens).await?;
        Ok((token, raw))
    }

    pub async fn revoke(&self, id: &str) -> Result<bool> {
        let mut tokens = self.tokens.write().await;
        // Find token by id and revoke it
        if let Some(token) = tokens.values_mut().find(|t| t.id == id) {
            token.revoked = true;
            self.save_locked(&tokens).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Write a brand-new store file for the missing-file case.
    async fn save_new_file(&self) -> Result<()> {
        if self.path.is_empty() {
            return Ok(());
        }
        self.save_locked(&HashMap::new()).await
    }

    /// Atomically persist the store: write temp file, fsync, rename.
    /// The file is created with `0600` permissions on Unix.
    async fn save_locked(&self, tokens: &HashMap<String, Token>) -> Result<()> {
        if self.path.is_empty() {
            return Ok(()); // No path set (empty store), skip saving
        }
        let token_list: Vec<&Token> = tokens.values().collect();
        let data = serde_json::to_string_pretty(&token_list)?;
        let tmp_path = format!("{}.tmp", self.path);
        write_restricted_file(&tmp_path, data.as_bytes())
            .await
            .with_context(|| format!("failed to write token store '{}'", self.path))?;
        tokio::fs::rename(&tmp_path, &self.path)
            .await
            .with_context(|| format!("failed to replace token store '{}'", self.path))?;
        Ok(())
    }
}

fn validate_stored_token(token: &Token) -> Result<()> {
    if token.id.is_empty() || token.id.len() > 128 {
        return Err(anyhow!("token id has invalid length"));
    }
    if token.token_hash.len() != 64 || !token.token_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(anyhow!("token_hash must be 64 lowercase hex chars"));
    }
    Ok(())
}

#[cfg(unix)]
async fn write_restricted_file(path: &str, data: &[u8]) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true).mode(0o600);
    let mut file = options.open(path).await?;
    use tokio::io::AsyncWriteExt;
    file.write_all(data).await?;
    file.flush().await?;
    file.sync_all().await?;
    // Harden permissions even if the file already existed with wider mode.
    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).await?;
    Ok(())
}

#[cfg(not(unix))]
async fn write_restricted_file(path: &str, data: &[u8]) -> Result<()> {
    tokio::fs::write(path, data).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_store() -> (TokenStore, tempfile_path::TempPath) {
        let path = tempfile_path::TempPath::new();
        let store = TokenStore::load(path.as_str()).await.unwrap();
        (store, path)
    }

    /// Minimal temp-file helper without adding a dev-dependency.
    mod tempfile_path {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        pub struct TempPath {
            path: std::path::PathBuf,
        }
        impl TempPath {
            pub fn new() -> Self {
                let id = COUNTER.fetch_add(1, Ordering::SeqCst);
                let path = std::env::temp_dir().join(format!(
                    "sonicboom-tokenstore-test-{}-{id}",
                    std::process::id()
                ));
                let _ = std::fs::remove_file(&path);
                Self { path }
            }
            pub fn as_str(&self) -> &str {
                self.path.to_str().unwrap()
            }
        }
        impl Drop for TempPath {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.path);
                let tmp = format!("{}.tmp", self.path.display());
                let _ = std::fs::remove_file(tmp);
            }
        }
    }

    #[tokio::test]
    async fn raw_token_is_not_stored_on_disk() {
        let (store, path) = temp_store().await;
        let (_token, raw) = store.create(None).await.unwrap();
        assert!(store.validate(&raw).await);
        let data = tokio::fs::read_to_string(path.as_str()).await.unwrap();
        assert!(
            !data.contains(&raw),
            "raw bearer token must never be persisted"
        );
        assert!(data.contains(&hash_token_value(&raw)));
    }

    #[tokio::test]
    async fn wrong_token_does_not_authenticate() {
        let (store, _path) = temp_store().await;
        let (_token, _raw) = store.create(None).await.unwrap();
        assert!(!store.validate("deadbeef").await);
        assert!(!store.validate("").await);
    }

    #[tokio::test]
    async fn revoked_and_expired_tokens_do_not_authenticate() {
        let (store, _path) = temp_store().await;
        let (token, raw) = store.create(None).await.unwrap();
        assert!(store.validate(&raw).await);
        assert!(store.revoke(&token.id).await.unwrap());
        assert!(!store.validate(&raw).await);

        let past = Utc::now() - chrono::Duration::seconds(60);
        let (_token2, raw2) = store.create(Some(past)).await.unwrap();
        assert!(!store.validate(&raw2).await);
    }

    #[tokio::test]
    async fn malformed_store_fails_instead_of_emptying() {
        let path = tempfile_path::TempPath::new();
        tokio::fs::write(path.as_str(), b"{ not valid json")
            .await
            .unwrap();
        assert!(TokenStore::load(path.as_str()).await.is_err());

        // Legacy plaintext shape must also fail closed.
        tokio::fs::write(
            path.as_str(),
            br#"[{"id":"x","value":"secret","created_at":"2026-01-01T00:00:00Z","expires_at":null,"revoked":false}]"#,
        )
        .await
        .unwrap();
        assert!(TokenStore::load(path.as_str()).await.is_err());
    }

    #[tokio::test]
    async fn missing_store_initializes_empty() {
        let path = tempfile_path::TempPath::new();
        let store = TokenStore::load(path.as_str()).await.unwrap();
        assert!(store.list().await.is_empty());
        assert!(!store.validate("anything").await);
        // File was created safely.
        let data = tokio::fs::read_to_string(path.as_str()).await.unwrap();
        assert_eq!(data.trim(), "[]");
    }
}
