//! Secret storage abstraction.
//!
//! Secrets (admin password, HuggingFace token) must never be
//! written to `config.toml`. They live in a dedicated secrets
//! file with restrictive permissions; the initial
//! implementation is a local file, with OS credential
//! managers (Windows Credential Manager, macOS Keychain,
//! Linux Secret Service) as future backends.

use std::collections::BTreeMap;
#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;

/// Failure to read/write a secret.
#[derive(Debug, thiserror::Error)]
pub enum SecretStoreError {
    #[error("secret store I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("secret store serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("secret store is corrupted: {0}")]
    Corrupted(String),
}

/// A key/value secret store.
pub trait SecretStore: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<String>, SecretStoreError>;
    fn set(&self, key: &str, value: &str) -> Result<(), SecretStoreError>;
    fn remove(&self, key: &str) -> Result<(), SecretStoreError>;
}

/// Well-known secret keys.
pub mod keys {
    /// Admin panel password.
    pub const ADMIN_PASSWORD: &str = "admin_password";
    /// HuggingFace access token for gated/private model downloads.
    pub const HF_TOKEN: &str = "hf_token";
}

/// File-backed secret store: a JSON map with `0600` permissions.
#[derive(Debug, Clone)]
pub struct FileSecretStore {
    path: PathBuf,
}

impl FileSecretStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Path of the secrets file (test introspection;
    /// never exposes secret values).
    #[cfg(test)]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn lock(&self) -> Result<std::fs::File, SecretStoreError> {
        if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(self.path.with_extension("json.lock"))?;
        file.lock()?;
        Ok(file)
    }

    fn read_map(&self) -> Result<BTreeMap<String, String>, SecretStoreError> {
        match std::fs::read_to_string(&self.path) {
            Ok(text) if text.trim().is_empty() => {
                Err(SecretStoreError::Corrupted("empty secrets file".into()))
            }
            Ok(text) => {
                serde_json::from_str(&text).map_err(|e| SecretStoreError::Corrupted(e.to_string()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
            Err(e) => Err(SecretStoreError::Io(e)),
        }
    }

    fn write_map(&self, map: &BTreeMap<String, String>) -> Result<(), SecretStoreError> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let temporary = self
            .path
            .with_extension(format!("json.{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| -> Result<(), SecretStoreError> {
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o600);
            }
            let mut file = opts.open(&temporary)?;
            use std::io::Write as _;
            file.write_all(serde_json::to_string_pretty(map)?.as_bytes())?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temporary, &self.path)?;
            #[cfg(unix)]
            if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::File::open(parent)?.sync_all()?;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temporary);
        }
        result?;
        Ok(())
    }
}

impl SecretStore for FileSecretStore {
    fn get(&self, key: &str) -> Result<Option<String>, SecretStoreError> {
        Ok(self.read_map()?.remove(key))
    }

    fn set(&self, key: &str, value: &str) -> Result<(), SecretStoreError> {
        let _lock = self.lock()?;
        let mut map = self.read_map()?;
        map.insert(key.to_string(), value.to_string());
        self.write_map(&map)
    }

    fn remove(&self, key: &str) -> Result<(), SecretStoreError> {
        let _lock = self.lock()?;
        let mut map = self.read_map()?;
        map.remove(key);
        self.write_map(&map)
    }
}

/// Load the admin password from the secret store, falling back to
/// the `SONICBOOM_ADMIN_PW` environment variable (headless/Docker
/// compatibility). The environment value is *not* persisted.
pub fn load_admin_password(store: &dyn SecretStore) -> Result<Option<String>, SecretStoreError> {
    if let Some(pw) = store.get(keys::ADMIN_PASSWORD)? {
        if !pw.is_empty() {
            return Ok(Some(pw));
        }
    }
    Ok(std::env::var("SONICBOOM_ADMIN_PW")
        .ok()
        .filter(|v| !v.is_empty()))
}

/// Store (or replace) the admin password in the
/// secret store. The password is never written
/// to `config.toml` (spec §7).
pub fn store_admin_password(
    store: &dyn SecretStore,
    password: &str,
) -> Result<(), SecretStoreError> {
    store.set(keys::ADMIN_PASSWORD, password)
}

/// Load the HuggingFace token from the secret store, falling back
/// to the `HF_TOKEN` environment variable.
pub fn load_hf_token(store: &dyn SecretStore) -> Result<Option<String>, SecretStoreError> {
    if let Some(token) = store.get(keys::HF_TOKEN)? {
        if !token.is_empty() {
            return Ok(Some(token));
        }
    }
    Ok(std::env::var("HF_TOKEN").ok().filter(|v| !v.is_empty()))
}

/// In-memory secret store for tests and embedding.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct MemorySecretStore {
    inner: std::sync::Mutex<BTreeMap<String, String>>,
}

#[cfg(test)]
impl SecretStore for MemorySecretStore {
    fn get(&self, key: &str) -> Result<Option<String>, SecretStoreError> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| SecretStoreError::Corrupted("secret store lock poisoned".to_string()))?;
        Ok(inner.get(key).cloned())
    }

    fn set(&self, key: &str, value: &str) -> Result<(), SecretStoreError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| SecretStoreError::Corrupted("secret store lock poisoned".to_string()))?;
        inner.insert(key.to_string(), value.to_string());
        Ok(())
    }

    fn remove(&self, key: &str) -> Result<(), SecretStoreError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| SecretStoreError::Corrupted("secret store lock poisoned".to_string()))?;
        inner.remove(key);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(tag: &str) -> FileSecretStore {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-secrets-test-{}-{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::create_dir_all(&dir);
        FileSecretStore::new(dir.join("secrets.json"))
    }

    #[test]
    fn set_get_round_trip() {
        let store = temp_store("roundtrip");
        let _ = std::fs::remove_file(store.path());
        assert_eq!(store.get(keys::ADMIN_PASSWORD).unwrap(), None);
        store.set(keys::ADMIN_PASSWORD, "s3cret-value").unwrap();
        assert_eq!(
            store.get(keys::ADMIN_PASSWORD).unwrap(),
            Some("s3cret-value".to_string())
        );
        store.remove(keys::ADMIN_PASSWORD).unwrap();
        assert_eq!(store.get(keys::ADMIN_PASSWORD).unwrap(), None);
        let _ = std::fs::remove_dir_all(store.path().parent().unwrap());
    }

    #[test]
    fn secrets_file_has_restrictive_permissions() {
        let store = temp_store("perms");
        let _ = std::fs::remove_file(store.path());
        store.set(keys::ADMIN_PASSWORD, "x").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(store.path())
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "secrets file must be 0600");
        }
        let _ = std::fs::remove_dir_all(store.path().parent().unwrap());
    }

    #[test]
    fn corrupted_store_is_reported() {
        let store = temp_store("corrupt");
        let _ = std::fs::remove_file(store.path());
        std::fs::write(store.path(), "not json").unwrap();
        assert!(store.get(keys::ADMIN_PASSWORD).is_err());
        let _ = std::fs::remove_dir_all(store.path().parent().unwrap());
    }
}
