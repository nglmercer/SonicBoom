//! Atomic configuration writes.
//!
//! Every internal write goes through a temporary file that is
//! flushed, fsynced, validated, and atomically renamed over the
//! live file — a crash mid-write can never replace a valid
//! `config.toml` with a partial one.

use std::path::Path;

use super::model::AppConfig;

/// Failure to atomically persist a configuration.
#[derive(Debug, thiserror::Error)]
pub enum AtomicWriteError {
    #[error("I/O error while writing config: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("candidate configuration failed validation: {0}")]
    Validation(String),
}

/// Serialize a configuration to TOML text.
pub fn to_toml_string(config: &AppConfig) -> Result<String, toml::ser::Error> {
    toml::to_string_pretty(config)
}

/// Atomically write `config` to `path`.
///
/// Flow: `path.tmp` -> write -> flush -> fsync -> validate ->
/// atomic rename -> `path`. The live file is never written
/// directly, so a failed write leaves the previous valid
/// configuration in place.
pub async fn atomic_write_config(path: &Path, config: &AppConfig) -> Result<(), AtomicWriteError> {
    // Validate before touching the filesystem: an invalid
    // candidate must never reach disk.
    super::validation::validate(config).map_err(AtomicWriteError::Validation)?;

    let text = to_toml_string(config)?;

    let tmp = path.with_extension(format!("toml.{}.tmp", uuid::Uuid::new_v4()));
    if let Err(error) = write_tmp(&tmp, text.as_bytes()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error.into());
    }

    // Atomic rename (POSIX rename(2); Windows uses
    // replace-file semantics via std::fs::rename).
    match std::fs::rename(&tmp, path) {
        Ok(()) => {}
        Err(e) => {
            // Best-effort cleanup of the temporary file.
            let _ = std::fs::remove_file(&tmp);
            return Err(AtomicWriteError::Io(e));
        }
    }
    // Persist the directory entry too so the rename itself is
    // durable before we report success.
    if let Some(parent) = path.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

fn write_tmp(tmp: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    use std::io::Write as _;
    if let Some(parent) = tmp.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o644);
    }
    let mut file = opts.open(tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Blocking variant of [`atomic_write_config`] for startup
/// and migration paths that run outside the async runtime.
pub fn atomic_write_config_blocking(
    path: &Path,
    config: &AppConfig,
) -> Result<(), AtomicWriteError> {
    super::validation::validate(config).map_err(AtomicWriteError::Validation)?;

    let text = to_toml_string(config)?;

    let tmp = path.with_extension(format!("toml.{}.tmp", uuid::Uuid::new_v4()));
    if let Err(error) = write_tmp(&tmp, text.as_bytes()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error.into());
    }

    match std::fs::rename(&tmp, path) {
        Ok(()) => {}
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(AtomicWriteError::Io(e));
        }
    }
    if let Some(parent) = path.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-atomic-test-{}-{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[tokio::test]
    async fn successful_write_persists_and_leaves_no_tmp() {
        let dir = temp_dir("success");
        let path = dir.join("config.toml");
        let _ = std::fs::remove_file(&path);
        let config = AppConfig::default();
        atomic_write_config(&path, &config).await.unwrap();
        assert!(path.is_file());
        assert!(
            !dir.join("config.toml.tmp").exists(),
            "tmp file must be cleaned up"
        );
        // Round-trip: the written file parses back.
        let text = std::fs::read_to_string(&path).unwrap();
        let parsed = super::super::loader::parse_toml(&text).unwrap();
        assert_eq!(parsed.server.port, config.server.port);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn invalid_config_is_never_written() {
        let dir = temp_dir("invalid");
        let path = dir.join("config.toml");
        // Pre-existing valid file.
        std::fs::write(&path, "# existing\n").unwrap();
        let mut config = AppConfig::default();
        config.server.port = 0; // invalid
        let err = atomic_write_config(&path, &config).await.unwrap_err();
        assert!(matches!(err, AtomicWriteError::Validation(_)));
        // The previous file is untouched and no tmp remains.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "# existing\n");
        assert!(!dir.join("config.toml.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn failed_rename_cleans_temporary_file_and_preserves_destination() {
        let dir = temp_dir("rename-failure");
        let destination = dir.join("config.toml");
        std::fs::create_dir_all(&destination).unwrap();
        std::fs::write(destination.join("existing"), "preserve me").unwrap();
        let result = atomic_write_config(&destination, &AppConfig::default()).await;
        assert!(matches!(result, Err(AtomicWriteError::Io(_))));
        assert_eq!(
            std::fs::read_to_string(destination.join("existing")).unwrap(),
            "preserve me"
        );
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            1,
            "temporary file must be removed"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn toml_round_trip_preserves_sections() {
        let mut config = AppConfig::default();
        config.server.port = 19000;
        config.audio.output_device = "CABLE Input".to_string();
        config.model.inference_steps = 10;
        config.security.trust_proxy = true;
        config.security.trusted_proxies = vec!["10.0.0.0/8".to_string()];
        let text = to_toml_string(&config).unwrap();
        let parsed = super::super::loader::parse_toml(&text).unwrap();
        assert_eq!(parsed.server.port, 19000);
        assert_eq!(parsed.audio.output_device, "CABLE Input");
        assert_eq!(parsed.model.inference_steps, 10);
        assert!(parsed.security.trust_proxy);
        assert_eq!(parsed.security.trusted_proxies.len(), 1);
    }
}
