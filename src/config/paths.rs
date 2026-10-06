//! Platform configuration and data directories.
//!
//! Desktop installs keep their configuration in the platform
//! application-data location rather than next to the executable:
//!
//! ```text
//! Windows: %APPDATA%\SonicBoom\
//! Linux:   $XDG_CONFIG_HOME/sonicboom  (or ~/.config/sonicboom)
//! macOS:   ~/Library/Application Support/SonicBoom
//! ```
//!
//! Portable/exe-adjacent configuration is still supported when a
//! `config.toml` exists next to the executable.

use std::path::PathBuf;

/// Application-specific directory name.
pub const APP_DIR_NAME: &str = "sonicboom";

/// The platform configuration directory (uncreated).
///
/// Linux honors `$XDG_CONFIG_HOME`; macOS uses
/// `~/Library/Application Support`; Windows uses `%APPDATA%`.
pub fn config_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("SonicBoom");
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return PathBuf::from(appdata).join("SonicBoom");
        }
        if let Some(home) = std::env::var_os("USERPROFILE") {
            return PathBuf::from(home)
                .join("AppData")
                .join("Roaming")
                .join("SonicBoom");
        }
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
            if !xdg.is_empty() {
                return PathBuf::from(xdg).join(APP_DIR_NAME);
            }
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(".config").join(APP_DIR_NAME);
        }
    }
    // Fallback: a `.sonicboom` directory under the current directory.
    PathBuf::from(".sonicboom")
}

/// The platform data directory (models, logs, audio). Same base as
/// the config directory for a single predictable layout.
pub fn data_dir() -> PathBuf {
    config_dir()
}

/// Resolve the active `config.toml` path.
///
/// Precedence:
/// 1. Explicit `SONICBOOM_CONFIG` environment variable.
/// 2. Exe-adjacent `config.toml` (portable installs).
/// 3. Platform configuration directory.
pub fn resolve_config_path() -> PathBuf {
    if let Some(explicit) = std::env::var_os("SONICBOOM_CONFIG") {
        if !explicit.is_empty() {
            return PathBuf::from(explicit);
        }
    }
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        let portable = exe_dir.join("config.toml");
        if portable.is_file() {
            return portable;
        }
    }
    config_dir().join("config.toml")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_dir_is_absolute() {
        let dir = config_dir();
        assert!(dir.is_absolute(), "config dir should be absolute: {dir:?}");
        assert!(
            dir.file_name().is_some_and(|name| {
                let name = name.to_string_lossy().to_lowercase();
                name == "sonicboom" || name == "sonicboom" || name.contains("sonicboom")
            }),
            "config dir should mention the app: {dir:?}"
        );
    }

    #[test]
    fn data_dir_matches_config_dir() {
        assert_eq!(data_dir(), config_dir());
    }
}
