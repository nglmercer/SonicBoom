//! Structured SonicBoom configuration.
//!
//! One central [`ConfigManager`] owns every configuration
//! change: GUI, REST API, CLI, MCP, the first-run wizard,
//! and the filesystem watcher all flow through it, so
//! validation, revisioning, atomic persistence, and
//! runtime reconfiguration happen in exactly one place.
//!
//! Layout:
//!
//! - [`model`] — sectioned `AppConfig` + defaults
//! - [`validation`] — fail-closed validation (incl.
//!   auth/bind security combinations)
//! - [`loader`] — `config.toml` + environment overrides
//!   with per-value source tracking
//! - [`writer`] — atomic writes (tmp -> fsync -> rename)
//! - [`manager`] — the central `ConfigManager`
//! - [`watcher`] — debounced hot-reload watcher
//! - [`diff`] — semantic config diffs and events
//! - [`schema`] — agent/GUI-facing setting metadata
//! - [`effective`] — effective values + sources
//! - [`migration`] — legacy `.env` migration
//! - [`secrets`] — secret storage (never in TOML)
//! - [`paths`] — platform configuration directories

pub mod diff;
pub mod effective;
pub mod loader;
pub mod manager;
pub mod migration;
pub mod model;
pub mod paths;
pub mod schema;
pub mod secrets;
pub mod validation;
pub mod watcher;
pub mod writer;

pub use manager::{ConfigManager, ConfigUpdate, UpdateError};
#[cfg(feature = "playback")]
pub use model::MAX_AUDIO_OUTPUT_DEVICE_LEN;
pub use model::{AppConfig, AuthMode};

pub use schema::schema;
pub use secrets::SecretStore;
