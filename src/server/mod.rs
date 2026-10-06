//! Independently restartable HTTP listener.
//!
//! `server.bind` / `server.port` changes restart only
//! this listener — the model, audio, tray, and the
//! application process stay alive.

pub mod manager;

pub use manager::ApiServerManager;
