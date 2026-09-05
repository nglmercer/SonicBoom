//! Reusable SonicBoom TTS engine components.
//!
//! The HTTP server, authentication, admin panel, tray, and playback queue
//! remain in the standalone binary. This library exposes only model download,
//! inference, and audio encoding for other runtimes such as the TikTools
//! process plugin.

pub mod engine;
pub mod tts;
