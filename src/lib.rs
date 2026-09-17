//! Reusable SonicBoom TTS engine components.
//!
//! The HTTP server, authentication, admin panel, tray, and playback queue
//! remain in the standalone binary. This library exposes only model download,
//! inference, and audio encoding for other runtimes such as the TikTools
//! process plugin.

pub mod engine;
pub mod tts;

// The GPU execution-provider features are mutually exclusive per platform
// (see the note on `coreml`/`cuda`/`rocm` in Cargo.toml): no prebuilt ONNX
// Runtime satisfies more than one, so combining them only produces a
// cryptic linker error. Fail here instead, with the offending combination.
#[cfg(all(feature = "coreml", feature = "cuda"))]
compile_error!("features `coreml` and `cuda` are mutually exclusive; enable at most one");
#[cfg(all(feature = "coreml", feature = "rocm"))]
compile_error!("features `coreml` and `rocm` are mutually exclusive; enable at most one");
#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("features `cuda` and `rocm` are mutually exclusive; enable at most one");
