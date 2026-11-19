//! Native TTS provider framework.
//!
//! Object-safe provider contract shared by Supertonic 3 (default),
//! Qwen3-TTS 1.7B GGUF/ONNX (opt-in) and Pocket TTS (ONNX, opt-in). All inference runs in the SonicBoom
//! process: non-`Send` model internals are encapsulated in dedicated
//! in-process worker threads with bounded channels, never subprocesses.

pub mod assets;
#[cfg(any(feature = "candle-qwen", feature = "candle-kyutai"))]
pub mod candle_models;
pub mod catalog;
pub mod error;
pub mod pocket_tts;
pub mod pool;
#[cfg(feature = "qwen-gguf")]
pub mod qwen_gguf;
pub mod registry;
pub mod supertonic;
pub mod types;
pub mod worker;

pub use error::{ProviderError, ProviderErrorKind};
pub use pool::ProviderPool;
pub use registry::TtsProviderRegistry;
pub use types::{
    CloneMode, DevicePreference, EffectiveDevice, PcmAudio, ProviderCapabilities, ProviderRequest,
    ProviderStatus, ProviderStatusState,
};

/// Object-safe voice-cloning provider contract.
///
/// Implementations must be `Send + Sync`. Model internals that are not
/// `Sync` (e.g. Pocket TTS mutable KV-cache states) are serialized
/// inside the provider behind a worker/mutex so concurrent callers can
/// never observe or corrupt each other's reference data.
pub trait TtsProvider: Send + Sync {
    /// Stable provider identifier, e.g. `supertonic`, `qwen3-gguf`, `pocket-tts`.
    fn id(&self) -> &'static str;
    /// Static capability metadata (models, languages, modes, devices).
    fn capabilities(&self) -> ProviderCapabilities;
    /// Current lifecycle status (enabled, readiness, effective device).
    fn status(&self) -> ProviderStatus;
    /// Load model assets for `model_id` (validates variant + files first).
    fn load(&self, model_id: &str) -> Result<(), ProviderError>;
    /// Unload model assets, freeing memory but keeping configuration.
    fn unload(&self);
    /// Logically invalidate conditioning after an authorized voice mutation.
    fn invalidate_voice(&self, _voice: &str) {}
    /// Blocking synthesis returning normalized PCM. Called with an
    /// admission-gate permit held by the blocking task itself.
    fn synthesize(&self, request: &ProviderRequest) -> Result<PcmAudio, ProviderError>;
    /// Cache a voice-conditioning reference, if the provider supports it.
    /// Default: providers without reference support reject explicitly.
    fn prepare_reference(
        &self,
        _model_id: &str,
        _reference_pcm: &[f32],
        _reference_rate: u32,
        _transcript: Option<&str>,
        _mode: CloneMode,
    ) -> Result<String, ProviderError> {
        Err(ProviderError::unsupported_model(
            self.id(),
            "<reference>",
            "this provider does not support reference conditioning",
        ))
    }
}
