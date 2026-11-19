//! Kyutai DSM-TTS (`kyutai/tts-1.6b-en_fr`) on candle.
//!
//! Ported from the Python reference in `kyutai-labs/moshi`
//! (`moshi/models/lm.py`, `moshi/models/tts.py`, `moshi/modules/transformer.py`).
//! Kyutai's own Rust server runs this checkpoint through Python, and the
//! `moshi-core` Rust LM targets an older layout, so the LM here is new code.
pub mod config;
pub mod lm;
pub mod machine;
pub mod spm;
pub mod tts;
