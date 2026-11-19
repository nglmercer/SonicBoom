//! Model identity is independent of compiled backends and runtime readiness.
//! Configuration, installed weights and provider status remain registry concerns.

pub const QWEN_GGUF_MODELS: &[&str] = &["qwen3-tts-1.7b-base"];
/// Qwen3-TTS Base (0.6B/1.7B safetensors) on candle: voice cloning with profiles.
pub const QWEN_CANDLE_MODELS: &[&str] = &["qwen3-tts-candle"];
/// Kyutai TTS 1.6B en/fr on candle: named voices (`kyutai/tts-voices` embeddings).
pub const KYUTAI_CANDLE_MODELS: &[&str] = &["kyutai-tts-1.6b-en-fr"];
pub const POCKET_TTS_MODELS: &[&str] = &[
    "pocket-tts-en",
    "pocket-tts-es",
    "pocket-tts-pt",
    "pocket-tts-de",
    "pocket-tts-it",
    "pocket-tts-fr-24l",
];
pub const SUPPORTED_OPENAI_MODELS: &[&str] = &[
    "tts-1",
    "tts-1-hd",
    "supertonic-3",
    QWEN_GGUF_MODELS[0],
    POCKET_TTS_MODELS[0],
    POCKET_TTS_MODELS[1],
    QWEN_CANDLE_MODELS[0],
    KYUTAI_CANDLE_MODELS[0],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelProvider {
    Supertonic,
    QwenGguf,
    PocketTts,
    QwenCandle,
    KyutaiCandle,
}

impl ModelProvider {
    pub fn id(self) -> &'static str {
        match self {
            Self::Supertonic => "supertonic",
            Self::QwenGguf => "qwen3-gguf",
            Self::PocketTts => "pocket-tts",
            Self::QwenCandle => "qwen3-candle",
            Self::KyutaiCandle => "kyutai-candle",
        }
    }

    pub fn compiled(self) -> bool {
        match self {
            Self::Supertonic => true,
            Self::QwenGguf => cfg!(feature = "qwen-gguf"),
            Self::PocketTts => cfg!(feature = "pocket-tts"),
            Self::QwenCandle => cfg!(feature = "candle-qwen"),
            Self::KyutaiCandle => cfg!(feature = "candle-kyutai"),
        }
    }
}

pub fn model_provider(model: &str) -> Option<ModelProvider> {
    if QWEN_GGUF_MODELS.contains(&model) {
        Some(ModelProvider::QwenGguf)
    } else if POCKET_TTS_MODELS.contains(&model) {
        Some(ModelProvider::PocketTts)
    } else if QWEN_CANDLE_MODELS.contains(&model) {
        Some(ModelProvider::QwenCandle)
    } else if KYUTAI_CANDLE_MODELS.contains(&model) {
        Some(ModelProvider::KyutaiCandle)
    } else if ["tts-1", "tts-1-hd", "supertonic-3"].contains(&model) {
        Some(ModelProvider::Supertonic)
    } else {
        None
    }
}
