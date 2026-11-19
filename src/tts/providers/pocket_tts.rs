//! Pocket TTS ONNX voice-cloning provider.
//!
//! Upstream runtime under review: `https://github.com/kyutai-labs/pocket-tts`
//! (MIT) with weights at `https://huggingface.co/kyutai/pocket-tts`
//! (CC-BY-4.0 plus a gated consent / prohibited-use clause). The native
//! ONNX bundles below are the third-party export
//! `https://huggingface.co/KevinAHM/pocket-tts-onnx` at pinned revision
//! `58a6d00cf13d239b6748cb0769f35c580a8f606c` (CC-BY-4.0, derived from the
//! gated upstream; redistribution needs owner/legal sign-off).
//!
//! Five ONNX graphs per language bundle run in-process on the shared
//! `ort =2.0.0-rc.13` CPU runtime (the same build Supertonic uses):
//! `text_conditioner`, `mimi_encoder` (reference -> voice embeddings),
//! `flow_lm_main` (autoregressive latent loop with KV-cache states),
//! `flow_lm_flow` (flow-matching head) and `mimi_decoder` (latents ->
//! 24 kHz PCM). Tokenization is a dependency-free pure-Rust unigram
//! encoder over the bundle's own `tokenizer.model` (identity normalizer,
//! byte fallback, verified numerically against Python sentencepiece).
//! No Python, no subprocess, no sidecar.
//!
//! Conditioning semantics are Pocket's own audio-prompt path: the
//! reference WAV is encoded to voice embeddings (with the bundle's
//! `bos_before_voice` prefix) and used to prefill the flow-LM state.
//! This is neither GPT ICL nor Qwen x-vector, so it reports a distinct
//! `audio_prompt` clone mode and never requires a transcript.
//!
//! Without the `pocket-tts` Cargo feature every load/synthesis path fails
//! with an explicit `model_missing` diagnostic (never silence, never a
//! Supertonic fallback).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::TtsProvider;
use super::assets::PinnedFile;
use super::error::ProviderError;
use super::types::{
    CloneMode, DevicePreference, EffectiveDevice, PcmAudio, ProviderCapabilities, ProviderRequest,
    ProviderStatus, ProviderStatusState,
};

mod manifests;
pub use manifests::*;

/// Stable provider identifier.
pub const POCKET_TTS_PROVIDER_ID: &str = "pocket-tts";
/// Pinned export revision (immutable commit on `KevinAHM/pocket-tts-onnx`).
pub const POCKET_TTS_UPSTREAM_REVISION: &str = "58a6d00cf13d239b6748cb0769f35c580a8f606c";
pub const POCKET_TTS_REPO: &str = "KevinAHM/pocket-tts-onnx";
/// Independently pinned language bundles. French uses the larger 24-layer preview.
pub use super::catalog::POCKET_TTS_MODELS;
pub const POCKET_TTS_LANGUAGES: &[&str] = &["en", "es", "pt", "de", "it", "fr"];

/// Exact language for a model; no cross-language fallback or three-letter aliases.
pub fn pocket_model_language(model: &str) -> Option<&'static str> {
    match model {
        "pocket-tts-en" => Some("en"),
        "pocket-tts-es" => Some("es"),
        "pocket-tts-pt" => Some("pt"),
        "pocket-tts-de" => Some("de"),
        "pocket-tts-it" => Some("it"),
        "pocket-tts-fr-24l" => Some("fr"),
        _ => None,
    }
}
/// Native sample rate of every validated bundle.
pub const POCKET_TTS_SAMPLE_RATE: u32 = 24_000;
/// Reference window (mirrors the upstream Python runtime's soft 1–30 s).
pub const POCKET_REF_MIN_SECS: f64 = crate::tts::voices::REFERENCE_MIN_SECS;
pub const POCKET_REF_MAX_SECS: f64 = crate::tts::voices::REFERENCE_MAX_SECS;
/// Reference audio is consumed as 24 kHz mono PCM (resampled on load).
pub const POCKET_REFERENCE_RATE: u32 = 24_000;
/// Hard output bound (ten minutes at 24 kHz), mirroring GPT-SoVITS policy.
pub const POCKET_MAX_SAMPLES: usize = 24_000 * 600;

fn bundle_subdir(model_id: &str) -> Option<&'static str> {
    match model_id {
        "pocket-tts-en" => Some("english_2026-04"),
        "pocket-tts-es" => Some("spanish"),
        "pocket-tts-pt" => Some("portuguese"),
        "pocket-tts-de" => Some("german"),
        "pocket-tts-it" => Some("italian"),
        "pocket-tts-fr-24l" => Some("french_24l"),
        _ => None,
    }
}

/// Files every validated bundle must contain (relative to the model dir).
pub const POCKET_TTS_REQUIRED_FILES: &[&str] = &[
    "bundle.json",
    "tokenizer.model",
    "bos_before_voice.npy",
    "text_conditioner.onnx",
    "mimi_encoder.onnx",
    "flow_lm_main_int8.onnx",
    "flow_lm_flow_int8.onnx",
    "mimi_decoder_int8.onnx",
];

/// Validated English bundle manifest: (relative path, SHA-256 hex, bytes).
/// Hashes measured from the downloaded artifacts (see Phase A evidence).
pub const POCKET_TTS_EN_MANIFEST: &[(&str, &str, u64)] = &[
    (
        "onnx/english_2026-04/bos_before_voice.npy",
        "f46edf4f7007b7ba4ea58831f49d003e59e167b4641c44bb3addfe9231a780b1",
        4224,
    ),
    (
        "onnx/english_2026-04/bundle.json",
        "bab643150f437f37df080a710520ff39ed9ebd9a339f8ebdc739f7eddfc28b3f",
        24381,
    ),
    (
        "onnx/english_2026-04/flow_lm_flow_int8.onnx",
        "3dd781ee5abee9e195320bf0106bebd6372a852b3b36352524ee78b40554635d",
        9962530,
    ),
    (
        "onnx/english_2026-04/flow_lm_main_int8.onnx",
        "f9bd8106b79a0192c1c43399ab938fb24900a95c1c599870d75a884e99000116",
        76341079,
    ),
    (
        "onnx/english_2026-04/mimi_decoder_int8.onnx",
        "3630450a3297a101792a6ac66619ebc70ab916b265e6220c2afaef8b1673f925",
        22684077,
    ),
    (
        "onnx/english_2026-04/mimi_encoder.onnx",
        "853e2ca623b8782d94c3745ec6133bfdff7ce33d9b11128bd29ea03f28d76e3d",
        39768446,
    ),
    (
        "onnx/english_2026-04/text_conditioner.onnx",
        "4ecee995fb69f85c7a7493d11f7b5ee15d9950facc7ab3f5c9c49ef1e03847bb",
        16388344,
    ),
    (
        "onnx/english_2026-04/tokenizer.model",
        "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6",
        59339,
    ),
];

/// Validated Spanish bundle manifest.
pub const POCKET_TTS_ES_MANIFEST: &[(&str, &str, u64)] = &[
    (
        "onnx/spanish/bos_before_voice.npy",
        "2c77a716ac206316d6656bdb4cf09e8a2fc9fa7a7f6cde4cceadc3fd7bb1d2a1",
        4224,
    ),
    (
        "onnx/spanish/bundle.json",
        "c9f18b5fc326c2a99f645588b79ef27ef3c9c6c3066f8df73bccb7f7c1f855b3",
        24365,
    ),
    (
        "onnx/spanish/flow_lm_flow_int8.onnx",
        "4cc630a8fc1d28e3b432349e492f5f81696aab88b1067b5282b15427fde19efe",
        9962530,
    ),
    (
        "onnx/spanish/flow_lm_main_int8.onnx",
        "9b020318976e0437bcc78ad0f4a49bf1de6e6e2193c6b0380398038601f9a270",
        76341079,
    ),
    (
        "onnx/spanish/mimi_decoder_int8.onnx",
        "309efd11b65c58cd176e554d09241f3e2c9aaabd8cbf74b8f01bdd3ec8f7d8e5",
        22684077,
    ),
    (
        "onnx/spanish/mimi_encoder.onnx",
        "6702fb117c58ff1f486f0b5153cafa41bca3811f08c560cc857ba445dd98a614",
        39768446,
    ),
    (
        "onnx/spanish/text_conditioner.onnx",
        "836ad5231fffe6d8fd23c8340d6d91676f466cd35057afd06b7db6ab69baae11",
        16388344,
    ),
    (
        "onnx/spanish/tokenizer.model",
        "aac2b96478e3e3068712bd5c45dc196f84b7341ea72587c33a2094957f0c1b0c",
        60895,
    ),
];

/// Pinned download manifest for a supported model ID. Unvalidated bundle
/// IDs (including unavailable Chinese, Japanese and Dutch exports) fail
/// with explicit guidance instead of a silent English fallback.
pub fn pocket_manifest(model_id: &str) -> Result<Vec<PinnedFile>, ProviderError> {
    let table = match model_id {
        "pocket-tts-en" => POCKET_TTS_EN_MANIFEST,
        "pocket-tts-es" => POCKET_TTS_ES_MANIFEST,
        "pocket-tts-pt" => POCKET_TTS_PT_MANIFEST,
        "pocket-tts-de" => POCKET_TTS_DE_MANIFEST,
        "pocket-tts-it" => POCKET_TTS_IT_MANIFEST,
        "pocket-tts-fr-24l" => POCKET_TTS_FR_MANIFEST,
        _ => {
            return Err(ProviderError::unsupported_model(
                POCKET_TTS_PROVIDER_ID,
                model_id,
                format!(
                    "only validated Pocket TTS bundles are supported; expected one of: {}",
                    POCKET_TTS_MODELS.join(", ")
                ),
            ));
        }
    };
    Ok(table
        .iter()
        .map(|(path, sha256, size)| PinnedFile {
            path,
            url: PinnedFile::huggingface(POCKET_TTS_REPO, POCKET_TTS_UPSTREAM_REVISION, path),
            sha256,
            size: *size,
        })
        .collect())
}

/// Pocket TTS provider configuration (`[providers.pocket_tts]`).
#[derive(Debug, Clone, PartialEq)]
pub struct PocketTtsConfig {
    pub enabled: bool,
    pub model_dir: PathBuf,
    pub model_id: String,
    pub revision: String,
    pub device: DevicePreference,
    pub cache_capacity: usize,
    pub temperature: f32,
    pub lsd_steps: usize,
    pub cpu_threads: usize,
}

impl Default for PocketTtsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: PathBuf::from("./models/pocket-tts"),
            model_id: POCKET_TTS_MODELS[0].to_string(),
            revision: POCKET_TTS_UPSTREAM_REVISION.to_string(),
            device: DevicePreference::Auto,
            cache_capacity: 16,
            temperature: 0.7,
            lsd_steps: 1,
            cpu_threads: 4,
        }
    }
}

/// Minimal SentencePiece unigram encoder over a bundle's `tokenizer.model`.
///
/// The validated tokenizers use the identity normalizer with dummy-prefix
/// whitespace encoding, unigram pieces and full 256-byte fallback, so a
/// Viterbi pass over the piece table reproduces Python sentencepiece
/// exactly with no third-party tokenizer dependency. Only the piece table
/// (field 1) of the ModelProto is parsed; every other field is skipped by
/// a small protobuf wire-format walker.
/// Minimal SentencePiece unigram encoder over a bundle's `tokenizer.model`.
///
/// The validated tokenizers use the identity normalizer with dummy-prefix
/// whitespace encoding, unigram pieces and full 256-byte fallback, so a
/// Viterbi pass over the piece table reproduces Python sentencepiece
/// exactly with no third-party tokenizer dependency. Only the piece table
/// (field 1: bytes, score, type) of the ModelProto is parsed; every other
/// field is skipped by a small protobuf wire-format walker.
pub mod tokenizer {
    use std::path::Path;

    /// SentencePiece ModelProto piece types we care about.
    const TYPE_NORMAL: u8 = 1;
    const TYPE_USER_DEFINED: u8 = 4;
    const TYPE_BYTE: u8 = 6;

    #[derive(Debug, Clone)]
    struct Piece {
        text: Vec<u8>,
        score: f32,
        ty: u8,
    }

    #[derive(Debug, Clone)]
    pub struct UnigramTokenizer {
        /// Pieces in model-id order.
        pieces: Vec<Piece>,
        /// Normal/user-defined piece ids grouped by first byte.
        match_index: Vec<usize>,
        first_by_byte: [usize; 256],
        /// Byte value -> byte-fallback piece id (`<0xXX>`).
        byte_id: [i64; 256],
    }

    fn read_varint(data: &[u8], mut pos: usize) -> Option<(u64, usize)> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *data.get(pos)?;
            pos += 1;
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Some((value, pos));
            }
        }
        None
    }

    fn skip_field(data: &[u8], wire: u8, pos: usize) -> Option<usize> {
        match wire {
            0 => read_varint(data, pos).map(|(_, p)| p),
            1 => Some(pos + 8),
            2 => {
                let (len, pos) = read_varint(data, pos)?;
                Some(pos + len as usize)
            }
            5 => Some(pos + 4),
            _ => None,
        }
    }

    fn parse_pieces(data: &[u8]) -> Option<Vec<Piece>> {
        let mut pieces = Vec::new();
        let mut pos = 0;
        while pos < data.len() {
            let (tag, next) = read_varint(data, pos)?;
            pos = next;
            let (field, wire) = ((tag >> 3) as u32, (tag & 7) as u8);
            if field == 1 && wire == 2 {
                let (len, start) = read_varint(data, pos)?;
                let end = start + len as usize;
                let msg = data.get(start..end)?;
                pos = end;
                let mut text = Vec::new();
                let mut score = 0.0f32;
                // NORMAL is the proto default: pieces without an explicit
                // type field are regular vocabulary entries.
                let mut ty = 1u8;
                let mut inner = 0;
                while inner < msg.len() {
                    let (itag, inner_next) = read_varint(msg, inner)?;
                    inner = inner_next;
                    let (f, w) = ((itag >> 3) as u32, (itag & 7) as u8);
                    match (f, w) {
                        (1, 2) => {
                            let (l, s) = read_varint(msg, inner)?;
                            let e = s + l as usize;
                            text = msg.get(s..e)?.to_vec();
                            inner = e;
                        }
                        (2, 5) => {
                            let bytes = msg.get(inner..inner + 4)?;
                            score = f32::from_le_bytes(bytes.try_into().ok()?);
                            inner += 4;
                        }
                        (3, 0) => {
                            let (v, n) = read_varint(msg, inner)?;
                            ty = v as u8;
                            inner = n;
                        }
                        _ => inner = skip_field(msg, w, inner)?,
                    }
                }
                pieces.push(Piece { text, score, ty });
            } else {
                pos = skip_field(data, wire, pos)?;
            }
        }
        if pieces.is_empty() {
            None
        } else {
            Some(pieces)
        }
    }

    fn byte_value(text: &[u8]) -> Option<u8> {
        // Parses `<0xXX>` (uppercase or lowercase hex).
        if text.len() != 6 || !text.starts_with(b"<0x") || !text.ends_with(b">") {
            return None;
        }
        let hex = std::str::from_utf8(&text[3..5]).ok()?;
        u8::from_str_radix(hex, 16).ok()
    }

    impl UnigramTokenizer {
        pub fn load(path: &Path) -> std::io::Result<Self> {
            let data = std::fs::read(path)?;
            Self::from_model_bytes(&data)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no pieces"))
        }

        fn from_model_bytes(data: &[u8]) -> Option<Self> {
            let pieces = parse_pieces(data)?;
            let mut byte_id = [0i64; 256];
            let mut byte_seen = [false; 256];
            for (id, piece) in pieces.iter().enumerate() {
                if piece.ty == TYPE_BYTE
                    && let Some(b) = byte_value(&piece.text)
                {
                    byte_id[b as usize] = id as i64;
                    byte_seen[b as usize] = true;
                }
            }
            if byte_seen.iter().any(|seen| !seen) {
                return None;
            }
            let mut match_index: Vec<usize> = pieces
                .iter()
                .enumerate()
                .filter(|(_, p)| p.ty == TYPE_NORMAL || p.ty == TYPE_USER_DEFINED)
                .map(|(id, _)| id)
                .collect();
            match_index.sort_by_key(|&id| pieces[id].text.first().copied().unwrap_or(0));
            let mut first_by_byte = [usize::MAX; 256];
            let mut current: Option<u8> = None;
            for (pos, &id) in match_index.iter().enumerate() {
                let byte = pieces[id].text.first().copied().unwrap_or(0);
                if current != Some(byte) {
                    current = Some(byte);
                    first_by_byte[byte as usize] = pos;
                }
            }
            Some(Self {
                pieces,
                match_index,
                first_by_byte,
                byte_id,
            })
        }

        /// Encode without adding BOS/EOS (matches upstream `Encode(text)`).
        /// A dummy `▁` prefix is prepended and ASCII spaces become `▁`
        /// (`escape_whitespaces`); every other byte is matched literally
        /// with single-byte fallback.
        pub fn encode(&self, text: &str) -> Vec<i64> {
            let mut input: Vec<u8> = Vec::with_capacity(text.len() + 3);
            input.extend_from_slice("▁".as_bytes());
            for byte in text.as_bytes() {
                if *byte == b' ' {
                    input.extend_from_slice("▁".as_bytes());
                } else {
                    input.push(*byte);
                }
            }
            let n = input.len();
            // best[i] = (score, piece id, byte length).
            let mut best: Vec<Option<(f32, i64, usize)>> = vec![None; n + 1];
            best[0] = Some((0.0, -1, 0));
            for i in 0..n {
                let Some((base, _, _)) = best[i] else {
                    continue;
                };
                let first = input[i];
                // Literal piece matches. Byte fallback is a last resort:
                // upstream only emits a byte piece where no vocab piece
                // matches, it never outscores real pieces.
                let mut matched = false;
                let mut k = self.first_by_byte[first as usize];
                if k != usize::MAX {
                    while k < self.match_index.len() {
                        let id = self.match_index[k] as i64;
                        let piece = &self.pieces[id as usize];
                        if piece.text.first().copied() != Some(first) {
                            break;
                        }
                        k += 1;
                        let len = piece.text.len();
                        if len > 0 && input[i..].starts_with(&piece.text) {
                            matched = true;
                            let score = base + piece.score;
                            let better = best[i + len].is_none_or(|(s, _, _)| score > s);
                            if better {
                                best[i + len] = Some((score, id, len));
                            }
                        }
                    }
                }
                if !matched {
                    let byte_id = self.byte_id[first as usize];
                    let score = base + self.pieces[byte_id as usize].score;
                    let better = best[i + 1].is_none_or(|(s, _, _)| score > s);
                    if better {
                        best[i + 1] = Some((score, byte_id, 1));
                    }
                }
            }
            let mut ids = Vec::new();
            let mut rev = Vec::new();
            let mut i = n;
            while i > 0 {
                match best[i] {
                    Some((_, id, len)) if len > 0 => {
                        rev.push(id);
                        i -= len;
                    }
                    _ => {
                        rev.push(0);
                        i -= 1;
                    }
                }
            }
            rev.reverse();
            ids.extend(rev);
            ids
        }

        /// Decode piece ids back to text: the leading dummy `▁` is
        /// dropped and every other `▁` becomes a space (upstream
        /// sentencepiece behavior).
        pub fn decode(&self, ids: &[i64]) -> String {
            let mut bytes = Vec::new();
            for id in ids {
                if let Some(piece) = self.pieces.get(*id as usize) {
                    if piece.ty == TYPE_BYTE {
                        // SentencePiece byte fallback is UTF-8 data, not the
                        // literal spelling "<0xXX>". Reassemble the bytes
                        // before decoding so segmentation preserves accents
                        // and any other out-of-vocabulary Unicode scalar.
                        if let Some(byte) = byte_value(&piece.text) {
                            bytes.push(byte);
                        }
                    } else {
                        bytes.extend_from_slice(&piece.text);
                    }
                }
            }
            let text = String::from_utf8_lossy(&bytes);
            text.strip_prefix('▁').unwrap_or(&text).replace('▁', " ")
        }
    }
}

/// Parsed `bundle.json` metadata driving shapes, prompts and state init.
#[derive(Debug, Clone)]
pub struct BundleMeta {
    pub sample_rate: u32,
    pub frame_rate: f32,
    pub samples_per_frame: usize,
    pub latent_dim: usize,
    pub conditioning_dim: usize,
    pub tokenizer_file: String,
    pub bos_before_voice_file: Option<String>,
    pub insert_bos_before_voice: bool,
    pub pad_with_spaces_for_short_inputs: bool,
    pub remove_semicolons: bool,
    pub model_recommended_frames_after_eos: Option<usize>,
    pub max_token_per_chunk: usize,
    pub flow_state_manifest: Vec<StateEntry>,
    pub mimi_state_manifest: Vec<StateEntry>,
}

/// One state-tensor entry from a bundle state manifest.
#[derive(Debug, Clone)]
pub struct StateEntry {
    pub input_name: String,
    pub output_name: String,
    pub dtype: StateDtype,
    pub fill: StateFill,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateDtype {
    F32,
    I64,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateFill {
    Nan,
    Ones,
    Zeros,
}

fn parse_state_manifest(value: &serde_json::Value) -> Result<Vec<StateEntry>, String> {
    let items = value
        .as_array()
        .ok_or_else(|| "state manifest is not an array".to_string())?;
    items
        .iter()
        .map(|e| {
            let get = |k: &str| e.get(k).ok_or_else(|| format!("state entry missing '{k}'"));
            let dtype = match get("dtype")?.as_str().unwrap_or("") {
                "float32" => StateDtype::F32,
                "int64" => StateDtype::I64,
                "bool" => StateDtype::Bool,
                d => return Err(format!("unsupported state dtype '{d}'")),
            };
            let fill = match get("fill")?.as_str().unwrap_or("zeros") {
                "nan" => StateFill::Nan,
                "ones" => StateFill::Ones,
                _ => StateFill::Zeros,
            };
            let shape = get("shape")?
                .as_array()
                .ok_or_else(|| "state shape is not an array".to_string())?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            Ok(StateEntry {
                input_name: get("input_name")?.as_str().unwrap_or("").to_string(),
                output_name: get("output_name")?.as_str().unwrap_or("").to_string(),
                dtype,
                fill,
                shape,
            })
        })
        .collect()
}

impl BundleMeta {
    pub fn load(bundle_dir: &Path) -> Result<Self, ProviderError> {
        let raw = std::fs::read_to_string(bundle_dir.join("bundle.json")).map_err(|e| {
            ProviderError::model_missing(format!("pocket-tts bundle.json unreadable: {e}"))
        })?;
        let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            ProviderError::model_missing(format!("pocket-tts bundle.json invalid: {e}"))
        })?;
        let get = |k: &str| {
            v.get(k)
                .ok_or_else(|| ProviderError::model_missing(format!("bundle.json missing '{k}'")))
        };
        let u32f = |k: &str| {
            get(k)?
                .as_u64()
                .ok_or_else(|| {
                    ProviderError::model_missing(format!("bundle.json '{k}' not a number"))
                })
                .map(|n| n as u32)
        };
        Ok(Self {
            sample_rate: u32f("sample_rate")?,
            frame_rate: get("frame_rate")?
                .as_f64()
                .ok_or_else(|| ProviderError::model_missing("bundle.json 'frame_rate'"))?
                as f32,
            samples_per_frame: u32f("samples_per_frame")? as usize,
            latent_dim: u32f("latent_dim")? as usize,
            conditioning_dim: u32f("conditioning_dim")? as usize,
            tokenizer_file: get("tokenizer_file")?
                .as_str()
                .ok_or_else(|| ProviderError::model_missing("bundle.json 'tokenizer_file'"))?
                .to_string(),
            bos_before_voice_file: v
                .get("bos_before_voice_file")
                .and_then(|s| s.as_str())
                .map(str::to_string),
            insert_bos_before_voice: v
                .get("insert_bos_before_voice")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            pad_with_spaces_for_short_inputs: v
                .get("pad_with_spaces_for_short_inputs")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            remove_semicolons: v
                .get("remove_semicolons")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            model_recommended_frames_after_eos: v
                .get("model_recommended_frames_after_eos")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize),
            max_token_per_chunk: v
                .get("max_token_per_chunk")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(50),
            flow_state_manifest: parse_state_manifest(get("flow_lm_state_manifest")?)
                .map_err(ProviderError::model_missing)?,
            mimi_state_manifest: parse_state_manifest(get("mimi_state_manifest")?)
                .map_err(ProviderError::model_missing)?,
        })
    }
}

/// Prepare text exactly like the upstream runtime: strip, single-space
/// newlines, optional semicolon rewrite, capitalize the first letter and
/// terminate with a period when the text ends alphanumerically.
pub fn prepare_text_prompt(
    text: &str,
    remove_semicolons: bool,
    pad_with_spaces: bool,
) -> Result<(String, usize), ProviderError> {
    let mut text = text.trim().to_string();
    if text.is_empty() {
        return Err(ProviderError::invalid_voice("text cannot be empty"));
    }
    // Single-pass rewrites, exactly like the upstream runtime.
    text = text.replace(['\n', '\r'], " ").replace("  ", " ");
    if remove_semicolons {
        text = text.replace(';', ",");
    }
    let word_count = text.split_whitespace().count();
    let guess = if word_count <= 4 { 3 } else { 1 };
    let mut chars: Vec<char> = text.chars().collect();
    if let Some(first) = chars.first_mut()
        && first.is_lowercase()
    {
        // `to_uppercase` may yield several chars (e.g. ß -> SS).
        let upper: String = first.to_uppercase().collect();
        let mut expanded: Vec<char> = upper.chars().collect();
        expanded.extend(chars.into_iter().skip(1));
        chars = expanded;
    }
    if let Some(last) = chars.last()
        && last.is_alphanumeric()
    {
        chars.push('.');
    }
    text = chars.into_iter().collect();
    if pad_with_spaces && text.split_whitespace().count() < 5 {
        text = format!("        {text}");
    }
    Ok((text, guess))
}

/// Minimal `.npy` loader for the bundle's `bos_before_voice` vector
/// (single float32 C-order array). Anything else is rejected explicitly.
pub fn load_npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let data = std::fs::read(path).map_err(|e| format!("npy unreadable: {e}"))?;
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".to_string());
    }
    let major = data[6];
    let header_len = if major == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize + 10
    } else if major == 2 {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize + 12
    } else {
        return Err(format!("unsupported .npy version {major}"));
    };
    let header = std::str::from_utf8(data.get(10..header_len).unwrap_or(&[]))
        .map_err(|_| "npy header is not UTF-8".to_string())?;
    let is_f32 = header.contains("<f4")
        || header.contains(">f4")
        || header.contains("|f4")
        || header.contains("float32");
    if !is_f32 {
        return Err("npy array is not float32".to_string());
    }
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return Err("fortran-order .npy is unsupported".to_string());
    }
    let shape_src = header
        .split_once("'shape'")
        .or_else(|| header.split_once("\"shape\""))
        .map(|(_, rest)| rest)
        .unwrap_or("");
    let shape_src = shape_src
        .split_once('(')
        .map(|(_, rest)| rest.split_once(')').map(|(s, _)| s).unwrap_or(""))
        .unwrap_or("");
    let shape: Vec<usize> = shape_src
        .split(',')
        .filter_map(|d| d.trim().parse::<usize>().ok())
        .collect();
    let body = data.get(header_len..).ok_or("npy body missing")?;
    if body.len() % 4 != 0 {
        return Err("npy body is not a multiple of 4 bytes".to_string());
    }
    let mut values = Vec::with_capacity(body.len() / 4);
    let (chunks, _) = body.as_chunks::<4>();
    for chunk in chunks {
        values.push(f32::from_le_bytes(*chunk));
    }
    Ok((shape, values))
}

#[cfg(feature = "pocket-tts")]
use ort::session::Session;
#[cfg(feature = "pocket-tts")]
use ort::value::{Tensor, Value};
#[cfg(feature = "pocket-tts")]
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Resident native Pocket TTS model: five CPU sessions plus bundle data.
/// All synthesis runs as one atomic unit under the provider mutex, so
/// request-local KV states can never leak between concurrent voices.
#[cfg(feature = "pocket-tts")]
pub struct PocketNative {
    meta: BundleMeta,
    tokenizer: tokenizer::UnigramTokenizer,
    bos: Vec<f32>,
    conditioner: Session,
    encoder: Session,
    lm_main: Session,
    lm_flow: Session,
    decoder: Session,
    eos_boundary_ids: Vec<i64>,
    fallback_ids: Vec<i64>,
}

#[cfg(feature = "pocket-tts")]
fn ort_session(path: PathBuf, cpu_threads: usize) -> Result<Session, ProviderError> {
    crate::onnx_runtime::initialize()
        .map_err(|e| ProviderError::inference_failed(format!("{e:#}")))?;
    let threads = cpu_threads.clamp(1, 32);
    Session::builder()
        .map_err(|e| ProviderError::inference_failed(format!("pocket-tts session setup: {e}")))?
        .with_intra_threads(threads)
        .map_err(|e| ProviderError::inference_failed(format!("pocket-tts thread setup: {e}")))?
        .commit_from_file(&path)
        .map_err(|e| {
            ProviderError::inference_failed(format!(
                "pocket-tts native load failed ({}): {e}",
                path.display()
            ))
        })
}

#[cfg(feature = "pocket-tts")]
use ort::session::SessionOutputs;
#[cfg(feature = "pocket-tts")]
use ort::value::DynValue;

/// Up-front reservation for the AR latent buffer, capped at the output
/// sample bound so a degenerate frame count cannot turn the reservation
/// itself into a huge allocation. The buffer still grows past the cap when
/// generation legitimately needs more room.
#[cfg(feature = "pocket-tts")]
fn latents_capacity(frame_limit: usize, latent_dim: usize) -> usize {
    frame_limit
        .saturating_mul(latent_dim)
        .min(POCKET_MAX_SAMPLES)
}

/// Owned KV-cache/state tensor. ORT outputs borrow the session, so every
/// step extracts into these and rebuilds inputs for the next step. Shapes
/// always come from the bundle manifest.
///
/// On ORT ownership and the per-frame copies in [`push_states`]: ORT's
/// `Tensor::from_array` takes ownership of its buffer while the state must
/// also persist for the next frame, so the state-to-tensor clone is
/// semantically mandatory without I/O binding (which ORT 2.x sessions here
/// do not expose safely); likewise `extract_states` must copy out of the
/// borrowed session outputs before they are dropped. The bindings that *are*
/// avoidable — the output `latents` reallocations and the per-frame flow
/// scratch buffer — are reused instead (see `run_chunk`).
#[cfg(feature = "pocket-tts")]
#[derive(Debug, Clone)]
enum StateTensor {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

#[cfg(feature = "pocket-tts")]
fn init_states(manifest: &[StateEntry]) -> Vec<StateTensor> {
    manifest
        .iter()
        .map(|entry| {
            let n: usize = entry.shape.iter().product();
            match entry.dtype {
                StateDtype::F32 => StateTensor::F32(match entry.fill {
                    StateFill::Nan => vec![f32::NAN; n],
                    StateFill::Ones => vec![1.0; n],
                    StateFill::Zeros => vec![0.0; n],
                }),
                StateDtype::I64 => StateTensor::I64(match entry.fill {
                    StateFill::Ones => vec![1i64; n],
                    _ => vec![0i64; n],
                }),
                StateDtype::Bool => StateTensor::Bool(vec![false; n]),
            }
        })
        .collect()
}

#[cfg(feature = "pocket-tts")]
fn push_states(
    inputs: &mut Vec<(String, Value)>,
    manifest: &[StateEntry],
    states: &[StateTensor],
) -> Result<(), ProviderError> {
    for (entry, state) in manifest.iter().zip(states.iter()) {
        let value: Value = match state {
            StateTensor::F32(values) => {
                Tensor::from_array((entry.shape.as_slice(), values.clone()))
                    .map_err(|e| {
                        ProviderError::inference_failed(format!("pocket-tts state input: {e}"))
                    })?
                    .into()
            }
            StateTensor::I64(values) => {
                Tensor::from_array((entry.shape.as_slice(), values.clone()))
                    .map_err(|e| {
                        ProviderError::inference_failed(format!("pocket-tts state input: {e}"))
                    })?
                    .into()
            }
            StateTensor::Bool(values) => {
                Tensor::from_array((entry.shape.as_slice(), values.clone()))
                    .map_err(|e| {
                        ProviderError::inference_failed(format!("pocket-tts state input: {e}"))
                    })?
                    .into()
            }
        };
        inputs.push((entry.input_name.clone(), value));
    }
    Ok(())
}

#[cfg(feature = "pocket-tts")]
fn extract_states(
    manifest: &[StateEntry],
    states: &mut [StateTensor],
    outputs: &SessionOutputs<'_>,
) -> Result<(), ProviderError> {
    for (entry, state) in manifest.iter().zip(states.iter_mut()) {
        let value: &DynValue = outputs.get(entry.output_name.as_str()).ok_or_else(|| {
            ProviderError::inference_failed(format!(
                "pocket-tts state output '{}' missing",
                entry.output_name
            ))
        })?;
        match (entry.dtype, state) {
            (StateDtype::F32, StateTensor::F32(slot)) => {
                let (_, values) = value.try_extract_tensor::<f32>().map_err(|e| {
                    ProviderError::inference_failed(format!("pocket-tts state decode: {e}"))
                })?;
                slot.clear();
                slot.extend_from_slice(values);
            }
            (StateDtype::I64, StateTensor::I64(slot)) => {
                let (_, values) = value.try_extract_tensor::<i64>().map_err(|e| {
                    ProviderError::inference_failed(format!("pocket-tts state decode: {e}"))
                })?;
                slot.clear();
                slot.extend_from_slice(values);
            }
            (StateDtype::Bool, StateTensor::Bool(slot)) => {
                let (_, values) = value.try_extract_tensor::<bool>().map_err(|e| {
                    ProviderError::inference_failed(format!("pocket-tts state decode: {e}"))
                })?;
                slot.clear();
                slot.extend_from_slice(values);
            }
            _ => {
                return Err(ProviderError::inference_failed(format!(
                    "pocket-tts state dtype changed for '{}'",
                    entry.input_name
                )));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "pocket-tts")]
fn take_f32(outputs: &SessionOutputs<'_>, name: &str) -> Result<Vec<f32>, ProviderError> {
    let value: &DynValue = outputs.get(name).ok_or_else(|| {
        ProviderError::inference_failed(format!("pocket-tts output '{name}' missing"))
    })?;
    let (_, values) = value.try_extract_tensor::<f32>().map_err(|e| {
        ProviderError::inference_failed(format!("pocket-tts output '{name}' unreadable: {e}"))
    })?;
    Ok(values.to_vec())
}

#[cfg(feature = "pocket-tts")]
impl PocketNative {
    fn open(model_dir: &Path, model_id: &str, cpu_threads: usize) -> Result<Self, ProviderError> {
        let subdir = bundle_subdir(model_id).ok_or_else(|| {
            ProviderError::unsupported_model(
                POCKET_TTS_PROVIDER_ID,
                model_id,
                format!("expected one of: {}", POCKET_TTS_MODELS.join(", ")),
            )
        })?;
        let bundle = model_dir.join("onnx").join(subdir);
        let meta = BundleMeta::load(&bundle)?;
        if meta.sample_rate != POCKET_TTS_SAMPLE_RATE {
            return Err(ProviderError::model_missing(format!(
                "pocket-tts bundle sample rate {} != {POCKET_TTS_SAMPLE_RATE}",
                meta.sample_rate
            )));
        }
        let tokenizer = tokenizer::UnigramTokenizer::load(&bundle.join(&meta.tokenizer_file))
            .map_err(|e| {
                ProviderError::model_missing(format!("pocket-tts tokenizer unreadable: {e}"))
            })?;
        let bos = match (&meta.bos_before_voice_file, meta.insert_bos_before_voice) {
            (Some(file), true) => {
                let (shape, values) =
                    load_npy_f32(&bundle.join(file)).map_err(ProviderError::model_missing)?;
                if shape.iter().product::<usize>() != values.len()
                    || values.len() != meta.conditioning_dim
                {
                    return Err(ProviderError::model_missing(format!(
                        "bos_before_voice shape {shape:?} != [1, 1, {}]",
                        meta.conditioning_dim
                    )));
                }
                values
            }
            _ => Vec::new(),
        };
        let file = |name: &str| bundle.join(name);
        let mut conditioner = ort_session(file("text_conditioner.onnx"), cpu_threads)?;
        let mut encoder = ort_session(file("mimi_encoder.onnx"), cpu_threads)?;
        let mut lm_main = ort_session(file("flow_lm_main_int8.onnx"), cpu_threads)?;
        let mut lm_flow = ort_session(file("flow_lm_flow_int8.onnx"), cpu_threads)?;
        let mut decoder = ort_session(file("mimi_decoder_int8.onnx"), cpu_threads)?;
        // Warm the session I/O bindings once so first-request latency is
        // steady; failures here are real load failures, not deferred.
        let _ = (
            &mut conditioner,
            &mut encoder,
            &mut lm_main,
            &mut lm_flow,
            &mut decoder,
        );
        let eos_boundary_ids: Vec<i64> = tokenizer.encode(".!...?").into_iter().skip(1).collect();
        let fallback_ids: Vec<i64> = tokenizer.encode(",;:").into_iter().skip(1).collect();
        Ok(Self {
            meta,
            tokenizer,
            bos,
            conditioner,
            encoder,
            lm_main,
            lm_flow,
            decoder,
            eos_boundary_ids,
            fallback_ids,
        })
    }

    /// Encode 24 kHz mono reference PCM to voice embeddings (frames x 1024).
    fn encode_voice(&mut self, pcm24: &[f32]) -> Result<(usize, Vec<f32>), ProviderError> {
        let inputs: Vec<(String, Value)> = vec![(
            "audio".to_string(),
            Tensor::from_array(([1, 1, pcm24.len()], pcm24.to_vec()))
                .map_err(|e| {
                    ProviderError::inference_failed(format!("pocket-tts voice input: {e}"))
                })?
                .into(),
        )];
        let outputs = self.encoder.run(inputs).map_err(|e| {
            ProviderError::inference_failed(format!("pocket-tts voice encoding failed: {e}"))
        })?;
        let latents = take_f32(&outputs, "latents")?;
        // Output shape is [1, frames, 1024]; recover the frame count.
        let frames = latents.len() / self.meta.conditioning_dim;
        if frames == 0 || latents.len() % self.meta.conditioning_dim != 0 {
            return Err(ProviderError::inference_failed(
                "pocket-tts voice encoder returned no frames",
            ));
        }
        if !latents.iter().all(|v| v.is_finite()) {
            return Err(ProviderError::inference_failed(
                "pocket-tts voice encoder returned non-finite embeddings",
            ));
        }
        Ok((frames, latents))
    }

    /// Prefill the flow-LM KV state from voice embeddings (with BOS prefix).
    fn prefill_voice(
        &mut self,
        frames: usize,
        embeddings: &[f32],
    ) -> Result<Vec<StateTensor>, ProviderError> {
        let dim = self.meta.conditioning_dim;
        let mut voice = Vec::with_capacity((frames + 1) * dim);
        if !self.bos.is_empty() {
            voice.extend_from_slice(&self.bos);
        }
        voice.extend_from_slice(embeddings);
        let total_frames = voice.len() / dim;
        let mut states = init_states(&self.meta.flow_state_manifest);
        let mut inputs: Vec<(String, Value)> = vec![
            (
                "sequence".to_string(),
                Tensor::from_array(([1, 0, self.meta.latent_dim], Vec::<f32>::new()))
                    .map_err(|e| {
                        ProviderError::inference_failed(format!("pocket-tts prefill input: {e}"))
                    })?
                    .into(),
            ),
            (
                "text_embeddings".to_string(),
                Tensor::from_array(([1, total_frames, dim], voice))
                    .map_err(|e| {
                        ProviderError::inference_failed(format!("pocket-tts prefill input: {e}"))
                    })?
                    .into(),
            ),
        ];
        push_states(&mut inputs, &self.meta.flow_state_manifest, &states)?;
        let outputs = self.lm_main.run(inputs).map_err(|e| {
            ProviderError::inference_failed(format!("pocket-tts voice prefill failed: {e}"))
        })?;
        extract_states(&self.meta.flow_state_manifest, &mut states, &outputs)?;
        Ok(states)
    }

    fn split_sentences(&self, text: &str) -> Result<Vec<String>, ProviderError> {
        let (prepared, _) = prepare_text_prompt(
            text,
            self.meta.remove_semicolons,
            self.meta.pad_with_spaces_for_short_inputs,
        )?;
        let prepared = prepared.trim().to_string();
        let tokens = self.tokenizer.encode(&prepared);
        let boundaries = boundary_indices(&tokens, &self.eos_boundary_ids);
        let mut segments: Vec<(usize, String)> = Vec::new();
        for pair in boundaries.windows(2) {
            let text = self.tokenizer.decode(&tokens[pair[0]..pair[1]]);
            segments.push((pair[1] - pair[0], text));
        }
        let mut refined: Vec<(usize, String)> = Vec::new();
        for (count, segment) in segments {
            if count <= self.meta.max_token_per_chunk {
                refined.push((count, segment));
                continue;
            }
            let sub_tokens = self.tokenizer.encode(segment.trim());
            let sub_bounds = boundary_indices(&sub_tokens, &self.fallback_ids);
            let mut subs: Vec<(usize, String)> = Vec::new();
            for pair in sub_bounds.windows(2) {
                subs.push((
                    pair[1] - pair[0],
                    self.tokenizer.decode(&sub_tokens[pair[0]..pair[1]]),
                ));
            }
            if subs.len() > 1 {
                refined.extend(subs);
            } else {
                refined.push((count, segment));
            }
        }
        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut current_count = 0usize;
        for (count, segment) in refined {
            if current.is_empty() {
                current = segment;
                current_count = count;
                continue;
            }
            if current_count + count > self.meta.max_token_per_chunk {
                chunks.push(current.trim().to_string());
                current = segment;
                current_count = count;
            } else {
                current = format!("{current} {segment}");
                current_count += count;
            }
        }
        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }
        Ok(chunks)
    }

    /// Run one text chunk: prefill text conditioning, then the AR latent
    /// loop with temperature sampling and LSD flow integration.
    fn run_chunk(
        &mut self,
        base_state: &[StateTensor],
        token_ids: &[i64],
        options: &ChunkOptions<'_>,
        rng: &mut StdRng,
    ) -> Result<Vec<f32>, ProviderError> {
        let frames_after_eos = options.frames_after_eos;
        let temperature = options.temperature;
        let lsd_steps = options.lsd_steps;
        let control = options.control;
        let dim = self.meta.conditioning_dim;
        let latent_dim = self.meta.latent_dim;
        // Text conditioning.
        let inputs: Vec<(String, Value)> = vec![(
            "token_ids".to_string(),
            Tensor::from_array(([1, token_ids.len()], token_ids.to_vec()))
                .map_err(|e| {
                    ProviderError::inference_failed(format!("pocket-tts token input: {e}"))
                })?
                .into(),
        )];
        let outputs = self.conditioner.run(inputs).map_err(|e| {
            ProviderError::inference_failed(format!("pocket-tts text conditioning failed: {e}"))
        })?;
        let text_emb = take_f32(&outputs, "embeddings")?;
        drop(outputs);
        // Prefill with the text embeddings over a clone of the voice state.
        let mut states: Vec<StateTensor> = base_state.to_vec();
        let mut inputs: Vec<(String, Value)> = vec![
            (
                "sequence".to_string(),
                Tensor::from_array(([1, 0, latent_dim], Vec::<f32>::new()))
                    .map_err(|e| {
                        ProviderError::inference_failed(format!("pocket-tts prefill input: {e}"))
                    })?
                    .into(),
            ),
            (
                "text_embeddings".to_string(),
                Tensor::from_array(([1, token_ids.len(), dim], text_emb))
                    .map_err(|e| {
                        ProviderError::inference_failed(format!("pocket-tts prefill input: {e}"))
                    })?
                    .into(),
            ),
        ];
        push_states(&mut inputs, &self.meta.flow_state_manifest, &states)?;
        let outputs = self.lm_main.run(inputs).map_err(|e| {
            ProviderError::inference_failed(format!("pocket-tts text prefill failed: {e}"))
        })?;
        extract_states(&self.meta.flow_state_manifest, &mut states, &outputs)?;
        drop(outputs);

        let frame_limit =
            ((token_ids.len() as f32 / 3.0 + 2.0).ceil() * self.meta.frame_rate) as usize;
        let dt = 1.0 / lsd_steps.max(1) as f32;
        let mut curr = vec![f32::NAN; latent_dim];
        // Reserve the full frame budget up front (capped at the output
        // sample bound) so per-frame appends never reallocate. The cap keeps
        // a corrupt/degenerate `frame_limit` from turning the reservation
        // itself into a huge allocation; growth beyond the cap still works.
        let mut latents: Vec<f32> = Vec::with_capacity(latents_capacity(frame_limit, latent_dim));
        // Reused across AR frames: every element is rewritten below on each
        // frame (zeros when temperature is 0), so reuse is value-identical
        // to a fresh allocation.
        let mut x = vec![0.0f32; latent_dim];
        let mut eos_step: Option<usize> = None;
        for step in 0..frame_limit {
            if control.is_some_and(|c| c.stopped()) {
                return Err(ProviderError::inference_failed(
                    "inference cancelled or deadline exceeded",
                ));
            }
            let mut inputs: Vec<(String, Value)> = vec![
                (
                    "sequence".to_string(),
                    Tensor::from_array(([1, 1, latent_dim], curr.clone()))
                        .map_err(|e| {
                            ProviderError::inference_failed(format!("pocket-tts AR input: {e}"))
                        })?
                        .into(),
                ),
                (
                    "text_embeddings".to_string(),
                    Tensor::from_array(([1, 0, dim], Vec::<f32>::new()))
                        .map_err(|e| {
                            ProviderError::inference_failed(format!("pocket-tts AR input: {e}"))
                        })?
                        .into(),
                ),
            ];
            push_states(&mut inputs, &self.meta.flow_state_manifest, &states)?;
            let outputs = self.lm_main.run(inputs).map_err(|e| {
                ProviderError::inference_failed(format!("pocket-tts AR step failed: {e}"))
            })?;
            let conditioning = take_f32(&outputs, "conditioning")?;
            let eos_logit = take_f32(&outputs, "eos_logit")?;
            extract_states(&self.meta.flow_state_manifest, &mut states, &outputs)?;
            drop(outputs);
            if eos_logit.first().is_some_and(|&v| v > -4.0) && eos_step.is_none() {
                eos_step = Some(step);
            }
            if eos_step.is_some_and(|s| step >= s + frames_after_eos) {
                break;
            }
            // Temperature sampling: x ~ N(0, sqrt(T)) per dim (Box-Muller).
            // `x` is the frame-reused scratch buffer: the sampling loop
            // below rewrites every element, and the zero-temperature path
            // resets it, so each frame starts from identical values.
            if temperature > 0.0 {
                let std = temperature.sqrt();
                for v in x.iter_mut() {
                    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                    let u2 = rng.random::<f64>();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    *v = (z * std as f64) as f32;
                }
            } else {
                x.fill(0.0);
            }
            // LSD Euler integration over the flow head.
            for j in 0..lsd_steps.max(1) {
                let s = j as f32 / lsd_steps.max(1) as f32;
                let t = s + dt;
                let inputs: Vec<(String, Value)> = vec![
                    (
                        "c".to_string(),
                        Tensor::from_array(([1, dim], conditioning.clone()))
                            .map_err(|e| {
                                ProviderError::inference_failed(format!("pocket-tts flow: {e}"))
                            })?
                            .into(),
                    ),
                    (
                        "s".to_string(),
                        Tensor::from_array(([1, 1], vec![s]))
                            .map_err(|e| {
                                ProviderError::inference_failed(format!("pocket-tts flow: {e}"))
                            })?
                            .into(),
                    ),
                    (
                        "t".to_string(),
                        Tensor::from_array(([1, 1], vec![t]))
                            .map_err(|e| {
                                ProviderError::inference_failed(format!("pocket-tts flow: {e}"))
                            })?
                            .into(),
                    ),
                    (
                        "x".to_string(),
                        Tensor::from_array(([1, latent_dim], x.clone()))
                            .map_err(|e| {
                                ProviderError::inference_failed(format!("pocket-tts flow: {e}"))
                            })?
                            .into(),
                    ),
                ];
                let outputs = self.lm_flow.run(inputs).map_err(|e| {
                    ProviderError::inference_failed(format!("pocket-tts flow step failed: {e}"))
                })?;
                let flow = take_f32(&outputs, "flow_dir")?;
                drop(outputs);
                for (x, f) in x.iter_mut().zip(flow.iter()) {
                    *x += *f * dt;
                }
            }
            latents.extend_from_slice(&x);
            // Double-buffer swap instead of a move: `curr` takes the new
            // frame (identical values to `curr = x`), while `x` reclaims the
            // previous frame's buffer for reuse next frame.
            std::mem::swap(&mut curr, &mut x);
            if control.is_some_and(|c| !c.update("generating", latents.len() / latent_dim)) {
                return Err(ProviderError::inference_failed(
                    "inference cancelled or deadline exceeded",
                ));
            }
            if latents.len() / latent_dim * self.meta.samples_per_frame > POCKET_MAX_SAMPLES {
                return Err(ProviderError::inference_failed(
                    "pocket-tts output exceeds ten-minute bound",
                ));
            }
        }
        Ok(latents)
    }

    /// Decode latent frames to PCM in 15-frame groups with state carry.
    fn decode_latents(&mut self, latents: &[f32]) -> Result<Vec<f32>, ProviderError> {
        let latent_dim = self.meta.latent_dim;
        if !latents.len().is_multiple_of(latent_dim) {
            return Err(ProviderError::inference_failed(
                "pocket-tts latent buffer has a partial frame",
            ));
        }
        let frames = latents.len() / latent_dim;
        let mut states = init_states(&self.meta.mimi_state_manifest);
        let mut pcm = Vec::with_capacity(frames * self.meta.samples_per_frame);
        for group in latents.chunks(15 * latent_dim) {
            let group_frames = group.len() / latent_dim;
            let mut inputs: Vec<(String, Value)> = vec![(
                "latent".to_string(),
                Tensor::from_array(([1, group_frames, latent_dim], group.to_vec()))
                    .map_err(|e| {
                        ProviderError::inference_failed(format!("pocket-tts decode input: {e}"))
                    })?
                    .into(),
            )];
            push_states(&mut inputs, &self.meta.mimi_state_manifest, &states)?;
            let outputs = self.decoder.run(inputs).map_err(|e| {
                ProviderError::inference_failed(format!("pocket-tts decode failed: {e}"))
            })?;
            let frame = take_f32(&outputs, "audio_frame")?;
            extract_states(&self.meta.mimi_state_manifest, &mut states, &outputs)?;
            drop(outputs);
            pcm.extend_from_slice(&frame);
        }
        Ok(pcm)
    }
}

/// Per-chunk generation options (keeps the AR-loop signature narrow).
#[cfg(feature = "pocket-tts")]
struct ChunkOptions<'a> {
    frames_after_eos: usize,
    temperature: f32,
    lsd_steps: usize,
    control: Option<&'a crate::tts::control::InferenceControl>,
}

/// Sentence-boundary indices mirroring the upstream splitter.
#[cfg(feature = "pocket-tts")]
fn boundary_indices(tokens: &[i64], boundary: &[i64]) -> Vec<usize> {
    let mut indices = vec![0];
    let mut previous_was_boundary = false;
    for (index, token) in tokens.iter().enumerate() {
        if boundary.contains(token) {
            previous_was_boundary = true;
        } else {
            if previous_was_boundary {
                indices.push(index);
            }
            previous_was_boundary = false;
        }
    }
    indices.push(tokens.len());
    indices
}

struct PocketState {
    loaded_model: Option<String>,
    load_error: Option<String>,
    ensuring: bool,
    download_progress: f32,
    downloading: bool,
}

/// Native Pocket TTS ONNX provider (audio-prompt voice cloning).
pub struct PocketTtsProvider {
    config: Mutex<PocketTtsConfig>,
    pending_config: Mutex<Option<PocketTtsConfig>>,
    lifecycle: Mutex<()>,
    invalidated_reference: std::sync::atomic::AtomicBool,
    state: Mutex<PocketState>,
    /// Resident native model. Synthesis runs as one atomic unit under
    /// this mutex: request A's voice state can never be replaced during
    /// request B's synthesis.
    #[cfg(feature = "pocket-tts")]
    model: Mutex<Option<PocketNative>>,
    /// Cached voice embeddings (frames x 1024 f32 LE with u64 frame
    /// count prefix), keyed by
    /// `(provider, revision, reference hash, transcript, audio_prompt)`.
    /// KV states are rebuilt per request from these, so concurrent voices
    /// stay isolated and memory stays bounded by the LRU.
    embeddings: Mutex<super::worker::ConditioningCache>,
    /// Last synthesized (voice, conditioning key) for fast-path reuse.
    last_voice: Mutex<Option<(String, String)>>,
}

impl PocketTtsProvider {
    pub fn new(config: PocketTtsConfig) -> Arc<Self> {
        Arc::new(Self {
            config: Mutex::new(config.clone()),
            pending_config: Mutex::new(None),
            lifecycle: Mutex::new(()),
            invalidated_reference: std::sync::atomic::AtomicBool::new(false),
            state: Mutex::new(PocketState {
                loaded_model: None,
                load_error: None,
                ensuring: false,
                download_progress: 0.0,
                downloading: false,
            }),
            #[cfg(feature = "pocket-tts")]
            model: Mutex::new(None),
            embeddings: Mutex::new(super::worker::ConditioningCache::new(config.cache_capacity)),
            last_voice: Mutex::new(None),
        })
    }

    pub fn reconfigure(&self, config: PocketTtsConfig) {
        let Ok(_lifecycle) = self.lifecycle.try_lock() else {
            if let Ok(mut pending) = self.pending_config.lock() {
                *pending = Some(config);
            }
            return;
        };
        if let Ok(mut pending) = self.pending_config.lock() {
            *pending = None;
        }
        let old = self.config_snapshot();
        let reload = old.enabled != config.enabled
            || old.model_dir != config.model_dir
            || old.model_id != config.model_id
            || old.revision != config.revision
            || old.device != config.device
            || old.cpu_threads != config.cpu_threads;
        if !reload {
            if let Ok(mut current) = self.config.lock() {
                *current = config;
            }
            return;
        }
        if let Ok(mut current) = self.config.lock() {
            *current = config;
        }
        #[cfg(feature = "pocket-tts")]
        if let Ok(mut model) = self.model.lock() {
            *model = None;
            if let Ok(mut last) = self.last_voice.lock() {
                *last = None;
            }
        }
        if let Ok(mut cache) = self.embeddings.lock() {
            *cache = super::worker::ConditioningCache::new(self.config_snapshot().cache_capacity);
        }
        if let Ok(mut state) = self.state.lock() {
            state.loaded_model = None;
            state.load_error = None;
        }
    }

    fn model_change_pending(&self) -> bool {
        let current = self.config_snapshot();
        self.pending_config
            .lock()
            .map(|p| {
                p.as_ref()
                    .is_some_and(|next| !Self::same_model(&current, next))
            })
            .unwrap_or(true)
    }

    pub fn reload_pending(&self) -> bool {
        self.pending_config
            .lock()
            .map(|p| p.is_some())
            .unwrap_or(true)
    }

    fn same_model(a: &PocketTtsConfig, b: &PocketTtsConfig) -> bool {
        a.enabled == b.enabled
            && a.model_dir == b.model_dir
            && a.model_id == b.model_id
            && a.revision == b.revision
            && a.device == b.device
            && a.cpu_threads == b.cpu_threads
    }

    fn config_snapshot(&self) -> PocketTtsConfig {
        self.config.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// Adopt `next` only when it differs (reconfigure drops the loaded
    /// model, so spurious resets would interrupt serving providers).
    pub fn sync_config(&self, next: PocketTtsConfig) {
        let changed = self
            .config
            .lock()
            .map(|current| *current != next)
            .unwrap_or(true);
        if changed || self.reload_pending() {
            self.reconfigure(next);
        }
    }

    fn check_model(model: &str) -> Result<(), ProviderError> {
        if POCKET_TTS_MODELS.contains(&model) {
            Ok(())
        } else {
            Err(ProviderError::unsupported_model(
                POCKET_TTS_PROVIDER_ID,
                model,
                format!(
                    "only validated Pocket TTS bundles are supported; expected one of: {}",
                    POCKET_TTS_MODELS.join(", ")
                ),
            ))
        }
    }

    fn check_language(language: &str) -> Result<(), ProviderError> {
        if POCKET_TTS_LANGUAGES.contains(&language) {
            Ok(())
        } else {
            // Reject languages without an exact verified bundle instead
            // of silently synthesizing another language.
            Err(ProviderError::unsupported_language(
                POCKET_TTS_PROVIDER_ID,
                language,
                POCKET_TTS_LANGUAGES,
            ))
        }
    }

    pub fn check_model_language(model: &str, language: &str) -> Result<(), ProviderError> {
        Self::check_model(model)?;
        Self::check_language(language)?;
        let expected = pocket_model_language(model).expect("model validated above");
        if language != expected {
            return Err(ProviderError::unsupported_language(
                POCKET_TTS_PROVIDER_ID,
                language,
                &[expected],
            ));
        }
        Ok(())
    }

    /// Validate a downloaded bundle directory (all eight pinned members).
    pub fn validate_bundle(model_dir: &Path, model_id: &str) -> Result<(), ProviderError> {
        let subdir = bundle_subdir(model_id).ok_or_else(|| {
            ProviderError::unsupported_model(
                POCKET_TTS_PROVIDER_ID,
                model_id,
                format!("expected one of: {}", POCKET_TTS_MODELS.join(", ")),
            )
        })?;
        let bundle = model_dir.join("onnx").join(subdir);
        let mut missing = Vec::new();
        for file in [
            "bundle.json",
            "tokenizer.model",
            "bos_before_voice.npy",
            "text_conditioner.onnx",
            "mimi_encoder.onnx",
            "flow_lm_main_int8.onnx",
            "flow_lm_flow_int8.onnx",
            "mimi_decoder_int8.onnx",
        ] {
            if !bundle.join(file).is_file() {
                missing.push(file.to_string());
            }
        }
        if !missing.is_empty() {
            return Err(ProviderError::model_missing(format!(
                "pocket-tts ONNX bundle incomplete in '{}': missing {} \
                 (enable [providers.pocket_tts] to download the pinned bundle)",
                bundle.display(),
                missing.join(", ")
            )));
        }
        Ok(())
    }

    pub fn validate_reference(pcm: &[f32], sample_rate: u32) -> Result<(), ProviderError> {
        if pcm.iter().any(|s| !s.is_finite()) {
            return Err(ProviderError::invalid_voice(
                "reference samples must be finite",
            ));
        }
        if pcm.is_empty() {
            return Err(ProviderError::invalid_voice(
                "reference audio is empty; provide a 3-10 s authorized recording",
            ));
        }
        if !(8000..=192000).contains(&sample_rate) {
            return Err(ProviderError::invalid_voice(
                "reference sample rate must be 8000..=192000 Hz",
            ));
        }
        let secs = pcm.len() as f64 / f64::from(sample_rate);
        if secs < POCKET_REF_MIN_SECS {
            return Err(ProviderError::invalid_voice(format!(
                "reference audio is too short ({secs:.1}s); provide 3-10 s (minimum {:.0}s)",
                POCKET_REF_MIN_SECS
            )));
        }
        if secs > POCKET_REF_MAX_SECS {
            return Err(ProviderError::invalid_voice(format!(
                "reference audio is too long ({secs:.1}s); keep it under {:.0}s",
                POCKET_REF_MAX_SECS
            )));
        }
        Ok(())
    }

    fn reference_hash(pcm: &[f32], sample_rate: u32) -> String {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(sample_rate.to_le_bytes());
        for sample in pcm {
            hasher.update(sample.to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }

    fn conditioning_key(
        &self,
        model_id: &str,
        reference_hash: &str,
        transcript: Option<&str>,
    ) -> String {
        super::worker::ConditioningCache::cache_key(
            POCKET_TTS_PROVIDER_ID,
            &format!(
                "{}|{}|{}",
                self.config_snapshot().revision,
                model_id,
                crate::tts::audio::REFERENCE_RESAMPLER_VERSION
            ),
            reference_hash,
            transcript,
            "audio_prompt",
        )
    }

    fn load_native(&self, model_id: &str) -> Result<(), ProviderError> {
        let config = self.config_snapshot();
        self.load_expected(model_id, &config)
    }

    fn load_expected(
        &self,
        model_id: &str,
        expected: &PocketTtsConfig,
    ) -> Result<(), ProviderError> {
        let _lifecycle = self
            .lifecycle
            .lock()
            .map_err(|_| ProviderError::inference_failed("pocket-tts lifecycle lock poisoned"))?;
        if self.model_change_pending() || !Self::same_model(&self.config_snapshot(), expected) {
            return Err(ProviderError::loading("obsolete pocket-tts load discarded"));
        }
        if let Some(loaded) = self
            .state
            .lock()
            .map_err(|_| ProviderError::inference_failed("pocket-tts state lock poisoned"))?
            .loaded_model
            .as_deref()
        {
            if loaded == model_id {
                return Ok(());
            }
            return Err(ProviderError::loading(format!(
                "pocket-tts has {loaded} resident; configure model_id={model_id} before loading"
            )));
        }
        Self::check_model(model_id)?;
        pocket_manifest(model_id)?;
        let config = self.config_snapshot();
        if !matches!(
            config.device,
            DevicePreference::Cpu | DevicePreference::Auto
        ) {
            return Err(ProviderError::device_unavailable(
                "provider 'pocket-tts' supports CPU only ('cpu'/'auto'); \
                 GPU execution providers are unsupported",
            ));
        }
        Self::validate_bundle(&config.model_dir, model_id)?;
        #[cfg(feature = "pocket-tts")]
        {
            super::assets::verify_local_files(&config.model_dir, &pocket_manifest(model_id)?)?;
            let _load_resource = super::worker::model_load_guard();
            let native = PocketNative::open(&config.model_dir, model_id, config.cpu_threads)?;
            if self.model_change_pending() {
                return Err(ProviderError::loading("obsolete pocket-tts load discarded"));
            }
            if let Ok(mut slot) = self.model.lock() {
                *slot = Some(native);
                if let Ok(mut last) = self.last_voice.lock() {
                    *last = None;
                }
            }
        }
        #[cfg(feature = "pocket-tts")]
        {
            if let Ok(mut state) = self.state.lock() {
                state.loaded_model = Some(model_id.to_string());
                state.load_error = None;
            }
            Ok(())
        }
        #[cfg(not(feature = "pocket-tts"))]
        {
            let _ = (config, model_id);
            Err(ProviderError::model_missing(
                "pocket-tts bundle validated but this build omits the native ONNX backend; \
                 rebuild with `--features pocket-tts` (CPU) for voice cloning",
            ))
        }
    }

    /// Start download + load in the background (deduplicated). Returns
    /// immediately; progress surfaces through [`TtsProvider::status`].
    pub fn ensure_background(self: &Arc<Self>, hf_token: Option<String>) {
        {
            let mut state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            if state.ensuring {
                return;
            }
            #[cfg(feature = "pocket-tts")]
            if state.loaded_model.is_some() {
                return;
            }
            state.ensuring = true;
            state.download_progress = 0.0;
            state.downloading = false;
        }
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let retry_token = hf_token.clone();
            let result = this.ensure_async(hf_token).await;
            if let Ok(mut state) = this.state.lock() {
                state.ensuring = false;
                state.downloading = false;
                match &result {
                    Ok(()) => state.load_error = None,
                    Err(error) if error.message().contains("obsolete") => state.load_error = None,
                    Err(error) => state.load_error = Some(error.message().to_string()),
                }
            }
            if result
                .as_ref()
                .is_err_and(|error| error.message().contains("obsolete"))
            {
                this.kick_ensure_if_needed(retry_token);
            }
            if let Err(error) = &result {
                tracing::error!("pocket-tts background prepare failed: {error}");
            }
        });
    }

    async fn ensure_async(self: &Arc<Self>, hf_token: Option<String>) -> Result<(), ProviderError> {
        if !cfg!(feature = "pocket-tts") {
            return Err(ProviderError::model_missing(
                "native backend is not compiled; rebuild with --features pocket-tts",
            ));
        }
        let config = self.config_snapshot();
        if !config.enabled {
            return Err(ProviderError::model_missing(
                "pocket-tts provider is disabled ([providers.pocket_tts] enabled=false)",
            ));
        }
        let manifest = pocket_manifest(&config.model_id)?;
        let progress_self = Arc::clone(self);
        let download_self = Arc::clone(self);
        super::assets::ensure_pinned_files(
            "pocket-tts",
            &config.model_dir,
            &manifest,
            hf_token.as_deref(),
            move |p| {
                if let Ok(mut state) = progress_self.state.lock() {
                    state.download_progress = p;
                }
            },
            move || {
                if let Ok(mut state) = download_self.state.lock() {
                    state.downloading = true;
                }
            },
            || false,
        )
        .await?;
        let loader = Arc::clone(self);
        let model_id = config.model_id.clone();
        tokio::task::spawn_blocking(move || loader.load_expected(&model_id, &config))
            .await
            .map_err(|e| {
                ProviderError::inference_failed(format!("pocket-tts load task failed: {e}"))
            })?
    }

    #[cfg(feature = "pocket-tts")]
    fn is_resident(&self) -> bool {
        self.state
            .lock()
            .map(|s| s.loaded_model.is_some())
            .unwrap_or(false)
            && self.model.lock().map(|m| m.is_some()).unwrap_or(false)
    }

    /// Trigger a background ensure when the provider is enabled but has no
    /// loaded model and no error yet (hot-reload path). Never blocks.
    pub fn kick_ensure_if_needed(self: &Arc<Self>, hf_token: Option<String>) {
        if self.reload_pending() {
            return;
        }
        #[cfg(feature = "pocket-tts")]
        let resident = self.is_resident();
        #[cfg(not(feature = "pocket-tts"))]
        let resident = self
            .state
            .lock()
            .map(|s| s.loaded_model.is_some())
            .unwrap_or(false);
        let needs = {
            let config = self.config_snapshot();
            let state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            config.enabled && !resident && state.load_error.is_none() && !state.ensuring
        };
        if needs {
            self.ensure_background(hf_token);
        }
    }

    #[cfg(feature = "pocket-tts")]
    fn require_loaded(&self) -> Result<(), ProviderError> {
        let loaded = self.model.lock().ok().is_some_and(|m| m.is_some());
        if loaded {
            return Ok(());
        }
        let state = self.state.lock().ok();
        match state {
            Some(state) if state.ensuring && state.download_progress >= 1.0 => Err(
                ProviderError::loading("pocket-tts assets verified; native model is loading"),
            ),
            Some(state) if state.ensuring && state.downloading => {
                Err(ProviderError::loading(format!(
                    "pocket-tts bundle is downloading ({:.0}% complete)",
                    state.download_progress * 100.0
                )))
            }
            Some(state) if state.ensuring => Err(ProviderError::loading(format!(
                "pocket-tts bundle is verifying cache ({:.0}% complete)",
                state.download_progress * 100.0
            ))),
            Some(state) => match &state.load_error {
                Some(detail) => Err(ProviderError::inference_failed(format!(
                    "pocket-tts model unavailable: {detail}"
                ))),
                None => Err(ProviderError::loading("pocket-tts model is not loaded yet")),
            },
            _ => Err(ProviderError::loading("pocket-tts model is not loaded yet")),
        }
    }

    #[cfg(feature = "pocket-tts")]
    fn cached_embeddings(&self, key: &str) -> Option<(usize, Vec<f32>)> {
        if let Ok(mut cache) = self.embeddings.lock()
            && let Some(bytes) = cache.get(key)
        {
            return decode_embeddings(&bytes);
        }
        None
    }

    #[cfg(feature = "pocket-tts")]
    fn store_embeddings(&self, key: &str, frames: usize, embeddings: &[f32]) {
        if let Ok(mut cache) = self.embeddings.lock() {
            cache.insert(key.to_string(), encode_embeddings(frames, embeddings));
        }
    }

    /// Encode + cache voice embeddings for 24 kHz mono reference PCM.
    /// Returns `(cache key, frames, embeddings)`.
    #[cfg(feature = "pocket-tts")]
    fn voice_embeddings(
        &self,
        model_id: &str,
        reference_pcm: &[f32],
        reference_rate: u32,
        transcript: Option<&str>,
    ) -> Result<(String, usize, Vec<f32>), ProviderError> {
        Self::validate_reference(reference_pcm, reference_rate)?;
        let hash = Self::reference_hash(reference_pcm, reference_rate);
        let key = self.conditioning_key(model_id, &hash, transcript);
        if let Some((frames, embeddings)) = self.cached_embeddings(&key) {
            return Ok((key, frames, embeddings));
        }
        let (frames, embeddings) = {
            let _lifecycle = crate::tts::control::lock_cancellable(&self.lifecycle)
                .map_err(ProviderError::inference_failed)?;
            if self
                .state
                .lock()
                .map_err(|_| ProviderError::inference_failed("pocket-tts state lock poisoned"))?
                .loaded_model
                .as_deref()
                != Some(model_id)
            {
                return Err(ProviderError::loading(
                    "Pocket reference model changed during preparation; retry",
                ));
            }
            let mut model = self
                .model
                .lock()
                .map_err(|_| ProviderError::inference_failed("pocket-tts model lock poisoned"))?;
            let native = model
                .as_mut()
                .ok_or_else(|| ProviderError::loading("pocket-tts model is not loaded"))?;
            let pcm24 = crate::tts::audio::resample_reference(
                reference_pcm,
                reference_rate,
                POCKET_REFERENCE_RATE,
            );
            native.encode_voice(&pcm24)?
        };
        self.store_embeddings(&key, frames, &embeddings);
        Ok((key, frames, embeddings))
    }
}

/// Serialize cached voice embeddings: u64 frame count + f32 LE samples.
#[cfg(feature = "pocket-tts")]
fn encode_embeddings(frames: usize, embeddings: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + embeddings.len() * 4);
    out.extend_from_slice(&(frames as u64).to_le_bytes());
    for value in embeddings {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(feature = "pocket-tts")]
fn decode_embeddings(bytes: &[u8]) -> Option<(usize, Vec<f32>)> {
    if bytes.len() < 8 || !(bytes.len() - 8).is_multiple_of(4) {
        return None;
    }
    let frames = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
    let mut embeddings = Vec::with_capacity((bytes.len() - 8) / 4);
    let (chunks, _) = bytes[8..].as_chunks::<4>();
    for chunk in chunks {
        embeddings.push(f32::from_le_bytes(*chunk));
    }
    if embeddings.len() % 1024 != 0 {
        return None;
    }
    Some((frames, embeddings))
}

impl TtsProvider for PocketTtsProvider {
    fn id(&self) -> &'static str {
        POCKET_TTS_PROVIDER_ID
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            provider_id: POCKET_TTS_PROVIDER_ID.to_string(),
            models: POCKET_TTS_MODELS.iter().map(|s| s.to_string()).collect(),
            languages: POCKET_TTS_LANGUAGES.iter().map(|s| s.to_string()).collect(),
            clone_modes: vec!["audio_prompt".to_string()],
            devices: vec!["cpu".to_string(), "auto".to_string()],
            requires_reference_audio: true,
            requires_reference_transcript: false,
            voices: vec![],
            sample_rate: Some(POCKET_TTS_SAMPLE_RATE),
        }
    }

    fn status(&self) -> ProviderStatus {
        let config = self.config_snapshot();
        if !config.enabled {
            return ProviderStatus {
                provider_id: POCKET_TTS_PROVIDER_ID.to_string(),
                enabled: false,
                state: ProviderStatusState::Disabled,
                active_model: None,
                effective_device: None,
                model_revision: Some(config.revision),
                backend_diagnostics: None,
                detail: Some(
                    "provider disabled in configuration ([providers.pocket_tts] enabled=false)"
                        .to_string(),
                ),
            };
        }
        let state = self.state.lock().ok();
        let loaded = state.as_ref().and_then(|s| s.loaded_model.clone());
        let error = state.as_ref().and_then(|s| s.load_error.clone());
        let ensuring = state.as_ref().is_some_and(|s| s.ensuring);
        let downloading = state.as_ref().is_some_and(|s| s.downloading);
        let progress = state.as_ref().map(|s| s.download_progress).unwrap_or(0.0);
        drop(state);
        #[cfg(not(feature = "pocket-tts"))]
        let native = false;
        #[cfg(feature = "pocket-tts")]
        let native = true;
        let installed = Self::validate_bundle(&config.model_dir, &config.model_id).is_ok();
        let state = if loaded.is_some() {
            ProviderStatusState::Ready
        } else if error.is_some() {
            ProviderStatusState::Error
        } else if ensuring || installed {
            ProviderStatusState::Loading
        } else {
            ProviderStatusState::NotInstalled
        };
        let detail = error.or_else(|| {
            if loaded.is_some() {
                None
            } else if ensuring && progress >= 1.0 {
                Some("pocket-tts assets verified; loading native model".into())
            } else if ensuring && downloading {
                Some(format!(
                    "pocket-tts bundle downloading ({:.0}% complete)",
                    progress * 100.0
                ))
            } else if ensuring {
                Some(format!(
                    "pocket-tts bundle verifying cache ({:.0}% complete)",
                    progress * 100.0
                ))
            } else if !native {
                Some(
                    "native ONNX backend not compiled; rebuild with `--features pocket-tts`"
                        .to_string(),
                )
            } else if installed {
                Some("pocket-tts bundle present; loading on first use".to_string())
            } else {
                Some(format!(
                    "pocket-tts ONNX bundle not installed in '{}' (enable [providers.pocket_tts] to download)",
                    config.model_dir.display()
                ))
            }
        });
        ProviderStatus {
            provider_id: POCKET_TTS_PROVIDER_ID.to_string(),
            enabled: true,
            state,
            active_model: loaded.or(Some(config.model_id)),
            effective_device: Some(EffectiveDevice::Cpu.as_str().to_string()),
            model_revision: Some(config.revision),
            backend_diagnostics: None,
            detail,
        }
    }

    fn load(&self, model_id: &str) -> Result<(), ProviderError> {
        Self::check_model(model_id)?;
        pocket_manifest(model_id)?;
        let config = self.config_snapshot();
        if !matches!(
            config.device,
            DevicePreference::Cpu | DevicePreference::Auto
        ) {
            return Err(ProviderError::device_unavailable(
                "provider 'pocket-tts' supports CPU only ('cpu'/'auto'); \
                 GPU execution providers are unsupported",
            ));
        }
        Self::validate_bundle(&config.model_dir, model_id)?;
        self.load_native(model_id)
    }

    fn unload(&self) {
        let Ok(_lifecycle) = self.lifecycle.lock() else {
            return;
        };
        #[cfg(feature = "pocket-tts")]
        if let Ok(mut model) = self.model.lock() {
            *model = None;
            if let Ok(mut last) = self.last_voice.lock() {
                *last = None;
            }
        }
        if let Ok(mut state) = self.state.lock() {
            state.loaded_model = None;
        }
    }

    fn invalidate_voice(&self, voice: &str) {
        self.invalidated_reference
            .store(true, std::sync::atomic::Ordering::Release);
        if let Ok(mut last) = self.last_voice.lock()
            && last.as_ref().is_some_and(|(id, _)| id == voice)
            && let Some((_, key)) = last.take()
            && let Ok(mut cache) = self.embeddings.lock()
        {
            cache.remove(&key);
        }
    }

    fn prepare_reference(
        &self,
        model_id: &str,
        reference_pcm: &[f32],
        reference_rate: u32,
        transcript: Option<&str>,
        mode: CloneMode,
    ) -> Result<String, ProviderError> {
        Self::check_model(model_id)?;
        if !matches!(mode, CloneMode::AudioPrompt) {
            return Err(ProviderError::invalid_voice(format!(
                "pocket-tts uses audio-prompt conditioning ('{}'), not '{}'",
                CloneMode::AudioPrompt.as_str(),
                mode.as_str()
            )));
        }
        Self::validate_reference(reference_pcm, reference_rate)?;
        let hash = Self::reference_hash(reference_pcm, reference_rate);
        let key = self.conditioning_key(model_id, &hash, transcript);
        // Eagerly warm cached embeddings when a model is loaded; otherwise
        // the key is derived now and materializes at synthesis.
        #[cfg(feature = "pocket-tts")]
        if self.is_resident()
            && self
                .state
                .lock()
                .map(|s| s.loaded_model.as_deref() == Some(model_id))
                .unwrap_or(false)
            && let Err(error) =
                self.voice_embeddings(model_id, reference_pcm, reference_rate, transcript)
        {
            return Err(error);
        }
        Ok(key)
    }

    fn synthesize(&self, request: &ProviderRequest) -> Result<PcmAudio, ProviderError> {
        Self::check_model_language(&request.model, &request.language)?;
        if !matches!(
            request.device,
            DevicePreference::Cpu | DevicePreference::Auto
        ) {
            return Err(ProviderError::device_unavailable(
                "provider 'pocket-tts' supports CPU only ('cpu'/'auto'); \
                 GPU execution providers are unsupported",
            ));
        }
        match request.clone_mode {
            None | Some(CloneMode::AudioPrompt) => {}
            Some(mode) => {
                return Err(ProviderError::invalid_voice(format!(
                    "pocket-tts uses audio-prompt conditioning, not '{}' \
                     (transcripts are accepted but unused)",
                    mode.as_str()
                )));
            }
        }
        if request.text.trim().is_empty() {
            return Err(ProviderError::invalid_voice("text cannot be empty"));
        }
        // Reference audio travels with the request: concurrent voices can
        // never observe each other's data through a shared mutable slot.
        // Transcripts are accepted but unused (audio-prompt conditioning).
        let reference = match request.reference_audio.as_ref() {
            Some(reference) => {
                Self::validate_reference(&reference.pcm, reference.sample_rate)?;
                reference
            }
            None => {
                return Err(ProviderError::invalid_voice(
                    "pocket-tts cloning requires reference audio; synthesize through \
                     an authorized voice profile (POST /api/voices/clones)",
                ));
            }
        };
        let config = self.config_snapshot();
        if !config.enabled {
            return Err(ProviderError::model_missing(
                "pocket-tts provider is disabled ([providers.pocket_tts] enabled=false)",
            ));
        }
        if let Err(missing) = Self::validate_bundle(&config.model_dir, &request.model) {
            if let Ok(mut state) = self.state.lock() {
                state.load_error = Some(missing.message().to_string());
            }
            return Err(missing);
        }
        #[cfg(not(feature = "pocket-tts"))]
        {
            let _ = (reference, config);
            Err(ProviderError::model_missing(
                "pocket-tts bundle validated but this build omits the native ONNX backend; \
                 rebuild with `--features pocket-tts` (CPU) for voice cloning",
            ))
        }
        #[cfg(feature = "pocket-tts")]
        {
            let _lifecycle = crate::tts::control::lock_cancellable(&self.lifecycle)
                .map_err(ProviderError::inference_failed)?;
            self.require_loaded()?;
            if let Some(control) = crate::tts::control::current() {
                control.set_context("pocket-tts", &request.model, &request.voice, "cpu");
            }
            if self.reload_pending() {
                return Err(ProviderError::loading(
                    "model reconfiguration pending; retry after the provider is ready",
                ));
            }
            let config = self.config_snapshot();
            if !config.enabled {
                return Err(ProviderError::model_missing("pocket-tts provider disabled"));
            }
            // One atomic unit under a single acquisition: voice embeddings
            // are resolved (or encoded) and every chunk is generated before
            // any other request can touch the model.
            let mut model = self
                .model
                .lock()
                .map_err(|_| ProviderError::inference_failed("pocket-tts model lock poisoned"))?;
            let loaded_matches = self
                .state
                .lock()
                .map_err(|_| ProviderError::inference_failed("pocket-tts state lock poisoned"))?
                .loaded_model
                .as_deref()
                == Some(request.model.as_str());
            if !loaded_matches {
                return Err(ProviderError::loading(format!(
                    "pocket-tts requested {}; configure model_id and load that bundle first",
                    request.model
                )));
            }
            let native = model
                .as_mut()
                .ok_or_else(|| ProviderError::loading("pocket-tts model is not loaded"))?;
            let hash = Self::reference_hash(&reference.pcm, reference.sample_rate);
            let key = self.conditioning_key(&request.model, &hash, None);
            let (frames, embeddings) = match self.cached_embeddings(&key) {
                Some(hit) => hit,
                None => {
                    let pcm24 = crate::tts::audio::resample_reference(
                        &reference.pcm,
                        reference.sample_rate,
                        POCKET_REFERENCE_RATE,
                    );
                    let (frames, embeddings) = native.encode_voice(&pcm24)?;
                    self.store_embeddings(&key, frames, &embeddings);
                    (frames, embeddings)
                }
            };
            let mut last = self.last_voice.lock().map_err(|_| {
                ProviderError::inference_failed("pocket-tts voice cache lock poisoned")
            })?;
            if self
                .invalidated_reference
                .swap(false, std::sync::atomic::Ordering::AcqRel)
            {
                *last = None;
            }
            *last = Some((request.voice.clone(), key));
            let control = crate::tts::control::current();
            // Voice-state prefill from this request's embeddings only.
            let base_state = native.prefill_voice(frames, &embeddings)?;
            let chunks = native.split_sentences(&request.text)?;
            if chunks.is_empty() {
                return Err(ProviderError::invalid_voice("text cannot be empty"));
            }
            let mut audio: Vec<f32> = Vec::new();
            // One request-local random stream, continued across sentences, like
            // the exporter's np.random stream. Never restart each chunk at seed.
            let seed = request.seed.unwrap_or_else(rand::random);
            if let Some(control) = &control {
                control.set_seed(seed);
            }
            let mut rng = StdRng::seed_from_u64(seed);
            for chunk in &chunks {
                if control.as_ref().is_some_and(|c| c.stopped()) {
                    return Err(ProviderError::inference_failed(
                        "inference cancelled or deadline exceeded",
                    ));
                }
                let (prepared, guess) = prepare_text_prompt(
                    chunk,
                    native.meta.remove_semicolons,
                    native.meta.pad_with_spaces_for_short_inputs,
                )?;
                let token_ids = native.tokenizer.encode(&prepared);
                if token_ids.is_empty() {
                    continue;
                }
                let frames_after_eos = native
                    .meta
                    .model_recommended_frames_after_eos
                    .unwrap_or(guess + 2);
                let chunk_options = ChunkOptions {
                    frames_after_eos,
                    temperature: config.temperature,
                    lsd_steps: config.lsd_steps.max(1),
                    control: control.as_deref(),
                };
                let latents =
                    native.run_chunk(&base_state, &token_ids, &chunk_options, &mut rng)?;
                if control.as_ref().is_some_and(|c| c.stopped()) {
                    return Err(ProviderError::inference_failed(
                        "inference cancelled or deadline exceeded",
                    ));
                }
                let pcm = native.decode_latents(&latents)?;
                audio.extend(pcm);
                if audio.len() > POCKET_MAX_SAMPLES {
                    return Err(ProviderError::inference_failed(
                        "pocket-tts output exceeds ten-minute bound",
                    ));
                }
            }
            drop(last);
            drop(model);
            if audio.is_empty()
                || audio.iter().any(|s| !s.is_finite())
                || !audio.iter().any(|s| s.abs() > 1e-4)
            {
                return Err(ProviderError::inference_failed(
                    "pocket-tts synthesis produced no audio",
                ));
            }
            Ok(PcmAudio {
                samples: audio,
                sample_rate: POCKET_TTS_SAMPLE_RATE,
                channels: 1,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::providers::types::ReferenceAudio;

    fn provider() -> Arc<PocketTtsProvider> {
        PocketTtsProvider::new(PocketTtsConfig {
            enabled: true,
            ..PocketTtsConfig::default()
        })
    }

    fn with_reference(mut request: ProviderRequest) -> ProviderRequest {
        request.reference_audio = Some(ReferenceAudio {
            pcm: vec![0.1; 48000],
            sample_rate: 24000,
        });
        request
    }

    #[test]
    #[cfg(feature = "pocket-tts")]
    fn push_states_rebuilds_inputs_without_changing_state() {
        let manifest = vec![
            StateEntry {
                input_name: "past_key_0".to_string(),
                output_name: "next_key_0".to_string(),
                dtype: StateDtype::F32,
                fill: StateFill::Zeros,
                shape: vec![1, 2, 2],
            },
            StateEntry {
                input_name: "past_mask".to_string(),
                output_name: "next_mask".to_string(),
                dtype: StateDtype::I64,
                fill: StateFill::Ones,
                shape: vec![1, 4],
            },
            StateEntry {
                input_name: "past_flag".to_string(),
                output_name: "next_flag".to_string(),
                dtype: StateDtype::Bool,
                fill: StateFill::Zeros,
                shape: vec![1, 1],
            },
        ];
        let states = vec![
            StateTensor::F32(vec![1.0, 2.0, 3.0, 4.0]),
            StateTensor::I64(vec![1, 1, 1, 1]),
            StateTensor::Bool(vec![true]),
        ];
        let mut inputs = vec![(
            "sequence".to_string(),
            Tensor::from_array(([1, 0, 2], Vec::<f32>::new()))
                .unwrap()
                .into(),
        )];
        push_states(&mut inputs, &manifest, &states).unwrap();
        assert_eq!(inputs.len(), 4);
        assert_eq!(inputs[1].0, "past_key_0");
        assert_eq!(inputs[2].0, "past_mask");
        assert_eq!(inputs[3].0, "past_flag");
        // Values round-trip exactly and the source states are untouched
        // (the tensor owns a copy; the state persists for the next frame).
        let (_, f32_values) = inputs[1].1.try_extract_tensor::<f32>().unwrap();
        assert_eq!(f32_values, &[1.0, 2.0, 3.0, 4.0]);
        let (_, i64_values) = inputs[2].1.try_extract_tensor::<i64>().unwrap();
        assert_eq!(i64_values, &[1, 1, 1, 1]);
        let (_, bool_values) = inputs[3].1.try_extract_tensor::<bool>().unwrap();
        assert_eq!(bool_values, &[true]);
        assert!(matches!(&states[0], StateTensor::F32(v) if v == &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    #[cfg(feature = "pocket-tts")]
    fn latents_reservation_is_bounded_and_exact_for_ordinary_shapes() {
        assert_eq!(latents_capacity(100, 128), 12_800);
        assert_eq!(latents_capacity(0, 128), 0);
        // Degenerate geometry can never turn the reservation into a huge
        // allocation: it saturates and caps at the output sample bound.
        assert_eq!(latents_capacity(usize::MAX, usize::MAX), POCKET_MAX_SAMPLES);
        assert_eq!(latents_capacity(1_000_000, 1_000_000), POCKET_MAX_SAMPLES);
    }

    #[test]
    fn manifest_pins_all_validated_bundles() {
        for model in POCKET_TTS_MODELS {
            let manifest = pocket_manifest(model).unwrap();
            assert_eq!(manifest.len(), 8, "{model}");
            for file in &manifest {
                assert!(file.url.contains(POCKET_TTS_UPSTREAM_REVISION));
                assert!(file.url.contains(POCKET_TTS_REPO));
            }
        }
    }

    #[test]
    fn unvalidated_variants_fail_explicitly() {
        for model in [
            "pocket-tts-fr",
            "pocket-tts-nl",
            "pocket-tts-zh",
            "pocket-tts-ja",
            "pocket-tts-german_24l",
            "gpt-sovits",
        ] {
            let error = pocket_manifest(model).unwrap_err();
            assert_eq!(error.code(), "unsupported_model", "{model}");
        }
    }

    #[test]
    fn unverified_languages_are_explicitly_unsupported() {
        let p = provider();
        for lang in ["nl", "zh", "ja", "por"] {
            let request = ProviderRequest::new(POCKET_TTS_MODELS[0], "profile", "hello", lang);
            let error = p.synthesize(&with_reference(request)).unwrap_err();
            assert_eq!(error.code(), "unsupported_language", "{lang}");
            assert_eq!(error.http_status(), 400);
        }
    }

    #[test]
    fn english_and_spanish_pass_language_validation() {
        let p = provider();
        for lang in ["en", "es"] {
            let request = ProviderRequest::new(
                if lang == "en" {
                    "pocket-tts-en"
                } else {
                    "pocket-tts-es"
                },
                "profile",
                "hello",
                lang,
            );
            let error = p.synthesize(&with_reference(request)).unwrap_err();
            // Bundle absent in CI: explicit missing/loading, never fallback.
            assert!(
                matches!(error.code(), "model_missing" | "loading"),
                "{lang}: {}",
                error
            );
        }
    }

    #[test]
    fn model_language_pairs_are_isolated() {
        for (model, language) in [("pocket-tts-en", "es"), ("pocket-tts-es", "en")] {
            let request = ProviderRequest::new(model, "profile", "hello", language);
            let error = provider().synthesize(&with_reference(request)).unwrap_err();
            assert_eq!(error.code(), "unsupported_language");
        }
        assert!(PocketTtsProvider::check_model_language("pocket-tts-en", "en").is_ok());
        assert!(PocketTtsProvider::check_model_language("pocket-tts-es", "es").is_ok());
    }

    #[test]
    fn loading_a_different_resident_bundle_fails_explicitly() {
        let p = provider();
        p.state.lock().unwrap().loaded_model = Some("pocket-tts-en".to_string());
        let config = p.config_snapshot();
        assert!(p.load_expected("pocket-tts-en", &config).is_ok());
        let error = p.load_expected("pocket-tts-es", &config).unwrap_err();
        assert_eq!(error.code(), "loading");
        assert!(error.message().contains("pocket-tts-en"));
        assert!(error.message().contains("pocket-tts-es"));
    }

    #[test]
    fn cloning_without_request_reference_audio_is_rejected() {
        let p = provider();
        let request = ProviderRequest::new(POCKET_TTS_MODELS[0], "profile", "hello", "en");
        let error = p.synthesize(&request).unwrap_err();
        assert_eq!(error.code(), "invalid_voice");
        assert!(error.message().contains("reference audio"));
    }

    #[test]
    fn transcript_is_optional_for_audio_prompt() {
        let p = provider();
        // No transcript at all: must NOT fail with a transcript error.
        let request = ProviderRequest::new(POCKET_TTS_MODELS[0], "profile", "hello", "en");
        let error = p.synthesize(&with_reference(request)).unwrap_err();
        assert!(
            !error.message().contains("transcript"),
            "unexpected: {}",
            error
        );
    }

    #[test]
    fn icl_and_xvector_modes_are_rejected_explicitly() {
        let p = provider();
        for mode in [CloneMode::Icl, CloneMode::XVector] {
            let mut request = ProviderRequest::new(POCKET_TTS_MODELS[0], "profile", "hello", "en");
            request.clone_mode = Some(mode);
            let error = p.synthesize(&with_reference(request)).unwrap_err();
            assert_eq!(error.code(), "invalid_voice", "{mode:?}");
            assert!(error.message().contains("audio-prompt"));
        }
    }

    #[test]
    fn gpu_requests_are_explicitly_unsupported() {
        let p = provider();
        let mut request = ProviderRequest::new(POCKET_TTS_MODELS[0], "profile", "hello", "en");
        request.device = DevicePreference::Cuda;
        let error = p.synthesize(&with_reference(request)).unwrap_err();
        assert_eq!(error.code(), "device_unavailable");
    }

    #[test]
    fn retired_gpt_model_ids_are_never_rerouted() {
        let p = provider();
        let request = ProviderRequest::new("gpt-sovits-v2pro-onnx", "profile", "hello", "en");
        let error = p.synthesize(&with_reference(request)).unwrap_err();
        assert_eq!(error.code(), "unsupported_model");
        assert!(!error.message().contains("supertonic"));
    }

    #[test]
    fn disabled_provider_reports_disabled_status() {
        let p = PocketTtsProvider::new(PocketTtsConfig::default());
        let status = p.status();
        assert!(!status.enabled);
        assert_eq!(status.state, ProviderStatusState::Disabled);
        assert_eq!(status.provider_id, "pocket-tts");
    }

    #[test]
    fn capabilities_advertise_verified_audio_prompt_languages() {
        let caps = provider().capabilities();
        assert_eq!(caps.provider_id, "pocket-tts");
        assert_eq!(caps.models, POCKET_TTS_MODELS);
        assert_eq!(caps.languages, POCKET_TTS_LANGUAGES);
        assert_eq!(caps.clone_modes, vec!["audio_prompt"]);
        assert!(caps.requires_reference_audio);
        assert!(!caps.requires_reference_transcript);
        assert_eq!(caps.sample_rate, Some(24_000));
    }

    #[test]
    fn malformed_bundle_names_missing_members() {
        let dir = std::env::temp_dir().join(format!("pocket-tts-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let error = PocketTtsProvider::validate_bundle(&dir, "pocket-tts-en").unwrap_err();
        assert_eq!(error.code(), "model_missing");
        assert!(error.message().contains("tokenizer.model"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reference_validation_bounds() {
        assert!(PocketTtsProvider::validate_reference(&vec![0.1; 48000], 16000).is_ok());
        assert!(PocketTtsProvider::validate_reference(&[], 16000).is_err());
        assert!(PocketTtsProvider::validate_reference(&vec![0.1; 100], 16000).is_err());
        assert!(PocketTtsProvider::validate_reference(&vec![0.1; 31 * 16000], 16000).is_err());
        assert!(PocketTtsProvider::validate_reference(&vec![f32::NAN; 48000], 16000).is_err());
        assert!(PocketTtsProvider::validate_reference(&vec![0.1; 48000], 4000).is_err());
    }

    #[test]
    fn delete_invalidates_only_the_matching_voice_entry() {
        let p = provider();
        let key_a = "pocket-tts|rev|aaa";
        let key_b = "pocket-tts|rev|bbb";
        if let Ok(mut cache) = p.embeddings.lock() {
            cache.insert(key_a.to_string(), vec![1]);
            cache.insert(key_b.to_string(), vec![2]);
        }
        if let Ok(mut last) = p.last_voice.lock() {
            *last = Some(("voice-a".to_string(), key_a.to_string()));
        }
        p.invalidate_voice("voice-a");
        if let Ok(mut cache) = p.embeddings.lock() {
            assert!(cache.get(key_a).is_none());
            assert_eq!(cache.get(key_b), Some(vec![2]));
        }
        assert!(p.last_voice.lock().map(|l| l.is_none()).unwrap_or(false));
        // Other voices and double deletion stay valid no-ops.
        p.invalidate_voice("voice-a");
        p.invalidate_voice("voice-b");
        if let Ok(mut cache) = p.embeddings.lock() {
            assert_eq!(cache.get(key_b), Some(vec![2]));
        }
    }

    #[test]
    fn reconfigure_drops_resident_state_and_keeps_disabled_default() {
        let p = provider();
        assert!(p.status().enabled);
        p.reconfigure(PocketTtsConfig::default());
        let status = p.status();
        assert!(!status.enabled);
        assert_eq!(status.state, ProviderStatusState::Disabled);
        // Identical reconfiguration is a no-op that records no failure.
        p.sync_config(PocketTtsConfig::default());
        assert!(!p.reload_pending());
    }

    #[test]
    fn concurrent_references_use_distinct_cache_keys() {
        let p = provider();
        let a = p
            .prepare_reference(
                POCKET_TTS_MODELS[0],
                &vec![0.1; 48000],
                16000,
                None,
                CloneMode::AudioPrompt,
            )
            .unwrap();
        let b = p
            .prepare_reference(
                POCKET_TTS_MODELS[0],
                &vec![0.2; 48000],
                16000,
                None,
                CloneMode::AudioPrompt,
            )
            .unwrap();
        assert_ne!(a, b, "distinct references must never share conditioning");
        assert!(a.starts_with("pocket-tts|"));
    }

    #[test]
    fn prepare_reference_rejects_non_audio_prompt_modes() {
        let p = provider();
        let error = p
            .prepare_reference(
                POCKET_TTS_MODELS[0],
                &vec![0.1; 48000],
                16000,
                Some("hello"),
                CloneMode::Icl,
            )
            .unwrap_err();
        assert_eq!(error.code(), "invalid_voice");
    }

    #[test]
    fn prompt_preparation_matches_upstream() {
        let (text, guess) = prepare_text_prompt("hello world", false, false).unwrap();
        assert_eq!(text, "Hello world.");
        assert_eq!(guess, 3);
        let (text, guess) =
            prepare_text_prompt("  this is a longer sentence here\n", false, false).unwrap();
        assert_eq!(text, "This is a longer sentence here.");
        assert_eq!(guess, 1);
        let (text, _) = prepare_text_prompt("a;b", true, false).unwrap();
        assert_eq!(text, "A,b.");
        let (text, _) = prepare_text_prompt("hi", false, true).unwrap();
        assert!(text.starts_with("        "), "{text}");
        assert!(prepare_text_prompt("   ", false, false).is_err());
    }

    #[test]
    #[cfg(feature = "pocket-tts")]
    fn embeddings_codec_round_trips() {
        let frames = 3;
        let embeddings: Vec<f32> = (0..frames * 1024).map(|v| v as f32 * 0.001).collect();
        let bytes = encode_embeddings(frames, &embeddings);
        let (back_frames, back) = decode_embeddings(&bytes).unwrap();
        assert_eq!(back_frames, frames);
        assert_eq!(back, embeddings);
        assert!(decode_embeddings(&[0u8; 7]).is_none());
        assert!(decode_embeddings(&[0u8; 12]).is_none());
    }

    #[test]
    fn npy_loader_reads_bos_vector() {
        // Minimal v1.0 float32 C-order .npy with shape (1, 1, 2).
        let mut file = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
        let header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 2), }";
        let pad = 64 - (10 + header.len()) % 64;
        let mut header = header.to_vec();
        header.extend(std::iter::repeat_n(b' ', pad));
        header.push(b'\n');
        file.extend_from_slice(&(header.len() as u16).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&1.5f32.to_le_bytes());
        file.extend_from_slice(&(-2.25f32).to_le_bytes());
        let dir = std::env::temp_dir().join(format!("pocket-npy-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v.npy");
        std::fs::write(&path, &file).unwrap();
        let (shape, values) = load_npy_f32(&path).unwrap();
        assert_eq!(shape, vec![1, 1, 2]);
        assert_eq!(values, vec![1.5, -2.25]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_unigram_model_encodes_and_decodes() {
        // Hand-built ModelProto: pieces "▁a" (id 1), "b" (id 2), 256 byte
        // pieces (ids 3..259). Verifies the protobuf walker, Viterbi
        // preference for longer pieces and single-byte fallback.
        fn piece(id_text: &[u8], score: f32, ty: u8) -> Vec<u8> {
            let mut msg = Vec::new();
            msg.push(0x0A);
            msg.push(id_text.len() as u8);
            msg.extend_from_slice(id_text);
            msg.push(0x15);
            msg.extend_from_slice(&score.to_le_bytes());
            msg.push(0x18);
            msg.push(ty);
            let mut field = Vec::new();
            field.push(0x0A);
            field.push(msg.len() as u8);
            field.extend_from_slice(&msg);
            field
        }
        let mut model = Vec::new();
        model.extend(piece("▁a".as_bytes(), 2.0, 1));
        model.extend(piece(b"b", 1.0, 1));
        for byte in 0..=255u8 {
            let name = format!("<0x{byte:02X}>");
            model.extend(piece(name.as_bytes(), 0.0, 6));
        }
        let dir = std::env::temp_dir().join(format!("pocket-tok-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.model");
        std::fs::write(&path, &model).unwrap();
        let tok = tokenizer::UnigramTokenizer::load(&path).unwrap();
        // "ab" -> "▁ab": the longer "▁a" piece (score 2.0) beats
        // single-byte matches; decode maps ▁ back to a space.
        let ids = tok.encode("ab");
        assert_eq!(ids, vec![0, 1], "{ids:?}");
        assert_eq!(tok.decode(&ids), "ab");
        // Uncovered bytes fall back to byte pieces: "▁" is E2 96 81 and
        // byte pieces sit at ids 2..258, so "\u{1}" (0x01) follows them.
        let ids = tok.encode("\u{1}");
        assert_eq!(ids, vec![2 + 0xE2, 2 + 0x96, 2 + 0x81, 2 + 0x01], "{ids:?}");
        assert_eq!(tok.decode(&ids), "\u{1}");
        for text in ["É", "sí", "¿Qué tal?", "🙂", "你好"] {
            assert_eq!(
                tok.decode(&tok.encode(text)),
                text,
                "UTF-8 fallback for {text}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_model_accepts_only_its_own_language() {
        for model in POCKET_TTS_MODELS {
            let language = pocket_model_language(model).unwrap();
            assert!(PocketTtsProvider::check_model_language(model, language).is_ok());
            for other in POCKET_TTS_LANGUAGES
                .iter()
                .filter(|other| **other != language)
            {
                let error = PocketTtsProvider::check_model_language(model, other).unwrap_err();
                assert_eq!(error.code(), "unsupported_language", "{model}: {other}");
            }
        }
    }

    #[test]
    fn conditioning_cache_is_isolated_by_model() {
        let p = provider();
        let keys: std::collections::HashSet<_> = POCKET_TTS_MODELS
            .iter()
            .map(|model| p.conditioning_key(model, "same-reference", None))
            .collect();
        assert_eq!(keys.len(), POCKET_TTS_MODELS.len());
    }

    /// Independent fixtures are generated by the exact pinned Python exporter.
    /// Temperature zero removes cross-language RNG implementation differences.
    #[test]
    #[ignore = "requires all six verified Pocket ONNX bundles"]
    #[cfg(feature = "pocket-tts")]
    fn real_bundles_match_pinned_exporter_prompts_states_and_decoder() {
        let root = std::env::var("POCKET_TTS_BUNDLE_DIR").expect("set POCKET_TTS_BUNDLE_DIR");
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmarks/tts-quality/fixtures/pocket-export-verified-ort128.json"
        )))
        .unwrap();
        assert_eq!(
            ort::info(),
            fixture["ort_build"].as_str().unwrap(),
            "export fixture/runtime build mismatch: run the documented ort/load-dynamic comparison against the same pinned Python CPU library"
        );
        let digest = |values: &[f32]| {
            use sha2::{Digest, Sha256};
            let mut hash = Sha256::new();
            for value in values {
                hash.update(value.to_le_bytes());
            }
            hex::encode(hash.finalize())
        };
        let pcm: Vec<f32> = (0..24000)
            .map(|i| ((i * 17 % 4096) as f32 - 2048.0) / 16384.0)
            .collect();
        for case in fixture["cases"].as_array().unwrap() {
            let model = case["model"].as_str().unwrap();
            let mut native = PocketNative::open(Path::new(&root), model, 4).unwrap();
            for text_case in case["cases"].as_array().unwrap() {
                let text = text_case["text"].as_str().unwrap();
                let (prepared, guess) = prepare_text_prompt(
                    text,
                    native.meta.remove_semicolons,
                    native.meta.pad_with_spaces_for_short_inputs,
                )
                .unwrap();
                assert_eq!(
                    prepared,
                    text_case["prepared"].as_str().unwrap(),
                    "{model}: preparation"
                );
                assert_eq!(guess, text_case["frames_guess"].as_u64().unwrap() as usize);
                assert_eq!(
                    serde_json::to_value(native.tokenizer.encode(&prepared)).unwrap(),
                    text_case["ids"],
                    "{model}: token ids"
                );
                assert_eq!(
                    serde_json::to_value(native.split_sentences(text).unwrap()).unwrap(),
                    text_case["chunks"],
                    "{model}: chunks"
                );
            }
            let (frames, embeddings) = native.encode_voice(&pcm).unwrap();
            assert_eq!(frames, case["voice_frames"].as_u64().unwrap() as usize);
            let base = native.prefill_voice(frames, &embeddings).unwrap();
            let (prepared, guess) = prepare_text_prompt(
                case["target"].as_str().unwrap(),
                native.meta.remove_semicolons,
                native.meta.pad_with_spaces_for_short_inputs,
            )
            .unwrap();
            let ids = native.tokenizer.encode(&prepared);
            let options = ChunkOptions {
                frames_after_eos: native
                    .meta
                    .model_recommended_frames_after_eos
                    .unwrap_or(guess + 2),
                temperature: 0.0,
                lsd_steps: 1,
                control: None,
            };
            let latents = native
                .run_chunk(&base, &ids, &options, &mut StdRng::seed_from_u64(42))
                .unwrap();
            let expected = case["latents"].as_array().unwrap();
            assert_eq!(
                latents.len(),
                expected.len(),
                "{model}: EOS/latent frame count"
            );
            let mut max_latent_error = 0.0f32;
            for (i, (value, expected)) in latents.iter().zip(expected).enumerate() {
                let expected = expected.as_f64().unwrap() as f32;
                max_latent_error = max_latent_error.max((*value - expected).abs());
                assert!(
                    (*value - expected).abs() <= 1e-3 + 1e-3 * expected.abs(),
                    "{model}: latent {i}: {value} vs {expected}"
                );
            }
            assert_eq!(
                digest(&latents),
                case["latents_sha256"].as_str().unwrap(),
                "{model}: all latent values"
            );
            let audio = native.decode_latents(&latents).unwrap();
            assert_eq!(
                audio.len(),
                case["sample_count"].as_u64().unwrap() as usize,
                "{model}: decoder length"
            );
            assert_eq!(
                digest(&audio),
                case["pcm_sha256"].as_str().unwrap(),
                "{model}: complete decoder PCM including chunk carry"
            );
            for sample in case["pcm_samples"].as_array().unwrap() {
                let index = sample[0].as_u64().unwrap() as usize;
                let expected = sample[1].as_f64().unwrap() as f32;
                assert!(
                    (audio[index] - expected).abs() <= 1e-3 + 1e-3 * expected.abs(),
                    "{model}: PCM {index}: {} vs {expected}",
                    audio[index]
                );
            }
            eprintln!(
                "{model}: exporter latent max abs error {max_latent_error}; {} frames, {} samples",
                latents.len() / 32,
                audio.len()
            );
        }
    }

    /// Uses actual ONNX execution to distinguish per-chunk random continuity,
    /// fresh request state and recovery after cancellation during inference.
    #[test]
    #[ignore = "requires authorized reference and real English ONNX bundle"]
    #[cfg(feature = "pocket-tts")]
    fn real_bundle_request_rng_and_active_cancellation_are_isolated() {
        use crate::tts::control::InferenceControl;
        use std::time::{Duration, Instant};
        let root = std::env::var("POCKET_TTS_BUNDLE_DIR").expect("set POCKET_TTS_BUNDLE_DIR");
        let path = std::env::var("POCKET_TTS_REF_WAV").expect("set POCKET_TTS_REF_WAV");
        let mut reader = hound::WavReader::open(path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        let pcm: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let pcm = crate::tts::audio::resample_reference(&pcm, spec.sample_rate, 24000);
        let mut native = PocketNative::open(Path::new(&root), "pocket-tts-en", 4).unwrap();
        let (frames, embeddings) = native.encode_voice(&pcm).unwrap();
        let base = native.prefill_voice(frames, &embeddings).unwrap();
        let ids = native.tokenizer.encode("Hello world.");
        let options = ChunkOptions {
            frames_after_eos: 5,
            temperature: 0.7,
            lsd_steps: 1,
            control: None,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let first = native.run_chunk(&base, &ids, &options, &mut rng).unwrap();
        let second_chunk = native.run_chunk(&base, &ids, &options, &mut rng).unwrap();
        assert_ne!(
            first, second_chunk,
            "identical chunks must continue the request RNG"
        );
        let first_pcm = native.decode_latents(&first).unwrap();
        let control = InferenceControl::new(Duration::from_secs(180));
        let watcher = control.clone();
        let cancellation = std::thread::spawn(move || {
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(120) {
                if watcher.snapshot().frames >= 1 {
                    watcher.cancel();
                    return true;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            watcher.cancel();
            false
        });
        let options_cancel = ChunkOptions {
            control: Some(&control),
            ..options
        };
        let result = native.run_chunk(&base, &ids, &options_cancel, &mut rng);
        assert!(
            cancellation.join().unwrap(),
            "must cancel after native frame generation"
        );
        assert!(result.unwrap_err().to_string().contains("cancelled"));
        let mut fresh_rng = StdRng::seed_from_u64(42);
        let recovered = native
            .run_chunk(&base, &ids, &options, &mut fresh_rng)
            .unwrap();
        assert_eq!(
            first, recovered,
            "fresh request must recover its deterministic latent sequence"
        );
        assert_eq!(
            first_pcm,
            native.decode_latents(&recovered).unwrap(),
            "decoder state must be fresh"
        );
    }

    /// Opt-in verification of downloaded language-local tokenizers and five
    /// native graphs. Uses Python SentencePiece fixtures, not self-derived IDs.
    #[test]
    #[ignore]
    #[cfg(feature = "pocket-tts")]
    fn real_bundle_new_languages_match_tokenizers_and_synthesize() {
        let root = std::env::var("POCKET_TTS_BUNDLE_DIR").expect("set POCKET_TTS_BUNDLE_DIR");
        let reference = std::env::var("POCKET_TTS_REF_WAV").expect("set POCKET_TTS_REF_WAV");
        let mut reader = hound::WavReader::open(reference).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        let pcm = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect::<Vec<_>>();
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmarks/pocket/fixtures/languages.json"
        )))
        .unwrap();
        for fixture in fixtures.as_array().unwrap() {
            let model = fixture["model"].as_str().unwrap();
            let language = fixture["language"].as_str().unwrap();
            let bundle = Path::new(&root)
                .join("onnx")
                .join(fixture["bundle"].as_str().unwrap());
            let files = pocket_manifest(model).unwrap();
            super::super::assets::verify_local_files(Path::new(&root), &files).unwrap();
            let tok = tokenizer::UnigramTokenizer::load(&bundle.join("tokenizer.model")).unwrap();
            for case in fixture["cases"].as_array().unwrap() {
                let text = case["text"].as_str().unwrap();
                let ids = case["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|id| id.as_i64().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(tok.encode(text), ids, "{model}: {text:?}");
            }
            let p = PocketTtsProvider::new(PocketTtsConfig {
                enabled: true,
                model_dir: PathBuf::from(&root),
                model_id: model.to_string(),
                ..PocketTtsConfig::default()
            });
            p.load(model).unwrap();
            let mut request = ProviderRequest::new(
                model,
                "synthetic-language-test",
                fixture["synthesis_text"].as_str().unwrap(),
                language,
            );
            request.reference_audio = Some(ReferenceAudio {
                pcm: pcm.clone(),
                sample_rate: spec.sample_rate,
            });
            request.seed = Some(49420);
            let start = std::time::Instant::now();
            let audio = p.synthesize(&request).unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            assert_eq!(audio.sample_rate, POCKET_TTS_SAMPLE_RATE);
            assert_eq!(audio.channels, 1);
            assert!(audio.duration_secs() > 1.0 && audio.is_audible());
            assert!(audio.samples.iter().all(|v| v.is_finite()));
            if let Ok(output) = std::env::var("POCKET_TTS_COMPARE_DIR") {
                std::fs::create_dir_all(&output).unwrap();
                let mut wav = hound::WavWriter::create(
                    Path::new(&output).join(format!("{model}.wav")),
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: audio.sample_rate,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    },
                )
                .unwrap();
                for sample in &audio.samples {
                    wav.write_sample(*sample).unwrap();
                }
                wav.finalize().unwrap();
            }
            eprintln!(
                "{}",
                serde_json::json!({"model":model,"language":language,
                "seconds":elapsed,"audio_seconds":audio.duration_secs(),"rtf":elapsed/audio.duration_secs(),
                "sentencepiece_cases":fixture["cases"].as_array().unwrap().len()})
            );
            p.unload();
        }
    }

    /// End-to-end synthesis against real bundles. Ignored by default
    /// (needs downloaded weights: set `POCKET_TTS_BUNDLE_DIR` to a model
    /// dir containing `onnx/<bundle>` plus `POCKET_TTS_REF_WAV` to an
    /// authorized reference WAV). Run explicitly with `-- --ignored`.
    /// Fails loudly on silent, empty or non-finite output.
    #[test]
    #[ignore]
    fn real_bundle_synthesizes_audible_cloned_speech() {
        let model_dir = std::env::var("POCKET_TTS_BUNDLE_DIR").unwrap_or_default();
        let ref_wav = std::env::var("POCKET_TTS_REF_WAV").unwrap_or_default();
        assert!(!model_dir.is_empty(), "set POCKET_TTS_BUNDLE_DIR");
        assert!(
            !ref_wav.is_empty(),
            "set POCKET_TTS_REF_WAV to an authorized reference"
        );
        let reader = hound::WavReader::open(&ref_wav).unwrap();
        let rate = reader.spec().sample_rate;
        let pcm: Vec<f32> = reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        for (model, lang, text) in [
            (
                "pocket-tts-en",
                "en",
                "Hello world, this is a test of voice cloning on a CPU.",
            ),
            (
                "pocket-tts-es",
                "es",
                "Hola mundo, esta es una prueba de clonación de voz.",
            ),
        ] {
            let bundle = Path::new(&model_dir).join("onnx");
            let subdir = if model == "pocket-tts-en" {
                "english_2026-04"
            } else {
                "spanish"
            };
            assert!(
                bundle.join(subdir).join("tokenizer.model").is_file(),
                "missing {subdir}/tokenizer.model in {}",
                bundle.display()
            );
            let p = PocketTtsProvider::new(PocketTtsConfig {
                enabled: true,
                model_dir: PathBuf::from(&model_dir),
                model_id: model.to_string(),
                ..PocketTtsConfig::default()
            });
            p.load(model).unwrap();
            let mut request = ProviderRequest::new(model, "e2e-voice", text, lang);
            request.reference_audio = Some(ReferenceAudio {
                pcm: pcm.clone(),
                sample_rate: rate,
            });
            request.seed = Some(0xC10C);
            let audio = p.synthesize(&request).unwrap();
            eprintln!(
                "{model}: {:.2}s audible={}",
                audio.duration_secs(),
                audio.is_audible()
            );
            assert_eq!(audio.sample_rate, POCKET_TTS_SAMPLE_RATE);
            assert_eq!(audio.channels, 1);
            assert!(audio.duration_secs() > 1.0, "{model}");
            assert!(audio.is_audible(), "{model}");
            assert!(audio.samples.iter().all(|s| s.is_finite()), "{model}");
            p.unload();
        }
        // Conditioning isolation: two authorized references must not mix.
        // Same text/seed through different voices must differ (and both be
        // audible); a shared global context would make them identical.
        let en_bundle = Path::new(&model_dir).join("onnx/english_2026-04");
        if en_bundle.join("tokenizer.model").is_file() {
            let alt_wav = std::env::var("POCKET_TTS_ALT_REF_WAV").unwrap_or_default();
            if !alt_wav.is_empty() {
                let alt_reader = hound::WavReader::open(&alt_wav).unwrap();
                let alt_rate = alt_reader.spec().sample_rate;
                let alt_pcm: Vec<f32> = alt_reader
                    .into_samples::<i16>()
                    .map(|s| s.unwrap() as f32 / 32768.0)
                    .collect();
                let p = PocketTtsProvider::new(PocketTtsConfig {
                    enabled: true,
                    model_dir: PathBuf::from(&model_dir),
                    model_id: "pocket-tts-en".to_string(),
                    ..PocketTtsConfig::default()
                });
                p.load("pocket-tts-en").unwrap();
                let synth = |wav_pcm: Vec<f32>, wav_rate: u32| {
                    let mut request =
                        ProviderRequest::new("pocket-tts-en", "iso-voice", "Hello world.", "en");
                    request.reference_audio = Some(ReferenceAudio {
                        pcm: wav_pcm,
                        sample_rate: wav_rate,
                    });
                    request.seed = Some(0xC10C);
                    p.synthesize(&request).unwrap().samples
                };
                let a = synth(pcm.clone(), rate);
                let b = synth(alt_pcm, alt_rate);
                assert!(PcmAudio::mono(a.clone(), POCKET_TTS_SAMPLE_RATE).is_audible());
                assert!(PcmAudio::mono(b.clone(), POCKET_TTS_SAMPLE_RATE).is_audible());
                assert_ne!(a, b, "different voices must not share conditioning");
                p.unload();
            }
        }
    }

    /// Real-bundle parity probe. Ignored by default (needs downloaded
    /// weights); run explicitly with `-- --ignored` after enabling
    /// `[providers.pocket_tts]` once. Fails loudly on any divergence.
    #[test]
    #[ignore]
    fn real_bundle_matches_phase_a_reference() {
        let dir = std::env::var("POCKET_TTS_BUNDLE_DIR")
            .unwrap_or_else(|_| "./models/pocket-tts".to_string());
        let bundle = Path::new(&dir).join("onnx/english_2026-04");
        // Loud failure beats silent green when weights were requested.
        assert!(
            bundle.join("tokenizer.model").is_file(),
            "missing tokenizer.model in {dir}"
        );
        let tok = tokenizer::UnigramTokenizer::load(&bundle.join("tokenizer.model")).unwrap();
        let ids = tok.encode("Hello world, this is a test of voice cloning on a CPU.");
        assert_eq!(
            ids,
            vec![
                2994, 578, 262, 285, 277, 267, 1115, 272, 1483, 331, 2101, 273, 288, 267, 444, 759,
                1917, 263
            ]
        );
        let (prepared, _) = prepare_text_prompt(
            "Hello world, this is a test of voice cloning on a CPU.",
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            prepared,
            "Hello world, this is a test of voice cloning on a CPU..".replace("..", ".")
        );
        let meta = BundleMeta::load(&bundle).unwrap();
        assert_eq!(meta.sample_rate, 24_000);
        assert_eq!(meta.latent_dim, 32);
        assert_eq!(meta.conditioning_dim, 1024);
    }
}
