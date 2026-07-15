use anyhow::Result;
use ort::session::Session;
use serde::Deserialize;
use std::collections::HashMap;

use crate::tts::{download::ModelPaths, text::TextProcessor};

// ============================================================================
// tts.json config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct TtsConfig {
    pub ae: AeConfig,
    pub ttl: TtlConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AeConfig {
    pub sample_rate: u32,
    pub base_chunk_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtlConfig {
    pub chunk_compress_factor: usize,
    pub latent_dim: usize,
}

// ============================================================================
// Voice style
// ============================================================================

#[derive(Debug, Deserialize)]
struct StyleTensor {
    data: serde_json::Value,
    dims: Vec<usize>,
}

impl StyleTensor {
    fn flatten(&self) -> Vec<f32> {
        fn collect(v: &serde_json::Value, out: &mut Vec<f32>) {
            match v {
                serde_json::Value::Array(arr) => {
                    for item in arr {
                        collect(item, out);
                    }
                }
                serde_json::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        out.push(f as f32);
                    }
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        collect(&self.data, &mut out);
        out
    }
}

#[derive(Debug, Deserialize)]
struct VoiceStyleRaw {
    style_ttl: StyleTensor,
    style_dp: StyleTensor,
}

#[derive(Debug)]
pub struct VoiceStyle {
    pub style_ttl: Vec<f32>,
    pub style_ttl_dims: (usize, usize, usize),
    pub style_dp: Vec<f32>,
    pub style_dp_dims: (usize, usize, usize),
}

// ============================================================================
// ModelHandle
// ============================================================================

pub struct ModelHandle {
    pub sessions: std::sync::Mutex<Sessions>,
    pub text_processor: TextProcessor,
    pub voice_styles: HashMap<String, VoiceStyle>,
    pub config: TtsConfig,
}

/// Holds all ONNX sessions together so they can be locked as a unit.
/// This prevents deadlocks and avoids contention between sessions during sequential inference.
pub struct Sessions {
    pub duration_predictor: Session,
    pub text_encoder: Session,
    pub vector_estimator: Session,
    pub vocoder: Session,
}

impl ModelHandle {
    pub fn sample_rate(&self) -> u32 {
        self.config.ae.sample_rate
    }
}

fn build_session(path: &std::path::Path) -> Result<Session> {
    let builder = Session::builder()?;

    #[cfg(target_os = "macos")]
    let builder = {
        use ort::execution_providers::CoreMLExecutionProvider;
        // Place .mlmodelc cache next to model file to prevent recompilation
        let cache_dir = path.with_extension("mlmodelc");
        let cache_dir_str = cache_dir.to_string_lossy().into_owned();
        builder.with_execution_providers([CoreMLExecutionProvider::default()
            .with_model_cache_dir(cache_dir_str)
            .build()])?
    };

    // NVIDIA CUDA acceleration (opt-in via `--features cuda`)
    #[cfg(feature = "cuda")]
    let builder = {
        use ort::execution_providers::CUDAExecutionProvider;
        builder.with_execution_providers([CUDAExecutionProvider::default().build()])?
    };

    // AMD ROCm acceleration (opt-in via `--features rocm`)
    #[cfg(feature = "rocm")]
    let builder = {
        use ort::execution_providers::ROCmExecutionProvider;
        builder.with_execution_providers([ROCmExecutionProvider::default().build()])?
    };

    Ok(builder.commit_from_file(path)?)
}

impl ModelHandle {
    pub fn load(paths: &ModelPaths) -> Result<Self> {
        tracing::info!("Loading ONNX sessions...");

        let sessions = std::sync::Mutex::new(Sessions {
            duration_predictor: build_session(&paths.duration_predictor)?,
            text_encoder: build_session(&paths.text_encoder)?,
            vector_estimator: build_session(&paths.vector_estimator)?,
            vocoder: build_session(&paths.vocoder)?,
        });

        let text_processor = TextProcessor::load(&paths.unicode_indexer)?;

        let config: TtsConfig = serde_json::from_str(&std::fs::read_to_string(&paths.tts_config)?)?;
        tracing::info!(
            "Config: sample_rate={}, base_chunk_size={}, chunk_compress={}, latent_dim={}",
            config.ae.sample_rate,
            config.ae.base_chunk_size,
            config.ttl.chunk_compress_factor,
            config.ttl.latent_dim,
        );

        let mut voice_styles = HashMap::new();
        for (name, path) in &paths.voice_files {
            let data = std::fs::read_to_string(path)?;
            let raw: VoiceStyleRaw = serde_json::from_str(&data)?;
            let ttl_dims = (
                raw.style_ttl.dims[0],
                raw.style_ttl.dims[1],
                raw.style_ttl.dims[2],
            );
            let dp_dims = (
                raw.style_dp.dims[0],
                raw.style_dp.dims[1],
                raw.style_dp.dims[2],
            );
            let style = VoiceStyle {
                style_ttl: raw.style_ttl.flatten(),
                style_ttl_dims: ttl_dims,
                style_dp: raw.style_dp.flatten(),
                style_dp_dims: dp_dims,
            };
            voice_styles.insert(name.clone(), style);
        }

        tracing::info!(
            "Model loaded successfully ({} voice styles)",
            voice_styles.len()
        );

        Ok(Self {
            sessions,
            text_processor,
            voice_styles,
            config,
        })
    }

    pub fn default_voice(&self) -> Option<&str> {
        for name in ["M1", "F1", "M2", "F2"] {
            if self.voice_styles.contains_key(name) {
                return Some(name);
            }
        }
        self.voice_styles.keys().next().map(|s| s.as_str())
    }
}
