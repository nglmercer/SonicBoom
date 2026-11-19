//! Inference backend seam for the four-graph Supertonic pipeline.
//!
//! `inference.rs` owns text chunking, length validation, the flow-matching
//! schedule and cancellation. A backend only executes the four graphs on plain
//! slices, so ONNX Runtime and Candle can be compared on identical inputs.

use anyhow::{Result, anyhow};
use ndarray::{Array, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;

use crate::tts::download::ModelPaths;

/// `(batch, rows, cols)` of a style tensor.
pub type Dims3 = (usize, usize, usize);

/// Which engine executes the Supertonic graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendChoice {
    /// ONNX Runtime (the shipping default).
    #[default]
    Ort,
    /// Candle's ONNX interpreter on the CPU.
    #[cfg(feature = "candle")]
    CandleCpu,
    /// Candle's ONNX interpreter on the Vulkan device.
    #[cfg(feature = "candle-vulkan")]
    CandleVulkan,
}

impl BackendChoice {
    /// The engine a build uses when configuration says `auto`: candle when it is
    /// compiled in (the migrated core), ONNX Runtime otherwise.
    pub fn auto() -> Self {
        #[cfg(feature = "candle")]
        {
            Self::CandleCpu
        }
        #[cfg(not(feature = "candle"))]
        {
            Self::Ort
        }
    }

    /// Resolve a configuration value (`auto`, `ort`, `candle-cpu`, `candle-vulkan`).
    pub fn resolve(name: &str) -> Result<Self, String> {
        let name = name.trim();
        if name.eq_ignore_ascii_case("auto") {
            return Ok(Self::auto());
        }
        Self::parse(name).ok_or_else(|| {
            format!("model.backend '{name}' is unknown or not compiled into this build")
        })
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ort" | "onnx" => Some(Self::Ort),
            #[cfg(feature = "candle")]
            "candle" | "candle-cpu" => Some(Self::CandleCpu),
            #[cfg(feature = "candle-vulkan")]
            "candle-vulkan" => Some(Self::CandleVulkan),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ort => "ort",
            #[cfg(feature = "candle")]
            Self::CandleCpu => "candle-cpu",
            #[cfg(feature = "candle-vulkan")]
            Self::CandleVulkan => "candle-vulkan",
        }
    }
}

/// Executes the Supertonic graphs. One chunk is processed as
/// `duration` → `encode_text` → N × `denoise_step` → `vocode`; the backend keeps
/// the per-chunk constants (text embedding, styles, masks) between calls.
pub trait SupertonicBackend: Send {
    fn name(&self) -> &'static str;

    /// Total predicted duration in seconds.
    fn duration(&mut self, ids: &[i64], style_dp: &[f32], dp_dims: Dims3) -> Result<f32>;

    /// Run the text encoder and retain the per-chunk constants.
    fn encode_text(
        &mut self,
        ids: &[i64],
        style_ttl: &[f32],
        ttl_dims: Dims3,
        latent_len: usize,
    ) -> Result<()>;

    /// One flow-matching step; returns the next latent (`[1, dim, len]` flat).
    fn denoise_step(
        &mut self,
        xt: &[f32],
        latent_dim: usize,
        latent_len: usize,
        step: usize,
        total_steps: usize,
    ) -> Result<Vec<f32>>;

    /// Latent to waveform samples.
    fn vocode(&mut self, latent: &[f32], latent_dim: usize, latent_len: usize) -> Result<Vec<f32>>;
}

pub fn load_backend(paths: &ModelPaths, choice: BackendChoice) -> Result<Box<dyn SupertonicBackend>> {
    match choice {
        BackendChoice::Ort => Ok(Box::new(OrtBackend::load(paths)?)),
        #[cfg(feature = "candle")]
        BackendChoice::CandleCpu => Ok(Box::new(candle_backend::CandleBackend::load(paths, false)?)),
        #[cfg(feature = "candle-vulkan")]
        BackendChoice::CandleVulkan => Ok(Box::new(candle_backend::CandleBackend::load(paths, true)?)),
    }
}

// ============================================================================
// ONNX Runtime
// ============================================================================

struct OrtChunk {
    text_emb: Tensor<f32>,
    style_ttl: Tensor<f32>,
    latent_mask: Tensor<f32>,
    text_mask: Tensor<f32>,
}

pub struct OrtBackend {
    duration_predictor: Session,
    text_encoder: Session,
    vector_estimator: Session,
    vocoder: Session,
    chunk: Option<OrtChunk>,
}

impl OrtBackend {
    pub fn load(paths: &ModelPaths) -> Result<Self> {
        Ok(Self {
            duration_predictor: crate::tts::model::build_session(&paths.duration_predictor)?,
            text_encoder: crate::tts::model::build_session(&paths.text_encoder)?,
            vector_estimator: crate::tts::model::build_session(&paths.vector_estimator)?,
            vocoder: crate::tts::model::build_session(&paths.vocoder)?,
            chunk: None,
        })
    }
}

impl SupertonicBackend for OrtBackend {
    fn name(&self) -> &'static str {
        "ort"
    }

    fn duration(&mut self, ids: &[i64], style_dp: &[f32], dp: Dims3) -> Result<f32> {
        let seq = ids.len();
        let text_ids = Array2::<i64>::from_shape_vec((1, seq), ids.to_vec())?;
        let style_dp = Array3::<f32>::from_shape_vec((1, dp.1, dp.2), style_dp.to_vec())?;
        let text_mask = Array3::<f32>::ones((1, 1, seq));
        let outputs = self.duration_predictor.run(ort::inputs![
            "text_ids"  => Tensor::from_array(text_ids)?,
            "style_dp"  => Tensor::from_array(style_dp)?,
            "text_mask" => Tensor::from_array(text_mask)?,
        ])?;
        let (_, data) = outputs["duration"].try_extract_tensor::<f32>()?;
        Ok(data.iter().sum())
    }

    fn encode_text(
        &mut self,
        ids: &[i64],
        style_ttl: &[f32],
        ttl: Dims3,
        latent_len: usize,
    ) -> Result<()> {
        let seq = ids.len();
        let text_ids = Array2::<i64>::from_shape_vec((1, seq), ids.to_vec())?;
        let style = Array3::<f32>::from_shape_vec((1, ttl.1, ttl.2), style_ttl.to_vec())?;
        let text_mask = Array3::<f32>::ones((1, 1, seq));
        let outputs = self.text_encoder.run(ort::inputs![
            "text_ids"  => Tensor::from_array(text_ids)?,
            "style_ttl" => Tensor::from_array(style.clone())?,
            "text_mask" => Tensor::from_array(text_mask.clone())?,
        ])?;
        let (shape, data) = outputs["text_emb"].try_extract_tensor::<f32>()?;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let text_emb = Array3::<f32>::from_shape_vec((dims[0], dims[1], dims[2]), data.to_vec())?;
        self.chunk = Some(OrtChunk {
            text_emb: Tensor::from_array(text_emb)?,
            style_ttl: Tensor::from_array(style)?,
            latent_mask: Tensor::from_array(Array3::<f32>::ones((1, 1, latent_len)))?,
            text_mask: Tensor::from_array(text_mask)?,
        });
        Ok(())
    }

    fn denoise_step(
        &mut self,
        xt: &[f32],
        latent_dim: usize,
        latent_len: usize,
        step: usize,
        total_steps: usize,
    ) -> Result<Vec<f32>> {
        let c = self
            .chunk
            .as_ref()
            .ok_or_else(|| anyhow!("denoise_step before encode_text"))?;
        let xt = Array3::<f32>::from_shape_vec((1, latent_dim, latent_len), xt.to_vec())?;
        let outputs = self.vector_estimator.run(ort::inputs![
            "noisy_latent" => Tensor::from_array(xt)?,
            "text_emb"     => c.text_emb.clone(),
            "style_ttl"    => c.style_ttl.clone(),
            "latent_mask"  => c.latent_mask.clone(),
            "text_mask"    => c.text_mask.clone(),
            "current_step" => Tensor::from_array(Array::from_elem(1, step as f32))?,
            "total_step"   => Tensor::from_array(Array::from_elem(1, total_steps as f32))?,
        ])?;
        let (_, data) = outputs["denoised_latent"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    fn vocode(&mut self, latent: &[f32], latent_dim: usize, latent_len: usize) -> Result<Vec<f32>> {
        let xt = Array3::<f32>::from_shape_vec((1, latent_dim, latent_len), latent.to_vec())?;
        let outputs = self.vocoder.run(ort::inputs![
            "latent" => Tensor::from_array(xt)?,
        ])?;
        let (_, data) = outputs["wav_tts"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

// ============================================================================
// Candle (candle-onnx interpreter)
// ============================================================================

#[cfg(feature = "candle")]
mod candle_backend {
    use super::*;
    use candle_core::{DType, Device, Tensor as CTensor};
    use candle_onnx::onnx::ModelProto;
    use std::collections::HashMap;

    struct Graph {
        model: ModelProto,
        init: HashMap<String, CTensor>,
    }

    impl Graph {
        fn load(path: &std::path::Path, device: &Device) -> Result<Self> {
            let model = candle_onnx::read_file(path)?;
            let mut init = candle_onnx::decode_initializers(&model)?;
            // Only floating point weights live on the accelerator; integer
            // constants (shapes, axes) are host data for the interpreter.
            for t in init.values_mut() {
                if matches!(t.dtype(), DType::F32 | DType::F16 | DType::BF16 | DType::F64) {
                    *t = t.to_device(device)?;
                }
            }
            Ok(Self { model, init })
        }

        fn run(&self, inputs: Vec<(&str, CTensor)>) -> Result<HashMap<String, CTensor>> {
            let inputs = inputs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
            Ok(candle_onnx::simple_eval_with_initializers(
                &self.model,
                &self.init,
                inputs,
            )?)
        }
    }

    struct Chunk {
        text_emb: CTensor,
        style_ttl: CTensor,
        latent_mask: CTensor,
        text_mask: CTensor,
    }

    pub struct CandleBackend {
        device: Device,
        dp: Graph,
        te: Graph,
        ve: Graph,
        voc: Graph,
        chunk: Option<Chunk>,
    }

    impl CandleBackend {
        pub fn load(paths: &ModelPaths, vulkan: bool) -> Result<Self> {
            let device = if vulkan {
                #[cfg(feature = "candle-vulkan")]
                {
                    Device::new_vulkan(0)?
                }
                #[cfg(not(feature = "candle-vulkan"))]
                {
                    return Err(anyhow!("built without the candle-vulkan feature"));
                }
            } else {
                Device::Cpu
            };
            Ok(Self {
                dp: Graph::load(&paths.duration_predictor, &device)?,
                te: Graph::load(&paths.text_encoder, &device)?,
                ve: Graph::load(&paths.vector_estimator, &device)?,
                voc: Graph::load(&paths.vocoder, &device)?,
                device,
                chunk: None,
            })
        }

        fn f32(&self, data: &[f32], shape: &[usize]) -> Result<CTensor> {
            Ok(CTensor::from_slice(data, shape, &self.device)?)
        }

        fn ones(&self, shape: &[usize]) -> Result<CTensor> {
            Ok(CTensor::ones(shape, DType::F32, &self.device)?)
        }

        fn ids(&self, ids: &[i64]) -> Result<CTensor> {
            // Token ids are index data: they stay on the host.
            Ok(CTensor::from_slice(ids, (1, ids.len()), &Device::Cpu)?)
        }
    }

    fn take(mut out: HashMap<String, CTensor>, name: &str) -> Result<CTensor> {
        out.remove(name)
            .ok_or_else(|| anyhow!("graph produced no output '{name}'"))
    }

    impl SupertonicBackend for CandleBackend {
        fn name(&self) -> &'static str {
            if matches!(self.device, Device::Cpu) {
                "candle-cpu"
            } else {
                "candle-vulkan"
            }
        }

        fn duration(&mut self, ids: &[i64], style_dp: &[f32], dp: Dims3) -> Result<f32> {
            let out = self.dp.run(vec![
                ("text_ids", self.ids(ids)?),
                ("style_dp", self.f32(style_dp, &[1, dp.1, dp.2])?),
                ("text_mask", self.ones(&[1, 1, ids.len()])?),
            ])?;
            let d = take(out, "duration")?.flatten_all()?.to_vec1::<f32>()?;
            Ok(d.iter().sum())
        }

        fn encode_text(
            &mut self,
            ids: &[i64],
            style_ttl: &[f32],
            ttl: Dims3,
            latent_len: usize,
        ) -> Result<()> {
            let style = self.f32(style_ttl, &[1, ttl.1, ttl.2])?;
            let text_mask = self.ones(&[1, 1, ids.len()])?;
            let out = self.te.run(vec![
                ("text_ids", self.ids(ids)?),
                ("style_ttl", style.clone()),
                ("text_mask", text_mask.clone()),
            ])?;
            self.chunk = Some(Chunk {
                text_emb: take(out, "text_emb")?,
                style_ttl: style,
                latent_mask: self.ones(&[1, 1, latent_len])?,
                text_mask,
            });
            Ok(())
        }

        fn denoise_step(
            &mut self,
            xt: &[f32],
            latent_dim: usize,
            latent_len: usize,
            step: usize,
            total_steps: usize,
        ) -> Result<Vec<f32>> {
            let c = self
                .chunk
                .as_ref()
                .ok_or_else(|| anyhow!("denoise_step before encode_text"))?;
            let out = self.ve.run(vec![
                ("noisy_latent", self.f32(xt, &[1, latent_dim, latent_len])?),
                ("text_emb", c.text_emb.clone()),
                ("style_ttl", c.style_ttl.clone()),
                ("latent_mask", c.latent_mask.clone()),
                ("text_mask", c.text_mask.clone()),
                ("current_step", self.f32(&[step as f32], &[1])?),
                ("total_step", self.f32(&[total_steps as f32], &[1])?),
            ])?;
            Ok(take(out, "denoised_latent")?.flatten_all()?.to_vec1::<f32>()?)
        }

        fn vocode(
            &mut self,
            latent: &[f32],
            latent_dim: usize,
            latent_len: usize,
        ) -> Result<Vec<f32>> {
            let out = self
                .voc
                .run(vec![("latent", self.f32(latent, &[1, latent_dim, latent_len])?)])?;
            Ok(take(out, "wav_tts")?.flatten_all()?.to_vec1::<f32>()?)
        }
    }
}
