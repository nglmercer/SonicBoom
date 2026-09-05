use std::{path::PathBuf, sync::Arc};

use anyhow::{Result, anyhow};

use crate::tts::{audio, download, inference, model::ModelHandle};

const MAX_TEXT_CHARS: usize = 10_000;
const MAX_INFERENCE_STEPS: usize = 50;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_dir: PathBuf,
    pub hf_token: Option<String>,
    pub inference_steps: usize,
}

impl EngineConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            hf_token: None,
            inference_steps: 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub language: String,
    pub voice: String,
    pub inference_steps: usize,
}

impl SynthesisRequest {
    pub fn validate(&self) -> Result<()> {
        let text = self.text.trim();
        if text.is_empty() {
            return Err(anyhow!("text cannot be empty"));
        }
        if text.chars().count() > MAX_TEXT_CHARS {
            return Err(anyhow!("text exceeds {MAX_TEXT_CHARS} characters"));
        }
        if self.language.is_empty()
            || self.language.len() > 16
            || !self
                .language
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(anyhow!("language must be a valid language tag"));
        }
        if self.voice.is_empty() || self.voice.len() > 16 {
            return Err(anyhow!("voice must be a valid voice name"));
        }
        if !(1..=MAX_INFERENCE_STEPS).contains(&self.inference_steps) {
            return Err(anyhow!(
                "inferenceSteps must be between 1 and {MAX_INFERENCE_STEPS}"
            ));
        }
        Ok(())
    }
}

pub struct SynthesisResult {
    pub wav: Vec<u8>,
    pub sample_rate: u32,
}

pub struct SonicBoomEngine {
    model: Arc<ModelHandle>,
    inference_lock: std::sync::Mutex<()>,
}

impl SonicBoomEngine {
    pub async fn prepare<F>(config: EngineConfig, on_progress: F) -> Result<Arc<Self>>
    where
        F: Fn(f32) + Send + Sync + 'static,
    {
        let paths =
            download::download_models(&config.model_dir, config.hf_token.as_deref(), on_progress)
                .await?;
        let model = tokio::task::spawn_blocking(move || ModelHandle::load(&paths))
            .await
            .map_err(|error| anyhow!("model loading task failed: {error}"))??;
        Ok(Arc::new(Self {
            model: Arc::new(model),
            inference_lock: std::sync::Mutex::new(()),
        }))
    }

    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.model.voice_styles.keys().map(String::as_str)
    }

    pub fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResult> {
        request.validate()?;
        if !self.model.voice_styles.contains_key(&request.voice) {
            return Err(anyhow!("voice '{}' is not available", request.voice));
        }
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| anyhow!("inference lock poisoned"))?;
        let samples = inference::synthesize(
            &self.model,
            request.text.trim(),
            &request.language,
            &request.voice,
            request.inference_steps,
        )?;
        let sample_rate = self.model.sample_rate();
        let wav = audio::encode_audio(&samples, sample_rate, audio::AudioFormat::Wav)?;
        Ok(SynthesisResult { wav, sample_rate })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> SynthesisRequest {
        SynthesisRequest {
            text: "hello".to_owned(),
            language: "en".to_owned(),
            voice: "F1".to_owned(),
            inference_steps: 5,
        }
    }

    #[test]
    fn validates_synthesis_inputs_without_loading_a_model() {
        assert!(request().validate().is_ok());

        let mut invalid = request();
        invalid.text.clear();
        assert!(invalid.validate().is_err());

        let mut invalid = request();
        invalid.language = "en_US".to_owned();
        assert!(invalid.validate().is_err());

        let mut invalid = request();
        invalid.inference_steps = MAX_INFERENCE_STEPS + 1;
        assert!(invalid.validate().is_err());
    }
}
