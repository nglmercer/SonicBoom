//! Checkpoint configuration (`config.json` of `kyutai/tts-1.6b-en_fr`).
use anyhow::{Context, Result};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Config {
    pub card: usize,
    pub n_q: usize,
    pub dep_q: usize,
    pub dim: usize,
    pub text_card: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub hidden_ff: usize,
    pub context: usize,
    pub max_period: f64,
    pub dep_dim: usize,
    pub dep_heads: usize,
    pub dep_layers: usize,
    pub dep_hidden_ff: usize,
    pub dep_low_rank: usize,
    pub dep_schedule: Vec<usize>,
    pub delays: Vec<usize>,
    pub cfg_values: Vec<String>,
    pub audio_delay_s: f64,
    pub second_stream_ahead: usize,
    pub speaker_dim: usize,
    pub max_speakers: usize,
}

fn us(v: &Value, k: &str) -> Result<usize> {
    v[k].as_u64().map(|x| x as usize).with_context(|| format!("config.json: missing {k}"))
}

/// `GatedMLP` hidden width as computed by `ActivationGating` in Python.
fn gating_hidden(dim: usize, dim_ff: usize) -> usize {
    if dim_ff == 4 * dim {
        (21 * dim) / 8
    } else {
        (2 * dim_ff) / 3
    }
}

impl Config {
    pub fn from_json(text: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(text)?;
        let dim = us(&v, "dim")?;
        let dep_dim = us(&v, "depformer_dim")?;
        let hidden_scale = v["hidden_scale"].as_f64().context("hidden_scale")?;
        let dim_ff = (hidden_scale * dim as f64) as usize;
        let dep_ff = us(&v, "depformer_dim_feedforward")?;
        let usv = |k: &str| -> Result<Vec<usize>> {
            v[k].as_array()
                .with_context(|| k.to_string())?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize).context(k.to_string()))
                .collect()
        };
        let cfg_values = v["conditioners"]["cfg"]["lut"]["possible_values"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        Ok(Self {
            card: us(&v, "card")?,
            n_q: us(&v, "n_q")?,
            dep_q: us(&v, "dep_q")?,
            dim,
            text_card: us(&v, "text_card")?,
            num_heads: us(&v, "num_heads")?,
            num_layers: us(&v, "num_layers")?,
            hidden_ff: gating_hidden(dim, dim_ff),
            context: us(&v, "context")?,
            max_period: v["max_period"].as_f64().unwrap_or(10000.0),
            dep_dim,
            dep_heads: us(&v, "depformer_num_heads")?,
            dep_layers: us(&v, "depformer_num_layers")?,
            dep_hidden_ff: gating_hidden(dep_dim, dep_ff),
            dep_low_rank: us(&v, "depformer_low_rank_embeddings")?,
            dep_schedule: usv("depformer_weights_per_step_schedule")?,
            delays: usv("delays")?,
            cfg_values,
            audio_delay_s: v["tts_config"]["audio_delay"].as_f64().context("audio_delay")?,
            second_stream_ahead: v["tts_config"]["second_stream_ahead"].as_u64().unwrap_or(0) as usize,
            speaker_dim: v["conditioners"]["speaker_wavs"]["tensor"]["dim"].as_u64().unwrap_or(512) as usize,
            max_speakers: 5,
        })
    }

    pub fn num_weight_sets(&self) -> usize {
        self.dep_schedule.iter().copied().max().map_or(0, |m| m + 1)
    }
}
