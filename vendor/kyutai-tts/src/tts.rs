//! End-to-end generation: text -> delayed audio tokens -> Mimi PCM.
//! A port of `TTSModel.generate` (moshi/models/tts.py) for a single request.
use crate::config::Config;
use crate::lm::{build_conditions, Lm, QuantPlan, ZERO_TOKEN};
use crate::machine::{script_to_entries, StateMachine, TokenIds};
use anyhow::{anyhow, Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use crate::spm::Spm;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Options {
    /// CFG-distillation conditioning, one of "1.0".."4.0" in 0.5 steps.
    pub cfg_value: String,
    pub temp: f64,
    pub top_k_audio: usize,
    pub top_k_text: usize,
    pub seed: u64,
    pub padding_between: usize,
    pub initial_padding: i64,
    pub max_padding: i64,
    pub final_padding: usize,
    /// Hard cap on generation steps (12.5 steps per second of audio).
    pub max_steps: usize,
    /// Number of Mimi codebooks to generate/decode (<= 32).
    pub n_q: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            cfg_value: "2.0".into(),
            temp: 0.6,
            top_k_audio: 250,
            top_k_text: 25,
            seed: 42,
            padding_between: 1,
            initial_padding: 2,
            max_padding: 8,
            final_padding: 4,
            max_steps: 1500,
            n_q: 32,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Timings {
    pub steps: usize,
    pub frames_decoded: usize,
    pub lm_main: Duration,
    pub depformer: Duration,
    /// GPU dispatches issued by the main LM and by the depformer (accelerator runs).
    pub lm_dispatches: u64,
    pub depformer_dispatches: u64,
    pub mimi: Duration,
    pub total: Duration,
    /// Time from the start of `synthesize` until the first PCM chunk.
    pub first_audio: Option<Duration>,
}

pub struct KyutaiTts {
    pub cfg: Config,
    pub device: Device,
    pub dtype: DType,
    lm: Lm,
    mimi: moshi::mimi::Mimi,
    sp: Spm,
    machine: StateMachine,
    delay_steps: usize,
    vb: VarBuilder<'static>,
    pub sample_rate: usize,
}

impl KyutaiTts {
    /// `dir` holds `config.json`, the LM and Mimi safetensors and the SPM model
    /// named in `config.json`; `voice` is a `kyutai/tts-voices` embedding file.
    pub fn load(dir: &Path, voice: &Path, device: &Device, dtype: DType, cfg_value: &str) -> Result<Self> {
        Self::load_quantized(dir, voice, device, dtype, cfg_value, None)
    }

    /// Like [`Self::load`], optionally quantizing the large matrices (the
    /// activations then run in `dtype`, which must be F32 for quantized matmuls).
    pub fn load_quantized(
        dir: &Path,
        voice: &Path,
        device: &Device,
        dtype: DType,
        cfg_value: &str,
        quant: Option<candle::quantized::GgmlDType>,
    ) -> Result<Self> {
        let cfg_text = std::fs::read_to_string(dir.join("config.json")).context("config.json")?;
        let raw: serde_json::Value = serde_json::from_str(&cfg_text)?;
        let cfg = Config::from_json(&cfg_text)?;
        let name = |k: &str| -> Result<String> {
            raw[k].as_str().map(String::from).ok_or_else(|| anyhow!("config.json: missing {k}"))
        };
        let lm_file = dir.join(name("moshi_name")?);
        let mimi_file = dir.join(name("mimi_name")?);
        let spm_file = dir.join(name("tokenizer_name")?);

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&lm_file], dtype, device)? };
        let voice_t = candle::safetensors::load(voice, device)?
            .remove("speaker_wavs")
            .ok_or_else(|| anyhow!("voice file has no speaker_wavs tensor"))?;
        let cond = build_conditions(&cfg, &vb, &voice_t, cfg_value, device)?;
        let plan = quant.map(|dtype| QuantPlan { dtype, device: device.clone() });
        // Large matrices are read as F32 on the CPU when quantizing, so they never
        // exist in full precision on the accelerator.
        let src_cpu;
        let weights_src = if plan.is_some() {
            src_cpu = unsafe { VarBuilder::from_mmaped_safetensors(&[&lm_file], DType::F32, &Device::Cpu)? };
            &src_cpu
        } else {
            &vb
        };
        let lm = Lm::load(&cfg, vb.clone(), weights_src, plan.as_ref(), device, dtype, &cond)?;
        let mimi = moshi::mimi::load(mimi_file.to_str().ok_or_else(|| anyhow!("bad path"))?, Some(cfg.n_q), device)?;
        let sp = Spm::open(&spm_file).context("sentencepiece model")?;
        let ids = TokenIds::new(cfg.text_card as i64 + 1);
        let machine = StateMachine {
            ids,
            second_stream_ahead: cfg.second_stream_ahead,
            max_padding: 8,
            initial_padding: 2,
        };
        let delay_steps = (cfg.audio_delay_s * 12.5) as usize;
        Ok(Self { cfg, device: device.clone(), dtype, lm, mimi, sp, machine, delay_steps, vb, sample_rate: 24000 })
    }

    /// Swap the speaker voice without reloading the LM.
    pub fn set_voice(&mut self, voice: &Path, cfg_value: &str) -> Result<()> {
        let t = candle::safetensors::load(voice, &self.device)?
            .remove("speaker_wavs")
            .ok_or_else(|| anyhow!("voice file has no speaker_wavs tensor"))?;
        let cond = build_conditions(&self.cfg, &self.vb, &t, cfg_value, &self.device)?;
        self.lm.set_conditions(&cond)
    }

    pub fn encode_word(&self, word: &str) -> Vec<i64> {
        self.sp.encode(word).into_iter().map(i64::from).collect()
    }

    /// Generate speech for `text`. `on_pcm` receives each 80 ms chunk as soon as
    /// it is decoded (streaming); the full waveform is also returned.
    pub fn synthesize(
        &mut self,
        text: &str,
        opts: &Options,
        mut on_pcm: impl FnMut(&[f32]),
    ) -> Result<(Vec<f32>, Timings)> {
        let started = Instant::now();
        let mut tm = Timings::default();
        let cfg = self.cfg.clone();
        let n_q = cfg.n_q;
        let ids = self.machine.ids;
        let machine = StateMachine { max_padding: opts.max_padding, initial_padding: opts.initial_padding, ..self.machine.clone() };
        let entries = script_to_entries(&|w| self.encode_word(w), &ids, text, opts.padding_between);
        if entries.is_empty() {
            return Err(anyhow!("empty text"));
        }
        let mut state = machine.new_state(entries);
        self.lm.reset();
        self.mimi.reset_state();

        let sampling = |k: usize| Sampling::TopK { k, temperature: opts.temp };
        let mut text_lp = LogitsProcessor::from_sampling(opts.seed, sampling(opts.top_k_text));
        let mut audio_lp = LogitsProcessor::from_sampling(opts.seed.wrapping_add(1), sampling(opts.top_k_audio));

        let max_delay = *cfg.delays.iter().max().unwrap_or(&0);
        let mut gen_text: Vec<i64> = Vec::new();
        let mut gen_audio: Vec<Vec<i64>> = Vec::new();
        let mut pcm_all: Vec<f32> = Vec::new();
        let init_audio = cfg.card as i64;

        for step in 0..opts.max_steps {
            if let Some(end) = state.end_step {
                if step >= end + self.delay_steps + opts.final_padding {
                    break;
                }
            }
            // Inputs: previous step's tokens, or the initial token while the
            // corresponding stream has not started (`offset <= delay`).
            let text_in = if step == 0 { cfg.text_card as i64 } else { gen_text[step - 1] };
            let audio_in: Vec<i64> = (0..n_q)
                .map(|k| {
                    if step <= cfg.delays[k + 1] {
                        init_audio
                    } else {
                        gen_audio[step - 1][k]
                    }
                })
                .collect();

            let t0 = Instant::now();
            let e0 = candle::Device::encode_count();
            let (logits, tout) = self.lm.forward_text(text_in, &audio_in)?;
            let text_tok = text_lp.sample(&logits.to_dtype(DType::F32)?)? as i64;
            tm.lm_main += t0.elapsed();
            tm.lm_dispatches += candle::Device::encode_count().saturating_sub(e0);

            let (out_tok, _consumed) = machine.process(step, &mut state, text_tok);
            gen_text.push(out_tok);

            let mut audio = vec![ZERO_TOKEN; n_q];
            if step >= self.delay_steps {
                let t1 = Instant::now();
                let e1 = candle::Device::encode_count();
                self.lm.depformer_begin();
                let mut prev = out_tok;
                for cb in 0..n_q.min(opts.n_q) {
                    let l = self.lm.depformer_step(cb, prev, &tout)?;
                    let tok = audio_lp.sample(&l.to_dtype(DType::F32)?)? as i64;
                    audio[cb] = tok;
                    prev = tok;
                }
                for (q, a) in audio.iter_mut().enumerate() {
                    if step < cfg.delays[q + 1] + self.delay_steps {
                        *a = ZERO_TOKEN;
                    }
                }
                tm.depformer += t1.elapsed();
                tm.depformer_dispatches += candle::Device::encode_count().saturating_sub(e1);
            }
            gen_audio.push(audio);
            tm.steps += 1;

            // Frame f is complete once its delayed codebooks exist (f + max_delay).
            if step >= max_delay {
                let f = step - max_delay;
                if f >= self.delay_steps {
                    let t2 = Instant::now();
                    let mut codes = Vec::with_capacity(n_q);
                    for k in 0..n_q {
                        let src = if cfg.delays[k + 1] == 0 { f } else { f + cfg.delays[k + 1] };
                        codes.push(gen_audio[src][k].max(0) as u32);
                    }
                    let codes = Tensor::from_vec(codes, (1, n_q, 1), &self.device)?;
                    let pcm = self.mimi.decode_step(&codes.into(), &().into())?;
                    if let Some(pcm) = pcm.as_option() {
                        let v = pcm.flatten_all()?.to_vec1::<f32>()?;
                        tm.frames_decoded += 1;
                        if tm.first_audio.is_none() {
                            tm.first_audio = Some(started.elapsed());
                        }
                        on_pcm(&v);
                        pcm_all.extend(v.into_iter().map(|s| s.clamp(-1.0, 1.0)));
                    }
                    tm.mimi += t2.elapsed();
                }
            }
        }
        tm.total = started.elapsed();
        Ok((pcm_all, tm))
    }
}
