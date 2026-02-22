use anyhow::{anyhow, Result};
use ndarray::{Array, Array2, Array3};
use ort::value::Tensor;
use rand::Rng;

use crate::tts::model::{ModelHandle, VoiceStyle};

pub fn synthesize(
    model: &ModelHandle,
    text: &str,
    lang: &str,
    voice_name: &str,
    inference_steps: usize,
) -> Result<Vec<f32>> {
    let style = model
        .voice_styles
        .get(voice_name)
        .ok_or_else(|| anyhow!("Voice style '{voice_name}' not found"))?;

    let cfg = &model.config;
    let sample_rate = cfg.ae.sample_rate as f32;
    let chunk_size = cfg.ae.base_chunk_size * cfg.ttl.chunk_compress_factor;
    let latent_dim_eff = cfg.ttl.latent_dim * cfg.ttl.chunk_compress_factor;

    let chunks = crate::tts::text::TextProcessor::split_sentences(text);
    let mut all_samples: Vec<f32> = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        if i > 0 {
            // 청크 사이 0.2초 무음
            all_samples.extend(vec![0.0f32; (sample_rate * 0.2) as usize]);
        }
        // 레퍼런스: 언어 태그로 감싸고 끝에 마침표 추가
        let tagged = preprocess_chunk(chunk, lang);
        let chunk_samples = synthesize_chunk(
            model, &tagged, style, inference_steps, chunk_size, latent_dim_eff, sample_rate,
        )?;
        all_samples.extend(chunk_samples);
    }

    Ok(all_samples)
}

/// 레퍼런스(helper.rs)의 preprocess_text 간소화 버전
fn preprocess_chunk(text: &str, lang: &str) -> String {
    let text = text.trim();
    // 끝에 구두점이 없으면 마침표 추가
    let needs_period = !text.ends_with(['.', '!', '?', ';', ':', ',', '\'', '"', ')', ']', '}', '…', '。']);
    let body = if needs_period {
        format!("{text}.")
    } else {
        text.to_string()
    };
    // 언어 태그 래핑
    format!("<{lang}>{body}</{lang}>")
}

fn synthesize_chunk(
    model: &ModelHandle,
    tagged_text: &str,
    style: &VoiceStyle,
    inference_steps: usize,
    chunk_size: usize,
    latent_dim_eff: usize,
    sample_rate: f32,
) -> Result<Vec<f32>> {
    let (ids, _) = model.text_processor.encode(tagged_text);
    let seq_len = ids.len();
    if seq_len == 0 {
        return Ok(Vec::new());
    }

    // text_ids: [1, seq_len]
    let text_ids = Array2::<i64>::from_shape_vec((1, seq_len), ids)?;
    // text_mask: [1, 1, seq_len]
    let text_mask = Array3::<f32>::ones((1, 1, seq_len));

    // style_dp: [1, dp_dim1, dp_dim2]
    let (_, dp1, dp2) = style.style_dp_dims;
    let style_dp = Array3::<f32>::from_shape_vec((1, dp1, dp2), style.style_dp.clone())?;

    // style_ttl: [1, ttl_dim1, ttl_dim2]
    let (_, ttl1, ttl2) = style.style_ttl_dims;
    let style_ttl = Array3::<f32>::from_shape_vec((1, ttl1, ttl2), style.style_ttl.clone())?;

    // 1. Duration predictor → duration(초)
    let duration_sec: f32 = {
        let mut session = model.duration_predictor.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "text_ids"  => Tensor::from_array(text_ids.clone())?,
            "style_dp"  => Tensor::from_array(style_dp.clone())?,
            "text_mask" => Tensor::from_array(text_mask.clone())?,
        ])?;
        let (_, data) = outputs["duration"].try_extract_tensor::<f32>()?;
        data.iter().sum()
    };

    // latent_len = ceil(duration_sec * sample_rate / chunk_size)
    let wav_len = (duration_sec * sample_rate) as usize;
    let latent_len = ((wav_len + chunk_size - 1) / chunk_size).max(1);

    // latent_mask: [1, 1, latent_len]
    let latent_mask = Array3::<f32>::ones((1, 1, latent_len));

    // 2. Text encoder
    let (text_emb_dims, text_emb_vec): (Vec<usize>, Vec<f32>) = {
        let mut session = model.text_encoder.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "text_ids"  => Tensor::from_array(text_ids.clone())?,
            "style_ttl" => Tensor::from_array(style_ttl.clone())?,
            "text_mask" => Tensor::from_array(text_mask.clone())?,
        ])?;
        let (shape, data) = outputs["text_emb"].try_extract_tensor::<f32>()?;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        (dims, data.to_vec())
    };

    let text_emb = Array3::<f32>::from_shape_vec(
        (text_emb_dims[0], text_emb_dims[1], text_emb_dims[2]),
        text_emb_vec,
    )?;

    // 3. 노이즈 초기화 (마스크 적용)
    let mut rng = rand::rng();
    let noise: Vec<f32> = (0..latent_dim_eff * latent_len)
        .map(|_| {
            let u1: f32 = rng.random_range(f32::EPSILON..1.0);
            let u2: f32 = rng.random_range(0.0f32..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    let mut xt = Array3::<f32>::from_shape_vec((1, latent_dim_eff, latent_len), noise)?;

    let total_steps = inference_steps as f32;
    let total_step_arr = Array::from_elem(1, total_steps);

    // 4. Flow matching loop
    // 레퍼런스: denoised_latent를 바로 xt로 대입 (Euler step 아님)
    for step in 0..inference_steps {
        let current_step_arr = Array::from_elem(1, step as f32);

        let (vel_dims, vel_vec): (Vec<usize>, Vec<f32>) = {
            let mut session = model.vector_estimator.lock().unwrap();
            let outputs = session.run(ort::inputs![
                "noisy_latent" => Tensor::from_array(xt.clone())?,
                "text_emb"     => Tensor::from_array(text_emb.clone())?,
                "style_ttl"    => Tensor::from_array(style_ttl.clone())?,
                "latent_mask"  => Tensor::from_array(latent_mask.clone())?,
                "text_mask"    => Tensor::from_array(text_mask.clone())?,
                "current_step" => Tensor::from_array(current_step_arr)?,
                "total_step"   => Tensor::from_array(total_step_arr.clone())?,
            ])?;
            let (shape, data) = outputs["denoised_latent"].try_extract_tensor::<f32>()?;
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            (dims, data.to_vec())
        };

        // 레퍼런스: xt = denoised (직접 대입)
        xt = Array3::<f32>::from_shape_vec((vel_dims[0], vel_dims[1], vel_dims[2]), vel_vec)?;
    }

    // 5. Vocoder
    let wav_vec: Vec<f32> = {
        let mut session = model.vocoder.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "latent" => Tensor::from_array(xt)?,
        ])?;
        let (_, data) = outputs["wav_tts"].try_extract_tensor::<f32>()?;
        data.to_vec()
    };

    // 6. duration으로 트리밍 (레퍼런스: wav[..wav_len])
    let trimmed_len = wav_len.min(wav_vec.len());
    Ok(wav_vec[..trimmed_len].to_vec())
}
