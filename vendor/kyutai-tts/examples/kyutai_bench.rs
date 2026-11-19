//! Kyutai TTS (candle) benchmark.
//! usage: kyutai_bench MODEL_DIR VOICE.safetensors DEVICE DTYPE [ITERS] [CASES] [OUT_DIR] [MAX_STEPS]
//! DEVICE: cpu|vulkan; DTYPE: f32|f16|bf16. RTF = wall time / audio seconds.
use anyhow::{bail, Result};
use candle::{DType, Device};
use kyutai_tts::tts::{KyutaiTts, Options};
use std::path::Path;
use std::time::Instant;

const CASES: &[(&str, &str)] = &[
    ("short_en", "Hello, this is a short test."),
    (
        "medium_en",
        "The quick brown fox jumps over the lazy dog. Speech synthesis should sound natural, \
         with clear pronunciation and a steady rhythm throughout the whole sentence.",
    ),
    (
        "long_en",
        "Real time factor measures how long a system takes to generate audio compared with the \
         duration of that audio. A value below one means the model is faster than playback and \
         can stream speech without gaps. We measure it on the target hardware, with warm models, \
         across several runs, and we report the median together with the slowest run so that \
         occasional stalls are not hidden. Memory use and start up time matter just as much for \
         a server that must stay resident.",
    ),
    ("short_fr", "Bonjour, ceci est un court test de synthèse vocale."),
];

fn vm_kib(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()))
        .unwrap_or(0)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_writer(std::io::stderr).with_ansi(false).init();
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        bail!("usage: kyutai_bench MODEL_DIR VOICE DEVICE DTYPE [ITERS] [CASES] [OUT_DIR] [MAX_STEPS]");
    }
    let device = match a[2].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "vulkan")]
        "vulkan" => Device::new_vulkan(0)?,
        d => bail!("unknown or unavailable device {d}"),
    };
    let (dtype, quant) = match a[3].as_str() {
        // Q8_0 weights with F32 activations.
        "q8" => (DType::F32, Some(candle::quantized::GgmlDType::Q8_0)),
        "q4" => (DType::F32, Some(candle::quantized::GgmlDType::Q4_0)),
        "f32" => (DType::F32, None),
        "f16" => (DType::F16, None),
        "bf16" => (DType::BF16, None),
        d => bail!("unknown dtype {d}"),
    };
    let _unused = match "f32" {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        d => bail!("unknown dtype {d}"),
    };
    let iters: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(2);
    let want: Vec<&str> = a.get(5).map(|s| s.split(',').collect()).unwrap_or(vec!["short_en"]);
    let out_dir = a.get(6);
    let mut opts = Options::default();
    if let Ok(v) = std::env::var("KYUTAI_SEED") {
        opts.seed = v.parse()?;
    }
    if let Some(m) = a.get(7) {
        opts.max_steps = m.parse()?;
    }
    if let Some(d) = out_dir {
        std::fs::create_dir_all(d)?;
    }
    let backend = format!("kyutai-tts-candle-{}-{}", a[2], a[3]);

    let t = Instant::now();
    let mut tts = KyutaiTts::load_quantized(Path::new(&a[0]), Path::new(&a[1]), &device, dtype, &opts.cfg_value, quant)?;
    let load_secs = t.elapsed().as_secs_f64();
    println!("{}", serde_json::json!({"event":"loaded","backend":backend,"load_secs":load_secs,"rss_kib":vm_kib("VmRSS:")}));

    for (name, text) in CASES.iter().filter(|(n, _)| want.contains(n)) {
        let mut rtfs = vec![];
        for i in 0..iters {
            let t = Instant::now();
            let enc0 = candle::Device::encode_count();
            let (pcm, tm) = tts.synthesize(text, &opts, |_| {})?;
            let dispatches = candle::Device::encode_count().saturating_sub(enc0);
            let wall = t.elapsed().as_secs_f64();
            let secs = pcm.len() as f64 / 24000.0;
            let rtf = wall / secs.max(1e-9);
            let peak = pcm.iter().fold(0f32, |m, s| m.max(s.abs()));
            rtfs.push(rtf);
            if i == 0 {
                if let Some(d) = out_dir {
                    let spec = hound::WavSpec { channels: 1, sample_rate: 24000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
                    let mut w = hound::WavWriter::create(format!("{d}/{backend}-{name}.wav"), spec)?;
                    for s in &pcm {
                        w.write_sample((s * 32767.0) as i16)?;
                    }
                    w.finalize()?;
                }
            }
            println!(
                "{}",
                serde_json::json!({"event":"run","backend":backend,"case":name,"iter":i,"chars":text.chars().count(),
                    "audio_secs":secs,"elapsed_secs":wall,"rtf":rtf,"peak":peak,"finite":pcm.iter().all(|s| s.is_finite()),
                    "ttfa_secs":tm.first_audio.map(|d| d.as_secs_f64()),
                    "stage_secs":{"lm_main":tm.lm_main.as_secs_f64(),"depformer":tm.depformer.as_secs_f64(),"mimi":tm.mimi.as_secs_f64(),
                    "steps":tm.steps,"frames_decoded":tm.frames_decoded},"dispatches":dispatches,"dispatches_per_step":dispatches as f64/tm.steps.max(1) as f64,"lm_dispatches":tm.lm_dispatches,"depformer_dispatches":tm.depformer_dispatches})
            );
        }
        rtfs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!("{}", serde_json::json!({"event":"summary","backend":backend,"case":name,"rtf_min":rtfs[0],"rtf_median":rtfs[rtfs.len()/2],"rtf_max":rtfs[rtfs.len()-1]}));
    }
    println!("{}", serde_json::json!({"event":"depformer_sections","dispatches":kyutai_tts::lm::SECTION_DISPATCHES.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).collect::<Vec<u64>>()}));
    println!("{}", serde_json::json!({"event":"done","backend":backend,"peak_rss_kib":vm_kib("VmHWM:")}));
    Ok(())
}
