//! Supertonic backend benchmark and parity check (ORT vs Candle).
//!
//! usage:
//!   supertonic_backend_bench bench  BACKEND MODELS_DIR [STEPS] [ITERATIONS] [OUT_DIR]
//!   supertonic_backend_bench parity MODELS_DIR BACKEND_A BACKEND_B [STEPS]
//!
//! BACKEND: ort | candle-cpu | candle-vulkan (the latter two need the matching
//! cargo feature). MODELS_DIR holds `onnx/` and `voice_styles/`.
//! Output is one JSON object per line. RTF = synthesis wall time / audio seconds.
use anyhow::{Result, anyhow, bail};
use sonicboom::tts::backend::{BackendChoice, Dims3, SupertonicBackend};
use sonicboom::tts::download::ModelPaths;
use sonicboom::tts::inference;
use sonicboom::tts::model::ModelHandle;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default, Debug, Clone)]
struct StageTimes {
    duration: Duration,
    encode_text: Duration,
    denoise: Duration,
    denoise_steps: usize,
    vocode: Duration,
}

/// Wraps a backend and accumulates per-stage wall time.
struct Timed {
    inner: Box<dyn SupertonicBackend>,
    t: Arc<Mutex<StageTimes>>,
}

impl SupertonicBackend for Timed {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn duration(&mut self, ids: &[i64], s: &[f32], d: Dims3) -> Result<f32> {
        let st = Instant::now();
        let r = self.inner.duration(ids, s, d);
        self.t.lock().unwrap().duration += st.elapsed();
        r
    }
    fn encode_text(&mut self, ids: &[i64], s: &[f32], d: Dims3, l: usize) -> Result<()> {
        let st = Instant::now();
        let r = self.inner.encode_text(ids, s, d, l);
        self.t.lock().unwrap().encode_text += st.elapsed();
        r
    }
    fn denoise_step(&mut self, x: &[f32], ld: usize, ll: usize, i: usize, n: usize) -> Result<Vec<f32>> {
        let st = Instant::now();
        let r = self.inner.denoise_step(x, ld, ll, i, n);
        let mut t = self.t.lock().unwrap();
        t.denoise += st.elapsed();
        t.denoise_steps += 1;
        r
    }
    fn vocode(&mut self, x: &[f32], ld: usize, ll: usize) -> Result<Vec<f32>> {
        let st = Instant::now();
        let r = self.inner.vocode(x, ld, ll);
        self.t.lock().unwrap().vocode += st.elapsed();
        r
    }
}

fn model_paths(dir: &Path) -> Result<ModelPaths> {
    let onnx = dir.join("onnx");
    let mut voices = Vec::new();
    for e in std::fs::read_dir(dir.join("voice_styles"))? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "json") {
            voices.push((p.file_stem().unwrap().to_string_lossy().into_owned(), p));
        }
    }
    voices.sort();
    Ok(ModelPaths {
        duration_predictor: onnx.join("duration_predictor.onnx"),
        text_encoder: onnx.join("text_encoder.onnx"),
        vector_estimator: onnx.join("vector_estimator.onnx"),
        vocoder: onnx.join("vocoder.onnx"),
        unicode_indexer: onnx.join("unicode_indexer.json"),
        tts_config: onnx.join("tts.json"),
        voice_files: voices,
    })
}

fn choice(s: &str) -> Result<BackendChoice> {
    BackendChoice::parse(s)
        .ok_or_else(|| anyhow!("backend '{s}' unknown or not compiled in (try ort|candle-cpu|candle-vulkan)"))
}

fn vm_kib(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

const CASES: &[(&str, &str, &str)] = &[
    ("short_en", "en", "Hello, this is a short test."),
    (
        "medium_en",
        "en",
        "The quick brown fox jumps over the lazy dog. Speech synthesis should sound natural, \
         with clear pronunciation and a steady rhythm throughout the whole sentence.",
    ),
    (
        "long_en",
        "en",
        "Real time factor measures how long a system takes to generate audio compared with the \
         duration of that audio. A value below one means the model is faster than playback and \
         can stream speech without gaps. We measure it on the target hardware, with warm models, \
         across several runs, and we report the median together with the slowest run so that \
         occasional stalls are not hidden. Memory use and start up time matter just as much for \
         a server that must stay resident.",
    ),
    (
        "short_es",
        "es",
        "Hola, esta es una prueba corta de síntesis de voz.",
    ),
];

fn run_bench(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        bail!("bench BACKEND MODELS_DIR [STEPS] [ITERATIONS] [OUT_DIR]");
    }
    let backend = choice(&args[0])?;
    let dir = PathBuf::from(&args[1]);
    let steps: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(5);
    let iters: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(5);
    let out_dir = args.get(4).map(PathBuf::from);
    if let Some(d) = &out_dir {
        std::fs::create_dir_all(d)?;
    }
    let paths = model_paths(&dir)?;

    let load_start = Instant::now();
    let mut model = ModelHandle::load_with(&paths, backend)?;
    let load_secs = load_start.elapsed().as_secs_f64();
    let times = Arc::new(Mutex::new(StageTimes::default()));
    {
        let mut guard = model.backend.lock().unwrap();
        let inner = std::mem::replace(&mut *guard, Box::new(NullBackend));
        *guard = Box::new(Timed { inner, t: times.clone() });
    }
    let _ = &mut model;
    let voice = model.default_voice().unwrap_or("M1").to_string();
    let rate = model.sample_rate() as f64;
    println!(
        "{}",
        serde_json::json!({"event":"loaded","backend":backend.as_str(),"load_secs":load_secs,
            "rss_kib":vm_kib("VmRSS:"),"steps":steps,"voice":voice})
    );

    for (name, lang, text) in CASES {
        let mut rtfs = Vec::new();
        for i in 0..iters {
            *times.lock().unwrap() = StageTimes::default();
            let start = Instant::now();
            let samples = inference::synthesize(&model, text, lang, &voice, steps, 200)?;
            let elapsed = start.elapsed().as_secs_f64();
            let audio = samples.len() as f64 / rate;
            let t = times.lock().unwrap().clone();
            let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
            let finite = samples.iter().all(|s| s.is_finite());
            let rtf = elapsed / audio.max(1e-9);
            rtfs.push(rtf);
            if i == 0 {
                if let Some(d) = &out_dir {
                    let spec = hound::WavSpec {
                        channels: 1,
                        sample_rate: rate as u32,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    };
                    let mut w = hound::WavWriter::create(
                        d.join(format!("{}-{}.wav", backend.as_str(), name)),
                        spec,
                    )?;
                    for s in &samples {
                        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
                    }
                    w.finalize()?;
                }
            }
            if i == 0 && std::env::var_os("CANDLE_ONNX_PROFILE").is_some() {
                // Per-op dispatch counts for the first run of each case (profile flag only).
                for (op, n) in candle_onnx::eval::take_dispatch_profile().into_iter().take(15) {
                    eprintln!("dispatch-profile case={name} op={op} dispatches={n}");
                }
            }
            println!(
                "{}",
                serde_json::json!({"event":"run","backend":backend.as_str(),"case":name,"iter":i,
                    "chars":text.chars().count(),"audio_secs":audio,"elapsed_secs":elapsed,"rtf":rtf,
                    "peak":peak,"finite":finite,
                    "stage_secs":{"duration":t.duration.as_secs_f64(),"encode_text":t.encode_text.as_secs_f64(),
                        "denoise":t.denoise.as_secs_f64(),"denoise_steps":t.denoise_steps,
                        "vocode":t.vocode.as_secs_f64()}})
            );
        }
        rtfs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{}",
            serde_json::json!({"event":"summary","backend":backend.as_str(),"case":name,
                "rtf_min":rtfs.first().copied(),"rtf_median":rtfs[rtfs.len()/2],"rtf_max":rtfs.last().copied()})
        );
    }
    println!(
        "{}",
        serde_json::json!({"event":"done","backend":backend.as_str(),"peak_rss_kib":vm_kib("VmHWM:")})
    );
    Ok(())
}

struct NullBackend;
impl SupertonicBackend for NullBackend {
    fn name(&self) -> &'static str {
        "null"
    }
    fn duration(&mut self, _: &[i64], _: &[f32], _: Dims3) -> Result<f32> {
        bail!("null backend")
    }
    fn encode_text(&mut self, _: &[i64], _: &[f32], _: Dims3, _: usize) -> Result<()> {
        bail!("null backend")
    }
    fn denoise_step(&mut self, _: &[f32], _: usize, _: usize, _: usize, _: usize) -> Result<Vec<f32>> {
        bail!("null backend")
    }
    fn vocode(&mut self, _: &[f32], _: usize, _: usize) -> Result<Vec<f32>> {
        bail!("null backend")
    }
}

fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let num: f64 = (0..n).map(|i| ((a[i] - b[i]) as f64).powi(2)).sum();
    let den: f64 = (0..n).map(|i| (a[i] as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Same fixed inputs through two backends; reports relative errors per stage.
fn run_parity(args: &[String]) -> Result<()> {
    if args.len() < 3 {
        bail!("parity MODELS_DIR BACKEND_A BACKEND_B [STEPS]");
    }
    let paths = model_paths(Path::new(&args[0]))?;
    let steps: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(5);
    let a = ModelHandle::load_with(&paths, choice(&args[1])?)?;
    let b = ModelHandle::load_with(&paths, choice(&args[2])?)?;
    let voice = a.default_voice().unwrap_or("M1").to_string();
    let style = &a.voice_styles[&voice];
    let (ids, _) = a.text_processor.encode("<en>Hello, this is a parity check.</en>");
    let cfg = &a.config;
    let latent_dim = cfg.ttl.latent_dim * cfg.ttl.chunk_compress_factor;

    let mut ba = a.backend.lock().unwrap();
    let mut bb = b.backend.lock().unwrap();
    let da = ba.duration(&ids, &style.style_dp, style.style_dp_dims)?;
    let db = bb.duration(&ids, &style.style_dp, style.style_dp_dims)?;
    println!("{}", serde_json::json!({"stage":"duration","a":da,"b":db,"abs_diff":(da-db).abs()}));

    let latent_len = ((da * cfg.ae.sample_rate as f32) as usize)
        .div_ceil(cfg.ae.base_chunk_size * cfg.ttl.chunk_compress_factor)
        .max(1);
    ba.encode_text(&ids, &style.style_ttl, style.style_ttl_dims, latent_len)?;
    bb.encode_text(&ids, &style.style_ttl, style.style_ttl_dims, latent_len)?;

    // Deterministic pseudo-noise (xorshift + Box-Muller).
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut xa: Vec<f32> = (0..latent_dim * latent_len)
        .map(|_| {
            let (u1, u2) = (next().max(1e-7), next());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    let mut xb = xa.clone();
    for s in 0..steps {
        xa = ba.denoise_step(&xa, latent_dim, latent_len, s, steps)?;
        xb = bb.denoise_step(&xb, latent_dim, latent_len, s, steps)?;
        println!("{}", serde_json::json!({"stage":"denoise","step":s,"rel_err":rel_err(&xa,&xb)}));
    }
    let wa = ba.vocode(&xa, latent_dim, latent_len)?;
    let wb = bb.vocode(&xb, latent_dim, latent_len)?;
    println!(
        "{}",
        serde_json::json!({"stage":"vocode","len_a":wa.len(),"len_b":wb.len(),"rel_err":rel_err(&wa,&wb)})
    );
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_writer(std::io::stderr).with_ansi(false).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => run_bench(&args[1..]),
        Some("parity") => run_parity(&args[1..]),
        _ => bail!("usage: supertonic_backend_bench bench|parity ..."),
    }
}
