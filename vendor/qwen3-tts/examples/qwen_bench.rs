//! Qwen3-TTS (candle) benchmark.
//! usage: qwen_bench MODEL_DIR DEVICE REF_WAV [x_vector|icl] [ITERS] [CASES] [OUT_DIR]
//! RTF = synthesis wall time / audio seconds. One JSON object per line.
use qwen3_tts::{parse_device, sync_device, AudioBuffer, Language, Qwen3TTS, SynthesisOptions};
use std::time::Instant;

const REF_TEXT: &str = "Hello, this is a voice cloning reference recording made for local testing.";
const CASES: &[(&str, &str)] = &[
    // Same sentence as benchmarks/results/2026-10-15-qwen-gguf (GGUF baseline).
    ("clone_test", "Hello, this is a native voice cloning test."),
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
];

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

fn main() -> anyhow::Result<()> {
    // Stage timings (generation / talker / acoustic / sampling) are emitted as tracing events.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        anyhow::bail!("usage: qwen_bench MODEL_DIR DEVICE REF_WAV [x_vector|icl] [ITERS] [CASES] [OUT_DIR]");
    }
    let mode = a.get(3).map(String::as_str).unwrap_or("x_vector");
    let iters: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let want: Vec<&str> = a.get(5).map(|s| s.split(',').collect()).unwrap_or(vec!["short_en", "medium_en"]);
    let out_dir = a.get(6);
    if let Some(d) = out_dir {
        std::fs::create_dir_all(d)?;
    }
    let device = parse_device(&a[1])?;
    let dtype_env = std::env::var("QWEN3_TTS_DTYPE").unwrap_or_else(|_| "default".into());

    let t = Instant::now();
    let model = Qwen3TTS::from_pretrained(&a[0], device.clone())?;
    sync_device(&device)?;
    let load_secs = t.elapsed().as_secs_f64();

    let reference = AudioBuffer::load(&a[2])?;
    let t = Instant::now();
    let prompt = model.create_voice_clone_prompt(&reference, if mode == "icl" { Some(REF_TEXT) } else { None })?;
    let prompt_secs = t.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"event":"loaded","backend":format!("qwen3-tts-candle-{}", a[1]),"dtype":dtype_env,
            "mode":mode,"load_secs":load_secs,"prompt_secs":prompt_secs,"rss_kib":vm_kib("VmRSS:")})
    );

    for (name, text) in CASES.iter().filter(|(n, _)| want.contains(n)) {
        let mut rtfs = vec![];
        for i in 0..iters {
            let options = SynthesisOptions {
                max_length: text.chars().count() * 2 + 25,
                seed: Some(42),
                ..SynthesisOptions::default()
            };
            let enc0 = candle_core::Device::encode_count();
            let t = Instant::now();
            let audio = model.synthesize_voice_clone(text, &prompt, Language::English, Some(options))?;
            sync_device(&device)?;
            let elapsed = t.elapsed().as_secs_f64();
            let dispatches = candle_core::Device::encode_count().saturating_sub(enc0);
            let secs = audio.samples.len() as f64 / audio.sample_rate as f64;
            let peak = audio.samples.iter().fold(0f32, |m, s| m.max(s.abs()));
            let finite = audio.samples.iter().all(|s| s.is_finite());
            let rtf = elapsed / secs.max(1e-9);
            rtfs.push(rtf);
            if i == 0 {
                if let Some(d) = out_dir {
                    audio.save(format!("{d}/qwen-candle-{}-{name}.wav", a[1]))?;
                }
            }
            println!(
                "{}",
                serde_json::json!({"event":"run","backend":format!("qwen3-tts-candle-{}", a[1]),"case":name,"iter":i,
                    "chars":text.chars().count(),"audio_secs":secs,"elapsed_secs":elapsed,"rtf":rtf,
                    "peak":peak,"finite":finite,"dispatches":dispatches})
            );
        }
        rtfs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!(
            "{}",
            serde_json::json!({"event":"summary","case":name,"rtf_min":rtfs[0],"rtf_median":rtfs[rtfs.len()/2],"rtf_max":rtfs[rtfs.len()-1]})
        );
    }
    println!("{}", serde_json::json!({"event":"done","peak_rss_kib":vm_kib("VmHWM:")}));
    Ok(())
}
