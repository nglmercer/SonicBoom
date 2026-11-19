use anyhow::{Result, anyhow};
use rand::Rng;

use crate::tts::model::{ModelHandle, VoiceStyle};

/// Lock the model sessions for the duration of inference.
/// Since the flow-matching pipeline is inherently sequential (each step depends on
/// the previous output), acquiring a single lock for the whole chunk avoids lock
/// churn and eliminates any risk of deadlocks between the per-session mutexes.
macro_rules! lock_sessions {
    ($model:expr) => {
        $model.backend.lock()
    };
}

/// Upper bound on audio the duration predictor may claim for a single text
/// chunk. Chunks are at most a few hundred characters (default 200), which at
/// very slow speech rates still map to well under a minute of audio; anything
/// beyond this indicates a corrupt or adversarial predictor output, and must
/// fail before any tensor is allocated.
pub(crate) const MAX_CHUNK_DURATION_SECS: f64 = 60.0;
/// Upper bound on total audio accumulated across all chunks of one request
/// (10 minutes). Guards whole-text accumulation independently of per-chunk
/// checks.
pub(crate) const MAX_REQUEST_DURATION_SECS: f64 = 600.0;
/// Upper bound on the latent tensor (`latent_dim_eff × latent_len` f32
/// elements) allocated per chunk, in bytes.
pub(crate) const MAX_LATENT_BYTES: usize = 256 * 1024 * 1024;
/// Silence inserted between chunks, in seconds.
const INTER_CHUNK_SILENCE_SECS: f64 = 0.2;
/// Sanity range for model-declared sample rates (1 Hz – 192 kHz).
const MAX_SAMPLE_RATE: u32 = 192_000;
/// Sanity range for the effective latent channel dimension.
const MAX_LATENT_DIM: usize = 8192;

/// Tensor lengths derived from a validated predicted duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkLengths {
    /// `wav_len`: predictor-seconds × sample rate, rounded up.
    pub wav_len: usize,
    /// `latent_len`: `wav_len.div_ceil(chunk_size)`, at least 1.
    pub latent_len: usize,
}

/// Validate a predicted `duration_sec` and derive allocation lengths.
///
/// Every finite/range/overflow check runs **before** any tensor allocation
/// so a `NaN`, infinite, negative, or huge predictor output fails with an
/// actionable error instead of panicking or allocating unreasonable tensors.
pub(crate) fn chunk_lengths_from_duration(
    duration_sec: f32,
    sample_rate: u32,
    chunk_size: usize,
    latent_dim_eff: usize,
) -> Result<ChunkLengths> {
    if !duration_sec.is_finite() || duration_sec <= 0.0 {
        return Err(anyhow!(
            "duration predictor returned a non-positive or non-finite duration; synthesis aborted"
        ));
    }
    let duration = duration_sec as f64;
    if duration > MAX_CHUNK_DURATION_SECS {
        return Err(anyhow!(
            "predicted chunk duration exceeds the per-chunk limit; synthesis aborted"
        ));
    }
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        return Err(anyhow!("model sample rate out of range; synthesis aborted"));
    }
    if chunk_size == 0 {
        return Err(anyhow!("model chunk size out of range; synthesis aborted"));
    }
    if latent_dim_eff == 0 || latent_dim_eff > MAX_LATENT_DIM {
        return Err(anyhow!(
            "model latent dimension out of range; synthesis aborted"
        ));
    }
    let wav_len_f = duration * f64::from(sample_rate);
    if !wav_len_f.is_finite() {
        return Err(anyhow!(
            "predicted audio length is not representable; synthesis aborted"
        ));
    }
    let max_chunk_samples = MAX_CHUNK_DURATION_SECS * f64::from(sample_rate);
    if wav_len_f > max_chunk_samples {
        return Err(anyhow!(
            "predicted audio length exceeds the per-chunk sample budget; synthesis aborted"
        ));
    }
    // Length derivation is bit-identical to the pre-fix formula
    // `(duration_sec * sample_rate) as usize`: the same f32 multiply and
    // truncation, so ordinary valid synthesis is unchanged. The range checks
    // above bound the product to at most `60 × 192000 = 11.52M`, hence the
    // `as` cast can neither saturate nor overflow on any supported platform.
    let wav_len = (duration_sec * sample_rate as f32) as usize;
    let latent_len = wav_len.div_ceil(chunk_size).max(1);
    let latent_elems = latent_dim_eff
        .checked_mul(latent_len)
        .ok_or_else(|| anyhow!("latent tensor shape overflows; synthesis aborted"))?;
    let latent_bytes = latent_elems
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| anyhow!("latent tensor shape overflows; synthesis aborted"))?;
    if latent_bytes > MAX_LATENT_BYTES {
        return Err(anyhow!(
            "predicted latent tensor exceeds the per-chunk byte budget; synthesis aborted"
        ));
    }
    Ok(ChunkLengths {
        wav_len,
        latent_len,
    })
}

/// Samples of inter-chunk silence for `sample_rate`, via checked arithmetic
/// (replaces the lossy `as usize` float cast).
pub(crate) fn inter_chunk_silence_samples(sample_rate: u32) -> Result<usize> {
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        return Err(anyhow!("model sample rate out of range; synthesis aborted"));
    }
    let samples = (f64::from(sample_rate) * INTER_CHUNK_SILENCE_SECS).ceil();
    if !samples.is_finite() || samples > usize::MAX as f64 {
        return Err(anyhow!(
            "inter-chunk silence length is not representable; synthesis aborted"
        ));
    }
    Ok(samples as usize)
}

/// Whole-request sample budget for `sample_rate` (10 minutes of audio).
pub(crate) fn max_request_samples(sample_rate: u32) -> Result<usize> {
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        return Err(anyhow!("model sample rate out of range; synthesis aborted"));
    }
    let budget = MAX_REQUEST_DURATION_SECS * f64::from(sample_rate);
    if !budget.is_finite() || budget > usize::MAX as f64 {
        return Err(anyhow!(
            "request sample budget is not representable; synthesis aborted"
        ));
    }
    Ok(budget as usize)
}

/// Accumulate `add` samples into a running request total, enforcing the
/// whole-text sample budget with checked arithmetic.
pub(crate) fn checked_accumulate_samples(total: usize, add: usize, budget: usize) -> Result<usize> {
    let next = total
        .checked_add(add)
        .ok_or_else(|| anyhow!("request audio length overflows; synthesis aborted"))?;
    if next > budget {
        return Err(anyhow!(
            "request audio length exceeds the per-request budget; synthesis aborted"
        ));
    }
    Ok(next)
}

pub fn synthesize(
    model: &ModelHandle,
    text: &str,
    lang: &str,
    voice_name: &str,
    inference_steps: usize,
    max_chunk_chars: usize,
) -> Result<Vec<f32>> {
    let style = model
        .voice_styles
        .get(voice_name)
        .ok_or_else(|| anyhow!("Voice style '{voice_name}' not found"))?;

    let cfg = &model.config;
    let sample_rate = cfg.ae.sample_rate as f32;
    let chunk_size = cfg.ae.base_chunk_size * cfg.ttl.chunk_compress_factor;
    let latent_dim_eff = cfg.ttl.latent_dim * cfg.ttl.chunk_compress_factor;

    // Long text is segmented through the shared policy interface after the
    // caller normalized the full text once, so numbers and dates spanning a
    // potential boundary were expanded before any cut. Chunking output is
    // identical to the previous direct call; only fragment tracking is new.
    let policy = crate::tts::segmentation::SegmentPolicy::supertonic(max_chunk_chars);
    let fragments = crate::tts::segmentation::segment_text(text, &policy);
    if let Some(control) = crate::tts::control::current() {
        control.set_fragments_total(fragments.len());
    }
    let sample_rate_u32 = sample_rate as u32;
    let request_budget = max_request_samples(sample_rate_u32)?;
    let mut total_samples: usize = 0;
    let mut all_samples: Vec<f32> = Vec::new();

    for fragment in fragments.iter() {
        if let Some(control) = crate::tts::control::current() {
            anyhow::ensure!(
                !control.stopped(),
                "inference cancelled or deadline exceeded"
            );
        }
        if fragment.index > 0 {
            // 0.2 seconds of silence between chunks (checked arithmetic;
            // counted against the whole-request budget like chunk audio).
            let silence = inter_chunk_silence_samples(sample_rate_u32)?;
            total_samples = checked_accumulate_samples(total_samples, silence, request_budget)?;
            all_samples.extend(vec![0.0f32; silence]);
        }
        // Reference: wrap with language tag and add period at the end
        let tagged = preprocess_chunk(&fragment.text, lang);
        let chunk_samples = synthesize_chunk(
            model,
            &tagged,
            style,
            inference_steps,
            chunk_size,
            latent_dim_eff,
            sample_rate,
        )?;
        total_samples =
            checked_accumulate_samples(total_samples, chunk_samples.len(), request_budget)?;
        all_samples.extend(chunk_samples);
        // Publish per-fragment completion in original order so progress
        // consumers can render TTFA-adjacent readiness while later
        // fragments still generate.
        if let Some(control) = crate::tts::control::current() {
            control.set_fragment_done(fragment.index);
        }
    }

    Ok(all_samples)
}

/// Simplified version of preprocess_text from reference (helper.rs)
fn preprocess_chunk(text: &str, lang: &str) -> String {
    let text = text.trim();
    // Add period at the end if no punctuation exists
    let needs_period = !text.ends_with([
        '.', '!', '?', ';', ':', ',', '\'', '"', ')', ']', '}', '…', '。',
    ]);
    let body = if needs_period {
        format!("{text}.")
    } else {
        text.to_string()
    };
    // Language tag wrapping
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
    if ids.is_empty() {
        return Ok(Vec::new());
    }

    // Lock the backend for the entire chunk — inference is strictly sequential.
    let mut backend = lock_sessions!(model).map_err(|e| anyhow!("Backend lock poisoned: {e}"))?;

    // 1. Duration predictor → duration (seconds)
    let duration_sec = backend.duration(&ids, &style.style_dp, style.style_dp_dims)?;

    // Validate the predictor output *before* allocating any tensor:
    // non-finite, non-positive, huge, or overflowing durations fail here.
    let sample_rate_u32 = sample_rate as u32;
    let ChunkLengths {
        wav_len,
        latent_len,
    } = chunk_lengths_from_duration(duration_sec, sample_rate_u32, chunk_size, latent_dim_eff)?;

    // 2. Text encoder (the backend keeps the per-chunk constants)
    backend.encode_text(&ids, &style.style_ttl, style.style_ttl_dims, latent_len)?;

    // 3. Initialize noise
    let mut rng = rand::rng();
    let mut xt: Vec<f32> = (0..latent_dim_eff * latent_len)
        .map(|_| {
            let u1: f32 = rng.random_range(f32::EPSILON..1.0);
            let u2: f32 = rng.random_range(0.0f32..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();

    // 4. Flow matching loop
    // Reference: assign denoised_latent directly to xt (not Euler step)
    for step in 0..inference_steps {
        if let Some(control) = crate::tts::control::current() {
            anyhow::ensure!(
                control.update("generating", 0),
                "inference cancelled or deadline exceeded"
            );
        }
        xt = backend.denoise_step(&xt, latent_dim_eff, latent_len, step, inference_steps)?;
    }

    // 5. Vocoder
    if let Some(control) = crate::tts::control::current() {
        anyhow::ensure!(
            control.update("decoding", 0),
            "inference cancelled or deadline exceeded"
        );
    }
    let wav_vec = backend.vocode(&xt, latent_dim_eff, latent_len)?;

    // 6. Trim by duration (reference: wav[..wav_len])
    let trimmed_len = wav_len.min(wav_vec.len());
    Ok(wav_vec[..trimmed_len].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 24_000;
    const CHUNK: usize = 480;
    const LATENT_DIM: usize = 128;

    #[test]
    fn preprocess_chunk_wraps_prepared_spanish_text_once() {
        assert_eq!(
            preprocess_chunk("Hay diez mil cinco solicitudes.", "es"),
            "<es>Hay diez mil cinco solicitudes.</es>"
        );
        // A missing terminator gets exactly one period, inside the tag.
        assert_eq!(
            preprocess_chunk("siete errores", "es"),
            "<es>siete errores.</es>"
        );
        // Existing sentence punctuation is preserved without doubling.
        assert_eq!(
            preprocess_chunk("¿Cuántas solicitudes?", "es"),
            "<es>¿Cuántas solicitudes?</es>"
        );
    }

    #[test]
    fn valid_durations_derive_exact_lengths() {
        // 1.0 s at 24 kHz, chunk 480 → 24000 samples, 50 latents.
        let out = chunk_lengths_from_duration(1.0, RATE, CHUNK, LATENT_DIM).unwrap();
        assert_eq!(
            out,
            ChunkLengths {
                wav_len: 24_000,
                latent_len: 50,
            }
        );
        // Fractional durations truncate exactly like the legacy `as usize`
        // cast (1.00001 s → 24000.24 → 24000), preserving valid-synthesis
        // lengths bit-for-bit.
        let out = chunk_lengths_from_duration(1.00001, RATE, CHUNK, LATENT_DIM).unwrap();
        assert_eq!(out.wav_len, 24_000);
        assert_eq!(out.latent_len, 50);
        // Tiny positive durations truncate to zero samples (legacy
        // behavior) while the latent length still floors at 1.
        let out = chunk_lengths_from_duration(f32::MIN_POSITIVE, RATE, CHUNK, LATENT_DIM).unwrap();
        assert_eq!(out.wav_len, 0);
        assert_eq!(out.latent_len, 1);
        // Exact durations are unchanged versus the old truncation path.
        let out = chunk_lengths_from_duration(2.5, RATE, CHUNK, LATENT_DIM).unwrap();
        assert_eq!(out.wav_len, 60_000);
    }

    #[test]
    fn valid_lengths_match_the_legacy_formula_exactly() {
        // For every in-range duration, the validated path must derive the
        // same lengths the pre-fix code computed, so valid synthesis output
        // (tensors, trim, audio) is bit-identical to before.
        let mut x = 0x12345678u32;
        let mut rng = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for _ in 0..5_000 {
            // Uniform-ish valid durations in (0, 60]: reuse f32 fractions.
            let frac = (rng() as f64 / u32::MAX as f64).clamp(0.0, 1.0);
            let duration = (frac * MAX_CHUNK_DURATION_SECS).max(f64::EPSILON) as f32;
            if duration <= 0.0 || duration as f64 > MAX_CHUNK_DURATION_SECS {
                continue;
            }
            let out = chunk_lengths_from_duration(duration, RATE, CHUNK, LATENT_DIM).unwrap();
            let legacy_wav = (duration * RATE as f32) as usize;
            let legacy_latent = legacy_wav.div_ceil(CHUNK).max(1);
            assert_eq!(out.wav_len, legacy_wav, "drift at {duration}");
            assert_eq!(out.latent_len, legacy_latent, "drift at {duration}");
        }
    }

    #[test]
    fn non_finite_and_non_positive_durations_fail() {
        for bad in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            -1.0,
            -1e10,
            f32::MIN,
        ] {
            assert!(
                chunk_lengths_from_duration(bad, RATE, CHUNK, LATENT_DIM).is_err(),
                "duration {bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn huge_and_boundary_durations_fail_before_allocation() {
        for bad in [1e10f32, 1e20, f32::MAX, 61.0, 3600.0] {
            assert!(
                chunk_lengths_from_duration(bad, RATE, CHUNK, LATENT_DIM).is_err(),
                "duration {bad} must be rejected"
            );
        }
        // Exactly the boundary is accepted; just above it is not.
        assert!(chunk_lengths_from_duration(60.0, RATE, CHUNK, LATENT_DIM).is_ok());
        assert!(chunk_lengths_from_duration(60.0001, RATE, CHUNK, LATENT_DIM).is_err());
        // At the boundary the tensor still fits the byte budget:
        // 60 s × 24 kHz = 1.44M samples → 3000 latents × 128ch × 4B = 1.5 MiB.
        let out = chunk_lengths_from_duration(60.0, RATE, CHUNK, LATENT_DIM).unwrap();
        assert_eq!(out.wav_len, 1_440_000);
        assert_eq!(out.latent_len, 3000);
    }

    #[test]
    fn degenerate_model_geometry_is_rejected() {
        assert!(chunk_lengths_from_duration(1.0, 0, CHUNK, LATENT_DIM).is_err());
        assert!(chunk_lengths_from_duration(1.0, 1_000_000, CHUNK, LATENT_DIM).is_err());
        assert!(chunk_lengths_from_duration(1.0, RATE, 0, LATENT_DIM).is_err());
        assert!(chunk_lengths_from_duration(1.0, RATE, CHUNK, 0).is_err());
        assert!(chunk_lengths_from_duration(1.0, RATE, CHUNK, 1_000_000).is_err());
    }

    #[test]
    fn latent_byte_budget_catches_small_chunk_attacks() {
        // 60 s × 24 kHz / 480 → 3000 latents × 8192ch × 4B ≈ 98 MiB —
        // still under budget, accepted.
        assert!(chunk_lengths_from_duration(60.0, RATE, 480, 8192).is_ok());
        // But the byte cap bites before memory does: 60 s at 192 kHz with
        // chunk 1 and 8192 channels → 11.52M latents × 8192 × 4B ≈ 377 GB.
        assert!(chunk_lengths_from_duration(60.0, 192_000, 1, 8192).is_err());
        // Overflow-shaped geometry (chunk 1, max dim, max rate) is rejected
        // by the byte budget rather than by wrapping arithmetic.
        assert!(chunk_lengths_from_duration(60.0, 192_000, 1, MAX_LATENT_DIM).is_err());
    }

    #[test]
    fn silence_and_request_budgets_use_checked_arithmetic() {
        assert_eq!(inter_chunk_silence_samples(24_000).unwrap(), 4800);
        assert_eq!(inter_chunk_silence_samples(1).unwrap(), 1);
        assert!(inter_chunk_silence_samples(0).is_err());
        assert!(inter_chunk_silence_samples(1_000_000).is_err());

        let budget = max_request_samples(24_000).unwrap();
        assert_eq!(budget, 14_400_000);
        assert!(max_request_samples(0).is_err());

        assert_eq!(checked_accumulate_samples(0, 100, budget).unwrap(), 100);
        assert_eq!(
            checked_accumulate_samples(usize::MAX, 1, usize::MAX)
                .unwrap_err()
                .to_string(),
            "request audio length overflows; synthesis aborted"
        );
        assert!(
            checked_accumulate_samples(budget, 1, budget)
                .unwrap_err()
                .to_string()
                .contains("per-request budget")
        );
        assert!(checked_accumulate_samples(budget, 0, budget).is_ok());
    }

    #[test]
    fn failure_messages_are_sanitized() {
        // Non-finite hostile inputs must not be interpolated into errors.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = chunk_lengths_from_duration(bad, RATE, CHUNK, LATENT_DIM)
                .unwrap_err()
                .to_string();
            assert!(
                !err.contains("NaN") && !err.contains("inf"),
                "leaky message: {err}"
            );
        }
        let err = chunk_lengths_from_duration(1e30, RATE, CHUNK, LATENT_DIM)
            .unwrap_err()
            .to_string();
        assert!(!err.contains("1e30"), "leaky message: {err}");
    }
}
