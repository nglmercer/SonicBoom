//! Managed voice-profile endpoints.
//!
//! Authenticated management of cloning voice profiles. Profiles are scoped
//! to the owning token (via its non-secret hash): one token's profiles are
//! invisible to another (unknown IDs report `404`, never `403` with
//! existence disclosure). Reference transcripts and audio bytes are never
//! logged.

use axum::{
    body::Body,
    extract::{FromRequest, Multipart, Path, Query, State},
    http::{StatusCode, header},
    response::IntoResponse,
};
use serde::Deserialize;

use sonicboom::tts::providers::types::{CloneMode, DevicePreference, ReferenceAudio};
use sonicboom::tts::voices::{
    NAME_MAX_CHARS, REFERENCE_MAX_CHANNELS, REFERENCE_MAX_RATE, REFERENCE_MAX_SECS,
    REFERENCE_MIN_RATE, REFERENCE_MIN_SECS, TRANSCRIPT_MAX_CHARS,
};
use sonicboom::tts::voices::{NewVoiceProfile, VoiceProfileStore};

use super::providers::{build_registry_for_state, provider_error_to_app};
use super::tts::{check_rate_limit, synthesize_provider_bounded};
use crate::auth::AuthenticatedToken;
use crate::error::AppError;

#[derive(Deserialize)]
pub struct CloneQuery {
    #[serde(default)]
    pub expected_revision: Option<u64>,
    pub name: String,
    pub provider: String,
    pub model: String,
    pub language: String,
    #[serde(default = "default_clone_mode")]
    pub mode: String,
    #[serde(default)]
    pub transcript: Option<String>,
    pub consent_basis: String,
    #[serde(default)]
    pub consent_note: Option<String>,
}

fn default_clone_mode() -> String {
    "icl".to_string()
}

pub(crate) fn parse_clone_mode_public(raw: &str) -> Result<CloneMode, AppError> {
    parse_clone_mode(raw)
}

fn parse_clone_mode(raw: &str) -> Result<CloneMode, AppError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "icl" => Ok(CloneMode::Icl),
        "x_vector" | "x-vector" | "xvector" => Ok(CloneMode::XVector),
        "audio_prompt" | "audio-prompt" | "audioprompt" => Ok(CloneMode::AudioPrompt),
        other => Err(AppError::BadRequest(format!(
            "unsupported clone mode '{other}' (expected one of: icl, x_vector, audio_prompt)"
        ))),
    }
}

/// Decode an uploaded WAV body to mono `f32` PCM (MIME **and** actual format
/// are validated; anything undecodable is `400`, non-WAV media is `415`).
fn decode_wav_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32), AppError> {
    let mut reader = hound::WavReader::new(bytes).map_err(|_| {
        AppError::BadRequest(
            "reference audio is not a decodable WAV file (check the file contents, not just the extension)"
                .to_string(),
        )
    })?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.channels > REFERENCE_MAX_CHANNELS {
        return Err(AppError::BadRequest(format!(
            "reference WAV has {} channels; mono or stereo is required",
            spec.channels
        )));
    }
    if spec.sample_rate == 0 {
        return Err(AppError::BadRequest(
            "reference WAV sample rate must be nonzero".to_string(),
        ));
    }
    if !(REFERENCE_MIN_RATE..=REFERENCE_MAX_RATE).contains(&spec.sample_rate)
        || reader.duration() as f64 / f64::from(spec.sample_rate) > REFERENCE_MAX_SECS
        || reader.len() > 16_000_000
        || u64::from(reader.len()) * u64::from(spec.bits_per_sample.div_ceil(8))
            > bytes.len() as u64
    {
        return Err(AppError::BadRequest("reference WAV duration, sample count or rate exceeds its bound, or its data is truncated".into()));
    }
    let pcm: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| AppError::BadRequest("reference WAV samples are corrupt".to_string()))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample.max(1) as f32;
            let scale = 2f32.powf(bits - 1.0);
            reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| AppError::BadRequest("reference WAV samples are corrupt".to_string()))?
                .into_iter()
                .map(|s| s as f32 / scale)
                .collect()
        }
    };
    // Stereo is mixed down to mono explicitly (documented, not silent: the
    // stored duration/hash reflect the mono mix).
    let mono = if spec.channels == 2 {
        let mut mixed = Vec::with_capacity(pcm.len() / 2);
        let mut pairs = pcm.into_iter();
        while let (Some(left), Some(right)) = (pairs.next(), pairs.next()) {
            mixed.push((left + right) / 2.0);
        }
        mixed
    } else {
        pcm
    };
    Ok((mono, spec.sample_rate))
}

/// Non-destructive health readout of the mono mix before conditioning.
///
/// Returns `(measurements, warnings, defects, guidance)`:
/// * `defects` are confirmed numerical faults (never silent: they always
///   surface, but still do not reject or modify the clip),
/// * `warnings` are conservative heuristics that must not reject a valid clip,
/// * `guidance` is advisory (e.g. the 5-10 s recording recommendation) and never
///   a model restriction.
fn reference_health(
    pcm: &[f32],
    rate: u32,
    source_channels: u16,
) -> (
    serde_json::Value,
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<&'static str>,
) {
    let mut defects = Vec::new();
    let mut warnings = Vec::new();
    let mut guidance = Vec::new();
    // Single O(n) measurement pass shared with the generation-quality module.
    let health = sonicboom::tts::quality::audio_health::analyze(pcm, rate);
    let duration_secs = pcm.len() as f64 / rate.max(1) as f64;
    let peak = health.peak;
    let rms = health.rms;
    let dc = health.dc_offset;
    let clipping = health.clipping_fraction;
    let low_amplitude = health.near_silent_fraction;
    if health.non_finite_samples > 0 {
        defects.push("reference_non_finite_samples");
    }
    if rms < 1e-4 {
        warnings.push("reference_nearly_silent");
    }
    if clipping > 0.1 {
        warnings.push("reference_heavily_clipped");
    }
    if dc.abs() > 0.1 {
        warnings.push("reference_large_dc_offset");
    }
    if (1e-4..0.01).contains(&rms) {
        warnings.push("reference_unusually_quiet");
    }
    // Frame-level (20 ms) energy profile: leading/trailing silence, dynamic
    // range, and the noise floor. Purely heuristic — a pure tone, a DC block,
    // or a constant-amplitude clip has no dynamic range and is never mistaken
    // for truncated speech or noisy speech.
    let frame = (rate as usize / 50).max(1);
    let energies: Vec<f64> = pcm
        .chunks(frame)
        .map(|window| {
            (window.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / window.len().max(1) as f64)
                .sqrt()
        })
        .collect();
    let silence_level = (peak as f64 * 0.01).max(1e-4);
    let (leading_secs, trailing_secs, speech_like, floor) = if energies.len() > 4 {
        let quiet = |value: f64| value < silence_level;
        let leading = energies.iter().take_while(|e| quiet(**e)).count();
        let trailing = energies.iter().rev().take_while(|e| quiet(**e)).count();
        let mut sorted = energies.clone();
        sorted.sort_by(f64::total_cmp);
        let floor = sorted[sorted.len() / 10];
        let loud = sorted[sorted.len() * 9 / 10];
        // Real pauses between speech produce a wide dynamic range; a steady
        // tone or a clipped square wave does not.
        let speech_like = loud > floor * 3.0 + 1e-6;
        let secs = |frames: usize| frames as f64 * frame as f64 / f64::from(rate.max(1));
        (secs(leading), secs(trailing), speech_like, floor)
    } else {
        (0.0, 0.0, false, 0.0)
    };
    if rms >= 1e-4 {
        if leading_secs > 1.0 {
            guidance.push("reference_leading_silence");
        }
        if trailing_secs > 1.0 {
            guidance.push("reference_trailing_silence");
        }
    }
    if speech_like {
        // The recording likely stops mid-speech: loud energy up to the final
        // sample with no natural trailing silence.
        if trailing_secs < 0.05 && energies.last().is_some_and(|e| e >= &silence_level) {
            warnings.push("reference_ends_during_speech");
        }
        // Noise floor: the quietest decile of the clip is far above digital
        // silence, suggesting background contamination.
        if floor > 0.005 && floor > peak as f64 * 0.02 {
            warnings.push("reference_possible_background_noise");
        }
    }
    if duration_secs < 3.0 {
        guidance.push("reference_short_for_cloning");
    }
    (
        serde_json::json!({"analysis":"mono_mix_before_resampling", "sample_count":pcm.len(),
        "source_channels":source_channels,"source_sample_rate":rate,
        "duration_secs":duration_secs,"peak":peak,"rms":rms,
        "dc_offset":dc,"clipping_fraction":clipping,"low_amplitude_fraction":low_amplitude,
        "non_finite_samples":health.non_finite_samples,
        "max_discontinuity":health.max_discontinuity,
        "impulsive_samples":health.impulsive_samples,
        "leading_silence_secs":leading_secs,"trailing_silence_secs":trailing_secs,
        "frame_count":energies.len()}),
        warnings,
        defects,
        guidance,
    )
}

/// One resident manifest and bounded decoded-reference LRU. All disk work
/// and mutation is serialized on blocking threads, never on Tokio executors.
#[derive(Debug, Default)]
pub struct CachedVoices {
    dir: String,
    store: Option<VoiceProfileStore>,
    stamp: Option<Vec<u64>>,
    references: std::collections::VecDeque<(String, Option<Vec<u64>>, ReferenceAudio)>,
}

/// Stored-reference size bound (16 MiB) and total decoded-reference cache
/// budget (64 MiB across at most 16 entries).
pub(crate) const MAX_STORED_REFERENCE_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const MAX_DECODED_CACHE_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_DECODED_CACHE_ENTRIES: usize = 16;

impl CachedVoices {
    /// Fast validated cache lookup: LRU-bump a decoded reference whose
    /// identity (profile id, owner, reference hash) and file stamp match.
    /// Pure metadata work — callers must still re-verify after any slow
    /// path (see [`verify_resident_identity`]).
    fn probe(&mut self, key: &str, stamp: &Option<Vec<u64>>) -> Option<ReferenceAudio> {
        let pos = self
            .references
            .iter()
            .position(|(k, identity, _)| k == key && identity == stamp && stamp.is_some())?;
        let entry = self.references.remove(pos).expect("cache index exists");
        let result = entry.2.clone();
        self.references.push_back(entry);
        Some(result)
    }

    /// Publish freshly decoded audio, evicting least-recently-used entries
    /// until both the entry-count and byte budgets hold. Oversized single
    /// references are decoded for the caller but not cached.
    fn publish(&mut self, key: String, stamp: Option<Vec<u64>>, reference: ReferenceAudio) {
        let size = reference.pcm.len() * 4;
        while self.references.len() >= MAX_DECODED_CACHE_ENTRIES
            || self
                .references
                .iter()
                .map(|(_, _, r)| r.pcm.len() * 4)
                .sum::<usize>()
                + size
                > MAX_DECODED_CACHE_BYTES
        {
            if self.references.pop_front().is_none() {
                break;
            }
        }
        self.references.retain(|(old, _, _)| old != &key);
        if size <= MAX_DECODED_CACHE_BYTES {
            self.references.push_back((key, stamp, reference));
        }
    }
}

/// Refresh the resident manifest when the directory or manifest stamp
/// changed, dropping decoded entries that the new manifest may have
/// invalidated. Callers must hold the voices lock.
fn ensure_resident_store(cached: &mut CachedVoices, dir: &str) -> Result<(), AppError> {
    let manifest = VoiceProfileStore::manifest_path(std::path::Path::new(dir));
    let stamp = file_stamp(&manifest);
    if cached.store.is_none() || cached.dir != dir || cached.stamp != stamp {
        cached.store = Some(
            VoiceProfileStore::open(std::path::PathBuf::from(dir)).map_err(AppError::internal)?,
        );
        cached.dir = dir.to_string();
        cached.stamp = stamp;
        cached.references.clear();
    }
    Ok(())
}

/// Re-authorize a profile against the resident store after any slow path
/// (disk read, decode): a deletion, owner change, or reference replacement
/// since the first check must never serve stale or foreign audio.
fn verify_resident_identity(
    store: &VoiceProfileStore,
    profile: &sonicboom::tts::voices::VoiceProfile,
) -> Result<(), AppError> {
    if store.get(&profile.id).is_some_and(|p| {
        p.reference_hash == profile.reference_hash && p.owner_token_hash == profile.owner_token_hash
    }) {
        Ok(())
    } else {
        Err(AppError::NotFound("unknown voice profile".into()))
    }
}
fn file_stamp(path: &std::path::Path) -> Option<Vec<u64>> {
    let m = std::fs::metadata(path).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some(vec![
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime() as u64,
            m.mtime_nsec() as u64,
            m.ctime() as u64,
            m.ctime_nsec() as u64,
        ])
    }
    #[cfg(not(unix))]
    {
        Some(vec![
            m.len(),
            m.modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_nanos() as u64,
        ])
    }
}
pub(crate) async fn with_store<T: Send + 'static>(
    state: &crate::AppState,
    work: impl FnOnce(&mut VoiceProfileStore) -> Result<T, AppError> + Send + 'static,
) -> Result<T, AppError> {
    let dir = state.config.get().await.voices.store_dir.clone();
    let cache = state.inference_gate.voices.clone();
    let preparation_lease = super::jobs::preparation_lease();
    let reference_lease = super::jobs::reference_lease();
    tokio::task::spawn_blocking(move || {
        let _preparation_lease = preparation_lease;
        let _reference_lease = reference_lease;
        let mut cached = cache
            .lock()
            .map_err(|_| AppError::internal("voice store lock poisoned"))?;
        std::fs::create_dir_all(&dir).map_err(|e| AppError::internal(e.to_string()))?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let lock = options
            .open(std::path::Path::new(&dir).join(".profiles.lock"))
            .map_err(|e| AppError::internal(e.to_string()))?;
        let started = std::time::Instant::now();
        loop {
            match lock.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock)
                    if started.elapsed() < std::time::Duration::from_secs(2) =>
                {
                    std::thread::sleep(std::time::Duration::from_millis(10))
                }
                Err(_) => {
                    return Err(AppError::ServiceUnavailable(
                        "voice store is busy or unavailable".into(),
                    ));
                }
            }
        }
        ensure_resident_store(&mut cached, &dir)?;
        let manifest = VoiceProfileStore::manifest_path(std::path::Path::new(&dir));
        let result = work(cached.store.as_mut().expect("store initialized"));
        cached.stamp = file_stamp(&manifest);
        result
    })
    .await
    .map_err(|e| AppError::internal(format!("voice store task failed: {e}")))?
}
/// Outcome of the locked lookup stage: either a validated decoded-cache hit
/// or the file path and stamp to read and decode without holding the lock.
enum StageResult {
    Hit(ReferenceAudio),
    Miss {
        path: std::path::PathBuf,
        stamp: Option<Vec<u64>>,
    },
}

/// Authorization is checked against resident metadata before looking in the
/// decoded cache. The ID, owner and reference hash scope the cache identity.
pub(crate) async fn owned_reference(
    state: &crate::AppState,
    id: String,
    owner: String,
) -> Result<(sonicboom::tts::voices::VoiceProfile, ReferenceAudio), AppError> {
    let profile = with_store(state, move |store| {
        store
            .get(&id)
            .filter(|p| p.owner_token_hash.as_deref() == Some(owner.as_str()))
            .cloned()
            .ok_or_else(|| AppError::NotFound("unknown voice profile".into()))
    })
    .await?;
    let cache = state.inference_gate.voices.clone();
    let profile_copy = profile.clone();
    let store_dir = state.config.get().await.voices.store_dir.clone();
    let key = format!(
        "{}:{}:{}",
        profile_copy.id,
        profile_copy.owner_token_hash.as_deref().unwrap_or(""),
        profile_copy.reference_hash
    );
    // Stage 1 (lock held, metadata only): refresh a stale resident store,
    // re-authorize against it, and probe the decoded cache. The lock is
    // released before any disk IO or decoding, so parallel reads no longer
    // serialize on the slow path.
    let staged = {
        let cache = cache.clone();
        let profile_copy = profile_copy.clone();
        let key = key.clone();
        let store_dir = store_dir.clone();
        let preparation_lease = super::jobs::preparation_lease();
        let reference_lease = super::jobs::reference_lease();
        tokio::task::spawn_blocking(move || {
            let _preparation_lease = preparation_lease;
            let _reference_lease = reference_lease;
            let mut cached = cache
                .lock()
                .map_err(|_| AppError::internal("voice cache lock poisoned"))?;
            // Recheck after acquiring the lock: deletion or store
            // reconfiguration may have occurred since the first check.
            ensure_resident_store(&mut cached, &store_dir)?;
            let store = cached
                .store
                .as_ref()
                .ok_or_else(|| AppError::NotFound("unknown voice profile".into()))?;
            verify_resident_identity(store, &profile_copy)?;
            let path = store
                .reference_path(&profile_copy.id)
                .map_err(AppError::internal)?;
            let stamp = file_stamp(&path);
            if let Some(hit) = cached.probe(&key, &stamp) {
                return Ok::<_, AppError>(StageResult::Hit(hit));
            }
            Ok::<_, AppError>(StageResult::Miss { path, stamp })
        })
        .await
        .map_err(|e| AppError::internal(format!("reference lookup task failed: {e}")))?
    }?;
    let (path, stamp) = match staged {
        StageResult::Hit(reference) => return Ok((profile, reference)),
        StageResult::Miss { path, stamp } => (path, stamp),
    };
    // Stage 2 (no lock): bounded disk read plus WAV decode.
    if std::fs::metadata(&path)
        .map_err(|e| AppError::internal(e.to_string()))?
        .len()
        > MAX_STORED_REFERENCE_BYTES
    {
        return Err(AppError::PayloadTooLarge(
            "stored reference exceeds 16 MiB bound".into(),
        ));
    }
    let (pcm, sample_rate) = {
        let bytes =
            std::fs::read(&path).map_err(|_| AppError::internal("reference audio unavailable"))?;
        decode_wav_bytes(&bytes)?
    };
    if file_stamp(&path) != stamp {
        return Err(AppError::ServiceUnavailable(
            "reference recording changed while decoding; retry".into(),
        ));
    }
    let reference = ReferenceAudio { pcm, sample_rate };
    // Stage 3 (lock held, metadata only): re-verify identity and stamp after
    // the slow path — a deletion, owner change, reference replacement, or
    // store reconfiguration mid-decode must never publish stale audio — then
    // re-probe (a concurrent request may have published first) and publish.
    let preparation_lease = super::jobs::preparation_lease();
    let reference_lease = super::jobs::reference_lease();
    let reference = tokio::task::spawn_blocking(move || {
        let _preparation_lease = preparation_lease;
        let _reference_lease = reference_lease;
        let mut cached = cache
            .lock()
            .map_err(|_| AppError::internal("voice cache lock poisoned"))?;
        ensure_resident_store(&mut cached, &store_dir)?;
        let store = cached
            .store
            .as_ref()
            .ok_or_else(|| AppError::NotFound("unknown voice profile".into()))?;
        verify_resident_identity(store, &profile_copy)?;
        if file_stamp(
            &store
                .reference_path(&profile_copy.id)
                .map_err(AppError::internal)?,
        ) != stamp
        {
            return Err(AppError::ServiceUnavailable(
                "reference recording changed while decoding; retry".into(),
            ));
        }
        if let Some(hit) = cached.probe(&key, &stamp) {
            return Ok(hit);
        }
        cached.publish(key, stamp, reference.clone());
        Ok(reference)
    })
    .await
    .map_err(|e| AppError::internal(format!("reference publish task failed: {e}")))??;
    Ok((profile, reference))
}

/// Public profile metadata. Ownership hashes, transcript and consent are
/// intentionally absent from this response DTO.
#[derive(serde::Serialize)]
struct VoiceSummary {
    id: String,
    name: String,
    provider: String,
    model: String,
    model_revision: String,
    language: String,
    clone_mode: String,
    reference_duration_secs: f64,
    reference_sample_rate: u32,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

impl From<&sonicboom::tts::voices::VoiceProfile> for VoiceSummary {
    fn from(p: &sonicboom::tts::voices::VoiceProfile) -> Self {
        Self {
            id: p.id.clone(),
            name: p.name.clone(),
            provider: p.provider.clone(),
            model: p.model.clone(),
            model_revision: p.model_revision.clone(),
            language: p.language.clone(),
            clone_mode: p.clone_mode.clone(),
            reference_duration_secs: p.reference_duration_secs,
            reference_sample_rate: p.reference_sample_rate,
            created_at: p.created_at,
            updated_at: p.updated_at,
        }
    }
}

fn voice_summary(p: &sonicboom::tts::voices::VoiceProfile) -> serde_json::Value {
    // Only fixed, serializable DTO fields; no private profile serialization.
    serde_json::to_value(VoiceSummary::from(p)).expect("voice summary is serializable")
}

/// Authenticated, current limits; provider modes come from the live registry.
pub async fn get_constraints(
    _token: AuthenticatedToken,
    State(state): State<crate::AppState>,
) -> axum::Json<serde_json::Value> {
    let registry = build_registry_for_state(&state).await;
    let config = state.config.get().await;
    let mut providers = serde_json::Map::new();
    for id in registry.provider_ids() {
        if let Some(provider) = registry.get(&id) {
            let cap = provider.capabilities();
            if cap.requires_reference_audio {
                let transcript_modes: Vec<_> = cap
                    .clone_modes
                    .iter()
                    .filter(|mode| {
                        cap.requires_reference_transcript
                            || CloneMode::parse(mode).is_some_and(CloneMode::requires_transcript)
                    })
                    .collect();
                providers.insert(id, serde_json::json!({
                    "duration_seconds":[REFERENCE_MIN_SECS,REFERENCE_MAX_SECS], "clone_modes":cap.clone_modes,
                    "models":cap.models,"languages":cap.languages,
                    "transcript_required_for":transcript_modes
                }));
            }
        }
    }
    axum::Json(serde_json::json!({"version":1,
        "revision":config.revision, "max_upload_bytes":config.voices.max_upload_bytes,
        "min_sample_rate":REFERENCE_MIN_RATE,"max_sample_rate":REFERENCE_MAX_RATE,"max_channels":REFERENCE_MAX_CHANNELS,
        "name_max_chars":NAME_MAX_CHARS,"transcript_max_chars":TRANSCRIPT_MAX_CHARS,
        "recommended_audio_seconds":[5,10],"providers":providers}))
}

/// `POST /api/voices/clones?name=..&provider=..&model=..&language=..`: upload
/// a reference WAV (`Content-Type: audio/wav`) and create a voice profile.
pub async fn post_clones(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    request: axum::extract::Request,
) -> Result<impl IntoResponse, AppError> {
    check_rate_limit(&state, &token)?;
    super::jobs::with_admission(
        &state,
        super::jobs::with_reference_memory(
            &state,
            post_clone_upload(token, State(state.clone()), request),
        ),
    )
    .await
}

async fn post_clone_upload(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    request: axum::extract::Request,
) -> Result<impl IntoResponse, AppError> {
    let (parts, body) = request.into_parts();
    let mut headers = parts.headers.clone();
    let multipart = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("multipart/form-data"));
    let limit = state.config.get().await.voices.max_upload_bytes;
    // Bound the complete envelope before multipart parsing or allocation.
    let bytes = axum::body::to_bytes(body, limit + 65_536)
        .await
        .map_err(|_| {
            AppError::PayloadTooLarge("voice upload exceeds its aggregate bound".into())
        })?;
    let (query, body) = if multipart {
        let request = axum::extract::Request::from_parts(parts, Body::from(bytes));
        let mut form = Multipart::from_request(request, &state)
            .await
            .map_err(|_| AppError::BadRequest("invalid multipart upload".into()))?;
        let mut metadata = None;
        let mut audio = None;
        while let Some(field) = form
            .next_field()
            .await
            .map_err(|_| AppError::BadRequest("invalid multipart field".into()))?
        {
            let name = field.name().unwrap_or("").to_string();
            if name == "file"
                && !matches!(
                    field.content_type(),
                    None | Some("audio/wav" | "audio/x-wav" | "application/octet-stream")
                )
            {
                return Err(AppError::UnsupportedMediaType(
                    "reference file part must contain WAV audio".into(),
                ));
            }
            let value = field
                .bytes()
                .await
                .map_err(|_| AppError::BadRequest("invalid multipart data".into()))?;
            match name.as_str() {
                "metadata" if metadata.is_none() && value.len() <= 60_000 => {
                    let document: serde_json::Value = serde_json::from_slice(&value)
                        .map_err(|_| AppError::BadRequest("invalid clone metadata".into()))?;
                    let object = document.as_object().ok_or_else(|| {
                        AppError::BadRequest("clone metadata must be an object".into())
                    })?;
                    if object.keys().any(|key| {
                        ![
                            "name",
                            "provider",
                            "model",
                            "language",
                            "mode",
                            "transcript",
                            "consent_basis",
                            "consent_note",
                            "expected_revision",
                        ]
                        .contains(&key.as_str())
                    }) {
                        return Err(AppError::BadRequest(
                            "unexpected clone metadata field".into(),
                        ));
                    }
                    // Deserialize the original bytes as well: duplicate JSON keys
                    // remain errors rather than being collapsed by Value.
                    metadata = Some(
                        serde_json::from_slice::<CloneQuery>(&value)
                            .map_err(|_| AppError::BadRequest("invalid clone metadata".into()))?,
                    );
                }
                "file" if audio.is_none() => {
                    audio = Some(value);
                }
                _ => {
                    return Err(AppError::BadRequest(
                        "unexpected, duplicate or oversized multipart field".into(),
                    ));
                }
            }
        }
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("audio/wav"),
        );
        (
            metadata.ok_or_else(|| AppError::BadRequest("metadata is required".into()))?,
            Body::from(audio.ok_or_else(|| AppError::BadRequest("file is required".into()))?),
        )
    } else {
        // Deprecated raw-WAV/query protocol retained for existing clients.
        let query = Query::<CloneQuery>::try_from_uri(&parts.uri)
            .map_err(|_| AppError::BadRequest("invalid clone parameters".into()))?
            .0;
        (query, Body::from(bytes))
    };
    post_clones_inner(token, State(state), Query(query), headers, body).await
}

async fn post_clones_inner(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<CloneQuery>,
    headers: axum::http::HeaderMap,
    body: Body,
) -> Result<impl IntoResponse, AppError> {
    let config = state.config.get().await;
    if query
        .expected_revision
        .is_some_and(|revision| revision != config.revision)
    {
        return Err(AppError::Conflict(
            "Voice constraints changed. Refresh the selection and retry with the preserved draft."
                .into(),
        ));
    }
    let limit = config.voices.max_upload_bytes + 1;
    let body = axum::body::to_bytes(body, limit).await.map_err(|_| {
        AppError::PayloadTooLarge(format!(
            "reference upload exceeds the configured limit of {} bytes",
            config.voices.max_upload_bytes
        ))
    })?;
    if body.len() > config.voices.max_upload_bytes {
        return Err(AppError::PayloadTooLarge(format!(
            "reference upload is {} bytes; the configured limit is {} bytes",
            body.len(),
            config.voices.max_upload_bytes
        )));
    }
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if content_type != "audio/wav" && content_type != "audio/x-wav" {
        return Err(AppError::UnsupportedMediaType(format!(
            "unsupported reference media '{content_type}' \
             (expected Content-Type: audio/wav with WAV contents; \
             FLAC/MP3 uploads are not supported by this build)"
        )));
    }
    if query.name.trim().is_empty() || query.name.chars().count() > NAME_MAX_CHARS {
        return Err(AppError::field(
            "name",
            "invalid_length",
            "profile name must be 1-64 characters",
            Some(serde_json::json!({"min":1,"max":NAME_MAX_CHARS})),
            Some(serde_json::json!(query.name.chars().count())),
        ));
    }
    let provider = query.provider.trim().to_ascii_lowercase();
    // The Candle provider is gone: Qwen cloning means the GGUF/ONNX backend.
    let cloning = provider == "pocket-tts"
        || (cfg!(feature = "qwen-gguf") && provider == "qwen3-gguf")
        || (cfg!(feature = "candle-qwen") && provider == "qwen3-candle");
    if !cloning {
        return Err(AppError::BadRequest(format!(
            "unsupported provider '{}' for voice cloning \
             (expected one of: {}; supertonic uses built-in style voices)",
            query.provider,
            if cfg!(feature = "qwen-gguf") || cfg!(feature = "candle-qwen") {
                "qwen3-gguf, qwen3-candle, pocket-tts"
            } else {
                "pocket-tts"
            }
        )));
    }
    let mode = parse_clone_mode(&query.mode)?;
    if provider == "pocket-tts" && mode != CloneMode::AudioPrompt {
        return Err(AppError::BadRequest(
            "pocket-tts supports only audio_prompt mode (reference WAV, no transcript needed)"
                .to_string(),
        ));
    }
    let transcript = query
        .transcript
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty());
    if transcript.is_some_and(|t| t.chars().count() > TRANSCRIPT_MAX_CHARS) {
        return Err(AppError::field(
            "transcript",
            "too_long",
            "reference transcript exceeds its bound",
            Some(serde_json::json!(TRANSCRIPT_MAX_CHARS)),
            Some(serde_json::json!(
                transcript.map(|text| text.chars().count()).unwrap_or(0)
            )),
        ));
    }
    if mode.requires_transcript() && transcript.is_none() {
        return Err(AppError::field(
            "transcript",
            "required",
            "ICL mode requires the exact reference transcript",
            None,
            None,
        ));
    }
    if mode == CloneMode::XVector && transcript.is_some() {
        return Err(AppError::BadRequest(
            "x_vector mode does not accept a transcript \
             (use icl mode for transcript-conditioned cloning)"
                .to_string(),
        ));
    }
    if query.consent_basis.trim().is_empty() {
        return Err(AppError::field(
            "consent_basis",
            "required",
            "consent_basis is required (e.g. owner, licensed, authorized)",
            None,
            None,
        ));
    }
    // Validate model/language against provider capabilities (no silent
    // fallback to another provider's pipeline).
    let registry = build_registry_for_state(&state).await;
    let capabilities = registry
        .get(&provider)
        .map(|p| p.capabilities())
        .ok_or_else(|| AppError::internal("cloning provider not registered"))?;
    if !capabilities
        .clone_modes
        .iter()
        .any(|value| value == mode.as_str())
    {
        return Err(AppError::BadRequest(
            "clone mode is not supported by the selected provider".into(),
        ));
    }
    if !capabilities.models.iter().any(|m| m == &query.model) {
        return Err(AppError::BadRequest(format!(
            "unsupported model '{}' for provider '{provider}' (expected one of: {})",
            query.model,
            capabilities.models.join(", ")
        )));
    }
    if !capabilities.languages.iter().any(|l| l == &query.language) {
        return Err(AppError::BadRequest(format!(
            "provider '{provider}' does not support language '{}' (supported: {})",
            query.language,
            capabilities.languages.join(", ")
        )));
    }
    if provider == "pocket-tts" {
        sonicboom::tts::providers::pocket_tts::PocketTtsProvider::check_model_language(
            &query.model,
            &query.language,
        )
        .map_err(provider_error_to_app)?;
    }
    let preparation_lease = super::jobs::preparation_lease();
    let reference_lease = super::jobs::reference_lease();
    let (pcm, rate, health, warnings, defects, guidance) = tokio::task::spawn_blocking(move || {
        let _preparation_lease = preparation_lease;
        let _reference_lease = reference_lease;
        let (pcm, rate) = decode_wav_bytes(&body)?;
        let channels = hound::WavReader::new(std::io::Cursor::new(&body))
            .map_err(|_| AppError::BadRequest("reference WAV header is invalid".into()))?
            .spec()
            .channels;
        let (health, warnings, defects, guidance) = reference_health(&pcm, rate, channels);
        Ok::<_, AppError>((pcm, rate, health, warnings, defects, guidance))
    })
    .await
    .map_err(|e| AppError::internal(e.to_string()))??;
    match provider.as_str() {
        #[cfg(feature = "qwen-gguf")]
        "qwen3-gguf" => {
            sonicboom::tts::providers::qwen_gguf::provider::QwenGgufProvider::validate_reference(
                &pcm, rate,
            )
            .map_err(provider_error_to_app)?
        }
        "pocket-tts" => {
            sonicboom::tts::providers::pocket_tts::PocketTtsProvider::validate_reference(&pcm, rate)
                .map_err(provider_error_to_app)?
        }
        "qwen3-candle" => {
            // Same bounds as a usable clone reference: 1-30 s of audio.
            let secs = pcm.len() as f64 / f64::from(rate.max(1));
            if !(1.0..=30.0).contains(&secs) {
                return Err(AppError::BadRequest(format!(
                    "reference audio must be 1-30 seconds long (got {secs:.1}s)"
                )));
            }
        }
        _ => {}
    }
    if state.config.get().await.revision != config.revision {
        return Err(AppError::Conflict(
            "Configuration changed while preparing the reference. Refresh and retry.".into(),
        ));
    }
    let input = NewVoiceProfile {
        name: query.name,
        provider: provider.clone(),
        model: query.model.clone(),
        model_revision: config_revision_for(&config, &provider),
        language: query.language.clone(),
        clone_mode: mode.as_str().to_string(),
        reference_pcm: pcm,
        reference_sample_rate: rate,
        transcript: transcript.map(str::to_string),
        owner_token_hash: Some(token.rate_limit_key()),
        consent_basis: query.consent_basis,
        consent_note: query.consent_note.unwrap_or_default(),
    };
    let max_profiles = config.voices.max_profiles;
    let profile = with_store(&state, move |store| {
        if store.list().len() >= max_profiles {
            return Err(AppError::TooManyRequests(
                "Voice profile capacity reached; delete unused profiles".into(),
            ));
        }
        store.create(input).map_err(AppError::BadRequest)
    })
    .await?;
    let mut response = voice_summary(&profile);
    response["revision"] = serde_json::json!(config.revision);
    response["reference_health"] = health;
    response["warnings"] = serde_json::json!(warnings);
    response["defects"] = serde_json::json!(defects);
    response["guidance"] = serde_json::json!(guidance);
    let body = serde_json::to_string(&response).map_err(|e| AppError::internal(e.to_string()))?;
    Ok((
        StatusCode::CREATED,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    ))
}

pub(crate) fn config_revision_for(
    config: &crate::config::model::AppConfig,
    provider: &str,
) -> String {
    match provider {
        #[cfg(feature = "qwen-gguf")]
        "qwen3-gguf" => sonicboom::tts::providers::qwen_gguf::models::REVISION.to_string(),
        "pocket-tts" => config.providers.pocket_tts.revision.clone(),
        // Candle bundles are operator-managed directories, not pinned downloads.
        "qwen3-candle" => "candle-safetensors".to_string(),
        // Retired providers (GPT-SoVITS) and anything unknown: a revision
        // that matches nothing, so legacy profiles fail closed with an
        // explicit retired/incompatible error instead of being reused.
        _ => "retired:unknown-provider".to_string(),
    }
}

/// `GET /api/voices`: list the caller's own profiles (private metadata
/// requires authorization; unauthenticated callers never reach here).
pub async fn get_voices(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
) -> Result<impl IntoResponse, AppError> {
    let owner = token.rate_limit_key();
    let profiles = with_store(&state, move |store| {
        Ok(store
            .list()
            .into_iter()
            .filter(|p| p.owner_token_hash.as_deref() == Some(owner.as_str()))
            .map(voice_summary)
            .collect::<Vec<_>>())
    })
    .await?;
    let body = serde_json::json!({ "object": "list", "data": profiles }).to_string();
    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    ))
}

/// `GET /api/voices/{id}`: one profile, or `404` (foreign/missing IDs are
/// indistinguishable to avoid existence disclosure).
pub async fn get_voice(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let owner = token.rate_limit_key();
    let profile = with_store(&state, move |store| {
        Ok(store
            .get(id.trim())
            .filter(|p| p.owner_token_hash.as_deref() == Some(owner.as_str()))
            .cloned())
    })
    .await?;
    match profile.as_ref() {
        Some(profile)
            if profile.owner_token_hash.as_deref() == Some(token.rate_limit_key().as_str()) =>
        {
            let mut detail = voice_summary(profile);
            detail["transcript"] = serde_json::json!(profile.transcript);
            let body = detail.to_string();
            Ok((
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                body,
            ))
        }
        _ => Err(AppError::NotFound("unknown voice profile".to_string())),
    }
}

/// `DELETE /api/voices/{id}`: delete a profile and its reference audio.
pub async fn delete_voice(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let owner = token.rate_limit_key();
    let delete_id = id.clone();
    let provider = with_store(&state, move |store| {
        if !store
            .get(delete_id.trim())
            .is_some_and(|p| p.owner_token_hash.as_deref() == Some(owner.as_str()))
        {
            return Err(AppError::NotFound("unknown voice profile".into()));
        }
        let provider = store
            .get(delete_id.trim())
            .expect("owned profile exists")
            .provider
            .clone();
        store.delete(delete_id.trim()).map_err(AppError::internal)?;
        Ok(provider)
    })
    .await?;
    match provider.as_str() {
        #[cfg(feature = "qwen-gguf")]
        "qwen3-gguf" => sonicboom::tts::providers::TtsProvider::invalidate_voice(
            state.providers.qwen_gguf.as_ref(),
            id.trim(),
        ),
        "pocket-tts" => sonicboom::tts::providers::TtsProvider::invalidate_voice(
            state.providers.pocket.as_ref(),
            id.trim(),
        ),
        _ => {}
    }
    state
        .inference_gate
        .voices
        .lock()
        .map_err(|_| AppError::internal("voice cache lock poisoned"))?
        .references
        .retain(|(key, _, _)| !key.starts_with(&format!("{}:", id.trim())));
    let body = serde_json::json!({ "deleted": id.trim() }).to_string();
    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    ))
}

#[derive(Deserialize, Default)]
pub struct PreviewQuery {
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub sampling_preset: Option<String>,
    #[serde(default)]
    pub seed: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub talker_temperature: Option<f32>,
    #[serde(default)]
    pub talker_top_k: Option<usize>,
    #[serde(default)]
    pub talker_top_p: Option<f32>,
    #[serde(default)]
    pub talker_repetition_penalty: Option<f32>,
    #[serde(default)]
    pub predictor_temperature: Option<f32>,
    #[serde(default)]
    pub predictor_top_k: Option<usize>,
    #[serde(default)]
    pub predictor_top_p: Option<f32>,
    #[serde(default)]
    pub predictor_repetition_penalty: Option<f32>,
    /// Text normalization mode: `auto` or `off` (absent = server default).
    #[serde(default)]
    pub normalization: Option<String>,
    /// Pronunciation locale, independent of the profile's base language.
    /// Absent or `auto` selects the language default.
    #[serde(default)]
    pub locale: Option<String>,
    /// Reading style: `precise`, `natural` or `digits` (absent = server
    /// default).
    #[serde(default)]
    pub reading_style: Option<String>,
}

impl PreviewQuery {
    /// The request's normalization options.
    pub(crate) fn speech_options(&self) -> super::speech_prep::SpeechOptions {
        super::speech_prep::SpeechOptions {
            normalization: self.normalization.clone(),
            locale: self.locale.clone(),
            reading_style: self.reading_style.clone(),
        }
    }

    /// Collect the optional sampling/quality overrides (unparsed).
    pub(crate) fn sampling_input(&self) -> super::sampling::SamplingInput {
        super::sampling::SamplingInput {
            preset: self.sampling_preset.clone(),
            seed: self.seed.clone(),
            quality: self.quality.clone(),
            talker_temperature: self.talker_temperature,
            talker_top_k: self.talker_top_k,
            talker_top_p: self.talker_top_p,
            talker_repetition_penalty: self.talker_repetition_penalty,
            predictor_temperature: self.predictor_temperature,
            predictor_top_k: self.predictor_top_k,
            predictor_top_p: self.predictor_top_p,
            predictor_repetition_penalty: self.predictor_repetition_penalty,
        }
    }
}

/// `POST /api/voices/{id}/preview`: synthesize `text/plain` with the
/// profile's provider/model/mode/reference. Shares the bounded-inference
/// gate, rate limits, body limits and timeout semantics.
pub async fn post_preview(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Path(id): Path<String>,
    Query(query): Query<PreviewQuery>,
    body: String,
) -> Result<impl IntoResponse, AppError> {
    super::jobs::with_admission(
        &state,
        super::jobs::with_reference_memory(
            &state,
            post_preview_inner(
                token,
                State(state.clone()),
                Path(id),
                Query(query),
                body,
                None,
            ),
        ),
    )
    .await
}

/// Preview semantics for text that was already prepared (async jobs
/// prepare at enqueue). `query.language` must be the language `speech`
/// was prepared for.
pub(crate) async fn post_preview_prepared(
    token: AuthenticatedToken,
    state: crate::AppState,
    id: String,
    query: PreviewQuery,
    body: String,
    speech: super::speech_prep::SpeechText,
) -> Result<axum::response::Response, AppError> {
    super::jobs::with_admission(
        &state,
        super::jobs::with_reference_memory(
            &state,
            post_preview_inner(
                token,
                State(state.clone()),
                Path(id),
                Query(query),
                body,
                Some(speech),
            ),
        ),
    )
    .await
}

async fn post_preview_inner(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Path(id): Path<String>,
    Query(query): Query<PreviewQuery>,
    body: String,
    prepared: Option<super::speech_prep::SpeechText>,
) -> Result<axum::response::Response, AppError> {
    check_rate_limit(&state, &token)?;
    let config = state.config.get().await;
    let text = body.trim().to_string();
    if text.is_empty() {
        return Err(AppError::BadRequest("text cannot be empty".to_string()));
    }
    if text.chars().count() > config.tts.max_text_length {
        return Err(AppError::BadRequest(format!(
            "text exceeds {} characters",
            config.tts.max_text_length
        )));
    }
    let (profile, reference) =
        owned_reference(&state, id.trim().to_string(), token.rate_limit_key()).await?;
    // Legacy GPT-SoVITS profiles are preserved on disk but unusable: their
    // conditioning is not Pocket conditioning and must never be silently
    // reinterpreted. The owner can recreate them as Pocket profiles from
    // the stored reference audio (see GET /api/voices/{id}).
    if profile.provider == "gpt-sovits" {
        return Err(AppError::BadRequest(
            "voice profile uses the retired GPT-SoVITS provider and is unavailable; recreate it \
             as a pocket-tts profile (mode audio_prompt) from its stored reference audio"
                .into(),
        ));
    }
    if profile.model_revision != config_revision_for(&config, &profile.provider) {
        return Err(AppError::BadRequest(
            "voice profile belongs to an incompatible model revision; create a new profile".into(),
        ));
    }
    let language = query.language.clone().unwrap_or(profile.language.clone());
    // Only the target text is prepared (once); the profile's reference
    // transcript and audio are passed through unchanged.
    let speech = match prepared {
        Some(speech) => speech,
        None => query.speech_options().prepare(&config, &language, &text)?,
    };
    let mode = parse_clone_mode(&profile.clone_mode)?;
    let format = query
        .format
        .as_deref()
        .map(sonicboom::tts::audio::AudioFormat::parse)
        .transpose()
        .map_err(|e| AppError::BadRequest(e.to_string()))?
        .unwrap_or(sonicboom::tts::audio::AudioFormat::Opus);
    let mut request =
        super::speech_prep::provider_request(&profile.model, &profile.id, speech, language);
    request.device = DevicePreference::Auto;
    request.clone_mode = Some(mode);
    request.reference_transcript = profile.transcript.clone();
    request.reference_audio = Some(reference);
    // Optional per-request sampling/quality overrides (validated strictly).
    let (sampling, quality) = query.sampling_input().into_parts()?;
    let registry = build_registry_for_state(&state).await;
    let provider = registry
        .resolve_model(&profile.model)
        .map_err(provider_error_to_app)?;
    // The provider validates ownership: a profile bound to one provider can
    // never be synthesized through another provider's pipeline.
    if provider.id() != profile.provider {
        return Err(AppError::BadRequest(format!(
            "voice profile '{}' belongs to provider '{}' and cannot be used as '{}'",
            profile.id,
            profile.provider,
            provider.id()
        )));
    }
    // Sampling and quality modes are Qwen3-TTS-only: reject them explicitly
    // for other providers rather than silently ignoring a requested setting.
    if (sampling.is_some() || quality.is_some()) && provider.id() != "qwen3-gguf" {
        return Err(AppError::BadRequest(format!(
            "sampling and quality options are not supported by provider '{}' (only qwen3-gguf)",
            provider.id()
        )));
    }
    request.sampling = sampling;
    request.quality = quality;
    let audio = synthesize_provider_bounded(&state, provider, request)
        .await
        .map_err(provider_error_to_app)?;
    let bytes = super::tts::encode_output(audio.samples, audio.sample_rate, format).await?;
    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        Body::from(bytes),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_health_warns_without_modifying_audio_or_logging_content() {
        let clean: Vec<f32> = (0..24000)
            .map(|i| 0.1 * (i as f32 * std::f32::consts::TAU * 440.0 / 24000.0).sin())
            .collect();
        let copy = clean.clone();
        let (health, warnings, defects, guidance) = reference_health(&clean, 24000, 2);
        assert!(
            warnings.is_empty(),
            "clean clip must not warn: {warnings:?}"
        );
        assert!(defects.is_empty());
        assert!(
            guidance.contains(&"reference_short_for_cloning"),
            "1 s is below guidance"
        );
        assert_eq!(health["source_channels"], 2);
        assert_eq!(health["duration_secs"], 1.0);
        assert_eq!(clean, copy, "audio must never be modified");
        assert!(health.get("pcm").is_none() && health.get("transcript").is_none());
        assert!(
            reference_health(&vec![0.0; 24000], 24000, 1)
                .1
                .contains(&"reference_nearly_silent")
        );
        let clipped: Vec<f32> = (0..24000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!(
            reference_health(&clipped, 24000, 1)
                .1
                .contains(&"reference_heavily_clipped")
        );
        assert!(
            reference_health(&vec![0.2; 24000], 24000, 1)
                .1
                .contains(&"reference_large_dc_offset")
        );
        // Unusually quiet but non-silent audio is a warning, not a rejection.
        let quiet: Vec<f32> = (0..24000)
            .map(|i| 0.005 * (i as f32 * std::f32::consts::TAU * 300.0 / 24000.0).sin())
            .collect();
        assert!(
            reference_health(&quiet, 24000, 1)
                .1
                .contains(&"reference_unusually_quiet")
        );
        // Confirmed numerical defect: non-finite samples are reported as a
        // defect, never silently dropped.
        let mut non_finite = clean.clone();
        non_finite[5] = f32::NAN;
        assert!(
            reference_health(&non_finite, 24000, 1)
                .2
                .contains(&"reference_non_finite_samples")
        );
    }

    /// Speech-like bursts separated by silence: the profile needed before the
    /// truncation/noise heuristics may fire (a pure tone never matches).
    fn speech_like(silence_level: f32) -> Vec<f32> {
        let mut samples = Vec::new();
        for block in 0..6 {
            for i in 0..2400 {
                samples.push(0.3 * (i as f32 * 300.0 * std::f32::consts::TAU / 24000.0).sin());
            }
            if block != 5 {
                for i in 0..2400 {
                    samples.push(
                        silence_level * (i as f32 * 5000.0 * std::f32::consts::TAU / 24000.0).sin(),
                    );
                }
            }
        }
        samples
    }

    #[test]
    fn reference_health_detects_truncated_speech_and_background_noise() {
        // Ends loudly with no trailing silence → likely truncated mid-speech.
        let truncated = speech_like(0.0);
        assert!(
            reference_health(&truncated, 24000, 1)
                .1
                .contains(&"reference_ends_during_speech")
        );
        // Same structure with a raised noise floor → background contamination.
        let noisy = speech_like(0.02);
        assert!(
            reference_health(&noisy, 24000, 1)
                .1
                .contains(&"reference_possible_background_noise")
        );
    }

    use sonicboom::tts::voices::{NewVoiceProfile, VoiceProfileStore};

    fn decoded(samples: usize, rate: u32) -> ReferenceAudio {
        ReferenceAudio {
            pcm: vec![0.25; samples],
            sample_rate: rate,
        }
    }

    #[test]
    fn probe_returns_only_stamp_matched_entries_in_lru_order() {
        let mut cached = CachedVoices::default();
        let stamp_a = Some(vec![1u64]);
        let stamp_b = Some(vec![2u64]);
        cached.publish("a".into(), stamp_a.clone(), decoded(8, 24_000));
        cached.publish("b".into(), stamp_b.clone(), decoded(8, 24_000));
        // Exact stamp hits and bumps recency.
        assert!(cached.probe("a", &stamp_a).is_some());
        // Wrong stamp, missing stamp, and unknown key all miss.
        assert!(cached.probe("a", &stamp_b).is_none());
        assert!(cached.probe("a", &None).is_none());
        assert!(cached.probe("ghost", &stamp_a).is_none());
        // `a` was bumped, so `b` is now least-recently-used.
        assert_eq!(cached.references.front().unwrap().0, "b");
        assert_eq!(cached.references.back().unwrap().0, "a");
    }

    #[test]
    fn publish_enforces_count_and_byte_budgets() {
        let mut cached = CachedVoices::default();
        for i in 0..MAX_DECODED_CACHE_ENTRIES {
            cached.publish(format!("v{i}"), Some(vec![i as u64]), decoded(4, 24_000));
        }
        assert_eq!(cached.references.len(), MAX_DECODED_CACHE_ENTRIES);
        cached.publish("new".into(), Some(vec![999]), decoded(4, 24_000));
        assert_eq!(cached.references.len(), MAX_DECODED_CACHE_ENTRIES);
        // `v0` was least-recently-used: evicted first.
        assert!(cached.probe("v0", &Some(vec![0])).is_none());
        assert!(cached.probe("new", &Some(vec![999])).is_some());
        // Re-publishing an existing key replaces it instead of duplicating.
        cached.publish("new".into(), Some(vec![1000]), decoded(4, 24_000));
        assert_eq!(
            cached
                .references
                .iter()
                .filter(|(k, _, _)| k == "new")
                .count(),
            1
        );
        // An oversized single reference is served but never cached.
        cached.publish(
            "huge".into(),
            Some(vec![7]),
            decoded(MAX_DECODED_CACHE_BYTES / 4 + 1, 24_000),
        );
        assert!(cached.probe("huge", &Some(vec![7])).is_none());
    }

    fn test_profile_input(owner: Option<&str>) -> NewVoiceProfile {
        NewVoiceProfile {
            name: "test".into(),
            provider: "pocket-tts".into(),
            model: "pocket-tts-en".into(),
            model_revision: "rev".into(),
            language: "en".into(),
            clone_mode: "audio_prompt".into(),
            reference_pcm: vec![0.1; 48_000],
            reference_sample_rate: 24_000,
            transcript: None,
            owner_token_hash: owner.map(str::to_string),
            consent_basis: "owner".into(),
            consent_note: String::new(),
        }
    }

    #[test]
    fn verify_resident_identity_rejects_deletion_and_replacement() {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-voices-verify-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut store = VoiceProfileStore::open(dir.clone()).unwrap();
        let profile = store
            .create(test_profile_input(Some("owner-hash")))
            .unwrap();

        // Exact identity verifies.
        assert!(verify_resident_identity(&store, &profile).is_ok());
        // Unknown ids never verify (404 isolation, never foreign audio).
        let mut ghost = profile.clone();
        ghost.id = uuid::Uuid::new_v4().to_string();
        assert!(verify_resident_identity(&store, &ghost).is_err());
        // A replaced reference hash fails even when id and owner match.
        let mut replaced = profile.clone();
        replaced.reference_hash = "tampered".into();
        assert!(verify_resident_identity(&store, &replaced).is_err());
        // An owner change fails even when id and reference match.
        let mut foreign = profile.clone();
        foreign.owner_token_hash = Some("someone-else".into());
        assert!(verify_resident_identity(&store, &foreign).is_err());
        // Deletion invalidates immediately.
        assert!(store.delete(&profile.id).unwrap());
        assert!(verify_resident_identity(&store, &profile).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ensure_resident_store_refreshes_on_manifest_change() {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-voices-ensure-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_str = dir.to_str().unwrap().to_string();
        let mut cached = CachedVoices::default();
        ensure_resident_store(&mut cached, &dir_str).unwrap();
        assert!(cached.store.is_some());
        // A concurrent mutation (profile created through another handle)
        // changes the manifest stamp: the next ensure reopens and drops
        // decoded entries the new manifest may have invalidated.
        let mut other = VoiceProfileStore::open(dir.clone()).unwrap();
        let profile = other.create(test_profile_input(None)).unwrap();
        cached.publish("stale".into(), Some(vec![1]), decoded(8, 24_000));
        ensure_resident_store(&mut cached, &dir_str).unwrap();
        assert!(cached.references.is_empty());
        assert!(verify_resident_identity(cached.store.as_ref().unwrap(), &profile).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_wav_header_is_rejected_before_sample_allocation() {
        let mut wav = sonicboom::tts::audio::encode_audio(
            &vec![0.1; 24000],
            24000,
            sonicboom::tts::audio::AudioFormat::Wav,
        )
        .unwrap();
        let offset = wav.windows(4).position(|w| w == b"data").unwrap() + 4;
        wav[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_wav_bytes(&wav).is_err());
    }
}
