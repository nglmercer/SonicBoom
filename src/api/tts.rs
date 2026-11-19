#[cfg(feature = "playback")]
use axum::Json;
use axum::{
    body::Body,
    extract::{Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::gate::RunBoundedError;
use super::speech_prep::{SpeechOptions, SpeechText};
use crate::{
    auth::AuthenticatedToken,
    error::AppError,
    tts::{ModelStatus, audio, inference},
};

/// Run model inference with the admission permit owned by the blocking
/// task itself, so an HTTP timeout/cancellation can never release the
/// concurrency slot while inference still runs (see
/// [`crate::api::gate::InferenceGate::run_bounded`]). The text is a
/// [`SpeechText`]: it has been prepared exactly once by
/// [`super::speech_prep::prepare_speech`].
pub(super) async fn synthesize_bounded(
    state: &crate::AppState,
    model_handle: std::sync::Arc<crate::tts::model::ModelHandle>,
    text: SpeechText,
    lang: String,
    voice_name: String,
) -> Result<Vec<f32>, AppError> {
    // Inference settings are read per request so
    // configuration changes apply immediately
    // (spec §46/§47).
    let config = state.config.get().await;
    let inference_steps = config.model.inference_steps;
    let max_chunk_chars = config.tts.max_chunk_chars;
    super::jobs::with_deadline(
        config.inference.generation_deadline_secs,
        state.inference_gate.run_bounded(move || {
            if let Some(c) = sonicboom::tts::control::current() {
                let device = if cfg!(any(feature = "cuda", feature = "rocm", feature = "coreml")) {
                    "onnxruntime_placement_unknown"
                } else {
                    "cpu"
                };
                c.set_context("supertonic", "supertonic-3", &voice_name, device);
            }
            inference::synthesize(
                &model_handle,
                text.as_str(),
                &lang,
                &voice_name,
                inference_steps,
                max_chunk_chars,
            )
        }),
    )
    .await
    .map_err(|e| match e {
        RunBoundedError::Saturated => AppError::TooManyRequests(
            "Server is busy. Too many pending inference requests.".to_string(),
        ),
        RunBoundedError::Cancelled => {
            AppError::ServiceUnavailable("Inference cancelled or deadline exceeded".into())
        }
        RunBoundedError::JoinFailed(detail) => {
            AppError::internal(format!("inference task failed: {detail}"))
        }
    })?
    .map_err(|e| AppError::internal(e.to_string()))
}

#[derive(Deserialize)]
pub struct TtsQuery {
    #[cfg(feature = "playback")]
    pub provider: Option<String>,
    pub voice: Option<String>,
    pub lang: Option<String>,
    /// Output format: "opus", "wav", "mp3", or "flac"
    #[serde(default)]
    pub format: Option<String>,
    /// Play immediately?
    #[cfg(feature = "playback")]
    pub play_now: Option<bool>,
    /// Text normalization mode: `auto` or `off`. Absent uses the server's
    /// configured default (`[tts.normalization]`).
    #[serde(default)]
    pub normalization: Option<String>,
    /// Pronunciation locale (e.g. `es-PE`). Absent or `auto` selects the
    /// speech language's documented default.
    #[serde(default)]
    pub locale: Option<String>,
    /// Reading style for written numbers and times: `precise`, `natural` or
    /// `digits`. Absent uses the server's configured default.
    #[serde(default)]
    pub reading_style: Option<String>,
}

impl TtsQuery {
    /// The request's normalization options.
    pub(crate) fn speech_options(&self) -> SpeechOptions {
        SpeechOptions {
            normalization: self.normalization.clone(),
            locale: self.locale.clone(),
            reading_style: self.reading_style.clone(),
        }
    }
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub status: String,
    pub progress: Option<f32>,
    pub error: Option<String>,
}

pub async fn get_status(State(state): State<crate::AppState>) -> impl IntoResponse {
    let status = state.model_status.read().await;
    let response = match &*status {
        ModelStatus::Idle => StatusResponse {
            status: "idle".to_string(),
            progress: None,
            error: None,
        },
        ModelStatus::Downloading { progress } => StatusResponse {
            status: "downloading".to_string(),
            progress: Some(*progress * 100.0),
            error: None,
        },
        ModelStatus::Loading => StatusResponse {
            status: "loading".to_string(),
            progress: None,
            error: None,
        },
        ModelStatus::Ready(_) => StatusResponse {
            status: "ready".to_string(),
            progress: None,
            error: None,
        },
        ModelStatus::Failed(reason) => {
            // Do not expose internal load failures; they are logged server-side.
            tracing::error!("model status requested while failed: {reason}");
            StatusResponse {
                status: "failed".to_string(),
                progress: None,
                error: Some("Model failed to load.".to_string()),
            }
        }
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&response).unwrap(),
    )
}

/// Run provider synthesis with the admission permit owned by the blocking
/// task itself (same cancellation safety as [`synthesize_bounded`]): an HTTP
/// timeout/cancellation can never release the concurrency slot while
/// inference still runs, and queued jobs are bounded by the shared gate.
///
/// Only requests built from a prepared [`SpeechText`]
/// ([`super::speech_prep::provider_request`]) are accepted: a request
/// without `speech_identity` carries raw text and is refused before any
/// inference.
pub(crate) async fn synthesize_provider_bounded(
    state: &crate::AppState,
    provider: std::sync::Arc<dyn sonicboom::tts::providers::TtsProvider>,
    request: sonicboom::tts::providers::types::ProviderRequest,
) -> Result<sonicboom::tts::providers::types::PcmAudio, sonicboom::tts::providers::ProviderError> {
    use sonicboom::tts::providers::ProviderError;
    if request.speech_identity.is_none() {
        return Err(ProviderError::inference_failed(
            "synthesis text was not prepared",
        ));
    }
    let config = state.config.get().await;
    super::jobs::with_deadline(
        config.inference.generation_deadline_secs,
        state
            .inference_gate
            .run_bounded(move || provider.synthesize(&request)),
    )
    .await
    .map_err(|e| match e {
        RunBoundedError::Saturated => {
            ProviderError::busy("Server is busy. Too many pending inference requests.".to_string())
        }
        RunBoundedError::Cancelled => {
            ProviderError::inference_failed("Inference cancelled or deadline exceeded")
        }
        RunBoundedError::JoinFailed(detail) => {
            ProviderError::inference_failed(format!("inference task failed: {detail}"))
        }
    })?
}

/// Shared pre-inference pipeline: rate limit -> validate -> admission control.
///
/// Token-bucket rejections become [`AppError::RateLimited`] (same `429` JSON
/// shape plus live `Retry-After` / `X-RateLimit-*` headers); inference and
/// playback saturation keep their own headerless `429` messages.
pub(crate) fn check_rate_limit(
    state: &crate::AppState,
    token: &AuthenticatedToken,
) -> Result<(), AppError> {
    if super::jobs::active_control().is_some() {
        return Ok(());
    }
    let decision = state.rate_limiter.check(&token.rate_limit_key());
    if decision.allowed {
        return Ok(());
    }
    Err(AppError::RateLimited {
        message: "Rate limit exceeded. Try again later.".to_string(),
        retry_after_secs: decision.retry_after_secs,
        limit: decision.limit,
        remaining: decision.remaining,
        reset_secs: decision.reset_secs,
    })
}

pub async fn post_tts(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<TtsQuery>,
    body: String,
) -> Result<Response, AppError> {
    super::jobs::with_admission(
        &state,
        post_tts_inner(token, State(state.clone()), Query(query), body, None),
    )
    .await
}

/// `/api/tts` semantics for text that was already prepared (async jobs
/// prepare at enqueue, before computing their idempotency fingerprint).
/// `query.lang` must be the language `speech` was prepared for.
pub(super) async fn post_tts_prepared(
    token: AuthenticatedToken,
    state: crate::AppState,
    query: TtsQuery,
    body: String,
    speech: SpeechText,
) -> Result<Response, AppError> {
    super::jobs::with_admission(
        &state,
        post_tts_inner(
            token,
            State(state.clone()),
            Query(query),
            body,
            Some(speech),
        ),
    )
    .await
}

/// Named-voice providers (Kyutai TTS) for the studio/job path: the voice is a file stem in the
/// provider's voices directory and the provider validates it. `speech` is already prepared.
#[cfg(feature = "playback")]
pub(super) async fn post_named_voice_prepared(
    state: &crate::AppState,
    model: &str,
    voice: &str,
    language: String,
    format: Option<&str>,
    speech: SpeechText,
) -> Result<Response, AppError> {
    // Omitted means Opus; an explicit unknown value is a client error.
    let format = match format {
        None => audio::AudioFormat::Opus,
        Some(raw) => audio::AudioFormat::parse(raw).map_err(|e| AppError::BadRequest(e.to_string()))?,
    };
    let registry = super::providers::build_registry_for_state(state).await;
    let provider = registry
        .resolve_model(model)
        .map_err(super::providers::provider_error_to_app)?;
    let mut request = super::speech_prep::provider_request(model, voice, speech, language);
    request.device = sonicboom::tts::providers::types::DevicePreference::Auto;
    let pcm = synthesize_provider_bounded(state, provider, request)
        .await
        .map_err(super::providers::provider_error_to_app)?;
    let bytes = encode_output(pcm.samples, pcm.sample_rate, format).await?;
    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        Body::from(bytes),
    )
        .into_response())
}

async fn post_tts_inner(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<TtsQuery>,
    body: String,
    prepared: Option<SpeechText>,
) -> Result<Response, AppError> {
    check_rate_limit(&state, &token)?;
    let lang = query.lang.clone().unwrap_or_else(|| "en".to_string());
    let voice = query.voice.clone().unwrap_or_else(|| "M1".to_string());
    // TTS limits are read per request (spec §46).
    let config = state.config.get().await;
    sonicboom::engine::validate_tts_input(
        &body,
        &lang,
        &voice,
        config.model.inference_steps,
        config.tts.max_text_length,
    )
    .map_err(|e| AppError::BadRequest(e.to_string()))?;
    // Text is prepared once, before the model check, so invalid
    // normalization options are a 400 even while the model loads.
    let speech = match prepared {
        Some(speech) => speech,
        None => query
            .speech_options()
            .prepare(&config, &lang, body.trim())?,
    };

    let model_handle = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Ready(handle) => Arc::clone(handle),
            ModelStatus::Downloading { progress } => {
                return Err(AppError::ServiceUnavailable(format!(
                    "Model is downloading ({:.0}% complete).",
                    progress * 100.0
                )));
            }
            ModelStatus::Loading => {
                return Err(AppError::ServiceUnavailable(
                    "Model is loading.".to_string(),
                ));
            }
            ModelStatus::Idle => {
                return Err(AppError::ServiceUnavailable(
                    "Model has not started loading yet.".to_string(),
                ));
            }
            ModelStatus::Failed(reason) => {
                tracing::error!("tts request while model failed: {reason}");
                return Err(AppError::internal("model unavailable"));
            }
        }
    };

    let voice_name = match query.voice {
        Some(ref v) if model_handle.voice_styles.contains_key(v.as_str()) => v.clone(),
        Some(_) => return Err(AppError::BadRequest("unknown Supertonic voice".into())),
        None => model_handle
            .default_voice()
            .ok_or_else(|| AppError::internal("no voice styles available"))?
            .to_string(),
    };
    let sample_rate = model_handle.sample_rate();

    // Determine output format: omitted means Opus, but an explicit
    // unknown value is a client error, never a silent default.
    let format = query
        .format
        .as_deref()
        .map(audio::AudioFormat::parse)
        .transpose()
        .map_err(|e| AppError::BadRequest(e.to_string()))?
        .unwrap_or(audio::AudioFormat::Opus);

    let samples = synthesize_bounded(&state, model_handle, speech, lang, voice_name).await?;

    let audio_bytes = encode_output(samples, sample_rate, format).await?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        Body::from(audio_bytes),
    )
        .into_response())
}

#[cfg(feature = "playback")]
pub async fn post_tts_and_play(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<TtsQuery>,
    body: String,
) -> Result<Response, AppError> {
    super::jobs::with_admission(
        &state,
        post_tts_and_play_inner(token, State(state.clone()), Query(query), body),
    )
    .await
}

#[cfg(feature = "playback")]
async fn post_tts_and_play_inner(
    token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<TtsQuery>,
    body: String,
) -> Result<Response, AppError> {
    check_rate_limit(&state, &token)?;
    let (provider_id, voice) = plugin_identity(query.provider.as_deref(), query.voice.as_deref())?;
    let config = state.config.get().await;
    let lang = query.lang.clone().unwrap_or_else(|| "en".into());
    sonicboom::engine::validate_tts_input(
        &body,
        &lang,
        if provider_id == "supertonic" {
            voice
        } else {
            "M1"
        },
        config.model.inference_steps,
        config.tts.max_text_length,
    )
    .map_err(|e| AppError::BadRequest(e.to_string()))?;
    let play_now = query.play_now.unwrap_or(false);
    let (samples, sample_rate) = if provider_id == "supertonic" {
        let speech = query
            .speech_options()
            .prepare(&config, &lang, body.trim())?;
        let model_handle = super::providers::supertonic_handle(&state)
            .await
            .ok_or_else(|| {
                AppError::ServiceUnavailable("Model is not ready for synthesis.".into())
            })?;
        let voice_name = if model_handle.voice_styles.contains_key(voice) {
            voice.to_string()
        } else if query
            .provider
            .as_deref()
            .is_none_or(|p| p.trim().is_empty())
        {
            model_handle
                .default_voice()
                .ok_or_else(|| AppError::internal("no voice styles available"))?
                .to_string()
        } else {
            return Err(AppError::BadRequest("unknown Supertonic voice".into()));
        };
        playback_manager(&state)?;
        let rate = model_handle.sample_rate();
        (
            synthesize_bounded(&state, model_handle, speech, lang, voice_name).await?,
            rate,
        )
    } else if provider_id == "kyutai-candle" {
        // Named voices (file stems in the provider's voices directory): no profile.
        let speech = query
            .speech_options()
            .prepare(&config, &lang, body.trim())?;
        let registry = super::providers::build_registry_for_state(&state).await;
        let provider = registry.get(provider_id).ok_or_else(|| {
            AppError::ServiceUnavailable("selected provider is not installed in this build".into())
        })?;
        let mut request = super::speech_prep::provider_request(
            sonicboom::tts::providers::catalog::KYUTAI_CANDLE_MODELS[0],
            voice,
            speech,
            lang,
        );
        request.device = sonicboom::tts::providers::types::DevicePreference::Auto;
        playback_manager(&state)?;
        let audio = synthesize_provider_bounded(&state, provider, request)
            .await
            .map_err(super::providers::provider_error_to_app)?;
        (audio.samples, audio.sample_rate)
    } else {
        let (provider, request) = prepare_plugin_clone(
            &state,
            &token,
            provider_id,
            voice,
            body.trim(),
            query.lang.clone(),
            &query.speech_options(),
        )
        .await?;
        playback_manager(&state)?;
        let audio = synthesize_provider_bounded(&state, provider, request)
            .await
            .map_err(super::providers::provider_error_to_app)?;
        (audio.samples, audio.sample_rate)
    };

    let audio_bytes = encode_output(samples, sample_rate, audio::AudioFormat::Wav).await?;
    enqueue_generated_wav(&state, audio_bytes, play_now).await
}

/// Outcome of admitting already-encoded audio into the server playback queue.
#[cfg(feature = "playback")]
#[derive(Debug, Clone)]
pub(crate) struct EnqueueOutcome {
    /// Server playback-queue id. Also the temp-file stem, so a client can
    /// correlate the accepted speech with the playback queue entry.
    pub id: String,
    /// True when playback was started immediately rather than queued. The
    /// device backend cannot confirm audible completion, so callers must not
    /// report either outcome as "heard".
    pub played_now: bool,
}

/// Writes `audio_bytes` to a private temp WAV and admits it to the playback
/// queue (immediately when `play_now`). Shared by the synchronous play path
/// and async jobs so both enforce identical cancellation, ownership and
/// cleanup rules. Never retries: a rejected admission deletes the temp file
/// rather than leaving an orphan, and a cancelled control withholds playback.
#[cfg(feature = "playback")]
pub(crate) async fn enqueue_encoded_wav(
    state: &crate::AppState,
    audio_bytes: &[u8],
    play_now: bool,
) -> Result<EnqueueOutcome, AppError> {
    let config = state.config.get().await;
    let audio_manager = playback_manager(state)?;
    // WAV is shared by every provider for rodio compatibility.

    // Ensure the temp directory exists with restrictive permissions.
    let temp_dir =
        crate::tts::queue::ensure_temp_dir(std::path::Path::new(&config.paths.temp_audio))
            .map_err(|e| AppError::internal(format!("temp audio dir unavailable: {e}")))?;

    // Save to a temp file readable only by the server user (synthesized
    // speech may be private). The filename matches the startup-sweep
    // pattern (`<uuid>.wav`) so leftovers are reaped on restart.
    let id = uuid::Uuid::new_v4().to_string();
    let filename = format!("{id}.wav");
    let path = temp_dir.join(&filename);

    crate::tts::queue::write_temp_wav(&path, audio_bytes)
        .await
        .map_err(|e| AppError::internal(format!("temp audio write failed: {e}")))?;

    // Ownership of the generated file transfers atomically on admission
    // (`temporary: true`): the audio thread registers the file before it
    // can complete, be stopped, or be replaced, so no interleaving can
    // orphan it and no separate `RegisterTemp` round-trip is needed. Every
    // failure path below deletes the newly generated file, since a rejected
    // or failed admission is never registered by the audio thread.
    if super::jobs::active_control().is_some_and(|c| c.stopped()) {
        let _ = tokio::fs::remove_file(&path).await;
        return Err(AppError::ServiceUnavailable(
            "Inference cancelled or deadline exceeded".into(),
        ));
    }
    use crate::tts::queue::AudioManagerError;
    let queued = if play_now {
        // Acknowledged start: `Ok` means playback is audible, anything else
        // maps to a precise status — never "Playing immediately" on failure.
        audio_manager
            .play_now_temp(id.clone(), path.clone())
            .await
            .map_err(|e| match e {
                AudioManagerError::Unavailable => {
                    AppError::ServiceUnavailable("Audio playback system unavailable.".to_string())
                }
                AudioManagerError::OutputUnavailable => {
                    AppError::ServiceUnavailable("Audio output unavailable.".to_string())
                }
                // The WAV was just synthesized by this server: undecodable
                // output is a server bug, never a client error.
                AudioManagerError::InvalidAudio => {
                    AppError::internal("synthesized playback audio undecodable")
                }
                AudioManagerError::QueueFull
                | AudioManagerError::UnknownDevice
                | AudioManagerError::DeviceError => {
                    AppError::internal(format!("unexpected play_now failure: {e}"))
                }
            })
    } else {
        audio_manager
            .add_to_queue_temp(id.clone(), path.clone())
            .await
            .map_err(|e| match e {
                AudioManagerError::QueueFull => AppError::TooManyRequests(
                    "Playback queue is full. Try again later.".to_string(),
                ),
                AudioManagerError::Unavailable => {
                    AppError::ServiceUnavailable("Audio playback system unavailable.".to_string())
                }
                AudioManagerError::OutputUnavailable => {
                    AppError::ServiceUnavailable("Audio output unavailable.".to_string())
                }
                AudioManagerError::InvalidAudio
                | AudioManagerError::UnknownDevice
                | AudioManagerError::DeviceError => {
                    AppError::internal(format!("unexpected enqueue failure: {e}"))
                }
            })
    };
    if let Err(error) = queued {
        let _ = tokio::fs::remove_file(&path).await;
        return Err(error);
    }

    Ok(EnqueueOutcome {
        id,
        played_now: play_now,
    })
}

/// Synchronous `/api/tts/play` response wrapper over [`enqueue_encoded_wav`].
#[cfg(feature = "playback")]
async fn enqueue_generated_wav(
    state: &crate::AppState,
    audio_bytes: Vec<u8>,
    play_now: bool,
) -> Result<Response, AppError> {
    let outcome = enqueue_encoded_wav(state, &audio_bytes, play_now).await?;
    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "success": true,
            "message": if outcome.played_now { "Playing immediately" } else { "Added to queue" },
            "id": outcome.id
        })),
    )
        .into_response())
}

/// Audio codecs execute off the async executor. Results cancelled during
/// encoding are withheld just like results cancelled during generation.
pub(crate) async fn encode_output(
    samples: Vec<f32>,
    rate: u32,
    format: audio::AudioFormat,
) -> Result<Vec<u8>, AppError> {
    let control = super::jobs::active_control();
    if let Some(c) = &control
        && !c.update("encoding", c.snapshot().frames)
    {
        return Err(AppError::ServiceUnavailable(
            "Inference cancelled or deadline exceeded".into(),
        ));
    }
    let lease = super::jobs::preparation_lease();
    let bytes = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        audio::encode_audio(&samples, rate, format)
    })
    .await
    .map_err(|e| AppError::internal(e.to_string()))?
    .map_err(|e| AppError::internal(e.to_string()))?;
    if control.is_some_and(|c| c.stopped()) {
        return Err(AppError::ServiceUnavailable(
            "Inference cancelled or deadline exceeded".into(),
        ));
    }
    Ok(bytes)
}

/// Qualified option values are accepted only when their provider agrees with the route.
#[cfg(any(feature = "playback", test))]
fn plugin_identity<'a>(
    provider: Option<&'a str>,
    voice: Option<&'a str>,
) -> Result<(&'a str, &'a str), AppError> {
    let provider = provider
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .unwrap_or("supertonic");
    if !matches!(
        provider,
        "supertonic" | "pocket-tts" | "qwen3-gguf" | "qwen3-candle" | "kyutai-candle"
    ) {
        return Err(AppError::BadRequest("unknown provider".into()));
    }
    let voice = voice
        .map(str::trim)
        .unwrap_or(if provider == "supertonic" { "M1" } else { "" });
    if let Some((namespace, raw)) = voice.split_once(':') {
        if namespace != provider || raw.is_empty() || raw.contains(':') {
            return Err(AppError::BadRequest("provider/voice mismatch".into()));
        }
        return Ok((provider, raw));
    }
    Ok((provider, voice))
}

#[cfg(test)]
mod plugin_identity_tests {
    use super::*;
    #[test]
    fn legacy_and_qualified_identity() {
        assert_eq!(plugin_identity(None, None).unwrap(), ("supertonic", "M1"));
        assert_eq!(
            plugin_identity(Some(""), Some("F1")).unwrap(),
            ("supertonic", "F1")
        );
        assert_eq!(
            plugin_identity(Some("pocket-tts"), Some("pocket-tts:id")).unwrap(),
            ("pocket-tts", "id")
        );
        assert!(plugin_identity(Some("pocket-tts"), Some("qwen3-gguf:id")).is_err());
        assert_eq!(
            plugin_identity(Some("kyutai-candle"), Some("kyutai-candle:calm")).unwrap(),
            ("kyutai-candle", "calm")
        );
        assert_eq!(
            plugin_identity(Some("qwen3-candle"), Some("id")).unwrap(),
            ("qwen3-candle", "id")
        );
        assert!(plugin_identity(Some("kyutai-candle"), Some("qwen3-candle:x")).is_err());
        assert!(plugin_identity(Some("unknown"), None).is_err());
    }
}

#[cfg(feature = "playback")]
async fn prepare_plugin_clone(
    state: &crate::AppState,
    token: &AuthenticatedToken,
    provider_id: &str,
    voice: &str,
    text: &str,
    language: Option<String>,
    options: &SpeechOptions,
) -> Result<
    (
        Arc<dyn sonicboom::tts::providers::TtsProvider>,
        sonicboom::tts::providers::ProviderRequest,
    ),
    AppError,
> {
    let config = state.config.get().await;
    let registry = super::providers::build_registry_for_state(state).await;
    let provider = registry.get(provider_id).ok_or_else(|| {
        AppError::ServiceUnavailable("selected provider is not installed in this build".into())
    })?;
    sonicboom::tts::voices::validate_profile_id(voice)
        .map_err(|_| AppError::BadRequest("an authorized voice profile is required".into()))?;
    let (profile, reference) =
        super::voices::owned_reference(state, voice.into(), token.rate_limit_key()).await?;
    let request = plugin_clone_request(
        &config,
        provider.as_ref(),
        provider_id,
        profile,
        reference,
        text,
        language,
        options,
    )?;
    Ok((provider, request))
}

#[cfg(feature = "playback")]
fn playback_manager(state: &crate::AppState) -> Result<&crate::tts::queue::AudioManager, AppError> {
    state.audio_manager.as_ref().as_ref().ok_or_else(|| {
        AppError::ServiceUnavailable("Audio playback system not initialized.".into())
    })
}

#[cfg(feature = "playback")]
#[allow(clippy::too_many_arguments)]
fn plugin_clone_request(
    config: &crate::config::AppConfig,
    provider: &dyn sonicboom::tts::providers::TtsProvider,
    provider_id: &str,
    profile: sonicboom::tts::voices::VoiceProfile,
    reference: sonicboom::tts::providers::types::ReferenceAudio,
    text: &str,
    language: Option<String>,
    options: &SpeechOptions,
) -> Result<sonicboom::tts::providers::ProviderRequest, AppError> {
    if profile.provider != provider_id
        || profile.model_revision != super::voices::config_revision_for(config, provider_id)
    {
        return Err(AppError::BadRequest(
            "voice provider or model revision mismatch".into(),
        ));
    }
    if !provider.status().enabled
        || provider.status().state != sonicboom::tts::providers::ProviderStatusState::Ready
    {
        return Err(AppError::ServiceUnavailable(
            "selected provider is unavailable".into(),
        ));
    }
    if provider.status().active_model.as_deref() != Some(profile.model.as_str()) {
        return Err(AppError::ServiceUnavailable(
            "the voice profile model is not resident; select its model in server configuration"
                .into(),
        ));
    }
    let language = language.unwrap_or(profile.language.clone());
    if provider_id == "pocket-tts" && language != profile.language {
        return Err(AppError::BadRequest(
            "language must match the voice profile".into(),
        ));
    }
    let capabilities = provider.capabilities();
    if !capabilities.models.contains(&profile.model)
        || !capabilities.languages.contains(&language)
        || !capabilities.clone_modes.contains(&profile.clone_mode)
    {
        return Err(AppError::BadRequest(
            "unsupported voice model, language or clone mode".into(),
        ));
    }
    // Only the target text is prepared; the reference transcript and audio
    // travel to the provider unchanged.
    let speech = options.prepare(config, &language, text)?;
    let mut request =
        super::speech_prep::provider_request(&profile.model, &profile.id, speech, language);
    request.clone_mode = Some(super::voices::parse_clone_mode_public(&profile.clone_mode)?);
    request.reference_transcript = profile.transcript;
    request.reference_audio = Some(reference);
    Ok(request)
}

#[cfg(all(test, feature = "playback"))]
mod plugin_play_tests {
    use super::*;
    use sonicboom::tts::providers::{
        PcmAudio, ProviderCapabilities, ProviderError, ProviderRequest, ProviderStatus,
        ProviderStatusState, TtsProvider,
    };
    struct Stub(&'static str, ProviderStatusState);
    impl TtsProvider for Stub {
        fn id(&self) -> &'static str {
            self.0
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                provider_id: self.0.into(),
                models: vec![
                    if self.0 == "pocket-tts" {
                        "pocket-tts-en"
                    } else {
                        "qwen3-tts-1.7b-base"
                    }
                    .into(),
                ],
                languages: vec!["en".into(), "es".into()],
                clone_modes: vec![
                    if self.0 == "pocket-tts" {
                        "audio_prompt"
                    } else {
                        "icl"
                    }
                    .into(),
                ],
                devices: vec!["cpu".into()],
                requires_reference_audio: true,
                requires_reference_transcript: self.0 == "qwen3-gguf",
                voices: vec![],
                sample_rate: Some(24_000),
            }
        }
        fn status(&self) -> ProviderStatus {
            ProviderStatus {
                provider_id: self.0.into(),
                enabled: self.1 != ProviderStatusState::Disabled,
                state: self.1,
                active_model: self.capabilities().models.first().cloned(),
                effective_device: Some("cpu".into()),
                model_revision: None,
                backend_diagnostics: None,
                detail: None,
            }
        }
        fn load(&self, _: &str) -> Result<(), ProviderError> {
            Ok(())
        }
        fn unload(&self) {}
        fn synthesize(&self, request: &ProviderRequest) -> Result<PcmAudio, ProviderError> {
            assert_eq!(request.text, "test");
            assert_eq!(
                request.reference_audio.as_ref().unwrap().sample_rate,
                24_000
            );
            assert_eq!(
                request.reference_audio.as_ref().unwrap().pcm,
                vec![0.25; 72_000]
            );
            Ok(PcmAudio::mono(vec![0.125; 2400], 24_000))
        }
    }
    fn profile(
        config: &crate::config::AppConfig,
        provider: &Stub,
    ) -> sonicboom::tts::voices::VoiceProfile {
        let now = chrono::Utc::now();
        sonicboom::tts::voices::VoiceProfile {
            id: uuid::Uuid::new_v4().to_string(),
            name: "Synthetic authorized fixture".into(),
            provider: provider.id().into(),
            model: provider.capabilities().models[0].clone(),
            model_revision: super::super::voices::config_revision_for(config, provider.id()),
            language: "en".into(),
            clone_mode: provider.capabilities().clone_modes[0].clone(),
            reference_hash: "synthetic-fixture".into(),
            reference_sample_rate: 24_000,
            reference_duration_secs: 3.0,
            transcript: (provider.id() == "qwen3-gguf").then(|| "different reference words".into()),
            owner_token_hash: Some("test-owner".into()),
            consent: sonicboom::tts::voices::VoiceConsent {
                basis: "owner".into(),
                note: String::new(),
                created_at: now,
            },
            created_at: now,
            updated_at: now,
        }
    }
    fn reference() -> sonicboom::tts::providers::types::ReferenceAudio {
        sonicboom::tts::providers::types::ReferenceAudio {
            pcm: vec![0.25; 72_000],
            sample_rate: 24_000,
        }
    }

    #[tokio::test]
    async fn clone_requests_keep_model_reference_mode_and_generated_wav_queue_ownership() {
        use crate::api::tests::{test_config, test_state};
        use crate::tts::queue::{AudioCommand, AudioManager, AudioManagerError};
        let dir = std::env::temp_dir().join(format!("plugin-queue-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let tokens =
            crate::auth::store::TokenStore::load(dir.join("tokens.json").to_str().unwrap())
                .await
                .unwrap();
        let mut config = test_config();
        config.paths.temp_audio = dir.join("audio").to_string_lossy().into_owned();
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let state = test_state(
            config.clone(),
            Arc::new(tokens),
            Arc::new(Some(AudioManager::for_test(tx))),
        );
        for id in ["pocket-tts", "qwen3-gguf"] {
            let provider = Arc::new(Stub(id, ProviderStatusState::Ready));
            let profile = profile(&config, &provider);
            let request = plugin_clone_request(
                &config,
                provider.as_ref(),
                id,
                profile.clone(),
                reference(),
                "test",
                None,
                &SpeechOptions::default(),
            )
            .unwrap();
            assert_eq!(request.model, profile.model);
            assert_eq!(request.voice, profile.id);
            assert_eq!(request.reference_transcript, profile.transcript);
            assert_eq!(request.clone_mode.unwrap().as_str(), profile.clone_mode);
            let audio = synthesize_provider_bounded(&state, provider, request)
                .await
                .unwrap();
            let bytes = encode_output(audio.samples, audio.sample_rate, audio::AudioFormat::Wav)
                .await
                .unwrap();
            for (play_now, reject) in [(false, false), (true, false), (false, true)] {
                let future = enqueue_generated_wav(&state, bytes.clone(), play_now);
                let consume = async {
                    let command = rx.recv().await.unwrap();
                    let (path, temporary, reply) = match command {
                        AudioCommand::Enqueue {
                            path,
                            temporary,
                            reply,
                            ..
                        } if !play_now => (path, temporary, reply),
                        AudioCommand::PlayNow {
                            path,
                            temporary,
                            reply,
                            ..
                        } if play_now => (path, temporary, reply),
                        _ => panic!("wrong playback command"),
                    };
                    assert!(temporary);
                    let wav = hound::WavReader::open(&path).unwrap();
                    assert_eq!(wav.spec().sample_rate, 24_000);
                    assert_eq!(wav.duration(), 2400);
                    reply
                        .send(if reject {
                            Err(AudioManagerError::QueueFull)
                        } else {
                            Ok(())
                        })
                        .unwrap();
                    path
                };
                let (result, path) = tokio::join!(future, consume);
                if reject {
                    assert!(matches!(result, Err(AppError::TooManyRequests(_))));
                    assert!(!path.exists());
                } else {
                    assert_eq!(result.unwrap().status(), StatusCode::OK);
                    std::fs::remove_file(path).unwrap();
                }
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }
    #[test]
    fn clone_validation_rejects_unavailable_revision_language_and_provider_mismatch() {
        let config = crate::api::tests::test_config();
        for status in [
            ProviderStatusState::Disabled,
            ProviderStatusState::Loading,
            ProviderStatusState::NotInstalled,
            ProviderStatusState::Error,
        ] {
            let provider = Stub("pocket-tts", status);
            assert!(matches!(
                plugin_clone_request(
                    &config,
                    &provider,
                    "pocket-tts",
                    profile(&config, &provider),
                    reference(),
                    "test",
                    None,
                    &SpeechOptions::default()
                ),
                Err(AppError::ServiceUnavailable(_))
            ));
        }
        let provider = Stub("pocket-tts", ProviderStatusState::Ready);
        let p = profile(&config, &provider);
        assert!(matches!(
            plugin_clone_request(
                &config,
                &provider,
                "pocket-tts",
                p.clone(),
                reference(),
                "test",
                Some("es".into()),
                &SpeechOptions::default()
            ),
            Err(AppError::BadRequest(_))
        ));
        assert!(matches!(
            plugin_clone_request(
                &config,
                &provider,
                "qwen3-gguf",
                p.clone(),
                reference(),
                "test",
                None,
                &SpeechOptions::default()
            ),
            Err(AppError::BadRequest(_))
        ));
        let mut stale = p;
        stale.model_revision = "different-revision".into();
        assert!(matches!(
            plugin_clone_request(
                &config,
                &provider,
                "pocket-tts",
                stale,
                reference(),
                "test",
                None,
                &SpeechOptions::default()
            ),
            Err(AppError::BadRequest(_))
        ));
    }
}
