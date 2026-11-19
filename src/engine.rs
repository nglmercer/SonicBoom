use std::{path::PathBuf, sync::Arc};

use anyhow::{Result, anyhow};

use crate::tts::{
    backend::BackendChoice,
    audio, download, inference,
    model::ModelHandle,
    normalization::{NormalizeError, NormalizeLimits, NormalizeOptions, ReadingStyle},
};

pub use crate::tts::normalization::{NormalizationMode, SpeechText};

const MAX_TEXT_CHARS: usize = 10_000;
const MAX_INFERENCE_STEPS: usize = 50;
/// Spoken-text budget of the reusable engine. Matches the server defaults
/// (`tts.normalization.max_normalized_chars` / `max_expansion_ratio`).
const MAX_SPOKEN_CHARS: usize = 20_000;
const MAX_EXPANSION_RATIO: f32 = 8.0;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_dir: PathBuf,
    pub hf_token: Option<String>,
    pub inference_steps: usize,
    /// Pinned HuggingFace revision. Defaults to [`download::DEFAULT_MODEL_REVISION`].
    pub model_revision: Option<String>,
    /// Optional expected-hashes manifest (see [`download::load_expected_hashes`]).
    pub model_hashes_path: Option<PathBuf>,
    /// Text normalization mode. Defaults to [`NormalizationMode::Auto`], the
    /// same default as the HTTP server; `Off` synthesizes the text unchanged.
    pub normalization: NormalizationMode,
    /// Pronunciation locale such as `"es-PE"`. `None` or `"auto"` selects the
    /// documented default of the speech language (server default). A
    /// concrete tag applies only to requests in its own language.
    pub locale: Option<String>,
    /// How written numbers and times are read. Defaults to
    /// [`ReadingStyle::Precise`], the compatibility style.
    pub reading_style: ReadingStyle,
    /// Inference engine for the Supertonic graphs; `auto` resolves per build
    /// (candle when compiled in, otherwise ONNX Runtime).
    pub backend: BackendChoice,
}

impl EngineConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            hf_token: None,
            inference_steps: 5,
            model_revision: None,
            model_hashes_path: None,
            normalization: NormalizationMode::Auto,
            locale: None,
            reading_style: ReadingStyle::default(),
            backend: BackendChoice::auto(),
        }
    }
}

/// Locale to request for one synthesis.
///
/// A non-empty `request` value always wins (`"auto"` selects the language
/// default). Otherwise a configured default (`"auto"` or a concrete tag)
/// applies only when its primary subtag is the speech language; a tag of
/// another language falls back to the speech language's documented default.
pub fn requested_locale<'a>(
    speech_language: &str,
    request: Option<&'a str>,
    configured_default: Option<&'a str>,
) -> Option<&'a str> {
    if let Some(value) = request.map(str::trim).filter(|value| !value.is_empty()) {
        return Some(value);
    }
    let configured = configured_default
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("auto"))?;
    let primary = configured.split('-').next().unwrap_or_default();
    primary
        .eq_ignore_ascii_case(speech_language.trim())
        .then_some(configured)
}

/// Resolve normalization options with the server's legacy-compatible rules.
///
/// Identical to [`NormalizeOptions::resolve`], except that a speech language
/// that is not a base code (for example `en-US`, which the legacy
/// validation accepts) passes through unchanged instead of failing, as long
/// as no concrete locale was requested. In `off` mode any language accepted
/// by the legacy validation passes through.
pub fn resolve_speech_options(
    mode: NormalizationMode,
    speech_language: &str,
    requested_locale: Option<&str>,
    limits: NormalizeLimits,
    report_ambiguities: bool,
) -> std::result::Result<NormalizeOptions, NormalizeError> {
    resolve_speech_options_with_style(
        mode,
        speech_language,
        requested_locale,
        limits,
        report_ambiguities,
        ReadingStyle::default(),
    )
}

/// [`resolve_speech_options`] with an explicit reading style.
#[allow(clippy::too_many_arguments)]
pub fn resolve_speech_options_with_style(
    mode: NormalizationMode,
    speech_language: &str,
    requested_locale: Option<&str>,
    limits: NormalizeLimits,
    report_ambiguities: bool,
    reading_style: ReadingStyle,
) -> std::result::Result<NormalizeOptions, NormalizeError> {
    match NormalizeOptions::resolve_with_style(
        mode,
        speech_language,
        requested_locale,
        limits,
        report_ambiguities,
        reading_style,
    ) {
        Err(NormalizeError::InvalidLanguage)
            if locale_is_default(requested_locale)
                && (mode == NormalizationMode::Off || is_well_formed_tag(speech_language)) =>
        {
            let tag = speech_language.trim().to_ascii_lowercase();
            if tag.is_empty() {
                return Err(NormalizeError::InvalidLanguage);
            }
            Ok(NormalizeOptions {
                mode,
                language: tag.clone(),
                locale: tag,
                locale_explicit: false,
                limits,
                report_ambiguities,
                reading_style,
            })
        }
        other => other,
    }
}

/// The single text-preparation step shared by every synthesis entry point
/// (HTTP routes, jobs and this engine): resolves options and runs the
/// normalizer once. The returned [`SpeechText`] is the only input that
/// synthesis accepts for user text.
pub fn prepare_speech_text(
    text: &str,
    speech_language: &str,
    mode: NormalizationMode,
    requested_locale: Option<&str>,
    limits: NormalizeLimits,
) -> std::result::Result<SpeechText, NormalizeError> {
    prepare_speech_text_with_style(
        text,
        speech_language,
        mode,
        requested_locale,
        limits,
        ReadingStyle::default(),
    )
}

/// [`prepare_speech_text`] with an explicit reading style.
pub fn prepare_speech_text_with_style(
    text: &str,
    speech_language: &str,
    mode: NormalizationMode,
    requested_locale: Option<&str>,
    limits: NormalizeLimits,
    reading_style: ReadingStyle,
) -> std::result::Result<SpeechText, NormalizeError> {
    let options = resolve_speech_options_with_style(
        mode,
        speech_language,
        requested_locale,
        limits,
        false,
        reading_style,
    )?;
    SpeechText::prepare(text, &options)
}

/// Build a provider request whose text is an already prepared
/// [`SpeechText`]; records its identity in
/// [`ProviderRequest::speech_identity`](crate::tts::providers::types::ProviderRequest::speech_identity).
pub fn provider_request_for(
    model: impl Into<String>,
    voice: impl Into<String>,
    speech: SpeechText,
    language: impl Into<String>,
) -> crate::tts::providers::types::ProviderRequest {
    let identity = speech.identity().to_string();
    let mut request = crate::tts::providers::types::ProviderRequest::new(
        model,
        voice,
        speech.into_string(),
        language,
    );
    request.speech_identity = Some(identity);
    request
}

fn locale_is_default(requested: Option<&str>) -> bool {
    requested
        .map(str::trim)
        .is_none_or(|value| value.is_empty() || value.eq_ignore_ascii_case("auto"))
}

/// Hyphen-separated tag with a 2-3 letter primary subtag and 1-8
/// alphanumeric subtags (BCP-47 shape only, no registry lookup).
fn is_well_formed_tag(raw: &str) -> bool {
    let raw = raw.trim();
    if raw.is_empty() || raw.len() > 64 {
        return false;
    }
    raw.split('-').enumerate().all(|(index, subtag)| {
        if index == 0 {
            (2..=3).contains(&subtag.len()) && subtag.bytes().all(|b| b.is_ascii_alphabetic())
        } else {
            (1..=8).contains(&subtag.len()) && subtag.bytes().all(|b| b.is_ascii_alphanumeric())
        }
    })
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub language: String,
    pub voice: String,
    pub inference_steps: usize,
}

/// Shared TTS input validation used by both the reusable engine and the
/// HTTP API so the rules cannot drift apart. Lengths are measured in Unicode
/// characters, not UTF-8 bytes.
pub fn validate_tts_input(
    text: &str,
    language: &str,
    voice: &str,
    inference_steps: usize,
    max_text_chars: usize,
) -> Result<()> {
    let text = text.trim();
    if text.is_empty() {
        return Err(anyhow!("text cannot be empty"));
    }
    if text.chars().count() > max_text_chars {
        return Err(anyhow!("text exceeds {max_text_chars} characters"));
    }
    if language.is_empty()
        || language.len() > 16
        || !language
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        return Err(anyhow!("language must be a valid language tag"));
    }
    if voice.is_empty() || voice.len() > 16 {
        return Err(anyhow!("voice must be a valid voice name"));
    }
    if !(1..=MAX_INFERENCE_STEPS).contains(&inference_steps) {
        return Err(anyhow!(
            "inferenceSteps must be between 1 and {MAX_INFERENCE_STEPS}"
        ));
    }
    Ok(())
}

impl SynthesisRequest {
    pub fn validate(&self) -> Result<()> {
        validate_tts_input(
            &self.text,
            &self.language,
            &self.voice,
            self.inference_steps,
            MAX_TEXT_CHARS,
        )
    }

    /// Prepare this request's text for synthesis exactly like the HTTP
    /// server does: the legacy trimming first, then normalization with
    /// `mode` and `locale` (`None`/`"auto"` = language default). Length
    /// validation of the original text stays in [`Self::validate`].
    pub fn prepare_speech(
        &self,
        mode: NormalizationMode,
        locale: Option<&str>,
    ) -> Result<SpeechText> {
        self.prepare_speech_with_style(mode, locale, ReadingStyle::default())
    }

    /// [`Self::prepare_speech`] with an explicit reading style.
    pub fn prepare_speech_with_style(
        &self,
        mode: NormalizationMode,
        locale: Option<&str>,
        reading_style: ReadingStyle,
    ) -> Result<SpeechText> {
        let limits = NormalizeLimits {
            max_input_chars: MAX_TEXT_CHARS,
            max_output_chars: MAX_SPOKEN_CHARS,
            max_expansion_ratio: MAX_EXPANSION_RATIO,
        };
        prepare_speech_text_with_style(
            self.text.trim(),
            &self.language,
            mode,
            requested_locale(&self.language, None, locale),
            limits,
            reading_style,
        )
        .map_err(|error| anyhow!("text normalization failed: {error}"))
    }

    /// Adapt a legacy request to the extensible provider contract without
    /// changing legacy validation or defaults. The returned request targets
    /// `model` (a Supertonic alias by default) and carries no reference
    /// audio: cloning providers reject it explicitly until a voice profile
    /// supplies authorized reference material. The text is carried raw
    /// (`speech_identity` is `None`); use [`provider_request_for`] with a
    /// prepared [`SpeechText`] for synthesis.
    pub fn to_provider_request(
        &self,
        model: impl Into<String>,
    ) -> crate::tts::providers::types::ProviderRequest {
        crate::tts::providers::types::ProviderRequest::new(
            model,
            self.voice.clone(),
            self.text.clone(),
            self.language.clone(),
        )
    }
}

pub struct SynthesisResult {
    pub wav: Vec<u8>,
    pub sample_rate: u32,
}

pub struct SonicBoomEngine {
    model: Arc<ModelHandle>,
    inference_lock: std::sync::Mutex<()>,
    normalization: NormalizationMode,
    locale: Option<String>,
    reading_style: ReadingStyle,
}

impl SonicBoomEngine {
    pub async fn prepare<F>(config: EngineConfig, on_progress: F) -> Result<Arc<Self>>
    where
        F: Fn(f32) + Send + Sync + 'static,
    {
        let revision = config
            .model_revision
            .clone()
            .unwrap_or_else(|| download::DEFAULT_MODEL_REVISION.to_string());
        let custom_manifest = config
            .model_hashes_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned());
        // Same trust policy as the HTTP server: a custom revision requires
        // an explicit trusted manifest for that exact revision.
        let (revision, expected_trust) =
            download::resolve_trust(&revision, custom_manifest.as_deref())?;
        let backend = config.backend;
        let normalization = config.normalization;
        let locale = config.locale.clone();
        let reading_style = config.reading_style;
        let paths = download::download_models_with_options(
            &config.model_dir,
            config.hf_token.as_deref(),
            &revision,
            &expected_trust,
            &download::DownloadLimits::default(),
            on_progress,
        )
        .await?;
        let model = tokio::task::spawn_blocking(move || ModelHandle::load_with(&paths, backend))
            .await
            .map_err(|error| anyhow!("model loading task failed: {error}"))??;
        Ok(Arc::new(Self {
            model: Arc::new(model),
            inference_lock: std::sync::Mutex::new(()),
            normalization,
            locale,
            reading_style,
        }))
    }

    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.model.voice_styles.keys().map(String::as_str)
    }

    /// Provider identity of this engine in the native registry
    /// (`supertonic`). The engine itself is unchanged: model download,
    /// voice styles, inference steps and output encoding behave exactly as
    /// before; the registry adapter only routes requests to it.
    pub fn provider_id(&self) -> &'static str {
        "supertonic"
    }

    pub fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResult> {
        request.validate()?;
        if !self.model.voice_styles.contains_key(&request.voice) {
            return Err(anyhow!("voice '{}' is not available", request.voice));
        }
        // Same preparation as the HTTP server, once, before chunking.
        let speech = request.prepare_speech_with_style(
            self.normalization,
            self.locale.as_deref(),
            self.reading_style,
        )?;
        // The lock guards the ONNX sessions, which are touched only by
        // `inference::synthesize`. WAV encoding touches no session state, so
        // the lock is released first: encoding must never block the next
        // model inference.
        let samples = {
            let _guard = self
                .inference_lock
                .lock()
                .map_err(|_| anyhow!("inference lock poisoned"))?;
            inference::synthesize(
                &self.model,
                speech.as_str(),
                &request.language,
                &request.voice,
                request.inference_steps,
                crate::tts::text::DEFAULT_MAX_CHUNK_CHARS,
            )?
        };
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
    fn legacy_requests_adapt_to_the_provider_contract() {
        let adapted = request().to_provider_request("supertonic-3");
        assert_eq!(adapted.model, "supertonic-3");
        assert_eq!(adapted.voice, "F1");
        assert_eq!(adapted.text, "hello");
        assert_eq!(adapted.language, "en");
        assert!(adapted.reference_audio.is_none());
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

    #[test]
    fn engine_defaults_match_server_normalization_defaults() {
        let config = EngineConfig::new("models");
        assert_eq!(config.normalization, NormalizationMode::Auto);
        assert_eq!(config.locale, None);
        assert_eq!(config.reading_style, ReadingStyle::Precise);
    }

    #[test]
    fn engine_preparation_matches_the_shared_helper() {
        let mut request = request();
        request.language = "es".into();
        request.text = "  Pago el 26/10/2026  ".into();
        let auto = request
            .prepare_speech(NormalizationMode::Auto, None)
            .unwrap();
        assert_eq!(
            auto.as_str(),
            "Pago el veintiséis de octubre de dos mil veintiséis"
        );
        assert_eq!(auto.identity(), "tn-v2.1|auto|es|es-ES|precise");
        let shared = prepare_speech_text(
            "Pago el 26/10/2026",
            "es",
            NormalizationMode::Auto,
            None,
            NormalizeLimits::default(),
        )
        .unwrap();
        assert_eq!(shared.as_str(), auto.as_str());
        assert_eq!(shared.identity(), auto.identity());
        // `off` hands inference the exact legacy (trimmed) text.
        let off = request
            .prepare_speech(NormalizationMode::Off, None)
            .unwrap();
        assert_eq!(off.as_str(), "Pago el 26/10/2026");
        assert_eq!(off.identity(), "tn-v2.1|off|es|es-ES|precise");
        // A configured locale of another language falls back to the default.
        let other = request
            .prepare_speech(NormalizationMode::Auto, Some("en-GB"))
            .unwrap();
        assert_eq!(other.identity(), "tn-v2.1|auto|es|es-ES|precise");
        let peru = request
            .prepare_speech(NormalizationMode::Auto, Some("es-PE"))
            .unwrap();
        assert_eq!(peru.identity(), "tn-v2.1|auto|es|es-PE|precise");
    }

    #[test]
    fn requested_locale_prefers_request_then_matching_config() {
        assert_eq!(
            requested_locale("es", Some("es-MX"), Some("es-PE")),
            Some("es-MX")
        );
        assert_eq!(
            requested_locale("es", Some("auto"), Some("es-PE")),
            Some("auto")
        );
        assert_eq!(
            requested_locale("es", Some(" "), Some("es-PE")),
            Some("es-PE")
        );
        assert_eq!(requested_locale("es", None, Some("es-PE")), Some("es-PE"));
        assert_eq!(requested_locale("en", None, Some("es-PE")), None);
        assert_eq!(requested_locale("es", None, Some("auto")), None);
        assert_eq!(requested_locale("es", None, None), None);
    }

    #[test]
    fn legacy_language_tags_pass_through_instead_of_failing() {
        let limits = NormalizeLimits::default();
        let tagged = prepare_speech_text(
            "I have 3 cats",
            "en-US",
            NormalizationMode::Auto,
            None,
            limits,
        )
        .unwrap();
        assert_eq!(tagged.as_str(), "I have 3 cats");
        assert_eq!(tagged.identity(), "tn-v2.1|auto|en-us|en-us|precise");
        // A concrete locale with a non-base language is still an error.
        assert_eq!(
            prepare_speech_text("x", "en-US", NormalizationMode::Auto, Some("en-US"), limits)
                .unwrap_err(),
            NormalizeError::InvalidLanguage
        );
        // Malformed codes are errors in auto mode, legacy pass-through in off.
        assert!(
            prepare_speech_text("x", "english", NormalizationMode::Auto, None, limits).is_err()
        );
        assert_eq!(
            prepare_speech_text("x", "english", NormalizationMode::Off, None, limits)
                .unwrap()
                .as_str(),
            "x"
        );
    }

    #[test]
    fn prepared_provider_requests_record_the_identity_only() {
        let speech = prepare_speech_text(
            "I have 3 cats",
            "en",
            NormalizationMode::Auto,
            None,
            NormalizeLimits::default(),
        )
        .unwrap();
        let identity = speech.identity().to_string();
        let spoken = speech.as_str().to_string();
        let request = provider_request_for("m", "v", speech, "en");
        assert_eq!(request.text, spoken);
        assert_eq!(request.speech_identity.as_deref(), Some(identity.as_str()));
        assert!(!identity.contains("cats"));
        assert!(request.reference_transcript.is_none());
        assert!(request_raw_identity_is_absent());
    }

    fn request_raw_identity_is_absent() -> bool {
        request()
            .to_provider_request("supertonic-3")
            .speech_identity
            .is_none()
    }
}
