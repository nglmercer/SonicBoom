//! `POST /api/tts/normalize`: authenticated, rate-limited preview of the
//! text normalizer.
//!
//! Runs exactly the preparation that synthesis runs
//! ([`super::speech_prep::resolve_options`] + the frozen normalizer) and
//! reports what would be spoken. No model inference, no file access, no job
//! creation. `original` is returned only to the caller who sent it and is
//! never logged; responses carry `Cache-Control: private, no-store`.

use std::collections::BTreeMap;

use axum::{
    Json,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use sonicboom::tts::normalization::{
    self, ChangeKind, Confidence, CoverageLevel, NormalizationMode, NormalizedText,
    RULESET_VERSION, ReadingStyle, WarningCode,
};

use crate::{AppState, auth::AuthenticatedToken, error::AppError};

/// Request body. Unknown fields are rejected.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizeRequest {
    /// Text to prepare (trimmed like synthesis input; non-empty, at most
    /// `tts.max_text_length` characters).
    pub text: String,
    /// Speech language (provider base code such as `es`).
    pub language: String,
    /// Pronunciation locale; absent or `auto` selects the language default.
    #[serde(default)]
    pub locale: Option<String>,
    /// `auto` or `off`; absent uses the server's configured default.
    #[serde(default)]
    pub mode: Option<String>,
    /// Reading style: `precise`, `natural` or `digits`; absent uses the
    /// server's configured default.
    #[serde(default)]
    pub reading_style: Option<String>,
    /// Optional model ID; when present, `language` must be valid for it.
    #[serde(default)]
    pub model: Option<String>,
}

/// One replaced span. Offsets are UTF-8 byte offsets into `original`.
#[derive(Debug, Serialize)]
pub struct ChangeView {
    pub kind: ChangeKind,
    pub before: String,
    pub after: String,
    pub start: usize,
    pub end: usize,
    pub confidence: Confidence,
}

/// One preserved span the caller may want to review (byte offsets).
#[derive(Debug, Serialize)]
pub struct WarningView {
    pub code: WarningCode,
    pub before: String,
    pub start: usize,
    pub end: usize,
}

/// Response body (`200`).
#[derive(Debug, Serialize)]
pub struct NormalizeResponse {
    pub original: String,
    pub spoken: String,
    pub language: String,
    pub locale: String,
    pub mode: NormalizationMode,
    pub reading_style: ReadingStyle,
    pub normalizer_version: &'static str,
    pub changes: Vec<ChangeView>,
    pub warnings: Vec<WarningView>,
    pub coverage: BTreeMap<String, CoverageLevel>,
}

impl NormalizeResponse {
    fn new(original: &str, normalized: NormalizedText, reading_style: ReadingStyle) -> Self {
        let slice = |start: usize, end: usize| original.get(start..end).unwrap_or_default();
        Self {
            original: original.to_string(),
            changes: normalized
                .changes
                .into_iter()
                .map(|change| ChangeView {
                    kind: change.kind,
                    before: slice(change.start, change.end).to_string(),
                    after: change.replacement,
                    start: change.start,
                    end: change.end,
                    confidence: change.confidence,
                })
                .collect(),
            warnings: normalized
                .warnings
                .into_iter()
                .map(|warning| WarningView {
                    code: warning.code,
                    before: slice(warning.start, warning.end).to_string(),
                    start: warning.start,
                    end: warning.end,
                })
                .collect(),
            spoken: normalized.spoken,
            language: normalized.language,
            locale: normalized.locale,
            mode: normalized.mode,
            reading_style,
            normalizer_version: RULESET_VERSION,
            coverage: normalized.coverage,
        }
    }
}

/// `POST /api/tts/normalize`.
pub async fn post_normalize(
    token: AuthenticatedToken,
    State(state): State<AppState>,
    body: String,
) -> Response {
    let mut response = match normalize_inner(&token, &state, &body).await {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(error) => error.into_response(),
    };
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    response
}

async fn normalize_inner(
    token: &AuthenticatedToken,
    state: &AppState,
    body: &str,
) -> Result<NormalizeResponse, AppError> {
    super::tts::check_rate_limit(state, token)?;
    let request: NormalizeRequest = serde_json::from_str(body).map_err(json_error)?;
    let config = state.config.get().await;
    let text = request.text.trim();
    let language = request.language.trim();
    // Same shape/length rules as the synthesis routes.
    sonicboom::engine::validate_tts_input(text, language, "preview", 1, config.tts.max_text_length)
        .map_err(|e| AppError::BadRequest(e.to_string()))?;
    if let Some(model) = request.model.as_deref() {
        validate_model_language(model.trim(), language)?;
    }
    let options = super::speech_prep::resolve_options(
        &config,
        request.mode.as_deref(),
        request.locale.as_deref(),
        request.reading_style.as_deref(),
        language,
    )?;
    let reading_style = options.reading_style;
    let normalized =
        normalization::normalize(text, &options).map_err(super::speech_prep::normalize_error)?;
    Ok(NormalizeResponse::new(text, normalized, reading_style))
}

/// Body errors never echo values from the request.
fn json_error(error: serde_json::Error) -> AppError {
    use serde_json::error::Category;
    AppError::BadRequest(match error.classify() {
        Category::Data => "request must be {\"text\", \"language\"} with optional \"locale\", \
                           \"mode\", \"reading_style\" and \"model\" string fields"
            .to_string(),
        _ => format!(
            "invalid JSON request body (line {}, column {})",
            error.line(),
            error.column()
        ),
    })
}

/// The language check synthesis applies for `model`, from static provider
/// metadata (no provider is started or loaded): the provider's advertised
/// languages, plus Pocket's per-model language binding.
fn validate_model_language(model: &str, language: &str) -> Result<(), AppError> {
    use sonicboom::tts::providers::catalog::{ModelProvider, model_provider};
    let unsupported = || {
        AppError::BadRequest(format!(
            "language '{language}' is not supported by model '{model}'"
        ))
    };
    match model_provider(model) {
        None => Err(AppError::BadRequest(format!("unsupported model '{model}'"))),
        Some(ModelProvider::Supertonic) => {
            if sonicboom::tts::providers::supertonic::SUPERTONIC_LANGUAGE_CHOICES
                .contains(&language)
            {
                Ok(())
            } else {
                Err(unsupported())
            }
        }
        Some(ModelProvider::PocketTts) => {
            sonicboom::tts::providers::pocket_tts::PocketTtsProvider::check_model_language(
                model, language,
            )
            .map_err(|_| unsupported())
        }
        Some(ModelProvider::QwenGguf | ModelProvider::QwenCandle) => {
            qwen_language(language).ok_or_else(unsupported)
        }
        Some(ModelProvider::KyutaiCandle) => {
            if ["en", "fr"].contains(&language) {
                Ok(())
            } else {
                Err(unsupported())
            }
        }
    }
}

#[cfg(feature = "qwen-gguf")]
fn qwen_language(language: &str) -> Option<()> {
    sonicboom::tts::providers::qwen_gguf::provider::QWEN_GGUF_LANGUAGES
        .contains(&language)
        .then_some(())
}

/// Without the backend, the published Qwen3-TTS language list still
/// describes the model; synthesis itself reports the provider as not
/// compiled.
#[cfg(not(feature = "qwen-gguf"))]
fn qwen_language(language: &str) -> Option<()> {
    ["en", "de", "es", "zh", "ja", "fr", "ko", "ru", "it", "pt"]
        .contains(&language)
        .then_some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::store::TokenStore;
    use std::sync::Arc;
    use tower::ServiceExt;

    async fn app() -> (axum::Router, String) {
        let path = std::env::temp_dir().join(format!(
            "sonicboom-normalize-test-{}.json",
            uuid::Uuid::new_v4()
        ));
        let store = TokenStore::load(path.to_str().unwrap()).await.unwrap();
        let (_, raw) = store.create(None).await.unwrap();
        let state = crate::api::tests::test_state(
            crate::api::tests::test_config(),
            Arc::new(store),
            Arc::new(None),
        );
        let _ = std::fs::remove_file(&path);
        (crate::api::router(state).await, raw)
    }

    async fn post(
        app: &axum::Router,
        token: Option<&str>,
        body: &str,
    ) -> (StatusCode, axum::http::HeaderMap, serde_json::Value) {
        let mut builder = axum::http::Request::post("/api/tts/normalize")
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(token) = token {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
        }
        let response = app
            .clone()
            .oneshot(
                builder
                    .body(axum::body::Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
            .await
            .unwrap();
        (
            status,
            headers,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    #[tokio::test]
    async fn preview_returns_the_contract_shape_with_no_store() {
        let (app, token) = app().await;
        let (status, headers, body) = post(
            &app,
            Some(&token),
            r#"{"text":"  La reunión es el 26/10/2026.  ","language":"es","locale":"es-PE","mode":"auto","model":"pocket-tts-es"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(headers[header::CACHE_CONTROL], "private, no-store");
        assert_eq!(body["original"], "La reunión es el 26/10/2026.");
        assert_eq!(
            body["spoken"],
            "La reunión es el veintiséis de octubre de dos mil veintiséis."
        );
        assert_eq!(body["language"], "es");
        assert_eq!(body["locale"], "es-PE");
        assert_eq!(body["mode"], "auto");
        assert_eq!(body["reading_style"], "precise");
        assert_eq!(body["normalizer_version"], RULESET_VERSION);
        let change = &body["changes"][0];
        assert_eq!(change["kind"], "date");
        assert_eq!(change["before"], "26/10/2026");
        assert_eq!(
            change["after"],
            "veintiséis de octubre de dos mil veintiséis"
        );
        assert_eq!(change["confidence"], "high");
        let original = body["original"].as_str().unwrap();
        let (start, end) = (
            change["start"].as_u64().unwrap() as usize,
            change["end"].as_u64().unwrap() as usize,
        );
        assert_eq!(&original[start..end], "26/10/2026");
        assert!(body["coverage"]["dates"].is_string());
        assert_eq!(body["coverage"].as_object().unwrap().len(), 11);
        assert!(body["warnings"].as_array().unwrap().is_empty());
        let keys: Vec<_> = body.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys.len(), 10, "exactly the contract fields: {keys:?}");

        // Reading style is applied by the same pipeline as synthesis.
        let (status, _, body) = post(
            &app,
            Some(&token),
            r#"{"text":"12:30","language":"es","reading_style":"natural"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["spoken"], "doce y media");
        assert_eq!(body["reading_style"], "natural");
        let (status, _, body) = post(
            &app,
            Some(&token),
            r#"{"text":"2026","language":"es","reading_style":"digits"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["spoken"], "dos cero dos seis");
        assert_eq!(body["changes"][0]["kind"], "number");

        // Ambiguous input is preserved and reported with its span.
        let (status, _, body) = post(
            &app,
            Some(&token),
            r#"{"text":"Nos vemos el 01/02/2026","language":"es"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["spoken"], "Nos vemos el 01/02/2026");
        assert_eq!(body["locale"], "es-ES");
        assert_eq!(body["warnings"][0]["code"], "ambiguous_date");
        assert_eq!(body["warnings"][0]["before"], "01/02/2026");

        // `off` returns the input unchanged.
        let (status, _, body) = post(
            &app,
            Some(&token),
            r#"{"text":"26/10/2026","language":"es","mode":"OFF"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["spoken"], "26/10/2026");
        assert_eq!(body["mode"], "off");
        assert!(body["changes"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn preview_resolves_padded_counts_and_matches_provider_input() {
        let (app, token) = app().await;
        let config = crate::api::tests::test_config();
        let text = "Hay 10.005 solicitudes y 00042 registros; el código 007.";
        let body = serde_json::json!({
            "text": text,
            "language": "es",
            "locale": "es-ES",
            "mode": "auto",
            "reading_style": "precise",
            "model": "supertonic-3",
        })
        .to_string();
        let (status, _, body) = post(&app, Some(&token), &body).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body["spoken"],
            "Hay diez mil cinco solicitudes y cuarenta y dos registros; el código 007."
        );
        assert_eq!(body["warnings"][0]["code"], "ambiguous_number");
        assert_eq!(body["warnings"][0]["before"], "007");
        assert_eq!(body["changes"][0]["before"], "10.005");
        assert_eq!(body["changes"][0]["after"], "diez mil cinco");
        assert_eq!(body["changes"][1]["before"], "00042");
        assert_eq!(body["changes"][1]["after"], "cuarenta y dos");
        assert_eq!(body["normalizer_version"], RULESET_VERSION);

        // The provider receives exactly this spoken text (the preview and
        // synthesis share one preparation path).
        let options = super::super::speech_prep::resolve_options(
            &config,
            Some("auto"),
            Some("es-ES"),
            Some("precise"),
            "es",
        )
        .unwrap();
        let prepared = super::super::speech_prep::prepare_speech(
            &config,
            Some("auto"),
            Some("es-ES"),
            Some("precise"),
            "es",
            text,
        )
        .unwrap();
        assert_eq!(prepared.as_str(), body["spoken"].as_str().unwrap());
        assert_eq!(prepared.identity(), "tn-v2.1|auto|es|es-ES|precise");
        assert_eq!(options.reading_style, ReadingStyle::Precise);
        let request =
            super::super::speech_prep::provider_request("supertonic-3", "F1", prepared, "es");
        assert_eq!(request.text, body["spoken"].as_str().unwrap());

        // Padded input in `digits` mode keeps the written sequence.
        let (status, _, body) = post(
            &app,
            Some(&token),
            r#"{"text":"007","language":"es","locale":"es-ES","mode":"auto","reading_style":"digits"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["spoken"], "cero cero siete");
        assert!(body["warnings"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn preview_accepts_maximum_length_escaped_text() {
        let (app, token) = app().await;
        for text in [
            "\"".repeat(10_000),
            "\\".repeat(10_000),
            "😀".repeat(10_000),
        ] {
            let body =
                serde_json::json!({"text": text, "language": "en", "mode": "off"}).to_string();
            let (status, _, response) = post(&app, Some(&token), &body).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(response["spoken"], text);
        }
        let body = format!(
            r#"{{"text":"{}","language":"en","mode":"off"}}"#,
            "\\ud83d\\ude00".repeat(10_000)
        );
        assert_eq!(post(&app, Some(&token), &body).await.0, StatusCode::OK);
    }

    #[tokio::test]
    async fn preview_requires_a_bearer_token() {
        let (app, _) = app().await;
        let (status, _, _) = post(&app, None, r#"{"text":"1","language":"en"}"#).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _, _) = post(&app, Some("wrong"), r#"{"text":"1","language":"en"}"#).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn preview_rejects_invalid_input_without_echoing_text() {
        let (app, token) = app().await;
        let long = format!(r#"{{"text":"{}","language":"en"}}"#, "a".repeat(10_001));
        for body in [
            r#"{"text":"   ","language":"es"}"#,
            r#"{"text":"secreto 5","language":"es","locale":"es-AR"}"#,
            r#"{"text":"secreto 5","language":"es","locale":"en-US"}"#,
            r#"{"text":"secreto 5","language":"es","locale":"es_ES"}"#,
            r#"{"text":"secreto 5","language":"es","mode":"strict"}"#,
            r#"{"text":"secreto 5","language":"es","reading_style":"loud"}"#,
            r#"{"text":"secreto 5","language":"es","reading_style":5}"#,
            r#"{"text":"secreto 5","language":"e$"}"#,
            r#"{"text":"secreto 5","language":"ja","model":"pocket-tts-es"}"#,
            r#"{"text":"secreto 5","language":"es","model":"no-such-model"}"#,
            r#"{"text":"secreto 5","language":"es","dictionary":"x"}"#,
            r#"{"text":5,"language":"es"}"#,
            r#"{"language":"es"}"#,
            "not json secreto",
            long.as_str(),
        ] {
            let (status, headers, response) = post(&app, Some(&token), body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(headers[header::CACHE_CONTROL], "private, no-store");
            assert!(!response.to_string().contains("secreto"), "{response}");
        }
    }

    #[test]
    fn model_language_check_matches_provider_metadata() {
        assert!(validate_model_language("pocket-tts-es", "es").is_ok());
        assert!(validate_model_language("pocket-tts-es", "en").is_err());
        assert!(validate_model_language("tts-1", "es").is_ok());
        assert!(validate_model_language("qwen3-tts-1.7b-base", "ko").is_ok());
        assert!(validate_model_language("qwen3-tts-1.7b-base", "ar").is_err());
        assert!(validate_model_language("gpt-sovits-v2-onnx", "en").is_err());
    }
}
