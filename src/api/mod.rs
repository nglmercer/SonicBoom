pub mod gate;
pub mod openai;
#[cfg(feature = "playback")]
pub mod queue;
pub mod rate_limit;
pub mod tts;

use crate::AppState;
use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};

pub fn router(state: AppState) -> Router {
    let tts_body_limit = state.config.tts_max_body_bytes;
    let openai_body_limit = state.config.openai_max_body_bytes;
    #[cfg(feature = "playback")]
    let queue_body_limit = state.config.queue_max_body_bytes;

    #[allow(unused_mut)]
    let mut router = Router::new()
        // Original TTS API
        .route(
            "/api/tts",
            post(tts::post_tts).route_layer(DefaultBodyLimit::max(tts_body_limit)),
        )
        .route("/api/status", get(tts::get_status))
        // OpenAI-compatible endpoints
        .route(
            "/v1/audio/speech",
            post(openai::post_speech).route_layer(DefaultBodyLimit::max(openai_body_limit)),
        )
        .route("/v1/models", get(openai::get_models))
        .route("/v1/models/list", get(openai::get_models))
        .route("/v1/voices", get(openai::get_voices));

    #[cfg(feature = "playback")]
    {
        // Audio playback and queue endpoints (only available when playback feature is enabled)
        router = router
            .route(
                "/api/tts/play",
                post(tts::post_tts_and_play).route_layer(DefaultBodyLimit::max(tts_body_limit)),
            )
            .route(
                "/api/queue",
                post(queue::queue_audio).route_layer(DefaultBodyLimit::max(queue_body_limit)),
            )
            .route("/api/queue/next", post(queue::play_next))
            .route("/api/queue/pause", post(queue::pause_audio))
            .route("/api/queue/resume", post(queue::resume_audio))
            .route("/api/queue/stop", post(queue::stop_audio))
            .route(
                "/api/queue/volume",
                post(queue::set_volume).route_layer(DefaultBodyLimit::max(queue_body_limit)),
            )
            .route("/api/queue/status", get(queue::get_queue_status));
    }

    router.with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::store::TokenStore;
    use crate::config::AppConfig;
    use crate::tts::ModelStatus;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    fn test_config() -> AppConfig {
        AppConfig {
            admin_id: "admin".to_string(),
            admin_pw: "long-enough-test-password".to_string(),
            enable_sample_token: false,
            token_store_path: String::new(),
            model_cache_dir: String::new(),
            model_revision: "test".to_string(),
            model_hashes_path: None,
            hf_token: None,
            inference_steps: 5,
            port: 3000,
            log_dir: String::new(),
            log_level: "info".to_string(),
            log_to_file: false,
            log_to_stdout: false,
            auth_required: true,
            allowed_audio_dir: Some("/tmp".to_string()),
            max_text_length: 10_000,
            request_timeout_secs: 5,
            max_concurrent_inference: 1,
            max_pending_inference: 8,
            max_chunk_chars: 200,
            tts_rate_limit_requests: 0, // disabled for auth tests
            tts_rate_limit_window_secs: 60,
            tts_max_body_bytes: 65_536,
            openai_max_body_bytes: 65_536,
            queue_max_body_bytes: 16_384,
            admin_max_body_bytes: 16_384,
            trust_proxy: false,
            trusted_proxies: vec![],
            cookie_secure: false,
            admin_session_expiry_secs: 60,
            temp_audio_dir: "./temp_audio".to_string(),
            enable_hsts: false,
            max_playback_queue_items: 100,
            model_download_connect_timeout_secs: 10,
            model_download_timeout_secs: 1800,
        }
    }

    async fn test_app() -> (Router, String) {
        let path = std::env::temp_dir().join(format!(
            "sonicboom-auth-test-{}-{}.json",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let store = TokenStore::load(path.to_str().unwrap()).await.unwrap();
        let (_token, raw) = store.create(None).await.unwrap();
        let state = AppState {
            model_status: Arc::new(RwLock::new(ModelStatus::Idle)),
            token_store: Arc::new(store),
            config: Arc::new(test_config()),
            audio_manager: Arc::new(None),
            inference_gate: Arc::new(gate::InferenceGate::new(1, 8)),
            rate_limiter: Arc::new(rate_limit::RateLimiter::new(0, 60)),
        };
        let _ = std::fs::remove_file(&path);
        (router(state), raw)
    }

    async fn post_status(app: Router, uri: &str, token: Option<&str>, body: &str) -> StatusCode {
        let mut builder = Request::post(uri);
        if let Some(token) = token {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let request = builder.body(Body::from(body.to_string())).unwrap();
        app.oneshot(request).await.unwrap().status()
    }

    #[tokio::test]
    async fn tts_requires_authentication() {
        let (app, valid) = test_app().await;
        assert_eq!(
            post_status(app.clone(), "/api/tts", None, "hello").await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            post_status(app.clone(), "/api/tts", Some("wrong-token"), "hello").await,
            StatusCode::UNAUTHORIZED
        );
        // Valid token passes auth (503 proves it reached the model check).
        assert_eq!(
            post_status(app, "/api/tts", Some(&valid), "hello").await,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn only_strict_bearer_syntax_is_accepted() {
        let (app, valid) = test_app().await;
        // Canonical form works (503 proves it passed authentication).
        assert_eq!(
            post_status(app.clone(), "/api/tts", Some(&valid), "hello").await,
            StatusCode::SERVICE_UNAVAILABLE
        );
        // Scheme is case-insensitive.
        let request = Request::post("/api/tts")
            .header("authorization", format!("bearer {valid}"))
            .body(Body::from("hello"))
            .unwrap();
        assert_eq!(
            app.clone().oneshot(request).await.unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        // Everything else is rejected.
        for header in [
            valid.clone(),
            "Basic dXNlcjpwYXNz".to_string(),
            "Token ".to_string() + &valid,
            "Bearer".to_string(),
            "Bearer ".to_string(),
            "Bearer  ".to_string() + &valid,
        ] {
            let request = Request::post("/api/tts")
                .header("authorization", header.clone())
                .body(Body::from("hello"))
                .unwrap();
            assert_eq!(
                app.clone().oneshot(request).await.unwrap().status(),
                StatusCode::UNAUTHORIZED,
                "accepted {header:?}"
            );
        }
    }

    #[tokio::test]
    async fn spoofed_referer_and_host_do_not_authenticate() {
        let (app, _) = test_app().await;
        let request = Request::post("/api/tts")
            .header("host", "localhost:3000")
            .header("referer", "http://localhost:3000/")
            .header("origin", "http://localhost:3000")
            .body(Body::from("hello"))
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );

        let (app, _) = test_app().await;
        let request = Request::post("/v1/audio/speech")
            .header("host", "localhost:3000")
            .header("referer", "http://localhost:3000/")
            .body(Body::from(r#"{"input":"hello"}"#))
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn openai_speech_requires_authentication() {
        let (app, valid) = test_app().await;
        assert_eq!(
            post_status(app.clone(), "/v1/audio/speech", None, r#"{"input":"hi"}"#).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            post_status(app, "/v1/audio/speech", Some(&valid), r#"{"input":"hi"}"#).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn oversized_body_is_rejected_before_handling() {
        let (app, valid) = test_app().await;
        let big = "x".repeat(70_000); // over the 64 KiB TTS body limit
        assert_eq!(
            post_status(app, "/api/tts", Some(&valid), &big).await,
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    #[cfg(feature = "playback")]
    #[tokio::test]
    async fn playback_endpoints_require_authentication() {
        for (method, uri, body) in [
            ("POST", "/api/tts/play", "hello"),
            ("POST", "/api/queue", r#"{"path":"x.wav"}"#),
            ("POST", "/api/queue/next", ""),
            ("POST", "/api/queue/pause", ""),
            ("POST", "/api/queue/resume", ""),
            ("POST", "/api/queue/stop", ""),
            ("POST", "/api/queue/volume", r#"{"volume":0.5}"#),
        ] {
            let (app, _) = test_app().await;
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();
            assert_eq!(
                app.oneshot(request).await.unwrap().status(),
                StatusCode::UNAUTHORIZED,
                "{method} {uri} must require authentication"
            );
        }

        let (app, _) = test_app().await;
        let request = Request::get("/api/queue/status")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED,
            "GET /api/queue/status must require authentication"
        );
    }
}
