#[cfg(feature = "playback")]
pub mod audio;
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
            .route("/api/queue/status", get(queue::get_queue_status))
            // Audio output device discovery/selection (playback only).
            .route("/api/audio/devices", get(audio::list_output_devices))
            .route(
                "/api/audio/output",
                get(audio::get_output_device)
                    .post(audio::set_output_device)
                    .route_layer(DefaultBodyLimit::max(queue_body_limit)),
            );
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
            port: 17842,
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
            tts_rate_limit_burst: 300,
            audio_output_device: "default".to_string(),
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

    #[cfg(feature = "playback")]
    async fn audio_test_parts() -> (
        Router,
        String,
        tokio::sync::mpsc::Receiver<crate::tts::queue::AudioCommand>,
    ) {
        let path = std::env::temp_dir().join(format!(
            "sonicboom-audio-test-{}-{}.json",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let store = TokenStore::load(path.to_str().unwrap()).await.unwrap();
        let (_token, raw) = store.create(None).await.unwrap();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let manager = crate::tts::queue::AudioManager::for_test(tx);
        let state = AppState {
            model_status: Arc::new(RwLock::new(ModelStatus::Idle)),
            token_store: Arc::new(store),
            config: Arc::new(test_config()),
            audio_manager: Arc::new(Some(manager)),
            inference_gate: Arc::new(gate::InferenceGate::new(1, 8)),
            rate_limiter: Arc::new(rate_limit::RateLimiter::new(0, 60)),
        };
        let _ = std::fs::remove_file(&path);
        (router(state), raw, rx)
    }

    #[cfg(feature = "playback")]
    fn authed(uri: &str, raw: &str, method: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("authorization", format!("Bearer {raw}"))
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[cfg(feature = "playback")]
    async fn body_json(response: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
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
            .header("host", "localhost:17842")
            .header("referer", "http://localhost:17842/")
            .header("origin", "http://localhost:17842")
            .body(Body::from("hello"))
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );

        let (app, _) = test_app().await;
        let request = Request::post("/v1/audio/speech")
            .header("host", "localhost:17842")
            .header("referer", "http://localhost:17842/")
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
            ("POST", "/api/audio/output", r#"{"device":"default"}"#),
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

        for uri in [
            "/api/queue/status",
            "/api/audio/devices",
            "/api/audio/output",
        ] {
            let (app, _) = test_app().await;
            let request = Request::get(uri).body(Body::empty()).unwrap();
            assert_eq!(
                app.oneshot(request).await.unwrap().status(),
                StatusCode::UNAUTHORIZED,
                "GET {uri} must require authentication"
            );
        }
    }

    #[cfg(not(feature = "playback"))]
    #[tokio::test]
    async fn audio_routes_are_absent_without_playback() {
        for (method, uri) in [
            ("GET", "/api/audio/devices"),
            ("GET", "/api/audio/output"),
            ("POST", "/api/audio/output"),
        ] {
            let (app, valid) = test_app().await;
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header("authorization", format!("Bearer {valid}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"device":"default"}"#))
                .unwrap();
            assert_eq!(
                app.oneshot(request).await.unwrap().status(),
                StatusCode::NOT_FOUND,
                "{method} {uri} must not exist without the playback feature"
            );
        }
    }

    #[tokio::test]
    async fn rate_limited_tts_carries_retry_headers() {
        // Tiny bucket: 2 requests, then the 3rd is a 429 with live metadata.
        // The model stays Idle so allowed requests pass the limiter and then
        // fail at the model check (503), proving limiter-before-model order.
        let path = std::env::temp_dir().join(format!(
            "sonicboom-ratelimit-test-{}-{}.json",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let store = TokenStore::load(path.to_str().unwrap()).await.unwrap();
        let (_token, raw) = store.create(None).await.unwrap();
        let mut config = test_config();
        config.tts_rate_limit_requests = 2;
        let state = AppState {
            model_status: Arc::new(RwLock::new(ModelStatus::Idle)),
            token_store: Arc::new(store),
            config: Arc::new(config),
            audio_manager: Arc::new(None),
            inference_gate: Arc::new(gate::InferenceGate::new(1, 8)),
            rate_limiter: Arc::new(rate_limit::RateLimiter::new(2, 60)),
        };
        let _ = std::fs::remove_file(&path);
        let app = router(state);

        for _ in 0..2 {
            assert_eq!(
                post_status(app.clone(), "/api/tts", Some(&raw), "hello").await,
                StatusCode::SERVICE_UNAVAILABLE
            );
        }
        let request = Request::post("/api/tts")
            .header("authorization", format!("Bearer {raw}"))
            .body(Body::from("hello"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let headers = response.headers();
        assert_eq!(headers.get("retry-after").unwrap(), "30");
        assert_eq!(headers.get("x-ratelimit-limit").unwrap(), "2");
        assert_eq!(headers.get("x-ratelimit-remaining").unwrap(), "0");
        assert!(headers.get("x-ratelimit-reset").is_some());
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"], "too_many_requests");
        assert_eq!(body["status"], 429);
    }

    #[cfg(feature = "playback")]
    #[tokio::test]
    async fn audio_devices_reports_list_and_selection() {
        use crate::tts::devices::{OutputDeviceInfo, OutputDeviceList};
        use crate::tts::queue::AudioCommand;

        let (app, valid, mut rx) = audio_test_parts().await;
        tokio::spawn(async move {
            match rx.recv().await {
                Some(AudioCommand::GetOutputDevices { reply }) => {
                    let _ = reply.send(Ok(OutputDeviceList {
                        devices: vec![
                            OutputDeviceInfo {
                                id: "default".to_string(),
                                name: "System Default".to_string(),
                                is_default: true,
                                is_selected: false,
                            },
                            OutputDeviceInfo {
                                id: "Speakers".to_string(),
                                name: "Speakers".to_string(),
                                is_default: false,
                                is_selected: true,
                            },
                        ],
                        selected: "Speakers".to_string(),
                    }));
                }
                other => panic!("unexpected command: {other:?}"),
            }
        });
        let response = app
            .oneshot(authed("/api/audio/devices", &valid, "GET", ""))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["selected"], "Speakers");
        assert_eq!(body["devices"][0]["id"], "default");
        assert_eq!(body["devices"][1]["id"], "Speakers");
        assert_eq!(body["devices"][1]["name"], "Speakers");
    }

    #[cfg(feature = "playback")]
    #[tokio::test]
    async fn audio_devices_enumeration_failure_is_500_not_empty_success() {
        use crate::tts::queue::{AudioCommand, AudioManagerError};

        let (app, valid, mut rx) = audio_test_parts().await;
        tokio::spawn(async move {
            if let Some(AudioCommand::GetOutputDevices { reply }) = rx.recv().await {
                let _ = reply.send(Err(AudioManagerError::DeviceError));
            }
        });
        let response = app
            .oneshot(authed("/api/audio/devices", &valid, "GET", ""))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[cfg(feature = "playback")]
    #[tokio::test]
    async fn audio_output_reports_configured_selection() {
        use crate::tts::devices::ActiveOutputDevice;
        use crate::tts::queue::AudioCommand;

        let (app, valid, mut rx) = audio_test_parts().await;
        tokio::spawn(async move {
            if let Some(AudioCommand::GetOutputDevice { reply }) = rx.recv().await {
                let _ = reply.send(ActiveOutputDevice {
                    device: "CABLE Input".to_string(),
                    resolved_name: Some("CABLE Input".to_string()),
                    available: true,
                });
            }
        });
        let response = app
            .oneshot(authed("/api/audio/output", &valid, "GET", ""))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["device"], "CABLE Input");
        assert_eq!(body["resolved_name"], "CABLE Input");
        assert_eq!(body["available"], true);
    }

    #[cfg(feature = "playback")]
    #[tokio::test]
    async fn audio_output_selection_success_and_unknown_rejected() {
        use crate::tts::devices::ActiveOutputDevice;
        use crate::tts::queue::{AudioCommand, AudioManagerError};

        // Successful switch commits and reports the resolved device.
        let (app, valid, mut rx) = audio_test_parts().await;
        tokio::spawn(async move {
            match rx.recv().await {
                Some(AudioCommand::SetOutputDevice { device, reply }) => {
                    assert_eq!(device, "CABLE Input");
                    let _ = reply.send(Ok(ActiveOutputDevice {
                        device: "CABLE Input".to_string(),
                        resolved_name: Some("CABLE Input".to_string()),
                        available: true,
                    }));
                }
                other => panic!("unexpected command: {other:?}"),
            }
        });
        let response = app
            .oneshot(authed(
                "/api/audio/output",
                &valid,
                "POST",
                r#"{"device":"CABLE Input"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["success"], true);
        assert_eq!(body["device"], "CABLE Input");
        assert_eq!(body["resolved_name"], "CABLE Input");

        // Unknown devices are 400 with no fallback.
        let (app, valid, mut rx) = audio_test_parts().await;
        tokio::spawn(async move {
            if let Some(AudioCommand::SetOutputDevice { reply, .. }) = rx.recv().await {
                let _ = reply.send(Err(AudioManagerError::UnknownDevice));
            }
        });
        let response = app
            .oneshot(authed(
                "/api/audio/output",
                &valid,
                "POST",
                r#"{"device":"Nope"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["success"], false);
    }

    #[cfg(feature = "playback")]
    #[tokio::test]
    async fn audio_output_rejects_empty_device_without_contacting_thread() {
        let (app, valid, rx) = audio_test_parts().await;
        drop(rx); // No command must be sent: validation happens first.
        let response = app
            .oneshot(authed(
                "/api/audio/output",
                &valid,
                "POST",
                r#"{"device":"  "}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[cfg(feature = "playback")]
    #[tokio::test]
    async fn audio_apis_are_unavailable_without_manager() {
        let (app, valid) = test_app().await; // audio_manager: None
        for (method, uri, body) in [
            ("GET", "/api/audio/devices", ""),
            ("GET", "/api/audio/output", ""),
            ("POST", "/api/audio/output", r#"{"device":"default"}"#),
        ] {
            let response = app
                .clone()
                .oneshot(authed(uri, &valid, method, body))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{method} {uri} without a manager must be 503"
            );
        }
    }
}
