pub mod client_ip;
pub mod handlers;
pub mod lockout;
pub mod session;
pub mod templates;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    middleware,
    routing::{get, post},
};
use handlers::AdminState;

use crate::security::no_store_cache;

pub fn router(state: AdminState) -> Router {
    let body_limit = state.config.admin_max_body_bytes;
    Router::new()
        .route("/admin", get(handlers::get_admin))
        .route(
            "/admin/login",
            get(handlers::get_login).post(handlers::post_login),
        )
        // Logout changes session state, so it must be POST (with CSRF).
        .route("/admin/logout", post(handlers::post_logout))
        .route("/admin/tokens", post(handlers::post_create_token))
        .route(
            "/admin/tokens/{id}/revoke",
            post(handlers::post_revoke_token),
        )
        .route_layer(DefaultBodyLimit::max(body_limit))
        // Admin pages carry auth state and one-time token displays.
        .layer(middleware::from_fn(no_store_cache))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{admin::lockout::LoginAttemptTracker, auth::store::TokenStore, config::AppConfig};
    use axum::{
        body::Body,
        http::{Request, StatusCode, header},
    };
    use std::sync::Arc;
    use tower::ServiceExt;
    use tower_sessions::{MemoryStore, SessionManagerLayer};

    fn test_router() -> Router {
        let config = Arc::new(AppConfig {
            admin_id: "admin".to_string(),
            admin_pw: "long-enough-test-password".to_string(),
            enable_sample_token: false,
            token_store_path: String::new(),
            model_cache_dir: String::new(),
            model_revision: crate::config::DEFAULT_MODEL_REVISION.to_string(),
            model_hashes_path: None,
            hf_token: None,
            inference_steps: 5,
            port: 17842,
            log_dir: String::new(),
            log_level: "info".to_string(),
            log_to_file: false,
            log_to_stdout: false,
            auth_required: true,
            allowed_audio_dir: None,
            max_text_length: 100,
            request_timeout_secs: 1,
            max_concurrent_inference: 1,
            max_pending_inference: 1,
            max_chunk_chars: 10,
            tts_rate_limit_requests: 0,
            tts_rate_limit_window_secs: 60,
            tts_rate_limit_burst: 300,
            audio_output_device: "default".to_string(),
            tts_max_body_bytes: 1024,
            openai_max_body_bytes: 1024,
            queue_max_body_bytes: 1024,
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
        });
        let state = AdminState {
            token_store: Arc::new(TokenStore::empty()),
            lockout: Arc::new(LoginAttemptTracker::default()),
            config,
        };
        router(state).layer(SessionManagerLayer::new(MemoryStore::default()))
    }

    #[tokio::test]
    async fn admin_pages_are_non_cacheable() {
        for (method, uri) in [
            ("GET", "/admin/login"),
            ("GET", "/admin"),
            ("POST", "/admin/tokens"),
            ("POST", "/admin/tokens/some-id/revoke"),
            ("POST", "/admin/logout"),
        ] {
            let app = test_router();
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(Body::from("csrf_token="))
                .unwrap();
            let response = app.oneshot(request).await.unwrap();
            assert!(
                response.status() != StatusCode::NOT_FOUND,
                "{method} {uri} unexpectedly 404"
            );
            assert_eq!(
                response.headers().get(header::CACHE_CONTROL).unwrap(),
                "no-store",
                "{method} {uri} missing no-store"
            );
        }
    }

    #[tokio::test]
    async fn login_page_does_not_leak_state() {
        let response = test_router()
            .oneshot(Request::get("/admin/login").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let html = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            !html.contains("csrf_token"),
            "login page needs no CSRF field"
        );
    }
}
