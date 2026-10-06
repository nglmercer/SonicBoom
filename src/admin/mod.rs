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

/// Build the admin router. The body limit is read
/// live from the configuration so limit changes
/// apply when the listener restarts.
pub async fn router(state: AdminState) -> Router {
    let body_limit = state.config.get().await.tts.admin_max_body_bytes;
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
    use crate::{
        admin::lockout::LoginAttemptTracker,
        auth::store::TokenStore,
        config::{AppConfig, ConfigManager, secrets::MemorySecretStore},
    };
    use axum::{
        body::Body,
        http::{Request, StatusCode, header},
    };
    use std::sync::Arc;
    use tower::ServiceExt;
    use tower_sessions::{MemoryStore, SessionManagerLayer};

    async fn test_router() -> Router {
        let config = ConfigManager::in_memory(AppConfig::default());
        let state = AdminState {
            token_store: Arc::new(TokenStore::empty()),
            lockout: Arc::new(LoginAttemptTracker::default()),
            config,
            secrets: Arc::new(MemorySecretStore::default()),
        };
        router(state)
            .await
            .layer(SessionManagerLayer::new(MemoryStore::default()))
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
            let app = test_router().await;
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
            .await
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
