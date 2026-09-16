use axum::{
    body::Body,
    extract::State,
    http::{HeaderValue, Request, header},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

use crate::config::AppConfig;

/// Add defense-in-depth security headers to responses.
///
/// - `X-Content-Type-Options: nosniff` on everything.
/// - `Referrer-Policy: no-referrer` so bearer tokens in URLs (none are used)
///   or page URLs never leak via navigation.
/// - `X-Frame-Options: DENY` plus CSP `frame-ancestors 'none'` on HTML pages
///   to prevent clickjacking of the admin UI.
/// - A strict CSP on HTML pages. All JavaScript and CSS live in same-origin
///   static assets, so no `'unsafe-inline'` is needed. The TTS demo page
///   plays audio from a blob URL, hence `media-src 'self' blob:`.
pub async fn security_headers(
    State(config): State<Arc<AppConfig>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    // HSTS is opt-in: only enable when the deployment is known-HTTPS
    // (direct TLS or a trusted TLS-terminating proxy for all traffic).
    if config.enable_hsts {
        headers.insert(
            header::STRICT_TRANSPORT_SECURITY,
            HeaderValue::from_static("max-age=63072000; includeSubDomains"),
        );
    }
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );

    let is_html = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/html"));
    if is_html {
        headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
        headers.insert(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static(
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'",
            ),
        );
    }
    response
}

/// Prevent caching of sensitive admin responses (auth state, one-time token
/// display, auth redirects). Applied as a layer over the whole admin router.
pub async fn no_store_cache(request: Request<Body>, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert("pragma", HeaderValue::from_static("no-cache"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
    };
    use tower::ServiceExt;

    async fn body_text(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    fn test_config(hsts: bool) -> Arc<AppConfig> {
        Arc::new(AppConfig {
            admin_id: "admin".to_string(),
            admin_pw: "long-enough-test-password".to_string(),
            enable_sample_token: false,
            token_store_path: String::new(),
            model_cache_dir: String::new(),
            model_revision: crate::config::DEFAULT_MODEL_REVISION.to_string(),
            model_hashes_path: None,
            hf_token: None,
            inference_steps: 5,
            port: 3000,
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
            tts_max_body_bytes: 1024,
            openai_max_body_bytes: 1024,
            queue_max_body_bytes: 1024,
            admin_max_body_bytes: 1024,
            trust_proxy: false,
            trusted_proxies: vec![],
            cookie_secure: false,
            admin_session_expiry_secs: 60,
            temp_audio_dir: "./temp_audio".to_string(),
            enable_hsts: hsts,
        })
    }

    fn app() -> Router {
        app_with_hsts(false)
    }

    fn app_with_hsts(hsts: bool) -> Router {
        Router::new()
            .route(
                "/html",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                        "<html></html>",
                    )
                }),
            )
            .route(
                "/audio",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "audio/ogg; codecs=opus")],
                        vec![0u8, 1u8],
                    )
                }),
            )
            .layer(middleware::from_fn_with_state(
                test_config(hsts),
                security_headers,
            ))
    }

    #[tokio::test]
    async fn html_responses_carry_strict_headers() {
        let response = app()
            .oneshot(Request::get("/html").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let headers = response.headers();
        assert_eq!(
            headers.get(header::X_CONTENT_TYPE_OPTIONS).unwrap(),
            "nosniff"
        );
        assert_eq!(headers.get(header::REFERRER_POLICY).unwrap(), "no-referrer");
        assert_eq!(headers.get(header::X_FRAME_OPTIONS).unwrap(), "DENY");
        let csp = headers
            .get(header::CONTENT_SECURITY_POLICY)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(csp.contains("frame-ancestors 'none'"), "csp: {csp}");
        assert!(!csp.contains("'unsafe-inline'"), "csp: {csp}");
        assert!(csp.contains("script-src 'self'"), "csp: {csp}");
        assert!(csp.contains("style-src 'self'"), "csp: {csp}");
        assert!(csp.contains("base-uri 'none'"), "csp: {csp}");
        assert!(csp.contains("object-src 'none'"), "csp: {csp}");
        assert!(csp.contains("media-src 'self' blob:"), "csp: {csp}");
        assert_eq!(body_text(response).await, "<html></html>");
    }

    #[tokio::test]
    async fn hsts_is_opt_in() {
        let response = app()
            .oneshot(Request::get("/html").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert!(
            response
                .headers()
                .get(header::STRICT_TRANSPORT_SECURITY)
                .is_none()
        );
        let response = app_with_hsts(true)
            .oneshot(Request::get("/html").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            response
                .headers()
                .get(header::STRICT_TRANSPORT_SECURITY)
                .unwrap(),
            "max-age=63072000; includeSubDomains"
        );
    }

    #[tokio::test]
    async fn no_store_middleware_marks_responses() {
        let app = Router::new()
            .route("/sensitive", get(|| async { "secret" }))
            .layer(middleware::from_fn(no_store_cache));
        let response = app
            .oneshot(Request::get("/sensitive").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CACHE_CONTROL).unwrap(),
            "no-store"
        );
        assert_eq!(response.headers().get("pragma").unwrap(), "no-cache");
    }

    #[tokio::test]
    async fn non_html_responses_skip_framing_headers() {
        let response = app()
            .oneshot(Request::get("/audio").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::X_CONTENT_TYPE_OPTIONS)
                .unwrap(),
            "nosniff"
        );
        assert!(response.headers().get(header::X_FRAME_OPTIONS).is_none());
    }
}
