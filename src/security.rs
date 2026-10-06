use axum::{
    body::Body,
    extract::State,
    http::{HeaderValue, Request, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;

use crate::config::ConfigManager;

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
    State(config): State<Arc<ConfigManager>>,
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
    // Read live: the header follows hot-reloaded configuration.
    if config.get().await.security.enable_hsts {
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

/// Reject browser cross-origin mutations and DNS rebinding in local mode.
pub async fn request_origin_guard(
    State(config): State<Arc<ConfigManager>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let cfg = config.get().await;
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok());
    if cfg.server.auth_mode == crate::config::AuthMode::Local {
        let valid_host = host
            .and_then(|host| format!("http://{host}").parse::<axum::http::Uri>().ok())
            .and_then(|uri| uri.host().map(str::to_string))
            .is_some_and(|host| {
                host.eq_ignore_ascii_case("localhost")
                    || host
                        .trim_matches(['[', ']'])
                        .parse::<std::net::IpAddr>()
                        .is_ok_and(|ip| ip.is_loopback())
            });
        if !valid_host {
            return axum::http::StatusCode::FORBIDDEN.into_response();
        }
    }
    if !matches!(
        *request.method(),
        axum::http::Method::GET | axum::http::Method::HEAD | axum::http::Method::OPTIONS
    ) {
        let cross_site = request
            .headers()
            .get("sec-fetch-site")
            .is_some_and(|v| v == "cross-site");
        let foreign_origin = request.headers().get(header::ORIGIN).is_some_and(|origin| {
            origin
                .to_str()
                .ok()
                .and_then(|origin| origin.parse::<axum::http::Uri>().ok())
                .and_then(|origin| origin.authority().map(|a| a.as_str().to_string()))
                .as_deref()
                != host
        });
        if cross_site || foreign_origin {
            return axum::http::StatusCode::FORBIDDEN.into_response();
        }
    }
    next.run(request).await
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

    use crate::config::AppConfig;

    async fn body_text(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    fn test_config(hsts: bool) -> Arc<ConfigManager> {
        let mut config = AppConfig::default();
        config.security.enable_hsts = hsts;
        ConfigManager::in_memory(config)
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
