use axum::{
    body::Body,
    http::{HeaderValue, Request, header},
    middleware::Next,
    response::Response,
};

/// Add defense-in-depth security headers to responses.
///
/// - `X-Content-Type-Options: nosniff` on everything.
/// - `Referrer-Policy: no-referrer` so bearer tokens in URLs (none are used)
///   or page URLs never leak via navigation.
/// - `X-Frame-Options: DENY` plus CSP `frame-ancestors 'none'` on HTML pages
///   to prevent clickjacking of the admin UI.
/// - A strict CSP on HTML pages. The built-in pages use inline `<style>` but
///   no external resources, so `default-src 'self'` with `'unsafe-inline'`
///   for style only is sufficient. (The TTS demo page uses inline script;
///   it is served same-origin and allowed via `'unsafe-inline'` for script
///   pending extraction to an external file.)
pub async fn security_headers(request: Request<Body>, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
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
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            ),
        );
    }
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

    fn app() -> Router {
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
            .layer(middleware::from_fn(security_headers))
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
        assert_eq!(body_text(response).await, "<html></html>");
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
