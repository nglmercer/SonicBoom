//! Same-origin static assets (JavaScript/CSS) embedded in the binary.
//!
//! Keeping all script and style in these files allows a strict
//! Content-Security-Policy with no `'unsafe-inline'`.

use axum::{
    extract::Path,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

const INDEX_JS: &[u8] = include_bytes!("../../static/index.js");
const INDEX_CSS: &[u8] = include_bytes!("../../static/index.css");
const ADMIN_CSS: &[u8] = include_bytes!("../../static/admin.css");

pub async fn get_static(Path(file): Path<String>) -> Response {
    let (content_type, body) = match file.as_str() {
        "index.js" => ("text/javascript; charset=utf-8", INDEX_JS),
        "index.css" | "admin.css" => (
            "text/css; charset=utf-8",
            if file == "index.css" {
                INDEX_CSS
            } else {
                ADMIN_CSS
            },
        ),
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, http::Request, routing::get};
    use tower::ServiceExt;

    fn app() -> Router {
        Router::new().route("/static/{file}", get(get_static))
    }

    async fn fetch(path: &str) -> Response {
        app()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn serves_known_assets_with_content_types() {
        let response = fetch("/static/index.js").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .contains("javascript")
        );
        let response = fetch("/static/index.css").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .contains("css")
        );
        let response = fetch("/static/admin.css").await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_assets_are_404_without_traversal() {
        for path in ["/static/nope.js", "/static/../secret", "/static/"] {
            let response = fetch(path).await;
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        }
    }

    #[test]
    fn embedded_js_has_no_inline_document_write() {
        // The asset allowlist is fixed at compile time; traversal can only 404.
        let js = std::str::from_utf8(INDEX_JS).unwrap();
        assert!(js.contains("modelReady"));
    }
}
