use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

/// Client-facing request identifier, logged server-side with the full error
/// so operators can correlate reports without leaking internals.
fn correlation_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("too many requests")]
    TooManyRequests(String),
    /// Token-bucket rejection. Same `429` JSON shape as [`AppError::TooManyRequests`]
    /// plus `Retry-After` / `X-RateLimit-*` headers computed from live limiter
    /// state. Inference saturation and playback-queue overflow keep using
    /// [`AppError::TooManyRequests`] (no rate-limit headers), so operators can
    /// tell the three `429` sources apart by message and header presence:
    /// - `RateLimited`: "Rate limit exceeded..." + `Retry-After`
    /// - inference saturated: "Server is busy..." (no headers)
    /// - playback queue full: "Playback queue is full..." (no headers)
    #[error("too many requests")]
    RateLimited {
        message: String,
        retry_after_secs: u64,
        limit: u32,
        remaining: u32,
        reset_secs: u64,
    },
    #[error("service unavailable: {0}")]
    ServiceUnavailable(String),
    #[error("internal server error: {0}")]
    Internal(String),
}

impl AppError {
    /// Internal errors carry a server-side detail that is logged but never
    /// sent to the client.
    pub fn internal(detail: impl Into<String>) -> Self {
        AppError::Internal(detail.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let request_id = correlation_id();
        let (status, error_code, message) = match &self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg.clone()),
            AppError::TooManyRequests(msg) => (
                StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                msg.clone(),
            ),
            AppError::RateLimited { message, .. } => (
                StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                message.clone(),
            ),
            AppError::ServiceUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                msg.clone(),
            ),
            AppError::Internal(detail) => {
                // Log the detail server-side; the client only gets a stable
                // message plus a correlation id.
                tracing::error!(request_id = %request_id, detail = %detail, "internal error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_server_error",
                    "An internal error occurred.".to_string(),
                )
            }
        };

        let body = Json(json!({
            "error": error_code,
            "message": message,
            "status": status.as_u16(),
            "request_id": request_id,
        }));

        let mut response = (status, body).into_response();
        // Token-bucket rejections carry live retry metadata; the other 429
        // sources (inference/queue saturation) intentionally have no headers.
        if let AppError::RateLimited {
            retry_after_secs,
            limit,
            remaining,
            reset_secs,
            ..
        } = &self
        {
            let headers = response.headers_mut();
            headers.insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_str(&(*retry_after_secs).max(1).to_string())
                    .unwrap_or_else(|_| axum::http::HeaderValue::from_static("1")),
            );
            for (name, value) in [
                ("x-ratelimit-limit", limit.to_string()),
                ("x-ratelimit-remaining", remaining.to_string()),
                ("x-ratelimit-reset", (*reset_secs).max(1).to_string()),
            ] {
                if let (Ok(name), Ok(value)) = (
                    axum::http::HeaderName::from_bytes(name.as_bytes()),
                    axum::http::HeaderValue::from_str(&value),
                ) {
                    headers.insert(name, value);
                }
            }
        }
        response
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::internal(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    async fn body_json(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn internal_errors_do_not_leak_details() {
        let secret_path = "/srv/secret/models/broken.onnx";
        let response = AppError::internal(format!("failed to open {secret_path}")).into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_json(response).await;
        let rendered = body.to_string();
        assert!(
            !rendered.contains(secret_path),
            "internal detail leaked: {rendered}"
        );
        assert_eq!(body["error"], "internal_server_error");
        assert!(body.get("request_id").is_some());
    }

    #[tokio::test]
    async fn rate_limit_maps_to_429() {
        let response = AppError::TooManyRequests("slow down".to_string()).into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn rate_limited_carries_retry_headers() {
        let response = AppError::RateLimited {
            message: "Rate limit exceeded. Try again later.".to_string(),
            retry_after_secs: 7,
            limit: 300,
            remaining: 0,
            reset_secs: 42,
        }
        .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let headers = response.headers();
        assert_eq!(headers.get("retry-after").unwrap(), "7");
        assert_eq!(headers.get("x-ratelimit-limit").unwrap(), "300");
        assert_eq!(headers.get("x-ratelimit-remaining").unwrap(), "0");
        assert_eq!(headers.get("x-ratelimit-reset").unwrap(), "42");
        let body = body_json(response).await;
        assert_eq!(body["error"], "too_many_requests");
        assert_eq!(body["status"], 429);
        assert!(body.get("request_id").is_some());
    }

    #[tokio::test]
    async fn saturation_429s_carry_no_rate_limit_headers() {
        for error in [
            AppError::TooManyRequests(
                "Server is busy. Too many pending inference requests.".to_string(),
            ),
            AppError::TooManyRequests("Playback queue is full. Try again later.".to_string()),
        ] {
            let response = error.into_response();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert!(
                response.headers().get("retry-after").is_none(),
                "saturation 429s must stay distinguishable from rate-limit 429s"
            );
        }
    }
}
