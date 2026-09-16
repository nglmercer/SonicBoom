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

        (status, body).into_response()
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
}
