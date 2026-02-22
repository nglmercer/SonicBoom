use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[allow(dead_code)]
    #[error("unauthorized")]
    Unauthorized,
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("service unavailable: {0}")]
    ServiceUnavailable(String),
    #[error("internal server error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_code, message) = match &self {
            AppError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                self.to_string(),
            ),
            AppError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "bad_request",
                msg.clone(),
            ),
            AppError::ServiceUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                msg.clone(),
            ),
            AppError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_server_error",
                msg.clone(),
            ),
        };

        let body = Json(json!({
            "error": error_code,
            "message": message,
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::Internal(err.to_string())
    }
}
