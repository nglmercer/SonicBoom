//! Server audio output device selection API (playback builds only).
//!
//! SonicBoom owns device discovery because HTTP clients (e.g. TikTools GUIs)
//! cannot enumerate the server host's audio devices themselves. All device
//! manipulation runs inside the dedicated playback thread; these handlers
//! only exchange plain data with it via [`AudioManager`].
//!
//! Runtime selection via `POST /api/audio/output` is process-local: it is
//! not persisted anywhere (SonicBoom never rewrites `.env`). The startup
//! selection comes from `AUDIO_OUTPUT_DEVICE`.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::tts::queue::AudioManagerError;
use crate::{AppState, auth::AuthenticatedToken};

/// Request body for `POST /api/audio/output`.
#[derive(Debug, Deserialize)]
pub struct SetOutputRequest {
    /// `default` (OS default output) or an explicit device id from
    /// `GET /api/audio/devices`.
    pub device: Option<String>,
}

/// Success body for `POST /api/audio/output`.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct SetOutputResponse {
    pub success: bool,
    /// Committed selection (`default` or the explicit id).
    pub device: String,
    /// Resolved hardware device name behind the committed selection.
    pub resolved_name: Option<String>,
}

fn failure(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "success": false,
            "message": message,
        })),
    )
        .into_response()
}

fn audio_manager(state: &AppState) -> Option<&crate::tts::queue::AudioManager> {
    state.audio_manager.as_ref().as_ref()
}

fn no_audio_manager() -> Response {
    failure(
        StatusCode::SERVICE_UNAVAILABLE,
        "Audio manager not initialized",
    )
}

/// `GET /api/audio/devices`: list output devices plus the current selection.
///
/// Output devices only (never input-only), always with the logical `default`
/// entry first. Enumeration failure is `500`, never an empty success.
pub async fn list_output_devices(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    let audio_manager = match audio_manager(&state) {
        Some(manager) => manager,
        None => return no_audio_manager(),
    };
    match audio_manager.output_devices().await {
        Ok(list) => (StatusCode::OK, Json(list)).into_response(),
        Err(AudioManagerError::DeviceError) => {
            tracing::error!("GET /api/audio/devices: OS enumeration failed");
            failure(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Audio device enumeration failed.",
            )
        }
        Err(_) => failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Audio playback system unavailable.",
        ),
    }
}

/// `GET /api/audio/output`: configured selection versus live hardware state.
pub async fn get_output_device(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    let audio_manager = match audio_manager(&state) {
        Some(manager) => manager,
        None => return no_audio_manager(),
    };
    match audio_manager.output_device().await {
        Ok(active) => (StatusCode::OK, Json(active)).into_response(),
        Err(_) => failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Audio playback system unavailable.",
        ),
    }
}

/// `POST /api/audio/output`: switch the output device at runtime.
///
/// Unknown devices are `400` with no fallback to another device; a failed
/// switch leaves the previous working output active. Process-local: the
/// selection does not survive restarts (see `AUDIO_OUTPUT_DEVICE`).
pub async fn set_output_device(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
    Json(request): Json<SetOutputRequest>,
) -> Response {
    let audio_manager = match audio_manager(&state) {
        Some(manager) => manager,
        None => return no_audio_manager(),
    };
    let device = request.device.unwrap_or_default();
    if device.trim().is_empty() {
        return failure(StatusCode::BAD_REQUEST, "device must not be empty");
    }
    if device.chars().count() > crate::config::MAX_AUDIO_OUTPUT_DEVICE_LEN {
        return failure(StatusCode::BAD_REQUEST, "device name is too long");
    }
    if device.chars().any(|c| c.is_control()) {
        return failure(
            StatusCode::BAD_REQUEST,
            "device must not contain control characters",
        );
    }
    match audio_manager.set_output_device(device).await {
        Ok(active) => (
            StatusCode::OK,
            Json(SetOutputResponse {
                success: true,
                device: active.device,
                resolved_name: active.resolved_name,
            }),
        )
            .into_response(),
        Err(AudioManagerError::UnknownDevice) => {
            failure(StatusCode::BAD_REQUEST, "Unknown audio output device.")
        }
        Err(AudioManagerError::OutputUnavailable) => {
            failure(StatusCode::SERVICE_UNAVAILABLE, "Audio output unavailable.")
        }
        Err(AudioManagerError::DeviceError) => {
            tracing::error!("POST /api/audio/output: OS enumeration failed");
            failure(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Audio device enumeration failed.",
            )
        }
        Err(_) => failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Audio playback system unavailable.",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_response_serializes_with_resolved_name() {
        let response = SetOutputResponse {
            success: true,
            device: "CABLE Input".to_string(),
            resolved_name: Some("CABLE Input".to_string()),
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["success"], true);
        assert_eq!(json["device"], "CABLE Input");
        assert_eq!(json["resolved_name"], "CABLE Input");
    }

    #[test]
    fn set_request_requires_device_field_shape() {
        let request: SetOutputRequest = serde_json::from_str(r#"{"device":"default"}"#).unwrap();
        assert_eq!(request.device.as_deref(), Some("default"));
        let request: SetOutputRequest = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(request.device, None);
    }
}
