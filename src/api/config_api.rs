//! Configuration REST API (spec §33–§37).
//!
//! Every operation delegates to the central
//! [`ConfigManager`]; nothing here mutates
//! configuration directly.
//!
//! - `GET    /api/config`          safe configuration
//! - `PATCH  /api/config`          partial update
//! - `GET    /api/config/schema`   machine-readable schema
//! - `GET    /api/config/status`   watcher/apply status
//! - `POST   /api/config/reload`   reload from disk

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::{AppState, config::ConfigUpdate};

/// Safe configuration snapshot for `GET /api/config`.
#[derive(Serialize)]
pub struct ConfigResponse {
    pub revision: u64,
    pub config: crate::config::AppConfig,
}

/// Request for `PATCH /api/config`.
#[derive(Deserialize)]
pub struct PatchRequest {
    /// Optimistic concurrency: reject the change
    /// when the current revision differs.
    #[serde(default)]
    pub expected_revision: Option<u64>,
    /// Partial configuration to merge into the
    /// current configuration.
    pub changes: serde_json::Value,
}

/// One applied change in a `PATCH` response.
#[derive(Serialize)]
pub struct AppliedChange {
    pub path: String,
    pub strategy: String,
    pub status: String,
}

/// Response for `PATCH /api/config`.
#[derive(Serialize)]
pub struct PatchResponse {
    pub success: bool,
    pub revision: u64,
    pub changes: Vec<AppliedChange>,
}

/// Response for `POST /api/config/reload`.
#[derive(Serialize)]
pub struct ReloadResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// `GET /api/config`
pub async fn get_config(
    _token: crate::auth::AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    let (revision, config) = state.config.snapshot().await;
    (StatusCode::OK, Json(ConfigResponse { revision, config })).into_response()
}

/// `PATCH /api/config`
///
/// The partial `changes` object is merged into the
/// current configuration, validated, and committed
/// through the [`ConfigManager`].
pub async fn patch_config(
    _token: crate::auth::AuthenticatedToken,
    State(state): State<AppState>,
    Json(request): Json<PatchRequest>,
) -> Response {
    if ["revision", "version", "setup_complete"]
        .iter()
        .any(|key| request.changes.get(*key).is_some())
    {
        return bad_request("revision, version and setup_complete are managed by SonicBoom");
    }
    if !request.changes.is_object() {
        return bad_request("changes must be an object");
    }
    // Merge the partial document into a copy of the
    // current configuration, then let the manager
    // validate and commit it.
    let (revision, current) = state.config.snapshot().await;
    let merged = match serde_json::to_value(&current) {
        Ok(value) => value,
        Err(e) => return server_error(e.to_string()),
    };
    let mut merged = merged;
    merge_json(&mut merged, &request.changes);
    let candidate = match serde_json::from_value(merged) {
        Ok(candidate) => candidate,
        Err(e) => return bad_request(format!("invalid configuration: {e}")),
    };
    let update = state
        .config
        .update(request.expected_revision.or(Some(revision)), |config| {
            *config = candidate;
        })
        .await;
    match update {
        Ok(ConfigUpdate { revision, changes }) => {
            let applied = changes
                .iter()
                .map(|change| AppliedChange {
                    path: change.path().to_string(),
                    strategy: change.strategy().as_str().to_string(),
                    status: "pending".to_string(),
                })
                .collect();
            (
                StatusCode::OK,
                Json(PatchResponse {
                    success: true,
                    revision,
                    changes: applied,
                }),
            )
                .into_response()
        }
        Err(e) => update_error(e),
    }
}

pub async fn open_config_file(
    _token: crate::auth::AuthenticatedToken,
    State(state): State<AppState>,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
) -> Response {
    if !peer.ip().is_loopback() {
        return StatusCode::FORBIDDEN.into_response();
    }
    #[cfg(feature = "gui")]
    {
        match open::that(state.config.path()) {
            Ok(()) => Json(serde_json::json!({"success": true})).into_response(),
            Err(error) => server_error(error.to_string()),
        }
    }
    #[cfg(not(feature = "gui"))]
    {
        let _ = state;
        StatusCode::NOT_IMPLEMENTED.into_response()
    }
}

pub async fn get_effective(
    _token: crate::auth::AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    Json(state.config.get_effective().await.describe()).into_response()
}

pub async fn validate_config_file(
    _token: crate::auth::AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    let candidate = crate::config::loader::load_from_file(state.config.path());
    match candidate {
        Ok(candidate) => match state.config.validate(&candidate.config).await {
            Ok(()) => Json(serde_json::json!({"valid": true})).into_response(),
            Err(error) => bad_request(error),
        },
        Err(error) => bad_request(error.to_string()),
    }
}

/// `GET /api/config/schema`
pub async fn get_schema(
    _token: crate::auth::AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    let _ = state;
    (StatusCode::OK, Json(crate::config::schema())).into_response()
}

/// `GET /api/config/status`
pub async fn get_status(
    _token: crate::auth::AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    let status = state.config.status().await;
    (StatusCode::OK, Json(status)).into_response()
}

/// `POST /api/config/reload`
pub async fn reload_config(
    _token: crate::auth::AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    match state.config.reload_from_disk().await {
        Ok(update) => (
            StatusCode::OK,
            Json(ReloadResponse {
                success: true,
                revision: Some(update.revision),
                error: None,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(ReloadResponse {
                success: false,
                revision: None,
                error: Some(e.to_string()),
            }),
        )
            .into_response(),
    }
}

/// Deeply merge `patch` into `base`.
fn merge_json(base: &mut serde_json::Value, patch: &serde_json::Value) {
    match (base, patch) {
        (serde_json::Value::Object(base_map), serde_json::Value::Object(patch_map)) => {
            for (key, value) in patch_map {
                let entry = base_map
                    .entry(key.clone())
                    .or_insert(serde_json::Value::Null);
                merge_json(entry, value);
            }
        }
        (base, patch) => *base = patch.clone(),
    }
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "success": false,
            "message": message.into(),
        })),
    )
        .into_response()
}

fn server_error(message: impl Into<String>) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "success": false,
            "message": message.into(),
        })),
    )
        .into_response()
}

fn update_error(e: crate::config::UpdateError) -> Response {
    match e {
        crate::config::UpdateError::Conflict { expected, current } => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "success": false,
                "message": "configuration conflict",
                "expected_revision": expected,
                "current_revision": current,
            })),
        )
            .into_response(),
        crate::config::UpdateError::Validation(e) => {
            bad_request(format!("invalid configuration: {e}"))
        }
        crate::config::UpdateError::Write(e) => {
            server_error(format!("failed to persist configuration: {e}"))
        }
    }
}
