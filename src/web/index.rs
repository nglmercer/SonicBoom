use axum::{
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::{AppState, tts::ModelStatus};

// Load the HTML template at compile time
const TEMPLATE: &str = include_str!("../../templates/index.html");

/// Liveness: the process is alive. Always 200 when the server can respond.
/// Used by container `HEALTHCHECK` / load-balancer liveness probes.
pub async fn get_health() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// Readiness: the model is loaded and the server can synthesize.
/// Used by load-balancer readiness probes and orchestrators.
pub async fn get_ready(State(state): State<AppState>) -> impl IntoResponse {
    let status = state.model_status.read().await;
    match &*status {
        ModelStatus::Ready(_) => (StatusCode::OK, "ready"),
        ModelStatus::Downloading { .. } => (StatusCode::SERVICE_UNAVAILABLE, "model downloading"),
        ModelStatus::Loading => (StatusCode::SERVICE_UNAVAILABLE, "model loading"),
        ModelStatus::Idle => (StatusCode::SERVICE_UNAVAILABLE, "model idle"),
        ModelStatus::Failed(_) => (StatusCode::SERVICE_UNAVAILABLE, "model failed to load"),
    }
}

pub async fn get_index(State(state): State<AppState>) -> Response {
    // Collect available voices
    let voices: Vec<String> = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Ready(handle) => {
                let mut names: Vec<String> = handle.voice_styles.keys().cloned().collect();
                names.sort();
                names
            }
            _ => vec![],
        }
    };

    let status_msg = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Idle => "Preparing model...".to_string(),
            ModelStatus::Downloading { progress } => {
                format!("Downloading model... ({:.0}%)", progress * 100.0)
            }
            ModelStatus::Loading => "Loading model...".to_string(),
            ModelStatus::Ready(_) => String::new(),
            // Never render internal load errors into the page.
            ModelStatus::Failed(_) => "Model load failed. Check server logs.".to_string(),
        }
    };

    let model_ready = !voices.is_empty();

    let voice_options: String = voices
        .iter()
        .map(|v| format!(r#"<option value="{v}">{v}</option>"#))
        .collect::<Vec<_>>()
        .join("\n");

    let html = TEMPLATE
        .replace("STATUS_MSG", &status_msg)
        .replace(
            "BLOCK_STATUS",
            if status_msg.is_empty() {
                "none"
            } else {
                "block"
            },
        )
        .replace(
            "VOICE_OPTIONS",
            if voice_options.is_empty() {
                "<option value=\"\">No voices available</option>"
            } else {
                &voice_options
            },
        )
        .replace("VOICE_DISABLED", if !model_ready { "disabled" } else { "" })
        .replace("BTN_DISABLED", if !model_ready { "disabled" } else { "" })
        .replace("MODEL_READY_JS", if model_ready { "true" } else { "false" });

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        html,
    )
        .into_response()
}
