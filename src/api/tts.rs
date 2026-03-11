use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    auth::AuthenticatedToken,
    error::AppError,
    tts::{audio, inference, ModelStatus},
};

#[derive(Deserialize)]
pub struct TtsQuery {
    pub voice: Option<String>,
    pub lang: Option<String>,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub status: String,
    pub progress: Option<f32>,
    pub error: Option<String>,
}

pub async fn get_status(State(state): State<crate::AppState>) -> impl IntoResponse {
    let status = state.model_status.read().await;
    let response = match &*status {
        ModelStatus::Idle => StatusResponse {
            status: "idle".to_string(),
            progress: None,
            error: None,
        },
        ModelStatus::Downloading { progress } => StatusResponse {
            status: "downloading".to_string(),
            progress: Some(*progress * 100.0),
            error: None,
        },
        ModelStatus::Loading => StatusResponse {
            status: "loading".to_string(),
            progress: None,
            error: None,
        },
        ModelStatus::Ready(_) => StatusResponse {
            status: "ready".to_string(),
            progress: None,
            error: None,
        },
        ModelStatus::Failed(reason) => StatusResponse {
            status: "failed".to_string(),
            progress: None,
            error: Some(reason.clone()),
        },
    };

    (StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], serde_json::to_string(&response).unwrap())
}

pub async fn post_tts(
    _token: AuthenticatedToken,
    State(state): State<crate::AppState>,
    Query(query): Query<TtsQuery>,
    body: String,
) -> Result<Response, AppError> {
    if body.trim().is_empty() {
        return Err(AppError::BadRequest("Text cannot be empty.".to_string()));
    }

    let model_handle = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Ready(handle) => Arc::clone(handle),
            ModelStatus::Downloading { progress } => {
                return Err(AppError::ServiceUnavailable(format!(
                    "Model is downloading ({:.0}% complete).",
                    progress * 100.0
                )));
            }
            ModelStatus::Loading => {
                return Err(AppError::ServiceUnavailable(
                    "Model is loading.".to_string(),
                ));
            }
            ModelStatus::Idle => {
                return Err(AppError::ServiceUnavailable(
                    "Model has not started loading yet.".to_string(),
                ));
            }
            ModelStatus::Failed(reason) => {
                return Err(AppError::Internal(format!("Model failed to load: {reason}")));
            }
        }
    };

    let text = body.trim().to_string();
    let voice_name = match query.voice {
        Some(ref v) if model_handle.voice_styles.contains_key(v.as_str()) => v.clone(),
        _ => model_handle
            .default_voice()
            .ok_or_else(|| AppError::Internal("No voice styles available.".to_string()))?
            .to_string(),
    };
    let lang = query.lang.unwrap_or_else(|| "en".to_string());
    let sample_rate = model_handle.sample_rate();
    let inference_steps = state.config.inference_steps;

    let samples = tokio::task::spawn_blocking(move || {
        inference::synthesize(&model_handle, &text, &lang, &voice_name, inference_steps)
    })
    .await
    .map_err(|e| AppError::Internal(e.to_string()))?
    .map_err(|e| AppError::Internal(e.to_string()))?;

    let opus_bytes = audio::encode_opus(&samples, sample_rate)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "audio/opus")],
        Body::from(opus_bytes),
    )
        .into_response())
}
