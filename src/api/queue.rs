use axum::{
    extract::State,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::AppState;
use crate::tts::queue::QueueStatus;

/// Request to add an audio file to the queue
#[derive(Debug, Deserialize)]
pub struct QueueAudioRequest {
    /// Unique identifier for the audio (will be returned in responses)
    pub id: Option<String>,
    /// Path to the audio file (absolute or relative to audio directory)
    pub path: String,
    /// If true, play immediately (clears queue)
    pub play_now: Option<bool>,
}

/// Request to control playback
#[derive(Debug, Deserialize)]
pub struct PlaybackControlRequest {
    /// Volume level (0.0 to 1.0)
    pub volume: Option<f32>,
}

/// Response for queue operations
#[derive(Debug, Serialize)]
pub struct QueueResponse {
    pub success: bool,
    pub message: String,
    pub id: Option<String>,
}

/// Add audio to the queue or play immediately
pub async fn queue_audio(
    State(state): State<AppState>,
    Json(req): Json<QueueAudioRequest>,
) -> Json<QueueResponse> {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return Json(QueueResponse {
                success: false,
                message: "Audio manager not initialized".to_string(),
                id: None,
            });
        }
    };

    let id = req.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let path = PathBuf::from(&req.path);

    // Check if file exists
    if !path.exists() {
        return Json(QueueResponse {
            success: false,
            message: format!("Audio file not found: {}", req.path),
            id: None,
        });
    }

    if req.play_now.unwrap_or(false) {
        audio_manager.play_now(id.clone(), path).await;
        Json(QueueResponse {
            success: true,
            message: "Playing immediately".to_string(),
            id: Some(id),
        })
    } else {
        audio_manager.add_to_queue(id.clone(), path).await;
        
        // Try to play next if nothing is playing
        let status = audio_manager.status().await;
        if !status.is_playing {
            audio_manager.play_next().await;
        }
        
        Json(QueueResponse {
            success: true,
            message: "Added to queue".to_string(),
            id: Some(id),
        })
    }
}

/// Play the next item in the queue
pub async fn play_next(State(state): State<AppState>) -> Json<QueueResponse> {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return Json(QueueResponse {
                success: false,
                message: "Audio manager not initialized".to_string(),
                id: None,
            });
        }
    };

    audio_manager.play_next().await;
    
    let status = audio_manager.status().await;
    Json(QueueResponse {
        success: status.is_playing,
        message: if status.is_playing { "Now playing".to_string() } else { "Queue is empty".to_string() },
        id: status.current.map(|c| c.id),
    })
}

/// Pause playback
pub async fn pause_audio(State(state): State<AppState>) -> Json<QueueResponse> {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return Json(QueueResponse {
                success: false,
                message: "Audio manager not initialized".to_string(),
                id: None,
            });
        }
    };

    audio_manager.pause().await;
    Json(QueueResponse {
        success: true,
        message: "Playback paused".to_string(),
        id: None,
    })
}

/// Resume playback
pub async fn resume_audio(State(state): State<AppState>) -> Json<QueueResponse> {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return Json(QueueResponse {
                success: false,
                message: "Audio manager not initialized".to_string(),
                id: None,
            });
        }
    };

    audio_manager.resume().await;
    Json(QueueResponse {
        success: true,
        message: "Playback resumed".to_string(),
        id: None,
    })
}

/// Stop playback and clear queue
pub async fn stop_audio(State(state): State<AppState>) -> Json<QueueResponse> {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return Json(QueueResponse {
                success: false,
                message: "Audio manager not initialized".to_string(),
                id: None,
            });
        }
    };

    audio_manager.stop().await;
    Json(QueueResponse {
        success: true,
        message: "Playback stopped and queue cleared".to_string(),
        id: None,
    })
}

/// Set volume
pub async fn set_volume(
    State(state): State<AppState>,
    Json(req): Json<PlaybackControlRequest>,
) -> Json<QueueResponse> {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return Json(QueueResponse {
                success: false,
                message: "Audio manager not initialized".to_string(),
                id: None,
            });
        }
    };

    let volume = req.volume.unwrap_or(1.0).clamp(0.0, 1.0);
    audio_manager.set_volume(volume).await;
    Json(QueueResponse {
        success: true,
        message: format!("Volume set to {}", volume),
        id: None,
    })
}

/// Get queue status
pub async fn get_queue_status(State(state): State<AppState>) -> Json<QueueStatus> {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return Json(QueueStatus {
                current: None,
                queue_length: 0,
                is_playing: false,
                is_paused: false,
                volume: 1.0,
            });
        }
    };

    Json(audio_manager.status().await)
}
