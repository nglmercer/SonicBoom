use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::tts::queue::{AudioManagerError, QueueStatus};
use crate::{AppState, auth::AuthenticatedToken};

/// Request to add an audio file to the queue
#[derive(Debug, Deserialize)]
pub struct QueueAudioRequest {
    /// Unique identifier for the audio (will be returned in responses)
    pub id: Option<String>,
    /// Path to the audio file (must be inside `ALLOWED_AUDIO_DIR`)
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

fn failure(status: StatusCode, message: &str) -> (StatusCode, Json<QueueResponse>) {
    (
        status,
        Json(QueueResponse {
            success: false,
            message: message.to_string(),
            id: None,
        }),
    )
}

/// Validate a user-supplied queue id: 1-128 chars, no control characters.
pub fn validate_queue_id(id: &str) -> Result<(), &'static str> {
    let len = id.chars().count();
    if len == 0 || len > 128 {
        return Err("queue id must be 1-128 characters");
    }
    if id.chars().any(|c| c.is_control()) {
        return Err("queue id must not contain control characters");
    }
    Ok(())
}

/// Validate a caller-supplied volume: `0.0..=1.0`, finite. Out-of-range or
/// non-finite values are caller errors (`400`), never silently clamped —
/// clamping stays only as a defensive measure inside the sink code.
pub fn validate_volume(volume: f32) -> Result<f32, &'static str> {
    if !volume.is_finite() {
        return Err("volume must be a finite number");
    }
    if !(0.0..=1.0).contains(&volume) {
        return Err("volume must be between 0.0 and 1.0");
    }
    Ok(volume)
}

/// Map a playback-subsystem failure to a stable API response: a dead audio
/// thread is `503`, a full queue is `429`.
fn audio_error_response(error: AudioManagerError) -> (StatusCode, Json<QueueResponse>) {
    match error {
        AudioManagerError::QueueFull => failure(
            StatusCode::TOO_MANY_REQUESTS,
            "Playback queue is full. Try again later.",
        ),
        AudioManagerError::Unavailable => failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Audio playback system unavailable.",
        ),
    }
}

const ALLOWED_AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg", "opus"];

/// Resolve `requested` strictly inside `allowed_dir`.
///
/// - Both sides are canonicalized; symlinks cannot escape the root.
/// - The target must exist and be a regular file (no directories).
/// - The extension must be an allowed audio format.
///
/// Error messages never echo full filesystem paths.
pub fn resolve_inside_allowed_dir(
    allowed_dir: &str,
    requested: &str,
) -> Result<PathBuf, &'static str> {
    let allowed_root = Path::new(allowed_dir)
        .canonicalize()
        .map_err(|_| "server audio directory is misconfigured")?;
    let candidate = Path::new(requested);
    // Resolve relative paths against the allowed root so `path` cannot be
    // interpreted relative to the server's working directory.
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        allowed_root.join(candidate)
    };
    let canonical = joined
        .canonicalize()
        .map_err(|_| "audio file not found or not accessible")?;
    if !canonical.starts_with(&allowed_root) {
        return Err("access denied: path outside allowed directory");
    }
    let metadata = std::fs::metadata(&canonical).map_err(|_| "audio file not accessible")?;
    if !metadata.is_file() {
        return Err("audio path must be a regular file");
    }
    let extension_ok = canonical
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            ALLOWED_AUDIO_EXTENSIONS
                .iter()
                .any(|allowed| ext.eq_ignore_ascii_case(allowed))
        });
    if !extension_ok {
        return Err("unsupported audio file extension");
    }
    Ok(canonical)
}

/// Add audio to the queue or play immediately
pub async fn queue_audio(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
    Json(req): Json<QueueAudioRequest>,
) -> impl IntoResponse {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Audio manager not initialized",
            );
        }
    };

    let allowed_dir = match state.config.allowed_audio_dir.as_deref() {
        Some(dir) => dir,
        None => {
            tracing::error!("filesystem queue used without ALLOWED_AUDIO_DIR");
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Filesystem queue is not configured on this server",
            );
        }
    };

    let id = req.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    if let Err(message) = validate_queue_id(&id) {
        return failure(StatusCode::BAD_REQUEST, message);
    }

    let path = match resolve_inside_allowed_dir(allowed_dir, &req.path) {
        Ok(path) => path,
        Err(message) => {
            let status = if message == "server audio directory is misconfigured" {
                StatusCode::INTERNAL_SERVER_ERROR
            } else if message == "access denied: path outside allowed directory" {
                StatusCode::FORBIDDEN
            } else {
                StatusCode::BAD_REQUEST
            };
            return failure(status, message);
        }
    };

    if req.play_now.unwrap_or(false) {
        if let Err(error) = audio_manager.play_now(id.clone(), path).await {
            return audio_error_response(error);
        }
        (
            StatusCode::OK,
            Json(QueueResponse {
                success: true,
                message: "Playing immediately".to_string(),
                id: Some(id),
            }),
        )
    } else {
        if let Err(error) = audio_manager.add_to_queue(id.clone(), path).await {
            return audio_error_response(error);
        }

        (
            StatusCode::OK,
            Json(QueueResponse {
                success: true,
                message: "Added to queue".to_string(),
                id: Some(id),
            }),
        )
    }
}

/// Play the next item in the queue
pub async fn play_next(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Audio manager not initialized",
            );
        }
    };

    if let Err(error) = audio_manager.play_next().await {
        return audio_error_response(error);
    }

    let status = match audio_manager.status().await {
        Ok(status) => status,
        Err(error) => return audio_error_response(error),
    };
    (
        StatusCode::OK,
        Json(QueueResponse {
            success: status.is_playing,
            message: if status.is_playing {
                "Now playing".to_string()
            } else {
                "Queue is empty".to_string()
            },
            id: status.current.map(|c| c.id),
        }),
    )
}

/// Pause playback
pub async fn pause_audio(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Audio manager not initialized",
            );
        }
    };

    if let Err(error) = audio_manager.pause().await {
        return audio_error_response(error);
    }
    (
        StatusCode::OK,
        Json(QueueResponse {
            success: true,
            message: "Playback paused".to_string(),
            id: None,
        }),
    )
}

/// Resume playback
pub async fn resume_audio(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Audio manager not initialized",
            );
        }
    };

    if let Err(error) = audio_manager.resume().await {
        return audio_error_response(error);
    }
    (
        StatusCode::OK,
        Json(QueueResponse {
            success: true,
            message: "Playback resumed".to_string(),
            id: None,
        }),
    )
}

/// Stop playback and clear queue
pub async fn stop_audio(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Audio manager not initialized",
            );
        }
    };

    if let Err(error) = audio_manager.stop().await {
        return audio_error_response(error);
    }
    (
        StatusCode::OK,
        Json(QueueResponse {
            success: true,
            message: "Playback stopped and queue cleared".to_string(),
            id: None,
        }),
    )
}

/// Set volume
pub async fn set_volume(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
    Json(req): Json<PlaybackControlRequest>,
) -> impl IntoResponse {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Audio manager not initialized",
            );
        }
    };

    let volume = match validate_volume(req.volume.unwrap_or(1.0)) {
        Ok(volume) => volume,
        Err(message) => return failure(StatusCode::BAD_REQUEST, message),
    };
    if let Err(error) = audio_manager.set_volume(volume).await {
        return audio_error_response(error);
    }
    (
        StatusCode::OK,
        Json(QueueResponse {
            success: true,
            message: format!("Volume set to {volume}"),
            id: None,
        }),
    )
}

/// Get queue status
pub async fn get_queue_status(
    _token: AuthenticatedToken,
    State(state): State<AppState>,
) -> Response {
    let audio_manager = match &*state.audio_manager {
        Some(manager) => manager,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(QueueStatus {
                    current: None,
                    queue_length: 0,
                    is_playing: false,
                    is_paused: false,
                    volume: 1.0,
                }),
            )
                .into_response();
        }
    };

    match audio_manager.status().await {
        Ok(status) => (StatusCode::OK, Json(status)).into_response(),
        Err(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(QueueResponse {
                success: false,
                message: "Audio playback system unavailable.".to_string(),
                id: None,
            }),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let id = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir()
                .join(format!("sonicboom-queue-test-{}-{id}", std::process::id()));
            std::fs::create_dir_all(path.join("allowed")).unwrap();
            std::fs::create_dir_all(path.join("outside")).unwrap();
            Self { path }
        }

        fn allowed(&self) -> PathBuf {
            self.path.join("allowed")
        }

        fn file(&self, name: &str) -> PathBuf {
            let path = self.allowed().join(name);
            std::fs::write(&path, b"fake audio").unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn file_inside_allowed_dir_is_allowed() {
        let dir = TempDir::new();
        let file = dir.file("song.wav");
        let resolved =
            resolve_inside_allowed_dir(dir.allowed().to_str().unwrap(), file.to_str().unwrap())
                .unwrap();
        assert!(resolved.starts_with(dir.allowed().canonicalize().unwrap()));
    }

    #[test]
    fn relative_path_resolves_inside_allowed_dir() {
        let dir = TempDir::new();
        dir.file("song.mp3");
        assert!(resolve_inside_allowed_dir(dir.allowed().to_str().unwrap(), "song.mp3").is_ok());
    }

    #[test]
    fn dotdot_escape_is_denied() {
        let dir = TempDir::new();
        let outside = dir.path.join("outside").join("evil.wav");
        std::fs::write(&outside, b"fake audio").unwrap();
        let attack = dir.allowed().join("..").join("outside").join("evil.wav");
        let err =
            resolve_inside_allowed_dir(dir.allowed().to_str().unwrap(), attack.to_str().unwrap())
                .unwrap_err();
        assert_eq!(err, "access denied: path outside allowed directory");
    }

    #[test]
    fn absolute_outside_path_is_denied() {
        let dir = TempDir::new();
        let err = resolve_inside_allowed_dir(dir.allowed().to_str().unwrap(), "/etc/hostname")
            .unwrap_err();
        assert!(
            err == "access denied: path outside allowed directory"
                || err == "audio file not found or not accessible"
                || err == "unsupported audio file extension",
            "unexpected: {err}"
        );
        assert!(!err.contains("/etc/hostname"), "path leaked: {err}");
    }

    #[test]
    fn missing_file_is_denied() {
        let dir = TempDir::new();
        let err =
            resolve_inside_allowed_dir(dir.allowed().to_str().unwrap(), "nope.wav").unwrap_err();
        assert_eq!(err, "audio file not found or not accessible");
    }

    #[test]
    fn directory_is_denied() {
        let dir = TempDir::new();
        let sub = dir.allowed().join("sub.wav");
        std::fs::create_dir_all(&sub).unwrap();
        let err =
            resolve_inside_allowed_dir(dir.allowed().to_str().unwrap(), sub.to_str().unwrap())
                .unwrap_err();
        assert_eq!(err, "audio path must be a regular file");
    }

    #[test]
    fn unsupported_extension_is_denied() {
        let dir = TempDir::new();
        let path = dir.allowed().join("run.sh");
        std::fs::write(&path, b"echo hi").unwrap();
        let err =
            resolve_inside_allowed_dir(dir.allowed().to_str().unwrap(), path.to_str().unwrap())
                .unwrap_err();
        assert_eq!(err, "unsupported audio file extension");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_denied() {
        let dir = TempDir::new();
        let outside = dir.path.join("outside").join("secret.wav");
        std::fs::write(&outside, b"fake audio").unwrap();
        let link = dir.allowed().join("link.wav");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        let err =
            resolve_inside_allowed_dir(dir.allowed().to_str().unwrap(), link.to_str().unwrap())
                .unwrap_err();
        assert_eq!(err, "access denied: path outside allowed directory");
    }

    #[test]
    fn queue_ids_are_bounded() {
        assert!(validate_queue_id("abc-123").is_ok());
        assert!(validate_queue_id("").is_err());
        assert!(validate_queue_id(&"x".repeat(129)).is_err());
        assert!(validate_queue_id("a\nb").is_err());
        assert!(validate_queue_id("a\0b").is_err());
    }

    #[test]
    fn volume_accepts_only_finite_unit_range() {
        assert_eq!(validate_volume(0.0), Ok(0.0));
        assert_eq!(validate_volume(1.0), Ok(1.0));
        assert_eq!(validate_volume(0.5), Ok(0.5));
        for bad in [
            -0.1,
            1.1,
            -1.0,
            100.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert!(
                validate_volume(bad).is_err(),
                "volume {bad} must be rejected, not clamped"
            );
        }
    }

    #[test]
    fn playback_failures_map_to_stable_statuses() {
        let (status, _) = audio_error_response(AudioManagerError::QueueFull);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        let (status, _) = audio_error_response(AudioManagerError::Unavailable);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }
}
