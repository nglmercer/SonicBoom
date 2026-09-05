#![forbid(unsafe_code)]
#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime},
};

use serde_json::{Map, Value};
use sonicboom::engine::{EngineConfig, SonicBoomEngine, SynthesisRequest};
use tiktools_plugin_sdk::prelude::*;

const DEFAULT_INFERENCE_STEPS: usize = 5;
const MAX_GENERATED_BYTES: u64 = 256 * 1024 * 1024;
const GENERATED_MAX_AGE: Duration = Duration::from_secs(60 * 60);
const PROGRESS_EVENT_TYPE: &str = "plugin.progress";

#[derive(Clone)]
enum EngineState {
    Idle,
    Downloading { progress: f32 },
    Loading,
    Ready(Arc<SonicBoomEngine>),
    Failed(String),
}

#[derive(Clone, PartialEq)]
struct ProgressSnapshot {
    status: &'static str,
    progress_percent: Option<u8>,
    message: String,
}

struct SonicBoomPlugin {
    state: Arc<Mutex<EngineState>>,
    generated_dir: Option<PathBuf>,
    last_progress: Option<ProgressSnapshot>,
    _preparation_thread: Option<thread::JoinHandle<()>>,
}

impl Default for SonicBoomPlugin {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(EngineState::Idle)),
            generated_dir: None,
            last_progress: None,
            _preparation_thread: None,
        }
    }
}

impl Plugin for SonicBoomPlugin {
    fn initialize(&mut self, _context: &PluginContext) -> PluginResult<()> {
        let data_dir = tiktools_plugin_sdk::process::data_dir()?;
        let generated_dir = data_dir.join("generated");
        fs::create_dir_all(data_dir.join("models")).map_err(|error| {
            PluginError::other(format!("could not create model directory: {error}"))
        })?;
        fs::create_dir_all(&generated_dir).map_err(|error| {
            PluginError::other(format!("could not create generated directory: {error}"))
        })?;
        cleanup_generated(&generated_dir);
        self.generated_dir = Some(generated_dir);
        self.start_preparation_if_needed()?;
        Ok(())
    }

    fn action(&mut self, _context: &PluginContext, call: ActionCall) -> PluginResult<ActionResult> {
        match call.action_type() {
            Some("sonicboom.tts.prepare") => self.prepare_action(),
            Some("sonicboom.tts.speak") => self.speak_action(call),
            Some(action) => Err(PluginError::unsupported(action)),
            None => Err(PluginError::invalid_request("action has no typeId")),
        }
    }

    fn poll(&mut self, _context: &PluginContext) -> PluginResult<PollResult> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::other("engine state lock poisoned"))?
            .clone();
        let Some(snapshot) = progress_snapshot(&state) else {
            return Ok(PollResult::default());
        };
        if self.last_progress.as_ref() == Some(&snapshot) {
            return Ok(PollResult::default());
        }
        self.last_progress = Some(snapshot.clone());
        Ok(PollResult::default().event(PluginEvent::new(
            PROGRESS_EVENT_TYPE,
            serde_json::json!({
                "status": snapshot.status,
                "progress": snapshot.progress_percent.map(|value| f32::from(value) / 100.0),
                "message": snapshot.message,
            }),
        )?))
    }
}

impl SonicBoomPlugin {
    fn start_preparation_if_needed(&mut self) -> PluginResult<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::other("engine state lock poisoned"))?;
        if !matches!(*state, EngineState::Idle) {
            return Ok(());
        }
        let data_dir = tiktools_plugin_sdk::process::data_dir()?;
        let model_dir = data_dir.join("models");
        let state_for_worker = Arc::clone(&self.state);
        *state = EngineState::Downloading { progress: 0.0 };
        drop(state);

        self._preparation_thread = Some(
            thread::Builder::new()
                .name("sonicboom-model-preparation".to_owned())
                .spawn(move || {
                    let runtime = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            set_failed(&state_for_worker, error.to_string());
                            return;
                        }
                    };
                    let config = EngineConfig::new(model_dir);
                    let state_for_progress = Arc::clone(&state_for_worker);
                    let result =
                        runtime.block_on(SonicBoomEngine::prepare(config, move |progress| {
                            let next = if progress >= 1.0 {
                                EngineState::Loading
                            } else {
                                EngineState::Downloading { progress }
                            };
                            if let Ok(mut state) = state_for_progress.lock() {
                                *state = next;
                            }
                        }));
                    match result {
                        Ok(engine) => {
                            if let Ok(mut state) = state_for_worker.lock() {
                                *state = EngineState::Ready(engine);
                            }
                        }
                        Err(error) => set_failed(&state_for_worker, error.to_string()),
                    }
                })
                .map_err(|error| {
                    PluginError::other(format!("could not start model worker: {error}"))
                })?,
        );
        Ok(())
    }

    fn prepare_action(&mut self) -> PluginResult<ActionResult> {
        self.start_preparation_if_needed()?;
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::other("engine state lock poisoned"))?
            .clone();
        match state {
            EngineState::Idle => Ok(ActionResult::summary(
                "SonicBoom model preparation started.",
            )),
            EngineState::Downloading { progress } => Ok(ActionResult::summary(format!(
                "Downloading SonicBoom model: {:.0}%.",
                progress * 100.0
            ))),
            EngineState::Loading => Ok(ActionResult::summary("SonicBoom model is loading.")),
            EngineState::Ready(_) => Ok(ActionResult::summary("SonicBoom model is ready.")),
            EngineState::Failed(error) => Err(PluginError::other(format!(
                "SonicBoom model failed to prepare: {error}"
            ))),
        }
    }

    fn speak_action(&mut self, call: ActionCall) -> PluginResult<ActionResult> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::other("engine state lock poisoned"))?
            .clone();
        let engine = match state {
            EngineState::Ready(engine) => engine,
            EngineState::Downloading { progress } => {
                return Err(PluginError::other(format!(
                    "SonicBoom model is downloading ({:.0}%).",
                    progress * 100.0
                )));
            }
            EngineState::Loading => {
                return Err(PluginError::other("SonicBoom model is loading."));
            }
            EngineState::Failed(error) => {
                return Err(PluginError::other(format!(
                    "SonicBoom model failed to load: {error}"
                )));
            }
            EngineState::Idle => {
                return Err(PluginError::other("SonicBoom model is not ready."));
            }
        };

        let config = call.config();
        let text = required_string(config, "text")?;
        let voice = optional_string(config, "voice").unwrap_or_else(|| "F1".to_owned());
        let language = optional_string(config, "language").unwrap_or_else(|| "en".to_owned());
        let volume = optional_number(config, "volume").unwrap_or(1.0);
        if !volume.is_finite() || !(0.0..=1.0).contains(&volume) {
            return Err(PluginError::invalid_request(
                "volume must be between 0 and 1",
            ));
        }
        let overlap = match optional_string(config, "overlap").as_deref() {
            None | Some("allow") => AudioOverlap::Allow,
            Some("restart") => AudioOverlap::Restart,
            Some("drop") => AudioOverlap::Drop,
            Some(value) => {
                return Err(PluginError::invalid_request(format!(
                    "unsupported overlap mode: {value}"
                )));
            }
        };
        let inference_steps = optional_u64(config, "inferenceSteps")
            .unwrap_or(DEFAULT_INFERENCE_STEPS as u64) as usize;
        let request = SynthesisRequest {
            text,
            language,
            voice,
            inference_steps,
        };
        request
            .validate()
            .map_err(|error| PluginError::invalid_request(error.to_string()))?;

        let result = engine
            .synthesize(&request)
            .map_err(|error| PluginError::other(format!("speech synthesis failed: {error}")))?;
        let generated_dir = self
            .generated_dir
            .as_deref()
            .ok_or_else(|| PluginError::other("plugin data directory is not initialized"))?;
        cleanup_generated(generated_dir);
        let path = write_generated_wav(generated_dir, &result.wav)?;
        cleanup_generated(generated_dir);

        Ok(
            ActionResult::summary("Speech synthesized").intent(HostIntent::audio_play(
                AudioPlayIntent::from_path(path.to_string_lossy().into_owned())
                    .volume(volume)
                    .overlap(overlap),
            )),
        )
    }
}

fn set_failed(state: &Arc<Mutex<EngineState>>, error: String) {
    eprintln!("SonicBoom model preparation failed: {error}");
    if let Ok(mut state) = state.lock() {
        *state = EngineState::Failed(error);
    }
}

fn progress_snapshot(state: &EngineState) -> Option<ProgressSnapshot> {
    match state {
        EngineState::Idle => None,
        EngineState::Downloading { progress } => Some(ProgressSnapshot {
            status: "downloading",
            progress_percent: Some((progress.clamp(0.0, 1.0) * 100.0).round() as u8),
            message: format!("Downloading SonicBoom model: {:.0}%.", progress * 100.0),
        }),
        EngineState::Loading => Some(ProgressSnapshot {
            status: "loading",
            progress_percent: None,
            message: "Loading SonicBoom model.".to_owned(),
        }),
        EngineState::Ready(_) => Some(ProgressSnapshot {
            status: "ready",
            progress_percent: Some(100),
            message: "SonicBoom model is ready.".to_owned(),
        }),
        EngineState::Failed(error) => Some(ProgressSnapshot {
            status: "failed",
            progress_percent: None,
            message: format!("SonicBoom model failed to prepare: {error}"),
        }),
    }
}

fn required_string(config: &Map<String, Value>, key: &str) -> PluginResult<String> {
    config
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| PluginError::invalid_request(format!("missing or empty {key}")))
}

fn optional_string(config: &Map<String, Value>, key: &str) -> Option<String> {
    config
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn optional_number(config: &Map<String, Value>, key: &str) -> Option<f32> {
    config.get(key).and_then(|value| match value {
        Value::Number(value) => value.as_f64().map(|value| value as f32),
        Value::String(value) => value.parse().ok(),
        _ => None,
    })
}

fn optional_u64(config: &Map<String, Value>, key: &str) -> Option<u64> {
    config.get(key).and_then(|value| match value {
        Value::Number(value) => value.as_u64(),
        Value::String(value) => value.parse().ok(),
        _ => None,
    })
}

fn write_generated_wav(generated_dir: &Path, bytes: &[u8]) -> PluginResult<PathBuf> {
    let id = uuid::Uuid::new_v4().simple().to_string();
    let path = generated_dir.join(format!("{id}.wav"));
    let temporary = generated_dir.join(format!("{id}.wav.tmp"));
    fs::write(&temporary, bytes)
        .map_err(|error| PluginError::other(format!("could not write generated audio: {error}")))?;
    fs::rename(&temporary, &path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        PluginError::other(format!("could not finalize generated audio: {error}"))
    })?;
    Ok(path)
}

fn cleanup_generated(generated_dir: &Path) {
    let Ok(root) = fs::canonicalize(generated_dir) else {
        return;
    };
    let now = SystemTime::now();
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(generated_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(canonical) = fs::canonicalize(&path) else {
            continue;
        };
        if !canonical.starts_with(&root) || !path.is_file() {
            continue;
        }
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        if path.extension().and_then(|extension| extension.to_str()) == Some("tmp") {
            let _ = fs::remove_file(&path);
            continue;
        }
        let modified = metadata.modified().unwrap_or(now);
        if now.duration_since(modified).unwrap_or_default() > GENERATED_MAX_AGE {
            let _ = fs::remove_file(&path);
            continue;
        }
        if path.extension().and_then(|extension| extension.to_str()) == Some("wav") {
            files.push((path, modified, metadata.len()));
        }
    }
    let mut total = files.iter().map(|(_, _, size)| *size).sum::<u64>();
    files.sort_by_key(|(_, modified, _)| *modified);
    for (path, _, size) in files {
        if total <= MAX_GENERATED_BYTES {
            break;
        }
        if fs::remove_file(path).is_ok() {
            total = total.saturating_sub(size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_audio_is_finalized_without_leaving_temporary_files() {
        let root = std::env::temp_dir().join(format!(
            "sonicboom-tiktools-plugin-test-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let generated_dir = root.join("generated");
        fs::create_dir_all(&generated_dir).expect("create generated test directory");

        let temporary = generated_dir.join("orphan.wav.tmp");
        fs::write(&temporary, b"stale").expect("write stale temporary file");
        let path = write_generated_wav(&generated_dir, b"RIFF-test").expect("write wav");

        assert_eq!(fs::read(&path).expect("read finalized wav"), b"RIFF-test");
        assert!(path.parent().is_some_and(|parent| parent == generated_dir));

        cleanup_generated(&generated_dir);
        assert!(!temporary.exists());
        fs::remove_dir_all(root).expect("remove generated test directory");
    }
}

tiktools_process_plugin!(SonicBoomPlugin);
