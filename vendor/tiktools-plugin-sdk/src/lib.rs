//! Developer-facing SDK for TikTools plugins.
//!
//! The SDK is intentionally runtime-neutral. It provides typed calls/results,
//! a small plugin trait, and adapters for the existing framed process and
//! native ABI boundaries. It does not depend on Tokio, the desktop crate,
//! Wry, Winit, or a particular WASM engine.

use std::{env, io};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;
use tiktools_plugin_api::{
    read_frame, write_frame, AudioOverlap, CapabilitySet, FrameError, MediaFileRef, PermissionSet,
    PluginBuffer, PluginRequest, PluginResponse, PluginStatus, MAX_FRAME_BYTES, METHOD_CALL,
    TIKTOOLS_PLUGIN_PROTOCOL_VERSION,
};

pub use tiktools_plugin_api;
pub use tiktools_plugin_macros::{tiktools_export_native_plugin, tiktools_process_plugin};

/// Errors exposed to plugin authors. Protocol adapters deliberately serialize
/// only this display text, never a Rust backtrace.
#[derive(Debug, Error)]
pub enum PluginError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("unsupported action: {0}")]
    UnsupportedAction(String),

    #[error("capability unavailable: {0}")]
    CapabilityUnavailable(String),

    #[error("plugin error: {0}")]
    Other(String),
}

impl PluginError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }

    pub fn unsupported(action: impl Into<String>) -> Self {
        Self::UnsupportedAction(action.into())
    }

    pub fn capability_unavailable(capability: impl Into<String>) -> Self {
        Self::CapabilityUnavailable(capability.into())
    }

    pub fn other(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }
}

pub type PluginResult<T> = Result<T, PluginError>;

/// A plugin's stable identity, independent of its runtime adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginIdentity {
    pub id: String,
    pub version: String,
}

impl PluginIdentity {
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
        }
    }
}

/// Runtime-neutral context supplied to plugin business logic.
///
/// File paths and host handles are intentionally absent. Future WASM
/// adapters can expose narrowly scoped host capabilities without changing the
/// trait or handing plugins arbitrary host internals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginContext {
    pub identity: PluginIdentity,
    /// Capabilities declared by the plugin manifest. These are not grants.
    pub declared_capabilities: CapabilitySet,
    /// Permissions declared by the plugin manifest. User grants are a
    /// separate policy layer and are intentionally not represented here yet.
    pub declared_permissions: PermissionSet,
}

impl PluginContext {
    pub fn new(
        identity: PluginIdentity,
        declared_capabilities: CapabilitySet,
        declared_permissions: PermissionSet,
    ) -> Self {
        Self {
            identity,
            declared_capabilities,
            declared_permissions,
        }
    }

    /// Builds the process context from the existing launcher contract.
    /// WASM adapters can construct the same shape from their manifest and
    /// explicit host policy without relying on environment variables.
    pub fn from_process_environment() -> Self {
        Self::new(
            PluginIdentity::new(
                env::var("TIKTOOLS_PLUGIN_ID").unwrap_or_else(|_| "unknown".to_owned()),
                env::var("TIKTOOLS_PLUGIN_VERSION").unwrap_or_else(|_| "0.0.0".to_owned()),
            ),
            CapabilitySet::from_strings(environment_list("TIKTOOLS_PLUGIN_CAPABILITIES")),
            PermissionSet::from_strings(environment_list("TIKTOOLS_PLUGIN_PERMISSIONS")),
        )
    }

    /// Returns the limited context available through native ABI v1.
    ///
    /// ABI v1 does not pass manifest metadata into `create`, so native
    /// plugins must not mistake this context for a manifest-backed grant.
    /// A future ABI revision can add an explicit initialization payload.
    pub fn for_native_abi_v1() -> Self {
        Self::new(
            PluginIdentity::new("unknown", "0.0.0"),
            CapabilitySet::default(),
            PermissionSet::default(),
        )
    }
}

fn environment_list(name: &str) -> Vec<String> {
    env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

/// Process-runtime helpers for paths supplied by the TikTools launcher.
///
/// These are intentionally outside `PluginContext`: WASM and native ABI
/// plugins do not necessarily have meaningful process paths.
pub mod process {
    use std::{env, ffi::OsString, path::PathBuf};

    use super::{PluginError, PluginResult};

    const DATA_DIR: &str = "TIKTOOLS_PLUGIN_DATA_DIR";
    const STORAGE_FILE: &str = "TIKTOOLS_PLUGIN_STORAGE_FILE";

    pub fn data_dir() -> PluginResult<PathBuf> {
        path_from_environment(DATA_DIR)
    }

    pub fn storage_file() -> PluginResult<PathBuf> {
        path_from_environment(STORAGE_FILE)
    }

    fn path_from_environment(name: &str) -> PluginResult<PathBuf> {
        path_from_os(name, env::var_os(name))
    }

    pub(crate) fn path_from_os(name: &str, value: Option<OsString>) -> PluginResult<PathBuf> {
        let value = value.ok_or_else(|| {
            PluginError::other(format!(
                "required process environment variable {name} is missing"
            ))
        })?;
        let path = PathBuf::from(value);
        if path.as_os_str().is_empty() {
            return Err(PluginError::other(format!(
                "required process environment variable {name} is empty"
            )));
        }
        Ok(path)
    }
}

/// A typed action call. The action descriptor remains a JSON value so plugin
/// authors can define their own config schema without host-side registration.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionCall {
    pub action: Value,
    pub event: Value,
}

impl ActionCall {
    pub fn action_type(&self) -> Option<&str> {
        self.action
            .get("typeId")
            .or_else(|| self.action.get("type"))
            .and_then(Value::as_str)
    }

    pub fn config(&self) -> &Map<String, Value> {
        static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
        self.action
            .get("config")
            .and_then(Value::as_object)
            .unwrap_or_else(|| EMPTY.get_or_init(Map::new))
    }
}

/// Typed calls at the SDK boundary. The serialized shape remains compatible
/// with the existing `{"type":"action"|"poll"}` process protocol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum PluginCall {
    Action {
        action: Value,
        #[serde(default)]
        event: Value,
    },
    Poll,
}

impl PluginCall {
    pub fn action(action: Value, event: Value) -> Self {
        Self::Action { action, event }
    }

    pub fn into_action(self) -> Option<ActionCall> {
        match self {
            Self::Action { action, event } => Some(ActionCall { action, event }),
            Self::Poll => None,
        }
    }
}

/// A plugin-defined event. The event type remains a string so new plugin event
/// namespaces are discoverable without recompiling the host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PluginEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: Value,
}

impl PluginEvent {
    pub fn new<T: Serialize>(event_type: impl Into<String>, data: T) -> PluginResult<Self> {
        let data = serde_json::to_value(data).map_err(|error| {
            PluginError::other(format!("could not encode plugin event: {error}"))
        })?;
        Ok(Self {
            event_type: event_type.into(),
            data,
        })
    }

    pub fn from_value(value: Value) -> PluginResult<Self> {
        serde_json::from_value(value)
            .map_err(|error| PluginError::invalid_request(format!("invalid plugin event: {error}")))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmitIntent {
    pub event_type: String,
    pub data: Value,
}

impl EmitIntent {
    pub fn new(event_type: impl Into<String>, data: Value) -> Self {
        Self {
            event_type: event_type.into(),
            data,
        }
    }

    pub fn serialized<T: Serialize>(event_type: impl Into<String>, data: T) -> PluginResult<Self> {
        Ok(Self::new(
            event_type,
            serde_json::to_value(data).map_err(|error| {
                PluginError::other(format!("could not encode emit intent: {error}"))
            })?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AudioPlayIntent {
    pub file_ref: MediaFileRef,
    #[serde(default = "default_volume")]
    pub volume: f32,
    #[serde(default)]
    pub overlap: AudioOverlap,
}

impl AudioPlayIntent {
    pub fn new(file_ref: MediaFileRef) -> Self {
        Self {
            file_ref,
            volume: 1.0,
            overlap: AudioOverlap::Allow,
        }
    }

    pub fn from_path(path: impl Into<String>) -> Self {
        Self::new(MediaFileRef::from_path(path))
    }

    pub fn volume(mut self, volume: f32) -> Self {
        self.volume = volume;
        self
    }

    pub fn overlap(mut self, overlap: AudioOverlap) -> Self {
        self.overlap = overlap;
        self
    }
}

fn default_volume() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data", rename_all = "kebab-case")]
pub enum HostIntent {
    Emit(EmitIntent),
    AudioPlay(AudioPlayIntent),
}

impl HostIntent {
    pub fn emit(event_type: impl Into<String>, data: Value) -> Self {
        Self::Emit(EmitIntent::new(event_type, data))
    }

    pub fn audio_play(intent: AudioPlayIntent) -> Self {
        Self::AudioPlay(intent)
    }
}

/// Typed result returned by an action or poll call.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PluginCallResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub intents: Vec<HostIntent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<PluginEvent>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ActionResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub intents: Vec<HostIntent>,
}

impl ActionResult {
    pub fn summary(summary: impl Into<String>) -> Self {
        Self {
            summary: Some(summary.into()),
            ..Self::default()
        }
    }

    pub fn log(mut self, message: impl Into<String>) -> Self {
        self.logs.push(message.into());
        self
    }

    pub fn intent(mut self, intent: HostIntent) -> Self {
        self.intents.push(intent);
        self
    }

    pub fn emit(mut self, event_type: impl Into<String>, data: Value) -> Self {
        self.intents.push(HostIntent::emit(event_type, data));
        self
    }

    pub fn emit_serialized<T: Serialize>(
        self,
        event_type: impl Into<String>,
        data: T,
    ) -> PluginResult<Self> {
        Ok(self.intent(HostIntent::Emit(EmitIntent::serialized(event_type, data)?)))
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PollResult {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<PluginEvent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logs: Vec<String>,
}

impl PollResult {
    pub fn event(mut self, event: PluginEvent) -> Self {
        self.events.push(event);
        self
    }

    pub fn log(mut self, message: impl Into<String>) -> Self {
        self.logs.push(message.into());
        self
    }
}

impl From<ActionResult> for PluginCallResult {
    fn from(result: ActionResult) -> Self {
        Self {
            summary: result.summary,
            logs: result.logs,
            intents: result.intents,
            events: Vec::new(),
        }
    }
}

impl From<PollResult> for PluginCallResult {
    fn from(result: PollResult) -> Self {
        Self {
            summary: None,
            logs: result.logs,
            intents: Vec::new(),
            events: result.events,
        }
    }
}

/// Runtime-neutral plugin business-logic trait.
pub trait Plugin: Send + 'static {
    fn initialize(&mut self, _context: &PluginContext) -> PluginResult<()> {
        Ok(())
    }

    fn action(
        &mut self,
        _context: &PluginContext,
        _call: ActionCall,
    ) -> PluginResult<ActionResult> {
        Err(PluginError::unsupported("action"))
    }

    fn poll(&mut self, _context: &PluginContext) -> PluginResult<PollResult> {
        Ok(PollResult::default())
    }

    fn shutdown(&mut self, _context: &PluginContext) -> PluginResult<()> {
        Ok(())
    }
}

pub fn dispatch_plugin_call<P: Plugin>(
    plugin: &mut P,
    context: &PluginContext,
    call: PluginCall,
) -> PluginResult<PluginCallResult> {
    match call {
        PluginCall::Action { action, event } => plugin
            .action(context, ActionCall { action, event })
            .map(PluginCallResult::from),
        PluginCall::Poll => plugin.poll(context).map(PluginCallResult::from),
    }
}

#[derive(Debug, Error)]
pub enum PluginProtocolError {
    #[error("plugin result must be a JSON object")]
    NotAnObject,
    #[error("plugin result field `{0}` is invalid")]
    InvalidField(&'static str),
    #[error("plugin result field `{field}` is invalid: {message}")]
    InvalidValue {
        field: &'static str,
        message: String,
    },
}

/// Converts both the current typed result shape and the legacy `emit` /
/// `playAudio` keys into one internal contract. This is the single legacy
/// compatibility boundary used by the host after a runtime call.
pub fn decode_plugin_result(value: Value) -> Result<PluginCallResult, PluginProtocolError> {
    let object = value.as_object().ok_or(PluginProtocolError::NotAnObject)?;
    let summary = decode_summary(object.get("summary"))?;
    let logs = decode_logs(object.get("logs"))?;
    let mut intents = decode_typed_intents(object.get("intents"))?;
    intents.extend(decode_legacy_emit_intents(object.get("emit"))?);
    intents.extend(decode_legacy_audio_intents(object.get("playAudio"))?);
    let events = decode_events(object.get("events"))?;
    Ok(PluginCallResult {
        summary,
        logs,
        intents,
        events,
    })
}

fn decode_summary(value: Option<&Value>) -> Result<Option<String>, PluginProtocolError> {
    let Some(value) = value else {
        return Ok(None);
    };
    value
        .as_str()
        .map(|value| Some(value.to_owned()))
        .ok_or(PluginProtocolError::InvalidField("summary"))
}

fn decode_logs(value: Option<&Value>) -> Result<Vec<String>, PluginProtocolError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    as_values(value)
        .into_iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or(PluginProtocolError::InvalidField("logs"))
        })
        .collect()
}

fn decode_typed_intents(value: Option<&Value>) -> Result<Vec<HostIntent>, PluginProtocolError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    as_values(value)
        .into_iter()
        .map(|value| {
            serde_json::from_value(value.clone()).map_err(|error| {
                PluginProtocolError::InvalidValue {
                    field: "intents",
                    message: error.to_string(),
                }
            })
        })
        .collect()
}

fn decode_legacy_emit_intents(
    value: Option<&Value>,
) -> Result<Vec<HostIntent>, PluginProtocolError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    as_values(value)
        .into_iter()
        .map(|value| {
            let object = value
                .as_object()
                .ok_or(PluginProtocolError::InvalidField("emit"))?;
            let event_type = object
                .get("type")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or(PluginProtocolError::InvalidField("emit"))?;
            Ok(HostIntent::Emit(EmitIntent::new(
                event_type,
                object.get("data").cloned().unwrap_or(Value::Null),
            )))
        })
        .collect()
}

fn decode_legacy_audio_intents(
    value: Option<&Value>,
) -> Result<Vec<HostIntent>, PluginProtocolError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    as_values(value)
        .into_iter()
        .map(|value| {
            let mut object = value
                .as_object()
                .cloned()
                .ok_or(PluginProtocolError::InvalidField("playAudio"))?;
            if object.get("fileRef").is_some_and(Value::is_string) {
                let file = object
                    .remove("fileRef")
                    .ok_or(PluginProtocolError::InvalidField("playAudio"))?;
                object.insert("fileRef".to_owned(), serde_json::json!({"path": file}));
            } else if object.get("fileRef").is_none() {
                for key in ["filePath", "file", "path"] {
                    if let Some(file) = object.remove(key) {
                        object.insert(
                            "fileRef".to_owned(),
                            if file.is_string() {
                                serde_json::json!({"path": file})
                            } else {
                                file
                            },
                        );
                        break;
                    }
                }
            }
            let intent = serde_json::from_value::<AudioPlayIntent>(Value::Object(object)).map_err(
                |error| PluginProtocolError::InvalidValue {
                    field: "playAudio",
                    message: error.to_string(),
                },
            )?;
            Ok(HostIntent::AudioPlay(intent))
        })
        .collect()
}

fn decode_events(value: Option<&Value>) -> Result<Vec<PluginEvent>, PluginProtocolError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    as_values(value)
        .into_iter()
        .map(|value| {
            serde_json::from_value(value.clone()).map_err(|error| {
                PluginProtocolError::InvalidValue {
                    field: "events",
                    message: error.to_string(),
                }
            })
        })
        .collect()
}

fn as_values(value: &Value) -> Vec<&Value> {
    match value {
        Value::Array(values) => values.iter().collect(),
        value => vec![value],
    }
}

fn response_ok(id: String, result: PluginCallResult) -> PluginResponse {
    PluginResponse {
        protocol_version: TIKTOOLS_PLUGIN_PROTOCOL_VERSION,
        id,
        ok: true,
        result: Some(
            serde_json::to_value(result).expect("SDK plugin call result should always serialize"),
        ),
        error: None,
    }
}

fn response_failure(id: String, error: impl Into<String>) -> PluginResponse {
    PluginResponse {
        protocol_version: TIKTOOLS_PLUGIN_PROTOCOL_VERSION,
        id,
        ok: false,
        result: None,
        error: Some(error.into()),
    }
}

/// Runs a plugin using the existing framed stdin/stdout process protocol.
pub fn run_process_plugin<P>() -> PluginResult<()>
where
    P: Plugin + Default,
{
    let context = PluginContext::from_process_environment();
    let mut plugin = P::default();
    plugin.initialize(&context)?;
    run_process_plugin_with(&mut plugin, &context)
}

pub fn run_process_plugin_with<P>(plugin: &mut P, context: &PluginContext) -> PluginResult<()>
where
    P: Plugin,
{
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut writer = io::BufWriter::new(stdout.lock());

    let loop_result = loop {
        let request = match read_frame::<_, PluginRequest>(&mut reader) {
            Ok(request) => request,
            Err(FrameError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof => {
                break Ok(())
            }
            Err(error) => break Err(PluginError::other(error.to_string())),
        };
        let response = handle_process_request(plugin, context, request);
        write_frame(&mut writer, &response)
            .map_err(|error| PluginError::other(error.to_string()))?;
    };
    let shutdown = plugin.shutdown(context);
    loop_result.and(shutdown)
}

fn handle_process_request<P: Plugin>(
    plugin: &mut P,
    context: &PluginContext,
    request: PluginRequest,
) -> PluginResponse {
    if request.protocol_version != TIKTOOLS_PLUGIN_PROTOCOL_VERSION {
        return response_failure(
            request.id,
            format!("unsupported protocol version {}", request.protocol_version),
        );
    }
    if request.method != METHOD_CALL {
        return response_failure(
            request.id,
            format!("unsupported method `{}`", request.method),
        );
    }
    let call = match serde_json::from_value::<PluginCall>(request.payload) {
        Ok(call) => call,
        Err(error) => return response_failure(request.id, format!("invalid plugin call: {error}")),
    };
    match dispatch_plugin_call(plugin, context, call) {
        Ok(result) => response_ok(request.id, result),
        Err(error) => response_failure(request.id, error.to_string()),
    }
}

pub mod native {
    //! Native ABI adapter used by `tiktools_export_native_plugin!`.

    use super::*;
    use std::{
        ffi::c_void,
        mem::ManuallyDrop,
        panic::{catch_unwind, AssertUnwindSafe},
        slice,
    };

    struct NativePluginState<P: Plugin> {
        plugin: P,
        context: PluginContext,
        initialization_error: Option<String>,
    }

    pub fn create<P>() -> *mut c_void
    where
        P: Plugin + Default,
    {
        let context = PluginContext::for_native_abi_v1();
        let mut plugin = P::default();
        let initialization_error = plugin
            .initialize(&context)
            .err()
            .map(|error| error.to_string());
        Box::into_raw(Box::new(NativePluginState {
            plugin,
            context,
            initialization_error,
        })) as *mut c_void
    }

    /// # Safety
    ///
    /// `context` must be a pointer returned by `create::<P>` that has not
    /// already been destroyed.
    pub unsafe fn destroy<P>(context: *mut c_void)
    where
        P: Plugin,
    {
        if context.is_null() {
            return;
        }
        // SAFETY: the pointer was allocated by `create::<P>` and is consumed
        // exactly once by the native ABI destroy function.
        let mut state = unsafe { Box::from_raw(context.cast::<NativePluginState<P>>()) };
        let _ = state.plugin.shutdown(&state.context);
    }

    /// # Safety
    ///
    /// `context` must come from `create::<P>`, `request_ptr` must refer to a
    /// readable request buffer for `request_len` bytes, and `response` must
    /// point to writable storage owned by the ABI caller.
    pub unsafe fn handle_message<P>(
        context: *mut c_void,
        request_ptr: *const u8,
        request_len: usize,
        response: *mut PluginBuffer,
    ) -> PluginStatus
    where
        P: Plugin,
    {
        let result =
            catch_unwind(AssertUnwindSafe(|| {
                if context.is_null() || response.is_null() {
                    return Err(PluginStatus::InvalidRequest);
                }
                if request_ptr.is_null() || request_len > MAX_FRAME_BYTES {
                    return Err(PluginStatus::InvalidRequest);
                }
                // SAFETY: the host supplies the request pointer and length for the
                // duration of this call; validation above bounds the slice.
                let request = unsafe { slice::from_raw_parts(request_ptr, request_len) };
                // SAFETY: the pointer was allocated by `create::<P>`.
                let state = unsafe { &mut *context.cast::<NativePluginState<P>>() };
                if state.initialization_error.is_some() {
                    return Err(PluginStatus::InternalError);
                }
                // Native ABI v1 receives the raw typed call. The process
                // adapter has an outer PluginRequest envelope, but adding
                // that envelope here would break existing native plugins.
                let call = serde_json::from_slice::<PluginCall>(request)
                    .map_err(|_| PluginStatus::InvalidRequest)?;
                let result = dispatch_plugin_call(&mut state.plugin, &state.context, call)
                    .map_err(|error| match error {
                        PluginError::InvalidRequest(_) | PluginError::UnsupportedAction(_) => {
                            PluginStatus::InvalidRequest
                        }
                        PluginError::CapabilityUnavailable(_) | PluginError::Other(_) => {
                            PluginStatus::InternalError
                        }
                    })?;
                let bytes = serde_json::to_vec(&result).map_err(|_| PluginStatus::InternalError)?;
                write_buffer(response, bytes);
                Ok(())
            }));
        match result {
            Ok(Ok(())) => PluginStatus::Ok,
            Ok(Err(status)) => status,
            Err(_) => PluginStatus::InternalError,
        }
    }

    /// # Safety
    ///
    /// `response` must point to a buffer previously initialized by this
    /// adapter, or to writable `PluginBuffer` storage.
    pub unsafe fn free_buffer(response: *mut PluginBuffer) {
        if response.is_null() {
            return;
        }
        // SAFETY: the host passes the same buffer that this adapter allocated.
        let buffer = unsafe { &mut *response };
        if !buffer.ptr.is_null() && buffer.capacity >= buffer.len {
            // SAFETY: pointer, length, and capacity came from `write_buffer`.
            unsafe { drop(Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.capacity)) };
        }
        *buffer = PluginBuffer::empty();
    }

    fn write_buffer(response: *mut PluginBuffer, bytes: Vec<u8>) {
        // SAFETY: the caller validated that `response` is non-null.
        let response = unsafe { &mut *response };
        if bytes.is_empty() {
            *response = PluginBuffer::empty();
            return;
        }
        let mut bytes = ManuallyDrop::new(bytes);
        *response = PluginBuffer {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
            capacity: bytes.capacity(),
        };
    }
}

pub mod prelude {
    pub use crate::{
        tiktools_export_native_plugin, tiktools_process_plugin, ActionCall, ActionResult,
        AudioPlayIntent, EmitIntent, HostIntent, Plugin, PluginCall, PluginCallResult,
        PluginContext, PluginError, PluginEvent, PluginIdentity, PluginResult, PollResult,
    };
    pub use tiktools_plugin_api::{AudioOverlap, MediaFileRef};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn action(
            &mut self,
            _context: &PluginContext,
            _call: ActionCall,
        ) -> PluginResult<ActionResult> {
            Ok(ActionResult::summary("action handled"))
        }
    }

    fn test_context() -> PluginContext {
        PluginContext::new(
            PluginIdentity::new("test.plugin", "1.0.0"),
            CapabilitySet::default(),
            PermissionSet::default(),
        )
    }

    #[test]
    fn typed_plugin_call_preserves_legacy_wire_shape() {
        let call = PluginCall::action(
            serde_json::json!({"typeId": "demo.action", "config": {"value": 1}}),
            serde_json::json!({"type": "tiktok.chat"}),
        );
        let value = serde_json::to_value(call).unwrap();
        assert_eq!(value["type"], "action");
        assert_eq!(value["action"]["typeId"], "demo.action");
    }

    #[test]
    fn compatibility_decoder_maps_legacy_intents_once() {
        let result = decode_plugin_result(serde_json::json!({
            "summary": "done",
            "logs": ["one"],
            "emit": [{"type": "demo.event", "data": {"ok": true}}],
            "playAudio": {"fileRef": {"path": "/tmp/alert.wav"}, "volume": 0.5}
        }))
        .unwrap();
        assert_eq!(result.summary.as_deref(), Some("done"));
        assert_eq!(result.logs, vec!["one"]);
        assert_eq!(result.intents.len(), 2);
        assert!(matches!(result.intents[0], HostIntent::Emit(_)));
        assert!(matches!(result.intents[1], HostIntent::AudioPlay(_)));
    }

    #[test]
    fn compatibility_decoder_rejects_malformed_legacy_fields() {
        for (field, value) in [
            ("summary", serde_json::json!(42)),
            ("logs", serde_json::json!(["ok", 42])),
            ("emit", serde_json::json!([{"data": {}}])),
            ("playAudio", serde_json::json!([{"volume": "loud"}])),
            ("events", serde_json::json!([{"type": "missing-data"}])),
        ] {
            let error = decode_plugin_result(serde_json::json!({field: value})).unwrap_err();
            assert!(error.to_string().contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn audio_intent_serializes_with_typed_file_reference() {
        let result = ActionResult::default().intent(HostIntent::audio_play(
            AudioPlayIntent::from_path("/tmp/alert.wav"),
        ));
        let value = serde_json::to_value(PluginCallResult::from(result)).unwrap();
        assert_eq!(value["intents"][0]["type"], "audio-play");
        assert_eq!(
            value["intents"][0]["data"]["fileRef"]["path"],
            "/tmp/alert.wav"
        );
    }

    #[test]
    fn process_path_helpers_reject_missing_and_empty_values() {
        assert!(process::path_from_os("TIKTOOLS_PLUGIN_DATA_DIR", None).is_err());
        assert!(process::path_from_os(
            "TIKTOOLS_PLUGIN_STORAGE_FILE",
            Some(std::ffi::OsString::new())
        )
        .is_err());
    }

    #[test]
    fn process_request_dispatches_raw_typed_call_inside_the_wire_envelope() {
        let call = PluginCall::action(
            serde_json::json!({"typeId": "demo.action"}),
            serde_json::json!({"type": "demo.event"}),
        );
        let request = PluginRequest::new(
            METHOD_CALL,
            METHOD_CALL,
            serde_json::to_value(call).unwrap(),
        );
        let response = handle_process_request(&mut TestPlugin, &test_context(), request);
        assert!(response.ok);
        assert_eq!(response.id, METHOD_CALL);
        let result: PluginCallResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.summary.as_deref(), Some("action handled"));
    }

    #[test]
    fn process_request_rejects_bad_protocol_method_and_call() {
        let context = test_context();
        let mut bad_version = PluginRequest::new(
            "version",
            METHOD_CALL,
            serde_json::json!({
                "type": "poll"
            }),
        );
        bad_version.protocol_version += 1;
        assert!(!handle_process_request(&mut TestPlugin, &context, bad_version).ok);

        let bad_method = PluginRequest::new(
            "method",
            "other",
            serde_json::json!({
                "type": "poll"
            }),
        );
        assert!(!handle_process_request(&mut TestPlugin, &context, bad_method).ok);

        let bad_call = PluginRequest::new(
            "call",
            METHOD_CALL,
            serde_json::json!({
                "type": "unknown"
            }),
        );
        assert!(!handle_process_request(&mut TestPlugin, &context, bad_call).ok);
    }
}
