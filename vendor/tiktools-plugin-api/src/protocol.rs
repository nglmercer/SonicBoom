//! JSON messages and length-delimited framing used by process plugins.

use std::io::{self, Read, Write};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
pub const METHOD_CALL: &str = "call";
pub const METHOD_CAPABILITY_REQUEST: &str = "capability.request";
pub const METHOD_CAPABILITY_RESPONSE: &str = "capability.response";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PluginRequest {
    pub protocol_version: u32,
    pub id: String,
    pub method: String,
    pub payload: Value,
}

impl PluginRequest {
    pub fn new(id: impl Into<String>, method: impl Into<String>, payload: Value) -> Self {
        Self {
            protocol_version: crate::TIKTOOLS_PLUGIN_PROTOCOL_VERSION,
            id: id.into(),
            method: method.into(),
            payload,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PluginResponse {
    pub protocol_version: u32,
    pub id: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Payload used by process/WASM runtimes when guest code asks the host for a
/// declared capability. The host may reject it before touching the filesystem
/// or audio device.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityRequest {
    pub request_id: String,
    pub capability: String,
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityResponse {
    pub request_id: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("could not read plugin frame: {0}")]
    Io(#[from] io::Error),
    #[error("plugin frame is larger than {MAX_FRAME_BYTES} bytes")]
    TooLarge,
    #[error("plugin frame is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn write_frame<W: Write, T: Serialize>(writer: &mut W, value: &T) -> Result<(), FrameError> {
    let payload = serde_json::to_vec(value)?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge);
    }
    let length = u32::try_from(payload.len()).map_err(|_| FrameError::TooLarge)?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

pub fn read_frame<R: Read, T: DeserializeOwned>(reader: &mut R) -> Result<T, FrameError> {
    let mut length_bytes = [0_u8; 4];
    reader.read_exact(&mut length_bytes)?;
    let length = u32::from_le_bytes(length_bytes) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge);
    }
    let mut payload = vec![0_u8; length];
    reader.read_exact(&mut payload)?;
    Ok(serde_json::from_slice(&payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_is_length_delimited() {
        let request = PluginRequest::new("1", "ping", serde_json::json!({"ok": true}));
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &request).unwrap();
        assert_eq!(
            u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize,
            bytes.len() - 4
        );
        let decoded: PluginRequest = read_frame(&mut bytes.as_slice()).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn capability_request_is_json_only() {
        let request = CapabilityRequest {
            request_id: "cap-1".to_owned(),
            capability: "audio.play".to_owned(),
            params: serde_json::json!({"fileRef":{"path":"/music/alert.wav"}}),
        };
        let decoded: CapabilityRequest =
            serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
        assert_eq!(decoded, request);
    }
}
