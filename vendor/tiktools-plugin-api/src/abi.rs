//! Intentionally small C ABI for trusted native plugins.
//!
//! The ABI only moves pointers, lengths, status codes, and serialized bytes.
//! It never moves `String`, `Vec`, `HashMap`, Rust trait objects, or futures
//! across a dynamic-library boundary.

use std::ffi::c_void;

use crate::{TIKTOOLS_PLUGIN_ABI_VERSION, TIKTOOLS_PLUGIN_PROTOCOL_VERSION};

pub const TIKTOOLS_PLUGIN_INIT_SYMBOL: &[u8] = b"tiktools_plugin_init\0";

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginStatus {
    Ok = 0,
    InvalidRequest = 1,
    InternalError = 2,
    Incompatible = 3,
}

impl PluginStatus {
    pub const fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }
}

/// A buffer allocated and owned by a plugin until `free_buffer` is called.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginBuffer {
    pub ptr: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

impl PluginBuffer {
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }
}

/// The only native functions the host calls after loading a plugin.
///
/// Plugins should export an `extern "C" fn tiktools_plugin_init() -> *const
/// TikToolsPluginApi` function returning a pointer to a static table.
#[repr(C)]
pub struct TikToolsPluginApi {
    pub abi_version: u32,
    pub protocol_version: u32,
    pub create: Option<unsafe extern "C" fn() -> *mut c_void>,
    pub destroy: Option<unsafe extern "C" fn(*mut c_void)>,
    pub handle_message: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            request_ptr: *const u8,
            request_len: usize,
            response: *mut PluginBuffer,
        ) -> PluginStatus,
    >,
    pub free_buffer: Option<unsafe extern "C" fn(*mut PluginBuffer)>,
}

impl TikToolsPluginApi {
    pub const fn is_compatible(&self) -> bool {
        self.abi_version == TIKTOOLS_PLUGIN_ABI_VERSION
            && self.protocol_version == TIKTOOLS_PLUGIN_PROTOCOL_VERSION
            && self.handle_message.is_some()
            && self.free_buffer.is_some()
    }
}

/// Exported symbol type. The library must retain ownership of the returned
/// API table for as long as the host keeps the plugin loaded.
pub type PluginInit = unsafe extern "C" fn() -> *const TikToolsPluginApi;
