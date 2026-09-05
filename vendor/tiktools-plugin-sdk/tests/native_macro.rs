use tiktools_plugin_sdk::prelude::*;
use tiktools_plugin_sdk::tiktools_plugin_api::{PluginBuffer, PluginStatus};

#[derive(Default)]
struct NativeExample;

impl Plugin for NativeExample {
    fn action(
        &mut self,
        _context: &PluginContext,
        _call: ActionCall,
    ) -> PluginResult<ActionResult> {
        Ok(ActionResult::summary("native call handled"))
    }
}

tiktools_export_native_plugin!(NativeExample);

#[test]
fn native_macro_exports_a_compatible_table() {
    let api = tiktools_plugin_init();
    assert!(!api.is_null());
    assert!(unsafe { &*api }.is_compatible());
}

#[test]
fn native_macro_round_trips_a_raw_typed_call() {
    let api = unsafe { &*tiktools_plugin_init() };
    let context = unsafe { (api.create.unwrap())() };
    let request = serde_json::to_vec(&PluginCall::action(
        serde_json::json!({"typeId": "demo.action"}),
        serde_json::json!({}),
    ))
    .unwrap();
    let mut response = PluginBuffer::empty();

    let status = unsafe {
        (api.handle_message.unwrap())(context, request.as_ptr(), request.len(), &mut response)
    };
    assert_eq!(status, PluginStatus::Ok);
    let result: PluginCallResult = unsafe {
        serde_json::from_slice(std::slice::from_raw_parts(response.ptr, response.len)).unwrap()
    };
    assert_eq!(result.summary.as_deref(), Some("native call handled"));

    unsafe { (api.free_buffer.unwrap())(&mut response) };
    unsafe { (api.destroy.unwrap())(context) };
}

#[test]
fn native_macro_rejects_a_null_request_pointer() {
    let api = unsafe { &*tiktools_plugin_init() };
    let context = unsafe { (api.create.unwrap())() };
    let mut response = PluginBuffer::empty();
    let status =
        unsafe { (api.handle_message.unwrap())(context, std::ptr::null(), 0, &mut response) };
    assert_eq!(status, PluginStatus::InvalidRequest);
    unsafe { (api.destroy.unwrap())(context) };
}
