//! Small, deliberately explicit macros for plugin entry-point plumbing.
//!
//! The generated code delegates all behavior and unsafe pointer handling to
//! `tiktools-plugin-sdk`. These macros only give plugin authors a concise,
//! stable entry point.

use proc_macro::TokenStream;

fn plugin_type(input: TokenStream) -> String {
    input.to_string().trim().to_owned()
}

/// Generates a process-plugin `main` function for a `Plugin + Default` type.
#[proc_macro]
pub fn tiktools_process_plugin(input: TokenStream) -> TokenStream {
    let plugin = plugin_type(input);
    format!(
        "fn main() -> ::tiktools_plugin_sdk::PluginResult<()> {{ ::tiktools_plugin_sdk::run_process_plugin::<{plugin}>() }}"
    )
    .parse()
    .expect("TikTools process-plugin macro output should parse")
}

/// Generates the reviewed native C ABI bridge for a `Plugin + Default` type.
#[proc_macro]
pub fn tiktools_export_native_plugin(input: TokenStream) -> TokenStream {
    let plugin = plugin_type(input);
    format!(
        r#"
        #[allow(non_upper_case_globals)]
        static TIKTOOLS_PLUGIN_API: ::tiktools_plugin_sdk::tiktools_plugin_api::TikToolsPluginApi =
            ::tiktools_plugin_sdk::tiktools_plugin_api::TikToolsPluginApi {{
                abi_version: ::tiktools_plugin_sdk::tiktools_plugin_api::TIKTOOLS_PLUGIN_ABI_VERSION,
                protocol_version: ::tiktools_plugin_sdk::tiktools_plugin_api::TIKTOOLS_PLUGIN_PROTOCOL_VERSION,
                create: Some(__tiktools_plugin_create),
                destroy: Some(__tiktools_plugin_destroy),
                handle_message: Some(__tiktools_plugin_handle_message),
                free_buffer: Some(__tiktools_plugin_free_buffer),
            }};

        unsafe extern "C" fn __tiktools_plugin_create() -> *mut ::std::ffi::c_void {{
            ::tiktools_plugin_sdk::native::create::<{plugin}>()
        }}

        unsafe extern "C" fn __tiktools_plugin_destroy(context: *mut ::std::ffi::c_void) {{
            unsafe {{ ::tiktools_plugin_sdk::native::destroy::<{plugin}>(context) }}
        }}

        unsafe extern "C" fn __tiktools_plugin_handle_message(
            context: *mut ::std::ffi::c_void,
            request_ptr: *const u8,
            request_len: usize,
            response: *mut ::tiktools_plugin_sdk::tiktools_plugin_api::PluginBuffer,
        ) -> ::tiktools_plugin_sdk::tiktools_plugin_api::PluginStatus {{
            unsafe {{
                ::tiktools_plugin_sdk::native::handle_message::<{plugin}>(
                    context,
                    request_ptr,
                    request_len,
                    response,
                )
            }}
        }}

        unsafe extern "C" fn __tiktools_plugin_free_buffer(
            response: *mut ::tiktools_plugin_sdk::tiktools_plugin_api::PluginBuffer,
        ) {{
            unsafe {{ ::tiktools_plugin_sdk::native::free_buffer(response) }}
        }}

        #[no_mangle]
        pub extern "C" fn tiktools_plugin_init()
            -> *const ::tiktools_plugin_sdk::tiktools_plugin_api::TikToolsPluginApi
        {{
            &TIKTOOLS_PLUGIN_API
        }}
        "#
    )
    .parse()
    .expect("TikTools native-plugin macro output should parse")
}
