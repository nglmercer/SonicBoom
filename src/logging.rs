//! Logging configuration for SonicBoom
//!
//! Provides structured logging with file rotation

use std::sync::OnceLock;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Global guard to keep the file writer alive
static LOG_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

/// Initialize the logging system
pub fn init(
    log_dir: &str,
    log_level: &str,
    log_to_file: bool,
    log_to_stdout: bool,
) {
    // Create log directory if it doesn't exist
    if log_to_file {
        if let Err(e) = std::fs::create_dir_all(log_dir) {
            eprintln!("Warning: Could not create log directory: {}", e);
        }
    }

    // Build the env filter
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| format!("SonicBoom={},tower_http=warn", log_level).into());

    // Create the base layer
    let base = tracing_subscriber::registry().with(env_filter);

    if log_to_file && log_to_stdout {
        // File writer
        let file_appender = tracing_appender::rolling::daily(log_dir, "sonicboom.log");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        let file_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_level(true)
            .with_ansi(false)
            .with_writer(non_blocking);

        let stdout_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_level(true)
            .with_ansi(true);

        // Keep guard alive
        let _ = LOG_GUARD.set(guard);

        base.with(file_layer).with(stdout_layer).init();
    } else if log_to_file {
        let file_appender = tracing_appender::rolling::daily(log_dir, "sonicboom.log");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        let file_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_level(true)
            .with_ansi(false)
            .with_writer(non_blocking);

        let _ = LOG_GUARD.set(guard);

        base.with(file_layer).init();
    } else if log_to_stdout {
        let stdout_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_level(true)
            .with_ansi(true);

        base.with(stdout_layer).init();
    } else {
        base.init();
    }
}

/// Log startup banner
pub fn log_startup(port: u16, log_dir: &str) {
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        port = port,
        log_dir = log_dir,
        "SonicBoom TTS Server starting"
    );
}
