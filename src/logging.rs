//! Logging configuration for SonicBoom
//!
//! Provides structured logging with file rotation

use std::sync::OnceLock;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, fmt::format::Writer, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Custom timer that shows only time (HH:MM:SS)
struct ShortTimer;

impl tracing_subscriber::fmt::time::FormatTime for ShortTimer {
    fn format_time(&self, w: &mut Writer<'_>) -> std::fmt::Result {
        use std::time::SystemTime;
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap();
        let secs = now.as_secs();
        let hours = (secs / 3600) % 24;
        let mins = (secs / 60) % 60;
        let secs = secs % 60;
        write!(w, "{:02}:{:02}:{:02}", hours, mins, secs)
    }
}

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

    // Build the env filter - include tower_http at trace level for request logging
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| format!("SonicBoom={},tower_http=trace,httparse=trace", log_level).into());

    // Base subscriber
    let base = tracing_subscriber::registry().with(env_filter);

    if log_to_file && log_to_stdout {
        // File writer - detailed format
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

        // Console - simplified format (level + target + message only)
        let stdout_layer = fmt::layer()
            .with_target(true)
            .with_level(true)
            .with_ansi(true)
            .with_timer(ShortTimer)
            .compact();

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
            .with_level(true)
            .with_ansi(true)
            .with_timer(ShortTimer)
            .compact();

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
