//! File-based logging for the Acton-AI kernel.
//!
//! This module provides automatic file logging with daily rotation.
//! Logs are written to an XDG-compliant location by default.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Configuration for kernel file logging.
///
/// Logs are written to rolling daily files in an XDG-compliant location.
/// By default, logs go to `~/.local/share/acton/logs/`.
///
/// # Example
///
/// ```rust
/// use acton_ai::kernel::LoggingConfig;
///
/// // Use defaults (logs to ~/.local/share/acton/logs/acton-ai.log)
/// let config = LoggingConfig::default();
///
/// // Customize for your application
/// let config = LoggingConfig::new()
///     .with_app_name("my-agent")
///     .with_level(acton_ai::kernel::LogLevel::Debug);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Whether file logging is enabled.
    pub enabled: bool,
    /// The application name used for log file naming.
    /// Log files are named `{app_name}.log` with daily rotation.
    pub app_name: String,
    /// Custom log directory. If None, uses XDG data dir + "acton/logs".
    pub log_dir: Option<PathBuf>,
    /// Log level filter.
    pub level: LogLevel,
}

impl LoggingConfig {
    /// Creates a new LoggingConfig with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a disabled logging configuration.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Sets the application name for log file naming.
    #[must_use]
    pub fn with_app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = name.into();
        self
    }

    /// Sets a custom log directory.
    #[must_use]
    pub fn with_log_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.log_dir = Some(path.into());
        self
    }

    /// Sets the log level filter.
    #[must_use]
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.level = level;
        self
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            app_name: "acton-ai".to_string(),
            log_dir: None,
            level: LogLevel::default(),
        }
    }
}

/// Log level filter for file logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level - most verbose.
    Trace,
    /// Debug level.
    Debug,
    /// Info level - default.
    #[default]
    Info,
    /// Warn level.
    Warn,
    /// Error level - least verbose.
    Error,
}

impl LogLevel {
    /// Converts to tracing_subscriber LevelFilter.
    #[must_use]
    pub fn to_filter(self) -> tracing_subscriber::filter::LevelFilter {
        match self {
            Self::Trace => tracing_subscriber::filter::LevelFilter::TRACE,
            Self::Debug => tracing_subscriber::filter::LevelFilter::DEBUG,
            Self::Info => tracing_subscriber::filter::LevelFilter::INFO,
            Self::Warn => tracing_subscriber::filter::LevelFilter::WARN,
            Self::Error => tracing_subscriber::filter::LevelFilter::ERROR,
        }
    }
}

/// Guard that must be held to keep file logging active.
///
/// When dropped, flushes pending logs and stops file logging.
/// The kernel stores this internally and drops it on shutdown.
///
/// Note: This type intentionally does not implement Clone, PartialEq, etc.
/// as the underlying WorkerGuard is not clonable and should not be compared.
pub struct LoggingGuard {
    _guard: tracing_appender::non_blocking::WorkerGuard,
}

impl std::fmt::Debug for LoggingGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoggingGuard").finish_non_exhaustive()
    }
}

/// Global storage for the logging guard.
/// This ensures the guard lives for the application's lifetime.
static LOGGING_GUARD: std::sync::OnceLock<LoggingGuard> = std::sync::OnceLock::new();

/// Stores the logging guard globally.
///
/// This is called automatically by `Kernel::spawn_with_config`.
/// The guard will be held for the application's lifetime.
fn store_logging_guard(guard: LoggingGuard) {
    // If a guard is already stored, this is a no-op.
    // This handles the case where multiple kernels are spawned.
    let _ = LOGGING_GUARD.set(guard);
}

/// Errors that can occur during logging initialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggingError {
    /// The specific error that occurred.
    pub kind: LoggingErrorKind,
}

/// Specific logging error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoggingErrorKind {
    /// Failed to determine XDG data directory.
    NoDataDir,
    /// Failed to create log directory.
    CreateDirFailed {
        /// The path that could not be created.
        path: PathBuf,
        /// The reason for failure.
        reason: String,
    },
    /// Subscriber initialization failed.
    SubscriberInitFailed {
        /// The reason for failure.
        reason: String,
    },
}

impl LoggingError {
    /// Creates a new LoggingError with the given kind.
    #[must_use]
    pub fn new(kind: LoggingErrorKind) -> Self {
        Self { kind }
    }

    /// Creates an error for missing XDG data directory.
    #[must_use]
    pub fn no_data_dir() -> Self {
        Self::new(LoggingErrorKind::NoDataDir)
    }

    /// Creates an error for failed directory creation.
    #[must_use]
    pub fn create_dir_failed(path: PathBuf, reason: impl Into<String>) -> Self {
        Self::new(LoggingErrorKind::CreateDirFailed {
            path,
            reason: reason.into(),
        })
    }

    /// Creates an error for subscriber initialization failure.
    #[must_use]
    pub fn subscriber_init_failed(reason: impl Into<String>) -> Self {
        Self::new(LoggingErrorKind::SubscriberInitFailed {
            reason: reason.into(),
        })
    }

    /// Returns true if this is a missing data directory error.
    #[must_use]
    pub fn is_no_data_dir(&self) -> bool {
        matches!(self.kind, LoggingErrorKind::NoDataDir)
    }
}

impl fmt::Display for LoggingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            LoggingErrorKind::NoDataDir => {
                write!(
                    f,
                    "could not determine XDG data directory; \
                     set XDG_DATA_HOME or use a custom log_dir"
                )
            }
            LoggingErrorKind::CreateDirFailed { path, reason } => {
                write!(
                    f,
                    "failed to create log directory '{}': {}; check permissions",
                    path.display(),
                    reason
                )
            }
            LoggingErrorKind::SubscriberInitFailed { reason } => {
                write!(
                    f,
                    "failed to initialize tracing subscriber: {}; \
                     a subscriber may already be set",
                    reason
                )
            }
        }
    }
}

impl std::error::Error for LoggingError {}

/// Resolves the log directory from configuration.
///
/// If `config.log_dir` is Some, uses that path.
/// Otherwise, uses XDG data dir + "acton/logs".
fn resolve_log_dir(config: &LoggingConfig) -> Result<PathBuf, LoggingError> {
    if let Some(ref custom_dir) = config.log_dir {
        return Ok(custom_dir.clone());
    }

    dirs::data_local_dir()
        .map(|dir| dir.join("acton").join("logs"))
        .ok_or_else(LoggingError::no_data_dir)
}

/// Initializes file-based logging with the given configuration.
///
/// Creates a daily rolling log file in the configured directory.
/// Returns a guard that must be held to keep logging active.
///
/// # Arguments
///
/// * `config` - Logging configuration
///
/// # Returns
///
/// `Ok(Some(LoggingGuard))` if logging was initialized successfully.
/// `Ok(None)` if logging is disabled in config.
/// `Err(LoggingError)` if initialization failed.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::kernel::{LoggingConfig, init_file_logging};
///
/// let config = LoggingConfig::default();
/// let _guard = init_file_logging(&config)?;
/// // Logging is active while _guard is held
/// ```
pub fn init_file_logging(config: &LoggingConfig) -> Result<Option<LoggingGuard>, LoggingError> {
    if !config.enabled {
        return Ok(None);
    }

    let log_dir = resolve_log_dir(config)?;

    // Create the log directory if it doesn't exist
    std::fs::create_dir_all(&log_dir)
        .map_err(|e| LoggingError::create_dir_failed(log_dir.clone(), e.to_string()))?;

    // Create a daily rolling file appender
    let file_appender =
        tracing_appender::rolling::daily(&log_dir, format!("{}.log", config.app_name));
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // Build the subscriber with file logging layer
    let result = tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .with_target(false),
        )
        .with(config.level.to_filter())
        .try_init();

    match result {
        Ok(()) => Ok(Some(LoggingGuard { _guard: guard })),
        Err(e) => Err(LoggingError::subscriber_init_failed(e.to_string())),
    }
}

/// Initializes file-based logging and stores the guard globally.
///
/// This is the recommended function for automatic kernel logging.
/// The guard is stored in a global static and will live for the application's lifetime.
///
/// # Arguments
///
/// * `config` - Logging configuration
///
/// # Returns
///
/// `Ok(true)` if logging was initialized successfully.
/// `Ok(false)` if logging is disabled or already initialized.
/// `Err(LoggingError)` if initialization failed.
pub fn init_and_store_logging(config: &LoggingConfig) -> Result<bool, LoggingError> {
    // Check if logging is already initialized
    if LOGGING_GUARD.get().is_some() {
        return Ok(false);
    }

    match init_file_logging(config)? {
        Some(guard) => {
            store_logging_guard(guard);
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Returns the resolved log directory path for the given configuration.
///
/// This is useful for applications that want to display where logs are being written.
///
/// # Arguments
///
/// * `config` - Logging configuration
///
/// # Returns
///
/// The resolved log directory path, or an error if XDG dir cannot be determined.
pub fn get_log_dir(config: &LoggingConfig) -> Result<PathBuf, LoggingError> {
    resolve_log_dir(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logging_config_default_values() {
        let config = LoggingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.app_name, "acton-ai");
        assert!(config.log_dir.is_none());
        assert_eq!(config.level, LogLevel::Info);
    }

    #[test]
    fn logging_config_new_equals_default() {
        let new = LoggingConfig::new();
        let default = LoggingConfig::default();
        assert_eq!(new, default);
    }

    #[test]
    fn logging_config_builder_pattern() {
        let config = LoggingConfig::new()
            .with_app_name("my-app")
            .with_log_dir("/tmp/logs")
            .with_level(LogLevel::Debug);

        assert_eq!(config.app_name, "my-app");
        assert_eq!(config.log_dir, Some(PathBuf::from("/tmp/logs")));
        assert_eq!(config.level, LogLevel::Debug);
    }

    #[test]
    fn logging_config_disabled() {
        let config = LoggingConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn log_level_default_is_info() {
        let level = LogLevel::default();
        assert_eq!(level, LogLevel::Info);
    }

    #[test]
    fn log_level_to_filter_mapping() {
        use tracing_subscriber::filter::LevelFilter;

        assert_eq!(LogLevel::Trace.to_filter(), LevelFilter::TRACE);
        assert_eq!(LogLevel::Debug.to_filter(), LevelFilter::DEBUG);
        assert_eq!(LogLevel::Info.to_filter(), LevelFilter::INFO);
        assert_eq!(LogLevel::Warn.to_filter(), LevelFilter::WARN);
        assert_eq!(LogLevel::Error.to_filter(), LevelFilter::ERROR);
    }

    #[test]
    fn logging_error_no_data_dir_display() {
        let error = LoggingError::no_data_dir();
        let message = error.to_string();
        assert!(message.contains("XDG"));
        assert!(message.contains("data directory"));
    }

    #[test]
    fn logging_error_no_data_dir_is_no_data_dir() {
        let error = LoggingError::no_data_dir();
        assert!(error.is_no_data_dir());
    }

    #[test]
    fn logging_error_create_dir_failed_display() {
        let error =
            LoggingError::create_dir_failed(PathBuf::from("/nonexistent/path"), "permission denied");
        let message = error.to_string();
        assert!(message.contains("/nonexistent/path"));
        assert!(message.contains("permission denied"));
    }

    #[test]
    fn logging_error_subscriber_init_failed_display() {
        let error = LoggingError::subscriber_init_failed("already initialized");
        let message = error.to_string();
        assert!(message.contains("already initialized"));
        assert!(message.contains("subscriber"));
    }

    #[test]
    fn logging_errors_are_clone() {
        let error1 = LoggingError::no_data_dir();
        let error2 = error1.clone();
        assert_eq!(error1, error2);
    }

    #[test]
    fn logging_errors_are_eq() {
        let error1 = LoggingError::no_data_dir();
        let error2 = LoggingError::no_data_dir();
        assert_eq!(error1, error2);

        let error3 = LoggingError::subscriber_init_failed("test");
        assert_ne!(error1, error3);
    }

    #[test]
    fn resolve_log_dir_uses_custom_when_provided() {
        let config = LoggingConfig::default().with_log_dir("/custom/logs");

        let resolved = resolve_log_dir(&config).unwrap();
        assert_eq!(resolved, PathBuf::from("/custom/logs"));
    }

    #[test]
    fn resolve_log_dir_uses_xdg_when_not_provided() {
        let config = LoggingConfig::default();

        // This test may fail if there's no home dir, but on most systems it should work
        if let Ok(resolved) = resolve_log_dir(&config) {
            assert!(resolved.to_string_lossy().contains("acton"));
            assert!(resolved.to_string_lossy().contains("logs"));
        }
    }

    #[test]
    fn init_file_logging_returns_none_when_disabled() {
        let config = LoggingConfig::disabled();
        let result = init_file_logging(&config);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn logging_config_serialization_roundtrip() {
        let config = LoggingConfig::new()
            .with_app_name("test-app")
            .with_level(LogLevel::Debug);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: LoggingConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }

    #[test]
    fn log_level_serialization_roundtrip() {
        for level in [
            LogLevel::Trace,
            LogLevel::Debug,
            LogLevel::Info,
            LogLevel::Warn,
            LogLevel::Error,
        ] {
            let json = serde_json::to_string(&level).unwrap();
            let deserialized: LogLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(level, deserialized);
        }
    }
}
