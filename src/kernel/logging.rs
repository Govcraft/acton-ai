//! Journald-based logging for the Acton-AI kernel.
//!
//! When running on Linux with systemd-journald available, kernel logs are
//! written to the journal tagged with the configured `app_name` as
//! `SYSLOG_IDENTIFIER`. On non-Linux hosts, or on Linux without a running
//! journald socket, initialization becomes a silent no-op — callers can still
//! install their own subscriber (e.g. stderr via `tracing-subscriber`).

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Configuration for kernel logging.
///
/// Logs are sent to systemd-journald under the configured `app_name` as
/// `SYSLOG_IDENTIFIER`. `level` filters which events are forwarded.
///
/// # Example
///
/// ```rust
/// use acton_ai::kernel::LoggingConfig;
///
/// // Use defaults (journald under SYSLOG_IDENTIFIER=acton-ai)
/// let config = LoggingConfig::default();
///
/// // Customize for your application
/// let config = LoggingConfig::new()
///     .with_app_name("my-agent")
///     .with_level(acton_ai::kernel::LogLevel::Debug);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Whether journald logging is enabled.
    pub enabled: bool,
    /// Application name sent to journald as `SYSLOG_IDENTIFIER`.
    pub app_name: String,
    /// Minimum log level forwarded to journald.
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

    /// Sets the application name sent to journald.
    #[must_use]
    pub fn with_app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = name.into();
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
            level: LogLevel::default(),
        }
    }
}

/// Log level filter for kernel logging.
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

/// Sentinel set once any caller successfully installs a tracing subscriber
/// via this module's helpers. Subsequent calls no-op rather than racing.
static LOGGING_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Marks the global tracing subscriber as already installed so subsequent
/// calls to [`init_and_store_logging`] become silent no-ops. The CLI bootstrap
/// uses this when it composes its own subscriber that already includes a
/// journald layer.
pub fn mark_subscriber_installed() {
    LOGGING_INSTALLED.store(true, Ordering::SeqCst);
}

/// Builds a journald tracing layer tagged with the given `app_name`.
///
/// Returns `Ok(None)` when journald is unavailable (non-Linux host, no
/// systemd socket, etc.), so callers can gracefully fall back to stderr-only
/// logging. Returns `Err` only when the config disables logging — in that
/// case there's simply nothing to build.
pub fn journald_layer<S>(
    config: &LoggingConfig,
) -> Option<tracing_subscriber::filter::Filtered<
    tracing_journald::Layer,
    tracing_subscriber::filter::LevelFilter,
    S,
>>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    use tracing_subscriber::Layer as _;

    if !config.enabled {
        return None;
    }

    match tracing_journald::layer() {
        Ok(layer) => Some(
            layer
                .with_syslog_identifier(config.app_name.clone())
                .with_filter(config.level.to_filter()),
        ),
        Err(e) => {
            // Best-effort: on non-Linux or without the journald socket we
            // simply return None and let the caller carry on.
            tracing::debug!(error = %e, "journald unavailable; skipping layer");
            None
        }
    }
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
    /// Subscriber initialization failed (another subscriber was already set).
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

    /// Creates an error for subscriber initialization failure.
    #[must_use]
    pub fn subscriber_init_failed(reason: impl Into<String>) -> Self {
        Self::new(LoggingErrorKind::SubscriberInitFailed {
            reason: reason.into(),
        })
    }
}

impl fmt::Display for LoggingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
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

/// Initializes journald-only logging with the given configuration.
///
/// Creates a subscriber that forwards tracing events to systemd-journald.
/// Callers that need additional layers (e.g. stderr) should compose their
/// own subscriber using [`journald_layer`] and call [`mark_subscriber_installed`]
/// so the kernel's auto-init silently no-ops.
///
/// # Returns
///
/// `Ok(true)` if a journald subscriber was installed.
/// `Ok(false)` if logging is disabled, another subscriber is already set, or
/// journald is unavailable on this host.
/// `Err(LoggingError)` only on unexpected `try_init` failures.
pub fn init_journald_logging(config: &LoggingConfig) -> Result<bool, LoggingError> {
    if !config.enabled {
        return Ok(false);
    }

    let Some(layer) = journald_layer::<tracing_subscriber::Registry>(config) else {
        return Ok(false);
    };

    match tracing_subscriber::registry().with(layer).try_init() {
        Ok(()) => {
            LOGGING_INSTALLED.store(true, Ordering::SeqCst);
            Ok(true)
        }
        Err(e) => Err(LoggingError::subscriber_init_failed(e.to_string())),
    }
}

/// Initializes journald logging unless another caller has already installed
/// a tracing subscriber via this module's helpers.
///
/// This is the entry point `Kernel::spawn_with_config` uses.
///
/// # Returns
///
/// `Ok(true)` if a journald subscriber was installed here.
/// `Ok(false)` if logging is disabled, a subscriber was already installed,
/// or journald is unavailable on this host.
pub fn init_and_store_logging(config: &LoggingConfig) -> Result<bool, LoggingError> {
    if LOGGING_INSTALLED.load(Ordering::SeqCst) {
        return Ok(false);
    }
    init_journald_logging(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logging_config_default_values() {
        let config = LoggingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.app_name, "acton-ai");
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
            .with_level(LogLevel::Debug);

        assert_eq!(config.app_name, "my-app");
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
    fn logging_error_subscriber_init_failed_display() {
        let error = LoggingError::subscriber_init_failed("already initialized");
        let message = error.to_string();
        assert!(message.contains("already initialized"));
        assert!(message.contains("subscriber"));
    }

    #[test]
    fn logging_errors_are_clone_and_eq() {
        let error1 = LoggingError::subscriber_init_failed("x");
        let error2 = error1.clone();
        assert_eq!(error1, error2);

        let error3 = LoggingError::subscriber_init_failed("y");
        assert_ne!(error1, error3);
    }

    #[test]
    fn init_journald_logging_returns_false_when_disabled() {
        let config = LoggingConfig::disabled();
        let result = init_journald_logging(&config);
        assert!(result.is_ok());
        assert!(!result.unwrap());
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
