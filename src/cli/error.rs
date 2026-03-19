//! CLI-specific error types.
//!
//! Maps library errors to user-facing CLI errors with appropriate exit codes.

use crate::error::ActonAIError;
use crate::memory::PersistenceError;
use std::fmt;

/// Exit codes for the CLI.
pub mod exit_code {
    /// Successful execution.
    pub const SUCCESS: i32 = 0;
    /// Runtime error (provider down, session not found, job failed).
    pub const RUNTIME_ERROR: i32 = 1;
    // Note: exit code 2 is used by clap for usage errors automatically.
}

/// CLI error with an associated exit code.
#[derive(Debug)]
pub struct CliError {
    /// The specific error that occurred.
    pub kind: CliErrorKind,
}

/// Specific CLI error types.
#[derive(Debug)]
pub enum CliErrorKind {
    /// Configuration error (bad config file, missing provider, etc.).
    Configuration(String),
    /// Session not found by name.
    SessionNotFound { name: String },
    /// Session already exists (when --create is not appropriate).
    SessionAlreadyExists { name: String },
    /// Job not found by name.
    JobNotFound {
        name: String,
        available: Vec<String>,
    },
    /// LLM provider is unavailable or failed.
    ProviderUnavailable { reason: String },
    /// I/O error (stdin read failure, etc.).
    Io(std::io::Error),
    /// Persistence/database error.
    Persistence(String),
    /// Framework error from the ActonAI library.
    Framework(ActonAIError),
    /// No input provided (neither --message nor stdin).
    NoInput,
}

impl CliError {
    /// Returns the exit code for this error.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        exit_code::RUNTIME_ERROR
    }

    /// Creates a configuration error.
    #[must_use]
    pub fn configuration(msg: impl Into<String>) -> Self {
        Self {
            kind: CliErrorKind::Configuration(msg.into()),
        }
    }

    /// Creates a session not found error.
    #[must_use]
    pub fn session_not_found(name: impl Into<String>) -> Self {
        Self {
            kind: CliErrorKind::SessionNotFound { name: name.into() },
        }
    }

    /// Creates a session already exists error.
    #[must_use]
    pub fn session_already_exists(name: impl Into<String>) -> Self {
        Self {
            kind: CliErrorKind::SessionAlreadyExists { name: name.into() },
        }
    }

    /// Creates a job not found error.
    #[must_use]
    pub fn job_not_found(name: impl Into<String>, available: Vec<String>) -> Self {
        Self {
            kind: CliErrorKind::JobNotFound {
                name: name.into(),
                available,
            },
        }
    }

    /// Creates a provider unavailable error.
    #[must_use]
    pub fn provider_unavailable(reason: impl Into<String>) -> Self {
        Self {
            kind: CliErrorKind::ProviderUnavailable {
                reason: reason.into(),
            },
        }
    }

    /// Creates a no input error.
    #[must_use]
    pub fn no_input() -> Self {
        Self {
            kind: CliErrorKind::NoInput,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            CliErrorKind::Configuration(msg) => write!(f, "configuration error: {msg}"),
            CliErrorKind::SessionNotFound { name } => {
                write!(f, "session '{name}' not found; use --create to create it")
            }
            CliErrorKind::SessionAlreadyExists { name } => {
                write!(f, "session '{name}' already exists")
            }
            CliErrorKind::JobNotFound { name, available } => {
                if available.is_empty() {
                    write!(f, "job '{name}' not found; no jobs defined in config")
                } else {
                    write!(
                        f,
                        "job '{name}' not found; available jobs: {}",
                        available.join(", ")
                    )
                }
            }
            CliErrorKind::ProviderUnavailable { reason } => {
                write!(f, "provider unavailable: {reason}")
            }
            CliErrorKind::Io(err) => write!(f, "I/O error: {err}"),
            CliErrorKind::Persistence(msg) => write!(f, "persistence error: {msg}"),
            CliErrorKind::Framework(err) => write!(f, "{err}"),
            CliErrorKind::NoInput => {
                write!(f, "no input provided; use --message or pipe via stdin")
            }
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            CliErrorKind::Io(err) => Some(err),
            CliErrorKind::Framework(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ActonAIError> for CliError {
    fn from(err: ActonAIError) -> Self {
        Self {
            kind: CliErrorKind::Framework(err),
        }
    }
}

impl From<PersistenceError> for CliError {
    fn from(err: PersistenceError) -> Self {
        Self {
            kind: CliErrorKind::Persistence(err.to_string()),
        }
    }
}

impl From<std::io::Error> for CliError {
    fn from(err: std::io::Error) -> Self {
        Self {
            kind: CliErrorKind::Io(err),
        }
    }
}
