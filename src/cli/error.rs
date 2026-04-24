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
    SessionNotFound {
        name: String,
        /// Existing session names (for "did you mean?" suggestions and
        /// listing).
        available: Vec<String>,
        /// Closest-match suggestion derived from `available` via strsim.
        suggestion: Option<String>,
    },
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

    /// Creates a session not found error, computing a "did you mean?"
    /// suggestion from the list of existing sessions via Jaro-Winkler
    /// similarity.
    #[must_use]
    pub fn session_not_found(name: impl Into<String>, available: Vec<String>) -> Self {
        let name = name.into();
        let suggestion = closest_match(&name, &available, 0.75);
        Self {
            kind: CliErrorKind::SessionNotFound {
                name,
                available,
                suggestion,
            },
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

impl CliError {
    /// Actionable advice rendered on a second, dim line after the primary
    /// `error: ...` line. Returns `None` when the error text already tells
    /// the user everything useful.
    #[must_use]
    pub fn hint(&self) -> Option<String> {
        match &self.kind {
            CliErrorKind::SessionNotFound {
                available,
                suggestion,
                ..
            } => {
                let mut parts = Vec::new();
                if let Some(s) = suggestion {
                    parts.push(format!("did you mean '{s}'?"));
                }
                if !available.is_empty() {
                    parts.push(format!(
                        "existing sessions: {}",
                        available.join(", ")
                    ));
                } else {
                    parts.push("pass --create to create a new session".to_string());
                }
                parts.push("run `acton-ai session list` for details".to_string());
                Some(parts.join("  "))
            }
            CliErrorKind::SessionAlreadyExists { .. } => Some(
                "pass a different --session name or drop --create to resume".to_string(),
            ),
            CliErrorKind::NoInput => Some(
                "pipe input (`echo hi | acton-ai chat`) or pass `-m \"...\"`".to_string(),
            ),
            CliErrorKind::ProviderUnavailable { .. } => Some(
                "check the provider config in acton-ai.toml; run with `-vv` for details"
                    .to_string(),
            ),
            CliErrorKind::Configuration(_) => {
                Some("see acton-ai.toml search paths: ./acton-ai.toml then ~/.config/acton-ai/config.toml".to_string())
            }
            _ => None,
        }
    }
}

/// Return the closest string in `candidates` to `input` above `threshold`,
/// measured by Jaro-Winkler similarity. Returns `None` when nothing is
/// close enough.
fn closest_match(input: &str, candidates: &[String], threshold: f64) -> Option<String> {
    candidates
        .iter()
        .map(|c| (c.clone(), strsim::jaro_winkler(input, c)))
        .filter(|(_, score)| *score >= threshold)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(c, _)| c)
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            CliErrorKind::Configuration(msg) => write!(f, "configuration error: {msg}"),
            CliErrorKind::SessionNotFound { name, .. } => {
                write!(f, "session '{name}' not found")
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
