//! Tool error types.
//!
//! Custom error types for tool operations including registration,
//! validation, execution, and sandbox errors.

use crate::types::CorrelationId;
use std::fmt;
use std::time::Duration;

/// Errors that can occur in tool operations.
///
/// This type uses Box<ToolErrorKind> to keep the error size small,
/// enabling efficient use in Result types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError {
    /// Optional correlation ID for tracking
    pub correlation_id: Option<CorrelationId>,
    /// The specific error that occurred (boxed for size efficiency)
    kind: Box<ToolErrorKind>,
}

/// Specific tool error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolErrorKind {
    /// Tool not found in registry
    NotFound {
        /// The name of the tool that was not found
        tool_name: String,
    },
    /// Tool already registered
    AlreadyRegistered {
        /// The name of the existing tool
        tool_name: String,
    },
    /// Tool execution failed
    ExecutionFailed {
        /// The name of the tool
        tool_name: String,
        /// Reason for failure
        reason: String,
    },
    /// Tool execution timed out
    Timeout {
        /// The name of the tool
        tool_name: String,
        /// The timeout duration that was exceeded
        duration: Duration,
    },
    /// Tool validation failed (invalid arguments)
    ValidationFailed {
        /// The name of the tool
        tool_name: String,
        /// What was invalid
        reason: String,
    },
    /// Sandbox error
    SandboxError {
        /// Description of the sandbox error
        message: String,
    },
    /// Registry is shutting down
    ShuttingDown,
    /// Internal error
    Internal {
        /// Description of the internal error
        message: String,
    },
}

impl ToolError {
    /// Creates a new ToolError with the given kind.
    #[must_use]
    pub fn new(kind: ToolErrorKind) -> Self {
        Self {
            correlation_id: None,
            kind: Box::new(kind),
        }
    }

    /// Creates a new ToolError with a correlation ID.
    #[must_use]
    pub fn with_correlation(correlation_id: CorrelationId, kind: ToolErrorKind) -> Self {
        Self {
            correlation_id: Some(correlation_id),
            kind: Box::new(kind),
        }
    }

    /// Returns a reference to the error kind.
    #[must_use]
    pub fn kind(&self) -> &ToolErrorKind {
        &self.kind
    }

    /// Creates a not found error.
    #[must_use]
    pub fn not_found(tool_name: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::NotFound {
            tool_name: tool_name.into(),
        })
    }

    /// Creates an already registered error.
    #[must_use]
    pub fn already_registered(tool_name: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::AlreadyRegistered {
            tool_name: tool_name.into(),
        })
    }

    /// Creates an execution failed error.
    #[must_use]
    pub fn execution_failed(tool_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::ExecutionFailed {
            tool_name: tool_name.into(),
            reason: reason.into(),
        })
    }

    /// Creates a timeout error.
    #[must_use]
    pub fn timeout(tool_name: impl Into<String>, duration: Duration) -> Self {
        Self::new(ToolErrorKind::Timeout {
            tool_name: tool_name.into(),
            duration,
        })
    }

    /// Creates a validation failed error.
    #[must_use]
    pub fn validation_failed(tool_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::ValidationFailed {
            tool_name: tool_name.into(),
            reason: reason.into(),
        })
    }

    /// Creates a sandbox error.
    #[must_use]
    pub fn sandbox_error(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::SandboxError {
            message: message.into(),
        })
    }

    /// Creates a shutting down error.
    #[must_use]
    pub fn shutting_down() -> Self {
        Self::new(ToolErrorKind::ShuttingDown)
    }

    /// Creates an internal error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Internal {
            message: message.into(),
        })
    }

    /// Returns true if this error is retriable.
    #[must_use]
    pub fn is_retriable(&self) -> bool {
        matches!(
            *self.kind,
            ToolErrorKind::Timeout { .. } | ToolErrorKind::SandboxError { .. }
        )
    }

    /// Returns true if this error indicates the tool was not found.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        matches!(*self.kind, ToolErrorKind::NotFound { .. })
    }

    /// Returns true if this error indicates the tool is already registered.
    #[must_use]
    pub fn is_already_registered(&self) -> bool {
        matches!(*self.kind, ToolErrorKind::AlreadyRegistered { .. })
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref corr_id) = self.correlation_id {
            write!(f, "[{}] ", corr_id)?;
        }

        match self.kind.as_ref() {
            ToolErrorKind::NotFound { tool_name } => {
                write!(
                    f,
                    "tool '{}' not found; verify the tool is registered",
                    tool_name
                )
            }
            ToolErrorKind::AlreadyRegistered { tool_name } => {
                write!(
                    f,
                    "tool '{}' is already registered; unregister it first or use a different name",
                    tool_name
                )
            }
            ToolErrorKind::ExecutionFailed { tool_name, reason } => {
                write!(f, "tool '{}' execution failed: {}", tool_name, reason)
            }
            ToolErrorKind::Timeout {
                tool_name,
                duration,
            } => {
                write!(
                    f,
                    "tool '{}' timed out after {} seconds",
                    tool_name,
                    duration.as_secs()
                )
            }
            ToolErrorKind::ValidationFailed { tool_name, reason } => {
                write!(
                    f,
                    "tool '{}' validation failed: {}; check the input arguments",
                    tool_name, reason
                )
            }
            ToolErrorKind::SandboxError { message } => {
                write!(f, "sandbox error: {}", message)
            }
            ToolErrorKind::ShuttingDown => {
                write!(
                    f,
                    "tool registry is shutting down; cannot accept new requests"
                )
            }
            ToolErrorKind::Internal { message } => {
                write!(f, "internal tool error: {}", message)
            }
        }
    }
}

impl std::error::Error for ToolError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_error_not_found_display() {
        let error = ToolError::not_found("calculator");
        let message = error.to_string();
        assert!(message.contains("calculator"));
        assert!(message.contains("not found"));
    }

    #[test]
    fn tool_error_already_registered_display() {
        let error = ToolError::already_registered("search");
        let message = error.to_string();
        assert!(message.contains("search"));
        assert!(message.contains("already registered"));
    }

    #[test]
    fn tool_error_execution_failed_display() {
        let error = ToolError::execution_failed("parser", "invalid JSON");
        let message = error.to_string();
        assert!(message.contains("parser"));
        assert!(message.contains("execution failed"));
        assert!(message.contains("invalid JSON"));
    }

    #[test]
    fn tool_error_timeout_display() {
        let error = ToolError::timeout("slow_tool", Duration::from_secs(30));
        let message = error.to_string();
        assert!(message.contains("slow_tool"));
        assert!(message.contains("timed out"));
        assert!(message.contains("30"));
    }

    #[test]
    fn tool_error_validation_failed_display() {
        let error = ToolError::validation_failed("query", "missing required field");
        let message = error.to_string();
        assert!(message.contains("query"));
        assert!(message.contains("validation failed"));
        assert!(message.contains("missing required field"));
    }

    #[test]
    fn tool_error_sandbox_display() {
        let error = ToolError::sandbox_error("memory limit exceeded");
        let message = error.to_string();
        assert!(message.contains("sandbox error"));
        assert!(message.contains("memory limit exceeded"));
    }

    #[test]
    fn tool_error_shutting_down_display() {
        let error = ToolError::shutting_down();
        let message = error.to_string();
        assert!(message.contains("shutting down"));
    }

    #[test]
    fn tool_error_internal_display() {
        let error = ToolError::internal("unexpected state");
        let message = error.to_string();
        assert!(message.contains("internal tool error"));
        assert!(message.contains("unexpected state"));
    }

    #[test]
    fn tool_error_with_correlation_id() {
        let corr_id = CorrelationId::new();
        let error = ToolError::with_correlation(
            corr_id.clone(),
            ToolErrorKind::NotFound {
                tool_name: "test".to_string(),
            },
        );
        let message = error.to_string();
        assert!(message.contains(&corr_id.to_string()));
    }

    #[test]
    fn tool_error_is_retriable_for_timeout() {
        let error = ToolError::timeout("slow_tool", Duration::from_secs(30));
        assert!(error.is_retriable());
    }

    #[test]
    fn tool_error_is_retriable_for_sandbox() {
        let error = ToolError::sandbox_error("transient failure");
        assert!(error.is_retriable());
    }

    #[test]
    fn tool_error_not_retriable_for_not_found() {
        let error = ToolError::not_found("missing");
        assert!(!error.is_retriable());
    }

    #[test]
    fn tool_error_not_retriable_for_already_registered() {
        let error = ToolError::already_registered("existing");
        assert!(!error.is_retriable());
    }

    #[test]
    fn tool_error_not_retriable_for_validation() {
        let error = ToolError::validation_failed("tool", "bad args");
        assert!(!error.is_retriable());
    }

    #[test]
    fn tool_error_is_not_found() {
        let error = ToolError::not_found("missing");
        assert!(error.is_not_found());

        let error2 = ToolError::already_registered("existing");
        assert!(!error2.is_not_found());
    }

    #[test]
    fn tool_error_is_already_registered() {
        let error = ToolError::already_registered("existing");
        assert!(error.is_already_registered());

        let error2 = ToolError::not_found("missing");
        assert!(!error2.is_already_registered());
    }

    #[test]
    fn errors_are_clone() {
        let error1 = ToolError::shutting_down();
        let error2 = error1.clone();
        assert_eq!(error1, error2);
    }

    #[test]
    fn errors_are_eq() {
        let error1 = ToolError::shutting_down();
        let error2 = ToolError::shutting_down();
        assert_eq!(error1, error2);

        let error3 = ToolError::not_found("different");
        assert_ne!(error1, error3);
    }

    #[test]
    fn tool_error_kind_accessor() {
        let error = ToolError::not_found("test");
        assert!(matches!(error.kind(), ToolErrorKind::NotFound { .. }));
    }
}
