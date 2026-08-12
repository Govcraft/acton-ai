//! Custom error types for the Acton-AI framework.
//!
//! This module contains all error types used throughout the framework.
//! Each error type implements Display, Debug, Clone, PartialEq, Eq, and std::error::Error.
//!
//! No external error crates (anyhow, thiserror, eyre) are used.

use crate::types::AgentId;
use std::fmt;

/// Errors that can occur in an Agent actor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentError {
    /// The agent that encountered the error, if known
    pub agent_id: Option<AgentId>,
    /// The specific error that occurred
    pub kind: AgentErrorKind,
}

/// Specific agent error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentErrorKind {
    /// Agent is not in a valid state for the requested operation
    InvalidState {
        /// Current state
        current: String,
        /// Expected state(s)
        expected: String,
    },
    /// Message processing failed
    ProcessingFailed {
        /// Description of what failed
        reason: String,
    },
    /// LLM request failed
    LLMRequestFailed {
        /// Error from the LLM provider
        reason: String,
    },
    /// Tool execution failed
    ToolExecutionFailed {
        /// Name of the tool
        tool_name: String,
        /// Reason for failure
        reason: String,
    },
    /// Agent is stopping and cannot process new messages
    Stopping,
    /// Configuration error
    InvalidConfig {
        /// Description of what was invalid
        field: String,
        /// Why it was invalid
        reason: String,
    },
}

impl AgentError {
    /// Creates a new AgentError with the given kind.
    #[must_use]
    pub fn new(agent_id: Option<AgentId>, kind: AgentErrorKind) -> Self {
        Self { agent_id, kind }
    }

    /// Creates an invalid state error.
    #[must_use]
    pub fn invalid_state(
        agent_id: Option<AgentId>,
        current: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self {
        Self::new(
            agent_id,
            AgentErrorKind::InvalidState {
                current: current.into(),
                expected: expected.into(),
            },
        )
    }

    /// Creates a processing failed error.
    #[must_use]
    pub fn processing_failed(agent_id: Option<AgentId>, reason: impl Into<String>) -> Self {
        Self::new(
            agent_id,
            AgentErrorKind::ProcessingFailed {
                reason: reason.into(),
            },
        )
    }

    /// Creates an LLM request failed error.
    #[must_use]
    pub fn llm_request_failed(agent_id: Option<AgentId>, reason: impl Into<String>) -> Self {
        Self::new(
            agent_id,
            AgentErrorKind::LLMRequestFailed {
                reason: reason.into(),
            },
        )
    }

    /// Creates a tool execution failed error.
    #[must_use]
    pub fn tool_execution_failed(
        agent_id: Option<AgentId>,
        tool_name: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::new(
            agent_id,
            AgentErrorKind::ToolExecutionFailed {
                tool_name: tool_name.into(),
                reason: reason.into(),
            },
        )
    }

    /// Creates a stopping error.
    #[must_use]
    pub fn stopping(agent_id: Option<AgentId>) -> Self {
        Self::new(agent_id, AgentErrorKind::Stopping)
    }

    /// Creates an invalid config error.
    #[must_use]
    pub fn invalid_config(
        agent_id: Option<AgentId>,
        field: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::new(
            agent_id,
            AgentErrorKind::InvalidConfig {
                field: field.into(),
                reason: reason.into(),
            },
        )
    }

    /// Returns true if this error indicates the agent is stopping.
    #[must_use]
    pub fn is_stopping(&self) -> bool {
        matches!(self.kind, AgentErrorKind::Stopping)
    }
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref id) = self.agent_id {
            write!(f, "agent '{}': ", id)?;
        } else {
            write!(f, "agent error: ")?;
        }

        match &self.kind {
            AgentErrorKind::InvalidState { current, expected } => {
                write!(f, "invalid state '{}'; expected {}", current, expected)
            }
            AgentErrorKind::ProcessingFailed { reason } => {
                write!(f, "message processing failed: {}", reason)
            }
            AgentErrorKind::LLMRequestFailed { reason } => {
                write!(f, "LLM request failed: {}", reason)
            }
            AgentErrorKind::ToolExecutionFailed { tool_name, reason } => {
                write!(f, "tool '{}' execution failed: {}", tool_name, reason)
            }
            AgentErrorKind::Stopping => {
                write!(f, "agent is stopping and cannot process new messages")
            }
            AgentErrorKind::InvalidConfig { field, reason } => {
                write!(f, "invalid configuration for '{}': {}", field, reason)
            }
        }
    }
}

impl std::error::Error for AgentError {}

// =============================================================================
// ActonAI High-Level API Error
// =============================================================================

/// Errors that can occur when using the high-level ActonAI API.
///
/// This error type covers all operations in the simplified facade API,
/// including launch, prompt execution, and stream handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActonAIError {
    /// The specific error that occurred
    pub kind: ActonAIErrorKind,
}

/// Specific high-level API error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActonAIErrorKind {
    /// Configuration error during builder setup
    Configuration {
        /// Description of what was invalid
        field: String,
        /// Why it was invalid
        reason: String,
    },
    /// Failed to launch the runtime
    LaunchFailed {
        /// Reason for the failure
        reason: String,
    },
    /// Prompt execution failed
    PromptFailed {
        /// Reason for the failure
        reason: String,
    },
    /// Stream processing error
    StreamError {
        /// Description of the error
        reason: String,
    },
    /// LLM provider error
    ProviderError {
        /// Error from the LLM provider
        reason: String,
    },
    /// Runtime was shut down
    RuntimeShutdown,
}

impl ActonAIError {
    /// Creates a new ActonAIError with the given kind.
    #[must_use]
    pub fn new(kind: ActonAIErrorKind) -> Self {
        Self { kind }
    }

    /// Creates a configuration error.
    #[must_use]
    pub fn configuration(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(ActonAIErrorKind::Configuration {
            field: field.into(),
            reason: reason.into(),
        })
    }

    /// Creates a launch failed error.
    #[must_use]
    pub fn launch_failed(reason: impl Into<String>) -> Self {
        Self::new(ActonAIErrorKind::LaunchFailed {
            reason: reason.into(),
        })
    }

    /// Creates a prompt failed error.
    #[must_use]
    pub fn prompt_failed(reason: impl Into<String>) -> Self {
        Self::new(ActonAIErrorKind::PromptFailed {
            reason: reason.into(),
        })
    }

    /// Creates a stream error.
    #[must_use]
    pub fn stream_error(reason: impl Into<String>) -> Self {
        Self::new(ActonAIErrorKind::StreamError {
            reason: reason.into(),
        })
    }

    /// Creates a provider error.
    #[must_use]
    pub fn provider_error(reason: impl Into<String>) -> Self {
        Self::new(ActonAIErrorKind::ProviderError {
            reason: reason.into(),
        })
    }

    /// Creates a runtime shutdown error.
    #[must_use]
    pub fn runtime_shutdown() -> Self {
        Self::new(ActonAIErrorKind::RuntimeShutdown)
    }

    /// Returns true if this error indicates a configuration problem.
    #[must_use]
    pub fn is_configuration(&self) -> bool {
        matches!(self.kind, ActonAIErrorKind::Configuration { .. })
    }

    /// Returns true if this error indicates the runtime was shut down.
    #[must_use]
    pub fn is_runtime_shutdown(&self) -> bool {
        matches!(self.kind, ActonAIErrorKind::RuntimeShutdown)
    }
}

impl fmt::Display for ActonAIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ActonAIErrorKind::Configuration { field, reason } => {
                write!(f, "configuration error for '{}': {}", field, reason)
            }
            ActonAIErrorKind::LaunchFailed { reason } => {
                write!(f, "failed to launch runtime: {}", reason)
            }
            ActonAIErrorKind::PromptFailed { reason } => {
                write!(f, "prompt execution failed: {}", reason)
            }
            ActonAIErrorKind::StreamError { reason } => {
                write!(f, "stream error: {}", reason)
            }
            ActonAIErrorKind::ProviderError { reason } => {
                write!(f, "LLM provider error: {}", reason)
            }
            ActonAIErrorKind::RuntimeShutdown => {
                write!(f, "runtime has been shut down")
            }
        }
    }
}

impl std::error::Error for ActonAIError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_error_with_agent_id_display() {
        let agent_id = AgentId::new();
        let error =
            AgentError::invalid_state(Some(agent_id.clone()), "Idle", "Thinking or Executing");

        let message = error.to_string();
        assert!(message.contains(&agent_id.to_string()));
        assert!(message.contains("invalid state"));
    }

    #[test]
    fn agent_error_without_agent_id_display() {
        let error = AgentError::processing_failed(None, "unexpected message format");

        let message = error.to_string();
        assert!(message.starts_with("agent error:"));
        assert!(message.contains("processing failed"));
    }

    #[test]
    fn agent_error_is_stopping() {
        let error = AgentError::stopping(None);
        assert!(error.is_stopping());

        let other = AgentError::processing_failed(None, "test");
        assert!(!other.is_stopping());
    }

    #[test]
    fn agent_error_tool_execution_failed() {
        let agent_id = AgentId::new();
        let error =
            AgentError::tool_execution_failed(Some(agent_id), "web_search", "connection timeout");

        let message = error.to_string();
        assert!(message.contains("web_search"));
        assert!(message.contains("connection timeout"));
    }

    // ActonAIError tests
    #[test]
    fn acton_ai_error_configuration_display() {
        let error = ActonAIError::configuration("app_name", "cannot be empty");

        let message = error.to_string();
        assert!(message.contains("app_name"));
        assert!(message.contains("cannot be empty"));
    }

    #[test]
    fn acton_ai_error_launch_failed_display() {
        let error = ActonAIError::launch_failed("failed to spawn provider");

        let message = error.to_string();
        assert!(message.contains("failed to launch"));
        assert!(message.contains("provider"));
    }

    #[test]
    fn acton_ai_error_prompt_failed_display() {
        let error = ActonAIError::prompt_failed("timeout waiting for response");

        let message = error.to_string();
        assert!(message.contains("prompt execution failed"));
        assert!(message.contains("timeout"));
    }

    #[test]
    fn acton_ai_error_stream_error_display() {
        let error = ActonAIError::stream_error("connection reset");

        let message = error.to_string();
        assert!(message.contains("stream error"));
        assert!(message.contains("connection reset"));
    }

    #[test]
    fn acton_ai_error_provider_error_display() {
        let error = ActonAIError::provider_error("rate limit exceeded");

        let message = error.to_string();
        assert!(message.contains("LLM provider"));
        assert!(message.contains("rate limit"));
    }

    #[test]
    fn acton_ai_error_runtime_shutdown_display() {
        let error = ActonAIError::runtime_shutdown();

        let message = error.to_string();
        assert!(message.contains("shut down"));
    }

    #[test]
    fn acton_ai_error_is_configuration() {
        let error = ActonAIError::configuration("field", "reason");
        assert!(error.is_configuration());

        let other = ActonAIError::runtime_shutdown();
        assert!(!other.is_configuration());
    }

    #[test]
    fn acton_ai_error_is_runtime_shutdown() {
        let error = ActonAIError::runtime_shutdown();
        assert!(error.is_runtime_shutdown());

        let other = ActonAIError::prompt_failed("test");
        assert!(!other.is_runtime_shutdown());
    }

    #[test]
    fn acton_ai_errors_are_clone() {
        let error1 = ActonAIError::runtime_shutdown();
        let error2 = error1.clone();
        assert_eq!(error1, error2);
    }

    #[test]
    fn acton_ai_errors_are_eq() {
        let error1 = ActonAIError::runtime_shutdown();
        let error2 = ActonAIError::runtime_shutdown();
        assert_eq!(error1, error2);

        let error3 = ActonAIError::prompt_failed("test");
        assert_ne!(error1, error3);
    }
}
