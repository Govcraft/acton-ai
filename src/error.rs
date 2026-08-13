//! Custom error types for the Acton-AI framework.
//!
//! This module contains all error types used throughout the framework.
//! Each error type implements Display, Debug, Clone, PartialEq, Eq, and std::error::Error.
//!
//! No external error crates (anyhow, thiserror, eyre) are used.

use crate::accounting::BudgetScope;
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
///
/// Marked `#[non_exhaustive]` so new failure kinds can be added without
/// breaking downstream `match`es.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
///
/// Marked `#[non_exhaustive]`: new capabilities add new failure kinds (the
/// `Mcp` variant arrived with MCP client support), so downstream `match`es
/// must carry a wildcard arm rather than breaking on every addition.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    /// An MCP server failed to connect, discover tools, or answer a call.
    ///
    /// Produced by converting a [`McpError`](crate::mcp::McpError); `reason`
    /// carries that error's rendered message.
    Mcp {
        /// The configured name of the MCP server.
        server: String,
        /// Rendered description of the underlying MCP failure.
        reason: String,
    },
    /// Typed structured output could not be produced.
    ///
    /// Raised by [`PromptBuilder::extract`](crate::prompt::PromptBuilder::extract)
    /// when the model never records an answer, when the answers it does
    /// record never deserialize into the target type, or when the target
    /// type's schema cannot be turned into something a provider accepts.
    /// `reason` names which of those happened.
    Extraction {
        /// Rendered description of the extraction failure.
        reason: String,
    },
    /// A configured spending cap was reached, so the request was refused
    /// before it was sent.
    ///
    /// Raised pre-flight by the prompt loop, which means no money was spent
    /// on the refused request. Raise the cap with
    /// [`ActonAIBuilder::budget`](crate::facade::ActonAIBuilder::budget) or
    /// the `[budget]` section of your config file, or start a runtime without
    /// one.
    BudgetExceeded {
        /// Which ceiling was hit — the process-wide total, or one provider.
        scope: BudgetScope,
        /// The ceiling, in integer micro-USD.
        limit_microusd: u64,
        /// Spend in that scope when the request was refused, in micro-USD.
        spent_microusd: u64,
    },
    /// Every provider in a failover chain was unusable, so the round could
    /// not be dispatched anywhere.
    ///
    /// Raised only when a chain is configured. Each attempt is recorded in
    /// order with the reason it was passed over — an open circuit, a spending
    /// cap, or the failure it returned — because with a chain in play "the
    /// request failed" is never the whole story.
    AllProvidersFailed {
        /// Each candidate in chain order, paired with why it did not serve.
        attempts: Vec<ProviderAttempt>,
    },
    /// The runtime is paused or draining, so the turn was refused before it
    /// started.
    ///
    /// Nothing was sent and nothing was spent. Turns already running are
    /// unaffected — pausing closes the door, it does not empty the room — so
    /// this is a transient condition a caller may retry after a `resume`.
    ///
    /// `Draining` is the terminal case: a draining runtime is on its way down
    /// and will not start accepting again.
    TurnsNotAdmitted {
        /// What the gate was set to when the turn was refused.
        state: crate::introspection::AdmissionState,
    },
}

/// One provider's turn in a failover chain, and how it ended.
///
/// Carried by [`ActonAIErrorKind::AllProvidersFailed`], which is only raised
/// once every entry has one of these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderAttempt {
    /// The configured provider name.
    pub provider: String,
    /// Why this provider did not serve the round.
    pub reason: String,
}

impl ProviderAttempt {
    /// Records that `provider` did not serve, for `reason`.
    #[must_use]
    pub fn new(provider: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ProviderAttempt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`: {}", self.provider, self.reason)
    }
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

    /// Creates an MCP error naming the server it concerns.
    #[must_use]
    pub fn mcp(server: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(ActonAIErrorKind::Mcp {
            server: server.into(),
            reason: reason.into(),
        })
    }

    /// Creates a structured-output extraction error.
    ///
    /// `reason` should name the failure class up front — validation
    /// exhausted, the model never called the tool, or schema generation —
    /// because that is what tells the caller whether to fix the prompt, the
    /// type, or the model choice.
    #[must_use]
    pub fn extraction(reason: impl Into<String>) -> Self {
        Self::new(ActonAIErrorKind::Extraction {
            reason: reason.into(),
        })
    }

    /// Creates a budget-exceeded error.
    ///
    /// ```rust
    /// use acton_ai::prelude::{ActonAIError, BudgetScope};
    ///
    /// let err = ActonAIError::budget_exceeded(BudgetScope::Total, 5_000_000, 5_200_000);
    ///
    /// assert!(err.is_budget_exceeded());
    /// assert!(err.to_string().contains("$5.0000"));
    /// ```
    #[must_use]
    pub fn budget_exceeded(scope: BudgetScope, limit_microusd: u64, spent_microusd: u64) -> Self {
        Self::new(ActonAIErrorKind::BudgetExceeded {
            scope,
            limit_microusd,
            spent_microusd,
        })
    }

    /// Creates an all-providers-failed error from the attempts, in chain order.
    ///
    /// ```rust
    /// use acton_ai::prelude::{ActonAIError, ProviderAttempt};
    ///
    /// let err = ActonAIError::all_providers_failed(vec![
    ///     ProviderAttempt::new("primary", "circuit open for another 27s"),
    ///     ProviderAttempt::new("backup", "HTTP 500"),
    /// ]);
    ///
    /// assert!(err.is_all_providers_failed());
    /// // Every candidate is named, so the chain can be diagnosed from the
    /// // error alone.
    /// assert!(err.to_string().contains("`primary`"));
    /// assert!(err.to_string().contains("`backup`"));
    /// ```
    #[must_use]
    pub fn all_providers_failed(attempts: Vec<ProviderAttempt>) -> Self {
        Self::new(ActonAIErrorKind::AllProvidersFailed { attempts })
    }

    /// Creates the error a paused or draining runtime refuses new turns with.
    ///
    /// ```rust
    /// use acton_ai::introspection::AdmissionState;
    /// use acton_ai::prelude::ActonAIError;
    ///
    /// let err = ActonAIError::turns_not_admitted(AdmissionState::Paused);
    ///
    /// assert!(err.is_turns_not_admitted());
    /// // The way back out is in the message, both ways of saying it.
    /// assert!(err.to_string().contains("acton-ai resume"));
    /// ```
    #[must_use]
    pub fn turns_not_admitted(state: crate::introspection::AdmissionState) -> Self {
        Self::new(ActonAIErrorKind::TurnsNotAdmitted { state })
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

    /// Returns true if a spending cap refused this request.
    ///
    /// The check that produces it is pre-flight, so nothing was spent on the
    /// refused request.
    #[must_use]
    pub fn is_budget_exceeded(&self) -> bool {
        matches!(self.kind, ActonAIErrorKind::BudgetExceeded { .. })
    }

    /// Returns true if a failover chain was exhausted without a dispatch.
    ///
    /// Only ever true when a chain is configured; a single-provider runtime
    /// reports the provider's own failure instead.
    #[must_use]
    pub fn is_all_providers_failed(&self) -> bool {
        matches!(self.kind, ActonAIErrorKind::AllProvidersFailed { .. })
    }

    /// Returns true if a paused or draining runtime refused this turn.
    ///
    /// Nothing was sent and nothing was spent, so a caller that retries after
    /// a `resume` loses nothing by having tried.
    #[must_use]
    pub fn is_turns_not_admitted(&self) -> bool {
        matches!(self.kind, ActonAIErrorKind::TurnsNotAdmitted { .. })
    }

    /// The per-provider attempts behind an
    /// [`ActonAIErrorKind::AllProvidersFailed`], in chain order.
    ///
    /// `None` for every other kind, so callers can branch on it without
    /// matching the kind themselves.
    #[must_use]
    pub fn provider_attempts(&self) -> Option<&[ProviderAttempt]> {
        match &self.kind {
            ActonAIErrorKind::AllProvidersFailed { attempts } => Some(attempts),
            _ => None,
        }
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
            ActonAIErrorKind::Mcp { server, reason } => {
                write!(f, "MCP server '{server}': {reason}")
            }
            ActonAIErrorKind::Extraction { reason } => {
                write!(f, "structured output extraction failed: {reason}")
            }
            ActonAIErrorKind::BudgetExceeded {
                scope,
                limit_microusd,
                spent_microusd,
            } => {
                write!(
                    f,
                    "budget exceeded for {scope}: ${:.4} spent against a ${:.4} cap; the request \
                     was refused before it was sent. Raise or remove the cap with \
                     ActonAIBuilder::budget(...) / .budget_usd(...), or the [budget] section of \
                     your config file",
                    crate::accounting::budget::usd(*spent_microusd),
                    crate::accounting::budget::usd(*limit_microusd),
                )
            }
            ActonAIErrorKind::TurnsNotAdmitted { state } => {
                // Both routes back out are named, because whoever reads this
                // may be an operator at a shell or a developer in a debugger,
                // and neither should have to go looking for the other's door.
                write!(
                    f,
                    "the runtime is {state} and is not admitting new turns, so this one was \
                     refused before anything was sent; turns already running are unaffected. \
                     Resume with `acton-ai resume` against this process's introspection socket, \
                     or ActonAI::resume() in-process",
                )
            }
            ActonAIErrorKind::AllProvidersFailed { attempts } => {
                write!(
                    f,
                    "every provider in the failover chain was unusable: {}",
                    attempts
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("; "),
                )
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
    fn acton_ai_error_extraction_display_names_the_failure_class() {
        let error = ActonAIError::extraction("the model never called structured_output");

        let message = error.to_string();
        assert!(message.contains("structured output extraction failed"));
        assert!(message.contains("never called structured_output"));
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
