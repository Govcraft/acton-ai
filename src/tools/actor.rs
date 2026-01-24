//! Tool actor trait and message types.
//!
//! Defines the base trait for tool actors that can be spawned per-agent.
//! Each tool becomes its own actor, supervised by the agent that uses it.

use crate::messages::ToolDefinition;
use crate::tools::error::ToolError;
use crate::types::CorrelationId;
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;

/// Message to execute a tool directly (sent to individual tool actors).
///
/// Unlike `ExecuteTool` which is routed through the registry, this message
/// is sent directly to a specific tool actor.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct ExecuteToolDirect {
    /// Correlation ID for tracking the request-response cycle
    pub correlation_id: CorrelationId,
    /// The tool call ID for matching responses
    pub tool_call_id: String,
    /// The arguments to pass to the tool
    pub args: Value,
}

impl ExecuteToolDirect {
    /// Creates a new execute tool direct message.
    #[must_use]
    pub fn new(
        correlation_id: CorrelationId,
        tool_call_id: impl Into<String>,
        args: Value,
    ) -> Self {
        Self {
            correlation_id,
            tool_call_id: tool_call_id.into(),
            args,
        }
    }
}

/// Response from a tool actor execution.
///
/// Broadcast by tool actors after execution completes.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct ToolActorResponse {
    /// Correlation ID matching the request
    pub correlation_id: CorrelationId,
    /// The tool call ID this responds to
    pub tool_call_id: String,
    /// The result of execution (success content or error message)
    pub result: Result<String, String>,
}

impl ToolActorResponse {
    /// Creates a successful response.
    #[must_use]
    pub fn success(
        correlation_id: CorrelationId,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            correlation_id,
            tool_call_id: tool_call_id.into(),
            result: Ok(content.into()),
        }
    }

    /// Creates an error response.
    #[must_use]
    pub fn error(
        correlation_id: CorrelationId,
        tool_call_id: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            correlation_id,
            tool_call_id: tool_call_id.into(),
            result: Err(error.into()),
        }
    }
}

/// Trait for tools that can be spawned as actors.
///
/// Each tool implements this trait to provide:
/// - A factory method to spawn the tool as an actor
/// - A definition method returning the tool's LLM-visible schema
/// - A name method for routing
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::actor::ToolActor;
///
/// struct MyTool;
///
/// impl ToolActor for MyTool {
///     fn name() -> &'static str {
///         "my_tool"
///     }
///
///     fn definition() -> ToolDefinition {
///         ToolDefinition {
///             name: "my_tool".to_string(),
///             description: "Does something useful".to_string(),
///             input_schema: serde_json::json!({
///                 "type": "object",
///                 "properties": {}
///             }),
///         }
///     }
///
///     async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
///         // ... spawn and configure actor
///     }
/// }
/// ```
pub trait ToolActor {
    /// Returns the tool name used for routing.
    fn name() -> &'static str;

    /// Returns the tool definition for the LLM.
    fn definition() -> ToolDefinition;

    /// Spawns the tool as an actor.
    ///
    /// The returned handle can be used to send `ExecuteToolDirect` messages.
    fn spawn(runtime: &mut ActorRuntime) -> impl Future<Output = ActorHandle> + Send;
}

/// Helper trait for executing tool logic asynchronously.
///
/// This separates the async execution logic from the actor state management,
/// making it easier to convert existing tools to actors.
pub trait ToolExecutor: Send + Sync {
    /// Executes the tool with the given arguments.
    fn execute(&self, args: Value) -> impl Future<Output = Result<Value, ToolError>> + Send;

    /// Validates the input arguments before execution.
    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let _ = args;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execute_tool_direct_new() {
        let corr_id = CorrelationId::new();
        let msg = ExecuteToolDirect::new(corr_id.clone(), "tc_123", serde_json::json!({}));

        assert_eq!(msg.correlation_id, corr_id);
        assert_eq!(msg.tool_call_id, "tc_123");
    }

    #[test]
    fn tool_actor_response_success() {
        let corr_id = CorrelationId::new();
        let resp = ToolActorResponse::success(corr_id.clone(), "tc_123", "result");

        assert_eq!(resp.correlation_id, corr_id);
        assert_eq!(resp.tool_call_id, "tc_123");
        assert!(resp.result.is_ok());
        assert_eq!(resp.result.unwrap(), "result");
    }

    #[test]
    fn tool_actor_response_error() {
        let corr_id = CorrelationId::new();
        let resp = ToolActorResponse::error(corr_id.clone(), "tc_123", "failed");

        assert_eq!(resp.correlation_id, corr_id);
        assert_eq!(resp.tool_call_id, "tc_123");
        assert!(resp.result.is_err());
        assert_eq!(resp.result.unwrap_err(), "failed");
    }
}
