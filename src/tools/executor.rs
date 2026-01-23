//! Tool Executor actor implementation.
//!
//! The Tool Executor is a supervised child actor that executes a single tool call.
//! It uses RestartPolicy::Temporary for one-shot execution.

use crate::messages::ToolResponse;
use crate::tools::definition::BoxedToolExecutor;
use crate::tools::error::ToolError;
use crate::types::CorrelationId;
use acton_reactive::prelude::*;
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;

/// Message to execute a tool.
#[acton_message]
pub struct Execute {
    /// The correlation ID for this execution
    pub correlation_id: CorrelationId,
    /// The tool call ID
    pub tool_call_id: String,
    /// The tool name
    pub tool_name: String,
    /// The arguments to pass to the tool
    pub args: Value,
}

/// The Tool Executor actor state.
///
/// Executes a single tool call and reports the result.
/// Uses RestartPolicy::Temporary - no restart on completion or failure.
#[acton_actor]
pub struct ToolExecutor {
    /// The tool executor implementation
    pub executor: Option<Arc<BoxedToolExecutor>>,
    /// When execution started
    pub started_at: Option<Instant>,
    /// The correlation ID for tracking
    pub correlation_id: Option<CorrelationId>,
}

/// Message to initialize the executor with its dependencies.
#[acton_message]
pub struct InitExecutor {
    /// The tool executor implementation
    pub executor: Arc<BoxedToolExecutor>,
}

impl ToolExecutor {
    /// Creates a new Tool Executor actor.
    ///
    /// This creates an actor configured with RestartPolicy::Temporary,
    /// meaning it will not be restarted after completion or failure.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The ActorRuntime
    /// * `name` - Unique name for this executor
    ///
    /// # Returns
    ///
    /// A configured but not yet started ManagedActor.
    pub fn create(runtime: &mut ActorRuntime, name: String) -> ManagedActor<Idle, ToolExecutor> {
        let mut builder = runtime.new_actor_with_name::<ToolExecutor>(name);

        // Set up lifecycle hooks
        builder
            .before_start(|_actor| {
                tracing::debug!("Tool Executor initializing");
                Reply::ready()
            })
            .before_stop(|actor| {
                if let Some(started_at) = actor.model.started_at {
                    tracing::debug!(
                        duration_ms = started_at.elapsed().as_millis() as u64,
                        "Tool Executor stopping"
                    );
                }
                Reply::ready()
            });

        // Configure message handlers
        configure_executor_handlers(&mut builder);

        builder
    }
}

/// Configures message handlers for the Tool Executor actor.
fn configure_executor_handlers(builder: &mut ManagedActor<Idle, ToolExecutor>) {
    // Handle initialization
    builder.mutate_on::<InitExecutor>(|actor, envelope| {
        let msg = envelope.message();
        actor.model.executor = Some(msg.executor.clone());
        actor.model.started_at = Some(Instant::now());
        tracing::debug!("Tool Executor initialized");
        Reply::ready()
    });

    // Handle execution with fallible handler
    builder
        .try_mutate_on::<Execute, (), ToolError>(|actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let tool_name = msg.tool_name.clone();
            let args = msg.args.clone();

            actor.model.correlation_id = Some(correlation_id.clone());

            let Some(ref executor) = actor.model.executor else {
                return Reply::try_err(ToolError::internal("executor not initialized"));
            };

            let executor = executor.clone();
            let broker = actor.broker().clone();
            let handle = actor.handle().clone();

            Reply::try_pending(async move {
                // Execute the tool
                let result = executor.execute(args).await;

                match result {
                    Ok(value) => {
                        let result_str = serde_json::to_string(&value)
                            .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e));

                        broker
                            .broadcast(ToolResponse {
                                correlation_id,
                                tool_call_id,
                                result: Ok(result_str),
                            })
                            .await;

                        // Signal completion - executor will stop
                        let _ = handle.stop().await;
                        Ok(())
                    }
                    Err(e) => {
                        broker
                            .broadcast(ToolResponse {
                                correlation_id: correlation_id.clone(),
                                tool_call_id,
                                result: Err(e.to_string()),
                            })
                            .await;

                        // Signal completion even on error
                        let _ = handle.stop().await;
                        Err(ToolError::execution_failed(&tool_name, e.to_string()))
                    }
                }
            })
        })
        .on_error::<Execute, ToolError>(|actor, envelope, error| {
            let tool_name = &envelope.message().tool_name;
            tracing::error!(
                tool_name = %tool_name,
                correlation_id = ?actor.model.correlation_id,
                error = %error,
                "Tool execution error"
            );
            Box::pin(async {})
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execute_message_fields() {
        let corr_id = CorrelationId::new();
        let msg = Execute {
            correlation_id: corr_id.clone(),
            tool_call_id: "tc_123".to_string(),
            tool_name: "test_tool".to_string(),
            args: serde_json::json!({"key": "value"}),
        };

        assert_eq!(msg.correlation_id, corr_id);
        assert_eq!(msg.tool_call_id, "tc_123");
        assert_eq!(msg.tool_name, "test_tool");
    }
}
