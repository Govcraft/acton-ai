//! Tool Registry actor implementation.
//!
//! The Tool Registry is the central registry for all tools in the system.
//! It handles tool registration, validation, and execution dispatch.

use crate::messages::{ExecuteTool, ToolDefinition, ToolResponse};
use crate::tools::definition::{BoxedToolExecutor, ToolConfig};
use crate::tools::error::{ToolError, ToolErrorKind};
use crate::tools::sandbox::SandboxFactory;
use acton_reactive::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Message to initialize the Tool Registry.
#[acton_message]
pub struct InitToolRegistry;

/// Message to register a tool with the registry.
#[acton_message]
pub struct RegisterTool {
    /// The tool configuration
    pub config: ToolConfig,
    /// The tool executor
    pub executor: Arc<BoxedToolExecutor>,
}

/// Message to unregister a tool from the registry.
#[acton_message]
pub struct UnregisterTool {
    /// The name of the tool to unregister
    pub tool_name: String,
}

/// Message to list all registered tools.
#[acton_message]
pub struct ListTools;

/// Message to configure the sandbox factory for sandboxed tool execution.
#[acton_message]
pub struct ConfigureSandbox {
    /// The sandbox factory to use for sandboxed tools
    pub factory: Arc<dyn SandboxFactory>,
}

/// Response containing the list of registered tools.
#[acton_message]
pub struct ToolListResponse {
    /// The list of tool definitions
    pub tools: Vec<ToolDefinition>,
}

/// The Tool Registry actor state.
///
/// Manages tool registration, validation, and execution dispatch.
#[acton_actor]
pub struct ToolRegistry {
    /// Registered tools by name
    pub tools: HashMap<String, RegisteredTool>,
    /// Optional sandbox factory for sandboxed tool execution
    pub sandbox_factory: Option<Arc<dyn SandboxFactory>>,
    /// Whether the registry is shutting down
    pub shutting_down: bool,
    /// Metrics
    pub metrics: RegistryMetrics,
}

/// A registered tool entry.
#[derive(Debug, Clone)]
pub struct RegisteredTool {
    /// The tool configuration
    pub config: ToolConfig,
    /// The tool executor (wrapped in Arc for cloning)
    pub executor: Arc<BoxedToolExecutor>,
}

/// Metrics for the Tool Registry.
#[derive(Debug, Clone, Default)]
pub struct RegistryMetrics {
    /// Total tools registered
    pub tools_registered: u64,
    /// Total tools unregistered
    pub tools_unregistered: u64,
    /// Total executions requested
    pub executions_requested: u64,
    /// Total executions succeeded
    pub executions_succeeded: u64,
    /// Total executions failed
    pub executions_failed: u64,
}

impl ToolRegistry {
    /// Spawns the Tool Registry actor.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The ActorRuntime
    ///
    /// # Returns
    ///
    /// The ActorHandle for the started registry.
    pub async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<ToolRegistry>("tool_registry".to_string());

        // Set up lifecycle hooks
        builder
            .before_start(|_actor| {
                tracing::debug!("Tool Registry initializing");
                Reply::ready()
            })
            .after_start(|actor| {
                tracing::info!(
                    tools_count = actor.model.tools.len(),
                    "Tool Registry ready to accept registrations"
                );
                Reply::ready()
            })
            .before_stop(|actor| {
                tracing::info!(
                    tools_registered = actor.model.metrics.tools_registered,
                    executions_requested = actor.model.metrics.executions_requested,
                    "Tool Registry shutting down"
                );
                Reply::ready()
            });

        // Configure message handlers
        configure_handlers(&mut builder);

        builder.start().await
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Checks if a tool is registered.
    #[must_use]
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
}

/// Configures message handlers for the Tool Registry actor.
fn configure_handlers(builder: &mut ManagedActor<Idle, ToolRegistry>) {
    // Handle initialization
    builder.mutate_on::<InitToolRegistry>(|_actor, _envelope| {
        tracing::info!("Tool Registry initialized");
        Reply::ready()
    });

    // Handle sandbox factory configuration
    builder.mutate_on::<ConfigureSandbox>(|actor, envelope| {
        let factory = envelope.message().factory.clone();
        let available = factory.is_available();
        actor.model.sandbox_factory = Some(factory);
        tracing::info!(sandbox_available = available, "Sandbox factory configured");
        Reply::ready()
    });

    // Handle tool registration with fallible handler
    builder
        .try_mutate_on::<RegisterTool, (), ToolError>(|actor, envelope| {
            if actor.model.shutting_down {
                return Reply::try_err(ToolError::shutting_down());
            }

            let msg = envelope.message();
            let tool_name = msg.config.definition.name.clone();

            // Check if already registered
            if actor.model.tools.contains_key(&tool_name) {
                return Reply::try_err(ToolError::already_registered(&tool_name));
            }

            // Register the tool
            actor.model.tools.insert(
                tool_name.clone(),
                RegisteredTool {
                    config: msg.config.clone(),
                    executor: msg.executor.clone(),
                },
            );
            actor.model.metrics.tools_registered += 1;

            tracing::info!(
                tool_name = %tool_name,
                sandboxed = msg.config.sandboxed,
                "Tool registered"
            );

            Reply::try_ok(())
        })
        .on_error::<RegisterTool, ToolError>(|_actor, envelope, error| {
            let tool_name = &envelope.message().config.definition.name;
            tracing::error!(
                tool_name = %tool_name,
                error = %error,
                "Tool registration failed"
            );
            Box::pin(async {})
        });

    // Handle tool unregistration
    builder
        .try_mutate_on::<UnregisterTool, (), ToolError>(|actor, envelope| {
            if actor.model.shutting_down {
                return Reply::try_err(ToolError::shutting_down());
            }

            let tool_name = &envelope.message().tool_name;

            if actor.model.tools.remove(tool_name).is_some() {
                actor.model.metrics.tools_unregistered += 1;
                tracing::info!(tool_name = %tool_name, "Tool unregistered");
                Reply::try_ok(())
            } else {
                Reply::try_err(ToolError::not_found(tool_name))
            }
        })
        .on_error::<UnregisterTool, ToolError>(|_actor, envelope, error| {
            let tool_name = &envelope.message().tool_name;
            tracing::error!(
                tool_name = %tool_name,
                error = %error,
                "Tool unregistration failed"
            );
            Box::pin(async {})
        });

    // Handle tool execution with fallible handler
    builder
        .try_mutate_on::<ExecuteTool, (), ToolError>(|actor, envelope| {
            if actor.model.shutting_down {
                return Reply::try_err(ToolError::shutting_down());
            }

            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_name = msg.tool_call.name.clone();
            let args = msg.tool_call.arguments.clone();
            let tool_call_id = msg.tool_call.id.clone();

            actor.model.metrics.executions_requested += 1;

            // Look up the tool
            let Some(registered) = actor.model.tools.get(&tool_name) else {
                return Reply::try_err(ToolError::with_correlation(
                    correlation_id,
                    ToolErrorKind::NotFound {
                        tool_name: tool_name.clone(),
                    },
                ));
            };

            let executor = registered.executor.clone();
            let is_sandboxed = registered.config.sandboxed;
            let sandbox_factory = actor.model.sandbox_factory.clone();
            let broker = actor.broker().clone();

            // Execute the tool
            Reply::try_pending(async move {
                // Validate arguments
                if let Err(e) = executor.validate_args(&args) {
                    broker
                        .broadcast(ToolResponse {
                            correlation_id: correlation_id.clone(),
                            tool_call_id: tool_call_id.clone(),
                            result: Err(e.to_string()),
                        })
                        .await;
                    return Err(e);
                }

                // Check if sandboxed execution is required
                if is_sandboxed {
                    let Some(factory) = sandbox_factory else {
                        // No sandbox factory configured - return error
                        let err = ToolError::sandbox_error(
                            "tool requires sandbox but no sandbox factory is configured",
                        );
                        broker
                            .broadcast(ToolResponse {
                                correlation_id: correlation_id.clone(),
                                tool_call_id: tool_call_id.clone(),
                                result: Err(err.to_string()),
                            })
                            .await;
                        return Err(err);
                    };

                    // Create sandbox and execute
                    match factory.create().await {
                        Ok(mut sandbox) => {
                            // Execute in sandbox
                            match sandbox.execute(&tool_name, args).await {
                                Ok(result) => {
                                    let result_str = serde_json::to_string(&result)
                                        .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e));

                                    sandbox.destroy();

                                    broker
                                        .broadcast(ToolResponse {
                                            correlation_id,
                                            tool_call_id,
                                            result: Ok(result_str),
                                        })
                                        .await;
                                    Ok(())
                                }
                                Err(e) => {
                                    sandbox.destroy();

                                    broker
                                        .broadcast(ToolResponse {
                                            correlation_id: correlation_id.clone(),
                                            tool_call_id,
                                            result: Err(e.to_string()),
                                        })
                                        .await;
                                    Err(ToolError::with_correlation(
                                        correlation_id,
                                        ToolErrorKind::ExecutionFailed {
                                            tool_name,
                                            reason: e.to_string(),
                                        },
                                    ))
                                }
                            }
                        }
                        Err(e) => {
                            broker
                                .broadcast(ToolResponse {
                                    correlation_id: correlation_id.clone(),
                                    tool_call_id,
                                    result: Err(e.to_string()),
                                })
                                .await;
                            Err(e)
                        }
                    }
                } else {
                    // Non-sandboxed execution - execute inline
                    match executor.execute(args).await {
                        Ok(result) => {
                            let result_str = serde_json::to_string(&result)
                                .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e));

                            broker
                                .broadcast(ToolResponse {
                                    correlation_id,
                                    tool_call_id,
                                    result: Ok(result_str),
                                })
                                .await;
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
                            Err(ToolError::with_correlation(
                                correlation_id,
                                ToolErrorKind::ExecutionFailed {
                                    tool_name,
                                    reason: e.to_string(),
                                },
                            ))
                        }
                    }
                }
            })
        })
        .on_error::<ExecuteTool, ToolError>(|actor, envelope, error| {
            let tool_name = &envelope.message().tool_call.name;
            tracing::error!(
                tool_name = %tool_name,
                correlation_id = %envelope.message().correlation_id,
                error = %error,
                "Tool execution failed"
            );
            actor.model.metrics.executions_failed += 1;
            Box::pin(async {})
        });

    // Handle list tools request (read-only)
    builder.act_on::<ListTools>(|actor, envelope| {
        let tools: Vec<ToolDefinition> = actor
            .model
            .tools
            .values()
            .map(|t| t.config.definition.clone())
            .collect();

        let reply = envelope.reply_envelope();

        Reply::pending(async move {
            reply.send(ToolListResponse { tools }).await;
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_metrics_default() {
        let metrics = RegistryMetrics::default();

        assert_eq!(metrics.tools_registered, 0);
        assert_eq!(metrics.tools_unregistered, 0);
        assert_eq!(metrics.executions_requested, 0);
        assert_eq!(metrics.executions_succeeded, 0);
        assert_eq!(metrics.executions_failed, 0);
    }

    #[test]
    fn registered_tool_is_clone() {
        use crate::messages::ToolDefinition;

        // We can't easily test RegisteredTool without a real executor,
        // but we can test the ToolConfig part
        let def = ToolDefinition {
            name: "test".to_string(),
            description: "desc".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let config = ToolConfig::new(def);
        let config2 = config.clone();
        assert_eq!(config.definition.name, config2.definition.name);
    }
}
