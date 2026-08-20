//! Tool Registry actor implementation.
//!
//! The Tool Registry is the central registry for all tools in the system.
//! It handles tool registration, validation, and execution dispatch.

use crate::messages::{ExecuteTool, ToolDefinition, ToolResponse};
use crate::tools::definition::{BoxedToolExecutor, ToolConfig};
use crate::tools::error::{ToolError, ToolErrorKind};
use acton_reactive::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// Response containing the list of registered tools.
#[acton_message]
pub struct ToolListResponse {
    /// The list of tool definitions
    pub tools: Vec<ToolDefinition>,
}

impl Request for ListTools {
    type Response = ToolListResponse;
}

/// The Tool Registry actor state.
///
/// Manages tool registration, validation, and execution dispatch.
#[acton_actor]
pub struct ToolRegistry {
    /// Registered tools by name
    pub tools: HashMap<String, RegisteredTool>,
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

/// Counters behind [`RegistryMetrics`].
///
/// Held behind an `Arc` so that the execution handler, which runs read-only and
/// hands its work to a `'static` future, can still record the outcome.
#[derive(Debug, Default)]
struct RegistryCounters {
    tools_registered: AtomicU64,
    tools_unregistered: AtomicU64,
    executions_requested: AtomicU64,
    executions_succeeded: AtomicU64,
    executions_failed: AtomicU64,
}

/// A handle to the Tool Registry's counters.
///
/// Cloning shares the same counters rather than copying their values; use
/// [`snapshot`](Self::snapshot) for a point-in-time copy.
#[derive(Debug, Clone, Default)]
pub struct RegistryMetrics {
    counters: Arc<RegistryCounters>,
}

/// A point-in-time copy of [`RegistryMetrics`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegistryMetricsSnapshot {
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

impl RegistryMetrics {
    /// Total tools registered.
    #[must_use]
    pub fn tools_registered(&self) -> u64 {
        self.counters.tools_registered.load(Ordering::Relaxed)
    }

    /// Total tools unregistered.
    #[must_use]
    pub fn tools_unregistered(&self) -> u64 {
        self.counters.tools_unregistered.load(Ordering::Relaxed)
    }

    /// Total executions requested.
    #[must_use]
    pub fn executions_requested(&self) -> u64 {
        self.counters.executions_requested.load(Ordering::Relaxed)
    }

    /// Total executions that completed successfully.
    #[must_use]
    pub fn executions_succeeded(&self) -> u64 {
        self.counters.executions_succeeded.load(Ordering::Relaxed)
    }

    /// Total executions that failed.
    #[must_use]
    pub fn executions_failed(&self) -> u64 {
        self.counters.executions_failed.load(Ordering::Relaxed)
    }

    /// Takes a point-in-time copy of every counter.
    #[must_use]
    pub fn snapshot(&self) -> RegistryMetricsSnapshot {
        RegistryMetricsSnapshot {
            tools_registered: self.tools_registered(),
            tools_unregistered: self.tools_unregistered(),
            executions_requested: self.executions_requested(),
            executions_succeeded: self.executions_succeeded(),
            executions_failed: self.executions_failed(),
        }
    }
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
                let metrics = actor.model.metrics.snapshot();
                tracing::info!(
                    tools_registered = metrics.tools_registered,
                    executions_requested = metrics.executions_requested,
                    executions_succeeded = metrics.executions_succeeded,
                    executions_failed = metrics.executions_failed,
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
            actor
                .model
                .metrics
                .counters
                .tools_registered
                .fetch_add(1, Ordering::Relaxed);

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
                actor
                    .model
                    .metrics
                    .counters
                    .tools_unregistered
                    .fetch_add(1, Ordering::Relaxed);
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

    // Handle tool execution.
    //
    // Read-only: it looks a tool up and dispatches it, never touching the
    // registry's own state. Under `try_mutate_on` the entire tool execution was
    // awaited inline on the message loop, so the registry could run exactly one
    // tool at a time and no other message — not even `ListTools` — could be
    // served meanwhile.
    builder
        .try_act_on::<ExecuteTool, (), ToolError>(|actor, envelope| {
            if actor.model.shutting_down {
                return Reply::try_err(ToolError::shutting_down());
            }

            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_name = msg.tool_call.name.clone();
            let args = msg.tool_call.arguments.clone();
            let tool_call_id = msg.tool_call.id.clone();
            let metrics = actor.model.metrics.clone();

            metrics
                .counters
                .executions_requested
                .fetch_add(1, Ordering::Relaxed);

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
            let broker = actor.broker().clone();

            // Execute the tool. The ToolRegistry always runs tools inline; the
            // `ToolConfig::sandboxed` flag is advisory metadata and is honored by
            // the facade's `PromptBuilder::use_builtins()` path, which routes
            // sandboxed builtins through a configured `SandboxFactory` before
            // reaching any registry.
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
                        metrics
                            .counters
                            .executions_succeeded
                            .fetch_add(1, Ordering::Relaxed);
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
            actor
                .model
                .metrics
                .counters
                .executions_failed
                .fetch_add(1, Ordering::Relaxed);
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
    fn registry_metrics_start_at_zero() {
        let metrics = RegistryMetrics::default();

        assert_eq!(metrics.snapshot(), RegistryMetricsSnapshot::default());
    }

    #[test]
    fn cloned_metrics_share_counters() {
        let metrics = RegistryMetrics::default();
        let clone = metrics.clone();

        clone
            .counters
            .executions_succeeded
            .fetch_add(2, Ordering::Relaxed);

        assert_eq!(
            metrics.executions_succeeded(),
            2,
            "the execution future records through a clone, so clones must share counters"
        );
    }

    #[test]
    fn snapshot_captures_every_counter() {
        let metrics = RegistryMetrics::default();
        let counters = &metrics.counters;

        counters.tools_registered.fetch_add(1, Ordering::Relaxed);
        counters.tools_unregistered.fetch_add(2, Ordering::Relaxed);
        counters
            .executions_requested
            .fetch_add(3, Ordering::Relaxed);
        counters
            .executions_succeeded
            .fetch_add(4, Ordering::Relaxed);
        counters.executions_failed.fetch_add(5, Ordering::Relaxed);

        assert_eq!(
            metrics.snapshot(),
            RegistryMetricsSnapshot {
                tools_registered: 1,
                tools_unregistered: 2,
                executions_requested: 3,
                executions_succeeded: 4,
                executions_failed: 5,
            }
        );
    }

    #[test]
    fn registered_tool_is_clone() {
        use crate::messages::ToolDefinition;

        // We can't easily test RegisteredTool without a real executor,
        // but we can test the ToolConfig part
        let def = ToolDefinition {
            idempotent: false,
            name: "test".to_string(),
            description: "desc".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let config = ToolConfig::new(def);
        let config2 = config.clone();
        assert_eq!(config.definition.name, config2.definition.name);
    }
}
