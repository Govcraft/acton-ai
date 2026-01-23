//! Kernel actor implementation.
//!
//! The Kernel is the central coordinator and supervisor of the entire
//! Acton-AI system. It manages agent lifecycles, routes inter-agent
//! communication, and handles agent failures through supervision.

use crate::kernel::KernelConfig;
use crate::messages::{
    AgentSpawned, GetAgentStatus, RouteMessage, SpawnAgent, StopAgent, SystemEvent,
};
use acton_reactive::prelude::*;
use std::collections::HashMap;

/// Metrics collected by the Kernel.
#[derive(Debug, Clone, Default)]
pub struct KernelMetrics {
    /// Total number of agents spawned
    pub agents_spawned: usize,
    /// Total number of agents stopped
    pub agents_stopped: usize,
    /// Total number of messages routed
    pub messages_routed: usize,
}

/// Message to initialize the kernel with configuration.
#[acton_message]
pub struct InitKernel {
    /// Kernel configuration
    pub config: KernelConfig,
}

/// The Kernel actor state.
///
/// The Kernel maintains a registry of all active agents and supervises
/// their lifecycles using the OneForOne supervision strategy.
#[acton_actor]
pub struct Kernel {
    /// Configuration for the kernel
    pub config: KernelConfig,
    /// Registry of active agents (AgentId -> ActorHandle)
    pub agents: HashMap<String, ActorHandle>,
    /// Metrics for monitoring
    pub metrics: KernelMetrics,
    /// Whether the kernel is shutting down
    pub shutting_down: bool,
}

impl Kernel {
    /// Spawns the Kernel actor with default configuration.
    ///
    /// This creates and starts the Kernel actor, which will supervise
    /// all agent actors in the system.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The ActorRuntime
    ///
    /// # Returns
    ///
    /// The ActorHandle for the started Kernel actor.
    pub async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        Self::spawn_with_config(runtime, KernelConfig::default()).await
    }

    /// Spawns the Kernel actor with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The ActorRuntime
    /// * `config` - Configuration for the kernel
    ///
    /// # Returns
    ///
    /// The ActorHandle for the started Kernel actor.
    pub async fn spawn_with_config(runtime: &mut ActorRuntime, config: KernelConfig) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<Kernel>("kernel".to_string());

        // Store config for use in init message
        let kernel_config = config.clone();

        // Set up lifecycle hooks (immutable, just for logging)
        builder
            .before_start(|_actor| {
                tracing::debug!("Kernel initializing");
                Reply::ready()
            })
            .after_start(|actor| {
                tracing::info!(
                    max_agents = actor.model.config.max_agents,
                    "Kernel ready to accept agent spawn requests"
                );
                Reply::ready()
            })
            .before_stop(|actor| {
                tracing::info!(
                    active_agents = actor.model.agents.len(),
                    total_spawned = actor.model.metrics.agents_spawned,
                    "Kernel shutting down"
                );
                Reply::ready()
            });

        // Configure message handlers
        configure_handlers(&mut builder);

        let handle = builder.start().await;

        // Initialize kernel with config
        handle.send(InitKernel { config: kernel_config }).await;

        handle
    }
}

/// Configures message handlers for the Kernel actor.
fn configure_handlers(builder: &mut ManagedActor<Idle, Kernel>) {
    // Handle kernel initialization
    builder.mutate_on::<InitKernel>(|actor, envelope| {
        actor.model.config = envelope.message().config.clone();
        actor.model.shutting_down = false;

        tracing::info!(
            max_agents = actor.model.config.max_agents,
            metrics_enabled = actor.model.config.enable_metrics,
            "Kernel configured"
        );

        Reply::ready()
    });

    // Handle SpawnAgent requests
    builder.mutate_on::<SpawnAgent>(|actor, envelope| {
        let config = envelope.message().config.clone();
        let reply = envelope.reply_envelope();

        // Check if we're shutting down
        if actor.model.shutting_down {
            tracing::warn!("Rejecting spawn request - kernel is shutting down");
            return Reply::ready();
        }

        // Check agent limit
        if actor.model.agents.len() >= actor.model.config.max_agents {
            tracing::warn!(
                current = actor.model.agents.len(),
                max = actor.model.config.max_agents,
                "Rejecting spawn request - agent limit reached"
            );
            return Reply::ready();
        }

        let agent_id = config.agent_id();
        tracing::info!(
            agent_id = %agent_id,
            name = ?config.name,
            "Spawning new agent"
        );

        // Store the agent ID for later use
        let spawned_id = agent_id.clone();

        // Update metrics
        actor.model.metrics.agents_spawned += 1;

        // Broadcast agent spawned event
        let broker = actor.broker().clone();

        Reply::pending(async move {
            broker
                .broadcast(SystemEvent::AgentSpawned {
                    id: spawned_id.clone(),
                })
                .await;

            reply.send(AgentSpawned { agent_id: spawned_id }).await;
        })
    });

    // Handle StopAgent requests
    builder.mutate_on::<StopAgent>(|actor, envelope| {
        let agent_id = &envelope.message().agent_id;
        let agent_id_str = agent_id.to_string();

        if let Some(handle) = actor.model.agents.remove(&agent_id_str) {
            tracing::info!(agent_id = %agent_id, "Stopping agent");

            actor.model.metrics.agents_stopped += 1;

            // Broadcast agent stopped event
            let broker = actor.broker().clone();
            let stopped_id = agent_id.clone();

            Reply::pending(async move {
                broker
                    .broadcast(SystemEvent::AgentStopped {
                        id: stopped_id,
                        reason: "requested".to_string(),
                    })
                    .await;

                // Send stop signal to the agent
                let _ = handle.stop().await;
            })
        } else {
            tracing::warn!(agent_id = %agent_id, "Agent not found for stop request");
            Reply::ready()
        }
    });

    // Handle RouteMessage - forward messages between agents
    builder.mutate_on::<RouteMessage>(|actor, envelope| {
        let msg = envelope.message();
        let to_str = msg.to.to_string();

        if let Some(target_handle) = actor.model.agents.get(&to_str) {
            tracing::debug!(
                from = %msg.from,
                to = %msg.to,
                payload_length = msg.payload.len(),
                "Routing message between agents"
            );

            actor.model.metrics.messages_routed += 1;

            let handle = target_handle.clone();
            let payload = msg.payload.clone();
            let from = msg.from.clone();

            Reply::pending(async move {
                // In Phase 1, we just log the routing
                // In later phases, this will forward to the target agent
                tracing::debug!(
                    from = %from,
                    payload = %payload,
                    "Message routed (Phase 1 stub)"
                );
                drop(handle);
            })
        } else {
            tracing::warn!(
                from = %msg.from,
                to = %msg.to,
                "Cannot route message - target agent not found"
            );
            Reply::ready()
        }
    });

    // Handle GetAgentStatus requests
    builder.act_on::<GetAgentStatus>(|actor, envelope| {
        let agent_id = &envelope.message().agent_id;
        let agent_id_str = agent_id.to_string();

        if let Some(agent_handle) = actor.model.agents.get(&agent_id_str) {
            let handle = agent_handle.clone();
            let id = agent_id.clone();

            Reply::pending(async move {
                // Forward the status request to the agent
                handle.send(GetAgentStatus { agent_id: id }).await;
            })
        } else {
            tracing::warn!(agent_id = %agent_id, "Agent not found for status request");
            Reply::ready()
        }
    });

    // Handle ChildTerminated events for supervised agents
    builder.mutate_on::<ChildTerminated>(|_actor, envelope| {
        let msg = envelope.message();
        tracing::info!(
            child_id = ?msg.child_id,
            reason = ?msg.reason,
            "Child actor terminated"
        );

        // Find and remove the terminated agent
        // Note: In a full implementation, we'd match on the Ern
        // For now, this is a placeholder for supervision handling

        Reply::ready()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_metrics_default() {
        let metrics = KernelMetrics::default();
        assert_eq!(metrics.agents_spawned, 0);
        assert_eq!(metrics.agents_stopped, 0);
        assert_eq!(metrics.messages_routed, 0);
    }
}
