//! Agent actor implementation.
//!
//! The Agent actor represents an individual AI agent with its own state,
//! conversation history, and reasoning loop.

use crate::agent::{AgentConfig, AgentState};
use crate::messages::{AgentStatusResponse, GetAgentStatus, GetStatus, Message, UserPrompt};
use crate::types::{AgentId, CorrelationId};
use acton_reactive::prelude::*;
use std::collections::HashMap;

/// Internal state for a pending LLM request.
#[derive(Debug, Clone, Default)]
pub struct PendingLLMRequest {
    /// The correlation ID for this request
    pub correlation_id: Option<CorrelationId>,
    /// The original user prompt that triggered this request
    pub original_prompt: String,
}

/// Message to initialize agent with configuration.
#[acton_message]
pub struct InitAgent {
    /// The agent configuration
    pub config: AgentConfig,
}

/// The Agent actor state.
///
/// Each agent maintains its own conversation history, state, and pending requests.
#[acton_actor]
pub struct Agent {
    /// Unique identifier for this agent
    pub id: Option<AgentId>,
    /// The system prompt defining agent behavior
    pub system_prompt: String,
    /// Optional display name
    pub name: Option<String>,
    /// Conversation history
    pub conversation: Vec<Message>,
    /// Current state in the reasoning loop
    pub state: AgentState,
    /// Maximum conversation history length
    pub max_conversation_length: usize,
    /// Whether streaming is enabled
    pub enable_streaming: bool,
    /// Pending LLM requests awaiting response
    pub pending_llm: HashMap<String, PendingLLMRequest>,
    /// Pending tool calls awaiting response
    pub pending_tools: HashMap<String, String>,
}

impl Agent {
    /// Creates a new Agent actor builder with the given runtime.
    ///
    /// This sets up the actor state and configures all message handlers.
    /// The actor is not started until `start()` is called on the returned builder.
    /// After starting, send an `InitAgent` message to configure the agent.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The Acton runtime to create the actor in
    ///
    /// # Returns
    ///
    /// A `ManagedActor<Idle, Agent>` that can be further configured and started.
    pub fn create(runtime: &mut ActorRuntime) -> ManagedActor<Idle, Agent> {
        let mut builder = runtime.new_actor::<Agent>();

        // Set up lifecycle hooks (immutable, just for logging)
        builder
            .before_start(|_actor| {
                tracing::debug!("Agent initializing");
                Reply::ready()
            })
            .after_start(|actor| {
                tracing::info!(
                    agent_id = ?actor.model.id,
                    "Agent ready for messages"
                );
                Reply::ready()
            })
            .before_stop(|actor| {
                tracing::info!(
                    agent_id = ?actor.model.id,
                    conversation_length = actor.model.conversation.len(),
                    "Agent stopping"
                );
                Reply::ready()
            });

        // Configure message handlers
        configure_handlers(&mut builder);

        builder
    }

    /// Adds a message to the conversation history, respecting max length.
    pub fn add_message(&mut self, message: Message) {
        self.conversation.push(message);

        // Trim conversation if it exceeds max length
        // Keep system messages and recent messages
        if self.conversation.len() > self.max_conversation_length {
            let excess = self.conversation.len() - self.max_conversation_length;
            // Remove from the beginning, but keep index 0 if it's a system message
            let start_index = if self
                .conversation
                .first()
                .map(|m| m.role == crate::messages::MessageRole::System)
                .unwrap_or(false)
            {
                1
            } else {
                0
            };
            self.conversation.drain(start_index..start_index + excess);
        }
    }

    /// Clears the conversation history.
    pub fn clear_conversation(&mut self) {
        self.conversation.clear();
    }

    /// Returns the current conversation length.
    #[must_use]
    pub fn conversation_length(&self) -> usize {
        self.conversation.len()
    }
}

/// Configures message handlers for the Agent actor.
fn configure_handlers(builder: &mut ManagedActor<Idle, Agent>) {
    // Handle initialization message
    builder.mutate_on::<InitAgent>(|actor, envelope| {
        let config = &envelope.message().config;

        actor.model.id = Some(config.agent_id());
        actor.model.system_prompt = config.system_prompt.clone();
        actor.model.name = config.name.clone();
        actor.model.max_conversation_length = config.max_conversation_length;
        actor.model.enable_streaming = config.enable_streaming;
        actor.model.state = AgentState::Idle;

        tracing::info!(
            agent_id = ?actor.model.id,
            name = ?actor.model.name,
            "Agent configured"
        );

        Reply::ready()
    });

    // Handle user prompts - starts the reasoning loop
    builder.mutate_on::<UserPrompt>(|actor, envelope| {
        let prompt = envelope.message();

        // Check if we can accept a new prompt
        if !actor.model.state.can_accept_prompt() {
            tracing::warn!(
                agent_id = ?actor.model.id,
                current_state = %actor.model.state,
                "Rejecting prompt - agent is busy"
            );
            return Reply::ready();
        }

        tracing::info!(
            agent_id = ?actor.model.id,
            correlation_id = %prompt.correlation_id,
            content_length = prompt.content.len(),
            "Received user prompt"
        );

        // Transition to Thinking state
        actor.model.state = AgentState::Thinking;

        // Add user message to conversation
        actor.model.add_message(Message::user(&prompt.content));

        // Store pending request
        let corr_id_str = prompt.correlation_id.to_string();
        actor.model.pending_llm.insert(
            corr_id_str.clone(),
            PendingLLMRequest {
                correlation_id: Some(prompt.correlation_id.clone()),
                original_prompt: prompt.content.clone(),
            },
        );

        // In Phase 1, we just simulate a response
        // In Phase 2, this will send an LLMRequest via the broker
        let agent_id_str = actor
            .model
            .id
            .as_ref()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // For now, simulate completing the reasoning loop
        // TODO: Replace with actual LLM integration in Phase 2
        actor.model.state = AgentState::Completed;
        actor.model.pending_llm.remove(&corr_id_str);

        // Add a placeholder assistant response
        actor.model.add_message(Message::assistant(format!(
            "Hello! I received your message. (Agent {} - Phase 1 stub)",
            agent_id_str
        )));

        tracing::debug!(
            agent_id = agent_id_str,
            "Completed processing prompt (stub)"
        );

        Reply::ready()
    });

    // Handle status requests (read-only)
    builder.act_on::<GetStatus>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let agent_id = actor.model.id.clone().unwrap_or_else(AgentId::new);
        let state = actor.model.state.to_string();
        let conversation_length = actor.model.conversation_length();

        Reply::pending(async move {
            reply
                .send(AgentStatusResponse {
                    agent_id,
                    state,
                    conversation_length,
                })
                .await;
        })
    });

    // Handle status requests from kernel
    builder.act_on::<GetAgentStatus>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let agent_id = actor.model.id.clone().unwrap_or_else(AgentId::new);
        let state = actor.model.state.to_string();
        let conversation_length = actor.model.conversation_length();

        Reply::pending(async move {
            reply
                .send(AgentStatusResponse {
                    agent_id,
                    state,
                    conversation_length,
                })
                .await;
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_message_respects_max_length() {
        let mut agent = Agent::default();
        agent.max_conversation_length = 5;

        for i in 0..10 {
            agent.add_message(Message::user(format!("Message {}", i)));
        }

        assert_eq!(agent.conversation.len(), 5);
    }

    #[test]
    fn add_message_preserves_system_message() {
        let mut agent = Agent::default();
        agent.max_conversation_length = 3;

        // Add system message first
        agent.add_message(Message::system("You are helpful"));

        // Add more messages than max
        for i in 0..5 {
            agent.add_message(Message::user(format!("Message {}", i)));
        }

        // System message should still be at index 0
        assert_eq!(agent.conversation.len(), 3);
        assert_eq!(
            agent.conversation[0].role,
            crate::messages::MessageRole::System
        );
    }

    #[test]
    fn clear_conversation_empties_history() {
        let mut agent = Agent::default();
        agent.max_conversation_length = 10;
        agent.add_message(Message::user("Hello"));
        agent.add_message(Message::assistant("Hi"));

        assert_eq!(agent.conversation_length(), 2);

        agent.clear_conversation();

        assert_eq!(agent.conversation_length(), 0);
    }
}
