//! Agent actor implementation.
//!
//! The Agent actor represents an individual AI agent with its own state,
//! conversation history, and reasoning loop.

use crate::agent::delegation::DelegationTracker;
use crate::agent::{AgentConfig, AgentState};
use crate::llm::StreamAccumulator;
use crate::messages::{
    AgentStatusResponse, GetAgentStatus, GetStatus, IncomingAgentMessage, IncomingTask, LLMRequest,
    LLMResponse, LLMStreamEnd, LLMStreamStart, LLMStreamToken, LLMStreamToolCall, Message,
    StopReason, TaskAccepted, TaskCompleted, TaskFailed, UserPrompt,
};
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
    /// Stream accumulator for receiving streamed responses
    pub stream_accumulator: StreamAccumulator,
    /// Tracker for delegated tasks (both outgoing and incoming)
    pub delegation_tracker: DelegationTracker,
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

        // Build conversation messages including system prompt
        let mut messages = Vec::new();
        if !actor.model.system_prompt.is_empty() {
            messages.push(Message::system(&actor.model.system_prompt));
        }
        messages.extend(actor.model.conversation.clone());

        // Create LLM request
        let llm_request = LLMRequest {
            correlation_id: prompt.correlation_id.clone(),
            agent_id: actor.model.id.clone().unwrap_or_default(),
            messages,
            tools: None, // Tools will be added in Phase 3
        };

        // Broadcast LLM request via broker for LLM Provider to pick up
        let broker = actor.broker().clone();

        Reply::pending(async move {
            broker.broadcast(llm_request).await;
        })
    });

    // Handle LLM stream start
    builder.mutate_on::<LLMStreamStart>(|actor, envelope| {
        let msg = envelope.message();
        let corr_id_str = msg.correlation_id.to_string();

        // Check if this stream is for us
        if !actor.model.pending_llm.contains_key(&corr_id_str) {
            return Reply::ready();
        }

        tracing::debug!(
            agent_id = ?actor.model.id,
            correlation_id = %msg.correlation_id,
            "LLM stream started"
        );

        // Start accumulating the stream
        actor
            .model
            .stream_accumulator
            .start_stream(&msg.correlation_id);

        Reply::ready()
    });

    // Handle LLM stream tokens
    builder.mutate_on::<LLMStreamToken>(|actor, envelope| {
        let msg = envelope.message();
        let corr_id_str = msg.correlation_id.to_string();

        // Check if this token is for us
        if !actor.model.pending_llm.contains_key(&corr_id_str) {
            return Reply::ready();
        }

        // Append token to accumulator
        actor
            .model
            .stream_accumulator
            .append_token(&msg.correlation_id, &msg.token);

        tracing::trace!(
            agent_id = ?actor.model.id,
            correlation_id = %msg.correlation_id,
            token_len = msg.token.len(),
            "Received stream token"
        );

        Reply::ready()
    });

    // Handle LLM stream tool calls with fallible handler pattern
    builder
        .try_mutate_on::<LLMStreamToolCall, (), crate::error::AgentError>(|actor, envelope| {
            let msg = envelope.message();
            let corr_id_str = msg.correlation_id.to_string();

            // Check if this tool call is for us
            if !actor.model.pending_llm.contains_key(&corr_id_str) {
                return Reply::try_ok(());
            }

            // Validate tool call has required fields
            if msg.tool_call.name.is_empty() {
                return Reply::try_err(crate::error::AgentError::processing_failed(
                    actor.model.id.clone(),
                    "Tool call missing name",
                ));
            }

            if msg.tool_call.id.is_empty() {
                return Reply::try_err(crate::error::AgentError::processing_failed(
                    actor.model.id.clone(),
                    "Tool call missing ID",
                ));
            }

            tracing::debug!(
                agent_id = ?actor.model.id,
                correlation_id = %msg.correlation_id,
                tool_name = %msg.tool_call.name,
                "Received tool call"
            );

            // Add tool call to accumulator
            actor
                .model
                .stream_accumulator
                .add_tool_call(&msg.correlation_id, msg.tool_call.clone());

            Reply::try_ok(())
        })
        .on_error::<LLMStreamToolCall, crate::error::AgentError>(|actor, envelope, error| {
            tracing::error!(
                agent_id = ?actor.model.id,
                correlation_id = %envelope.message().correlation_id,
                error = %error,
                "Tool call processing failed"
            );
            Box::pin(async {})
        });

    // Handle LLM stream end with fallible handler pattern
    builder
        .try_mutate_on::<LLMStreamEnd, (), crate::error::AgentError>(|actor, envelope| {
            let msg = envelope.message();
            let corr_id_str = msg.correlation_id.to_string();

            // Check if this stream end is for us
            if !actor.model.pending_llm.contains_key(&corr_id_str) {
                return Reply::try_ok(());
            }

            tracing::debug!(
                agent_id = ?actor.model.id,
                correlation_id = %msg.correlation_id,
                stop_reason = ?msg.stop_reason,
                "LLM stream ended"
            );

            // Complete the stream and get accumulated content
            let stream = actor
                .model
                .stream_accumulator
                .end_stream(&msg.correlation_id, msg.stop_reason);

            let Some(stream) = stream else {
                // Stream was never started or already ended - this is an error
                return Reply::try_err(crate::error::AgentError::processing_failed(
                    actor.model.id.clone(),
                    format!("No active stream for correlation_id {}", msg.correlation_id),
                ));
            };

            // Add assistant response to conversation
            if stream.has_tool_calls() {
                actor.model.add_message(Message::assistant_with_tools(
                    stream.content.clone(),
                    stream.tool_calls.clone(),
                ));
                // Transition to Executing state for tool calls
                actor.model.state = AgentState::Executing;
            } else {
                actor
                    .model
                    .add_message(Message::assistant(stream.content.clone()));
                // Return to Idle state (or Completed based on stop reason)
                actor.model.state = match msg.stop_reason {
                    StopReason::MaxTokens => AgentState::Completed,
                    _ => AgentState::Idle,
                };
            }

            tracing::info!(
                agent_id = ?actor.model.id,
                correlation_id = %msg.correlation_id,
                content_len = stream.content.len(),
                tool_calls = stream.tool_calls.len(),
                "Completed LLM response"
            );

            // Remove from pending
            actor.model.pending_llm.remove(&corr_id_str);

            Reply::try_ok(())
        })
        .on_error::<LLMStreamEnd, crate::error::AgentError>(|actor, envelope, error| {
            let corr_id_str = envelope.message().correlation_id.to_string();

            tracing::error!(
                agent_id = ?actor.model.id,
                correlation_id = %envelope.message().correlation_id,
                error = %error,
                "Stream finalization failed"
            );

            // Clean up the pending request on error
            actor.model.pending_llm.remove(&corr_id_str);

            // Reset state to Idle on error
            actor.model.state = AgentState::Idle;

            Box::pin(async {})
        });

    // Handle complete LLM responses (non-streaming fallback)
    builder.mutate_on::<LLMResponse>(|actor, envelope| {
        let msg = envelope.message();
        let corr_id_str = msg.correlation_id.to_string();

        // Check if this response is for us
        if !actor.model.pending_llm.contains_key(&corr_id_str) {
            return Reply::ready();
        }

        // If we already processed via streaming, skip the complete response
        if actor
            .model
            .stream_accumulator
            .get_stream(&msg.correlation_id)
            .is_some()
        {
            return Reply::ready();
        }

        tracing::debug!(
            agent_id = ?actor.model.id,
            correlation_id = %msg.correlation_id,
            content_len = msg.content.len(),
            "Received complete LLM response"
        );

        // Add assistant response to conversation
        if let Some(tool_calls) = &msg.tool_calls {
            actor.model.add_message(Message::assistant_with_tools(
                msg.content.clone(),
                tool_calls.clone(),
            ));
            actor.model.state = AgentState::Executing;
        } else {
            actor
                .model
                .add_message(Message::assistant(msg.content.clone()));
            actor.model.state = match msg.stop_reason {
                StopReason::MaxTokens => AgentState::Completed,
                _ => AgentState::Idle,
            };
        }

        // Remove from pending
        actor.model.pending_llm.remove(&corr_id_str);

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

    // =========================================================================
    // Multi-Agent Message Handlers (Phase 6)
    // =========================================================================

    // Handle incoming messages from other agents
    builder.mutate_on::<IncomingAgentMessage>(|actor, envelope| {
        let msg = envelope.message();

        tracing::info!(
            agent_id = ?actor.model.id,
            from = %msg.from,
            content_len = msg.content.len(),
            "Received message from another agent"
        );

        // Add to conversation as a user message (from another agent)
        let content = format!("[From Agent {}]: {}", msg.from, msg.content);
        actor.model.add_message(Message::user(content));

        Reply::ready()
    });

    // Handle incoming task delegations
    builder.mutate_on::<IncomingTask>(|actor, envelope| {
        let msg = envelope.message();

        tracing::info!(
            agent_id = ?actor.model.id,
            from = %msg.from,
            task_id = %msg.task_id,
            task_type = %msg.task_type,
            "Received delegated task"
        );

        // Track the incoming task
        actor.model.delegation_tracker.track_incoming(
            msg.task_id.clone(),
            msg.from.clone(),
            msg.task_type.clone(),
        );

        // Auto-accept the task and broadcast TaskAccepted
        actor.model.delegation_tracker.accept_incoming(&msg.task_id);

        let broker = actor.broker().clone();
        let agent_id = actor.model.id.clone().unwrap_or_default();
        let task_id = msg.task_id.clone();

        Reply::pending(async move {
            broker.broadcast(TaskAccepted { task_id, agent_id }).await;
        })
    });

    // Handle task acceptance notifications
    builder.mutate_on::<TaskAccepted>(|actor, envelope| {
        let msg = envelope.message();

        if let Some(task) = actor
            .model
            .delegation_tracker
            .get_outgoing_mut(&msg.task_id)
        {
            task.accept();
            tracing::debug!(
                task_id = %msg.task_id,
                agent_id = %msg.agent_id,
                "Delegated task accepted"
            );
        }

        Reply::ready()
    });

    // Handle task completion notifications
    builder.mutate_on::<TaskCompleted>(|actor, envelope| {
        let msg = envelope.message();

        if let Some(task) = actor
            .model
            .delegation_tracker
            .get_outgoing_mut(&msg.task_id)
        {
            task.complete(msg.result.clone());
            tracing::info!(
                task_id = %msg.task_id,
                "Delegated task completed"
            );
        }

        Reply::ready()
    });

    // Handle task failure notifications
    builder.mutate_on::<TaskFailed>(|actor, envelope| {
        let msg = envelope.message();

        if let Some(task) = actor
            .model
            .delegation_tracker
            .get_outgoing_mut(&msg.task_id)
        {
            task.fail(&msg.error);
            tracing::warn!(
                task_id = %msg.task_id,
                error = %msg.error,
                "Delegated task failed"
            );
        }

        Reply::ready()
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
