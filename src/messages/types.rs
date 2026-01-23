//! Core message type definitions.
//!
//! All messages implement Send + Sync + Debug + Clone + 'static as required by acton-reactive.

use crate::agent::AgentConfig;
use crate::types::{AgentId, CorrelationId};
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// Kernel Messages
// =============================================================================

/// Request to spawn a new agent.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct SpawnAgent {
    /// Configuration for the new agent
    pub config: AgentConfig,
}

/// Response after spawning an agent.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct AgentSpawned {
    /// The ID of the newly spawned agent
    pub agent_id: AgentId,
}

/// Request to stop an agent.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct StopAgent {
    /// The ID of the agent to stop
    pub agent_id: AgentId,
}

/// Request for agent status.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct GetAgentStatus {
    /// The ID of the agent to query
    pub agent_id: AgentId,
}

/// Response with agent status.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct AgentStatusResponse {
    /// The ID of the agent
    pub agent_id: AgentId,
    /// The current state of the agent
    pub state: String,
    /// Number of messages in conversation
    pub conversation_length: usize,
}

/// Message routed between agents.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct RouteMessage {
    /// The agent sending the message
    pub from: AgentId,
    /// The agent receiving the message
    pub to: AgentId,
    /// The message content
    pub payload: String,
}

// =============================================================================
// Agent Messages
// =============================================================================

/// User prompt sent to an agent.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct UserPrompt {
    /// Unique identifier for this request-response cycle
    pub correlation_id: CorrelationId,
    /// The user's message content
    pub content: String,
}

impl UserPrompt {
    /// Creates a new UserPrompt with a fresh correlation ID.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            correlation_id: CorrelationId::new(),
            content: content.into(),
        }
    }
}

/// Request the current status of an agent (read-only).
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct GetStatus;

/// Internal message to update agent state.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct UpdateState {
    /// The new state string
    pub new_state: String,
}

// =============================================================================
// Conversation Messages
// =============================================================================

/// A message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender
    pub role: MessageRole,
    /// The content of the message
    pub content: String,
    /// Optional tool calls in this message
    pub tool_calls: Option<Vec<ToolCall>>,
    /// ID of the tool call this message responds to
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Creates a new user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a new assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a new assistant message with tool calls.
    #[must_use]
    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// Creates a new tool response message.
    #[must_use]
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Creates a new system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// The role of a message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System instructions
    System,
    /// User input
    User,
    /// Assistant response
    Assistant,
    /// Tool response
    Tool,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

// =============================================================================
// LLM Messages
// =============================================================================

/// Request to the LLM provider.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct LLMRequest {
    /// Correlation ID for matching request to response
    pub correlation_id: CorrelationId,
    /// The agent making the request
    pub agent_id: AgentId,
    /// The messages to send to the LLM
    pub messages: Vec<Message>,
    /// Optional tool definitions available to the LLM
    pub tools: Option<Vec<ToolDefinition>>,
}

/// Complete response from the LLM (non-streaming).
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct LLMResponse {
    /// Correlation ID matching the request
    pub correlation_id: CorrelationId,
    /// The generated content
    pub content: String,
    /// Tool calls requested by the LLM
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The reason the model stopped generating
    pub stop_reason: StopReason,
}

/// The reason the LLM stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Normal completion
    EndTurn,
    /// Reached maximum tokens
    MaxTokens,
    /// Model wants to call tools
    ToolUse,
    /// User-initiated stop
    StopSequence,
}

// =============================================================================
// Streaming LLM Messages
// =============================================================================

/// Indicates the start of a streaming LLM response.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct LLMStreamStart {
    /// Correlation ID for this stream
    pub correlation_id: CorrelationId,
}

/// A single token in a streaming response.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct LLMStreamToken {
    /// Correlation ID for this stream
    pub correlation_id: CorrelationId,
    /// The token text
    pub token: String,
}

/// A tool call in a streaming response.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct LLMStreamToolCall {
    /// Correlation ID for this stream
    pub correlation_id: CorrelationId,
    /// The tool call
    pub tool_call: ToolCall,
}

/// Indicates the end of a streaming LLM response.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct LLMStreamEnd {
    /// Correlation ID for this stream
    pub correlation_id: CorrelationId,
    /// The reason the stream ended
    pub stop_reason: StopReason,
}

// =============================================================================
// Tool Messages
// =============================================================================

/// Definition of a tool that can be called by the LLM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON Schema for the tool's input parameters
    pub input_schema: serde_json::Value,
}

/// A tool call requested by the LLM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// The name of the tool to call
    pub name: String,
    /// The arguments to pass to the tool (as JSON)
    pub arguments: serde_json::Value,
}

/// Request to execute a tool.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct ExecuteTool {
    /// Correlation ID for matching to response
    pub correlation_id: CorrelationId,
    /// The tool call to execute
    pub tool_call: ToolCall,
    /// The agent requesting the tool execution
    pub requesting_agent: AgentId,
}

/// Response from tool execution.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct ToolResponse {
    /// Correlation ID matching the request
    pub correlation_id: CorrelationId,
    /// The ID of the tool call this responds to
    pub tool_call_id: String,
    /// The result of the tool execution (success content or error message)
    pub result: Result<String, String>,
}

// =============================================================================
// System Events (Pub/Sub)
// =============================================================================

/// System-wide events broadcast via the broker.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub enum SystemEvent {
    /// An agent was spawned
    AgentSpawned {
        /// The ID of the spawned agent
        id: AgentId,
    },
    /// An agent stopped
    AgentStopped {
        /// The ID of the stopped agent
        id: AgentId,
        /// The reason for stopping
        reason: String,
    },
    /// A tool was registered
    ToolRegistered {
        /// The name of the registered tool
        name: String,
    },
    /// Rate limit was hit
    RateLimitHit {
        /// The provider that hit the limit
        provider: String,
        /// Seconds until retry is allowed
        retry_after_secs: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_prompt_creates_correlation_id() {
        let prompt1 = UserPrompt::new("Hello");
        let prompt2 = UserPrompt::new("World");

        assert_ne!(prompt1.correlation_id, prompt2.correlation_id);
    }

    #[test]
    fn message_user_creation() {
        let msg = Message::user("Hello, agent!");

        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "Hello, agent!");
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
    }

    #[test]
    fn message_assistant_creation() {
        let msg = Message::assistant("I can help with that.");

        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.content, "I can help with that.");
    }

    #[test]
    fn message_assistant_with_tools() {
        let tool_calls = vec![ToolCall {
            id: "tc_123".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "Rust actors"}),
        }];

        let msg = Message::assistant_with_tools("Let me search for that.", tool_calls);

        assert_eq!(msg.role, MessageRole::Assistant);
        assert!(msg.tool_calls.is_some());
        assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn message_tool_response() {
        let msg = Message::tool("tc_123", "Search results: ...");

        assert_eq!(msg.role, MessageRole::Tool);
        assert_eq!(msg.tool_call_id, Some("tc_123".to_string()));
    }

    #[test]
    fn message_role_display() {
        assert_eq!(MessageRole::System.to_string(), "system");
        assert_eq!(MessageRole::User.to_string(), "user");
        assert_eq!(MessageRole::Assistant.to_string(), "assistant");
        assert_eq!(MessageRole::Tool.to_string(), "tool");
    }

    #[test]
    fn tool_definition_serialization() {
        let tool = ToolDefinition {
            name: "calculator".to_string(),
            description: "Performs basic arithmetic".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }),
        };

        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&json).unwrap();

        assert_eq!(tool, deserialized);
    }

    #[test]
    fn system_event_agent_spawned() {
        let agent_id = AgentId::new();
        let event = SystemEvent::AgentSpawned { id: agent_id.clone() };

        if let SystemEvent::AgentSpawned { id } = event {
            assert_eq!(id, agent_id);
        } else {
            panic!("Expected AgentSpawned event");
        }
    }

    #[test]
    fn stop_reason_serialization() {
        let reasons = vec![
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::ToolUse,
            StopReason::StopSequence,
        ];

        for reason in reasons {
            let json = serde_json::to_string(&reason).unwrap();
            let deserialized: StopReason = serde_json::from_str(&json).unwrap();
            assert_eq!(reason, deserialized);
        }
    }
}
