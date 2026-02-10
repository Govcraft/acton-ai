//! Core message type definitions.
//!
//! All messages implement Send + Sync + Debug + Clone + 'static as required by acton-reactive.

use crate::agent::AgentConfig;
use crate::llm::SamplingParams;
use crate::types::{AgentId, CorrelationId, TaskId};
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
    /// Optional sampling parameters for this request
    pub sampling: Option<SamplingParams>,
}

impl LLMRequest {
    /// Creates a simple request with just user content.
    ///
    /// IDs are generated internally - users don't need to manage them.
    /// This is the simplest way to create an LLM request.
    ///
    /// # Example
    ///
    /// ```
    /// use acton_ai::messages::LLMRequest;
    ///
    /// let request = LLMRequest::simple("What is the capital of France?");
    /// assert!(!request.messages.is_empty());
    /// ```
    #[must_use]
    pub fn simple(content: impl Into<String>) -> Self {
        Self {
            correlation_id: CorrelationId::new(),
            agent_id: AgentId::new(),
            messages: vec![Message::user(content)],
            tools: None,
            sampling: None,
        }
    }

    /// Creates a request with a system prompt and user content.
    ///
    /// IDs are generated internally.
    ///
    /// # Example
    ///
    /// ```
    /// use acton_ai::messages::LLMRequest;
    ///
    /// let request = LLMRequest::with_system(
    ///     "You are a helpful assistant.",
    ///     "What is 2 + 2?"
    /// );
    /// assert_eq!(request.messages.len(), 2);
    /// ```
    #[must_use]
    pub fn with_system(system: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            correlation_id: CorrelationId::new(),
            agent_id: AgentId::new(),
            messages: vec![Message::system(system), Message::user(content)],
            tools: None,
            sampling: None,
        }
    }

    /// Creates a builder for advanced request configuration.
    ///
    /// Use the builder when you need to:
    /// - Set explicit correlation or agent IDs (for tracking/persistence)
    /// - Add multiple messages
    /// - Include tool definitions
    ///
    /// # Example
    ///
    /// ```
    /// use acton_ai::messages::LLMRequest;
    ///
    /// let request = LLMRequest::builder()
    ///     .system("You are a helpful assistant.")
    ///     .user("Hello!")
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder() -> LLMRequestBuilder {
        LLMRequestBuilder::default()
    }
}

/// Builder for constructing LLM requests with advanced options.
///
/// Use `LLMRequest::builder()` to create an instance.
///
/// # Example
///
/// ```
/// use acton_ai::messages::LLMRequest;
/// use acton_ai::types::CorrelationId;
///
/// let request = LLMRequest::builder()
///     .correlation_id(CorrelationId::new())
///     .system("You are an expert.")
///     .user("Explain Rust ownership.")
///     .build();
/// ```
#[derive(Default)]
pub struct LLMRequestBuilder {
    correlation_id: Option<CorrelationId>,
    agent_id: Option<AgentId>,
    messages: Vec<Message>,
    tools: Option<Vec<ToolDefinition>>,
    sampling: Option<SamplingParams>,
}

impl LLMRequestBuilder {
    /// Sets an explicit correlation ID.
    ///
    /// Use this when you need to track requests across systems
    /// or match requests to responses manually.
    #[must_use]
    pub fn correlation_id(mut self, id: CorrelationId) -> Self {
        self.correlation_id = Some(id);
        self
    }

    /// Sets an explicit agent ID.
    ///
    /// Use this in multi-agent scenarios where you need to
    /// identify which agent made the request.
    #[must_use]
    pub fn agent_id(mut self, id: AgentId) -> Self {
        self.agent_id = Some(id);
        self
    }

    /// Adds a system message.
    #[must_use]
    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::system(content));
        self
    }

    /// Adds a user message.
    #[must_use]
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::user(content));
        self
    }

    /// Adds an assistant message.
    #[must_use]
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::assistant(content));
        self
    }

    /// Adds a custom message.
    #[must_use]
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Adds multiple messages.
    #[must_use]
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Sets the tool definitions available to the LLM.
    #[must_use]
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Adds a single tool definition.
    #[must_use]
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.get_or_insert_with(Vec::new).push(tool);
        self
    }

    /// Sets the sampling parameters for this request.
    #[must_use]
    pub fn sampling(mut self, params: SamplingParams) -> Self {
        self.sampling = Some(params);
        self
    }

    /// Builds the LLM request.
    ///
    /// IDs are auto-generated if not explicitly set.
    #[must_use]
    pub fn build(self) -> LLMRequest {
        LLMRequest {
            correlation_id: self.correlation_id.unwrap_or_default(),
            agent_id: self.agent_id.unwrap_or_default(),
            messages: self.messages,
            tools: self.tools,
            sampling: self.sampling,
        }
    }
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

// =============================================================================
// Multi-Agent Messages (Phase 6)
// =============================================================================

/// Direct message between agents.
///
/// Agents can send messages directly to other agents via the kernel's routing.
/// The kernel forwards messages to the target agent.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct AgentMessage {
    /// The agent sending the message
    pub from: AgentId,
    /// The target agent to receive the message
    pub to: AgentId,
    /// The message content
    pub content: String,
    /// Optional metadata (JSON value for extensibility)
    pub metadata: Option<serde_json::Value>,
}

impl AgentMessage {
    /// Creates a new agent message.
    #[must_use]
    pub fn new(from: AgentId, to: AgentId, content: impl Into<String>) -> Self {
        Self {
            from,
            to,
            content: content.into(),
            metadata: None,
        }
    }

    /// Adds metadata to the message.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Request to delegate a task to another agent.
///
/// The delegating agent creates this message to assign work to a specialist agent.
/// The task_id serves as a correlation ID for tracking the result.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct DelegateTask {
    /// The agent delegating the task
    pub from: AgentId,
    /// The agent to perform the task
    pub to: AgentId,
    /// Unique identifier for this task
    pub task_id: TaskId,
    /// The type of task (e.g., "code_review", "summarize", "translate")
    pub task_type: String,
    /// Task payload as JSON
    pub payload: serde_json::Value,
    /// Optional deadline for the task
    pub deadline: Option<std::time::Duration>,
}

impl DelegateTask {
    /// Creates a new task delegation.
    #[must_use]
    pub fn new(
        from: AgentId,
        to: AgentId,
        task_type: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            from,
            to,
            task_id: TaskId::new(),
            task_type: task_type.into(),
            payload,
            deadline: None,
        }
    }

    /// Sets a deadline for task completion.
    #[must_use]
    pub fn with_deadline(mut self, deadline: std::time::Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

/// Acknowledgment that a task was accepted.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct TaskAccepted {
    /// The task that was accepted
    pub task_id: TaskId,
    /// The agent that accepted the task
    pub agent_id: AgentId,
}

/// Notification that a delegated task completed successfully.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct TaskCompleted {
    /// The task that completed
    pub task_id: TaskId,
    /// The result of the task as JSON
    pub result: serde_json::Value,
}

/// Notification that a delegated task failed.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct TaskFailed {
    /// The task that failed
    pub task_id: TaskId,
    /// The error message
    pub error: String,
}

/// Announcement of agent capabilities for discovery.
///
/// Agents broadcast this message to announce what they can do.
/// Other agents or the kernel can track these capabilities.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct AnnounceCapabilities {
    /// The agent announcing its capabilities
    pub agent_id: AgentId,
    /// List of capability strings (e.g., "code_review", "translation", "summarization")
    pub capabilities: Vec<String>,
}

impl AnnounceCapabilities {
    /// Creates a new capability announcement.
    #[must_use]
    pub fn new(agent_id: AgentId, capabilities: Vec<String>) -> Self {
        Self {
            agent_id,
            capabilities,
        }
    }
}

/// Request to find an agent with a specific capability.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct FindCapableAgent {
    /// The capability to search for
    pub capability: String,
    /// Correlation ID for the response
    pub correlation_id: CorrelationId,
}

impl FindCapableAgent {
    /// Creates a new capability search request.
    #[must_use]
    pub fn new(capability: impl Into<String>) -> Self {
        Self {
            capability: capability.into(),
            correlation_id: CorrelationId::new(),
        }
    }
}

/// Response to a capability search request.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct CapableAgentFound {
    /// The correlation ID from the request
    pub correlation_id: CorrelationId,
    /// The agent with the capability, if found
    pub agent_id: Option<AgentId>,
    /// The capability that was searched for
    pub capability: String,
}

/// Incoming message from another agent (delivered to agent).
///
/// This is what an agent receives when another agent sends it a message.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct IncomingAgentMessage {
    /// The agent that sent the message
    pub from: AgentId,
    /// The message content
    pub content: String,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

impl From<AgentMessage> for IncomingAgentMessage {
    fn from(msg: AgentMessage) -> Self {
        Self {
            from: msg.from,
            content: msg.content,
            metadata: msg.metadata,
        }
    }
}

/// Incoming task delegation (delivered to agent).
///
/// This is what an agent receives when another agent delegates a task to it.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct IncomingTask {
    /// The agent that delegated the task
    pub from: AgentId,
    /// The task identifier
    pub task_id: TaskId,
    /// The type of task
    pub task_type: String,
    /// The task payload
    pub payload: serde_json::Value,
    /// Optional deadline
    pub deadline: Option<std::time::Duration>,
}

impl IncomingTask {
    /// Creates an IncomingTask from a DelegateTask message.
    #[must_use]
    pub fn from_delegate(msg: &DelegateTask) -> Self {
        Self {
            from: msg.from.clone(),
            task_id: msg.task_id.clone(),
            task_type: msg.task_type.clone(),
            payload: msg.payload.clone(),
            deadline: msg.deadline,
        }
    }
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
        let event = SystemEvent::AgentSpawned {
            id: agent_id.clone(),
        };

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

    // LLMRequest convenience method tests
    #[test]
    fn llm_request_simple_creates_user_message() {
        let request = LLMRequest::simple("Hello");

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, MessageRole::User);
        assert_eq!(request.messages[0].content, "Hello");
        assert!(request.tools.is_none());
    }

    #[test]
    fn llm_request_simple_generates_ids() {
        let request1 = LLMRequest::simple("Hello");
        let request2 = LLMRequest::simple("World");

        assert_ne!(request1.correlation_id, request2.correlation_id);
        assert_ne!(request1.agent_id, request2.agent_id);
    }

    #[test]
    fn llm_request_with_system_creates_two_messages() {
        let request = LLMRequest::with_system("Be helpful", "Hello");

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, MessageRole::System);
        assert_eq!(request.messages[0].content, "Be helpful");
        assert_eq!(request.messages[1].role, MessageRole::User);
        assert_eq!(request.messages[1].content, "Hello");
    }

    #[test]
    fn llm_request_builder_basic() {
        let request = LLMRequest::builder().user("Hello").build();

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].content, "Hello");
    }

    #[test]
    fn llm_request_builder_with_system_and_user() {
        let request = LLMRequest::builder()
            .system("Be concise")
            .user("What is 2+2?")
            .build();

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, MessageRole::System);
        assert_eq!(request.messages[1].role, MessageRole::User);
    }

    #[test]
    fn llm_request_builder_with_explicit_ids() {
        let corr_id = CorrelationId::new();
        let agent_id = AgentId::new();

        let request = LLMRequest::builder()
            .correlation_id(corr_id.clone())
            .agent_id(agent_id.clone())
            .user("Hello")
            .build();

        assert_eq!(request.correlation_id, corr_id);
        assert_eq!(request.agent_id, agent_id);
    }

    #[test]
    fn llm_request_builder_with_tools() {
        let tool = ToolDefinition {
            name: "calculator".to_string(),
            description: "Math".to_string(),
            input_schema: serde_json::json!({}),
        };

        let request = LLMRequest::builder()
            .user("Calculate 2+2")
            .tool(tool.clone())
            .build();

        assert!(request.tools.is_some());
        assert_eq!(request.tools.as_ref().unwrap().len(), 1);
        assert_eq!(request.tools.as_ref().unwrap()[0].name, "calculator");
    }

    #[test]
    fn llm_request_builder_with_multiple_tools() {
        let tools = vec![
            ToolDefinition {
                name: "calc".to_string(),
                description: "Math".to_string(),
                input_schema: serde_json::json!({}),
            },
            ToolDefinition {
                name: "search".to_string(),
                description: "Search".to_string(),
                input_schema: serde_json::json!({}),
            },
        ];

        let request = LLMRequest::builder().user("Hello").tools(tools).build();

        assert!(request.tools.is_some());
        assert_eq!(request.tools.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn llm_request_builder_with_assistant() {
        let request = LLMRequest::builder()
            .user("Hello")
            .assistant("Hi there!")
            .user("How are you?")
            .build();

        assert_eq!(request.messages.len(), 3);
        assert_eq!(request.messages[1].role, MessageRole::Assistant);
    }

    #[test]
    fn llm_request_builder_with_custom_message() {
        let custom_msg = Message::tool("tc_123", "Result: 4");

        let request = LLMRequest::builder()
            .user("Calculate 2+2")
            .message(custom_msg)
            .build();

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[1].role, MessageRole::Tool);
    }

    #[test]
    fn llm_request_builder_generates_ids_when_not_set() {
        let request1 = LLMRequest::builder().user("Hello").build();
        let request2 = LLMRequest::builder().user("World").build();

        assert_ne!(request1.correlation_id, request2.correlation_id);
        assert_ne!(request1.agent_id, request2.agent_id);
    }
}
