//! Core message type definitions.
//!
//! All messages implement Send + Sync + Debug + Clone + 'static as required by acton-reactive.

use crate::llm::SamplingParams;
use crate::types::{AgentId, CorrelationId, TaskId, TurnId};
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// Agent Lifecycle Messages
// =============================================================================

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

/// How the model should choose among the offered tools.
///
/// This is a closed protocol concept shared by the Anthropic Messages API and
/// the OpenAI chat-completions API, so it is deliberately **not**
/// `#[non_exhaustive]` — the four variants below cover the whole space both
/// wire formats express, and callers benefit from exhaustive `match`es.
///
/// Each client maps the variants onto its own wire encoding:
///
/// | Variant        | Anthropic                        | OpenAI-compatible                                    |
/// |----------------|----------------------------------|------------------------------------------------------|
/// | [`Self::Auto`] | `{"type":"auto"}`                | `"auto"`                                             |
/// | [`Self::Any`]  | `{"type":"any"}`                 | `"required"`                                         |
/// | [`Self::Tool`] | `{"type":"tool","name":…}`       | `{"type":"function","function":{"name":…}}`          |
/// | [`Self::None`] | `{"type":"none"}`                | `"none"`                                             |
///
/// When [`LLMRequest::tool_choice`] is `Option::None` the key is omitted from
/// the request body entirely, leaving the provider's own default in force.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool or answer directly.
    Auto,
    /// Require the model to call one of the offered tools, its choice which.
    Any,
    /// Require the model to call the named tool specifically.
    Tool(String),
    /// Forbid tool calls; the model must answer with text.
    None,
}

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
    /// How the model should choose among `tools`.
    ///
    /// `None` (the default) omits the field from the wire request, so the
    /// provider's own default applies — that is the behavior every request
    /// had before this field existed.
    pub tool_choice: Option<ToolChoice>,
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
            tool_choice: None,
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
            tool_choice: None,
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
    tool_choice: Option<ToolChoice>,
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

    /// Sets how the model should choose among the request's tools.
    ///
    /// Leave unset to send no `tool_choice` at all, which keeps the
    /// provider's default behavior.
    #[must_use]
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
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
            tool_choice: self.tool_choice,
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
    /// The request failed terminally and will not be retried.
    ///
    /// Emitted by the provider when a request errors without a retry (or
    /// after retries are exhausted), so collectors can distinguish a failed
    /// round from an empty successful one. Details are in the provider logs.
    Error,
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
    /// Token usage the provider reported for this round.
    ///
    /// [`Usage::default()`] (all zeros) when the provider reported nothing —
    /// absent usage is never an error, so a zeroed value means "not reported",
    /// not "nothing was spent".
    pub usage: Usage,
    /// The model that actually served this round.
    ///
    /// Per-dispatch, not per-provider: a provider that degraded to its
    /// `fallback_model` because it was rate limited reports the model that
    /// really answered, which is what the round span and the latency
    /// histogram are then labelled with. Empty when no dispatch reached a
    /// client at all.
    pub model: String,
}

// =============================================================================
// Token Usage
// =============================================================================

/// Token counts reported by a provider for one request.
///
/// This is the framework's single canonical usage type; each client converts
/// its own wire shape into it. Fields a provider does not report stay `0` —
/// a zeroed `Usage` means "the provider told us nothing", not "no tokens were
/// spent", so never render it as a definitive figure.
///
/// `input_tokens` counts **uncached** input only, matching Anthropic's wire
/// semantics. The OpenAI client subtracts `cached_tokens` from `prompt_tokens`
/// so both providers agree, which keeps per-rate cost arithmetic uniform.
///
/// Deliberately not `#[non_exhaustive]`: tests and callers legitimately
/// construct it. New fields may still arrive in a minor bump of this 0.x
/// crate, so prefer `..Usage::default()` when constructing it literally.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Uncached input (prompt) tokens.
    pub input_tokens: u64,
    /// Generated output (completion) tokens.
    pub output_tokens: u64,
    /// Input tokens served from a prompt cache. `0` when unreported.
    pub cache_read_tokens: u64,
    /// Input tokens written into a prompt cache. `0` when unreported.
    pub cache_creation_tokens: u64,
}

impl Usage {
    /// Returns true when every counter is zero, i.e. the provider reported
    /// nothing usable.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.cache_read_tokens == 0
            && self.cache_creation_tokens == 0
    }

    /// Total tokens across every counter.
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.output_tokens)
            .saturating_add(self.cache_read_tokens)
            .saturating_add(self.cache_creation_tokens)
    }
}

impl core::ops::Add for Usage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.saturating_add(rhs.input_tokens),
            output_tokens: self.output_tokens.saturating_add(rhs.output_tokens),
            cache_read_tokens: self.cache_read_tokens.saturating_add(rhs.cache_read_tokens),
            cache_creation_tokens: self
                .cache_creation_tokens
                .saturating_add(rhs.cache_creation_tokens),
        }
    }
}

impl core::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::iter::Sum for Usage {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, usage| acc + usage)
    }
}

/// Broadcast whenever a provider finishes a request, successful or not.
///
/// Providers publish this unconditionally: it is a tiny message, and with no
/// subscriber a broadcast is a no-op. Whether anything tallies it is decided
/// by [`ActonAIBuilder::usage_tracking`](crate::facade::ActonAIBuilder::usage_tracking),
/// which governs only whether the accountant actor exists.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct UsageReport {
    /// The **configured** provider name — the key under which the runtime
    /// registered it, not the vendor (`"claude"`, not `"anthropic"`).
    pub provider: String,
    /// The model that served the request.
    pub model: String,
    /// Correlation ID of the request this usage belongs to.
    pub correlation_id: CorrelationId,
    /// The agent that made the request.
    pub agent_id: AgentId,
    /// What the provider reported. May be [`Usage::default()`].
    pub usage: Usage,
}

/// Result of executing a tool call, broadcast after `PromptBuilder::collect`
/// runs the tool. Consumers (e.g. the chat REPL) render this inline so the
/// user can see tool success/failure in the same timeline as the
/// preceding [`LLMStreamToolCall`].
///
/// This is a CLI-observability event — the tool result also lives in
/// `ExecutedToolCall` returned from `CollectedResponse` and in the tool
/// message appended to conversation history.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct LLMStreamToolResult {
    /// Correlation ID for the stream this result belongs to
    pub correlation_id: CorrelationId,
    /// ID of the tool call that produced this result (matches
    /// [`ToolCall::id`]).
    pub tool_call_id: String,
    /// Name of the tool that ran
    pub tool_name: String,
    /// True if the tool returned `Ok`, false if it returned `Err`
    pub success: bool,
    /// Short human-readable preview of the result or error, flattened to
    /// one line and truncated to ~200 chars. Full payload is still
    /// available via `CollectedResponse::tool_calls`.
    pub summary: String,
}

/// The lifecycle bracket of one turn and of the tools it runs.
///
/// Broadcast by the prompt loop so an observer can answer "what is this
/// process doing *right now*" without being on the turn's call path. The
/// [`IntrospectionActor`](crate::introspection::IntrospectionActor) is the
/// only subscriber in this crate.
///
/// # Why a bracket and not inference
///
/// In-flight state could almost be inferred from the existing stream events —
/// [`LLMStreamStart`]/[`LLMStreamEnd`] per round, [`LLMStreamToolCall`] and
/// [`LLMStreamToolResult`] per tool. Almost: a *turn* spans many rounds, so
/// rounds cannot bound it; and the model can emit tool calls the loop
/// deliberately never executes (the siblings of a `structured_output` call,
/// or anything past the round limit), so a call-minus-result count leaks and
/// the reported in-flight number drifts upward for the life of the process.
/// An explicit bracket cannot drift.
///
/// Published unconditionally, like [`UsageReport`]: these are tiny messages
/// and a broadcast with no subscriber is a no-op, so a runtime that never
/// arms introspection pays nothing but the send.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub enum TurnLifecycle {
    /// A turn was admitted and is now running.
    TurnStarted {
        /// Identifies this turn across its whole lifecycle.
        turn_id: TurnId,
    },
    /// A turn ended, successfully or with an error.
    ///
    /// Always published for a turn that published [`Self::TurnStarted`],
    /// whatever the outcome.
    TurnFinished {
        /// The turn that ended.
        turn_id: TurnId,
    },
    /// A turn was never admitted, because admission was closed.
    ///
    /// Published *instead of* the [`Self::TurnStarted`]/[`Self::TurnFinished`]
    /// pair, never alongside it: a refused turn never ran, so counting it as
    /// in-flight even momentarily would make a drain look unfinished.
    TurnRefused,
    /// A tool call started executing inside a turn.
    ToolStarted {
        /// The turn running this tool.
        turn_id: TurnId,
        /// The provider-assigned call ID (matches [`ToolCall::id`]).
        tool_call_id: String,
        /// The tool being executed.
        tool_name: String,
    },
    /// A tool call finished executing, successfully or not.
    ToolFinished {
        /// The turn that ran this tool.
        turn_id: TurnId,
        /// The call that finished.
        tool_call_id: String,
    },
    /// A turn's history was compacted between rounds.
    ///
    /// Published only by a runtime with
    /// [`CompactionConfig`](crate::memory::CompactionConfig) in force, and
    /// only on a round where compaction actually changed the history: the
    /// elided span was summarized by the turn's own provider and the summary
    /// spliced in where the removed messages were. It is the one lifecycle
    /// event that reports work done *to* a turn rather than *by* it, and it
    /// exists because compaction rewrites what the model sees: an operator
    /// debugging a model that "forgot" something needs to be able to see that
    /// the framework took it away — and what it was told instead.
    ContextCompacted {
        /// The turn whose history was compacted.
        turn_id: TurnId,
        /// Estimated tokens before compaction.
        tokens_before: u64,
        /// Estimated tokens after compaction.
        tokens_after: u64,
        /// Messages the summary replaced.
        messages_elided: u64,
    },
}

/// A plan the model published, broadcast the moment it is recorded.
///
/// The `update_plan` built-in tool validates the plan; the prompt round loop
/// — the single owner of a turn's current plan — records it and publishes
/// this event. Published unconditionally, like [`UsageReport`] and
/// [`TurnLifecycle`]: a broadcast nobody is subscribed to costs one send, and
/// a UI that wants to draw the model's progress should not have to be wired
/// in specially to get it. A plan the validator refused is never published —
/// the refusal goes back to the model as that call's tool result instead.
///
/// The turn's *final* plan is also returned on
/// [`CollectedResponse::plan`](crate::stream::CollectedResponse::plan), for
/// callers who want the state rather than the stream.
///
/// Subscribe on an actor's **builder**, before `start()`:
///
/// ```rust,ignore
/// builder.handle().subscribe::<PlanUpdated>().await;
/// ```
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct PlanUpdated {
    /// The turn whose model published this plan.
    pub turn_id: TurnId,
    /// The round that made the call.
    pub correlation_id: CorrelationId,
    /// The provider-assigned call ID, matching the [`ToolCall::id`] it came
    /// from.
    pub tool_call_id: String,
    /// The plan as recorded, already validated.
    pub plan: crate::tools::plan::Plan,
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

impl IncomingAgentMessage {
    /// Creates a new incoming agent message.
    #[must_use]
    pub fn new(from: AgentId, content: impl Into<String>) -> Self {
        Self {
            from,
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

/// Incoming task delegation.
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
    /// Creates a new incoming task with a fresh task ID.
    #[must_use]
    pub fn new(from: AgentId, task_type: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            from,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Usage` with a distinct value in every field, so a fold that drops or
    /// crosses a field is visible in the assertion.
    fn usage(input: u64, output: u64, read: u64, creation: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: read,
            cache_creation_tokens: creation,
        }
    }

    #[test]
    fn usage_default_is_all_zeros() {
        assert_eq!(Usage::default(), usage(0, 0, 0, 0));
        assert!(Usage::default().is_empty());
    }

    #[test]
    fn usage_add_sums_each_field_independently() {
        let total = usage(1, 2, 3, 4) + usage(10, 20, 30, 40);

        assert_eq!(total, usage(11, 22, 33, 44));
    }

    #[test]
    fn usage_add_assign_accumulates_in_place() {
        let mut total = Usage::default();

        total += usage(1, 2, 3, 4);
        total += usage(10, 20, 30, 40);

        assert_eq!(total, usage(11, 22, 33, 44));
    }

    #[test]
    fn usage_add_saturates_instead_of_overflowing() {
        let max = usage(u64::MAX, u64::MAX, u64::MAX, u64::MAX);

        assert_eq!(max + usage(1, 1, 1, 1), max);
    }

    #[test]
    fn usage_sums_over_an_iterator() {
        let rounds = vec![usage(1, 2, 0, 0), usage(3, 4, 0, 0), usage(5, 6, 0, 0)];

        let total: Usage = rounds.into_iter().sum();

        assert_eq!(total, usage(9, 12, 0, 0));
    }

    #[test]
    fn usage_total_tokens_counts_every_field() {
        assert_eq!(usage(1, 2, 3, 4).total_tokens(), 10);
    }

    #[test]
    fn usage_is_empty_only_when_every_field_is_zero() {
        assert!(!usage(0, 0, 0, 1).is_empty());
        assert!(!usage(0, 0, 1, 0).is_empty());
        assert!(!usage(0, 1, 0, 0).is_empty());
        assert!(!usage(1, 0, 0, 0).is_empty());
    }

    #[test]
    fn usage_round_trips_through_json() {
        let original = usage(7, 8, 9, 10);

        let json = serde_json::to_string(&original).unwrap();
        let parsed: Usage = serde_json::from_str(&json).unwrap();

        assert_eq!(original, parsed);
    }

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
