//! Fluent prompt builder for LLM requests.
//!
//! This module provides the `PromptBuilder` for constructing and sending
//! prompts to the LLM with a fluent, ergonomic API.
//!
//! # Simple Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! let response = runtime
//!     .prompt("Explain Rust ownership.")
//!     .system("You are a Rust expert. Be concise.")
//!     .on_token(|token| print!("{token}"))
//!     .collect()
//!     .await?;
//! ```
//!
//! # Tool Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! let response = runtime
//!     .prompt("What is 42 * 17?")
//!     .system("Use the calculator for math.")
//!     .tool(
//!         "calculator",
//!         "Computes math expressions",
//!         json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
//!         |args| async move {
//!             let expr = args["expr"].as_str().unwrap();
//!             Ok(json!({"result": compute(expr)}))
//!         },
//!     )
//!     .on_token(|t| print!("{t}"))
//!     .collect()
//!     .await?;
//! ```

use crate::accounting::{BudgetDecision, CheckBudget};
use crate::checkpoint::{
    plan_from_record, resolve_pending_call, CheckpointRecord, CheckpointSink, FinalAnswer,
    PendingCallAction, PendingCallState, PendingRound, PendingToolCall, ResumePlan, RoundProgress,
    TurnFingerprint, TurnInputs,
};
use crate::conversation::StreamToken;
use crate::error::ActonAIError;
use crate::extract::{
    ensure_name_is_available, repairs_exhausted_error, validation_feedback, StructuredSpec,
    MAX_VALIDATION_REPAIRS, STRUCTURED_OUTPUT_TOOL,
};
use crate::facade::ActonAI;
use crate::llm::{CheckHealth, FailoverEvent, ProviderHealth, SamplingParams};
use crate::messages::{
    LLMRequest, LLMStreamEnd, LLMStreamStart, LLMStreamToken, LLMStreamToolCall, Message,
    StopReason, ToolCall, ToolChoice, ToolDefinition, TurnLifecycle, TurnOutcome, Usage,
};
use crate::stream::{CollectedResponse, ExecutedToolCall, StreamContext};
use crate::tools::ToolError;
use crate::types::{AgentId, CorrelationId, TurnId};
use acton_reactive::prelude::*;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Notify;

/// Framework fallback for the agentic tool-call loop.
///
/// Used when neither a `[defaults]` TOML block nor a builder override
/// supplies `max_tool_rounds`. Per-prompt calls to
/// [`PromptBuilder::max_tool_rounds`] still win over this value.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 10;

/// Nudge appended when an extraction round ends in prose instead of a
/// recorded answer. Sent once per extraction, immediately before the round
/// that forces the `structured_output` call.
const STRUCTURED_OUTPUT_NUDGE: &str = "Record your final answer now by calling structured_output.";

/// Type alias for start callbacks.
///
/// The [`StreamContext`] names the turn and the round that is starting, so a
/// caller multiplexing several concurrent prompts can route the event without
/// inventing a correlation scheme of its own.
type StartCallback = Box<dyn FnMut(&StreamContext) + Send + 'static>;

/// Type alias for token callbacks.
type TokenCallback = Box<dyn FnMut(&str) + Send + 'static>;

/// Type alias for end callbacks.
///
/// Receives the same [`StreamContext`] the round's start callback saw,
/// alongside the reason the provider stopped.
type EndCallback = Box<dyn FnMut(&StreamContext, StopReason) + Send + 'static>;

/// Type alias for tool result callbacks.
///
/// Called after a tool executes with the result (success or error).
type ToolResultCallback = Box<dyn FnMut(Result<&serde_json::Value, &str>) + Send + 'static>;

/// Type alias for tool execution futures.
///
/// Re-exported from [`crate::tools`] so the fluent API and the [`Tool`] trait
/// speak the same future type rather than two structurally-equal aliases.
use crate::tools::ToolFuture;

/// Trait for tool execution functions.
///
/// This trait allows both closures and custom executors to be used
/// as tool handlers in the fluent API.
pub trait ToolExecutorFn: Send + Sync {
    /// Executes the tool with the given arguments.
    fn call(&self, args: serde_json::Value) -> ToolFuture;
}

/// Adapter to wrap async closures as `ToolExecutorFn`.
struct ClosureToolExecutor<F> {
    func: F,
}

impl<F, Fut> ToolExecutorFn for ClosureToolExecutor<F>
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync,
    Fut: Future<Output = Result<serde_json::Value, ToolError>> + Send + 'static,
{
    fn call(&self, args: serde_json::Value) -> ToolFuture {
        Box::pin((self.func)(args))
    }
}

/// Adapter that lets a [`Tool`](crate::tools::Tool) implementation stand in
/// wherever the prompt loop expects a `ToolExecutorFn`.
///
/// The two traits already agree on the call shape — `serde_json::Value` in,
/// [`ToolFuture`] out — so this forwards without adaptation. It exists only
/// because the loop is generic over `ToolExecutorFn`, and making `Tool` a
/// subtrait of it would leak a loop-internal trait into the public API that
/// tool authors implement.
struct TraitToolExecutor<T> {
    tool: T,
}

impl<T> ToolExecutorFn for TraitToolExecutor<T>
where
    T: crate::tools::Tool,
{
    fn call(&self, args: serde_json::Value) -> ToolFuture {
        self.tool.call(args)
    }
}

/// Lets the public builtin execution path stand in as a `ToolExecutorFn`.
///
/// [`BuiltinExecutor`](crate::tools::builtins::BuiltinExecutor) is what
/// [`ActonAI::builtin_executor`](crate::facade::ActonAI::builtin_executor)
/// hands to embedders, and it is also what `use_builtins()` registers — one
/// construction site, one execution path, so what an embedder wraps is
/// literally what the loop runs. The sandbox-or-in-process decision lives
/// inside the executor; this impl only bridges the trait.
impl ToolExecutorFn for crate::tools::builtins::BuiltinExecutor {
    fn call(&self, args: serde_json::Value) -> ToolFuture {
        crate::tools::builtins::BuiltinExecutor::call(self, args)
    }
}

/// A tool specification combining definition, executor, and optional result callback.
pub struct ToolSpec {
    /// The tool definition sent to the LLM
    pub definition: ToolDefinition,
    /// The executor for this tool
    executor: Arc<dyn ToolExecutorFn>,
    /// Optional callback invoked when the tool returns a result.
    ///
    /// Behind a `Mutex` purely so the spec — and with it [`PromptBuilder`] —
    /// is `Sync`; the loop is the only caller and uses `get_mut`, so the
    /// lock is never contended.
    on_result: Option<std::sync::Mutex<ToolResultCallback>>,
}

impl std::fmt::Debug for ToolSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolSpec")
            .field("definition", &self.definition)
            .finish_non_exhaustive()
    }
}

impl Clone for ToolSpec {
    fn clone(&self) -> Self {
        Self {
            definition: self.definition.clone(),
            executor: self.executor.clone(),
            // Callbacks cannot be cloned (FnMut is not Clone)
            on_result: None,
        }
    }
}

/// A tool staged outside any single prompt: a definition plus a shared
/// executor, with no per-prompt result callback.
///
/// [`ToolSpec`] carries an optional `FnMut` callback slot, which makes it
/// `Send` but not `Sync` — fine inside one `PromptBuilder`, fatal inside
/// `ActonAIInner`, which lives behind an `Arc` and must stay `Send + Sync`.
/// Runtime-wide tools (staged with
/// [`ActonAIBuilder::with_tool`](crate::facade::ActonAIBuilder::with_tool))
/// and per-conversation tools (staged with
/// [`ConversationBuilder::with_tool`](crate::conversation::ConversationBuilder::with_tool))
/// are therefore held in this callback-free shape and converted with
/// [`to_tool_spec`](Self::to_tool_spec) at injection time — which puts them
/// in the same list the prompt loop executes, behind the same policy gate
/// and audit trail as the built-ins.
#[derive(Clone)]
pub(crate) struct SharedToolSpec {
    definition: ToolDefinition,
    executor: Arc<dyn ToolExecutorFn>,
}

impl std::fmt::Debug for SharedToolSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedToolSpec")
            .field("definition", &self.definition)
            .finish_non_exhaustive()
    }
}

impl SharedToolSpec {
    /// Builds a spec from a definition and an async closure — the same
    /// wrapping [`PromptBuilder::with_tool`] performs.
    pub(crate) fn from_closure<F, Fut>(definition: ToolDefinition, executor: F) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, ToolError>> + Send + 'static,
    {
        Self {
            definition,
            executor: Arc::new(ClosureToolExecutor { func: executor }),
        }
    }

    /// Builds a spec from a definition and a boxed
    /// [`ToolExecutorTrait`](crate::tools::ToolExecutorTrait) executor.
    ///
    /// Arguments are run through the executor's
    /// [`validate_args`](crate::tools::ToolExecutorTrait::validate_args)
    /// before execution — the executor declared a validator, so skipping it
    /// here would make registration through the facade quietly laxer than
    /// registration anywhere else.
    pub(crate) fn from_executor(
        definition: ToolDefinition,
        executor: Arc<crate::tools::BoxedToolExecutor>,
    ) -> Self {
        Self {
            definition,
            executor: Arc::new(ExecutorTraitAdapter { executor }),
        }
    }

    /// Builds a spec from a value implementing the [`Tool`](crate::tools::Tool)
    /// trait — the shape the `#[tool]` attribute macro generates.
    pub(crate) fn from_tool_impl<T>(tool: T) -> Self
    where
        T: crate::tools::Tool,
    {
        let definition = ToolDefinition {
            // The Tool trait predates the idempotency flag and offers no way
            // to declare it, so trait-built tools take the conservative
            // default: a crash-recovery resume never re-runs them blindly.
            idempotent: false,
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            input_schema: tool.input_schema(),
        };
        Self {
            definition,
            executor: Arc::new(TraitToolExecutor { tool }),
        }
    }

    /// The name the model calls this tool by.
    pub(crate) fn name(&self) -> &str {
        &self.definition.name
    }

    /// Converts into the per-prompt shape the loop executes.
    ///
    /// Cheap: the definition is cloned, the executor `Arc` is shared.
    pub(crate) fn to_tool_spec(&self) -> ToolSpec {
        ToolSpec {
            definition: self.definition.clone(),
            executor: Arc::clone(&self.executor),
            on_result: None,
        }
    }
}

/// Adapter that runs an `Arc<BoxedToolExecutor>` — the executor shape
/// [`ToolExecutorTrait`](crate::tools::ToolExecutorTrait) implementors
/// produce — wherever the prompt loop expects a `ToolExecutorFn`.
///
/// Unlike [`BuiltinToolExecutorAdapter`] it honors the executor's own
/// `validate_args` hook: built-ins validate inside `execute`, but an external
/// executor was promised its validator runs first.
struct ExecutorTraitAdapter {
    executor: Arc<crate::tools::BoxedToolExecutor>,
}

impl ToolExecutorFn for ExecutorTraitAdapter {
    fn call(&self, args: serde_json::Value) -> ToolFuture {
        let executor = Arc::clone(&self.executor);
        Box::pin(async move {
            executor.validate_args(&args)?;
            executor.execute(args).await
        })
    }
}

/// Type alias for wrapped start callback (shared across rounds).
type WrappedStartCallback = Arc<std::sync::Mutex<StartCallback>>;

/// Type alias for wrapped token callback (shared across rounds).
type WrappedTokenCallback = Arc<std::sync::Mutex<TokenCallback>>;

/// Type alias for wrapped end callback (shared across rounds).
type WrappedEndCallback = Arc<std::sync::Mutex<EndCallback>>;

/// A fluent builder for constructing and sending LLM prompts.
///
/// Created via `ActonAI::prompt()`, this builder allows you to configure
/// the request and set up callbacks for streaming responses.
///
/// # Example
///
/// ```rust,ignore
/// runtime
///     .prompt("What is 2 + 2?")
///     .system("Be concise.")
///     .on_token(|t| print!("{t}"))
///     .collect()
///     .await?;
/// ```
///
/// # Multi-Provider Example
///
/// ```rust,ignore
/// // Use a specific provider for this prompt
/// runtime
///     .prompt("Complex reasoning task")
///     .provider("claude")  // Use the "claude" provider
///     .collect()
///     .await?;
/// ```
///
/// # Sending a turn through an actor handler
///
/// The builder — and the future [`collect`](Self::collect) /
/// [`extract`](Self::extract) returns — is `Send + Sync`. This is a
/// deliberate contract, not an accident of the fields: acton-reactive's
/// `Reply::pending` stores handler futures as
/// `Pin<Box<dyn Future + Send + Sync>>`, so an embedder driving turns from
/// inside its own actors (an ACP agent daemon, a per-session actor) can
/// await a turn directly in a handler instead of spawning a detached task
/// and losing cancellation and supervision:
///
/// ```rust,ignore
/// session.act_on::<UserTurn>(|actor, ctx| {
///     let ai = actor.model.ai.clone();
///     let reply = ctx.reply_envelope();
///     let content = ctx.message().content.clone();
///     Reply::pending(async move {
///         let response = ai.prompt(content).collect().await;
///         reply.send(TurnDone::from(response)).await;
///     })
/// });
/// ```
///
/// What keeps it true: the `FnMut` callbacks are stored pre-wrapped in the
/// `Arc<Mutex<_>>` the stream collector needs anyway, and the two `dyn
/// Future + Send` trait objects the loop awaits (tool executions, the
/// approval hook) are polled through `sync_wrapper::SyncFuture`. The
/// `tests/prompt_builder_sync.rs` suite turns a regression here into a
/// compile error.
pub struct PromptBuilder {
    /// The ActonAI runtime (cheaply cloned via Arc)
    runtime: ActonAI,
    /// The user's prompt content
    user_content: String,
    /// Optional system prompt
    system_prompt: Option<String>,
    /// Optional conversation history (replaces user_content when set)
    conversation_history: Option<Vec<Message>>,
    /// Callback for stream start.
    ///
    /// Stored pre-wrapped in the `Arc<Mutex<_>>` the stream collector needs
    /// anyway, rather than as a bare `Box<dyn FnMut>`, because a boxed
    /// `FnMut` is `!Sync` and would make the whole builder `!Sync` — see the
    /// "Sending a turn through an actor handler" section on [`PromptBuilder`].
    on_start: Option<WrappedStartCallback>,
    /// Callback for each token. Wrapped for the same reason as `on_start`.
    on_token: Option<WrappedTokenCallback>,
    /// Callback for stream end. Wrapped for the same reason as `on_start`.
    on_end: Option<WrappedEndCallback>,
    /// Registered tools with inline executors
    tools: Vec<ToolSpec>,
    /// Maximum tool execution rounds (default: 10)
    max_tool_rounds: usize,
    /// Name of the provider to use (None = default provider)
    provider_name: Option<String>,
    /// Optional actor handle to receive [`StreamToken`] messages
    token_target: Option<ActorHandle>,
    /// Optional sampling parameters for this prompt
    sampling: Option<SamplingParams>,
    /// Set by [`PromptBuilder::extract`]: the schema the model must fill in
    /// and the type-erased check that its answer really parses.
    structured: Option<StructuredSpec>,
    /// The conversation this turn belongs to, when it belongs to one.
    ///
    /// Carried for the audit trail: a one-shot `prompt()` has no conversation
    /// to name, while a turn driven by [`Conversation`](crate::Conversation)
    /// does, and an auditor reading the trail needs to be able to tell which
    /// calls belonged to the same exchange.
    conversation_id: Option<crate::types::ConversationId>,
    /// Where this turn records its progress, and where a rerun looks for it.
    ///
    /// Empty unless [`PromptBuilder::checkpoint`] filled it in, and every call
    /// on an empty sink is a no-op — a prompt that did not ask for
    /// checkpointing pays nothing for the feature existing.
    checkpoint: CheckpointSink,
    /// A record handed over by [`ActonAI::resume_turn`](crate::ActonAI::resume_turn),
    /// which the loop trusts in place of a fingerprint-checked lookup.
    ///
    /// The operator path holds only the record — the original inputs were
    /// never stored — so the facade, not the fingerprint, is the authority
    /// that it belongs to this turn. `None` on every other prompt.
    resume_seed: Option<Box<CheckpointRecord>>,
    /// Caller-supplied turn identity, when the caller wants to know the id
    /// before the loop runs. `None` mints a fresh one at admission.
    turn_id: Option<TurnId>,
}

impl PromptBuilder {
    /// Creates a new prompt builder with the given content.
    ///
    /// This is called internally by `ActonAI::prompt()`.
    #[must_use]
    pub(crate) fn new(runtime: ActonAI, user_content: String) -> Self {
        let max_tool_rounds = runtime.default_max_tool_rounds();
        Self {
            runtime,
            user_content,
            system_prompt: None,
            conversation_history: None,
            on_start: None,
            on_token: None,
            on_end: None,
            tools: Vec::new(),
            max_tool_rounds,
            provider_name: None,
            token_target: None,
            sampling: None,
            structured: None,
            conversation_id: None,
            checkpoint: CheckpointSink::disabled(),
            resume_seed: None,
            turn_id: None,
        }
    }

    /// Returns the current `max_tool_rounds` value that will be enforced.
    #[must_use]
    pub fn current_max_tool_rounds(&self) -> usize {
        self.max_tool_rounds
    }

    /// Sets the system prompt for this request.
    ///
    /// The system prompt provides context and instructions to the LLM
    /// about how to respond.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime
    ///     .prompt("What is the capital of France?")
    ///     .system("Be concise. Answer in one word if possible.")
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets conversation history for multi-turn conversations.
    ///
    /// When set, this replaces the initial user content passed to `prompt()`.
    /// Use this for multi-turn conversations where you need to include
    /// prior exchanges between the user and assistant.
    ///
    /// The system prompt (if set via `.system()`) is automatically prepended
    /// to the conversation history.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::prelude::*;
    ///
    /// // Build conversation history
    /// let mut history = vec![
    ///     Message::user("What is Rust?"),
    ///     Message::assistant("Rust is a systems programming language..."),
    /// ];
    ///
    /// // Add new user message
    /// history.push(Message::user("How does ownership work?"));
    ///
    /// // Send with full history
    /// let response = runtime
    ///     .prompt("")  // Ignored when messages() is set
    ///     .system("You are a helpful Rust expert.")
    ///     .messages(history)
    ///     .on_token(|t| print!("{t}"))
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.conversation_history = Some(messages.into_iter().collect());
        self
    }

    /// Sets a callback to be called when the stream starts.
    ///
    /// This is useful for displaying a "thinking" indicator or spinner. The
    /// callback receives the [`StreamContext`] naming the turn and the round
    /// that is starting, so an embedder driving several prompts at once (an
    /// ACP agent fanning turns out to client sessions, say) can route the
    /// event without keeping a side table. The callback fires once per
    /// provider round, so a turn that loops through tools fires it several
    /// times under the same [`StreamContext::turn_id`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime
    ///     .prompt("Hello")
    ///     .on_start(|ctx| println!("Thinking... (turn {})", ctx.turn_id))
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn on_start<F>(mut self, f: F) -> Self
    where
        F: FnMut(&StreamContext) + Send + 'static,
    {
        self.on_start = Some(Arc::new(std::sync::Mutex::new(Box::new(f))));
        self
    }

    /// Sets a callback to be called for each token.
    ///
    /// Tokens are delivered in order as they are received from the LLM.
    /// This is the primary way to stream output to the user.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime
    ///     .prompt("Tell me a story.")
    ///     .on_token(|token| print!("{token}"))
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn on_token<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str) + Send + 'static,
    {
        self.on_token = Some(Arc::new(std::sync::Mutex::new(Box::new(f))));
        self
    }

    /// Sets a callback to be called when the stream ends.
    ///
    /// The callback receives the [`StreamContext`] the round's start callback
    /// saw, plus the stop reason indicating why the LLM stopped generating.
    /// Like [`on_start`](Self::on_start) it fires once per provider round.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime
    ///     .prompt("Hello")
    ///     .on_end(|_ctx, reason| println!("\n[Finished: {reason:?}]"))
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn on_end<F>(mut self, f: F) -> Self
    where
        F: FnMut(&StreamContext, StopReason) + Send + 'static,
    {
        self.on_end = Some(Arc::new(std::sync::Mutex::new(Box::new(f))));
        self
    }

    /// Registers a tool with an inline executor closure.
    ///
    /// This is the most ergonomic way to add tools to a prompt. The closure
    /// receives the tool arguments as JSON and should return the result.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime
    ///     .prompt("What is 42 * 17?")
    ///     .tool(
    ///         "calculator",
    ///         "Computes mathematical expressions",
    ///         json!({
    ///             "type": "object",
    ///             "properties": {
    ///                 "expression": {"type": "string"}
    ///             },
    ///             "required": ["expression"]
    ///         }),
    ///         |args| async move {
    ///             let expr = args["expression"].as_str().unwrap();
    ///             Ok(json!({"result": calculate(expr)}))
    ///         },
    ///     )
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn tool<F, Fut>(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
        executor: F,
    ) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, ToolError>> + Send + 'static,
    {
        let definition = ToolDefinition {
            idempotent: false,
            name: name.into(),
            description: description.into(),
            input_schema,
        };

        let spec = ToolSpec {
            definition,
            executor: Arc::new(ClosureToolExecutor { func: executor }),
            on_result: None,
        };

        self.tools.push(spec);
        self
    }

    /// Registers a tool using a `ToolDefinition`.
    ///
    /// This is a convenience method for when you have a pre-built `ToolDefinition`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let calculator = ToolDefinition {
    ///     name: "calculator".to_string(),
    ///     description: "Evaluates math expressions".to_string(),
    ///     input_schema: json!({
    ///         "type": "object",
    ///         "properties": {
    ///             "expression": { "type": "string" }
    ///         },
    ///     }),
    /// };
    ///
    /// runtime
    ///     .prompt("What is 2 + 2?")
    ///     .with_tool(calculator, |args| async move {
    ///         let expr = args["expression"].as_str().unwrap();
    ///         Ok(json!({"result": calculate(expr)}))
    ///     })
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_tool<F, Fut>(mut self, definition: ToolDefinition, executor: F) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, ToolError>> + Send + 'static,
    {
        let spec = ToolSpec {
            definition,
            executor: Arc::new(ClosureToolExecutor { func: executor }),
            on_result: None,
        };

        self.tools.push(spec);
        self
    }

    /// Appends pre-built tool specs to this prompt.
    ///
    /// This is how the facade injects runtime-wide custom tools (staged with
    /// [`ActonAIBuilder::with_tool`](crate::facade::ActonAIBuilder::with_tool))
    /// and how a [`Conversation`](crate::conversation::Conversation) carries
    /// its per-conversation tools into every turn. Injected specs sit in the
    /// same list as everything else, so the policy gate and the audit trail
    /// see them exactly as they see built-ins.
    #[must_use]
    pub(crate) fn with_tool_specs(mut self, specs: impl IntoIterator<Item = ToolSpec>) -> Self {
        self.tools.extend(specs);
        self
    }

    /// Registers a value implementing the [`Tool`] trait.
    ///
    /// [`Tool`] bundles the name, description, schema, and executor that
    /// [`tool`](Self::tool) takes as four separate arguments, so a tool can be
    /// defined once and registered anywhere. The usual way to get one is the
    /// [`#[tool]`](macro@crate::tool) attribute, which generates the
    /// implementation from an `async fn`:
    ///
    /// ```rust
    /// use acton_ai::prelude::*;
    ///
    /// /// Adds two numbers.
    /// #[tool]
    /// async fn add(a: i64, b: i64) -> Result<serde_json::Value, ToolError> {
    ///     Ok(serde_json::json!({ "sum": a + b }))
    /// }
    ///
    /// # async fn run(runtime: ActonAI) -> Result<(), ActonAIError> {
    /// let response = runtime
    ///     .prompt("What is 42 + 17?")
    ///     .add_tool(Add)
    ///     .collect()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Why not `with_tool`?
    ///
    /// [`with_tool`](Self::with_tool) already means "a `ToolDefinition` plus a
    /// closure" and predates this trait. Rust cannot overload on arity, so
    /// this is a separate method rather than a second form of that one.
    #[must_use]
    pub fn add_tool<T>(mut self, tool: T) -> Self
    where
        T: crate::tools::Tool,
    {
        let definition = ToolDefinition {
            idempotent: false,
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            input_schema: tool.input_schema(),
        };

        self.tools.push(ToolSpec {
            definition,
            executor: Arc::new(TraitToolExecutor { tool }),
            on_result: None,
        });
        self
    }

    /// Registers a tool using a `ToolDefinition` with a result callback.
    ///
    /// The callback is invoked after the tool executes, receiving either the
    /// successful result value or an error message. This is useful for logging,
    /// debugging, or updating UI state when a tool completes.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let calculator = ToolDefinition {
    ///     name: "calculator".to_string(),
    ///     description: "Evaluates math expressions".to_string(),
    ///     input_schema: json!({
    ///         "type": "object",
    ///         "properties": {
    ///             "expression": { "type": "string" }
    ///         },
    ///     }),
    /// };
    ///
    /// runtime
    ///     .prompt("What is 2 + 2?")
    ///     .with_tool_callback(
    ///         calculator,
    ///         |args| async move {
    ///             let expr = args["expression"].as_str().unwrap();
    ///             Ok(json!({"result": calculate(expr)}))
    ///         },
    ///         |result| {
    ///             match result {
    ///                 Ok(value) => println!("Calculator returned: {value}"),
    ///                 Err(e) => println!("Calculator failed: {e}"),
    ///             }
    ///         },
    ///     )
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_tool_callback<F, Fut, C>(
        mut self,
        definition: ToolDefinition,
        executor: F,
        on_result: C,
    ) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, ToolError>> + Send + 'static,
        C: FnMut(Result<&serde_json::Value, &str>) + Send + 'static,
    {
        let spec = ToolSpec {
            definition,
            executor: Arc::new(ClosureToolExecutor { func: executor }),
            on_result: Some(std::sync::Mutex::new(Box::new(on_result))),
        };

        self.tools.push(spec);
        self
    }

    /// Sets the maximum number of tool execution rounds.
    ///
    /// This prevents infinite loops if the LLM keeps requesting tools.
    /// Default is 10 rounds.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime
    ///     .prompt("Complex task")
    ///     .tool(...)
    ///     .max_tool_rounds(5)
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn max_tool_rounds(mut self, max: usize) -> Self {
        self.max_tool_rounds = max;
        self
    }

    /// Sets the provider to use for this prompt.
    ///
    /// When multiple providers are configured, this selects which one
    /// handles this specific prompt. If not called, the default provider
    /// is used.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use a specific provider for complex reasoning
    /// runtime
    ///     .prompt("Analyze this complex problem...")
    ///     .provider("claude")
    ///     .collect()
    ///     .await?;
    ///
    /// // Use a fast/cheap provider for simple tasks
    /// runtime
    ///     .prompt("Summarize this text")
    ///     .provider("fast")
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn provider(mut self, name: impl Into<String>) -> Self {
        self.provider_name = Some(name.into());
        self
    }

    /// Sets the sampling parameters for this prompt.
    ///
    /// These override any provider-level defaults.
    #[must_use]
    pub fn sampling(mut self, params: SamplingParams) -> Self {
        self.sampling = Some(params);
        self
    }

    /// Sets the temperature for this prompt.
    ///
    /// Overrides any provider-level default temperature.
    #[must_use]
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.sampling
            .get_or_insert_with(SamplingParams::default)
            .temperature = Some(temperature);
        self
    }

    /// Sets top_p (nucleus) sampling for this prompt.
    #[must_use]
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.sampling
            .get_or_insert_with(SamplingParams::default)
            .top_p = Some(top_p);
        self
    }

    /// Sets top_k sampling for this prompt.
    #[must_use]
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.sampling
            .get_or_insert_with(SamplingParams::default)
            .top_k = Some(top_k);
        self
    }

    /// Sets stop sequences for this prompt.
    #[must_use]
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.sampling
            .get_or_insert_with(SamplingParams::default)
            .stop_sequences = Some(sequences);
        self
    }

    /// Sets the frequency penalty for this prompt.
    #[must_use]
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.sampling
            .get_or_insert_with(SamplingParams::default)
            .frequency_penalty = Some(penalty);
        self
    }

    /// Sets the presence penalty for this prompt.
    #[must_use]
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.sampling
            .get_or_insert_with(SamplingParams::default)
            .presence_penalty = Some(penalty);
        self
    }

    /// Sets the seed for deterministic generation.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.sampling
            .get_or_insert_with(SamplingParams::default)
            .seed = Some(seed);
        self
    }

    /// Sets a target actor to receive [`StreamToken`] messages during streaming.
    ///
    /// When set, each token received from the LLM is forwarded as a [`StreamToken`]
    /// message to the target actor. The target actor must have a handler registered
    /// for `StreamToken`.
    ///
    /// This is used internally by [`Conversation::send_streaming`](crate::conversation::Conversation::send_streaming).
    #[must_use]
    pub fn token_target(mut self, handle: ActorHandle) -> Self {
        self.token_target = Some(handle);
        self
    }

    /// Names the conversation this turn belongs to.
    ///
    /// Recorded on every audit entry the turn produces, so a reader of the
    /// trail can group the tool calls of one exchange. Set automatically by
    /// [`Conversation`](crate::Conversation); a one-shot prompt has no
    /// conversation and leaves it unset.
    #[must_use]
    pub fn conversation_id(mut self, id: crate::types::ConversationId) -> Self {
        self.conversation_id = Some(id);
        self
    }

    /// Makes this turn resumable, recording its progress under `id` through
    /// the given [`MemoryStore`](crate::memory::MemoryStore) handle.
    ///
    /// The turn saves a checkpoint after every round that completes, so a
    /// process that dies mid-turn leaves behind the conversation so far, the
    /// rounds already spent, and the tools already executed. Running the same
    /// prompt again with the same `id` picks up from there instead of
    /// re-dispatching and re-executing everything; running it again after the
    /// turn finished replays the stored answer without contacting a provider
    /// at all.
    ///
    /// The `id` is the caller's to keep. Persist it alongside whatever work
    /// the turn belongs to — a job row, a queue message — because a
    /// `CheckpointId` that nobody wrote down is a checkpoint nobody can
    /// resume.
    ///
    /// # What counts as the same turn
    ///
    /// A resume is refused, with
    /// [`ActonAIErrorKind::Checkpoint`](crate::error::ActonAIErrorKind::Checkpoint),
    /// when the prompt, system prompt, tool set, provider, or round ceiling
    /// differ from the ones the checkpoint was written for. That is
    /// deliberate: resuming a record written for a different question would
    /// splice two turns together. Change any of those and start a new
    /// checkpoint.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::types::CheckpointId;
    ///
    /// let id = CheckpointId::new();
    /// let answer = runtime
    ///     .prompt("Summarize every .rs file under src/")
    ///     .use_builtins()
    ///     .checkpoint(store.clone(), id.clone())
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn checkpoint(mut self, store: ActorHandle, id: crate::types::CheckpointId) -> Self {
        self.checkpoint = CheckpointSink::to_store(store, id);
        self
    }

    /// Seeds this turn from a record the caller vouches for.
    ///
    /// Used by [`ActonAI::resume_turn`](crate::ActonAI::resume_turn): the record's own
    /// messages are the turn, so the loop skips the fingerprint check and
    /// picks up exactly where the record says the turn stopped. Progress
    /// keeps being written under the record's own ID.
    pub(crate) fn resume_from(mut self, store: ActorHandle, record: CheckpointRecord) -> Self {
        self.checkpoint = CheckpointSink::to_store(store, record.id.clone());
        self.resume_seed = Some(Box::new(record));
        self
    }

    /// Supplies the turn's identity instead of letting the loop mint one.
    ///
    /// Exists for embedders that must announce a turn to their own client
    /// before the loop starts — an ACP agent has to answer a `session/prompt`
    /// with an id it can immediately bind lifecycle events to, and minting
    /// the id itself is the only way to have it that early. Every
    /// [`TurnLifecycle`] event, every
    /// [`LLMStreamToolResult`](crate::messages::LLMStreamToolResult), the
    /// audit trail, and [`CollectedResponse::turn_id`] then carry exactly
    /// this id.
    ///
    /// Callers that only need the id *after* the turn can skip this and read
    /// it from [`CollectedResponse::turn_id`] instead.
    ///
    /// # Uniqueness
    ///
    /// Nothing forces a supplied id to be unique, and the runtime's own
    /// accounting does not need it to be: the introspection actor counts
    /// starts and finishes per id, so a cancel-and-retry that reuses the id
    /// it already announced stays correct even when the cancelled turn's
    /// interrupted finish lands after the retry started. Avoid holding two
    /// turns *live* under one id all the same — lifecycle events key on the
    /// id, so your own subscribers cannot tell the two turns' events apart.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let turn_id = TurnId::new();
    /// notify_client_of_new_turn(&turn_id);
    ///
    /// let response = runtime
    ///     .prompt("Deploy the fix")
    ///     .turn_id(turn_id.clone())
    ///     .collect()
    ///     .await?;
    /// assert_eq!(response.turn_id, turn_id);
    /// ```
    #[must_use]
    pub fn turn_id(mut self, id: TurnId) -> Self {
        self.turn_id = Some(id);
        self
    }

    /// Enables the built-in tools configured on the runtime.
    ///
    /// This method adds all tools that were configured via
    /// [`with_builtins`](crate::ActonAIBuilder::with_builtins) or
    /// [`with_builtin_tools`](crate::ActonAIBuilder::with_builtin_tools)
    /// to this prompt, making them available to the LLM.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_builtin_tools(&["bash", "read_file"])
    ///     .launch()
    ///     .await?;
    ///
    /// // The LLM can now use bash and read_file tools
    /// runtime
    ///     .prompt("List files in the current directory")
    ///     .use_builtins()  // Enable the configured built-in tools
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn use_builtins(mut self) -> Self {
        if let Some(builtins) = self.runtime.builtins() {
            for (name, config) in builtins.configs() {
                // `builtin_executor` owns the sandbox-or-in-process decision:
                // only tools configured `sandboxed` route through the
                // subprocess, and only when a sandbox factory exists. Going
                // through the facade keeps this the same executor an
                // embedder gets from `ActonAI::builtin_executor`.
                if let Some(executor) = self.runtime.builtin_executor(name) {
                    self.tools.push(ToolSpec {
                        definition: config.definition.clone(),
                        executor: Arc::new(executor),
                        on_result: None,
                    });
                }
            }
        }
        self
    }

    /// Sends the prompt and returns a typed, schema-validated value.
    ///
    /// Instead of handing you prose to parse, this appends a synthetic tool
    /// named `structured_output` whose input schema is the JSON Schema of
    /// `T`, and constrains the request so the model has to call it. The
    /// arguments of that call become your `T`. If they don't deserialize,
    /// the serde error is handed back to the model as a tool result and it
    /// is asked to correct itself, up to
    /// [`MAX_VALIDATION_REPAIRS`](crate::extract::MAX_VALIDATION_REPAIRS)
    /// times.
    ///
    /// Real tools still work: anything registered via [`Self::tool`],
    /// [`Self::use_builtins`], or MCP may run first, and extraction is the
    /// terminal step. While other tools are available the model chooses
    /// freely; if a round ends in prose with no answer recorded, it is asked
    /// once more with the choice forced.
    ///
    /// `T` must implement [`serde::Deserialize`] and
    /// [`schemars::JsonSchema`]. Add `schemars` to your own dependencies and
    /// derive it there; the crate is also re-exported as
    /// [`acton_ai::schemars`](crate::schemars).
    ///
    /// Streaming callbacks such as [`Self::on_token`] keep firing throughout.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The runtime has been shut down
    /// - A tool named `structured_output` is already registered on this
    ///   prompt (rename it; extraction will not silently shadow it)
    /// - The model never records an answer, or never records one that
    ///   deserializes into `T` within the repair budget — in which case the
    ///   error carries the serde error and a truncated dump of what the
    ///   model actually produced
    /// - Maximum tool rounds exceeded
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::prelude::*;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Debug, Deserialize, JsonSchema)]
    /// struct Invoice {
    ///     vendor: String,
    ///     total_cents: u64,
    /// }
    ///
    /// let invoice: Invoice = runtime
    ///     .prompt("Extract the invoice from this email: ...")
    ///     .extract::<Invoice>()
    ///     .await?;
    ///
    /// println!("{} owes {} cents", invoice.vendor, invoice.total_cents);
    /// ```
    pub async fn extract<T>(mut self) -> Result<T, ActonAIError>
    where
        T: DeserializeOwned + JsonSchema,
    {
        if self.runtime.is_shutdown() {
            return Err(ActonAIError::runtime_shutdown());
        }
        ensure_name_is_available(self.tools.iter().map(|t| t.definition.name.as_str()))?;
        self.structured = Some(StructuredSpec::for_type::<T>());

        let session = build_stream_collector(&self.runtime).await;
        let result = self.collect_structured(&session).await;
        session.shutdown().await;
        let (value, _response) = result?;

        // The loop already ran `from_value::<T>` on this exact value, so
        // reaching the error arm means something is deeply wrong rather than
        // that the model misbehaved — but it is still not an unwrap.
        serde_json::from_value::<T>(value).map_err(|e| {
            ActonAIError::extraction(format!(
                "an answer that passed validation failed to deserialize: {e}"
            ))
        })
    }

    /// Runs the prompt loop in extraction mode and returns the recorded
    /// answer alongside the usual collected response.
    ///
    /// Kept non-generic so the loop itself never has to know `T`; the caller
    /// ([`Self::extract`]) owns the type parameter.
    pub(crate) async fn collect_structured(
        self,
        session: &StreamCollectorSession,
    ) -> Result<(serde_json::Value, CollectedResponse), ActonAIError> {
        let (response, captured) = self.run_prompt_loop(session).await?;
        match captured {
            Some(value) => Ok((value, response)),
            None => Err(ActonAIError::extraction(
                "the model never called structured_output",
            )),
        }
    }

    /// Sends the prompt and collects the complete response.
    ///
    /// This method:
    /// 1. Creates a temporary actor to collect tokens
    /// 2. Subscribes to streaming events
    /// 3. Sends the request to the LLM provider
    /// 4. If tools are registered and the LLM requests them:
    ///    - Executes the requested tools
    ///    - Sends tool results back to the LLM
    ///    - Repeats until the LLM completes (EndTurn)
    /// 5. Returns the collected response
    ///
    /// Callbacks (`on_start`, `on_token`, `on_end`) are called during streaming.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The runtime has been shut down
    /// - The stream fails to complete
    /// - Maximum tool rounds exceeded
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = runtime
    ///     .prompt("What is 2 + 2?")
    ///     .on_token(|t| print!("{t}"))
    ///     .collect()
    ///     .await?;
    ///
    /// println!("\nFull response: {}", response.text);
    /// ```
    pub async fn collect(self) -> Result<CollectedResponse, ActonAIError> {
        if self.runtime.is_shutdown() {
            return Err(ActonAIError::runtime_shutdown());
        }

        // Build a session scoped to this one-off `collect()` call. Callers
        // that issue many `collect()`s in sequence (like Conversation)
        // should use `collect_with_session` instead so the subscription
        // lives across all calls — that avoids stacking dead subscribers
        // in the acton-reactive broker.
        let session = build_stream_collector(&self.runtime).await;
        let result = self.collect_inner(&session).await;
        session.shutdown().await;
        result
    }

    /// Run `collect()`'s core tool/streaming loop against a caller-owned
    /// [`StreamCollectorSession`]. Used by long-lived callers such as
    /// [`crate::conversation::Conversation`] that want one persistent
    /// subscriber for every turn instead of a fresh one per call.
    pub(crate) async fn collect_with_session(
        self,
        session: &StreamCollectorSession,
    ) -> Result<CollectedResponse, ActonAIError> {
        if self.runtime.is_shutdown() {
            return Err(ActonAIError::runtime_shutdown());
        }
        self.collect_inner(session).await
    }

    async fn collect_inner(
        self,
        session: &StreamCollectorSession,
    ) -> Result<CollectedResponse, ActonAIError> {
        let (response, _captured) = self.run_prompt_loop(session).await?;
        Ok(response)
    }

    /// The shared prompt/tool loop behind both [`Self::collect`] and
    /// [`Self::extract`].
    ///
    /// Returns the collected response plus, when the builder is in
    /// extraction mode, the arguments of the `structured_output` call that
    /// passed validation. The loop is deliberately non-generic: extraction
    /// state travels as a [`StructuredSpec`], never as a type parameter.
    async fn run_prompt_loop(
        self,
        session: &StreamCollectorSession,
    ) -> Result<(CollectedResponse, Option<serde_json::Value>), ActonAIError> {
        // The turn span brackets the whole run, so it is opened out here and
        // closed on every exit path — including the failures, which are the
        // spans an operator most wants to find. `run_rounds` reports what it
        // accumulated through `stats` so the outcome can be recorded even
        // when it returns early with an error.
        let billed_provider = self
            .provider_name
            .clone()
            .unwrap_or_else(|| self.runtime.default_provider_name().to_string());

        // Admission is checked exactly here: before the span opens, before a
        // provider is resolved, before anything is sent. A refusal must cost
        // nothing, or `pause` would be a way to spend money slowly.
        let broker = self.runtime.runtime().broker();
        let admission = self.runtime.admission_state();
        if !admission.admits() {
            broker.broadcast(TurnLifecycle::TurnRefused).await;
            return Err(ActonAIError::turns_not_admitted(admission));
        }

        // From here the turn is admitted, and every exit path below must
        // publish `TurnFinished` — a turn counted as started and never
        // finished holds a drain open forever. "Every exit path" includes the
        // one the compiler cannot see: the caller dropping this future
        // mid-await. The guard owns that duty, so cancellation publishes an
        // `Interrupted` finish instead of leaving the start unmatched.
        //
        // The caller's id when one was supplied, so an embedder that
        // announced the turn to its own client before calling `collect` sees
        // that exact id on every event; minted here otherwise.
        let turn_id = self.turn_id.clone().unwrap_or_default();
        broker
            .broadcast(TurnLifecycle::TurnStarted {
                turn_id: turn_id.clone(),
            })
            .await;
        let guard = TurnFinishedGuard::new(broker, turn_id.clone());

        let turn =
            crate::telemetry::spans::TurnSpan::start(&billed_provider, self.structured.is_some());

        // Cloned before the builder is consumed, so a turn that fails can be
        // marked failed on the way out. The sink is a handle and an ID; the
        // clone costs nothing and is empty when no checkpoint was configured.
        let checkpoint = self.checkpoint.clone();

        let mut stats = TurnStats::default();
        // The claim is the turn's exclusivity: a checkpoint ID has exactly
        // one live owner per process. It is taken before the loop reads the
        // record, so two loops aiming at the same ID — a live turn and an
        // operator's resume, or a retry racing the resume_auto background
        // task — can never both load the same pending round and settle it
        // twice. A refused claim skips the loop entirely and, deliberately,
        // never touches the record: it belongs to the running owner.
        let result = match checkpoint.claim().await {
            Ok(()) => {
                let result = self.run_rounds(session, &turn, &mut stats, &turn_id).await;

                // A failed turn's record stays exactly as it was, so it is still
                // resumable; only its status changes, which is what lets an operator
                // list the turns that fell over separately from the ones still
                // running (terminal records — Completed, Abandoned — are left
                // untouched; see `checkpoint::fail`). A failure to record that
                // must not replace the error the caller actually needs to see.
                if result.is_err() && checkpoint.is_enabled() {
                    if let Err(error) = checkpoint.mark_failed().await {
                        tracing::warn!(%error, "could not mark the turn's checkpoint failed");
                    }
                }

                // Released win or lose, and before TurnFinished: once the
                // turn is over, the ID is free for the next attempt.
                checkpoint.release().await;
                result
            }
            Err(error) => Err(error),
        };

        guard
            .finish(match &result {
                Ok(_) => TurnOutcome::Completed,
                Err(_) => TurnOutcome::Failed,
            })
            .await;

        let outcome = match &result {
            Ok(_) => crate::telemetry::metrics::OUTCOME_OK,
            Err(error) => outcome_for(error),
        };
        turn.finish(stats.rounds, &stats.usage, outcome);

        result
    }

    /// The prompt/tool loop proper.
    ///
    /// Split out of [`Self::run_prompt_loop`] so the turn span has exactly one
    /// place to be closed however the loop ends. Everything it accumulates
    /// that the span needs travels in `stats`, because most of the ways this
    /// returns are early errors.
    async fn run_rounds(
        self,
        session: &StreamCollectorSession,
        turn: &crate::telemetry::spans::TurnSpan,
        stats: &mut TurnStats,
        turn_id: &TurnId,
    ) -> Result<(CollectedResponse, Option<serde_json::Value>), ActonAIError> {
        // Destructure self to take ownership of all fields
        let PromptBuilder {
            runtime,
            user_content,
            system_prompt,
            conversation_history,
            on_start,
            on_token,
            on_end,
            mut tools,
            max_tool_rounds,
            provider_name,
            token_target,
            sampling,
            structured,
            conversation_id,
            checkpoint,
            resume_seed,
            // Already resolved into the `turn_id` parameter by
            // `run_prompt_loop`, which needed it before the span opened.
            turn_id: _,
        } = self;

        // Resolve the provider handle
        let provider_handle = if let Some(ref name) = provider_name {
            runtime.provider_handle_named(name).ok_or_else(|| {
                ActonAIError::configuration(
                    "provider",
                    format!(
                        "provider '{}' not found; available: {}",
                        name,
                        runtime.provider_names().collect::<Vec<_>>().join(", ")
                    ),
                )
            })?
        } else {
            runtime.provider_handle()
        };

        // Pre-flight budget state, resolved once. `None` when nothing is
        // capped, which is what keeps the unbudgeted loop free of an ask.
        let budget_accountant = runtime.budget_accountant().cloned();
        let billed_provider = provider_name
            .clone()
            .unwrap_or_else(|| runtime.default_provider_name().to_string());

        // The gate and the trail, resolved once. Both are `None` in a runtime
        // that configured neither, and every added branch below is guarded by
        // that `Option` — a loop with no policy and no audit does exactly what
        // it did before.
        let tool_policy = runtime.tool_policy().cloned();
        let audit = runtime
            .audit()
            .map(|(handle, config)| (handle.clone(), config.clone()));

        // Per-turn invocation counts. Owned outright by this task, so a plain
        // map: the caps are per turn, and a turn is exactly this loop.
        let mut turn_counts = crate::policy::TurnCounts::new();

        // The compaction step, resolved once. `None` in a runtime that
        // configured no policy, which is what keeps the plan-and-discard work
        // out of every round of every unconfigured turn. Summarization goes
        // to the same provider the turn itself uses, under the same budget.
        let mut compaction = CompactionGate::resolve(
            &runtime,
            provider_handle.clone(),
            billed_provider.clone(),
            budget_accountant.clone(),
        );
        let mut compaction_records: Vec<crate::memory::CompactionRecord> = Vec::new();

        // Build the initial messages
        let mut messages = Vec::new();
        if let Some(ref system) = system_prompt {
            messages.push(Message::system(system));
        }

        // Use conversation history if provided, otherwise use user_content
        if let Some(history) = conversation_history {
            messages.extend(history);
        } else {
            messages.push(Message::user(&user_content));
        }

        // Collect tool definitions. In extraction mode the synthetic
        // `structured_output` tool rides along with the caller's own tools;
        // its name was checked for collisions back in `extract`.
        let mut tool_definitions: Vec<ToolDefinition> =
            tools.iter().map(|t| t.definition.clone()).collect();
        let has_caller_tools = !tool_definitions.is_empty();
        if let Some(ref spec) = structured {
            tool_definitions.push(spec.tool_definition());
        }
        let has_tools = !tool_definitions.is_empty();

        // Extraction state. With no caller tools to work through there is
        // nothing for the model to do but answer, so the choice is forced
        // from round one; otherwise it works freely until it stalls.
        let mut force_structured = structured.is_some() && !has_caller_tools;
        let mut already_nudged = false;
        let mut repairs = 0_usize;
        let mut captured: Option<serde_json::Value> = None;

        // Track executed tool calls and total tokens
        let mut executed_tool_calls = Vec::new();
        // The turn's current plan, when the model has published one via the
        // `update_plan` tool. This local is the plan's one owner: it lives
        // exactly as long as the turn, carries across every round of the tool
        // loop, and every observer sees it only through the `PlanUpdated`
        // broadcasts issued below and the final `CollectedResponse::plan`.
        let mut turn_plan: Option<crate::tools::plan::Plan> = None;
        let mut total_token_count = 0;
        let mut final_text;
        let final_stop_reason;
        // Loop iterations, which is what `max_tool_rounds` bounds. Distinct
        // from `stats.rounds`, which counts dispatches that actually happened.
        let mut iteration = 0;

        // Checkpoint state, resolved once. `checkpoint_inputs` is what decides
        // whether a stored record describes *this* turn; the fingerprint
        // derived from it is written onto every record below. Both are built
        // even when no checkpoint is configured, because they cost a hash over
        // strings the loop already holds and building them unconditionally
        // keeps the branching flat.
        let checkpoint_tool_names: Vec<String> = tool_definitions
            .iter()
            .map(|definition| definition.name.clone())
            .collect();
        let checkpoint_schema = structured
            .as_ref()
            .map(|spec| spec.tool_definition().input_schema.to_string());
        let checkpoint_inputs = TurnInputs {
            system_prompt: system_prompt.as_deref(),
            user_content: &user_content,
            tool_names: &checkpoint_tool_names,
            provider: &billed_provider,
            max_tool_rounds,
            structured_schema: checkpoint_schema.as_deref(),
        };
        // A seeded resume keeps the record's own fingerprint. The seeded
        // builder's inputs are synthetic — `prompt(String::new())` plus
        // whatever tools the runtime holds today — and stamping their hash
        // onto the record would break the original caller's documented
        // retry: a later `.prompt(P).checkpoint(store, id)` must still
        // compare equal against the fingerprint the turn was started with.
        let checkpoint_fingerprint = match resume_seed {
            Some(ref record) => record.fingerprint.clone(),
            None => TurnFingerprint::of(&checkpoint_inputs),
        };

        // The seed path is the operator's authority; the sink path is the
        // fingerprint's. Exactly one of the two produces the plan.
        let plan = match resume_seed {
            Some(record) => plan_from_record(&record, max_tool_rounds)?,
            None => checkpoint.plan(&checkpoint_inputs).await?,
        };

        // Whether this turn picked up an earlier attempt's work. Stamped onto
        // every audit entry the turn writes, so a reader of the trail can
        // tell first-run executions from post-crash ones.
        let mut turn_resumed = false;
        // Failed attempts already recorded against this checkpoint, carried
        // through every progress write so a resume does not reset the count
        // the unattended sweep bounds itself on.
        let mut turn_resume_attempts = 0_u32;
        // The interrupted round to settle before anything is dispatched, when
        // the previous attempt died between a round's tool calls.
        let mut pending_settlement: Option<PendingRound> = None;

        match plan {
            ResumePlan::Start => {}
            ResumePlan::Replay {
                response,
                structured_output,
            } => {
                // The turn already finished. Handing its answer straight back
                // is the whole point of having saved it: re-running finished
                // work must not cost a round, or a retry-on-restart loop pays
                // for the same answer every time it comes up.
                tracing::info!(
                    checkpoint = %DisplayId(checkpoint.id()),
                    "replaying a finished turn from its checkpoint",
                );
                return Ok((*response, structured_output));
            }
            ResumePlan::Resume {
                messages: saved,
                rounds_completed,
                tool_calls,
                token_count,
                usage,
                pending_round,
                resume_attempts,
            } => {
                // `iteration` is seeded rather than reset, so the resumed turn
                // runs under what is left of the original round budget instead
                // of a fresh one.
                tracing::info!(
                    checkpoint = %DisplayId(checkpoint.id()),
                    rounds_completed,
                    "resuming a turn from its checkpoint",
                );
                messages = saved;
                iteration = rounds_completed;
                executed_tool_calls = tool_calls;
                total_token_count = token_count;
                stats.usage = usage;
                turn_resumed = true;
                turn_resume_attempts = resume_attempts;
                pending_settlement = pending_round;
            }
        }

        // Settle an interrupted round before the model hears anything. Each
        // call the dead process left behind is resolved on the tool's own
        // idempotency declaration: a finished result is reused, an unstarted
        // call runs, a started idempotent call runs again, and a started
        // non-idempotent call is NOT re-run — its uncertainty becomes the
        // tool result the model reads.
        if let Some(pending) = pending_settlement.take() {
            let PendingRound {
                assistant_text,
                calls,
            } = pending;
            let settle_correlation_id = CorrelationId::new();
            let mut results: Vec<String> = Vec::with_capacity(calls.len());
            for entry in &calls {
                let idempotent = tools.iter().any(|spec| {
                    spec.definition.name == entry.call.name && spec.definition.idempotent
                });
                match resolve_pending_call(&entry.state, &entry.call.name, idempotent) {
                    PendingCallAction::UseStored { result } => results.push(result),
                    PendingCallAction::Uncertain { feedback } => {
                        // The caller's record says the same thing the model
                        // reads: the call was not run again, and why.
                        executed_tool_calls.push(ExecutedToolCall::error(
                            &entry.call.id,
                            &entry.call.name,
                            entry.call.arguments.clone(),
                            feedback.clone(),
                        ));
                        // The trail hears it too. The first attempt died
                        // before its entry could be written, so this is the
                        // only place the chain can account for a call the
                        // response's tool_calls will show — and the decision
                        // not to re-run it is itself the auditable event.
                        let step = ToolStep {
                            provider_handle: &provider_handle,
                            turn,
                            turn_id,
                            correlation_id: &settle_correlation_id,
                            conversation_id: conversation_id.as_ref(),
                            policy: tool_policy.as_ref(),
                            audit: audit.as_ref(),
                            resumed: true,
                        };
                        step.record_uncertain(&entry.call, &feedback).await;
                        results.push(feedback);
                    }
                    PendingCallAction::Execute => {
                        // The same augmentation the main round loop applies:
                        // `get_context_remaining` reads the turn's live
                        // message state, whose one owner is this loop, so the
                        // measured budget is injected at call time. The
                        // original call, not the augmented one, is what goes
                        // back into `messages` below.
                        let call_with_state;
                        let tool_call = if entry.call.name
                            == crate::tools::builtins::GET_CONTEXT_REMAINING_TOOL
                        {
                            call_with_state = ToolCall {
                                arguments: crate::tools::builtins::inject_context_state(
                                    &entry.call.arguments,
                                    runtime.context_window(),
                                    &messages,
                                ),
                                ..entry.call.clone()
                            };
                            &call_with_state
                        } else {
                            &entry.call
                        };
                        let step = ToolStep {
                            provider_handle: &provider_handle,
                            turn,
                            turn_id,
                            correlation_id: &settle_correlation_id,
                            conversation_id: conversation_id.as_ref(),
                            policy: tool_policy.as_ref(),
                            audit: audit.as_ref(),
                            resumed: true,
                        };
                        let (outcome, executed) =
                            step.run(&mut tools, &mut turn_counts, tool_call).await;
                        // An `update_plan` the settlement itself ran is this
                        // attempt's own execution, not reconstruction of a
                        // previous round's state: the plan owner is updated
                        // and observers hear `PlanUpdated`, exactly as if the
                        // main loop had run the call.
                        if let ToolOutcome::Ran(Ok(value)) = &outcome {
                            if let Some(plan) =
                                crate::tools::plan::plan_from_tool_result(&tool_call.name, value)
                            {
                                provider_handle
                                    .broadcast(crate::messages::PlanUpdated {
                                        turn_id: turn_id.clone(),
                                        correlation_id: settle_correlation_id.clone(),
                                        tool_call_id: tool_call.id.clone(),
                                        plan: plan.clone(),
                                    })
                                    .await;
                                turn_plan = Some(plan);
                            }
                        }
                        results.push(outcome.as_tool_result());
                        executed_tool_calls.push(executed);
                    }
                }
            }

            // Close the round out in the conversation exactly as the loop
            // below would have, had the first attempt survived it.
            let raw_calls: Vec<ToolCall> = calls.into_iter().map(|entry| entry.call).collect();
            messages.push(Message::assistant_with_tools(
                assistant_text,
                raw_calls.clone(),
            ));
            for (call, result) in raw_calls.iter().zip(results) {
                messages.push(Message::tool(&call.id, result));
            }

            // And in the checkpoint: the settled round is a boundary now, so
            // a second crash resumes past it rather than settling it twice.
            checkpoint
                .record_progress(
                    conversation_id.as_ref(),
                    &checkpoint_fingerprint,
                    RoundProgress {
                        rounds_completed: iteration,
                        messages: messages.clone(),
                        tool_calls: executed_tool_calls.clone(),
                        token_count: total_token_count,
                        usage: stats.usage,
                        resume_attempts: turn_resume_attempts,
                        pending_round: None,
                    },
                )
                .await?;
        }

        // The providers this turn may dispatch to, in order, resolved once.
        // The first entry is always the caller's provider; the rest is its
        // configured chain. A runtime with no chains gets a one-element list,
        // and `chained` stays false — which is what keeps every added round
        // trip and every added error path out of the unchained loop.
        let mut candidates: Vec<(String, ActorHandle)> =
            vec![(billed_provider.clone(), provider_handle.clone())];
        for name in runtime
            .provider_failover(&billed_provider)
            .unwrap_or_default()
        {
            let handle = runtime.provider_handle_named(name).ok_or_else(|| {
                ActonAIError::configuration(
                    format!("providers.{billed_provider}.failover"),
                    format!("failover target '{name}' is not a running provider"),
                )
            })?;
            candidates.push((name.clone(), handle));
        }
        let chained = candidates.len() > 1;

        // Only resolved when there is a chain: `FailedOver` is the only event
        // the loop itself publishes, and it can only happen with one.
        let broker = chained.then(|| runtime.runtime().broker());

        // The callbacks arrive already wrapped in the Arc<Mutex> the stream
        // collector shares across rounds; the builder stores them that way so
        // that it stays `Sync` (a bare boxed FnMut is not).

        loop {
            iteration += 1;
            if iteration > max_tool_rounds {
                return Err(ActonAIError::prompt_failed(format!(
                    "exceeded maximum tool rounds ({max_tool_rounds})",
                )));
            }

            // Before the request is built, and therefore before a candidate
            // is chosen: every provider in the chain gets the same history,
            // and a failover must not be the thing that decides whether the
            // history was compacted.
            if let Some(ref mut gate) = compaction {
                if let Some(record) = gate
                    .apply(
                        session,
                        turn_id,
                        &mut messages,
                        stats,
                        &mut total_token_count,
                    )
                    .await
                {
                    compaction_records.push(record);
                }
            }

            // Constrain the choice only while extracting. Plain `collect()`
            // keeps sending no `tool_choice` at all, so nothing changes for
            // callers who never ask for a typed answer.
            let tool_choice = structured.as_ref().map(|_| {
                if force_structured {
                    ToolChoice::Tool(STRUCTURED_OUTPUT_TOOL.to_string())
                } else {
                    ToolChoice::Auto
                }
            });

            // Walk the chain until one provider serves this round. Without a
            // chain the walk is one candidate long and every failure returns
            // straight out, exactly as it did before failover existed.
            let mut attempts: Vec<crate::error::ProviderAttempt> = Vec::new();
            let mut served: Option<(String, CorrelationId, RoundResult)> = None;

            for (position, (candidate, handle)) in candidates.iter().enumerate() {
                let attempt: Result<(CorrelationId, RoundResult), String> = 'candidate: {
                    // Ask the breaker first, and only when there is somewhere
                    // to go: an open circuit is worth knowing about precisely
                    // because it means trying the next provider instead. With
                    // no chain the provider refuses the request itself and
                    // this round trip would buy nothing.
                    if chained {
                        match handle.ask(CheckHealth).await {
                            Ok(ProviderHealth::Open { remaining }) => {
                                break 'candidate Err(format!(
                                    "circuit open for another {}s",
                                    remaining.as_secs(),
                                ));
                            }
                            Ok(_) => {}
                            Err(error) => {
                                break 'candidate Err(format!(
                                    "could not reach the provider to check its circuit: {error}"
                                ));
                            }
                        }
                    }

                    // Every round is a request that costs money, so every
                    // round is checked — the initial one, each tool round, and
                    // each structured-output nudge alike.
                    if let Some(ref accountant) = budget_accountant {
                        if let Err(error) = check_budget(accountant, candidate).await {
                            // A cap on one provider is a reason to try the
                            // next one; with nowhere to go it is the caller's
                            // answer.
                            if !chained {
                                return Err(error);
                            }
                            break 'candidate Err(error.to_string());
                        }
                    }

                    // Generate new IDs for this dispatch. A second candidate
                    // is a second request, so it gets its own correlation ID
                    // rather than reusing the failed one's.
                    let correlation_id = CorrelationId::new();
                    let agent_id = AgentId::new();
                    // Kept for the round span, which is opened after the
                    // request has taken ownership of the original.
                    let round_agent_id = agent_id.to_string();

                    let request = LLMRequest {
                        correlation_id: correlation_id.clone(),
                        agent_id,
                        messages: messages.clone(),
                        tools: if has_tools {
                            Some(tool_definitions.clone())
                        } else {
                            None
                        },
                        sampling: sampling.clone(),
                        tool_choice: tool_choice.clone(),
                    };

                    // Collect stream response — reuses the caller-owned
                    // collector. Keep a clone so we can tag tool-result
                    // broadcasts with the round's correlation ID further down.
                    let round_correlation_id = correlation_id.clone();
                    let round_callbacks = StreamRoundCallbacks {
                        context: Some(StreamContext::new(
                            turn_id.clone(),
                            round_correlation_id.clone(),
                        )),
                        on_start: on_start.clone(),
                        on_token: on_token.clone(),
                        on_end: on_end.clone(),
                        token_target: token_target.clone(),
                    };
                    // Counted here rather than at the top of the loop, so a
                    // turn refused by the budget check reports the rounds that
                    // actually dispatched — not the iteration that never got
                    // to. A failover dispatch is another round by the same
                    // rule: it went out.
                    stats.rounds += 1;

                    // One span and one latency sample per provider dispatch,
                    // opened as a child of the turn.
                    let round_span = turn.round(
                        candidate,
                        &round_correlation_id.to_string(),
                        &round_agent_id,
                        stats.rounds,
                    );
                    let round_started = std::time::Instant::now();

                    let round = run_stream_round(
                        session,
                        handle,
                        &request,
                        correlation_id,
                        round_callbacks,
                    )
                    .await;

                    let elapsed = round_started.elapsed().as_secs_f64();

                    // A transport failure never reaches the collector, so the
                    // round still has to be closed and measured before the
                    // error escapes. Nothing was served, so the only model to
                    // label it with is the configured one.
                    let round = match round {
                        Ok(round) => round,
                        Err(error) => {
                            let configured = runtime
                                .provider_model(candidate)
                                .unwrap_or_default()
                                .to_string();
                            crate::telemetry::metrics::record_request_duration(
                                candidate,
                                &configured,
                                crate::telemetry::metrics::OUTCOME_ERROR,
                                elapsed,
                            );
                            round_span.finish(
                                &Usage::default(),
                                StopReason::Error,
                                crate::telemetry::metrics::OUTCOME_ERROR,
                                &configured,
                            );
                            if !chained {
                                return Err(error);
                            }
                            break 'candidate Err(error.to_string());
                        }
                    };

                    // The model that actually served, which a rate limit may
                    // have degraded away from the configured one. Empty only
                    // on a round the provider refused before dispatching.
                    let served_model = if round.model.is_empty() {
                        runtime
                            .provider_model(candidate)
                            .unwrap_or_default()
                            .to_string()
                    } else {
                        round.model.clone()
                    };

                    // A round that ends in `StopReason::Error` cost time and
                    // possibly tokens, so it is recorded as a round that
                    // happened and failed — not as one that never ran.
                    let round_outcome = if round.stop_reason == StopReason::Error {
                        crate::telemetry::metrics::OUTCOME_ERROR
                    } else {
                        crate::telemetry::metrics::OUTCOME_OK
                    };
                    crate::telemetry::metrics::record_request_duration(
                        candidate,
                        &served_model,
                        round_outcome,
                        elapsed,
                    );
                    round_span.finish(
                        &round.usage,
                        round.stop_reason,
                        round_outcome,
                        &served_model,
                    );

                    // Folded even for a failed candidate: those tokens were
                    // spent whether or not the answer was usable.
                    total_token_count += round.token_count;
                    stats.usage += round.usage;

                    if round.stop_reason == StopReason::Error {
                        if !chained {
                            return Err(ActonAIError::prompt_failed(
                                "LLM request failed; see provider logs for details",
                            ));
                        }
                        break 'candidate Err(
                            "the round failed; see provider logs for details".to_string()
                        );
                    }

                    Ok((round_correlation_id, round))
                };

                match attempt {
                    Ok((round_correlation_id, round)) => {
                        served = Some((candidate.clone(), round_correlation_id, round));
                        break;
                    }
                    Err(reason) => {
                        tracing::warn!(provider = %candidate, %reason, "provider did not serve the round");
                        attempts.push(crate::error::ProviderAttempt::new(candidate, reason));
                        // One event per hop, so a trace shows the whole walk
                        // rather than only where it ended up.
                        if let (Some(broker), Some((next, _))) =
                            (broker.as_ref(), candidates.get(position + 1))
                        {
                            broker
                                .broadcast(FailoverEvent::FailedOver {
                                    from: candidate.clone(),
                                    to: next.clone(),
                                })
                                .await;
                        }
                    }
                }
            }

            // Only reachable with a chain: an unchained walk either serves or
            // has already returned the provider's own error.
            let Some((serving_provider, round_correlation_id, round)) = served else {
                return Err(ActonAIError::all_providers_failed(attempts));
            };
            if serving_provider != billed_provider {
                tracing::info!(
                    primary = %billed_provider,
                    served_by = %serving_provider,
                    "round served by a failover provider",
                );
            }

            let RoundResult {
                text,
                stop_reason,
                tool_calls,
                ..
            } = round;
            final_text = text.clone();

            // A recorded answer ends the run whatever the stop reason says —
            // providers are not consistent about reporting `ToolUse` when the
            // only call they emit is the terminal one.
            if let Some(ref spec) = structured {
                if let Some(call) = tool_calls
                    .iter()
                    .find(|call| call.name == STRUCTURED_OUTPUT_TOOL)
                {
                    match spec.validate(&call.arguments) {
                        Ok(()) => {
                            if tool_calls.len() > 1 {
                                tracing::debug!(
                                    skipped = tool_calls.len() - 1,
                                    "structured_output captured; sibling tool calls in the \
                                     same round were not executed"
                                );
                            }
                            captured = Some(call.arguments.clone());
                            final_stop_reason = stop_reason;
                            break;
                        }
                        Err(message) => {
                            repairs += 1;
                            if repairs > MAX_VALIDATION_REPAIRS {
                                return Err(repairs_exhausted_error(&message, &call.arguments));
                            }

                            // Echo back only the offending call, not its
                            // siblings: every tool call in an assistant
                            // message needs a matching tool result, and the
                            // siblings deliberately never ran.
                            messages.push(Message::assistant_with_tools(text, vec![call.clone()]));
                            messages.push(Message::tool(&call.id, validation_feedback(&message)));
                            force_structured = true;
                            continue;
                        }
                    }
                }
            }

            match stop_reason {
                StopReason::ToolUse if !tool_calls.is_empty() => {
                    // The per-call ledger, kept only when this turn
                    // checkpoints. Written before the first tool runs and
                    // rewritten around every call — each write one upsert of
                    // the whole record, so the ledger and the progress it
                    // belongs to can never disagree. A process that dies
                    // mid-round leaves behind exactly which calls finished,
                    // which never began, and which are in doubt.
                    let mut pending = checkpoint.is_enabled().then(|| PendingRound {
                        assistant_text: text.clone(),
                        calls: tool_calls
                            .iter()
                            .map(|call| PendingToolCall {
                                call: call.clone(),
                                state: PendingCallState::Pending,
                            })
                            .collect(),
                    });
                    if let Some(ref ledger) = pending {
                        checkpoint
                            .record_progress(
                                conversation_id.as_ref(),
                                &checkpoint_fingerprint,
                                RoundProgress {
                                    rounds_completed: iteration,
                                    messages: messages.clone(),
                                    tool_calls: executed_tool_calls.clone(),
                                    token_count: total_token_count,
                                    usage: stats.usage,
                                    resume_attempts: turn_resume_attempts,
                                    pending_round: Some(ledger.clone()),
                                },
                            )
                            .await?;
                    }

                    // Execute tools and continue
                    let mut tool_results = Vec::new();
                    for (index, tool_call) in tool_calls.iter().enumerate() {
                        // `Started` lands before execution begins, so a crash
                        // during the call is distinguishable from one before
                        // it — the distinction the idempotency rules turn on.
                        if let Some(ref mut ledger) = pending {
                            ledger.calls[index].state = PendingCallState::Started;
                            checkpoint
                                .record_progress(
                                    conversation_id.as_ref(),
                                    &checkpoint_fingerprint,
                                    RoundProgress {
                                        rounds_completed: iteration,
                                        messages: messages.clone(),
                                        tool_calls: executed_tool_calls.clone(),
                                        token_count: total_token_count,
                                        usage: stats.usage,
                                        resume_attempts: turn_resume_attempts,
                                        pending_round: Some(ledger.clone()),
                                    },
                                )
                                .await?;
                        }
                        // `get_context_remaining` reads the turn's live
                        // message state. That state has exactly one owner —
                        // this loop — so the measured budget is injected into
                        // the call's arguments here, at call time, and the
                        // tool itself stays pure arithmetic. The original
                        // call, not the augmented one, is what goes back into
                        // `messages` below.
                        let call_with_state;
                        let tool_call = if tool_call.name
                            == crate::tools::builtins::GET_CONTEXT_REMAINING_TOOL
                        {
                            call_with_state = ToolCall {
                                arguments: crate::tools::builtins::inject_context_state(
                                    &tool_call.arguments,
                                    runtime.context_window(),
                                    &messages,
                                ),
                                ..tool_call.clone()
                            };
                            &call_with_state
                        } else {
                            tool_call
                        };
                        let step = ToolStep {
                            provider_handle: &provider_handle,
                            turn,
                            turn_id,
                            correlation_id: &round_correlation_id,
                            conversation_id: conversation_id.as_ref(),
                            policy: tool_policy.as_ref(),
                            audit: audit.as_ref(),
                            resumed: turn_resumed,
                        };

                        let (outcome, executed) =
                            step.run(&mut tools, &mut turn_counts, tool_call).await;

                        // A successful `update_plan` replaces the turn's plan
                        // and is announced. The decision is entirely
                        // `plan_from_tool_result`, a pure function: any other
                        // tool, a failed or refused call, or a result that no
                        // longer validates yields `None`, so a rejected plan
                        // never disturbs the one already recorded. Fire-and-
                        // forget like the events around it — a plan is an
                        // observation, and a turn must not wait on anyone
                        // watching it.
                        if let ToolOutcome::Ran(Ok(value)) = &outcome {
                            if let Some(plan) =
                                crate::tools::plan::plan_from_tool_result(&tool_call.name, value)
                            {
                                provider_handle
                                    .broadcast(crate::messages::PlanUpdated {
                                        turn_id: turn_id.clone(),
                                        correlation_id: round_correlation_id.clone(),
                                        tool_call_id: tool_call.id.clone(),
                                        plan: plan.clone(),
                                    })
                                    .await;
                                turn_plan = Some(plan);
                            }
                        }

                        executed_tool_calls.push(executed);
                        // `Completed` lands with the result in the same write,
                        // so a resume can reuse it instead of running anything.
                        if let Some(ref mut ledger) = pending {
                            ledger.calls[index].state = PendingCallState::Completed {
                                result: outcome.as_tool_result(),
                            };
                            checkpoint
                                .record_progress(
                                    conversation_id.as_ref(),
                                    &checkpoint_fingerprint,
                                    RoundProgress {
                                        rounds_completed: iteration,
                                        messages: messages.clone(),
                                        tool_calls: executed_tool_calls.clone(),
                                        token_count: total_token_count,
                                        usage: stats.usage,
                                        resume_attempts: turn_resume_attempts,
                                        pending_round: Some(ledger.clone()),
                                    },
                                )
                                .await?;
                        }
                        tool_results.push(outcome);
                    }

                    // Add assistant message with tool calls to conversation
                    messages.push(Message::assistant_with_tools(text, tool_calls.clone()));

                    // Add tool result messages. A refused call gets the
                    // gate's own words rather than an error string: it is a
                    // normal outcome the model is expected to work around,
                    // and dressing it as a failure invites a retry.
                    for (tool_call, outcome) in tool_calls.iter().zip(tool_results.iter()) {
                        messages.push(Message::tool(&tool_call.id, outcome.as_tool_result()));
                    }

                    // The resumable point. `messages` is now exactly what the
                    // next round would send, so a run that dies after this
                    // write picks up here rather than re-executing the tools
                    // above. Guarded on `is_enabled` because building the
                    // progress clones the whole conversation, which an
                    // uncheckpointed turn has no reason to pay for.
                    if checkpoint.is_enabled() {
                        checkpoint
                            .record_progress(
                                conversation_id.as_ref(),
                                &checkpoint_fingerprint,
                                RoundProgress {
                                    rounds_completed: iteration,
                                    messages: messages.clone(),
                                    tool_calls: executed_tool_calls.clone(),
                                    token_count: total_token_count,
                                    usage: stats.usage,
                                    resume_attempts: turn_resume_attempts,
                                    pending_round: None,
                                },
                            )
                            .await?;
                    }
                }
                // The round produced no answer and nothing left to execute:
                // either the turn genuinely ended, or the provider reported
                // `ToolUse` with no calls attached.
                _ => {
                    if structured.is_some() {
                        if already_nudged {
                            return Err(ActonAIError::extraction(
                                "the model ended its turn without calling structured_output, \
                                 even after being asked to record its answer",
                            ));
                        }
                        already_nudged = true;
                        force_structured = true;
                        if !text.is_empty() {
                            messages.push(Message::assistant(text));
                        }
                        messages.push(Message::user(STRUCTURED_OUTPUT_NUDGE));
                        continue;
                    }
                    // Conversation complete
                    final_stop_reason = stop_reason;
                    break;
                }
            }
        }

        // One completion write covers both ways out of the loop — a recorded
        // structured answer and an ordinary end of turn — so there is exactly
        // one place a finished turn is marked finished.
        if checkpoint.is_enabled() {
            checkpoint
                .record_completion(
                    conversation_id.as_ref(),
                    &checkpoint_fingerprint,
                    RoundProgress {
                        rounds_completed: iteration,
                        messages,
                        tool_calls: executed_tool_calls.clone(),
                        token_count: total_token_count,
                        usage: stats.usage,
                        resume_attempts: turn_resume_attempts,
                        pending_round: None,
                    },
                    FinalAnswer {
                        text: final_text.clone(),
                        stop_reason: final_stop_reason,
                        structured_output: captured.clone(),
                    },
                )
                .await?;
        }

        Ok((
            CollectedResponse::with_tool_calls(
                final_text,
                final_stop_reason,
                total_token_count,
                executed_tool_calls,
            )
            .with_usage(stats.usage)
            .with_plan(turn_plan)
            .with_compactions(compaction_records)
            // The id every lifecycle event of this turn carried — supplied by
            // the caller or minted at admission — so the response can be
            // matched against events observed while the turn ran.
            .with_turn_id(turn_id.clone()),
            captured,
        ))
    }
}

/// Renders an optional checkpoint ID for a tracing field.
///
/// A `None` prints as `-` rather than as `None`, so an operator grepping the
/// logs of a mixed workload can tell an unconfigured turn from a configured
/// one at a glance.
struct DisplayId<'a>(Option<&'a crate::types::CheckpointId>);

impl std::fmt::Display for DisplayId<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(id) => write!(f, "{id}"),
            None => f.write_str("-"),
        }
    }
}

/// Owns the duty to publish [`TurnLifecycle::TurnFinished`] exactly once.
///
/// The prompt loop broadcasts `TurnStarted` and *must* balance it with a
/// `TurnFinished`, or the introspection actor counts the turn as in-flight
/// forever and a drain waiting on it never completes. The `Ok`/`Err` paths
/// through [`PromptBuilder::run_prompt_loop`] do that themselves via
/// [`finish`](Self::finish) — but a caller can also *drop* the `collect()` /
/// `extract()` future mid-await (a user pressed cancel, a `select!` took the
/// other arm), and a dropped future never reaches any of its own code again.
/// This guard's `Drop` is the only thing that still runs on that path, so
/// the balancing broadcast lives here.
///
/// # Why `Drop` spawns instead of blocking
///
/// `Drop` is synchronous and usually runs inside an async context, so it can
/// neither `.await` the broadcast nor block on it without risking a deadlock
/// on the very runtime that is dropping the future. It therefore hands the
/// send to a spawned task — the same fire-and-forget shape
/// [`StreamCollectorSessionInner`]'s `Drop` already uses — on the runtime
/// handle captured when the guard was built. Captured, not looked up at
/// drop time: `Drop` can run on a thread with no Tokio context at all — an
/// embedder stores the `Send + Sync` `collect()` future in its own session
/// table and a UI thread, a C-FFI callback, or a watchdog `std::thread`
/// drops the entry — and on that thread `Handle::try_current()` offers
/// nothing while the runtime that started the turn is still healthy and
/// still counting the turn in-flight. The ambient handle remains the
/// fallback for the degenerate case of a guard built with no handle
/// current.
///
/// # Why the normal path cannot double-fire
///
/// [`finish`](Self::finish) marks the guard fired *after* its broadcast
/// completes, and `Drop` checks that mark. If the future is cancelled *inside*
/// `finish`'s own broadcast — the one window where both could act — the
/// duplicate `TurnFinished` is harmless by design: the introspection actor
/// keeps in-flight turns in a `HashSet`, where a second remove is a no-op.
/// The failure direction is chosen deliberately: an occasional duplicate
/// finish is idempotent, while a missing one wedges a drain forever.
struct TurnFinishedGuard {
    broker: BrokerRef,
    turn_id: TurnId,
    /// Where `Drop` spawns the balancing broadcast. See the type-level note
    /// on why this is captured at construction instead of looked up at drop.
    handle: Option<tokio::runtime::Handle>,
    fired: bool,
}

impl TurnFinishedGuard {
    /// Arms the guard for `turn_id`. Call immediately after `TurnStarted`.
    fn new(broker: BrokerRef, turn_id: TurnId) -> Self {
        Self {
            broker,
            turn_id,
            // Effectively always `Some`: `new` runs inside the prompt loop's
            // async context. `try_current` rather than `current` so a guard
            // built anywhere stranger degrades instead of panicking.
            handle: tokio::runtime::Handle::try_current().ok(),
            fired: false,
        }
    }

    /// Publishes the balancing `TurnFinished` with how the turn ended.
    ///
    /// Consumes the guard: after this, its `Drop` is a no-op.
    async fn finish(mut self, outcome: TurnOutcome) {
        self.broker
            .broadcast(TurnLifecycle::TurnFinished {
                turn_id: self.turn_id.clone(),
                outcome,
            })
            .await;
        self.fired = true;
    }
}

impl Drop for TurnFinishedGuard {
    fn drop(&mut self) {
        if self.fired {
            return;
        }
        let broker = self.broker.clone();
        let turn_id = self.turn_id.clone();
        let handle = self
            .handle
            .take()
            .or_else(|| tokio::runtime::Handle::try_current().ok());
        if let Some(handle) = handle {
            handle.spawn(async move {
                broker
                    .broadcast(TurnLifecycle::TurnFinished {
                        turn_id,
                        outcome: TurnOutcome::Interrupted,
                    })
                    .await;
            });
        }
    }
}

/// What the turn span needs from a run that may not have finished.
///
/// Passed into [`PromptBuilder::run_rounds`] by `&mut` so the totals survive
/// an early error return and the span can still record how far the turn got.
#[derive(Debug, Default)]
pub(crate) struct TurnStats {
    /// Rounds started, including the one that failed.
    rounds: usize,
    /// Usage summed across every round of the tool loop: one `collect()` can
    /// drive several provider requests, and the caller is billed for all of
    /// them.
    usage: Usage,
}

/// The auto-compaction step of the prompt loop.
///
/// Resolved once per turn and only when the runtime has both a context window
/// and a policy. Runs between rounds — before a request is built, never while
/// a tool exchange is in flight — so a compaction can never split a
/// `tool_use` from its `tool_result`.
///
/// The summary is written by the **same provider** the turn dispatches to:
/// the model that will continue the conversation is the one that decides what
/// the conversation still needs. That request is a paid round like any other,
/// so it passes the same budget check, and its usage folds into the turn's
/// totals.
///
/// This is the only place a history the caller handed us is rewritten, and it
/// is deliberately loud about it: an `info` log and a lifecycle broadcast on
/// every pass, and a [`CompactionRecord`](crate::memory::CompactionRecord)
/// handed back so the caller's persistence can store what happened. A
/// framework that silently deletes context is indistinguishable, from the
/// outside, from a model that ignores it.
struct CompactionGate {
    window: crate::memory::ContextWindow,
    config: crate::memory::CompactionConfig,
    broker: ActorHandle,
    /// The turn's own provider, which also writes the summaries.
    provider: ActorHandle,
    /// The name the summarization spend is billed under.
    provider_name: String,
    /// The budget gate, when one is configured: a summary costs money too.
    accountant: Option<ActorHandle>,
    /// Latched on the first failed or refused summarization. A provider that
    /// cannot summarize now is unlikely to summarize on the very next round,
    /// and without the latch every remaining round of the turn would pay for
    /// — and wait on — another doomed attempt.
    stalled: bool,
}

impl CompactionGate {
    /// Builds the gate, or `None` when this runtime does not compact.
    fn resolve(
        runtime: &ActonAI,
        provider: ActorHandle,
        provider_name: String,
        accountant: Option<ActorHandle>,
    ) -> Option<Self> {
        let config = *runtime.compaction()?;
        // No window means no budget to measure against, so a policy alone is
        // inert rather than an error: `without_context_window` is an explicit
        // choice to ship the whole history.
        let window = runtime.context_window()?.clone();
        Some(Self {
            window,
            config,
            broker: runtime.runtime().broker(),
            provider,
            provider_name,
            accountant,
            stalled: false,
        })
    }

    /// Compacts `messages` in place if the policy calls for it, returning the
    /// record of what happened when it does.
    ///
    /// Every failure path returns `None` and leaves `messages` untouched:
    /// a turn whose summarization fails proceeds with its full history and
    /// takes its chances at the provider, which can only be better than
    /// proceeding with a hole where its history used to be.
    async fn apply(
        &mut self,
        session: &StreamCollectorSession,
        turn_id: &TurnId,
        messages: &mut Vec<Message>,
        stats: &mut TurnStats,
        total_token_count: &mut usize,
    ) -> Option<crate::memory::CompactionRecord> {
        if self.stalled {
            return None;
        }

        let plan = crate::memory::plan_compaction(&self.window, &self.config, messages)?;

        // The summary is a paid request, so it faces the same budget gate as
        // the rounds it exists to protect. A refusal stalls the gate rather
        // than erroring the turn: the caller's own dispatch will run the same
        // check and produce the caller-facing answer.
        if let Some(ref accountant) = self.accountant {
            if let Err(error) = check_budget(accountant, &self.provider_name).await {
                tracing::warn!(
                    error = %error,
                    "budget refused the summarization request; \
                     continuing this turn without compaction",
                );
                self.stalled = true;
                return None;
            }
        }

        // A fresh correlation ID: this is its own request, not part of any
        // round's stream. Default callbacks keep the caller's token hooks
        // quiet — the summary is framework traffic, not the model's answer.
        let correlation_id = CorrelationId::new();
        let request = LLMRequest {
            correlation_id: correlation_id.clone(),
            agent_id: AgentId::new(),
            messages: crate::memory::summarization_messages(&plan),
            tools: None,
            sampling: None,
            tool_choice: None,
        };

        stats.rounds += 1;
        let round = match run_stream_round(
            session,
            &self.provider,
            &request,
            correlation_id,
            StreamRoundCallbacks::default(),
        )
        .await
        {
            Ok(round) => round,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "summarization request failed; continuing this turn without compaction",
                );
                self.stalled = true;
                return None;
            }
        };

        // Spent whether or not the summary is usable.
        *total_token_count += round.token_count;
        stats.usage += round.usage;

        let summary = round.text.trim().to_string();
        if round.stop_reason == StopReason::Error || summary.is_empty() {
            tracing::warn!(
                stop_reason = ?round.stop_reason,
                "provider returned no usable summary; \
                 continuing this turn without compaction",
            );
            self.stalled = true;
            return None;
        }

        // Declined when the summary outweighs what it would replace. Not
        // latched, unlike a failure: the history grows on every round of a
        // tool loop, so the next plan elides a larger span and the same-sized
        // summary can succeed against it.
        let Some((compacted, outcome)) =
            crate::memory::finish_compaction(&self.window, &plan, &summary)
        else {
            tracing::debug!(
                "summary would not shrink the history; \
                 leaving it uncompacted this round",
            );
            return None;
        };

        tracing::info!(
            messages_before = outcome.messages_before,
            messages_after = outcome.messages_after,
            messages_elided = outcome.messages_elided,
            tokens_before = outcome.tokens_before,
            tokens_after = outcome.tokens_after,
            max_tokens = self.window.config().max_tokens,
            "compacted conversation history to stay within the context window",
        );

        *messages = compacted;

        self.broker
            .broadcast(TurnLifecycle::ContextCompacted {
                turn_id: turn_id.clone(),
                tokens_before: outcome.tokens_before as u64,
                tokens_after: outcome.tokens_after as u64,
                messages_elided: outcome.messages_elided as u64,
            })
            .await;

        Some(crate::memory::CompactionRecord { summary, outcome })
    }
}

/// The span outcome label for an error.
///
/// These strings are what operators filter and group on, so they are a
/// deliberately small, stable vocabulary rather than rendered messages —
/// `budget_exceeded` in particular is the one a team watches once a cap is in
/// force.
fn outcome_for(error: &ActonAIError) -> &'static str {
    use crate::error::ActonAIErrorKind as Kind;
    match &error.kind {
        Kind::BudgetExceeded { .. } => "budget_exceeded",
        Kind::Configuration { .. } => "configuration",
        Kind::LaunchFailed { .. } => "launch_failed",
        Kind::PromptFailed { .. } => "prompt_failed",
        Kind::StreamError { .. } => "stream_error",
        Kind::ProviderError { .. } => "provider_error",
        Kind::RuntimeShutdown => "runtime_shutdown",
        Kind::Mcp { .. } => "mcp",
        Kind::Extraction { .. } => "extraction",
        Kind::AllProvidersFailed { .. } => "all_providers_failed",
        // Distinct from every failure above: nothing broke and nothing was
        // spent. A team draining for a deploy needs these separable from real
        // errors, or a clean rollout looks like an incident.
        Kind::TurnsNotAdmitted { .. } => "turns_not_admitted",
        Kind::Checkpoint { .. } => "checkpoint",
    }
}

/// Callbacks and token target that apply to a single stream round.
///
/// Sent into the long-lived [`StreamCollectorSession`] via
/// [`ResetStreamRound`] before each round so the collector's event handlers
/// dispatch to the correct caller-supplied hooks without needing to rebuild
/// broker subscriptions.
///
/// Implements `Debug` manually because the boxed `FnMut` callbacks inside
/// the wrapped slots don't themselves implement `Debug`, and
/// `#[acton_actor]` auto-derives `Debug` on any actor model that holds
/// this type.
#[derive(Clone, Default)]
pub(crate) struct StreamRoundCallbacks {
    /// The turn+round identity handed to the start and end callbacks.
    ///
    /// `Some` whenever the round loop installed this set; `None` only in the
    /// idle state between rounds, where no callback can fire because the
    /// correlation filter is also clear.
    pub(crate) context: Option<StreamContext>,
    pub(crate) on_start: Option<WrappedStartCallback>,
    pub(crate) on_token: Option<WrappedTokenCallback>,
    pub(crate) on_end: Option<WrappedEndCallback>,
    pub(crate) token_target: Option<ActorHandle>,
}

impl std::fmt::Debug for StreamRoundCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamRoundCallbacks")
            .field("context", &self.context)
            .field("on_start", &self.on_start.is_some())
            .field("on_token", &self.on_token.is_some())
            .field("on_end", &self.on_end.is_some())
            .field("token_target", &self.token_target.is_some())
            .finish()
    }
}

/// Long-lived stream collector. Owns broker subscriptions for a whole
/// [`crate::conversation::Conversation`] (or a single `collect()` call when
/// used ad-hoc); each tool round sends a [`ResetStreamRound`] to swap the
/// correlation-ID filter and the round's callbacks before firing the next
/// request.
///
/// Reusing one session across all turns avoids stacking dead broker
/// subscribers — `acton-reactive`'s `UnsubscribeBroker` ships as a no-op,
/// so every new subscriber that later stops leaves a closed channel in the
/// broker's table, producing `Recipient channel is closed` errors on every
/// subsequent broadcast.
///
/// The session is cheap to clone (it's an [`Arc`] under the hood) so it can
/// be shared between the owner and transient handler closures. The inner
/// actor handle is stopped exactly once when the last clone drops.
#[derive(Clone)]
pub(crate) struct StreamCollectorSession {
    inner: Arc<StreamCollectorSessionInner>,
}

struct StreamCollectorSessionInner {
    /// `None` after [`StreamCollectorSession::shutdown`] runs. `Drop`
    /// treats that as a no-op so we never double-stop the actor.
    handle: std::sync::Mutex<Option<ActorHandle>>,
    completion: Arc<Notify>,
    result_container: Arc<std::sync::Mutex<Option<CollectorResultData>>>,
}

impl StreamCollectorSession {
    /// Stop the underlying actor explicitly.
    ///
    /// Callers that want deterministic shutdown (like one-off `collect()`)
    /// should call this; long-lived owners (like Conversation) can just
    /// drop the last clone and rely on the best-effort cleanup in `Drop`.
    pub(crate) async fn shutdown(self) {
        let handle = self.inner.handle.lock().ok().and_then(|mut g| g.take());
        if let Some(h) = handle {
            let _ = h.stop().await;
        }
    }

    fn handle(&self) -> Option<ActorHandle> {
        self.inner.handle.lock().ok().and_then(|g| g.clone())
    }
}

impl Drop for StreamCollectorSessionInner {
    // Best-effort async stop. If we're inside a tokio runtime, spawn a
    // task to stop the actor so the broker prunes its subscription slot;
    // otherwise the runtime-level shutdown will reap it.
    fn drop(&mut self) {
        let handle = self.handle.lock().ok().and_then(|mut g| g.take());
        if let Some(h) = handle {
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move {
                    let _ = h.stop().await;
                });
            }
        }
    }
}

/// Build and start a long-lived `StreamCollector` actor subscribed to all
/// four streaming event types. The caller drives individual rounds via
/// [`run_stream_round`], which reuses this handle — and its subscriptions —
/// for every round of every turn.
pub(crate) async fn build_stream_collector(runtime: &ActonAI) -> StreamCollectorSession {
    let completion = Arc::new(Notify::new());
    let completion_signal = completion.clone();

    let result_container: Arc<std::sync::Mutex<Option<CollectorResultData>>> =
        Arc::new(std::sync::Mutex::new(None));
    let result_container_for_handler = result_container.clone();

    let mut actor_runtime = runtime.runtime().clone();
    let mut collector = actor_runtime.new_actor::<StreamCollector>();

    // Stream start — fire the caller's on_start callback for the current round.
    collector.mutate_on::<LLMStreamStart>(move |actor, envelope| {
        if actor.model.expected_correlation_id.as_ref() != Some(&envelope.message().correlation_id)
        {
            return Reply::ready();
        }
        if let (Some(callback), Some(context)) =
            (&actor.model.round.on_start, &actor.model.round.context)
        {
            if let Ok(mut f) = callback.lock() {
                f(context);
            }
        }
        Reply::ready()
    });

    // Stream token — accumulate, fire caller's callback, forward to target.
    collector.mutate_on::<LLMStreamToken>(move |actor, envelope| {
        if actor.model.expected_correlation_id.as_ref() != Some(&envelope.message().correlation_id)
        {
            return Reply::ready();
        }
        let token = envelope.message().token.clone();
        actor.model.buffer.push_str(&token);
        actor.model.token_count += 1;

        if let Some(ref callback) = actor.model.round.on_token {
            if let Ok(mut f) = callback.lock() {
                f(&token);
            }
        }

        if let Some(ref target) = actor.model.round.token_target {
            let target = target.clone();
            return Reply::pending(async move {
                target.send(StreamToken { text: token }).await;
            });
        }
        Reply::ready()
    });

    // Stream tool call — accumulate into per-round state.
    collector.mutate_on::<LLMStreamToolCall>(move |actor, envelope| {
        if actor.model.expected_correlation_id.as_ref() != Some(&envelope.message().correlation_id)
        {
            return Reply::ready();
        }
        actor
            .model
            .tool_calls
            .push(envelope.message().tool_call.clone());
        Reply::ready()
    });

    // Stream end — take the accumulated state into the shared result slot
    // and signal completion so the caller can pick up the round result.
    collector.mutate_on::<LLMStreamEnd>(move |actor, envelope| {
        if actor.model.expected_correlation_id.as_ref() != Some(&envelope.message().correlation_id)
        {
            return Reply::ready();
        }
        actor.model.stop_reason = Some(envelope.message().stop_reason);
        actor.model.usage = envelope.message().usage;
        if let (Some(callback), Some(context)) =
            (&actor.model.round.on_end, &actor.model.round.context)
        {
            if let Ok(mut f) = callback.lock() {
                f(context, envelope.message().stop_reason);
            }
        }

        if let Ok(mut container) = result_container_for_handler.lock() {
            *container = Some(CollectorResultData {
                buffer: std::mem::take(&mut actor.model.buffer),
                stop_reason: actor.model.stop_reason,
                token_count: actor.model.token_count,
                usage: actor.model.usage,
                tool_calls: std::mem::take(&mut actor.model.tool_calls),
                model: envelope.message().model.clone(),
            });
        }
        // Clear the correlation-ID filter and drop callbacks + target so
        // late stray events from the just-finished round don't land in
        // the next round's state.
        actor.model.expected_correlation_id = None;
        actor.model.round = StreamRoundCallbacks::default();

        completion_signal.notify_one();
        Reply::ready()
    });

    // Reset per-round state — reliably delivered BEFORE any event for the
    // new correlation_id because the provider is only told to send the
    // request after this message has been acknowledged. Installs the new
    // round's callbacks and token target so event handlers dispatch to
    // the right place.
    collector.mutate_on::<ResetStreamRound>(move |actor, envelope| {
        let msg = envelope.message();
        actor.model.buffer.clear();
        actor.model.token_count = 0;
        actor.model.usage = Usage::default();
        actor.model.stop_reason = None;
        actor.model.tool_calls.clear();
        actor.model.expected_correlation_id = Some(msg.expected_id.clone());
        actor.model.round = msg.callbacks.clone();
        Reply::ready()
    });

    // Subscribe BEFORE starting so no broadcast can slip past us.
    collector.handle().subscribe::<LLMStreamStart>().await;
    collector.handle().subscribe::<LLMStreamToken>().await;
    collector.handle().subscribe::<LLMStreamToolCall>().await;
    collector.handle().subscribe::<LLMStreamEnd>().await;

    let handle = collector.start().await;
    StreamCollectorSession {
        inner: Arc::new(StreamCollectorSessionInner {
            handle: std::sync::Mutex::new(Some(handle)),
            completion,
            result_container,
        }),
    }
}

/// Asks the accountant whether one more request to `provider` may go out.
///
/// Called before every provider dispatch, so a refusal costs nothing beyond
/// the round trip to a local actor. A denial names the scope that has to
/// change; unpriced usage is reported as the configuration problem it is,
/// naming the providers to price and the flag that accepts the blind spot.
async fn check_budget(accountant: &ActorHandle, provider: &str) -> Result<(), ActonAIError> {
    let decision = accountant
        .ask(CheckBudget::for_provider(provider))
        .await
        .map_err(|e| {
            ActonAIError::provider_error(format!(
                "could not reach the cost accountant to check the budget: {e}"
            ))
        })?;

    match decision {
        BudgetDecision::Allowed => Ok(()),
        BudgetDecision::Denied {
            scope,
            limit_microusd,
            spent_microusd,
        } => Err(ActonAIError::budget_exceeded(
            scope,
            limit_microusd,
            spent_microusd,
        )),
        BudgetDecision::Unpriced { providers } => Err(ActonAIError::configuration(
            "budget",
            format!(
                "a budget is set but provider(s) {} reported usage with no configured pricing, \
                 so the cap cannot be trusted; add [providers.<name>.pricing] for them, or \
                 accept the blind spot with Budget::allow_unpriced() / `allow_unpriced = true` \
                 under [budget]",
                providers
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )),
    }
}

/// Run a single stream round on the supplied long-lived collector.
///
/// Installs `callbacks` for this round before firing the request so the
/// collector's event handlers dispatch to the right hooks, then waits on
/// the completion notify for the round's final result.
pub(crate) async fn run_stream_round(
    session: &StreamCollectorSession,
    provider_handle: &ActorHandle,
    request: &LLMRequest,
    correlation_id: CorrelationId,
    callbacks: StreamRoundCallbacks,
) -> Result<RoundResult, ActonAIError> {
    // Resolve the collector's live actor handle. Returns an error if the
    // session was already shut down — defensive, but shouldn't happen on
    // any normal path.
    let handle = session.handle().ok_or_else(|| {
        ActonAIError::prompt_failed("stream collector session is shut down".to_string())
    })?;

    // Ack: reset state and install the new round's correlation ID + callbacks.
    handle
        .send(ResetStreamRound {
            expected_id: correlation_id,
            callbacks,
        })
        .await;

    // Fire the request.
    provider_handle.send(request.clone()).await;

    // Wait for the stream-end handler to fill the result slot.
    session.inner.completion.notified().await;

    let result = session
        .inner
        .result_container
        .lock()
        .ok()
        .and_then(|mut guard| guard.take())
        .ok_or_else(|| {
            ActonAIError::prompt_failed("failed to retrieve collected stream data".to_string())
        })?;

    Ok(RoundResult {
        text: result.buffer,
        stop_reason: result.stop_reason.unwrap_or(StopReason::EndTurn),
        token_count: result.token_count,
        usage: result.usage,
        tool_calls: result.tool_calls,
        model: result.model,
    })
}

/// What one provider dispatch produced.
///
/// A struct rather than a tuple because `model` is the sixth field and the
/// three `String`/`usize` positions were already easy to transpose.
pub(crate) struct RoundResult {
    /// Text accumulated from the round's tokens.
    pub(crate) text: String,
    /// Why the provider stopped.
    pub(crate) stop_reason: StopReason,
    /// How many tokens the caller's stream saw.
    pub(crate) token_count: usize,
    /// Usage the provider reported for the round.
    pub(crate) usage: Usage,
    /// Tool calls the round emitted.
    pub(crate) tool_calls: Vec<ToolCall>,
    /// The model that actually served the round.
    pub(crate) model: String,
}

/// Render a successful tool result as a single-line preview for the
/// [`LLMStreamToolResult`] broadcast. Picks a salient string field when the
/// payload is a JSON object (`output`, `stdout`, `content`, `text`,
/// `result`), otherwise falls back to a compact JSON rendering. Newlines
/// are flattened so the preview stays on one line; oversized strings are
/// truncated with an ellipsis.
fn summarize_tool_value(value: &serde_json::Value, max: usize) -> String {
    if let Some(obj) = value.as_object() {
        for key in ["output", "stdout", "content", "text", "result"] {
            if let Some(v) = obj.get(key).and_then(|v| v.as_str()) {
                return flatten_and_truncate(v, max);
            }
        }
    }
    if let Some(s) = value.as_str() {
        return flatten_and_truncate(s, max);
    }
    let rendered = serde_json::to_string(value).unwrap_or_else(|_| "<result>".to_string());
    flatten_and_truncate(&rendered, max)
}

/// Render a tool error message as a compact single-line preview.
fn summarize_error(msg: &str, max: usize) -> String {
    flatten_and_truncate(msg, max)
}

fn flatten_and_truncate(s: &str, max: usize) -> String {
    let flat: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    if flat.chars().count() <= max {
        flat
    } else {
        let cut: String = flat.chars().take(max).collect();
        format!("{cut}…")
    }
}

/// What happened to one tool call, in the form the next round needs it.
///
/// The loop cannot use a plain `Result` here because refusal is a third
/// thing: not a value the tool returned, and not a failure it suffered. The
/// model is told about all three, but in different words.
#[derive(Debug)]
enum ToolOutcome {
    /// The tool ran, successfully or not.
    Ran(Result<serde_json::Value, ToolError>),
    /// The gate refused the call, with the text the model reads.
    Refused(String),
}

impl ToolOutcome {
    /// The string that goes back to the model as this call's tool result.
    fn as_tool_result(&self) -> String {
        match self {
            Self::Ran(Ok(value)) => serde_json::to_string(value).unwrap_or_default(),
            Self::Ran(Err(error)) => format!("Error: {error}"),
            Self::Refused(feedback) => feedback.clone(),
        }
    }
}

/// Everything one tool call needs that does not change between calls.
///
/// Assembled per call from borrows the round loop already holds; nothing here
/// is cloned. It exists so [`Self::run`] can carry the whole gate → execute →
/// record sequence without adding eight more locals to a loop body that is
/// already long.
struct ToolStep<'a> {
    /// The provider actor, used only as a broadcast source.
    provider_handle: &'a ActorHandle,
    /// The turn's span, parent of this call's tool span.
    turn: &'a crate::telemetry::spans::TurnSpan,
    /// The turn being served.
    turn_id: &'a TurnId,
    /// The correlation ID of the round that asked for this call.
    correlation_id: &'a CorrelationId,
    /// The conversation this turn belongs to, when it belongs to one.
    conversation_id: Option<&'a crate::types::ConversationId>,
    /// The gate, when one is configured.
    policy: Option<&'a crate::policy::ToolPolicy>,
    /// The audit log and its settings, when a trail is configured.
    audit: Option<&'a (ActorHandle, crate::audit::AuditConfig)>,
    /// Whether the call runs inside a turn resumed from a checkpoint.
    ///
    /// Stamped onto the audit entry, so the trail distinguishes a first-run
    /// execution from one performed by a restarted process picking up an
    /// interrupted turn.
    resumed: bool,
}

impl ToolStep<'_> {
    /// Gates one tool call, runs it if admitted, and records what happened.
    ///
    /// Returns the outcome the round loop feeds back to the model and the
    /// record it keeps for the caller. Every path through this function
    /// produces both — a refused call and a failed call are as reportable as
    /// a successful one, and an audit trail with holes in it is not evidence
    /// of anything.
    async fn run(
        &self,
        tools: &mut [ToolSpec],
        counts: &mut crate::policy::TurnCounts,
        tool_call: &ToolCall,
    ) -> (ToolOutcome, ExecutedToolCall) {
        let started = std::time::Instant::now();

        // The bracket opens before the gate is consulted, so a call the gate
        // goes on to refuse is announced exactly like one that runs. This is
        // the doctrine `TurnLifecycle` documents: a consumer mapping these
        // onto a protocol where a result may only follow an announced call
        // must never have to synthesize a start it never saw. The cost is
        // that a refused tool appears in an observer's status line for as
        // long as the gate deliberates, which is the honest picture — the
        // call was proposed, and something had to decide about it.
        self.start_tool(tool_call).await;

        let gate = self.decide(tool_call, counts).await;

        let (arguments, decided_by) = match gate {
            crate::policy::GateOutcome::Deny { reason, decided_by } => {
                return self.refuse(tool_call, &reason, decided_by, started).await;
            }
            crate::policy::GateOutcome::Allow {
                arguments,
                decided_by,
            } => (arguments, decided_by),
        };

        // Recorded only once the call is admitted, so a refusal does not
        // consume the very budget it was refused against.
        counts.record(&tool_call.name);

        self.execute(tools, tool_call, arguments, decided_by, started)
            .await
    }

    /// Opens the observation bracket for one call.
    ///
    /// Carries the arguments the *model* proposed, not the ones that will
    /// run: at this point the gate has not been consulted, so a hook has had
    /// no chance to rewrite them and the proposed set is the only one that
    /// exists. What actually ran is the audit trail's business.
    ///
    /// When a trail is configured, the arguments pass through its redactor
    /// first, exactly as the trail's own entries do: this broadcast fans out
    /// to every [`TurnLifecycle`] subscriber's mailbox — the introspection
    /// actor, an embedder's lifecycle forwarder — and those mailboxes are
    /// precisely the places the redaction config promises a secret never
    /// reaches. Publishing raw here would keep the secret out of the audit
    /// file while delivering it to every UI and log downstream of the
    /// broker.
    async fn start_tool(&self, tool_call: &ToolCall) {
        let arguments = match self.audit {
            Some((_, config)) => config.redactor().redact(&tool_call.arguments),
            None => tool_call.arguments.clone(),
        };
        self.provider_handle
            .broadcast(TurnLifecycle::ToolStarted {
                turn_id: self.turn_id.clone(),
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                arguments,
            })
            .await;
    }

    /// Closes the bracket [`Self::start_tool`] opened, however the call ended.
    ///
    /// Both events carry the same verdict and the same text, so an observer
    /// watching only one of them reaches the same conclusion. Callers settle
    /// `success` and `summary` before calling this, which is what makes that
    /// possible.
    async fn finish_tool(&self, tool_call: &ToolCall, success: bool, summary: String) {
        self.provider_handle
            .broadcast(TurnLifecycle::ToolFinished {
                turn_id: self.turn_id.clone(),
                tool_call_id: tool_call.id.clone(),
                success,
                summary: summary.clone(),
            })
            .await;

        self.provider_handle
            .broadcast(crate::messages::LLMStreamToolResult {
                correlation_id: self.correlation_id.clone(),
                turn_id: self.turn_id.clone(),
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                success,
                summary,
            })
            .await;
    }

    /// Applies the policy, or admits everything when there is none.
    async fn decide(
        &self,
        tool_call: &ToolCall,
        counts: &crate::policy::TurnCounts,
    ) -> crate::policy::GateOutcome {
        let Some(policy) = self.policy else {
            return crate::policy::GateOutcome::Allow {
                arguments: tool_call.arguments.clone(),
                decided_by: crate::policy::Decider::NoPolicy,
            };
        };

        policy
            .decide(
                crate::policy::ToolInvocation {
                    tool_name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                    tool_call_id: tool_call.id.clone(),
                    correlation_id: self.correlation_id.clone(),
                    turn_id: self.turn_id.clone(),
                },
                counts,
            )
            .await
    }

    /// Handles a call the gate refused.
    async fn refuse(
        &self,
        tool_call: &ToolCall,
        reason: &crate::policy::DenialReason,
        decided_by: crate::policy::Decider,
        started: std::time::Instant,
    ) -> (ToolOutcome, ExecutedToolCall) {
        tracing::info!(
            tool = %tool_call.name,
            decided_by = %decided_by,
            %reason,
            "tool call refused by policy",
        );

        // Closes the bracket `run` opened before the gate was consulted. A
        // refusal is something an operator watching a chat session needs to
        // see, and `success: false` is exactly what the REPL already renders
        // as a failed call.
        let summary = summarize_error(&reason.to_string(), 200);
        self.finish_tool(tool_call, false, summary).await;

        self.record(
            tool_call,
            &tool_call.arguments,
            crate::audit::AuditOutcome::Denied {
                reason: reason.to_string(),
            },
            crate::audit::AuditDecision::refused(decided_by),
            started,
        )
        .await;

        // `ExecutedToolCall` has no refusal variant and public fields, so
        // adding one would break every caller that builds it by literal. The
        // caller-facing record says the call did not happen and why; the
        // structured decision lives in the audit entry, which is the artifact
        // that has to be precise about it.
        let executed = ExecutedToolCall::error(
            &tool_call.id,
            &tool_call.name,
            tool_call.arguments.clone(),
            format!("denied by policy: {reason}"),
        );

        (
            ToolOutcome::Refused(crate::policy::denial_feedback(reason)),
            executed,
        )
    }

    /// Runs a call the gate admitted, with the arguments the gate settled on.
    async fn execute(
        &self,
        tools: &mut [ToolSpec],
        tool_call: &ToolCall,
        arguments: serde_json::Value,
        decided_by: crate::policy::Decider,
        started: std::time::Instant,
    ) -> (ToolOutcome, ExecutedToolCall) {
        // Child of the turn, not of the round: the tool runs after its round
        // has closed, and nesting it under a finished span would misreport
        // when it ran.
        let tool_span = self.turn.tool(&tool_call.name);

        // Kept only when there is a trail to write it to: the entry must
        // describe the call that actually ran, and a hook may have rewritten
        // it into something the model never proposed.
        let recorded_arguments = self.audit.map(|_| arguments.clone());

        let result = execute_tool_with_callback(tools, tool_call, arguments).await;

        // Settled before the bracket closes, because `finish_tool` puts the
        // same verdict and the same text on both events it publishes: an
        // observer watching only one of them has to reach the same conclusion
        // as one watching the other.
        let (success, summary) = match &result {
            Ok(value) => (true, summarize_tool_value(value, 200)),
            Err(e) => (false, summarize_error(&e.to_string(), 200)),
        };
        self.finish_tool(tool_call, success, summary.clone()).await;

        let tool_outcome = if result.is_ok() {
            crate::telemetry::metrics::OUTCOME_OK
        } else {
            crate::telemetry::metrics::OUTCOME_ERROR
        };
        crate::telemetry::metrics::record_tool_duration(
            &tool_call.name,
            tool_outcome,
            started.elapsed().as_secs_f64(),
        );
        // Name and outcome only. The arguments and the result are user data
        // of unbounded size and are never recorded — see
        // `crate::telemetry::spans`.
        tool_span.finish(tool_outcome);

        let audit_outcome = match (&result, self.audit) {
            // Summarized from a redacted copy of the result, not the raw one.
            // A tool that echoes part of its input back — and plenty do —
            // would otherwise write into the file the very secret that was
            // stripped out of the arguments one field earlier.
            (Ok(value), Some((_, config))) => crate::audit::AuditOutcome::Success {
                summary: summarize_tool_value(&config.redactor().redact(value), 200),
            },
            (Ok(_), None) => crate::audit::AuditOutcome::Success { summary },
            // An error is prose, not structured data, so key-based redaction
            // has nothing to match on. It is bounded, and it is the tool's own
            // words about its own failure.
            (Err(_), _) => crate::audit::AuditOutcome::Error { message: summary },
        };
        self.record(
            tool_call,
            recorded_arguments.as_ref().unwrap_or(&tool_call.arguments),
            audit_outcome,
            crate::audit::AuditDecision::approved(decided_by),
            started,
        )
        .await;

        let executed = match &result {
            Ok(value) => ExecutedToolCall::success(
                &tool_call.id,
                &tool_call.name,
                tool_call.arguments.clone(),
                value.clone(),
            ),
            Err(e) => ExecutedToolCall::error(
                &tool_call.id,
                &tool_call.name,
                tool_call.arguments.clone(),
                e.to_string(),
            ),
        };

        (ToolOutcome::Ran(result), executed)
    }

    /// Writes the settlement's verdict on a call it declined to re-run.
    ///
    /// A started, non-idempotent call from an interrupted round may or may
    /// not have had its effect, and the decision not to run it again is
    /// itself an auditable event. The first attempt crashed before it could
    /// record anything, so without this entry the trail would hold no trace
    /// of a call the response's `tool_calls` reports — on the one path whose
    /// stated purpose is making crash recovery auditable.
    async fn record_uncertain(&self, tool_call: &ToolCall, feedback: &str) {
        self.record(
            tool_call,
            &tool_call.arguments,
            crate::audit::AuditOutcome::Uncertain {
                message: feedback.to_string(),
            },
            crate::audit::AuditDecision::refused(crate::policy::Decider::Settlement),
            std::time::Instant::now(),
        )
        .await;
    }

    /// Appends one entry to the trail, if a trail is configured.
    ///
    /// Fire-and-forget by design: the entry is `send`, not `ask`, so the turn
    /// is never blocked waiting on a disk. Ordering still holds — the audit
    /// actor's mailbox is FIFO and its handler is sequential — which is what
    /// the hash chain needs and all it needs.
    async fn record(
        &self,
        tool_call: &ToolCall,
        arguments: &serde_json::Value,
        outcome: crate::audit::AuditOutcome,
        decision: crate::audit::AuditDecision,
        started: std::time::Instant,
    ) {
        let Some((handle, config)) = self.audit else {
            return;
        };

        let record = crate::audit::InvocationRecord {
            timestamp: chrono::Utc::now().to_rfc3339(),
            correlation_id: self.correlation_id.clone(),
            conversation_id: self.conversation_id.cloned(),
            turn_id: self.turn_id.clone(),
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            // Redacted here, at the boundary, so a secret never reaches the
            // actor's mailbox — let alone its file.
            arguments: config.redactor().redact(arguments),
            outcome,
            decision,
            duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            resumed: self.resumed,
        };

        handle
            .send(crate::audit::RecordInvocation::new(record))
            .await;
    }
}

/// Executes a single tool call and invokes the result callback if present.
///
/// `arguments` is passed separately rather than read off `tool_call` because
/// an approval hook may rewrite them — the call the model asked for and the
/// call that runs are not always the same, and the difference is exactly what
/// the gate exists to express.
async fn execute_tool_with_callback(
    tools: &mut [ToolSpec],
    tool_call: &ToolCall,
    arguments: serde_json::Value,
) -> Result<serde_json::Value, ToolError> {
    let Some(spec) = tools
        .iter_mut()
        .find(|spec| spec.definition.name == tool_call.name)
    else {
        return Err(ToolError::not_found(&tool_call.name));
    };

    // `SyncFuture` because `ToolFuture` is a `dyn Future + Send` without
    // `Sync`, and a `!Sync` future held across this await would make the
    // whole turn future `!Sync` — which would force embedders back onto a
    // spawned task instead of `Reply::pending`. The wrapper is sound because
    // a future being polled is never shared, and it costs nothing at runtime.
    let result = sync_wrapper::SyncFuture::new(spec.executor.call(arguments)).await;

    // Invoke the result callback if present. `get_mut` rather than `lock`:
    // we hold `&mut`, so the Mutex is only satisfying `Sync`, not guarding
    // anything — and a poisoned lock (a callback that panicked earlier)
    // simply skips the callback rather than failing the tool.
    if let Some(callback) = spec.on_result.as_mut() {
        if let Ok(callback) = callback.get_mut() {
            match &result {
                Ok(value) => callback(Ok(value)),
                Err(e) => {
                    let error_str = e.to_string();
                    callback(Err(&error_str));
                }
            }
        }
    }

    result
}

/// Internal actor for collecting stream tokens.
///
/// This actor owns all state for collecting streaming responses plus the
/// caller-supplied per-round callbacks. Handlers read callbacks from
/// `actor.model`; the caller updates them per round via
/// [`ResetStreamRound`]. No external Mutex-protected shared state is
/// needed.
///
/// The collector is **long-lived across every round of every turn it
/// serves** — subscribing once at construction and swapping per-round
/// callbacks + correlation filter through [`ResetStreamRound`]. Spawning
/// a fresh collector per round leaks broker subscriptions because
/// `acton-reactive`'s `UnsubscribeBroker` ships as a no-op (see
/// `common/src/message/unsubscribe_broker.rs`), so every stopped
/// subscriber leaves a dead channel that the broker still broadcasts to.
#[acton_actor]
struct StreamCollector {
    /// Accumulated response buffer for the current round
    buffer: String,
    /// Count of tokens received in the current round
    token_count: usize,
    /// Provider-reported usage for the current round, taken from the round's
    /// terminal `LLMStreamEnd`.
    usage: Usage,
    /// Stop reason when the current round's stream ends
    stop_reason: Option<StopReason>,
    /// Accumulated tool calls from the current round
    tool_calls: Vec<ToolCall>,
    /// Correlation ID of the round currently being collected. Handlers
    /// ignore any event whose correlation ID doesn't match — protects the
    /// collector from stray events emitted by other concurrent streams
    /// that the provider may broadcast to the same broker channel.
    expected_correlation_id: Option<CorrelationId>,
    /// Caller-supplied callbacks + token target for the current round.
    /// Swapped in at the start of each round by [`ResetStreamRound`].
    round: StreamRoundCallbacks,
}

/// Per-round reset message. Sent to the collector before starting each
/// round's LLM request: clears accumulated state, installs the new
/// correlation-ID filter, and swaps in the new round's callbacks +
/// token target so the event handlers dispatch to the correct caller.
#[derive(Clone, Debug)]
struct ResetStreamRound {
    expected_id: CorrelationId,
    callbacks: StreamRoundCallbacks,
}

/// Collected stream data returned from the actor.
#[derive(Debug, Clone, Default)]
struct CollectorResultData {
    /// Accumulated text from tokens
    buffer: String,
    /// Reason the stream stopped
    stop_reason: Option<StopReason>,
    /// Number of tokens received
    token_count: usize,
    /// Provider-reported usage for the round
    usage: Usage,
    /// Tool calls received during streaming
    tool_calls: Vec<ToolCall>,
    /// The model the provider reported as having served this round.
    ///
    /// Not the configured model when a rate limit degraded the round onto a
    /// `fallback_model`, which is exactly why it is carried out of the
    /// terminal event rather than read back from configuration.
    model: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_turn_gets_its_own_span_outcome() {
        let refused = outcome_for(&ActonAIError::turns_not_admitted(
            crate::introspection::AdmissionState::Paused,
        ));

        // A team draining for a deploy filters on this label. Sharing one with
        // a real failure would make every clean rollout look like an incident.
        assert_eq!(refused, "turns_not_admitted");
        assert_ne!(refused, outcome_for(&ActonAIError::prompt_failed("boom")));
        assert_ne!(
            refused,
            outcome_for(&ActonAIError::provider_error("upstream 500"))
        );
    }

    #[test]
    fn a_draining_refusal_shares_the_paused_outcome() {
        // The two differ for an operator reading `status`, not for a span:
        // both mean "we chose not to start this", and splitting the label
        // would fragment the very query the label exists for.
        assert_eq!(
            outcome_for(&ActonAIError::turns_not_admitted(
                crate::introspection::AdmissionState::Draining
            )),
            outcome_for(&ActonAIError::turns_not_admitted(
                crate::introspection::AdmissionState::Paused
            ))
        );
    }

    #[test]
    fn tool_spec_debug_impl() {
        let spec = ToolSpec {
            definition: ToolDefinition {
                idempotent: false,
                name: "test".to_string(),
                description: "Test tool".to_string(),
                input_schema: serde_json::json!({}),
            },
            executor: Arc::new(ClosureToolExecutor {
                func: |_args: serde_json::Value| async { Ok(serde_json::json!({})) },
            }),
            on_result: None,
        };

        let debug = format!("{:?}", spec);
        assert!(debug.contains("test"));
    }

    #[test]
    fn tool_spec_clone() {
        let spec = ToolSpec {
            definition: ToolDefinition {
                idempotent: false,
                name: "test".to_string(),
                description: "Test tool".to_string(),
                input_schema: serde_json::json!({}),
            },
            executor: Arc::new(ClosureToolExecutor {
                func: |_args: serde_json::Value| async { Ok(serde_json::json!({})) },
            }),
            on_result: Some(std::sync::Mutex::new(Box::new(|_result| {}))),
        };

        let cloned = spec.clone();
        assert_eq!(cloned.definition.name, "test");
        // Callbacks are not cloned
        assert!(cloned.on_result.is_none());
    }

    #[test]
    fn collected_response_new_creates_correctly() {
        let response = CollectedResponse::new("Hello world".to_string(), StopReason::EndTurn, 2);

        assert_eq!(response.text, "Hello world");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.token_count, 2);
        assert!(response.tool_calls.is_empty());
    }

    #[test]
    fn collected_response_is_complete() {
        let complete = CollectedResponse::new("test".to_string(), StopReason::EndTurn, 1);
        assert!(complete.is_complete());

        let incomplete = CollectedResponse::new("test".to_string(), StopReason::MaxTokens, 1);
        assert!(!incomplete.is_complete());
    }

    #[test]
    fn collected_response_is_truncated() {
        let truncated = CollectedResponse::new("test".to_string(), StopReason::MaxTokens, 1);
        assert!(truncated.is_truncated());

        let complete = CollectedResponse::new("test".to_string(), StopReason::EndTurn, 1);
        assert!(!complete.is_truncated());
    }
}
