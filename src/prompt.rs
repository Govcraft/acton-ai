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

use crate::conversation::StreamToken;
use crate::error::ActonAIError;
use crate::facade::ActonAI;
use crate::llm::SamplingParams;
use crate::messages::{
    LLMRequest, LLMStreamEnd, LLMStreamStart, LLMStreamToken, LLMStreamToolCall, Message,
    StopReason, ToolCall, ToolDefinition,
};
use crate::stream::{CollectedResponse, ExecutedToolCall};
use crate::tools::ToolError;
use crate::types::{AgentId, CorrelationId};
use acton_reactive::prelude::*;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Notify;

/// Framework fallback for the agentic tool-call loop.
///
/// Used when neither a `[defaults]` TOML block nor a builder override
/// supplies `max_tool_rounds`. Per-prompt calls to
/// [`PromptBuilder::max_tool_rounds`] still win over this value.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 10;

/// Type alias for start callbacks.
type StartCallback = Box<dyn FnMut() + Send + 'static>;

/// Type alias for token callbacks.
type TokenCallback = Box<dyn FnMut(&str) + Send + 'static>;

/// Type alias for end callbacks.
type EndCallback = Box<dyn FnMut(StopReason) + Send + 'static>;

/// Type alias for tool result callbacks.
///
/// Called after a tool executes with the result (success or error).
type ToolResultCallback = Box<dyn FnMut(Result<&serde_json::Value, &str>) + Send + 'static>;

/// Type alias for tool execution futures.
type ToolFuture = Pin<Box<dyn Future<Output = Result<serde_json::Value, ToolError>> + Send>>;

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

/// Adapter to wrap built-in tool executors as `ToolExecutorFn`.
///
/// When `sandbox` is `Some`, tool invocations are dispatched through the
/// sandbox factory's `Sandbox::execute` path. Otherwise the in-process
/// executor runs the tool directly. This is how facade-configured
/// `ProcessSandbox` actually reaches the bash/write_file/edit_file call
/// sites — prior to this wiring the factory existed in memory but the
/// prompt path skipped it entirely.
struct BuiltinToolExecutorAdapter {
    tool_name: String,
    executor: Arc<crate::tools::BoxedToolExecutor>,
    sandbox: Option<Arc<dyn crate::tools::sandbox::SandboxFactory>>,
}

impl ToolExecutorFn for BuiltinToolExecutorAdapter {
    fn call(&self, args: serde_json::Value) -> ToolFuture {
        match self.sandbox.clone() {
            Some(factory) => {
                let name = self.tool_name.clone();
                Box::pin(async move {
                    let mut sandbox = factory.create().await?;
                    let result = sandbox.execute(&name, args).await;
                    sandbox.destroy();
                    result
                })
            }
            None => {
                let executor = Arc::clone(&self.executor);
                Box::pin(async move { executor.execute(args).await })
            }
        }
    }
}

/// A tool specification combining definition, executor, and optional result callback.
pub struct ToolSpec {
    /// The tool definition sent to the LLM
    pub definition: ToolDefinition,
    /// The executor for this tool
    executor: Arc<dyn ToolExecutorFn>,
    /// Optional callback invoked when the tool returns a result
    on_result: Option<ToolResultCallback>,
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
pub struct PromptBuilder {
    /// The ActonAI runtime (cheaply cloned via Arc)
    runtime: ActonAI,
    /// The user's prompt content
    user_content: String,
    /// Optional system prompt
    system_prompt: Option<String>,
    /// Optional conversation history (replaces user_content when set)
    conversation_history: Option<Vec<Message>>,
    /// Callback for stream start
    on_start: Option<StartCallback>,
    /// Callback for each token
    on_token: Option<TokenCallback>,
    /// Callback for stream end
    on_end: Option<EndCallback>,
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
    /// This is useful for displaying a "thinking" indicator or spinner.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime
    ///     .prompt("Hello")
    ///     .on_start(|| println!("Thinking..."))
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn on_start<F>(mut self, f: F) -> Self
    where
        F: FnMut() + Send + 'static,
    {
        self.on_start = Some(Box::new(f));
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
        self.on_token = Some(Box::new(f));
        self
    }

    /// Sets a callback to be called when the stream ends.
    ///
    /// The callback receives the stop reason indicating why the LLM
    /// stopped generating.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime
    ///     .prompt("Hello")
    ///     .on_end(|reason| println!("\n[Finished: {reason:?}]"))
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn on_end<F>(mut self, f: F) -> Self
    where
        F: FnMut(StopReason) + Send + 'static,
    {
        self.on_end = Some(Box::new(f));
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
            on_result: Some(Box::new(on_result)),
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
            let factory = self.runtime.sandbox_factory().cloned();
            for (name, config) in builtins.configs() {
                if let Some(executor) = builtins.get_executor(name) {
                    // Only sandboxed tools route through the factory. Non-
                    // sandboxed tools (e.g. `calculate`, `read_file`) skip
                    // the subprocess roundtrip even when a factory exists.
                    let sandbox = if config.sandboxed {
                        factory.clone()
                    } else {
                        None
                    };
                    let adapter = BuiltinToolExecutorAdapter {
                        tool_name: name.clone(),
                        executor,
                        sandbox,
                    };
                    self.tools.push(ToolSpec {
                        definition: config.definition.clone(),
                        executor: Arc::new(adapter),
                        on_result: None,
                    });
                }
            }
        }
        self
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

        // Collect tool definitions
        let tool_definitions: Vec<ToolDefinition> =
            tools.iter().map(|t| t.definition.clone()).collect();
        let has_tools = !tool_definitions.is_empty();

        // Track executed tool calls and total tokens
        let mut executed_tool_calls = Vec::new();
        let mut total_token_count = 0;
        let mut final_text;
        let mut rounds = 0;

        // Wrap callbacks in Arc<Mutex> for sharing across multiple rounds
        let on_start: Option<WrappedStartCallback> =
            on_start.map(|f| Arc::new(std::sync::Mutex::new(f)));
        let on_token: Option<WrappedTokenCallback> =
            on_token.map(|f| Arc::new(std::sync::Mutex::new(f)));
        let on_end: Option<WrappedEndCallback> = on_end.map(|f| Arc::new(std::sync::Mutex::new(f)));

        // Build one long-lived StreamCollector for this entire `collect()`
        // call. Spawning a fresh collector per round leaks broker
        // subscriptions — `UnsubscribeBroker` ships unimplemented in
        // acton-reactive, so the only reliable way to avoid dead-channel
        // spam is to reuse a single subscriber across rounds.
        let collector_session = build_stream_collector(
            &runtime,
            &StreamRoundCallbacks {
                on_start: on_start.clone(),
                on_token: on_token.clone(),
                on_end: on_end.clone(),
                token_target: token_target.clone(),
            },
        )
        .await;

        loop {
            rounds += 1;
            if rounds > max_tool_rounds {
                collector_session.shutdown().await;
                return Err(ActonAIError::prompt_failed(format!(
                    "exceeded maximum tool rounds ({max_tool_rounds})",
                )));
            }

            // Generate new IDs for this round
            let correlation_id = CorrelationId::new();
            let agent_id = AgentId::new();

            // Create the request
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
            };

            // Collect stream response — reuses the long-lived collector.
            // Keep a clone so we can tag tool-result broadcasts with the
            // round's correlation ID further down.
            let round_correlation_id = correlation_id.clone();
            let (text, stop_reason, token_count, tool_calls) = match run_stream_round(
                &collector_session,
                &provider_handle,
                &request,
                correlation_id,
            )
            .await
            {
                Ok(data) => data,
                Err(e) => {
                    collector_session.shutdown().await;
                    return Err(e);
                }
            };

            final_text = text.clone();
            total_token_count += token_count;

            match stop_reason {
                StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence => {
                    // Conversation complete
                    break;
                }
                StopReason::ToolUse => {
                    if tool_calls.is_empty() {
                        // No tool calls but ToolUse stop reason - treat as complete
                        break;
                    }

                    // Execute tools and continue
                    let mut tool_results = Vec::new();
                    for tool_call in &tool_calls {
                        let result = execute_tool_with_callback(&mut tools, tool_call).await;

                        // Broadcast a compact result event so observers
                        // (the CLI chat REPL) can render success/failure
                        // inline alongside the preceding tool-call line.
                        let (success, summary) = match &result {
                            Ok(value) => (true, summarize_tool_value(value, 200)),
                            Err(e) => (false, summarize_error(&e.to_string(), 200)),
                        };
                        provider_handle
                            .broadcast(crate::messages::LLMStreamToolResult {
                                correlation_id: round_correlation_id.clone(),
                                tool_call_id: tool_call.id.clone(),
                                tool_name: tool_call.name.clone(),
                                success,
                                summary,
                            })
                            .await;

                        // Record the executed tool call
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
                        executed_tool_calls.push(executed);
                        tool_results.push(result);
                    }

                    // Add assistant message with tool calls to conversation
                    messages.push(Message::assistant_with_tools(text, tool_calls.clone()));

                    // Add tool result messages
                    for (tool_call, result) in tool_calls.iter().zip(tool_results.iter()) {
                        let result_str = match result {
                            Ok(v) => serde_json::to_string(v).unwrap_or_default(),
                            Err(e) => format!("Error: {e}"),
                        };
                        messages.push(Message::tool(&tool_call.id, result_str));
                    }
                }
            }
        }

        // One shutdown for the long-lived collector — no broker
        // subscriptions leaked across rounds.
        collector_session.shutdown().await;

        Ok(CollectedResponse::with_tool_calls(
            final_text,
            StopReason::EndTurn,
            total_token_count,
            executed_tool_calls,
        ))
    }
}

/// Callbacks and token target shared across every round of a single
/// `collect()` call.
struct StreamRoundCallbacks {
    on_start: Option<WrappedStartCallback>,
    on_token: Option<WrappedTokenCallback>,
    on_end: Option<WrappedEndCallback>,
    token_target: Option<ActorHandle>,
}

/// Long-lived stream collector. Owns broker subscriptions for the entire
/// `collect()` call; each tool round sends a [`ResetStreamRound`] to reset
/// state and swap the correlation-ID filter before firing the next request.
struct StreamCollectorSession {
    handle: ActorHandle,
    completion: Arc<Notify>,
    result_container: Arc<std::sync::Mutex<Option<CollectorResultData>>>,
}

impl StreamCollectorSession {
    /// Stop the underlying actor. Call once after all rounds complete.
    async fn shutdown(self) {
        let _ = self.handle.stop().await;
    }
}

/// Build and start a long-lived `StreamCollector` actor subscribed to all
/// four streaming event types. The caller drives individual rounds via
/// [`run_stream_round`], which reuses this handle for every round.
async fn build_stream_collector(
    runtime: &ActonAI,
    callbacks: &StreamRoundCallbacks,
) -> StreamCollectorSession {
    let completion = Arc::new(Notify::new());
    let completion_signal = completion.clone();

    let result_container: Arc<std::sync::Mutex<Option<CollectorResultData>>> =
        Arc::new(std::sync::Mutex::new(None));
    let result_container_for_handler = result_container.clone();

    let mut actor_runtime = runtime.runtime().clone();
    let mut collector = actor_runtime.new_actor::<StreamCollector>();

    // Stream start — fire the caller's on_start callback for our round.
    let on_start_clone = callbacks.on_start.clone();
    collector.mutate_on::<LLMStreamStart>(move |actor, envelope| {
        if actor.model.expected_correlation_id.as_ref()
            != Some(&envelope.message().correlation_id)
        {
            return Reply::ready();
        }
        if let Some(ref callback) = on_start_clone {
            if let Ok(mut f) = callback.lock() {
                f();
            }
        }
        Reply::ready()
    });

    // Stream token — accumulate, fire caller's callback, forward to target.
    let on_token_clone = callbacks.on_token.clone();
    let token_target_clone = callbacks.token_target.clone();
    collector.mutate_on::<LLMStreamToken>(move |actor, envelope| {
        if actor.model.expected_correlation_id.as_ref()
            != Some(&envelope.message().correlation_id)
        {
            return Reply::ready();
        }
        let token = envelope.message().token.clone();
        actor.model.buffer.push_str(&token);
        actor.model.token_count += 1;

        if let Some(ref callback) = on_token_clone {
            if let Ok(mut f) = callback.lock() {
                f(&token);
            }
        }

        if let Some(ref target) = token_target_clone {
            let target = target.clone();
            return Reply::pending(async move {
                target.send(StreamToken { text: token }).await;
            });
        }
        Reply::ready()
    });

    // Stream tool call — accumulate into per-round state.
    collector.mutate_on::<LLMStreamToolCall>(move |actor, envelope| {
        if actor.model.expected_correlation_id.as_ref()
            != Some(&envelope.message().correlation_id)
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
    let on_end_clone = callbacks.on_end.clone();
    collector.mutate_on::<LLMStreamEnd>(move |actor, envelope| {
        if actor.model.expected_correlation_id.as_ref()
            != Some(&envelope.message().correlation_id)
        {
            return Reply::ready();
        }
        actor.model.stop_reason = Some(envelope.message().stop_reason);
        if let Some(ref callback) = on_end_clone {
            if let Ok(mut f) = callback.lock() {
                f(envelope.message().stop_reason);
            }
        }

        if let Ok(mut container) = result_container_for_handler.lock() {
            *container = Some(CollectorResultData {
                buffer: std::mem::take(&mut actor.model.buffer),
                stop_reason: actor.model.stop_reason,
                token_count: actor.model.token_count,
                tool_calls: std::mem::take(&mut actor.model.tool_calls),
            });
        }
        // Clear the correlation-ID filter so stray late events from the
        // just-finished round don't land in the next round's state.
        actor.model.expected_correlation_id = None;

        completion_signal.notify_one();
        Reply::ready()
    });

    // Reset per-round state — reliably delivered BEFORE any event for the
    // new correlation_id because the provider is only told to send the
    // request after this message has been acknowledged.
    collector.mutate_on::<ResetStreamRound>(move |actor, envelope| {
        let msg = envelope.message();
        actor.model.buffer.clear();
        actor.model.token_count = 0;
        actor.model.stop_reason = None;
        actor.model.tool_calls.clear();
        actor.model.expected_correlation_id = Some(msg.expected_id.clone());
        Reply::ready()
    });

    // Subscribe BEFORE starting so no broadcast can slip past us.
    collector.handle().subscribe::<LLMStreamStart>().await;
    collector.handle().subscribe::<LLMStreamToken>().await;
    collector.handle().subscribe::<LLMStreamToolCall>().await;
    collector.handle().subscribe::<LLMStreamEnd>().await;

    let handle = collector.start().await;
    StreamCollectorSession {
        handle,
        completion,
        result_container,
    }
}

/// Run a single stream round on the supplied long-lived collector.
async fn run_stream_round(
    session: &StreamCollectorSession,
    provider_handle: &ActorHandle,
    request: &LLMRequest,
    correlation_id: CorrelationId,
) -> Result<(String, StopReason, usize, Vec<ToolCall>), ActonAIError> {
    // Ack: reset state and install the new round's correlation ID.
    session
        .handle
        .send(ResetStreamRound {
            expected_id: correlation_id,
        })
        .await;

    // Fire the request.
    provider_handle.send(request.clone()).await;

    // Wait for the stream-end handler to fill the result slot.
    session.completion.notified().await;

    let result = session
        .result_container
        .lock()
        .ok()
        .and_then(|mut guard| guard.take())
        .ok_or_else(|| {
            ActonAIError::prompt_failed("failed to retrieve collected stream data".to_string())
        })?;

    Ok((
        result.buffer,
        result.stop_reason.unwrap_or(StopReason::EndTurn),
        result.token_count,
        result.tool_calls,
    ))
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

/// Executes a single tool call and invokes the result callback if present.
async fn execute_tool_with_callback(
    tools: &mut [ToolSpec],
    tool_call: &ToolCall,
) -> Result<serde_json::Value, ToolError> {
    // Find the tool by name
    for spec in tools.iter_mut() {
        if spec.definition.name == tool_call.name {
            let result = spec.executor.call(tool_call.arguments.clone()).await;

            // Invoke the result callback if present
            if let Some(ref mut callback) = spec.on_result {
                match &result {
                    Ok(value) => callback(Ok(value)),
                    Err(e) => {
                        let error_str = e.to_string();
                        callback(Err(&error_str));
                    }
                }
            }

            return result;
        }
    }

    Err(ToolError::not_found(&tool_call.name))
}

/// Internal actor for collecting stream tokens.
///
/// This actor owns all state for collecting streaming responses, eliminating
/// the need for external Mutex-protected shared state. All mutable state
/// (buffer, token_count, stop_reason, tool_calls) is owned by the actor
/// and accessed directly in handlers via `actor.model`.
///
/// The collector is **long-lived across all tool rounds of a single
/// `collect()` call** — subscribing once at the top and resetting per-round
/// state via [`ResetStreamRound`]. Spawning a fresh collector per round
/// leaks broker subscriptions (acton-reactive's `UnsubscribeBroker` ships
/// as an unimplemented stub: see `common/src/message/unsubscribe_broker.rs`).
#[acton_actor]
struct StreamCollector {
    /// Accumulated response buffer for the current round
    buffer: String,
    /// Count of tokens received in the current round
    token_count: usize,
    /// Stop reason when the current round's stream ends
    stop_reason: Option<StopReason>,
    /// Accumulated tool calls from the current round
    tool_calls: Vec<ToolCall>,
    /// Correlation ID of the round currently being collected. Handlers
    /// ignore any event whose correlation ID doesn't match — protects the
    /// collector from stray events emitted by other concurrent streams
    /// that the provider may broadcast to the same broker channel.
    expected_correlation_id: Option<CorrelationId>,
}

/// Per-round reset message. Sent to the collector before starting each
/// round's LLM request so state is fresh and the correlation-ID filter
/// accepts the new round's events.
#[derive(Clone, Debug)]
struct ResetStreamRound {
    expected_id: CorrelationId,
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
    /// Tool calls received during streaming
    tool_calls: Vec<ToolCall>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_spec_debug_impl() {
        let spec = ToolSpec {
            definition: ToolDefinition {
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
                name: "test".to_string(),
                description: "Test tool".to_string(),
                input_schema: serde_json::json!({}),
            },
            executor: Arc::new(ClosureToolExecutor {
                func: |_args: serde_json::Value| async { Ok(serde_json::json!({})) },
            }),
            on_result: Some(Box::new(|_result| {})),
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
