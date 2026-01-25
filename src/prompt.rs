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

use crate::error::ActonAIError;
use crate::facade::ActonAI;
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
struct BuiltinToolExecutorAdapter {
    executor: Arc<crate::tools::BoxedToolExecutor>,
}

impl ToolExecutorFn for BuiltinToolExecutorAdapter {
    fn call(&self, args: serde_json::Value) -> ToolFuture {
        let executor = Arc::clone(&self.executor);
        Box::pin(async move { executor.execute(args).await })
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
pub struct PromptBuilder<'a> {
    /// Reference to the ActonAI runtime
    runtime: &'a ActonAI,
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
}

impl<'a> PromptBuilder<'a> {
    /// Creates a new prompt builder with the given content.
    ///
    /// This is called internally by `ActonAI::prompt()`.
    #[must_use]
    pub(crate) fn new(runtime: &'a ActonAI, user_content: String) -> Self {
        Self {
            runtime,
            user_content,
            system_prompt: None,
            conversation_history: None,
            on_start: None,
            on_token: None,
            on_end: None,
            tools: Vec::new(),
            max_tool_rounds: 10,
            provider_name: None,
        }
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
                if let Some(executor) = builtins.get_executor(name) {
                    let adapter = BuiltinToolExecutorAdapter { executor };
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
        let on_end: Option<WrappedEndCallback> =
            on_end.map(|f| Arc::new(std::sync::Mutex::new(f)));

        loop {
            rounds += 1;
            if rounds > max_tool_rounds {
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
            };

            // Collect stream response
            let (text, stop_reason, token_count, tool_calls) = collect_stream_round(
                runtime,
                &provider_handle,
                &request,
                correlation_id,
                on_start.clone(),
                on_token.clone(),
                on_end.clone(),
            )
            .await?;

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

        Ok(CollectedResponse::with_tool_calls(
            final_text,
            StopReason::EndTurn,
            total_token_count,
            executed_tool_calls,
        ))
    }
}

/// Collects a single stream round.
///
/// This function creates a `StreamCollector` actor that owns all collection state
/// internally. All mutable state (buffer, token_count, stop_reason, tool_calls)
/// is owned by the actor and accessed directly in handlers via `actor.model`,
/// eliminating the need for external Mutex-protected shared state during streaming.
///
/// The only synchronization is a single-use Mutex for the final result, which is
/// filled once when streaming ends and read once to retrieve results. This is
/// a minimal, single-write single-read pattern with no contention during streaming.
async fn collect_stream_round(
    runtime: &ActonAI,
    provider_handle: &ActorHandle,
    request: &LLMRequest,
    correlation_id: CorrelationId,
    on_start: Option<WrappedStartCallback>,
    on_token: Option<WrappedTokenCallback>,
    on_end: Option<WrappedEndCallback>,
) -> Result<(String, StopReason, usize, Vec<ToolCall>), ActonAIError> {
    // Set up completion signal
    let stream_done = Arc::new(Notify::new());
    let stream_done_signal = stream_done.clone();

    // Result container - filled once when stream ends, read once to retrieve
    // This replaces the original per-token Mutex with a single-use pattern
    let result_container: Arc<std::sync::Mutex<Option<CollectorResultData>>> =
        Arc::new(std::sync::Mutex::new(None));
    let result_container_clone = result_container.clone();

    // Create the collector actor
    let mut actor_runtime = runtime.runtime().clone();
    let mut collector = actor_runtime.new_actor::<StreamCollector>();

    // Clone for the start handler
    let on_start_clone = on_start.clone();
    let expected_id = correlation_id.clone();

    // Handle stream start - callback captured in closure
    collector.mutate_on::<LLMStreamStart>(move |_actor, envelope| {
        if envelope.message().correlation_id == expected_id {
            if let Some(ref callback) = on_start_clone {
                if let Ok(mut f) = callback.lock() {
                    f();
                }
            }
        }
        Reply::ready()
    });

    // Clone for the token handler
    let on_token_clone = on_token.clone();
    let expected_id = correlation_id.clone();

    // Handle tokens - accumulates to actor-owned buffer (no Mutex during streaming)
    collector.mutate_on::<LLMStreamToken>(move |actor, envelope| {
        if envelope.message().correlation_id == expected_id {
            let token = &envelope.message().token;

            // State owned by actor - no external Mutex access during streaming
            actor.model.buffer.push_str(token);
            actor.model.token_count += 1;

            if let Some(ref callback) = on_token_clone {
                if let Ok(mut f) = callback.lock() {
                    f(token);
                }
            }
        }
        Reply::ready()
    });

    // Clone for the tool call handler
    let expected_id = correlation_id.clone();

    // Handle tool calls - accumulates to actor-owned vec (no Mutex during streaming)
    collector.mutate_on::<LLMStreamToolCall>(move |actor, envelope| {
        if envelope.message().correlation_id == expected_id {
            // State owned by actor - no external Mutex access during streaming
            actor.model.tool_calls.push(envelope.message().tool_call.clone());
        }
        Reply::ready()
    });

    // Clone for the end handler
    let on_end_clone = on_end.clone();
    let expected_id = correlation_id.clone();

    // Handle stream end - collects results, signals completion
    collector.mutate_on::<LLMStreamEnd>(move |actor, envelope| {
        if envelope.message().correlation_id == expected_id {
            // Set stop reason in actor state
            actor.model.stop_reason = Some(envelope.message().stop_reason);

            // Invoke end callback
            if let Some(ref callback) = on_end_clone {
                if let Ok(mut f) = callback.lock() {
                    f(envelope.message().stop_reason);
                }
            }

            // Collect all results from actor state into the result container
            // This is a single write - no contention during streaming
            if let Ok(mut container) = result_container_clone.lock() {
                *container = Some(CollectorResultData {
                    buffer: std::mem::take(&mut actor.model.buffer),
                    stop_reason: actor.model.stop_reason,
                    token_count: actor.model.token_count,
                    tool_calls: std::mem::take(&mut actor.model.tool_calls),
                });
            }

            // Signal completion
            stream_done_signal.notify_one();
        }
        Reply::ready()
    });

    // Subscribe to streaming events BEFORE starting
    collector.handle().subscribe::<LLMStreamStart>().await;
    collector.handle().subscribe::<LLMStreamToken>().await;
    collector.handle().subscribe::<LLMStreamToolCall>().await;
    collector.handle().subscribe::<LLMStreamEnd>().await;

    // Start the collector
    let collector_handle = collector.start().await;

    // Send the request to the provider
    provider_handle.send(request.clone()).await;

    // Wait for stream completion
    stream_done.notified().await;

    // Stop the collector
    let _ = collector_handle.stop().await;

    // Extract the collected data - single read, no contention
    let result = result_container
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
#[acton_actor]
struct StreamCollector {
    /// Accumulated response buffer
    buffer: String,
    /// Count of tokens received
    token_count: usize,
    /// Stop reason when stream ends
    stop_reason: Option<StopReason>,
    /// Accumulated tool calls
    tool_calls: Vec<ToolCall>,
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
