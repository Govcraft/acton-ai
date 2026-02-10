//! Managed conversation abstraction for multi-turn interactions.
//!
//! This module provides the [`Conversation`] type which handles automatic history
//! management, reducing boilerplate in multi-turn conversation scenarios.
//!
//! `Conversation` is an actor-based wrapper: a [`ConversationActor`] owns the history,
//! the mailbox serializes sends, and the public [`Conversation`] handle is
//! `Clone + Send + 'static` with all methods taking `&self`.
//!
//! # Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! let runtime = ActonAI::builder()
//!     .app_name("my-app")
//!     .ollama("qwen2.5:7b")
//!     .with_builtins()
//!     .launch()
//!     .await?;
//!
//! let conv = runtime.conversation()
//!     .system("You are a helpful assistant.")
//!     .build()
//!     .await;
//!
//! // History is automatically managed
//! let response = conv.send("What is Rust?").await?;
//! println!("Assistant: {}", response.text);
//!
//! let response = conv.send("How does ownership work?").await?;
//! println!("Assistant: {}", response.text);
//! ```

use crate::error::ActonAIError;
use crate::facade::ActonAI;
use crate::messages::{Message, ToolDefinition};
use crate::stream::CollectedResponse;
use acton_reactive::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

/// Type alias for input mapper functions used in [`ChatConfig`].
type InputMapperFn = Box<dyn FnMut(&str) -> String + Send>;

/// Default system prompt used by [`Conversation::run_chat`] when no system prompt is set.
///
/// This prompt provides sensible defaults for a general-purpose chat assistant with
/// tool access and exit detection.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a helpful assistant with access to various tools. \
Use tools when appropriate to help the user. \
When the user wants to end the conversation (says goodbye, bye, quit, exit, etc.), \
use the exit_conversation tool.";

// =========================================================================
// ChatConfig (unchanged from original)
// =========================================================================

/// Configuration for the chat loop.
///
/// Created via [`ChatConfig::new()`] and passed to
/// [`Conversation::run_chat_with()`](Conversation::run_chat_with).
///
/// For custom token handling (e.g., sending to a websocket), use the lower-level
/// [`Conversation::send_streaming()`] API in your own loop instead.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::prelude::*;
///
/// conv.run_chat_with(
///     ChatConfig::new()
///         .user_prompt(">>> ")
///         .assistant_prompt("AI: ")
///         .map_input(|s| format!("[context] {}", s))
/// ).await?;
/// ```
pub struct ChatConfig {
    /// The prompt shown before user input (default: "You: ")
    user_prompt: String,
    /// The prefix shown before assistant responses (default: "Assistant: ")
    assistant_prompt: String,
    /// Optional function to transform user input before sending
    input_mapper: Option<InputMapperFn>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            user_prompt: "You: ".to_string(),
            assistant_prompt: "Assistant: ".to_string(),
            input_mapper: None,
        }
    }
}

impl ChatConfig {
    /// Creates a new chat configuration with default settings.
    ///
    /// Default settings:
    /// - User prompt: "You: "
    /// - Assistant prompt: "Assistant: "
    /// - Tokens printed to stdout
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ChatConfig::new()
    ///     .user_prompt(">>> ");
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the prompt shown before user input.
    ///
    /// Default: "You: "
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// ChatConfig::new().user_prompt(">>> ")
    /// ```
    #[must_use]
    pub fn user_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.user_prompt = prompt.into();
        self
    }

    /// Sets the prefix shown before assistant responses.
    ///
    /// Default: "Assistant: "
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// ChatConfig::new().assistant_prompt("AI: ")
    /// ```
    #[must_use]
    pub fn assistant_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.assistant_prompt = prompt.into();
        self
    }

    /// Sets a function to transform user input before sending to the LLM.
    ///
    /// Use this to add context, preprocess input, or inject system information.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// ChatConfig::new()
    ///     .map_input(|s| format!("[User context: admin] {}", s))
    /// ```
    #[must_use]
    pub fn map_input<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str) -> String + Send + 'static,
    {
        self.input_mapper = Some(Box::new(f));
        self
    }
}

impl std::fmt::Debug for ChatConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatConfig")
            .field("user_prompt", &self.user_prompt)
            .field("assistant_prompt", &self.assistant_prompt)
            .field("has_input_mapper", &self.input_mapper.is_some())
            .finish()
    }
}

// =========================================================================
// Exit tool definition (unchanged)
// =========================================================================

/// Creates the exit tool definition for conversation termination.
fn exit_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "exit_conversation".to_string(),
        description: "Call this tool when the user wants to end the conversation, \
                      say goodbye, or leave. Examples: 'bye', 'goodbye', 'I'm done', \
                      'quit', 'exit', 'see ya', 'thanks, that's all'."
            .to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "farewell": {
                    "type": "string",
                    "description": "A friendly farewell message to the user"
                }
            },
            "required": ["farewell"]
        }),
    }
}

// =========================================================================
// Actor messages
// =========================================================================

/// Wrapper → ConversationActor: initiate a send.
#[derive(Clone, Debug)]
struct ConvSend {
    content: String,
    token_target: Option<ActorHandle>,
    result_tx: mpsc::Sender<Result<CollectedResponse, ActonAIError>>,
}

/// Async block → ConversationActor: LLM completed, add assistant message.
#[derive(Clone, Debug)]
struct ConvAddAssistant {
    text: String,
}

/// A streamed token message sent from the conversation to a user-provided actor.
///
/// Register a handler for this message type on your actor to receive
/// individual tokens as they are streamed from the LLM.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::prelude::*;
///
/// actor.mutate_on::<StreamToken>(|_actor, ctx| {
///     print!("{}", ctx.message().text);
///     std::io::stdout().flush().ok();
///     Reply::ready()
/// });
/// ```
#[derive(Clone, Debug)]
pub struct StreamToken {
    /// The token text.
    pub text: String,
}

/// Wrapper → ConversationActor: clear history (fire-and-forget).
#[derive(Clone, Debug)]
struct ConvClear;

/// Wrapper → ConversationActor: update system prompt (fire-and-forget).
#[derive(Clone, Debug)]
struct ConvSetSystemPrompt {
    prompt: Option<String>,
}

// =========================================================================
// ConversationActor
// =========================================================================

/// Actor that owns the conversation history.
///
/// All mutations are serialized through the mailbox. The `Reply::pending`
/// future blocks the mailbox during the LLM call, guaranteeing that
/// `ConvAddAssistant` is processed before the next `ConvSend`.
#[acton_actor]
struct ConversationActor {
    history: Vec<Message>,
}

/// Actor used internally by [`Conversation::run_chat_with`] to print tokens to stdout.
#[derive(Default, Debug)]
struct StdoutTokenPrinter;

// =========================================================================
// Handler configuration
// =========================================================================

/// Shared state captured by handler closures at build time.
///
/// Groups the shared state that `configure_handlers` needs, keeping the
/// function signature clean.
struct HandlerState {
    runtime: ActonAI,
    self_handle: ActorHandle,
    history_tx: Arc<watch::Sender<Vec<Message>>>,
    history_len: Arc<AtomicUsize>,
    exit_requested: Arc<AtomicBool>,
    exit_tool_enabled: Arc<AtomicBool>,
    system_prompt_rx: watch::Receiver<Option<String>>,
    system_prompt_tx: watch::Sender<Option<String>>,
}

/// Registers all message handlers on the `ConversationActor` builder.
fn configure_handlers(builder: &mut ManagedActor<Idle, ConversationActor>, state: HandlerState) {
    let HandlerState {
        runtime,
        self_handle,
        history_tx,
        history_len,
        exit_requested,
        exit_tool_enabled,
        system_prompt_rx,
        system_prompt_tx,
    } = state;
    // ----- ConvSend: push user msg, run LLM call, await it -----
    {
        let runtime = runtime.clone();
        let self_handle = self_handle.clone();
        let history_tx = history_tx.clone();
        let history_len = history_len.clone();
        let exit_requested = exit_requested.clone();
        let exit_tool_enabled = exit_tool_enabled.clone();
        let system_prompt_rx = system_prompt_rx.clone();

        builder.mutate_on::<ConvSend>(move |actor, ctx| {
            let msg = ctx.message().clone();

            // Sync: push user message to history
            actor.model.history.push(Message::user(&msg.content));

            // Update watch channel and atomic
            let _ = history_tx.send(actor.model.history.clone());
            history_len.store(actor.model.history.len(), Ordering::SeqCst);

            // Clone everything for the async block
            let history = actor.model.history.clone();
            let system_prompt = system_prompt_rx.borrow().clone();
            let runtime = runtime.clone();
            let exit_requested = exit_requested.clone();
            let exit_tool_enabled_val = exit_tool_enabled.load(Ordering::SeqCst);
            let result_tx = msg.result_tx;
            let token_target = msg.token_target;
            let self_handle = self_handle.clone();

            // The LLM call runs in a spawned task because PromptBuilder
            // contains non-Sync callbacks (FnMut). The spawned task only
            // requires Send. JoinHandle is Send+Sync, satisfying the
            // FutureBox = Pin<Box<dyn Future + Send + Sync>> requirement.
            Reply::pending(async move {
                let llm_result = tokio::spawn(async move {
                    // Build prompt with full history
                    let mut builder = runtime.continue_with(history);

                    if let Some(ref system) = system_prompt {
                        builder = builder.system(system);
                    }

                    // Inject exit tool if enabled
                    if exit_tool_enabled_val {
                        let exit_flag = exit_requested.clone();
                        builder = builder.with_tool_callback(
                            exit_tool_definition(),
                            move |_args| {
                                let flag = exit_flag.clone();
                                async move {
                                    flag.store(true, Ordering::SeqCst);
                                    Ok(serde_json::json!({"status": "goodbye"}))
                                }
                            },
                            |_result| {},
                        );
                    }

                    // Add token target if provided
                    if let Some(target) = token_target {
                        builder = builder.token_target(target);
                    }

                    // Collect response
                    builder.collect().await
                })
                .await;

                // Unwrap the JoinHandle result
                let result = match llm_result {
                    Ok(r) => r,
                    Err(join_err) => Err(ActonAIError::prompt_failed(join_err.to_string())),
                };

                // On success, send assistant message back to actor
                // (queued BEFORE result is sent to caller → FIFO guarantees ordering)
                if let Ok(ref response) = result {
                    self_handle
                        .send(ConvAddAssistant {
                            text: response.text.clone(),
                        })
                        .await;
                }

                // Send result to caller
                let _ = result_tx.send(result).await;
            })
        });
    }

    // ----- ConvAddAssistant: push assistant msg (sync only) -----
    {
        let history_tx = history_tx.clone();
        let history_len = history_len.clone();

        builder.mutate_on::<ConvAddAssistant>(move |actor, ctx| {
            let text = &ctx.message().text;
            actor.model.history.push(Message::assistant(text));

            let _ = history_tx.send(actor.model.history.clone());
            history_len.store(actor.model.history.len(), Ordering::SeqCst);

            Reply::ready()
        });
    }

    // ----- ConvClear: clear history (sync only) -----
    {
        let history_tx = history_tx.clone();
        let history_len = history_len.clone();

        builder.mutate_on::<ConvClear>(move |actor, _ctx| {
            actor.model.history.clear();

            let _ = history_tx.send(actor.model.history.clone());
            history_len.store(0, Ordering::SeqCst);

            Reply::ready()
        });
    }

    // ----- ConvSetSystemPrompt: update watch channel (sync only) -----
    builder.mutate_on::<ConvSetSystemPrompt>(move |_actor, ctx| {
        let prompt = ctx.message().prompt.clone();
        let _ = system_prompt_tx.send(prompt);

        Reply::ready()
    });
}

// =========================================================================
// Conversation — thin, Clone + Send + 'static wrapper
// =========================================================================

/// A managed conversation with automatic history tracking.
///
/// `Conversation` is backed by an actor that serializes all operations through
/// its mailbox. This makes `Conversation` `Clone + Send + 'static` with all
/// methods taking `&self`, enabling safe sharing across tasks.
///
/// Each call to [`send`](Self::send) automatically:
/// 1. Adds the user message to history
/// 2. Sends the request to the LLM
/// 3. Adds the assistant response to history
/// 4. Returns the response
///
/// Create a `Conversation` using [`ActonAI::conversation()`](ActonAI::conversation)
/// and [`ConversationBuilder::build().await`](ConversationBuilder::build).
///
/// # Example
///
/// ```rust,ignore
/// let conv = runtime.conversation()
///     .system("You are a helpful assistant.")
///     .build()
///     .await;
///
/// loop {
///     let input = read_user_input();
///     let response = conv.send(&input).await?;
///     println!("{}", response.text);
/// }
/// ```
pub struct Conversation {
    /// Handle to the ConversationActor.
    handle: ActorHandle,
    /// The ActonAI runtime (cheaply cloned via Arc).
    runtime: ActonAI,
    /// Exit flag — set when the exit tool is called.
    exit_requested: Arc<AtomicBool>,
    /// Whether the exit tool is enabled.
    exit_tool_enabled: Arc<AtomicBool>,
    /// Broadcast receiver for history changes.
    history_rx: watch::Receiver<Vec<Message>>,
    /// Lock-free history length.
    history_len: Arc<AtomicUsize>,
    /// Broadcast receiver for system prompt changes.
    system_prompt_rx: watch::Receiver<Option<String>>,
}

// Compile-time assertion: Conversation is Clone + Send + 'static.
const _: () = {
    #[allow(dead_code)]
    fn assert_clone_send_static<T: Clone + Send + 'static>() {}
    #[allow(dead_code)]
    fn assert_conversation() {
        assert_clone_send_static::<Conversation>();
    }
};

impl Clone for Conversation {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            runtime: self.runtime.clone(),
            exit_requested: self.exit_requested.clone(),
            exit_tool_enabled: self.exit_tool_enabled.clone(),
            history_rx: self.history_rx.clone(),
            history_len: self.history_len.clone(),
            system_prompt_rx: self.system_prompt_rx.clone(),
        }
    }
}

impl Conversation {
    /// Sends a message and receives a response, automatically managing history.
    ///
    /// This is the primary method for interacting with a conversation. It:
    /// 1. Adds the user message to the conversation history
    /// 2. Sends the request to the LLM with the full history
    /// 3. Adds the assistant's response to history
    /// 4. Returns the collected response
    ///
    /// If builtins were configured with [`with_builtins`](crate::ActonAIBuilder::with_builtins),
    /// they are automatically available to the LLM.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = conv.send("What is the capital of France?").await?;
    /// println!("{}", response.text);  // "Paris"
    ///
    /// // Conversation remembers context
    /// let response = conv.send("What about Germany?").await?;
    /// println!("{}", response.text);  // "Berlin"
    /// ```
    pub async fn send(
        &self,
        content: impl Into<String>,
    ) -> Result<CollectedResponse, ActonAIError> {
        let (tx, mut rx) = mpsc::channel(1);
        self.handle
            .send(ConvSend {
                content: content.into(),
                token_target: None,
                result_tx: tx,
            })
            .await;

        rx.recv().await.unwrap_or_else(|| {
            Err(ActonAIError::prompt_failed(
                "conversation actor dropped".to_string(),
            ))
        })
    }

    /// Sends a message with streaming tokens delivered to a user-provided actor.
    ///
    /// The `token_handle` actor must have a handler registered for [`StreamToken`]
    /// messages. Tokens are delivered in order as they arrive from the LLM.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create a token-handling actor
    /// let mut printer = runtime.runtime().clone().new_actor::<MyPrinter>();
    /// printer.mutate_on::<StreamToken>(|_actor, ctx| {
    ///     print!("{}", ctx.message().text);
    ///     Reply::ready()
    /// });
    /// let printer_handle = printer.start().await;
    ///
    /// let response = conv.send_streaming("Tell me a story", &printer_handle).await?;
    /// ```
    pub async fn send_streaming(
        &self,
        content: impl Into<String>,
        token_handle: &ActorHandle,
    ) -> Result<CollectedResponse, ActonAIError> {
        let (tx, mut rx) = mpsc::channel(1);
        self.handle
            .send(ConvSend {
                content: content.into(),
                token_target: Some(token_handle.clone()),
                result_tx: tx,
            })
            .await;

        rx.recv().await.unwrap_or_else(|| {
            Err(ActonAIError::prompt_failed(
                "conversation actor dropped".to_string(),
            ))
        })
    }

    /// Returns a snapshot of the conversation history.
    ///
    /// This is useful for:
    /// - Serializing the conversation for persistence
    /// - Inspecting the conversation for debugging
    /// - Building custom UIs that display history
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for message in conv.history() {
    ///     println!("{}: {}", message.role, message.content);
    /// }
    /// ```
    #[must_use]
    pub fn history(&self) -> Vec<Message> {
        self.history_rx.borrow().clone()
    }

    /// Clears the conversation history.
    ///
    /// This resets the conversation to a fresh state while keeping the system
    /// prompt. Use this to start a new topic without creating a new `Conversation`.
    ///
    /// The clear is sent as a fire-and-forget message to the actor and will be
    /// processed after any in-flight sends complete.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// conv.send("Topic A discussion...").await?;
    /// conv.clear();  // Start fresh
    /// conv.send("Topic B discussion...").await?;
    /// ```
    pub fn clear(&self) {
        let handle = self.handle.clone();
        tokio::spawn(async move {
            handle.send(ConvClear).await;
        });
    }

    /// Returns the number of messages in the conversation history.
    #[must_use]
    pub fn len(&self) -> usize {
        self.history_len.load(Ordering::SeqCst)
    }

    /// Returns true if the conversation history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history_len.load(Ordering::SeqCst) == 0
    }

    /// Returns the system prompt, if set.
    #[must_use]
    pub fn system_prompt(&self) -> Option<String> {
        self.system_prompt_rx.borrow().clone()
    }

    /// Sets or updates the system prompt.
    ///
    /// This can be used to change the assistant's behavior mid-conversation.
    /// The change is sent as a fire-and-forget message and will take effect
    /// on the next [`send`](Self::send) call.
    pub fn set_system_prompt(&self, prompt: impl Into<String>) {
        let handle = self.handle.clone();
        let prompt = prompt.into();
        tokio::spawn(async move {
            handle
                .send(ConvSetSystemPrompt {
                    prompt: Some(prompt),
                })
                .await;
        });
    }

    /// Clears the system prompt.
    pub fn clear_system_prompt(&self) {
        let handle = self.handle.clone();
        tokio::spawn(async move {
            handle.send(ConvSetSystemPrompt { prompt: None }).await;
        });
    }

    /// Returns `true` if the exit tool has been called.
    ///
    /// Use this to check if the conversation should end. The exit flag
    /// is set when the LLM calls the `exit_conversation` tool, which
    /// it does when the user indicates they want to leave.
    ///
    /// This method is only meaningful when the conversation was built
    /// with [`with_exit_tool`](ConversationBuilder::with_exit_tool).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// loop {
    ///     let response = conv.send(&input).await?;
    ///
    ///     if conv.should_exit() {
    ///         println!("Goodbye!");
    ///         break;
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn should_exit(&self) -> bool {
        self.exit_requested.load(Ordering::SeqCst)
    }

    /// Clears the exit flag.
    ///
    /// Use this if you want to reset the conversation state and continue
    /// after the exit tool was called. For example, you might ask for
    /// confirmation before actually exiting.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if conv.should_exit() {
    ///     println!("Are you sure you want to leave?");
    ///     let confirmation = read_input();
    ///     if confirmation != "yes" {
    ///         conv.clear_exit();
    ///         continue;
    ///     }
    /// }
    /// ```
    pub fn clear_exit(&self) {
        self.exit_requested.store(false, Ordering::SeqCst);
    }

    /// Returns a clone of the exit flag `Arc<AtomicBool>`.
    ///
    /// This is useful if you need to share the exit state with other
    /// components or check it from a different context.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let exit_flag = conv.exit_requested();
    ///
    /// // Check from another context
    /// if exit_flag.load(Ordering::SeqCst) {
    ///     // Handle exit
    /// }
    /// ```
    #[must_use]
    pub fn exit_requested(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.exit_requested)
    }

    /// Returns whether the exit tool is enabled for this conversation.
    #[must_use]
    pub fn is_exit_tool_enabled(&self) -> bool {
        self.exit_tool_enabled.load(Ordering::SeqCst)
    }

    /// Runs a complete terminal chat loop.
    ///
    /// This method provides a minimal API for building terminal chat applications.
    /// It handles:
    /// - Reading input from stdin with "You: " prompt
    /// - Sending to the LLM and streaming the response
    /// - Printing "Assistant: " prefix before responses
    /// - Streaming tokens to stdout
    /// - Checking `should_exit()` and breaking when true
    /// - Breaking on EOF (Ctrl+D)
    ///
    /// The exit tool is automatically enabled since the chat loop depends on it
    /// for graceful termination. If no system prompt is set, a sensible default
    /// is used (see [`DEFAULT_SYSTEM_PROMPT`]).
    ///
    /// # Errors
    ///
    /// Returns an error if an LLM request fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// ActonAI::builder()
    ///     .app_name("chat")
    ///     .ollama("qwen2.5:7b")
    ///     .with_builtins()
    ///     .launch()
    ///     .await?
    ///     .conversation()
    ///     .run_chat()
    ///     .await?;
    /// ```
    pub async fn run_chat(&self) -> Result<(), ActonAIError> {
        self.run_chat_with(ChatConfig::default()).await
    }

    /// Runs a complete terminal chat loop with custom configuration.
    ///
    /// Like [`run_chat`](Self::run_chat), but allows customization of prompts,
    /// input transformation, and token handling via [`ChatConfig`].
    ///
    /// The exit tool is automatically enabled since the chat loop depends on it
    /// for graceful termination. If no system prompt is set, a sensible default
    /// is used (see [`DEFAULT_SYSTEM_PROMPT`]).
    ///
    /// # Errors
    ///
    /// Returns an error if an LLM request fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// conv.run_chat_with(
    ///     ChatConfig::new()
    ///         .user_prompt(">>> ")
    ///         .assistant_prompt("AI: ")
    ///         .map_input(|s| format!("[context] {}", s))
    /// ).await?;
    /// ```
    pub async fn run_chat_with(&self, mut config: ChatConfig) -> Result<(), ActonAIError> {
        use std::io::{BufRead, Write};

        // Auto-enable exit tool since chat loop depends on it
        self.exit_tool_enabled.store(true, Ordering::SeqCst);

        // Use default system prompt if none is set
        if self.system_prompt_rx.borrow().is_none() {
            self.handle
                .send(ConvSetSystemPrompt {
                    prompt: Some(DEFAULT_SYSTEM_PROMPT.to_string()),
                })
                .await;
        }

        // Create a temporary token actor for stdout printing
        let mut actor_runtime = self.runtime.runtime().clone();
        let mut token_actor = actor_runtime.new_actor::<StdoutTokenPrinter>();
        token_actor.mutate_on::<StreamToken>(|_actor, ctx| {
            print!("{}", ctx.message().text);
            std::io::stdout().flush().ok();
            Reply::ready()
        });
        let token_handle = token_actor.start().await;

        let stdin = std::io::stdin();

        let result = loop {
            // Print user prompt and flush
            print!("{}", config.user_prompt);
            std::io::stdout().flush().ok();

            // Read input line
            let mut input = String::new();
            if stdin.lock().read_line(&mut input).unwrap_or(0) == 0 {
                break Ok(()); // EOF
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            // Apply input mapper if present
            let content = match config.input_mapper.as_mut() {
                Some(mapper) => mapper(input),
                None => input.to_string(),
            };

            // Print assistant prompt
            print!("{}", config.assistant_prompt);
            std::io::stdout().flush().ok();

            // Stream response to stdout via the token actor
            match self.send_streaming(&content, &token_handle).await {
                Ok(_) => {
                    println!();
                }
                Err(e) => {
                    break Err(e);
                }
            }

            // Check for exit
            if self.should_exit() {
                break Ok(());
            }
        };

        // Stop the temporary token actor
        let _ = token_handle.stop().await;

        result
    }
}

impl std::fmt::Debug for Conversation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conversation")
            .field("history_len", &self.history_len.load(Ordering::SeqCst))
            .field(
                "has_system_prompt",
                &self.system_prompt_rx.borrow().is_some(),
            )
            .field(
                "exit_tool_enabled",
                &self.exit_tool_enabled.load(Ordering::SeqCst),
            )
            .field(
                "exit_requested",
                &self.exit_requested.load(Ordering::SeqCst),
            )
            .finish_non_exhaustive()
    }
}

// =========================================================================
// ConversationBuilder
// =========================================================================

/// Builder for creating a [`Conversation`].
///
/// Created via [`ActonAI::conversation()`](ActonAI::conversation).
///
/// # Example
///
/// ```rust,ignore
/// let conv = runtime.conversation()
///     .system("You are a helpful assistant.")
///     .restore(saved_history)
///     .build()
///     .await;
/// ```
pub struct ConversationBuilder {
    runtime: ActonAI,
    system_prompt: Option<String>,
    history: Vec<Message>,
    /// Whether to enable the built-in exit tool
    exit_tool_enabled: bool,
}

impl ConversationBuilder {
    /// Creates a new conversation builder.
    pub(crate) fn new(runtime: ActonAI) -> Self {
        Self {
            runtime,
            system_prompt: None,
            history: Vec::new(),
            exit_tool_enabled: false,
        }
    }

    /// Sets the system prompt for the conversation.
    ///
    /// The system prompt provides context and instructions to the LLM
    /// about how to respond. It's applied to every message in the conversation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let conv = runtime.conversation()
    ///     .system("You are a concise assistant. Answer in one sentence.")
    ///     .build()
    ///     .await;
    /// ```
    #[must_use]
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Restores conversation history from a previous session.
    ///
    /// Use this to continue a conversation that was previously saved or
    /// to inject context from an external source.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Load saved history
    /// let saved: Vec<Message> = load_from_database();
    ///
    /// let conv = runtime.conversation()
    ///     .system("Be helpful.")
    ///     .restore(saved)
    ///     .build()
    ///     .await;
    /// ```
    #[must_use]
    pub fn restore(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.history = messages.into_iter().collect();
        self
    }

    /// Enables the built-in exit tool for this conversation.
    ///
    /// When enabled, an `exit_conversation` tool is automatically available
    /// to the LLM. When the LLM calls this tool (typically when the user
    /// wants to end the conversation), the conversation's exit flag is set.
    ///
    /// Check the flag with [`Conversation::should_exit`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let conv = runtime.conversation()
    ///     .system("Be helpful. Use exit_conversation when the user says goodbye.")
    ///     .with_exit_tool()
    ///     .build()
    ///     .await;
    ///
    /// loop {
    ///     let input = read_input();
    ///     let response = conv.send(&input).await?;
    ///
    ///     if conv.should_exit() {
    ///         break;
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn with_exit_tool(mut self) -> Self {
        self.exit_tool_enabled = true;
        self
    }

    /// Explicitly disables the exit tool for this conversation.
    ///
    /// By default, [`run_chat`](Conversation::run_chat) auto-enables the exit tool.
    /// Use this method if you want to build a conversation that will be used
    /// with `run_chat` but you want to handle exit logic differently.
    ///
    /// Note: If you later call [`run_chat`](Conversation::run_chat) or
    /// [`run_chat_with`](Conversation::run_chat_with), the exit tool will
    /// still be auto-enabled since the chat loop depends on it.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let conv = runtime.conversation()
    ///     .system("You are helpful.")
    ///     .without_exit_tool()
    ///     .build()
    ///     .await;
    ///
    /// // Manual control over conversation
    /// loop {
    ///     let response = conv.send(&input).await?;
    ///     if custom_exit_condition(&response) {
    ///         break;
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn without_exit_tool(mut self) -> Self {
        self.exit_tool_enabled = false;
        self
    }

    /// Builds the conversation by spawning a [`ConversationActor`].
    ///
    /// After calling this, you can use [`Conversation::send`] to interact
    /// with the LLM. The returned `Conversation` is `Clone + Send + 'static`.
    pub async fn build(self) -> Conversation {
        let initial_history = self.history;

        // Create watch channels with initial values
        let (history_tx, history_rx) = watch::channel(initial_history.clone());
        let (system_prompt_tx, system_prompt_rx) = watch::channel(self.system_prompt);

        let history_len = Arc::new(AtomicUsize::new(initial_history.len()));
        let exit_requested = Arc::new(AtomicBool::new(false));
        let exit_tool_enabled = Arc::new(AtomicBool::new(self.exit_tool_enabled));

        // Create the actor
        let mut actor_runtime = self.runtime.runtime().clone();
        let mut actor_builder = actor_runtime.new_actor::<ConversationActor>();

        // Set initial history in the actor model
        actor_builder.model.history = initial_history;

        let actor_handle = actor_builder.handle().clone();

        // Configure all message handlers
        configure_handlers(
            &mut actor_builder,
            HandlerState {
                runtime: self.runtime.clone(),
                self_handle: actor_handle.clone(),
                history_tx: Arc::new(history_tx),
                history_len: history_len.clone(),
                exit_requested: exit_requested.clone(),
                exit_tool_enabled: exit_tool_enabled.clone(),
                system_prompt_rx: system_prompt_rx.clone(),
                system_prompt_tx,
            },
        );

        // Start the actor
        let _started = actor_builder.start().await;

        Conversation {
            handle: actor_handle,
            runtime: self.runtime,
            exit_requested,
            exit_tool_enabled,
            history_rx,
            history_len,
            system_prompt_rx,
        }
    }

    /// Builds the conversation and immediately runs the chat loop.
    ///
    /// This is a convenience method that combines [`build()`](Self::build)
    /// and [`Conversation::run_chat()`](Conversation::run_chat) into a single call.
    ///
    /// The exit tool is automatically enabled and a default system prompt
    /// is used if none was set.
    ///
    /// # Errors
    ///
    /// Returns an error if an LLM request fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Minimal chat - all defaults
    /// ActonAI::builder()
    ///     .app_name("chat")
    ///     .ollama("qwen2.5:7b")
    ///     .with_builtins()
    ///     .launch()
    ///     .await?
    ///     .conversation()
    ///     .run_chat()
    ///     .await?;
    ///
    /// // Or with custom system prompt
    /// runtime.conversation()
    ///     .system("You are a coding assistant.")
    ///     .run_chat()
    ///     .await?;
    /// ```
    pub async fn run_chat(self) -> Result<(), ActonAIError> {
        self.build().await.run_chat().await
    }

    /// Builds the conversation and immediately runs the chat loop with custom config.
    ///
    /// This is a convenience method that combines [`build()`](Self::build)
    /// and [`Conversation::run_chat_with()`](Conversation::run_chat_with) into a single call.
    ///
    /// # Errors
    ///
    /// Returns an error if an LLM request fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime.conversation()
    ///     .system("You are helpful.")
    ///     .run_chat_with(
    ///         ChatConfig::new()
    ///             .user_prompt(">>> ")
    ///             .assistant_prompt("AI: ")
    ///     )
    ///     .await?;
    /// ```
    pub async fn run_chat_with(self, config: ChatConfig) -> Result<(), ActonAIError> {
        self.build().await.run_chat_with(config).await
    }
}

impl std::fmt::Debug for ConversationBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationBuilder")
            .field("has_system_prompt", &self.system_prompt.is_some())
            .field("history_len", &self.history.len())
            .field("exit_tool_enabled", &self.exit_tool_enabled)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_tool_definition_has_required_fields() {
        let def = exit_tool_definition();
        assert_eq!(def.name, "exit_conversation");
        assert!(def.description.contains("goodbye"));

        // Check schema has farewell property
        let props = def.input_schema.get("properties").unwrap();
        assert!(props.get("farewell").is_some());

        // Check farewell is required
        let required = def.input_schema.get("required").unwrap();
        let required_arr = required.as_array().unwrap();
        assert!(required_arr.iter().any(|v| v.as_str() == Some("farewell")));
    }

    #[test]
    fn exit_flag_atomic_operations() {
        let flag = Arc::new(AtomicBool::new(false));
        assert!(!flag.load(Ordering::SeqCst));

        flag.store(true, Ordering::SeqCst);
        assert!(flag.load(Ordering::SeqCst));

        flag.store(false, Ordering::SeqCst);
        assert!(!flag.load(Ordering::SeqCst));
    }

    #[test]
    fn chat_config_default_values() {
        let config = ChatConfig::new();
        assert_eq!(config.user_prompt, "You: ");
        assert_eq!(config.assistant_prompt, "Assistant: ");
        assert!(config.input_mapper.is_none());
    }

    #[test]
    fn chat_config_custom_user_prompt() {
        let config = ChatConfig::new().user_prompt(">>> ");
        assert_eq!(config.user_prompt, ">>> ");
    }

    #[test]
    fn chat_config_custom_assistant_prompt() {
        let config = ChatConfig::new().assistant_prompt("AI: ");
        assert_eq!(config.assistant_prompt, "AI: ");
    }

    #[test]
    fn chat_config_with_input_mapper() {
        let mut config = ChatConfig::new().map_input(|s| format!("[test] {}", s));

        let mapper = config.input_mapper.as_mut().unwrap();
        assert_eq!(mapper("hello"), "[test] hello");
    }

    #[test]
    fn chat_config_debug_impl() {
        let config = ChatConfig::new()
            .user_prompt("test> ")
            .map_input(|s| s.to_string());

        let debug = format!("{:?}", config);
        assert!(debug.contains("test> "));
        assert!(debug.contains("has_input_mapper"));
    }

    #[test]
    fn chat_config_chaining() {
        let config = ChatConfig::new()
            .user_prompt("U> ")
            .assistant_prompt("A> ")
            .map_input(|s| s.to_uppercase());

        assert_eq!(config.user_prompt, "U> ");
        assert_eq!(config.assistant_prompt, "A> ");
        assert!(config.input_mapper.is_some());
    }

    #[test]
    fn default_system_prompt_is_sensible() {
        assert!(DEFAULT_SYSTEM_PROMPT.contains("helpful"));
        assert!(DEFAULT_SYSTEM_PROMPT.contains("exit_conversation"));
    }
}
