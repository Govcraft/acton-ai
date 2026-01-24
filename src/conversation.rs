//! Managed conversation abstraction for multi-turn interactions.
//!
//! This module provides the [`Conversation`] type which handles automatic history
//! management, reducing boilerplate in multi-turn conversation scenarios.
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
//! let mut conv = runtime.conversation()
//!     .system("You are a helpful assistant.")
//!     .build();
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
use crate::prompt::PromptBuilder;
use crate::stream::CollectedResponse;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

/// A managed conversation with automatic history tracking.
///
/// `Conversation` eliminates the boilerplate of manually managing conversation
/// history. Each call to [`send`](Self::send) automatically:
/// 1. Adds the user message to history
/// 2. Sends the request to the LLM
/// 3. Adds the assistant response to history
/// 4. Returns the response
///
/// Create a `Conversation` using [`ActonAI::conversation()`](ActonAI::conversation).
///
/// # Example
///
/// ```rust,ignore
/// let mut conv = runtime.conversation()
///     .system("You are a helpful assistant.")
///     .build();
///
/// loop {
///     let input = read_user_input();
///     let response = conv.send(&input).await?;
///     println!("{}", response.text);
/// }
/// ```
pub struct Conversation<'a> {
    /// Reference to the ActonAI runtime
    runtime: &'a ActonAI,
    /// Accumulated conversation history
    history: Vec<Message>,
    /// Optional system prompt
    system_prompt: Option<String>,
    /// Exit flag - set when the exit tool is called
    exit_requested: Arc<AtomicBool>,
    /// Whether the exit tool is enabled
    exit_tool_enabled: bool,
}

impl<'a> Conversation<'a> {
    /// Creates a new conversation.
    fn new(
        runtime: &'a ActonAI,
        system_prompt: Option<String>,
        history: Vec<Message>,
        exit_tool_enabled: bool,
    ) -> Self {
        Self {
            runtime,
            history,
            system_prompt,
            exit_requested: Arc::new(AtomicBool::new(false)),
            exit_tool_enabled,
        }
    }

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
        &mut self,
        content: impl Into<String>,
    ) -> Result<CollectedResponse, ActonAIError> {
        self.send_with(content, |b| b).await
    }

    /// Sends a message with additional prompt configuration.
    ///
    /// This allows per-message customization like adding tools, setting callbacks,
    /// or modifying other prompt options while still getting automatic history management.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = conv.send_with("Process this data", |b| {
    ///     b.tool(
    ///         "process",
    ///         "Process data",
    ///         schema,
    ///         |args| async move { Ok(json!({})) },
    ///     )
    ///     .on_token(|t| print!("{t}"))
    /// }).await?;
    /// ```
    pub async fn send_with<F>(
        &mut self,
        content: impl Into<String>,
        configure: F,
    ) -> Result<CollectedResponse, ActonAIError>
    where
        F: FnOnce(PromptBuilder<'a>) -> PromptBuilder<'a>,
    {
        let user_content = content.into();

        // Add user message to history
        self.history.push(Message::user(&user_content));

        // Build the prompt with history
        let mut builder = self.runtime.continue_with(self.history.clone());

        // Apply system prompt if set
        if let Some(ref system) = self.system_prompt {
            builder = builder.system(system);
        }

        // Inject exit tool if enabled
        if self.exit_tool_enabled {
            let exit_flag = Arc::clone(&self.exit_requested);
            builder = builder.with_tool_callback(
                exit_tool_definition(),
                move |_args| {
                    let flag = Arc::clone(&exit_flag);
                    async move {
                        flag.store(true, Ordering::SeqCst);
                        Ok(serde_json::json!({"status": "goodbye"}))
                    }
                },
                |_result| {},
            );
        }

        // Apply user customization
        builder = configure(builder);

        // Send and collect response
        let response = builder.collect().await?;

        // Add assistant response to history
        self.history.push(Message::assistant(&response.text));

        Ok(response)
    }

    /// Sends a message with a streaming token callback.
    ///
    /// This is a convenience method for the common case of wanting to stream
    /// tokens while still getting automatic history management.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = conv.send_streaming("Tell me a story", |token| {
    ///     print!("{token}");
    ///     std::io::stdout().flush().ok();
    /// }).await?;
    /// println!();
    /// ```
    pub async fn send_streaming<F>(
        &mut self,
        content: impl Into<String>,
        on_token: F,
    ) -> Result<CollectedResponse, ActonAIError>
    where
        F: FnMut(&str) + Send + 'static,
    {
        self.send_with(content, |b| b.on_token(on_token)).await
    }

    /// Returns a reference to the conversation history.
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
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Returns a mutable reference to the conversation history.
    ///
    /// This allows manual manipulation of history when needed, such as:
    /// - Removing messages to shorten context
    /// - Editing messages to correct errors
    /// - Inserting messages for context injection
    #[must_use]
    pub fn history_mut(&mut self) -> &mut Vec<Message> {
        &mut self.history
    }

    /// Clears the conversation history.
    ///
    /// This resets the conversation to a fresh state while keeping the system
    /// prompt. Use this to start a new topic without creating a new `Conversation`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// conv.send("Topic A discussion...").await?;
    /// conv.clear();  // Start fresh
    /// conv.send("Topic B discussion...").await?;
    /// ```
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Returns the number of messages in the conversation history.
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns true if the conversation history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Returns the system prompt, if set.
    #[must_use]
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Sets or updates the system prompt.
    ///
    /// This can be used to change the assistant's behavior mid-conversation.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = Some(prompt.into());
    }

    /// Clears the system prompt.
    pub fn clear_system_prompt(&mut self) {
        self.system_prompt = None;
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
    pub fn clear_exit(&mut self) {
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
        self.exit_tool_enabled
    }
}

impl std::fmt::Debug for Conversation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conversation")
            .field("history_len", &self.history.len())
            .field("has_system_prompt", &self.system_prompt.is_some())
            .field("exit_tool_enabled", &self.exit_tool_enabled)
            .field("exit_requested", &self.exit_requested.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

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
///     .build();
/// ```
pub struct ConversationBuilder<'a> {
    runtime: &'a ActonAI,
    system_prompt: Option<String>,
    history: Vec<Message>,
    /// Whether to enable the built-in exit tool
    exit_tool_enabled: bool,
}

impl<'a> ConversationBuilder<'a> {
    /// Creates a new conversation builder.
    pub(crate) fn new(runtime: &'a ActonAI) -> Self {
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
    ///     .build();
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
    ///     .build();
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
    /// let mut conv = runtime.conversation()
    ///     .system("Be helpful. Use exit_conversation when the user says goodbye.")
    ///     .with_exit_tool()
    ///     .build();
    ///
    /// loop {
    ///     let input = read_input();
    ///     conv.send_streaming(&input, |t| print!("{t}")).await?;
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

    /// Builds the conversation.
    ///
    /// After calling this, you can use [`Conversation::send`] to interact
    /// with the LLM.
    #[must_use]
    pub fn build(self) -> Conversation<'a> {
        Conversation::new(
            self.runtime,
            self.system_prompt,
            self.history,
            self.exit_tool_enabled,
        )
    }
}

impl std::fmt::Debug for ConversationBuilder<'_> {
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
}
