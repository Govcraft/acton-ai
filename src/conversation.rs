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
use crate::messages::Message;
use crate::prompt::PromptBuilder;
use crate::stream::CollectedResponse;

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
}

impl<'a> Conversation<'a> {
    /// Creates a new conversation.
    fn new(runtime: &'a ActonAI, system_prompt: Option<String>, history: Vec<Message>) -> Self {
        Self {
            runtime,
            history,
            system_prompt,
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
    pub async fn send(&mut self, content: impl Into<String>) -> Result<CollectedResponse, ActonAIError> {
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
}

impl std::fmt::Debug for Conversation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conversation")
            .field("history_len", &self.history.len())
            .field("has_system_prompt", &self.system_prompt.is_some())
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
}

impl<'a> ConversationBuilder<'a> {
    /// Creates a new conversation builder.
    pub(crate) fn new(runtime: &'a ActonAI) -> Self {
        Self {
            runtime,
            system_prompt: None,
            history: Vec::new(),
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

    /// Builds the conversation.
    ///
    /// After calling this, you can use [`Conversation::send`] to interact
    /// with the LLM.
    #[must_use]
    pub fn build(self) -> Conversation<'a> {
        Conversation::new(self.runtime, self.system_prompt, self.history)
    }
}

impl std::fmt::Debug for ConversationBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationBuilder")
            .field("has_system_prompt", &self.system_prompt.is_some())
            .field("history_len", &self.history.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    // Note: Full integration tests require a running LLM.
    // Unit tests for types are limited since they require ActonAI instances.

    #[test]
    fn module_compiles() {
        // This test verifies the module compiles correctly.
        // Full integration tests are in examples/conversation.rs
        assert!(true);
    }
}
