//! Fluent prompt builder for LLM requests.
//!
//! This module provides the `PromptBuilder` for constructing and sending
//! prompts to the LLM with a fluent, ergonomic API.
//!
//! # Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! let response = runtime
//!     .prompt("Explain Rust ownership.")
//!     .system("You are a Rust expert. Be concise.")
//!     .on_start(|| println!("Thinking..."))
//!     .on_token(|token| print!("{token}"))
//!     .on_end(|reason| println!("\n[{reason:?}]"))
//!     .collect()
//!     .await?;
//!
//! println!("Total tokens: {}", response.token_count);
//! ```

use crate::error::ActonAIError;
use crate::facade::ActonAI;
use crate::messages::{LLMRequest, LLMStreamEnd, LLMStreamStart, LLMStreamToken, Message};
use crate::stream::CollectedResponse;
use crate::types::{AgentId, CorrelationId};
use acton_reactive::prelude::*;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

/// Type alias for start callbacks.
type StartCallback = Box<dyn FnMut() + Send + 'static>;

/// Type alias for token callbacks.
type TokenCallback = Box<dyn FnMut(&str) + Send + 'static>;

/// Type alias for end callbacks.
type EndCallback = Box<dyn FnMut(crate::messages::StopReason) + Send + 'static>;

/// Shared state for collecting stream data.
#[derive(Default)]
struct SharedCollectorState {
    buffer: String,
    token_count: usize,
    stop_reason: Option<crate::messages::StopReason>,
}

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
pub struct PromptBuilder<'a> {
    /// Reference to the ActonAI runtime
    runtime: &'a ActonAI,
    /// The user's prompt content
    user_content: String,
    /// Optional system prompt
    system_prompt: Option<String>,
    /// Callback for stream start
    on_start: Option<StartCallback>,
    /// Callback for each token
    on_token: Option<TokenCallback>,
    /// Callback for stream end
    on_end: Option<EndCallback>,
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
            on_start: None,
            on_token: None,
            on_end: None,
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
        F: FnMut(crate::messages::StopReason) + Send + 'static,
    {
        self.on_end = Some(Box::new(f));
        self
    }

    /// Sends the prompt and collects the complete response.
    ///
    /// This method:
    /// 1. Creates a temporary actor to collect tokens
    /// 2. Subscribes to streaming events
    /// 3. Sends the request to the LLM provider
    /// 4. Waits for the stream to complete
    /// 5. Returns the collected response
    ///
    /// Callbacks (`on_start`, `on_token`, `on_end`) are called during streaming.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The runtime has been shut down
    /// - The stream fails to complete
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

        // Build the messages
        let mut messages = Vec::new();
        if let Some(system) = self.system_prompt {
            messages.push(Message::system(system));
        }
        messages.push(Message::user(self.user_content));

        // Generate IDs internally - users don't need to see these
        let correlation_id = CorrelationId::new();
        let agent_id = AgentId::new();

        // Create the request
        let request = LLMRequest {
            correlation_id: correlation_id.clone(),
            agent_id,
            messages,
            tools: None,
        };

        // Set up completion signal and shared state
        let stream_done = Arc::new(Notify::new());
        let stream_done_signal = stream_done.clone();
        let shared_state = Arc::new(Mutex::new(SharedCollectorState::default()));

        // Wrap callbacks in Arc<Mutex> so they can be shared across handlers
        let on_start = self
            .on_start
            .map(|f| Arc::new(std::sync::Mutex::new(f)));
        let on_token = self
            .on_token
            .map(|f| Arc::new(std::sync::Mutex::new(f)));
        let on_end = self.on_end.map(|f| Arc::new(std::sync::Mutex::new(f)));

        // Create the collector actor
        // We need to clone the runtime reference to use in the actor setup
        let mut runtime = self.runtime.runtime().clone();
        let mut collector = runtime.new_actor::<StreamCollector>();

        // Clone for the start handler
        let on_start_clone = on_start.clone();
        let expected_id = correlation_id.clone();

        // Handle stream start
        collector.mutate_on::<LLMStreamStart>(move |_actor, envelope| {
            // Only process events for our correlation ID
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
        let shared_state_clone = shared_state.clone();
        let expected_id = correlation_id.clone();

        // Handle tokens
        collector.mutate_on::<LLMStreamToken>(move |_actor, envelope| {
            // Only process events for our correlation ID
            if envelope.message().correlation_id == expected_id {
                let token = &envelope.message().token;

                // Update shared state
                if let Ok(mut state) = shared_state_clone.try_lock() {
                    state.buffer.push_str(token);
                    state.token_count += 1;
                }

                if let Some(ref callback) = on_token_clone {
                    if let Ok(mut f) = callback.lock() {
                        f(token);
                    }
                }
            }
            Reply::ready()
        });

        // Clone for the end handler
        let on_end_clone = on_end.clone();
        let shared_state_clone = shared_state.clone();
        let expected_id = correlation_id.clone();

        // Handle stream end
        collector.mutate_on::<LLMStreamEnd>(move |_actor, envelope| {
            // Only process events for our correlation ID
            if envelope.message().correlation_id == expected_id {
                // Update shared state
                if let Ok(mut state) = shared_state_clone.try_lock() {
                    state.stop_reason = Some(envelope.message().stop_reason);
                }

                if let Some(ref callback) = on_end_clone {
                    if let Ok(mut f) = callback.lock() {
                        f(envelope.message().stop_reason);
                    }
                }

                stream_done_signal.notify_one();
            }
            Reply::ready()
        });

        // Subscribe to streaming events BEFORE starting
        collector.handle().subscribe::<LLMStreamStart>().await;
        collector.handle().subscribe::<LLMStreamToken>().await;
        collector.handle().subscribe::<LLMStreamEnd>().await;

        // Start the collector
        let collector_handle = collector.start().await;

        // Send the request to the provider
        self.runtime.provider_handle().send(request).await;

        // Wait for stream completion
        stream_done.notified().await;

        // Stop the collector (ignoring any error since we're done)
        let _ = collector_handle.stop().await;

        // Extract the collected data from shared state
        let state = shared_state.lock().await;
        let response = CollectedResponse::new(
            state.buffer.clone(),
            state
                .stop_reason
                .unwrap_or(crate::messages::StopReason::EndTurn),
            state.token_count,
        );

        Ok(response)
    }
}

/// Internal actor for collecting stream tokens.
/// The actual state is stored in the shared Arc<Mutex<>> so we can
/// access it after the stream completes.
#[acton_actor]
struct StreamCollector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collected_response_new_creates_correctly() {
        let response =
            CollectedResponse::new("Hello world".to_string(), crate::messages::StopReason::EndTurn, 2);

        assert_eq!(response.text, "Hello world");
        assert_eq!(response.stop_reason, crate::messages::StopReason::EndTurn);
        assert_eq!(response.token_count, 2);
    }

    #[test]
    fn collected_response_is_complete() {
        let complete =
            CollectedResponse::new("test".to_string(), crate::messages::StopReason::EndTurn, 1);
        assert!(complete.is_complete());

        let incomplete = CollectedResponse::new(
            "test".to_string(),
            crate::messages::StopReason::MaxTokens,
            1,
        );
        assert!(!incomplete.is_complete());
    }

    #[test]
    fn collected_response_is_truncated() {
        let truncated = CollectedResponse::new(
            "test".to_string(),
            crate::messages::StopReason::MaxTokens,
            1,
        );
        assert!(truncated.is_truncated());

        let complete =
            CollectedResponse::new("test".to_string(), crate::messages::StopReason::EndTurn, 1);
        assert!(!complete.is_truncated());
    }
}
