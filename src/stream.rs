//! Stream handling for LLM responses.
//!
//! This module provides the `StreamHandler` trait for custom actors that need
//! to handle streaming LLM responses with their own state management.
//!
//! # Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! #[acton_actor]
//! struct TokenCollector {
//!     buffer: String,
//!     token_count: usize,
//! }
//!
//! impl StreamHandler for TokenCollector {
//!     fn on_token(&mut self, token: &str) {
//!         self.buffer.push_str(token);
//!         self.token_count += 1;
//!     }
//!
//!     fn on_end(&mut self, reason: StopReason) -> StreamAction {
//!         println!("Collected {} tokens: {}", self.token_count, self.buffer);
//!         StreamAction::Complete
//!     }
//! }
//! ```

use crate::messages::StopReason;
use crate::types::CorrelationId;

/// Action to take after processing a stream event.
///
/// Returned from `StreamHandler::on_end` to indicate what should happen
/// after the stream completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamAction {
    /// Continue listening for more streams.
    ///
    /// Use this when the handler should stay active to handle
    /// subsequent LLM requests.
    #[default]
    Continue,

    /// Signal completion to any waiters.
    ///
    /// Use this when you want to notify code that's waiting for
    /// the stream to complete (e.g., via `collect().await`).
    Complete,

    /// Stop the actor after this stream.
    ///
    /// Use this when the handler should be cleaned up after
    /// processing this stream.
    Stop,
}

impl StreamAction {
    /// Returns true if the action is Continue.
    #[must_use]
    pub fn is_continue(&self) -> bool {
        matches!(self, Self::Continue)
    }

    /// Returns true if the action is Complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }

    /// Returns true if the action is Stop.
    #[must_use]
    pub fn is_stop(&self) -> bool {
        matches!(self, Self::Stop)
    }
}

/// Trait for handling streaming LLM responses.
///
/// Implement this trait on your actor to receive streaming tokens
/// with full access to actor state. This is the Layer 3 API for
/// users who need custom stream processing with stateful actors.
///
/// # Example
///
/// ```rust,ignore
/// #[acton_actor]
/// struct MyCollector {
///     buffer: String,
///     word_count: usize,
/// }
///
/// impl StreamHandler for MyCollector {
///     fn on_start(&mut self, correlation_id: &CorrelationId) {
///         println!("Stream {} starting", correlation_id);
///         self.buffer.clear();
///         self.word_count = 0;
///     }
///
///     fn on_token(&mut self, token: &str) {
///         self.buffer.push_str(token);
///         // Count words as they come in
///         self.word_count += token.split_whitespace().count();
///     }
///
///     fn on_end(&mut self, reason: StopReason) -> StreamAction {
///         println!("Stream ended: {:?}, {} words", reason, self.word_count);
///         StreamAction::Complete
///     }
/// }
/// ```
///
/// # Subscription
///
/// When using `ActonAI::spawn_stream_handler()`, the handler is automatically
/// subscribed to `LLMStreamStart`, `LLMStreamToken`, and `LLMStreamEnd` messages.
/// No manual subscription is required.
pub trait StreamHandler: Send + 'static {
    /// Called when a new stream starts.
    ///
    /// Use this to initialize any per-stream state.
    ///
    /// # Arguments
    ///
    /// * `correlation_id` - The correlation ID for this stream, useful for
    ///   tracking which request this stream belongs to.
    fn on_start(&mut self, _correlation_id: &CorrelationId) {}

    /// Called for each token in the stream.
    ///
    /// This is the only required method. Tokens are delivered in order
    /// as they are received from the LLM.
    ///
    /// # Arguments
    ///
    /// * `token` - The token text. May be a partial word, punctuation, or whitespace.
    fn on_token(&mut self, token: &str);

    /// Called when the stream ends.
    ///
    /// Returns a `StreamAction` indicating what should happen next.
    ///
    /// # Arguments
    ///
    /// * `stop_reason` - Why the LLM stopped generating:
    ///   - `EndTurn`: Normal completion
    ///   - `MaxTokens`: Reached token limit
    ///   - `ToolUse`: Model wants to call tools
    ///   - `StopSequence`: Hit a stop sequence
    ///
    /// # Returns
    ///
    /// - `StreamAction::Continue`: Keep listening for more streams
    /// - `StreamAction::Complete`: Signal waiters that this stream is done
    /// - `StreamAction::Stop`: Stop the actor
    fn on_end(&mut self, _stop_reason: StopReason) -> StreamAction {
        StreamAction::Continue
    }
}

/// Response collected from a completed stream.
///
/// Returned by `PromptBuilder::collect()` after the stream completes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectedResponse {
    /// The complete text generated by the LLM.
    pub text: String,

    /// Why the LLM stopped generating.
    pub stop_reason: StopReason,

    /// Number of tokens in the response.
    pub token_count: usize,
}

impl CollectedResponse {
    /// Creates a new collected response.
    #[must_use]
    pub fn new(text: String, stop_reason: StopReason, token_count: usize) -> Self {
        Self {
            text,
            stop_reason,
            token_count,
        }
    }

    /// Returns true if the response completed normally.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self.stop_reason, StopReason::EndTurn)
    }

    /// Returns true if the response was truncated due to token limit.
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        matches!(self.stop_reason, StopReason::MaxTokens)
    }

    /// Returns true if the model wants to call tools.
    #[must_use]
    pub fn needs_tool_call(&self) -> bool {
        matches!(self.stop_reason, StopReason::ToolUse)
    }
}

impl Default for CollectedResponse {
    fn default() -> Self {
        Self {
            text: String::new(),
            stop_reason: StopReason::EndTurn,
            token_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_action_default_is_continue() {
        assert_eq!(StreamAction::default(), StreamAction::Continue);
    }

    #[test]
    fn stream_action_is_continue() {
        assert!(StreamAction::Continue.is_continue());
        assert!(!StreamAction::Complete.is_continue());
        assert!(!StreamAction::Stop.is_continue());
    }

    #[test]
    fn stream_action_is_complete() {
        assert!(!StreamAction::Continue.is_complete());
        assert!(StreamAction::Complete.is_complete());
        assert!(!StreamAction::Stop.is_complete());
    }

    #[test]
    fn stream_action_is_stop() {
        assert!(!StreamAction::Continue.is_stop());
        assert!(!StreamAction::Complete.is_stop());
        assert!(StreamAction::Stop.is_stop());
    }

    #[test]
    fn collected_response_new() {
        let response = CollectedResponse::new("Hello".to_string(), StopReason::EndTurn, 1);

        assert_eq!(response.text, "Hello");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.token_count, 1);
    }

    #[test]
    fn collected_response_is_complete() {
        let complete = CollectedResponse::new("test".to_string(), StopReason::EndTurn, 1);
        assert!(complete.is_complete());

        let truncated = CollectedResponse::new("test".to_string(), StopReason::MaxTokens, 1);
        assert!(!truncated.is_complete());
    }

    #[test]
    fn collected_response_is_truncated() {
        let truncated = CollectedResponse::new("test".to_string(), StopReason::MaxTokens, 1);
        assert!(truncated.is_truncated());

        let complete = CollectedResponse::new("test".to_string(), StopReason::EndTurn, 1);
        assert!(!complete.is_truncated());
    }

    #[test]
    fn collected_response_needs_tool_call() {
        let tool_use = CollectedResponse::new("".to_string(), StopReason::ToolUse, 0);
        assert!(tool_use.needs_tool_call());

        let complete = CollectedResponse::new("test".to_string(), StopReason::EndTurn, 1);
        assert!(!complete.needs_tool_call());
    }

    #[test]
    fn collected_response_default() {
        let response = CollectedResponse::default();

        assert!(response.text.is_empty());
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.token_count, 0);
    }
}
