//! LLM client trait abstraction.
//!
//! This module defines the `LLMClient` trait which abstracts over different
//! LLM providers (Anthropic, OpenAI, Ollama, etc.) allowing the LLMProvider
//! actor to work with any backend.

use crate::llm::config::SamplingParams;
use crate::llm::error::LLMError;
use crate::messages::{Message, StopReason, ToolCall, ToolDefinition};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

/// A streaming event from an LLM provider.
#[derive(Debug, Clone)]
pub enum LLMStreamEvent {
    /// Stream has started with the given response ID
    Start {
        /// The response ID from the provider
        id: String,
    },
    /// A text token was generated
    Token {
        /// The text content of the token
        text: String,
    },
    /// A tool call was generated
    ToolCall {
        /// The tool call details
        tool_call: ToolCall,
    },
    /// The stream has ended
    End {
        /// The reason the stream ended
        stop_reason: StopReason,
    },
    /// An error occurred during streaming
    Error {
        /// The type of error
        error_type: String,
        /// The error message
        message: String,
    },
}

/// Response from a non-streaming LLM request.
#[derive(Debug, Clone)]
pub struct LLMClientResponse {
    /// The generated text content
    pub content: String,
    /// Tool calls requested by the model
    pub tool_calls: Vec<ToolCall>,
    /// The reason the model stopped generating
    pub stop_reason: StopReason,
}

/// Type alias for boxed stream of LLM events.
pub type LLMEventStream = Pin<Box<dyn Stream<Item = Result<LLMStreamEvent, LLMError>> + Send>>;

/// Trait for LLM API clients.
///
/// This trait abstracts over different LLM providers (Anthropic, OpenAI, Ollama, etc.)
/// allowing the LLMProvider actor to work with any backend.
///
/// # Example
///
/// ```ignore
/// use acton_ai::llm::{LLMClient, ProviderConfig, AnthropicClient};
///
/// let config = ProviderConfig::anthropic("api-key");
/// let client = AnthropicClient::new(config)?;
///
/// let messages = vec![Message::user("Hello!")];
/// let response = client.send_request(&messages, None).await?;
/// ```
#[async_trait]
pub trait LLMClient: Send + Sync + std::fmt::Debug {
    /// Sends a non-streaming request to the LLM.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation messages
    /// * `tools` - Optional tool definitions available to the LLM
    /// * `sampling` - Optional sampling parameters for this request
    ///
    /// # Returns
    ///
    /// The complete response from the LLM.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    async fn send_request(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        sampling: Option<&SamplingParams>,
    ) -> Result<LLMClientResponse, LLMError>;

    /// Sends a streaming request to the LLM.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation messages
    /// * `tools` - Optional tool definitions available to the LLM
    /// * `sampling` - Optional sampling parameters for this request
    ///
    /// # Returns
    ///
    /// A stream of events from the LLM.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails to start.
    async fn send_streaming_request(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        sampling: Option<&SamplingParams>,
    ) -> Result<LLMEventStream, LLMError>;

    /// Returns the name of this provider for logging/metrics.
    fn provider_name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_stream_event_is_debug() {
        let event = LLMStreamEvent::Token {
            text: "Hello".to_string(),
        };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("Token"));
        assert!(debug_str.contains("Hello"));
    }

    #[test]
    fn llm_stream_event_is_clone() {
        let event = LLMStreamEvent::Start {
            id: "test-id".to_string(),
        };
        let cloned = event.clone();
        assert!(matches!(cloned, LLMStreamEvent::Start { id } if id == "test-id"));
    }

    #[test]
    fn llm_client_response_is_debug() {
        let response = LLMClientResponse {
            content: "Hello".to_string(),
            tool_calls: vec![],
            stop_reason: StopReason::EndTurn,
        };
        let debug_str = format!("{:?}", response);
        assert!(debug_str.contains("Hello"));
        assert!(debug_str.contains("EndTurn"));
    }

    #[test]
    fn llm_client_response_is_clone() {
        let response = LLMClientResponse {
            content: "Hello".to_string(),
            tool_calls: vec![],
            stop_reason: StopReason::EndTurn,
        };
        let cloned = response.clone();
        assert_eq!(cloned.content, "Hello");
        assert_eq!(cloned.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn llm_stream_event_error_variant() {
        let event = LLMStreamEvent::Error {
            error_type: "rate_limit".to_string(),
            message: "Too many requests".to_string(),
        };
        assert!(matches!(
            event,
            LLMStreamEvent::Error { error_type, message }
            if error_type == "rate_limit" && message == "Too many requests"
        ));
    }

    #[test]
    fn llm_stream_event_end_variant() {
        let event = LLMStreamEvent::End {
            stop_reason: StopReason::ToolUse,
        };
        assert!(matches!(
            event,
            LLMStreamEvent::End { stop_reason }
            if stop_reason == StopReason::ToolUse
        ));
    }

    #[test]
    fn llm_stream_event_tool_call_variant() {
        let tool_call = ToolCall {
            id: "tc_123".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "test"}),
        };
        let event = LLMStreamEvent::ToolCall {
            tool_call: tool_call.clone(),
        };
        assert!(matches!(
            event,
            LLMStreamEvent::ToolCall { tool_call: tc }
            if tc.id == "tc_123" && tc.name == "search"
        ));
    }
}
