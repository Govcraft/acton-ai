//! Anthropic API client.
//!
//! HTTP client for communicating with the Anthropic Claude API,
//! including streaming SSE response handling.

use crate::llm::client::{LLMClient, LLMClientResponse, LLMEventStream, LLMStreamEvent};
use crate::llm::config::ProviderConfig;
use crate::llm::error::LLMError;
use crate::messages::{Message, MessageRole, StopReason, ToolCall, ToolDefinition};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Client for the Anthropic Claude API.
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    /// HTTP client
    client: Client,
    /// Configuration
    config: ProviderConfig,
}

/// Request body for the Anthropic messages API.
#[derive(Debug, Clone, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiTool>>,
    stream: bool,
}

/// A message in the API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiMessage {
    role: String,
    content: ApiContent,
}

/// Content in the API format (can be string or array of content blocks).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ApiContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block in the API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// Tool definition in the API format.
#[derive(Debug, Clone, Serialize)]
struct ApiTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

/// Response from the Anthropic messages API (non-streaming).
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    /// Unique ID for this response
    pub id: String,
    /// The model that generated the response
    pub model: String,
    /// The stop reason
    pub stop_reason: Option<String>,
    /// The content blocks
    pub content: Vec<ResponseContentBlock>,
    /// Usage statistics
    pub usage: Usage,
}

/// A content block in the response.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Usage statistics from the API.
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    /// Input tokens used
    pub input_tokens: u32,
    /// Output tokens generated
    pub output_tokens: u32,
}

/// Error response from the Anthropic API.
#[derive(Debug, Clone, Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetail,
}

/// Error detail from the API.
#[derive(Debug, Clone, Deserialize)]
struct ApiErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

/// SSE event types from the streaming API.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Stream has started
    MessageStart {
        /// Response ID
        id: String,
    },
    /// Content block started
    ContentBlockStart {
        /// Index of the content block
        index: usize,
        /// Type of content block
        block_type: String,
        /// Tool ID (for tool_use blocks)
        tool_id: Option<String>,
        /// Tool name (for tool_use blocks)
        tool_name: Option<String>,
    },
    /// Text delta in content
    ContentBlockDelta {
        /// Index of the content block
        index: usize,
        /// Delta type
        delta_type: String,
        /// Text content (for text deltas)
        text: Option<String>,
        /// Partial JSON (for tool input deltas)
        partial_json: Option<String>,
    },
    /// Content block stopped
    ContentBlockStop {
        /// Index of the content block
        index: usize,
    },
    /// Message completed
    MessageDelta {
        /// Stop reason
        stop_reason: Option<String>,
    },
    /// Stream ended
    MessageStop,
    /// Ping event (keep-alive)
    Ping,
    /// Error event
    Error {
        /// Error type
        error_type: String,
        /// Error message
        message: String,
    },
}

/// Raw SSE event data from the API.
#[derive(Debug, Clone, Deserialize)]
struct RawStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    message: Option<serde_json::Value>,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    content_block: Option<serde_json::Value>,
    #[serde(default)]
    delta: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

impl AnthropicClient {
    /// Creates a new Anthropic client with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Provider configuration including API key and settings
    ///
    /// # Returns
    ///
    /// A new `AnthropicClient` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn new(config: ProviderConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| LLMError::network(format!("failed to create HTTP client: {}", e)))?;

        Ok(Self { client, config })
    }

    /// Sends a messages request to the Anthropic API (non-streaming).
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation messages
    /// * `tools` - Optional tool definitions
    ///
    /// # Returns
    ///
    /// The API response with generated content.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the API returns an error.
    pub async fn send_messages(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<MessagesResponse, LLMError> {
        let (system, api_messages) = self.convert_messages(messages);

        let request_body = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            system,
            messages: api_messages,
            tools: tools.map(|t| self.convert_tools(t)),
            stream: false,
        };

        let response = self
            .client
            .post(self.config.messages_endpoint())
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.api_version)
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| self.map_reqwest_error(e))?;

        self.handle_response(response).await
    }

    /// Sends a streaming messages request to the Anthropic API.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation messages
    /// * `tools` - Optional tool definitions
    ///
    /// # Returns
    ///
    /// A stream of SSE events.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn send_messages_streaming(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<impl futures::Stream<Item = Result<StreamEvent, LLMError>>, LLMError> {
        let (system, api_messages) = self.convert_messages(messages);

        let request_body = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            system,
            messages: api_messages,
            tools: tools.map(|t| self.convert_tools(t)),
            stream: true,
        };

        let response = self
            .client
            .post(self.config.messages_endpoint())
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.api_version)
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| self.map_reqwest_error(e))?;

        let status = response.status();
        if !status.is_success() {
            let error = self.parse_error_response(response).await;
            return Err(error);
        }

        let stream = response.bytes_stream().map(move |result| {
            result
                .map_err(|e| LLMError::stream_error(format!("stream read error: {}", e)))
                .and_then(|bytes| {
                    let text = String::from_utf8_lossy(&bytes);
                    Self::parse_sse_events(&text)
                })
        });

        // Flatten the nested stream
        Ok(stream.flat_map(|result| {
            futures::stream::iter(match result {
                Ok(events) => events.into_iter().map(Ok).collect::<Vec<_>>(),
                Err(e) => vec![Err(e)],
            })
        }))
    }

    /// Converts internal messages to API format.
    fn convert_messages(&self, messages: &[Message]) -> (Option<String>, Vec<ApiMessage>) {
        let mut system = None;
        let mut api_messages = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    system = Some(msg.content.clone());
                }
                MessageRole::User => {
                    api_messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: ApiContent::Text(msg.content.clone()),
                    });
                }
                MessageRole::Assistant => {
                    if let Some(tool_calls) = &msg.tool_calls {
                        let blocks: Vec<ContentBlock> = std::iter::once(ContentBlock::Text {
                            text: msg.content.clone(),
                        })
                        .chain(tool_calls.iter().map(|tc| ContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            input: tc.arguments.clone(),
                        }))
                        .collect();

                        api_messages.push(ApiMessage {
                            role: "assistant".to_string(),
                            content: ApiContent::Blocks(blocks),
                        });
                    } else {
                        api_messages.push(ApiMessage {
                            role: "assistant".to_string(),
                            content: ApiContent::Text(msg.content.clone()),
                        });
                    }
                }
                MessageRole::Tool => {
                    if let Some(tool_call_id) = &msg.tool_call_id {
                        api_messages.push(ApiMessage {
                            role: "user".to_string(),
                            content: ApiContent::Blocks(vec![ContentBlock::ToolResult {
                                tool_use_id: tool_call_id.clone(),
                                content: msg.content.clone(),
                            }]),
                        });
                    }
                }
            }
        }

        (system, api_messages)
    }

    /// Converts tool definitions to API format.
    fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<ApiTool> {
        tools
            .iter()
            .map(|t| ApiTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema.clone(),
            })
            .collect()
    }

    /// Handles a successful API response.
    async fn handle_response(
        &self,
        response: reqwest::Response,
    ) -> Result<MessagesResponse, LLMError> {
        let status = response.status();

        if status.is_success() {
            response
                .json::<MessagesResponse>()
                .await
                .map_err(|e| LLMError::parse_error(format!("failed to parse response: {}", e)))
        } else {
            Err(self.parse_error_response(response).await)
        }
    }

    /// Parses an error response from the API.
    async fn parse_error_response(&self, response: reqwest::Response) -> LLMError {
        let status = response.status();
        let status_code = status.as_u16();

        // Check for rate limit headers
        if status_code == 429 {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60);

            return LLMError::rate_limited(Duration::from_secs(retry_after));
        }

        // Try to parse error body
        let error_body = response.text().await.unwrap_or_default();

        if let Ok(api_error) = serde_json::from_str::<ApiErrorResponse>(&error_body) {
            match api_error.error.error_type.as_str() {
                "authentication_error" => LLMError::authentication_failed(&api_error.error.message),
                "invalid_request_error" => LLMError::invalid_request(&api_error.error.message),
                "overloaded_error" => LLMError::model_overloaded(&self.config.model),
                _ => LLMError::api_error(
                    status_code,
                    api_error.error.message,
                    Some(api_error.error.error_type),
                ),
            }
        } else {
            LLMError::api_error(
                status_code,
                if error_body.is_empty() {
                    status.canonical_reason().unwrap_or("Unknown error")
                } else {
                    &error_body
                },
                None,
            )
        }
    }

    /// Maps a reqwest error to an LLMError.
    fn map_reqwest_error(&self, error: reqwest::Error) -> LLMError {
        if error.is_timeout() {
            LLMError::timeout(self.config.timeout)
        } else if error.is_connect() {
            LLMError::network(format!("connection failed: {}", error))
        } else {
            LLMError::network(error.to_string())
        }
    }

    /// Parses SSE events from a text chunk.
    fn parse_sse_events(text: &str) -> Result<Vec<StreamEvent>, LLMError> {
        let mut events = Vec::new();

        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }

                if let Ok(raw_event) = serde_json::from_str::<RawStreamEvent>(data) {
                    if let Some(event) = Self::convert_raw_event(raw_event)? {
                        events.push(event);
                    }
                }
            }
        }

        Ok(events)
    }

    /// Converts a raw SSE event to a typed event.
    fn convert_raw_event(raw: RawStreamEvent) -> Result<Option<StreamEvent>, LLMError> {
        match raw.event_type.as_str() {
            "message_start" => {
                let id = raw
                    .message
                    .and_then(|m| m.get("id").and_then(|v| v.as_str().map(String::from)))
                    .unwrap_or_default();
                Ok(Some(StreamEvent::MessageStart { id }))
            }
            "content_block_start" => {
                let index = raw.index.unwrap_or(0);
                let (block_type, tool_id, tool_name) = raw
                    .content_block
                    .map(|cb| {
                        let block_type = cb
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("text")
                            .to_string();
                        let tool_id = cb.get("id").and_then(|v| v.as_str().map(String::from));
                        let tool_name = cb.get("name").and_then(|v| v.as_str().map(String::from));
                        (block_type, tool_id, tool_name)
                    })
                    .unwrap_or(("text".to_string(), None, None));

                Ok(Some(StreamEvent::ContentBlockStart {
                    index,
                    block_type,
                    tool_id,
                    tool_name,
                }))
            }
            "content_block_delta" => {
                let index = raw.index.unwrap_or(0);
                let (delta_type, text, partial_json) = raw
                    .delta
                    .map(|d| {
                        let delta_type = d
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("text_delta")
                            .to_string();
                        let text = d.get("text").and_then(|v| v.as_str().map(String::from));
                        let partial_json = d
                            .get("partial_json")
                            .and_then(|v| v.as_str().map(String::from));
                        (delta_type, text, partial_json)
                    })
                    .unwrap_or(("text_delta".to_string(), None, None));

                Ok(Some(StreamEvent::ContentBlockDelta {
                    index,
                    delta_type,
                    text,
                    partial_json,
                }))
            }
            "content_block_stop" => {
                let index = raw.index.unwrap_or(0);
                Ok(Some(StreamEvent::ContentBlockStop { index }))
            }
            "message_delta" => {
                let stop_reason = raw.delta.and_then(|d| {
                    d.get("stop_reason")
                        .and_then(|v| v.as_str().map(String::from))
                });
                Ok(Some(StreamEvent::MessageDelta { stop_reason }))
            }
            "message_stop" => Ok(Some(StreamEvent::MessageStop)),
            "ping" => Ok(Some(StreamEvent::Ping)),
            "error" => {
                let (error_type, message) = raw
                    .error
                    .map(|e| {
                        let error_type = e
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("error")
                            .to_string();
                        let message = e
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown error")
                            .to_string();
                        (error_type, message)
                    })
                    .unwrap_or(("error".to_string(), "Unknown error".to_string()));

                Ok(Some(StreamEvent::Error {
                    error_type,
                    message,
                }))
            }
            _ => Ok(None), // Ignore unknown event types
        }
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
    async fn send_request(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<LLMClientResponse, LLMError> {
        let response = self.send_messages(messages, tools).await?;
        Ok(LLMClientResponse {
            content: extract_text_content(&response),
            tool_calls: extract_tool_calls(&response),
            stop_reason: response
                .stop_reason
                .as_ref()
                .map(|s| parse_stop_reason(s))
                .unwrap_or(StopReason::EndTurn),
        })
    }

    async fn send_streaming_request(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<LLMEventStream, LLMError> {
        let stream = self.send_messages_streaming(messages, tools).await?;
        Ok(Box::pin(convert_anthropic_stream(stream)))
    }

    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
}

/// Converts Anthropic stream events to unified LLMStreamEvent.
fn convert_anthropic_stream(
    stream: impl futures::Stream<Item = Result<StreamEvent, LLMError>> + Send + 'static,
) -> impl futures::Stream<Item = Result<LLMStreamEvent, LLMError>> + Send {
    stream.filter_map(|result| async move {
        match result {
            Ok(event) => match event {
                StreamEvent::MessageStart { id } => Some(Ok(LLMStreamEvent::Start { id })),
                StreamEvent::ContentBlockDelta { text, .. } => {
                    text.map(|t| Ok(LLMStreamEvent::Token { text: t }))
                }
                StreamEvent::MessageDelta { stop_reason } => stop_reason.map(|reason| {
                    Ok(LLMStreamEvent::End {
                        stop_reason: parse_stop_reason(&reason),
                    })
                }),
                StreamEvent::MessageStop => Some(Ok(LLMStreamEvent::End {
                    stop_reason: StopReason::EndTurn,
                })),
                StreamEvent::Error {
                    error_type,
                    message,
                } => Some(Ok(LLMStreamEvent::Error {
                    error_type,
                    message,
                })),
                StreamEvent::Ping
                | StreamEvent::ContentBlockStart { .. }
                | StreamEvent::ContentBlockStop { .. } => None,
            },
            Err(e) => Some(Err(e)),
        }
    })
}

/// Converts an API stop reason string to our `StopReason` enum.
#[must_use]
pub fn parse_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "tool_use" => StopReason::ToolUse,
        "stop_sequence" => StopReason::StopSequence,
        _ => StopReason::EndTurn,
    }
}

/// Extracts text content from a response.
#[must_use]
pub fn extract_text_content(response: &MessagesResponse) -> String {
    response
        .content
        .iter()
        .filter_map(|block| match block {
            ResponseContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Extracts tool calls from a response.
#[must_use]
pub fn extract_tool_calls(response: &MessagesResponse) -> Vec<ToolCall> {
    response
        .content
        .iter()
        .filter_map(|block| match block {
            ResponseContentBlock::ToolUse { id, name, input } => Some(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: input.clone(),
            }),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stop_reason_end_turn() {
        assert_eq!(parse_stop_reason("end_turn"), StopReason::EndTurn);
    }

    #[test]
    fn parse_stop_reason_max_tokens() {
        assert_eq!(parse_stop_reason("max_tokens"), StopReason::MaxTokens);
    }

    #[test]
    fn parse_stop_reason_tool_use() {
        assert_eq!(parse_stop_reason("tool_use"), StopReason::ToolUse);
    }

    #[test]
    fn parse_stop_reason_unknown_defaults_to_end_turn() {
        assert_eq!(parse_stop_reason("unknown"), StopReason::EndTurn);
    }

    #[test]
    fn convert_messages_extracts_system() {
        let config = ProviderConfig::new("test-key");
        let client = AnthropicClient::new(config).unwrap();

        let messages = vec![Message::system("You are helpful"), Message::user("Hello")];

        let (system, api_messages) = client.convert_messages(&messages);

        assert_eq!(system, Some("You are helpful".to_string()));
        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, "user");
    }

    #[test]
    fn convert_messages_handles_tool_calls() {
        let config = ProviderConfig::new("test-key");
        let client = AnthropicClient::new(config).unwrap();

        let tool_calls = vec![ToolCall {
            id: "tc_123".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "rust"}),
        }];

        let messages = vec![
            Message::user("Search for rust"),
            Message::assistant_with_tools("I'll search for that.", tool_calls),
            Message::tool("tc_123", "Search results: ..."),
        ];

        let (_, api_messages) = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 3);
        assert_eq!(api_messages[0].role, "user");
        assert_eq!(api_messages[1].role, "assistant");
        assert_eq!(api_messages[2].role, "user"); // Tool results are user messages
    }

    #[test]
    fn convert_tools() {
        let config = ProviderConfig::new("test-key");
        let client = AnthropicClient::new(config).unwrap();

        let tools = vec![ToolDefinition {
            name: "calculator".to_string(),
            description: "Performs math".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }),
        }];

        let api_tools = client.convert_tools(&tools);

        assert_eq!(api_tools.len(), 1);
        assert_eq!(api_tools[0].name, "calculator");
    }

    #[test]
    fn parse_sse_events_text_delta() {
        let text = r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;

        let events = AnthropicClient::parse_sse_events(text).unwrap();

        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ContentBlockDelta {
                index,
                delta_type,
                text,
                ..
            } => {
                assert_eq!(*index, 0);
                assert_eq!(delta_type, "text_delta");
                assert_eq!(text, &Some("Hello".to_string()));
            }
            _ => panic!("Expected ContentBlockDelta"),
        }
    }

    #[test]
    fn parse_sse_events_message_stop() {
        let text = r#"data: {"type":"message_stop"}"#;

        let events = AnthropicClient::parse_sse_events(text).unwrap();

        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::MessageStop));
    }

    #[test]
    fn parse_sse_events_done_marker() {
        let text = "data: [DONE]";

        let events = AnthropicClient::parse_sse_events(text).unwrap();

        assert!(events.is_empty());
    }

    #[test]
    fn parse_sse_events_ping() {
        let text = r#"data: {"type":"ping"}"#;

        let events = AnthropicClient::parse_sse_events(text).unwrap();

        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Ping));
    }

    #[test]
    fn extract_text_content_from_response() {
        let response = MessagesResponse {
            id: "msg_123".to_string(),
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some("end_turn".to_string()),
            content: vec![
                ResponseContentBlock::Text {
                    text: "Hello ".to_string(),
                },
                ResponseContentBlock::Text {
                    text: "World".to_string(),
                },
            ],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        assert_eq!(extract_text_content(&response), "Hello World");
    }

    #[test]
    fn extract_tool_calls_from_response() {
        let response = MessagesResponse {
            id: "msg_123".to_string(),
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some("tool_use".to_string()),
            content: vec![
                ResponseContentBlock::Text {
                    text: "I'll use a tool".to_string(),
                },
                ResponseContentBlock::ToolUse {
                    id: "tc_456".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"query": "rust"}),
                },
            ],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let tool_calls = extract_tool_calls(&response);

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tc_456");
        assert_eq!(tool_calls[0].name, "search");
    }

    #[test]
    fn anthropic_client_implements_llm_client() {
        let config = ProviderConfig::anthropic("test-key");
        let client = AnthropicClient::new(config).unwrap();
        let _boxed: Box<dyn LLMClient> = Box::new(client);
    }

    #[test]
    fn anthropic_client_provider_name() {
        let config = ProviderConfig::anthropic("test-key");
        let client = AnthropicClient::new(config).unwrap();
        assert_eq!(client.provider_name(), "anthropic");
    }
}
