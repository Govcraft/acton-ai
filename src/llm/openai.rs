//! OpenAI-compatible API client.
//!
//! HTTP client for communicating with OpenAI-compatible APIs including
//! OpenAI, Ollama, vLLM, LocalAI, and other compatible endpoints.

use crate::llm::client::{LLMClient, LLMClientResponse, LLMEventStream, LLMStreamEvent};
use crate::llm::config::ProviderConfig;
use crate::llm::error::LLMError;
use crate::messages::{Message, MessageRole, StopReason, ToolCall, ToolDefinition};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Client for OpenAI-compatible APIs (OpenAI, Ollama, vLLM, LocalAI, etc.).
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    /// HTTP client
    client: Client,
    /// Base URL for the API
    base_url: String,
    /// API key (optional for local providers like Ollama)
    api_key: Option<String>,
    /// Model name
    model: String,
    /// Maximum tokens to generate
    max_tokens: u32,
}

/// Request body for OpenAI chat completions API.
#[derive(Debug, Clone, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    stream: bool,
}

/// A message in OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// A tool definition in OpenAI format.
#[derive(Debug, Clone, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

/// A function definition in OpenAI format.
#[derive(Debug, Clone, Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// A tool call in OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunctionCall,
}

/// A function call in OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

/// Non-streaming response from OpenAI API.
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionResponse {
    #[allow(dead_code)]
    id: String,
    choices: Vec<ChatCompletionChoice>,
}

/// A choice in the response.
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionChoice {
    #[allow(dead_code)]
    index: usize,
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

/// Streaming chunk from OpenAI API.
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionChunk {
    id: String,
    choices: Vec<ChatCompletionChunkChoice>,
}

/// A choice in a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionChunkChoice {
    #[allow(dead_code)]
    index: usize,
    delta: ChatCompletionDelta,
    finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

/// Tool call delta in streaming.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAIFunctionCallDelta>,
}

/// Function call delta in streaming.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIFunctionCallDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Error response from OpenAI API.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIErrorDetail,
}

/// Error detail from the API.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIErrorDetail {
    #[serde(rename = "type")]
    error_type: Option<String>,
    message: String,
}

/// Accumulator for building tool calls from streaming deltas.
#[derive(Debug, Clone, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl OpenAIClient {
    /// Creates a new OpenAI-compatible client.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL for the API (e.g., "http://localhost:11434/v1")
    /// * `config` - Provider configuration with model, max_tokens, timeout
    ///
    /// # Returns
    ///
    /// A new `OpenAIClient` instance.
    ///
    /// # Errors
    ///
    /// Returns `LLMError::network` if the HTTP client cannot be created.
    pub fn new(base_url: String, config: &ProviderConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| LLMError::network(format!("failed to create HTTP client: {}", e)))?;

        let api_key = if config.api_key.is_empty() {
            None
        } else {
            Some(config.api_key.clone())
        };

        Ok(Self {
            client,
            base_url,
            api_key,
            model: config.model.clone(),
            max_tokens: config.max_tokens,
        })
    }

    /// Creates a client configured for Ollama.
    ///
    /// # Arguments
    ///
    /// * `config` - Provider configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn for_ollama(config: &ProviderConfig) -> Result<Self, LLMError> {
        Self::new("http://localhost:11434/v1".to_string(), config)
    }

    /// Creates a client configured for OpenAI.
    ///
    /// # Arguments
    ///
    /// * `config` - Provider configuration with API key
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn for_openai(config: &ProviderConfig) -> Result<Self, LLMError> {
        Self::new("https://api.openai.com/v1".to_string(), config)
    }

    /// Returns the chat completions endpoint URL.
    fn chat_completions_endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Converts internal messages to OpenAI API format.
    fn convert_messages(&self, messages: &[Message]) -> Vec<OpenAIMessage> {
        messages
            .iter()
            .map(|msg| match msg.role {
                MessageRole::System => OpenAIMessage {
                    role: "system".to_string(),
                    content: Some(msg.content.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                MessageRole::User => OpenAIMessage {
                    role: "user".to_string(),
                    content: Some(msg.content.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                MessageRole::Assistant => {
                    let tool_calls = msg.tool_calls.as_ref().map(|tcs| {
                        tcs.iter()
                            .map(|tc| OpenAIToolCall {
                                id: tc.id.clone(),
                                call_type: "function".to_string(),
                                function: OpenAIFunctionCall {
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.to_string(),
                                },
                            })
                            .collect()
                    });

                    OpenAIMessage {
                        role: "assistant".to_string(),
                        content: if msg.content.is_empty() {
                            None
                        } else {
                            Some(msg.content.clone())
                        },
                        tool_calls,
                        tool_call_id: None,
                    }
                }
                MessageRole::Tool => OpenAIMessage {
                    role: "tool".to_string(),
                    content: Some(msg.content.clone()),
                    tool_calls: None,
                    tool_call_id: msg.tool_call_id.clone(),
                },
            })
            .collect()
    }

    /// Converts tool definitions to OpenAI API format.
    fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<OpenAITool> {
        tools
            .iter()
            .map(|t| OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect()
    }

    /// Parses OpenAI stop reason to internal format.
    #[must_use]
    pub fn parse_stop_reason(reason: Option<&str>) -> StopReason {
        match reason {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            Some("tool_calls") => StopReason::ToolUse,
            _ => StopReason::EndTurn,
        }
    }

    /// Builds the request with optional authorization header.
    fn build_request(
        &self,
        request_body: &ChatCompletionRequest,
    ) -> Result<reqwest::RequestBuilder, LLMError> {
        let mut request = self
            .client
            .post(self.chat_completions_endpoint())
            .header("content-type", "application/json")
            .json(request_body);

        if let Some(ref api_key) = self.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        Ok(request)
    }

    /// Parses an error response from the API.
    async fn parse_error_response(&self, response: reqwest::Response) -> LLMError {
        let status = response.status();
        let status_code = status.as_u16();

        // Check for rate limit
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

        if let Ok(api_error) = serde_json::from_str::<OpenAIErrorResponse>(&error_body) {
            let error_type = api_error.error.error_type.as_deref().unwrap_or("unknown");

            match error_type {
                "authentication_error" | "invalid_api_key" => {
                    LLMError::authentication_failed(&api_error.error.message)
                }
                "invalid_request_error" => LLMError::invalid_request(&api_error.error.message),
                _ => LLMError::api_error(
                    status_code,
                    api_error.error.message,
                    api_error.error.error_type,
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

    /// Parses SSE events from a text chunk.
    fn parse_sse_line(line: &str) -> Option<Result<ChatCompletionChunk, LLMError>> {
        let data = line.strip_prefix("data: ")?;

        if data == "[DONE]" {
            return None;
        }

        Some(
            serde_json::from_str::<ChatCompletionChunk>(data)
                .map_err(|e| LLMError::parse_error(format!("failed to parse SSE event: {}", e))),
        )
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn send_request(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<LLMClientResponse, LLMError> {
        let api_messages = self.convert_messages(messages);

        let request_body = ChatCompletionRequest {
            model: self.model.clone(),
            max_tokens: Some(self.max_tokens),
            messages: api_messages,
            tools: tools.map(|t| self.convert_tools(t)),
            stream: false,
        };

        let request = self.build_request(&request_body)?;

        let response = request
            .send()
            .await
            .map_err(|e| LLMError::network(format!("request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(self.parse_error_response(response).await);
        }

        let completion: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| LLMError::parse_error(format!("failed to parse response: {}", e)))?;

        let choice = completion.choices.first().ok_or_else(|| {
            LLMError::parse_error("response contained no choices".to_string())
        })?;

        let content = choice.message.content.clone().unwrap_or_default();

        let tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .filter_map(|tc| {
                        let arguments: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).ok()?;
                        Some(ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let stop_reason = Self::parse_stop_reason(choice.finish_reason.as_deref());

        Ok(LLMClientResponse {
            content,
            tool_calls,
            stop_reason,
        })
    }

    async fn send_streaming_request(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<LLMEventStream, LLMError> {
        let api_messages = self.convert_messages(messages);

        let request_body = ChatCompletionRequest {
            model: self.model.clone(),
            max_tokens: Some(self.max_tokens),
            messages: api_messages,
            tools: tools.map(|t| self.convert_tools(t)),
            stream: true,
        };

        let request = self.build_request(&request_body)?;

        let response = request
            .send()
            .await
            .map_err(|e| LLMError::network(format!("request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(self.parse_error_response(response).await);
        }

        let stream = response.bytes_stream();

        // State for accumulating tool calls across chunks
        let tool_call_accumulators: std::sync::Arc<
            tokio::sync::Mutex<std::collections::HashMap<usize, ToolCallAccumulator>>,
        > = std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new()));

        let event_stream = stream.flat_map(move |result| {
            let tool_call_accumulators = tool_call_accumulators.clone();

            let events: Vec<Result<LLMStreamEvent, LLMError>> = match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut events = Vec::new();
                    let mut first_id = None;

                    for line in text.lines() {
                        if let Some(chunk_result) = Self::parse_sse_line(line) {
                            match chunk_result {
                                Ok(chunk) => {
                                    // Capture first ID for Start event
                                    if first_id.is_none() && !chunk.id.is_empty() {
                                        first_id = Some(chunk.id.clone());
                                    }

                                    for choice in chunk.choices {
                                        // Handle content delta
                                        if let Some(content) = choice.delta.content {
                                            if !content.is_empty() {
                                                events.push(Ok(LLMStreamEvent::Token {
                                                    text: content,
                                                }));
                                            }
                                        }

                                        // Handle tool call deltas
                                        if let Some(tool_deltas) = choice.delta.tool_calls {
                                            for delta in tool_deltas {
                                                let mut accumulators =
                                                    tool_call_accumulators.blocking_lock();
                                                let acc = accumulators
                                                    .entry(delta.index)
                                                    .or_default();

                                                if let Some(id) = delta.id {
                                                    acc.id = Some(id);
                                                }

                                                if let Some(ref func) = delta.function {
                                                    if let Some(ref name) = func.name {
                                                        acc.name = Some(name.clone());
                                                    }
                                                    if let Some(ref args) = func.arguments {
                                                        acc.arguments.push_str(args);
                                                    }
                                                }
                                            }
                                        }

                                        // Handle finish reason
                                        if let Some(ref reason) = choice.finish_reason {
                                            // Emit any accumulated tool calls
                                            let accumulators =
                                                tool_call_accumulators.blocking_lock();
                                            for acc in accumulators.values() {
                                                if let (Some(id), Some(name)) =
                                                    (&acc.id, &acc.name)
                                                {
                                                    let arguments: serde_json::Value =
                                                        serde_json::from_str(&acc.arguments)
                                                            .unwrap_or(serde_json::json!({}));

                                                    events.push(Ok(LLMStreamEvent::ToolCall {
                                                        tool_call: ToolCall {
                                                            id: id.clone(),
                                                            name: name.clone(),
                                                            arguments,
                                                        },
                                                    }));
                                                }
                                            }

                                            events.push(Ok(LLMStreamEvent::End {
                                                stop_reason: Self::parse_stop_reason(
                                                    Some(reason),
                                                ),
                                            }));
                                        }
                                    }
                                }
                                Err(e) => {
                                    events.push(Err(e));
                                }
                            }
                        }
                    }

                    // Add Start event if we have an ID and haven't emitted one yet
                    if let Some(id) = first_id {
                        events.insert(0, Ok(LLMStreamEvent::Start { id }));
                    }

                    events
                }
                Err(e) => {
                    vec![Err(LLMError::stream_error(format!(
                        "stream read error: {}",
                        e
                    )))]
                }
            };

            futures::stream::iter(events)
        });

        Ok(Box::pin(event_stream))
    }

    fn provider_name(&self) -> &'static str {
        "openai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_client() -> OpenAIClient {
        let config = ProviderConfig::ollama("llama3.2");
        OpenAIClient::for_ollama(&config).unwrap()
    }

    #[test]
    fn openai_client_is_debug() {
        let client = create_test_client();
        let debug_str = format!("{:?}", client);
        assert!(debug_str.contains("OpenAIClient"));
    }

    #[test]
    fn openai_client_for_ollama() {
        let config = ProviderConfig::ollama("llama3.2");
        let client = OpenAIClient::for_ollama(&config).unwrap();
        assert_eq!(client.base_url, "http://localhost:11434/v1");
        assert!(client.api_key.is_none());
        assert_eq!(client.model, "llama3.2");
    }

    #[test]
    fn openai_client_for_openai() {
        let config = ProviderConfig::openai("test-key");
        let client = OpenAIClient::for_openai(&config).unwrap();
        assert_eq!(client.base_url, "https://api.openai.com/v1");
        assert_eq!(client.api_key, Some("test-key".to_string()));
        assert_eq!(client.model, "gpt-4o");
    }

    #[test]
    fn openai_client_chat_completions_endpoint() {
        let client = create_test_client();
        assert_eq!(
            client.chat_completions_endpoint(),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn openai_convert_user_message() {
        let client = create_test_client();
        let messages = vec![Message::user("Hello!")];
        let api_messages = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, "user");
        assert_eq!(api_messages[0].content, Some("Hello!".to_string()));
    }

    #[test]
    fn openai_convert_system_message() {
        let client = create_test_client();
        let messages = vec![Message::system("You are helpful.")];
        let api_messages = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, "system");
        assert_eq!(api_messages[0].content, Some("You are helpful.".to_string()));
    }

    #[test]
    fn openai_convert_assistant_message() {
        let client = create_test_client();
        let messages = vec![Message::assistant("I can help.")];
        let api_messages = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, "assistant");
        assert_eq!(api_messages[0].content, Some("I can help.".to_string()));
    }

    #[test]
    fn openai_convert_assistant_message_with_tools() {
        let client = create_test_client();
        let tool_calls = vec![ToolCall {
            id: "tc_123".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "rust"}),
        }];
        let messages = vec![Message::assistant_with_tools("I'll search.", tool_calls)];
        let api_messages = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, "assistant");
        assert!(api_messages[0].tool_calls.is_some());

        let tool_calls = api_messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tc_123");
        assert_eq!(tool_calls[0].function.name, "search");
    }

    #[test]
    fn openai_convert_tool_response() {
        let client = create_test_client();
        let messages = vec![Message::tool("tc_123", "Search results...")];
        let api_messages = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, "tool");
        assert_eq!(api_messages[0].tool_call_id, Some("tc_123".to_string()));
        assert_eq!(api_messages[0].content, Some("Search results...".to_string()));
    }

    #[test]
    fn openai_convert_tools() {
        let client = create_test_client();
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
        assert_eq!(api_tools[0].tool_type, "function");
        assert_eq!(api_tools[0].function.name, "calculator");
        assert_eq!(api_tools[0].function.description, "Performs math");
    }

    #[test]
    fn openai_parse_stop_reason_stop() {
        assert_eq!(
            OpenAIClient::parse_stop_reason(Some("stop")),
            StopReason::EndTurn
        );
    }

    #[test]
    fn openai_parse_stop_reason_length() {
        assert_eq!(
            OpenAIClient::parse_stop_reason(Some("length")),
            StopReason::MaxTokens
        );
    }

    #[test]
    fn openai_parse_stop_reason_tool_calls() {
        assert_eq!(
            OpenAIClient::parse_stop_reason(Some("tool_calls")),
            StopReason::ToolUse
        );
    }

    #[test]
    fn openai_parse_stop_reason_unknown_defaults_to_end_turn() {
        assert_eq!(
            OpenAIClient::parse_stop_reason(Some("unknown")),
            StopReason::EndTurn
        );
        assert_eq!(OpenAIClient::parse_stop_reason(None), StopReason::EndTurn);
    }

    #[test]
    fn openai_parse_sse_text_delta() {
        let line =
            r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#;

        let result = OpenAIClient::parse_sse_line(line).unwrap().unwrap();
        assert_eq!(result.id, "chatcmpl-123");
        assert_eq!(result.choices.len(), 1);
        assert_eq!(result.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn openai_parse_sse_done_marker() {
        let line = "data: [DONE]";
        let result = OpenAIClient::parse_sse_line(line);
        assert!(result.is_none());
    }

    #[test]
    fn openai_parse_sse_finish_reason() {
        let line = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;

        let result = OpenAIClient::parse_sse_line(line).unwrap().unwrap();
        assert_eq!(result.choices[0].finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn openai_parse_sse_tool_call() {
        let line = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","function":{"name":"search"}}]}}]}"#;

        let result = OpenAIClient::parse_sse_line(line).unwrap().unwrap();
        let tool_calls = result.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, Some("call_123".to_string()));
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name,
            Some("search".to_string())
        );
    }

    #[test]
    fn openai_client_implements_llm_client() {
        let config = ProviderConfig::ollama("llama3.2");
        let client = OpenAIClient::for_ollama(&config).unwrap();
        let _boxed: Box<dyn LLMClient> = Box::new(client);
    }

    #[test]
    fn openai_client_provider_name() {
        let client = create_test_client();
        assert_eq!(client.provider_name(), "openai");
    }
}
