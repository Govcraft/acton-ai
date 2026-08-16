//! Anthropic API client.
//!
//! HTTP client for communicating with the Anthropic Claude API,
//! including streaming SSE response handling.

use crate::llm::client::{LLMClient, LLMClientResponse, LLMEventStream, LLMStreamEvent};
use crate::llm::config::{ProviderConfig, SamplingParams};
use crate::llm::error::LLMError;
use crate::messages::{
    Message, MessageRole, StopReason, ToolCall, ToolChoice, ToolDefinition, Usage,
};
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ApiToolChoice>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
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

/// Tool-choice constraint in the Anthropic Messages API format.
///
/// The API models this as an object with a discriminating `type` key, so it
/// serializes as an internally tagged enum: `{"type":"auto"}`,
/// `{"type":"any"}`, `{"type":"tool","name":"…"}`, `{"type":"none"}`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiToolChoice {
    Auto,
    Any,
    Tool { name: String },
    None,
}

impl From<&ToolChoice> for ApiToolChoice {
    fn from(choice: &ToolChoice) -> Self {
        match choice {
            ToolChoice::Auto => Self::Auto,
            ToolChoice::Any => Self::Any,
            ToolChoice::Tool(name) => Self::Tool { name: name.clone() },
            ToolChoice::None => Self::None,
        }
    }
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
    #[serde(default)]
    pub usage: ApiUsage,
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

/// Usage statistics as they appear on the Anthropic wire.
///
/// Every field is optional because the API reports different subsets in
/// different places: `message_start` carries input and cache counts (plus a
/// provisional `output_tokens`), while `message_delta` carries the final
/// cumulative `output_tokens` and usually nothing else. Absent fields must
/// never fail deserialization, so each is an `Option` that
/// [`ApiUsage::apply_to`] simply skips.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ApiUsage {
    /// Uncached input tokens.
    #[serde(default)]
    pub input_tokens: Option<u64>,
    /// Output tokens generated so far (cumulative within a stream).
    #[serde(default)]
    pub output_tokens: Option<u64>,
    /// Input tokens served from the prompt cache.
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    /// Input tokens written into the prompt cache.
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
}

impl ApiUsage {
    /// Folds whatever this event reported onto `total`.
    ///
    /// Each field **overwrites** rather than adds, because Anthropic reports
    /// cumulative figures within one stream: `message_delta` restates the
    /// running `output_tokens` total, so adding would double-count. A field
    /// the event omits leaves the running total untouched, which is what
    /// keeps `message_start`'s input count alive through the final
    /// `message_delta`.
    pub fn apply_to(&self, total: &mut Usage) {
        if let Some(input) = self.input_tokens {
            total.input_tokens = input;
        }
        if let Some(output) = self.output_tokens {
            total.output_tokens = output;
        }
        if let Some(read) = self.cache_read_input_tokens {
            total.cache_read_tokens = read;
        }
        if let Some(creation) = self.cache_creation_input_tokens {
            total.cache_creation_tokens = creation;
        }
    }

    /// Converts a standalone (non-streaming) usage object.
    #[must_use]
    pub fn to_usage(&self) -> Usage {
        let mut usage = Usage::default();
        self.apply_to(&mut usage);
        usage
    }
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
        /// Usage reported alongside the opening message: input tokens and,
        /// when prompt caching is in play, the cache counters.
        usage: ApiUsage,
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
        /// Final cumulative usage for the message. Note this rides on the
        /// event itself, **not** inside `delta` — verified against the
        /// Anthropic SSE reference.
        usage: ApiUsage,
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
    /// Top-level usage. Present on `message_delta`; `message_start` instead
    /// nests its usage inside `message`.
    #[serde(default)]
    usage: Option<ApiUsage>,
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
        sampling: Option<&SamplingParams>,
        tool_choice: Option<&ToolChoice>,
    ) -> Result<MessagesResponse, LLMError> {
        let (system, api_messages) = self.convert_messages(messages);

        let request_body = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            system,
            messages: api_messages,
            tools: tools.map(|t| self.convert_tools(t)),
            tool_choice: tool_choice.map(ApiToolChoice::from),
            stream: false,
            temperature: sampling.and_then(|s| s.temperature),
            top_k: sampling.and_then(|s| s.top_k),
            top_p: sampling.and_then(|s| s.top_p),
            stop_sequences: sampling.and_then(|s| s.stop_sequences.clone()),
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
        sampling: Option<&SamplingParams>,
        tool_choice: Option<&ToolChoice>,
    ) -> Result<impl futures::Stream<Item = Result<StreamEvent, LLMError>>, LLMError> {
        let (system, api_messages) = self.convert_messages(messages);

        let request_body = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            system,
            messages: api_messages,
            tools: tools.map(|t| self.convert_tools(t)),
            tool_choice: tool_choice.map(ApiToolChoice::from),
            stream: true,
            temperature: sampling.and_then(|s| s.temperature),
            top_k: sampling.and_then(|s| s.top_k),
            top_p: sampling.and_then(|s| s.top_p),
            stop_sequences: sampling.and_then(|s| s.stop_sequences.clone()),
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
    ///
    /// The history is structurally repaired first (see
    /// [`crate::llm::sanitize`]), so this only has the wire format to worry
    /// about. Two Anthropic rules shape what follows:
    ///
    /// - A text block must be non-empty, so a tool-calling turn with no
    ///   preamble contributes only its `tool_use` blocks.
    /// - Tool results are `user` messages, and every result answering one
    ///   assistant turn must ride in a *single* user message. Adjacent
    ///   same-role messages are therefore coalesced rather than pushed
    ///   individually.
    fn convert_messages(&self, messages: &[Message]) -> (Option<String>, Vec<ApiMessage>) {
        let messages = crate::llm::sanitize::sanitize_history(messages);
        let mut system = None;
        let mut api_messages: Vec<ApiMessage> = Vec::new();

        for msg in &messages {
            match msg.role {
                MessageRole::System => {
                    system = Some(msg.content.clone());
                }
                MessageRole::User => {
                    push_or_coalesce(
                        &mut api_messages,
                        "user",
                        ApiContent::Text(msg.content.clone()),
                    );
                }
                MessageRole::Assistant => {
                    let content = if let Some(tool_calls) = &msg.tool_calls {
                        let mut blocks = Vec::with_capacity(tool_calls.len() + 1);
                        // An empty text block is rejected outright, and a
                        // tool call with no preamble text is the common case.
                        if !msg.content.trim().is_empty() {
                            blocks.push(ContentBlock::Text {
                                text: msg.content.clone(),
                            });
                        }
                        blocks.extend(tool_calls.iter().map(|tc| ContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            input: tc.arguments.clone(),
                        }));
                        ApiContent::Blocks(blocks)
                    } else {
                        ApiContent::Text(msg.content.clone())
                    };

                    push_or_coalesce(&mut api_messages, "assistant", content);
                }
                MessageRole::Tool => {
                    if let Some(tool_call_id) = &msg.tool_call_id {
                        push_or_coalesce(
                            &mut api_messages,
                            "user",
                            ApiContent::Blocks(vec![ContentBlock::ToolResult {
                                tool_use_id: tool_call_id.clone(),
                                content: msg.content.clone(),
                            }]),
                        );
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
                let message = raw.message;
                let id = message
                    .as_ref()
                    .and_then(|m| m.get("id").and_then(|v| v.as_str().map(String::from)))
                    .unwrap_or_default();
                // A usage block that fails to parse is not worth failing the
                // whole stream over — degrade to "reported nothing".
                let usage = message
                    .and_then(|m| m.get("usage").cloned())
                    .and_then(|u| serde_json::from_value::<ApiUsage>(u).ok())
                    .unwrap_or_default();
                Ok(Some(StreamEvent::MessageStart { id, usage }))
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
                Ok(Some(StreamEvent::MessageDelta {
                    stop_reason,
                    usage: raw.usage.unwrap_or_default(),
                }))
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
        sampling: Option<&SamplingParams>,
        tool_choice: Option<&ToolChoice>,
    ) -> Result<LLMClientResponse, LLMError> {
        let response = self
            .send_messages(messages, tools, sampling, tool_choice)
            .await?;
        Ok(LLMClientResponse {
            content: extract_text_content(&response),
            tool_calls: extract_tool_calls(&response),
            stop_reason: response
                .stop_reason
                .as_ref()
                .map(|s| parse_stop_reason(s))
                .unwrap_or(StopReason::EndTurn),
            usage: response.usage.to_usage(),
        })
    }

    async fn send_streaming_request(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        sampling: Option<&SamplingParams>,
        tool_choice: Option<&ToolChoice>,
    ) -> Result<LLMEventStream, LLMError> {
        let stream = self
            .send_messages_streaming(messages, tools, sampling, tool_choice)
            .await?;
        Ok(Box::pin(convert_anthropic_stream(stream)))
    }

    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
}

/// Converts Anthropic stream events to unified LLMStreamEvent.
///
/// Two things have to survive across events, so both ride in a [`StreamState`]
/// threaded through with `scan`:
///
/// - Usage arrives split across two events — `message_start` carries the input
///   and cache counts, `message_delta` the final output count — and is attached
///   to whichever terminal event closes the round.
/// - A `tool_use` block is announced, streamed as JSON fragments, then closed,
///   so a tool call is only whole at `content_block_stop`.
fn convert_anthropic_stream(
    stream: impl futures::Stream<Item = Result<StreamEvent, LLMError>> + Send + 'static,
) -> impl futures::Stream<Item = Result<LLMStreamEvent, LLMError>> + Send {
    stream
        .scan(StreamState::default(), |state, result| {
            // Never yields `None`: the stream ends when the source does, not
            // when an event happens to carry nothing worth emitting.
            futures::future::ready(Some(convert_one_event(state, result)))
        })
        .filter_map(futures::future::ready)
}

/// Appends `content` to the last API message when it shares `role`, otherwise
/// starts a new message.
///
/// Anthropic requires alternating roles, and requires every tool result
/// answering one assistant turn to arrive in a single user message. Both fall
/// out of coalescing on the way in.
fn push_or_coalesce(messages: &mut Vec<ApiMessage>, role: &str, content: ApiContent) {
    if let Some(last) = messages.last_mut() {
        if last.role == role {
            append_content(&mut last.content, content);
            return;
        }
    }
    messages.push(ApiMessage {
        role: role.to_string(),
        content,
    });
}

/// Concatenates two content bodies, promoting plain text to a block list so
/// the two representations can be joined.
fn append_content(existing: &mut ApiContent, incoming: ApiContent) {
    let mut blocks = match std::mem::replace(existing, ApiContent::Blocks(Vec::new())) {
        ApiContent::Text(text) => vec![ContentBlock::Text { text }],
        ApiContent::Blocks(blocks) => blocks,
    };

    match incoming {
        ApiContent::Text(text) => blocks.push(ContentBlock::Text { text }),
        ApiContent::Blocks(incoming) => blocks.extend(incoming),
    }

    *existing = ApiContent::Blocks(blocks);
}

/// A `tool_use` content block being assembled across SSE deltas.
///
/// Anthropic announces the id and name in `content_block_start`, then streams
/// the arguments as a sequence of `input_json_delta` fragments that are only
/// valid JSON once concatenated.
#[derive(Debug, Clone)]
struct ToolBlock {
    id: String,
    name: String,
    /// Concatenated `partial_json` fragments. Empty for a tool taking no
    /// arguments, which Anthropic streams with no deltas at all.
    json: String,
}

/// State threaded through the SSE stream by [`Self::convert_stream`].
#[derive(Debug, Default)]
struct StreamState {
    /// Running usage total, folded from whichever events report it.
    usage: Usage,
    /// Tool blocks open right now, keyed by their content-block index.
    /// Anthropic may interleave blocks, so this cannot be a single slot.
    tools: std::collections::HashMap<usize, ToolBlock>,
}

/// Folds one Anthropic event into the running stream state and maps it onto
/// the unified event type. Returns `None` for events with no unified
/// counterpart (pings, the boundaries of a text block).
fn convert_one_event(
    state: &mut StreamState,
    result: Result<StreamEvent, LLMError>,
) -> Option<Result<LLMStreamEvent, LLMError>> {
    let event = match result {
        Ok(event) => event,
        Err(e) => return Some(Err(e)),
    };

    match event {
        StreamEvent::MessageStart { id, usage } => {
            usage.apply_to(&mut state.usage);
            Some(Ok(LLMStreamEvent::Start { id }))
        }
        StreamEvent::ContentBlockStart {
            index,
            block_type,
            tool_id,
            tool_name,
        } => {
            // Only `tool_use` blocks carry an id and a name; a text block
            // opens with neither and needs no state.
            if block_type == "tool_use" {
                if let (Some(id), Some(name)) = (tool_id, tool_name) {
                    state.tools.insert(
                        index,
                        ToolBlock {
                            id,
                            name,
                            json: String::new(),
                        },
                    );
                }
            }
            None
        }
        StreamEvent::ContentBlockDelta {
            index,
            text,
            partial_json,
            ..
        } => {
            // An argument fragment belongs to the tool block at this index,
            // never to the caller's token stream.
            if let Some(fragment) = partial_json {
                if let Some(block) = state.tools.get_mut(&index) {
                    block.json.push_str(&fragment);
                }
                return None;
            }
            text.map(|t| Ok(LLMStreamEvent::Token { text: t }))
        }
        StreamEvent::ContentBlockStop { index } => {
            // The close of a tool block is the only point at which its
            // arguments are complete enough to parse.
            let block = state.tools.remove(&index)?;
            let arguments = if block.json.trim().is_empty() {
                serde_json::Value::Object(serde_json::Map::new())
            } else {
                match serde_json::from_str(&block.json) {
                    Ok(value) => value,
                    // Never fall back to empty arguments: that runs the tool
                    // with inputs the model did not ask for.
                    Err(e) => {
                        return Some(Err(LLMError::parse_error(format!(
                            "tool call `{}` streamed unparseable arguments: {e}",
                            block.name
                        ))))
                    }
                }
            };
            Some(Ok(LLMStreamEvent::ToolCall {
                tool_call: ToolCall {
                    id: block.id,
                    name: block.name,
                    arguments,
                },
            }))
        }
        StreamEvent::MessageDelta { stop_reason, usage } => {
            usage.apply_to(&mut state.usage);
            let total = state.usage;
            stop_reason.map(|reason| {
                Ok(LLMStreamEvent::End {
                    stop_reason: parse_stop_reason(&reason),
                    usage: total,
                })
            })
        }
        StreamEvent::MessageStop => Some(Ok(LLMStreamEvent::End {
            stop_reason: StopReason::EndTurn,
            usage: state.usage,
        })),
        StreamEvent::Error {
            error_type,
            message,
        } => Some(Ok(LLMStreamEvent::Error {
            error_type,
            message,
        })),
        StreamEvent::Ping => None,
    }
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

    fn tool_choice_body(choice: Option<&ToolChoice>) -> serde_json::Value {
        let request = MessagesRequest {
            model: "claude-test".to_string(),
            max_tokens: 16,
            system: None,
            messages: Vec::new(),
            tools: None,
            tool_choice: choice.map(ApiToolChoice::from),
            stream: false,
            temperature: None,
            top_k: None,
            top_p: None,
            stop_sequences: None,
        };
        serde_json::to_value(&request).unwrap()
    }

    #[test]
    fn anthropic_tool_choice_auto_serializes_as_typed_object() {
        let body = tool_choice_body(Some(&ToolChoice::Auto));
        assert_eq!(body["tool_choice"], serde_json::json!({"type": "auto"}));
    }

    #[test]
    fn anthropic_tool_choice_any_serializes_as_typed_object() {
        let body = tool_choice_body(Some(&ToolChoice::Any));
        assert_eq!(body["tool_choice"], serde_json::json!({"type": "any"}));
    }

    #[test]
    fn anthropic_tool_choice_named_tool_serializes_with_name() {
        let body = tool_choice_body(Some(&ToolChoice::Tool("structured_output".to_string())));
        assert_eq!(
            body["tool_choice"],
            serde_json::json!({"type": "tool", "name": "structured_output"})
        );
    }

    #[test]
    fn anthropic_tool_choice_none_serializes_as_typed_object() {
        let body = tool_choice_body(Some(&ToolChoice::None));
        assert_eq!(body["tool_choice"], serde_json::json!({"type": "none"}));
    }

    #[test]
    fn anthropic_tool_choice_absent_omits_the_key() {
        let body = tool_choice_body(None);
        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice must not appear in the body when unset: {body}"
        );
    }

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
    fn tool_call_with_no_preamble_emits_no_empty_text_block() {
        let config = ProviderConfig::new("test-key");
        let client = AnthropicClient::new(config).unwrap();

        let messages = vec![
            Message::user("search for rust"),
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "tc_1".to_string(),
                    name: "search".to_string(),
                    arguments: serde_json::json!({}),
                }],
            ),
            Message::tool("tc_1", "results"),
        ];

        let (_, api_messages) = client.convert_messages(&messages);
        let wire = serde_json::to_string(&api_messages).unwrap();

        assert!(
            !wire.contains(r#"{"type":"text","text":""}"#),
            "empty text block reached the wire: {wire}"
        );
        assert!(wire.contains("tool_use"));
    }

    #[test]
    fn parallel_tool_results_ride_in_one_user_message() {
        let config = ProviderConfig::new("test-key");
        let client = AnthropicClient::new(config).unwrap();

        let calls = vec![
            ToolCall {
                id: "tc_1".to_string(),
                name: "search".to_string(),
                arguments: serde_json::json!({}),
            },
            ToolCall {
                id: "tc_2".to_string(),
                name: "search".to_string(),
                arguments: serde_json::json!({}),
            },
        ];
        let messages = vec![
            Message::user("search twice"),
            Message::assistant_with_tools("on it", calls),
            Message::tool("tc_1", "first"),
            Message::tool("tc_2", "second"),
        ];

        let (_, api_messages) = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 3);
        assert_eq!(api_messages[2].role, "user");
        let ApiContent::Blocks(blocks) = &api_messages[2].content else {
            panic!("tool results must serialize as blocks")
        };
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn roles_never_repeat_on_the_wire() {
        let config = ProviderConfig::new("test-key");
        let client = AnthropicClient::new(config).unwrap();

        let messages = vec![
            Message::user("extract the data"),
            Message::user("record your answer"),
        ];

        let (_, api_messages) = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert!(api_messages
            .windows(2)
            .all(|pair| pair[0].role != pair[1].role));
    }

    #[test]
    fn dangling_tool_call_never_reaches_the_wire() {
        let config = ProviderConfig::new("test-key");
        let client = AnthropicClient::new(config).unwrap();

        // The shape a turn leaves behind when it dies between issuing a call
        // and recording its result.
        let messages = vec![
            Message::user("search for rust"),
            Message::assistant_with_tools(
                "looking",
                vec![ToolCall {
                    id: "tc_1".to_string(),
                    name: "search".to_string(),
                    arguments: serde_json::json!({}),
                }],
            ),
            Message::user("never mind"),
        ];

        let (_, api_messages) = client.convert_messages(&messages);
        let wire = serde_json::to_string(&api_messages).unwrap();

        assert!(!wire.contains("tool_use"), "dangling call survived: {wire}");
    }

    #[test]
    fn streaming_reassembles_a_tool_call_from_json_fragments() {
        let events = events_from_sse(&[
            r#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":10}}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tc_1","name":"search"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"rust\"}"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}"#,
        ]);

        let tool_call = events
            .iter()
            .find_map(|event| match event {
                Ok(LLMStreamEvent::ToolCall { tool_call }) => Some(tool_call),
                _ => None,
            })
            .expect("the stream must yield the tool call it described");

        assert_eq!(tool_call.id, "tc_1");
        assert_eq!(tool_call.name, "search");
        assert_eq!(tool_call.arguments["query"], "rust");
    }

    #[test]
    fn streaming_tool_call_with_no_arguments_yields_an_empty_object() {
        let events = events_from_sse(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tc_1","name":"now"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
        ]);

        let tool_call = events
            .iter()
            .find_map(|event| match event {
                Ok(LLMStreamEvent::ToolCall { tool_call }) => Some(tool_call),
                _ => None,
            })
            .expect("a tool taking no arguments still produces a call");

        assert_eq!(tool_call.arguments, serde_json::json!({}));
    }

    #[test]
    fn truncated_tool_arguments_error_rather_than_running_empty() {
        let events = events_from_sse(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tc_1","name":"bash"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"rm -"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
        ]);

        assert!(
            events.iter().any(Result::is_err),
            "a half-streamed argument list must not be rounded down to {{}}"
        );
        assert!(events
            .iter()
            .all(|event| !matches!(event, Ok(LLMStreamEvent::ToolCall { .. }))));
    }

    #[test]
    fn text_deltas_are_not_confused_with_tool_arguments() {
        let events = events_from_sse(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
        ]);

        let tokens: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Ok(LLMStreamEvent::Token { text }) => Some(text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(tokens, vec!["hello"]);
        assert!(events
            .iter()
            .all(|event| !matches!(event, Ok(LLMStreamEvent::ToolCall { .. }))));
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

    // -------------------------------------------------------------------
    // Usage extraction
    //
    // Wire shapes below are the documented Anthropic SSE payloads. The one
    // that is easy to get wrong: on `message_delta`, `usage` is a **sibling**
    // of `delta`, not a field inside it.
    // -------------------------------------------------------------------

    /// Runs a scripted set of SSE lines through the same conversion the
    /// client uses, returning the usage attached to the terminal event.
    fn usage_from_sse(lines: &[&str]) -> Usage {
        let mut ended = None;

        for event in events_from_sse(lines) {
            if let Ok(LLMStreamEvent::End { usage, .. }) = event {
                ended = Some(usage);
            }
        }

        ended.expect("the scripted stream must produce a terminal event")
    }

    /// Runs a scripted set of SSE lines through the same conversion the
    /// client uses, returning every unified event it produced.
    fn events_from_sse(lines: &[&str]) -> Vec<Result<LLMStreamEvent, LLMError>> {
        let mut state = StreamState::default();

        lines
            .iter()
            .flat_map(|line| AnthropicClient::parse_sse_events(line).unwrap())
            .filter_map(|event| convert_one_event(&mut state, Ok(event)))
            .collect()
    }

    #[test]
    fn message_start_carries_input_and_cache_tokens() {
        let usage = usage_from_sse(&[
            r#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":25,"output_tokens":1,"cache_read_input_tokens":100,"cache_creation_input_tokens":40}}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}"#,
        ]);

        assert_eq!(usage.input_tokens, 25);
        assert_eq!(usage.cache_read_tokens, 100);
        assert_eq!(usage.cache_creation_tokens, 40);
    }

    #[test]
    fn message_delta_output_tokens_replace_the_provisional_count() {
        // message_start reports output_tokens=1 as a placeholder; the final
        // figure on message_delta is cumulative, so it must overwrite rather
        // than add — 15, never 16.
        let usage = usage_from_sse(&[
            r#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":25,"output_tokens":1}}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}"#,
        ]);

        assert_eq!(usage.output_tokens, 15);
    }

    #[test]
    fn message_delta_usage_is_read_from_the_event_not_the_delta() {
        // A parser that looked inside `delta` would report 0 here.
        let usage = usage_from_sse(&[
            r#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":7}}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":99}}"#,
        ]);

        assert_eq!(usage.output_tokens, 99);
    }

    #[test]
    fn absent_cache_fields_default_to_zero() {
        let usage = usage_from_sse(&[
            r#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":25,"output_tokens":1}}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}"#,
        ]);

        assert_eq!(usage.cache_read_tokens, 0);
        assert_eq!(usage.cache_creation_tokens, 0);
    }

    #[test]
    fn a_stream_with_no_usage_at_all_degrades_to_default() {
        let usage = usage_from_sse(&[
            r#"data: {"type":"message_start","message":{"id":"msg_1"}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        ]);

        assert_eq!(usage, Usage::default());
    }

    #[test]
    fn unknown_usage_fields_do_not_fail_deserialization() {
        // Forward compatibility: a new counter must not break the round.
        let usage = usage_from_sse(&[
            r#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":5,"some_future_counter":123}}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);

        assert_eq!(usage.input_tokens, 5);
    }

    #[test]
    fn message_stop_closes_the_round_with_the_accumulated_usage() {
        let usage = usage_from_sse(&[
            r#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":11,"output_tokens":2}}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);

        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 2);
    }

    #[test]
    fn non_streaming_usage_converts_from_the_wire_shape() {
        let api = ApiUsage {
            input_tokens: Some(3),
            output_tokens: Some(4),
            cache_read_input_tokens: Some(5),
            cache_creation_input_tokens: Some(6),
        };

        assert_eq!(
            api.to_usage(),
            Usage {
                input_tokens: 3,
                output_tokens: 4,
                cache_read_tokens: 5,
                cache_creation_tokens: 6,
            }
        );
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
            usage: ApiUsage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                ..ApiUsage::default()
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
            usage: ApiUsage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                ..ApiUsage::default()
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
