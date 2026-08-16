//! OpenAI-compatible API client.
//!
//! HTTP client for communicating with OpenAI-compatible APIs including
//! OpenAI, Ollama, vLLM, LocalAI, and other compatible endpoints.

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
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    stream: bool,
    /// Only ever set on streaming requests, where it asks the server to send
    /// a final usage chunk. Omitted entirely otherwise — non-streaming
    /// responses carry `usage` unconditionally, and some OpenAI-compatible
    /// servers reject the key on a non-streaming call.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

/// Streaming-only options block.
///
/// `include_usage` asks the server to append a final chunk carrying `usage`
/// and an **empty** `choices` array. Servers that do not implement
/// `stream_options` ignore the key, which is why absent usage has to degrade
/// to [`Usage::default()`] rather than erroring.
#[derive(Debug, Clone, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// Usage as it appears on the OpenAI wire.
#[derive(Debug, Clone, Default, Deserialize)]
struct OpenAIUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Breakdown of `prompt_tokens`. Only OpenAI proper reports this; compatible
/// servers usually omit it.
#[derive(Debug, Clone, Default, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

impl OpenAIUsage {
    /// Converts to the canonical [`Usage`].
    ///
    /// OpenAI's `prompt_tokens` is *inclusive* of cached tokens, whereas
    /// Anthropic's `input_tokens` excludes them. The cached count is
    /// subtracted here so `Usage::input_tokens` means the same thing for both
    /// providers — which is what lets the pricing table apply one input rate
    /// and one cache-read rate without knowing the vendor.
    fn to_usage(&self) -> Usage {
        let cache_read = self
            .prompt_tokens_details
            .as_ref()
            .map_or(0, |details| details.cached_tokens);

        Usage {
            input_tokens: self.prompt_tokens.saturating_sub(cache_read),
            output_tokens: self.completion_tokens,
            cache_read_tokens: cache_read,
            // OpenAI-compatible servers do not report cache writes.
            cache_creation_tokens: 0,
        }
    }
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
    #[serde(default)]
    usage: Option<OpenAIUsage>,
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
    /// Empty on the final usage chunk when `stream_options.include_usage` is
    /// set — the parser must tolerate that rather than index `choices[0]`.
    #[serde(default)]
    choices: Vec<ChatCompletionChunkChoice>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
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

/// Parses the arguments of a tool call, which this API carries as a JSON
/// string rather than as JSON.
///
/// An absent or blank argument string means "no arguments" and yields an empty
/// object. Anything else that fails to parse is an error: a truncated or
/// malformed fragment must never be silently rounded down to `{}`, because the
/// tool would then run against inputs the model never supplied.
fn parse_tool_arguments(tool_name: &str, raw: &str) -> Result<serde_json::Value, LLMError> {
    if raw.trim().is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }

    serde_json::from_str(raw).map_err(|e| {
        LLMError::parse_error(format!(
            "tool call `{tool_name}` supplied unparseable arguments: {e}"
        ))
    })
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
    /// Converts internal messages to API format.
    ///
    /// The history is structurally repaired first (see
    /// [`crate::llm::sanitize`]) so that no `tool` message arrives without the
    /// `tool_call_id` this format requires, and no `tool_calls` array is left
    /// without its answering messages.
    fn convert_messages(&self, messages: &[Message]) -> Vec<OpenAIMessage> {
        crate::llm::sanitize::sanitize_history(messages)
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

    /// Encodes a [`ToolChoice`] in the OpenAI chat-completions wire format.
    ///
    /// OpenAI accepts either a bare string (`"auto"`, `"required"`, `"none"`)
    /// or an object naming one function, so the return type is a raw
    /// [`serde_json::Value`] rather than a typed struct. Callers pass
    /// `None` through untouched so the key is omitted entirely.
    fn convert_tool_choice(choice: &ToolChoice) -> serde_json::Value {
        match choice {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::Any => serde_json::json!("required"),
            ToolChoice::Tool(name) => serde_json::json!({
                "type": "function",
                "function": { "name": name },
            }),
            ToolChoice::None => serde_json::json!("none"),
        }
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
        sampling: Option<&SamplingParams>,
        tool_choice: Option<&ToolChoice>,
    ) -> Result<LLMClientResponse, LLMError> {
        let api_messages = self.convert_messages(messages);

        let request_body = ChatCompletionRequest {
            model: self.model.clone(),
            max_tokens: Some(self.max_tokens),
            messages: api_messages,
            tools: tools.map(|t| self.convert_tools(t)),
            tool_choice: tool_choice.map(Self::convert_tool_choice),
            stream: false,
            stream_options: None,
            temperature: sampling.and_then(|s| s.temperature),
            top_p: sampling.and_then(|s| s.top_p),
            frequency_penalty: sampling.and_then(|s| s.frequency_penalty),
            presence_penalty: sampling.and_then(|s| s.presence_penalty),
            seed: sampling.and_then(|s| s.seed),
            stop: sampling.and_then(|s| s.stop_sequences.clone()),
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

        let choice = completion
            .choices
            .first()
            .ok_or_else(|| LLMError::parse_error("response contained no choices".to_string()))?;

        let content = choice.message.content.clone().unwrap_or_default();

        // A call whose arguments will not parse is an error, not something to
        // quietly leave out: dropping it strands the model mid-plan and the
        // turn ends as though it had simply chosen to answer.
        let tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| {
                        Ok(ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: parse_tool_arguments(
                                &tc.function.name,
                                &tc.function.arguments,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, LLMError>>()
            })
            .transpose()?
            .unwrap_or_default();

        // If tool calls are present, force ToolUse stop reason even if the
        // provider (e.g. Ollama) sent "stop" instead of "tool_calls".
        let stop_reason = if !tool_calls.is_empty() {
            StopReason::ToolUse
        } else {
            Self::parse_stop_reason(choice.finish_reason.as_deref())
        };

        Ok(LLMClientResponse {
            content,
            tool_calls,
            stop_reason,
            usage: completion.usage.map(|u| u.to_usage()).unwrap_or_default(),
        })
    }

    async fn send_streaming_request(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        sampling: Option<&SamplingParams>,
        tool_choice: Option<&ToolChoice>,
    ) -> Result<LLMEventStream, LLMError> {
        use std::collections::{HashMap, VecDeque};

        let api_messages = self.convert_messages(messages);

        let request_body = ChatCompletionRequest {
            model: self.model.clone(),
            max_tokens: Some(self.max_tokens),
            messages: api_messages,
            tools: tools.map(|t| self.convert_tools(t)),
            tool_choice: tool_choice.map(Self::convert_tool_choice),
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature: sampling.and_then(|s| s.temperature),
            top_p: sampling.and_then(|s| s.top_p),
            frequency_penalty: sampling.and_then(|s| s.frequency_penalty),
            presence_penalty: sampling.and_then(|s| s.presence_penalty),
            seed: sampling.and_then(|s| s.seed),
            stop: sampling.and_then(|s| s.stop_sequences.clone()),
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

        // State carried through the unfold iteration.
        // This eliminates the need for Arc<Mutex<>> by owning state directly.
        struct StreamState<S> {
            stream: S,
            tool_accumulators: HashMap<usize, ToolCallAccumulator>,
            pending_events: VecDeque<Result<LLMStreamEvent, LLMError>>,
            /// Usage from the final `stream_options` chunk, if the server
            /// sends one. Stays default when it does not.
            usage: Usage,
            /// The round's stop reason, held back until usage can ride with
            /// it. OpenAI sends the usage chunk *after* the chunk carrying
            /// `finish_reason`, so emitting `End` on sight of the finish
            /// reason would publish a round whose usage had not arrived yet.
            pending_end: Option<StopReason>,
            /// Set by the `data: [DONE]` sentinel, which is the last thing on
            /// the wire and therefore the cue to release `pending_end`
            /// without waiting for the socket to close.
            saw_done: bool,
        }

        let event_stream = futures::stream::unfold(
            StreamState {
                stream,
                tool_accumulators: HashMap::new(),
                pending_events: VecDeque::new(),
                usage: Usage::default(),
                pending_end: None,
                saw_done: false,
            },
            |mut state| async move {
                loop {
                    // Return any pending events first
                    if let Some(event) = state.pending_events.pop_front() {
                        return Some((event, state));
                    }

                    // Get next chunk from stream. A stream that ends
                    // without a `[DONE]` sentinel still has to close its
                    // round, so flush any held terminal event first.
                    let Some(result) = state.stream.next().await else {
                        let stop_reason = state.pending_end.take()?;
                        let usage = state.usage;
                        return Some((Ok(LLMStreamEvent::End { stop_reason, usage }), state));
                    };

                    match result {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            let mut first_id = None;

                            for line in text.lines() {
                                if line.trim() == "data: [DONE]" {
                                    state.saw_done = true;
                                }
                                if let Some(chunk_result) = OpenAIClient::parse_sse_line(line) {
                                    match chunk_result {
                                        Ok(chunk) => {
                                            // Capture first ID for Start event
                                            if first_id.is_none() && !chunk.id.is_empty() {
                                                first_id = Some(chunk.id.clone());
                                            }

                                            // The final `include_usage` chunk
                                            // carries usage and an EMPTY
                                            // `choices` array; the loop below
                                            // simply runs zero times for it.
                                            if let Some(usage) = chunk.usage {
                                                state.usage = usage.to_usage();
                                            }

                                            for choice in chunk.choices {
                                                // Handle content delta
                                                if let Some(content) = choice.delta.content {
                                                    if !content.is_empty() {
                                                        state.pending_events.push_back(Ok(
                                                            LLMStreamEvent::Token { text: content },
                                                        ));
                                                    }
                                                }

                                                // Handle tool call deltas
                                                if let Some(tool_deltas) = choice.delta.tool_calls {
                                                    for delta in tool_deltas {
                                                        let acc = state
                                                            .tool_accumulators
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
                                                    let mut emitted_tool_calls = false;
                                                    for acc in state.tool_accumulators.values() {
                                                        if let (Some(id), Some(name)) =
                                                            (&acc.id, &acc.name)
                                                        {
                                                            // A stream cut mid-JSON leaves a
                                                            // fragment. Defaulting it to `{}`
                                                            // would run the tool with arguments
                                                            // the model never sent, so the
                                                            // round fails instead.
                                                            let arguments =
                                                                match parse_tool_arguments(
                                                                    name,
                                                                    &acc.arguments,
                                                                ) {
                                                                    Ok(arguments) => arguments,
                                                                    Err(e) => {
                                                                        state
                                                                            .pending_events
                                                                            .push_back(Err(e));
                                                                        continue;
                                                                    }
                                                                };

                                                            state.pending_events.push_back(Ok(
                                                                LLMStreamEvent::ToolCall {
                                                                    tool_call: ToolCall {
                                                                        id: id.clone(),
                                                                        name: name.clone(),
                                                                        arguments,
                                                                    },
                                                                },
                                                            ));
                                                            emitted_tool_calls = true;
                                                        }
                                                    }

                                                    // If tool calls were emitted, force ToolUse
                                                    // stop reason even if the provider (e.g. Ollama)
                                                    // sent "stop" instead of "tool_calls".
                                                    let stop_reason = if emitted_tool_calls {
                                                        StopReason::ToolUse
                                                    } else {
                                                        OpenAIClient::parse_stop_reason(Some(
                                                            reason,
                                                        ))
                                                    };

                                                    // Held back until the
                                                    // usage chunk (or the end
                                                    // of the stream) arrives.
                                                    state.pending_end = Some(stop_reason);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            state.pending_events.push_back(Err(e));
                                        }
                                    }
                                }
                            }

                            // Add Start event at the front if we have an ID
                            if let Some(id) = first_id {
                                state
                                    .pending_events
                                    .push_front(Ok(LLMStreamEvent::Start { id }));
                            }

                            // `[DONE]` is the last thing the server sends, so
                            // whatever usage was going to arrive has arrived.
                            if state.saw_done {
                                if let Some(stop_reason) = state.pending_end.take() {
                                    let usage = state.usage;
                                    state
                                        .pending_events
                                        .push_back(Ok(LLMStreamEvent::End { stop_reason, usage }));
                                }
                            }

                            // Loop continues to return first pending event (or get next chunk if none)
                        }
                        Err(e) => {
                            return Some((
                                Err(LLMError::stream_error(format!("stream read error: {}", e))),
                                state,
                            ));
                        }
                    }
                }
            },
        );

        Ok(Box::pin(event_stream))
    }

    fn provider_name(&self) -> &'static str {
        "openai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_choice_body(choice: Option<&ToolChoice>) -> serde_json::Value {
        let request = ChatCompletionRequest {
            model: "gpt-test".to_string(),
            messages: Vec::new(),
            max_tokens: Some(16),
            tools: None,
            tool_choice: choice.map(OpenAIClient::convert_tool_choice),
            stream: false,
            stream_options: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop: None,
        };
        serde_json::to_value(&request).unwrap()
    }

    #[test]
    fn openai_tool_choice_auto_serializes_as_string() {
        assert_eq!(
            tool_choice_body(Some(&ToolChoice::Auto))["tool_choice"],
            "auto"
        );
    }

    #[test]
    fn openai_tool_choice_any_serializes_as_required() {
        assert_eq!(
            tool_choice_body(Some(&ToolChoice::Any))["tool_choice"],
            "required"
        );
    }

    #[test]
    fn openai_tool_choice_named_tool_serializes_as_function_object() {
        let body = tool_choice_body(Some(&ToolChoice::Tool("structured_output".to_string())));
        assert_eq!(
            body["tool_choice"],
            serde_json::json!({
                "type": "function",
                "function": {"name": "structured_output"}
            })
        );
    }

    #[test]
    fn openai_tool_choice_none_serializes_as_string() {
        assert_eq!(
            tool_choice_body(Some(&ToolChoice::None))["tool_choice"],
            "none"
        );
    }

    #[test]
    fn openai_tool_choice_absent_omits_the_key() {
        let body = tool_choice_body(None);
        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice must not appear in the body when unset: {body}"
        );
    }

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
        assert_eq!(
            api_messages[0].content,
            Some("You are helpful.".to_string())
        );
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
        // A complete exchange: a call with no answering result is stripped
        // before serialization, so the fixture has to include the result.
        let messages = vec![
            Message::user("search for rust"),
            Message::assistant_with_tools("I'll search.", tool_calls),
            Message::tool("tc_123", "Search results..."),
        ];
        let api_messages = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 3);
        assert_eq!(api_messages[1].role, "assistant");
        assert!(api_messages[1].tool_calls.is_some());

        let tool_calls = api_messages[1].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tc_123");
        assert_eq!(tool_calls[0].function.name, "search");
    }

    #[test]
    fn openai_convert_tool_response() {
        let client = create_test_client();
        let messages = vec![
            Message::user("search for rust"),
            Message::assistant_with_tools(
                "I'll search.",
                vec![ToolCall {
                    id: "tc_123".to_string(),
                    name: "search".to_string(),
                    arguments: serde_json::json!({"query": "rust"}),
                }],
            ),
            Message::tool("tc_123", "Search results..."),
        ];
        let api_messages = client.convert_messages(&messages);

        assert_eq!(api_messages.len(), 3);
        assert_eq!(api_messages[2].role, "tool");
        assert_eq!(api_messages[2].tool_call_id, Some("tc_123".to_string()));
        assert_eq!(
            api_messages[2].content,
            Some("Search results...".to_string())
        );
    }

    #[test]
    fn openai_tool_message_without_an_id_is_never_serialized() {
        let client = create_test_client();
        // `tool_call_id` is required by this format; without it the message
        // serializes as a `tool` role naming no call, which is rejected.
        let messages = vec![
            Message::user("search"),
            Message {
                role: MessageRole::Tool,
                content: "results".to_string(),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let api_messages = client.convert_messages(&messages);

        assert!(api_messages.iter().all(|m| m.role != "tool"));
    }

    #[test]
    fn openai_unparseable_tool_arguments_are_an_error() {
        let result = parse_tool_arguments("bash", "{\"command\":\"rm -");

        assert!(
            result.is_err(),
            "a truncated argument list must not become {{}}"
        );
    }

    #[test]
    fn openai_absent_tool_arguments_mean_no_arguments() {
        let parsed = parse_tool_arguments("now", "").unwrap();

        assert_eq!(parsed, serde_json::json!({}));
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
