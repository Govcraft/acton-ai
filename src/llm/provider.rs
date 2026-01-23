//! LLM Provider actor implementation.
//!
//! The LLM Provider actor manages API calls to language models with
//! rate limiting, retry logic, and streaming support.

use crate::llm::anthropic::{
    extract_text_content, extract_tool_calls, parse_stop_reason, AnthropicClient, StreamEvent,
};
use crate::llm::config::ProviderConfig;
use crate::llm::streaming::StreamAccumulator;
use crate::messages::{
    LLMRequest, LLMResponse, LLMStreamEnd, LLMStreamStart, LLMStreamToken, LLMStreamToolCall,
    StopReason, SystemEvent,
};
use acton_reactive::prelude::*;
use futures::StreamExt;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Message to initialize the LLM Provider with configuration.
#[acton_message]
pub struct InitLLMProvider {
    /// Provider configuration
    pub config: ProviderConfig,
}

/// Internal message for processing queued requests.
#[acton_message]
struct ProcessQueue;

/// Internal message for retry after rate limit.
#[acton_message]
struct RetryAfterRateLimit {
    /// The request to retry
    request: LLMRequest,
    /// Current retry attempt
    attempt: u32,
}

/// Pending request in the queue.
#[derive(Debug, Clone)]
struct PendingRequest {
    /// The LLM request
    request: LLMRequest,
    /// Number of retry attempts (used for retry logic)
    _attempts: u32,
    /// When the request was queued (used for timeout tracking)
    _queued_at: Instant,
}

/// Rate limiter state.
#[derive(Debug, Clone, Default)]
struct RateLimiterState {
    /// Requests made in the current window
    requests_in_window: u32,
    /// Tokens used in the current window
    tokens_in_window: u32,
    /// Start of the current window
    window_start: Option<Instant>,
    /// If rate limited, when we can retry
    rate_limited_until: Option<Instant>,
}

impl RateLimiterState {
    /// Checks if a request can be made now.
    fn can_make_request(&self, config: &ProviderConfig) -> bool {
        // Check if we're in a rate limit window
        if let Some(until) = self.rate_limited_until {
            if Instant::now() < until {
                return false;
            }
        }

        // Check if window has expired (reset counters)
        if let Some(start) = self.window_start {
            if start.elapsed() >= Duration::from_secs(60) {
                return true; // Window expired, counters will be reset
            }
        } else {
            return true; // No window started yet
        }

        // Check rate limits
        self.requests_in_window < config.rate_limit.requests_per_minute
    }

    /// Records a request being made.
    fn record_request(&mut self, estimated_tokens: u32) {
        let now = Instant::now();

        // Reset window if expired
        if let Some(start) = self.window_start {
            if start.elapsed() >= Duration::from_secs(60) {
                self.requests_in_window = 0;
                self.tokens_in_window = 0;
                self.window_start = Some(now);
            }
        } else {
            self.window_start = Some(now);
        }

        self.requests_in_window += 1;
        self.tokens_in_window += estimated_tokens;
    }

    /// Records a rate limit hit.
    /// Currently used only in tests but will be used when handling rate limit responses.
    #[cfg(test)]
    fn record_rate_limit(&mut self, retry_after: Duration) {
        self.rate_limited_until = Some(Instant::now() + retry_after);
    }

    /// Clears the rate limit.
    /// Currently used only in tests but will be used when rate limit window expires.
    #[cfg(test)]
    fn clear_rate_limit(&mut self) {
        self.rate_limited_until = None;
    }
}

/// The LLM Provider actor state.
///
/// Manages API calls to language models with rate limiting and retries.
#[acton_actor]
pub struct LLMProvider {
    /// Configuration
    pub config: Option<ProviderConfig>,
    /// HTTP client
    client: Option<AnthropicClient>,
    /// Request queue
    queue: VecDeque<PendingRequest>,
    /// Rate limiter state
    rate_limiter: RateLimiterState,
    /// Active streams (for future correlation-based stream management)
    _streams: StreamAccumulator,
    /// Whether the provider is shutting down
    shutting_down: bool,
    /// Metrics
    metrics: ProviderMetrics,
}

/// Metrics for the LLM Provider.
#[derive(Debug, Clone, Default)]
pub struct ProviderMetrics {
    /// Total requests made
    pub requests_total: u64,
    /// Successful requests
    pub requests_success: u64,
    /// Failed requests
    pub requests_failed: u64,
    /// Requests that hit rate limits (tracked for monitoring, to be used with rate limit handling)
    pub _rate_limits_hit: u64,
    /// Total tokens used (input + output, to be populated from API response usage)
    pub _tokens_used: u64,
}

impl LLMProvider {
    /// Spawns the LLM Provider actor with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The ActorRuntime
    /// * `config` - Provider configuration
    ///
    /// # Returns
    ///
    /// The ActorHandle for the started provider.
    pub async fn spawn(runtime: &mut ActorRuntime, config: ProviderConfig) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<LLMProvider>("llm_provider".to_string());

        // Store config for initialization
        let provider_config = config.clone();

        // Set up lifecycle hooks
        builder
            .before_start(|_actor| {
                tracing::debug!("LLM Provider initializing");
                Reply::ready()
            })
            .after_start(|actor| {
                if let Some(config) = &actor.model.config {
                    tracing::info!(
                        model = %config.model,
                        "LLM Provider ready to accept requests"
                    );
                }
                Reply::ready()
            })
            .before_stop(|actor| {
                tracing::info!(
                    requests_total = actor.model.metrics.requests_total,
                    requests_success = actor.model.metrics.requests_success,
                    requests_failed = actor.model.metrics.requests_failed,
                    "LLM Provider shutting down"
                );
                Reply::ready()
            });

        // Configure message handlers
        configure_handlers(&mut builder);

        let handle = builder.start().await;

        // Initialize with config
        handle.send(InitLLMProvider { config: provider_config }).await;

        handle
    }
}

/// Configures message handlers for the LLM Provider actor.
fn configure_handlers(builder: &mut ManagedActor<Idle, LLMProvider>) {
    // Handle initialization
    builder.mutate_on::<InitLLMProvider>(|actor, envelope| {
        let config = envelope.message().config.clone();

        // Create the HTTP client
        match AnthropicClient::new(config.clone()) {
            Ok(client) => {
                actor.model.client = Some(client);
                actor.model.config = Some(config);
                actor.model.shutting_down = false;
                tracing::info!("LLM Provider configured");
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to create Anthropic client");
            }
        }

        Reply::ready()
    });

    // Handle LLM requests
    builder.mutate_on::<LLMRequest>(|actor, envelope| {
        let request = envelope.message().clone();
        let correlation_id = request.correlation_id.clone();

        if actor.model.shutting_down {
            tracing::warn!(
                correlation_id = %correlation_id,
                "Rejecting request - provider is shutting down"
            );
            return Reply::ready();
        }

        let Some(ref config) = actor.model.config else {
            tracing::error!(
                correlation_id = %correlation_id,
                "Provider not configured"
            );
            return Reply::ready();
        };

        // Check rate limits
        if !actor.model.rate_limiter.can_make_request(config) {
            if config.rate_limit.queue_when_limited {
                // Check queue size
                if actor.model.queue.len() >= config.rate_limit.max_queue_size {
                    tracing::warn!(
                        correlation_id = %correlation_id,
                        queue_size = actor.model.queue.len(),
                        "Request queue full, rejecting request"
                    );
                    return Reply::ready();
                }

                // Queue the request
                actor.model.queue.push_back(PendingRequest {
                    request,
                    _attempts: 0,
                    _queued_at: Instant::now(),
                });

                tracing::debug!(
                    correlation_id = %correlation_id,
                    queue_size = actor.model.queue.len(),
                    "Request queued due to rate limit"
                );

                return Reply::ready();
            } else {
                tracing::warn!(
                    correlation_id = %correlation_id,
                    "Rate limited and queueing disabled"
                );
                return Reply::ready();
            }
        }

        // Process the request
        let client = actor.model.client.clone();
        let broker = actor.broker().clone();
        let streaming = actor.model.config.as_ref()
            .map(|c| c.rate_limit.queue_when_limited) // Streaming is enabled by default
            .unwrap_or(true);

        // Record the request
        actor.model.rate_limiter.record_request(estimate_tokens(&request));
        actor.model.metrics.requests_total += 1;

        Reply::pending(async move {
            if let Some(client) = client {
                if streaming {
                    process_streaming_request(&client, &request, &broker).await;
                } else {
                    process_non_streaming_request(&client, &request, &broker).await;
                }
            }
        })
    });

    // Handle queue processing
    builder.mutate_on::<ProcessQueue>(|actor, _envelope| {
        let Some(ref config) = actor.model.config else {
            return Reply::ready();
        };

        // Process queued requests if rate limit allows
        while actor.model.rate_limiter.can_make_request(config) {
            if let Some(pending) = actor.model.queue.pop_front() {
                let client = actor.model.client.clone();
                let broker = actor.broker().clone();
                let request = pending.request;

                actor.model.rate_limiter.record_request(estimate_tokens(&request));
                actor.model.metrics.requests_total += 1;

                // Spawn the request processing
                tokio::spawn(async move {
                    if let Some(client) = client {
                        process_streaming_request(&client, &request, &broker).await;
                    }
                });
            } else {
                break;
            }
        }

        Reply::ready()
    });

    // Handle retry after rate limit
    builder.mutate_on::<RetryAfterRateLimit>(|actor, envelope| {
        let msg = envelope.message();
        let request = msg.request.clone();
        let attempt = msg.attempt;

        let Some(ref config) = actor.model.config else {
            return Reply::ready();
        };

        // Check if we've exceeded max retries
        if attempt >= config.retry.max_retries {
            tracing::warn!(
                correlation_id = %request.correlation_id,
                attempts = attempt,
                "Max retries exceeded"
            );
            actor.model.metrics.requests_failed += 1;
            return Reply::ready();
        }

        // Re-queue the request
        actor.model.queue.push_front(PendingRequest {
            request,
            _attempts: attempt,
            _queued_at: Instant::now(),
        });

        Reply::ready()
    });
}

/// Processes a streaming request.
async fn process_streaming_request(
    client: &AnthropicClient,
    request: &LLMRequest,
    broker: &ActorHandle,
) {
    let correlation_id = &request.correlation_id;

    // Convert tools if present
    let tools = request.tools.as_deref();

    // Send stream start
    broker
        .broadcast(LLMStreamStart {
            correlation_id: correlation_id.clone(),
        })
        .await;

    // Start streaming request
    match client.send_messages_streaming(&request.messages, tools).await {
        Ok(mut stream) => {
            let mut accumulated_text = String::new();
            let mut tool_calls = Vec::new();
            let mut stop_reason = StopReason::EndTurn;
            let mut current_tool_id: Option<String> = None;
            let mut current_tool_name: Option<String> = None;
            let mut current_tool_input = String::new();

            while let Some(result) = stream.next().await {
                match result {
                    Ok(event) => {
                        match event {
                            StreamEvent::ContentBlockStart {
                                block_type,
                                tool_id,
                                tool_name,
                                ..
                            } => {
                                if block_type == "tool_use" {
                                    current_tool_id = tool_id;
                                    current_tool_name = tool_name;
                                    current_tool_input.clear();
                                }
                            }
                            StreamEvent::ContentBlockDelta {
                                text,
                                partial_json,
                                ..
                            } => {
                                if let Some(text) = text {
                                    accumulated_text.push_str(&text);

                                    // Broadcast token
                                    broker
                                        .broadcast(LLMStreamToken {
                                            correlation_id: correlation_id.clone(),
                                            token: text,
                                        })
                                        .await;
                                }

                                if let Some(json) = partial_json {
                                    current_tool_input.push_str(&json);
                                }
                            }
                            StreamEvent::ContentBlockStop { .. } => {
                                // If we were building a tool call, finalize it
                                if let (Some(id), Some(name)) =
                                    (current_tool_id.take(), current_tool_name.take())
                                {
                                    let input: serde_json::Value =
                                        serde_json::from_str(&current_tool_input)
                                            .unwrap_or(serde_json::json!({}));

                                    let tool_call = crate::messages::ToolCall {
                                        id: id.clone(),
                                        name: name.clone(),
                                        arguments: input,
                                    };

                                    // Broadcast tool call
                                    broker
                                        .broadcast(LLMStreamToolCall {
                                            correlation_id: correlation_id.clone(),
                                            tool_call: tool_call.clone(),
                                        })
                                        .await;

                                    tool_calls.push(tool_call);
                                    current_tool_input.clear();
                                }
                            }
                            StreamEvent::MessageDelta {
                                stop_reason: Some(reason),
                            } => {
                                stop_reason = parse_stop_reason(&reason);
                            }
                            StreamEvent::MessageStop => {
                                break;
                            }
                            StreamEvent::Error {
                                error_type,
                                message,
                            } => {
                                tracing::error!(
                                    correlation_id = %correlation_id,
                                    error_type = %error_type,
                                    message = %message,
                                    "Stream error"
                                );
                                break;
                            }
                            StreamEvent::Ping | StreamEvent::MessageStart { .. } | StreamEvent::MessageDelta { stop_reason: None } => {
                                // Ignore ping and message start events
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            correlation_id = %correlation_id,
                            error = %e,
                            "Stream error"
                        );
                        break;
                    }
                }
            }

            // Send stream end
            broker
                .broadcast(LLMStreamEnd {
                    correlation_id: correlation_id.clone(),
                    stop_reason,
                })
                .await;

            // Also broadcast the complete response for non-streaming consumers
            broker
                .broadcast(LLMResponse {
                    correlation_id: correlation_id.clone(),
                    content: accumulated_text,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    stop_reason,
                })
                .await;
        }
        Err(e) => {
            tracing::error!(
                correlation_id = %correlation_id,
                error = %e,
                "Failed to start streaming request"
            );

            // Broadcast rate limit event if applicable
            if let Some(retry_after) = e.retry_after() {
                broker
                    .broadcast(SystemEvent::RateLimitHit {
                        provider: "anthropic".to_string(),
                        retry_after_secs: retry_after.as_secs(),
                    })
                    .await;
            }

            // Send stream end with error
            broker
                .broadcast(LLMStreamEnd {
                    correlation_id: correlation_id.clone(),
                    stop_reason: StopReason::EndTurn,
                })
                .await;
        }
    }
}

/// Processes a non-streaming request.
async fn process_non_streaming_request(
    client: &AnthropicClient,
    request: &LLMRequest,
    broker: &ActorHandle,
) {
    let correlation_id = &request.correlation_id;

    // Convert tools if present
    let tools = request.tools.as_deref();

    match client.send_messages(&request.messages, tools).await {
        Ok(response) => {
            let content = extract_text_content(&response);
            let tool_calls = extract_tool_calls(&response);
            let stop_reason = response
                .stop_reason
                .as_ref()
                .map(|s| parse_stop_reason(s))
                .unwrap_or(StopReason::EndTurn);

            broker
                .broadcast(LLMResponse {
                    correlation_id: correlation_id.clone(),
                    content,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    stop_reason,
                })
                .await;
        }
        Err(e) => {
            tracing::error!(
                correlation_id = %correlation_id,
                error = %e,
                "Non-streaming request failed"
            );

            // Broadcast rate limit event if applicable
            if let Some(retry_after) = e.retry_after() {
                broker
                    .broadcast(SystemEvent::RateLimitHit {
                        provider: "anthropic".to_string(),
                        retry_after_secs: retry_after.as_secs(),
                    })
                    .await;
            }
        }
    }
}

/// Estimates the number of tokens in a request.
fn estimate_tokens(request: &LLMRequest) -> u32 {
    // Rough estimate: 4 characters per token
    let char_count: usize = request
        .messages
        .iter()
        .map(|m| m.content.len())
        .sum();

    (char_count / 4) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_allows_initial_request() {
        let state = RateLimiterState::default();
        let config = ProviderConfig::new("test-key");

        assert!(state.can_make_request(&config));
    }

    #[test]
    fn rate_limiter_tracks_requests() {
        let mut state = RateLimiterState::default();
        let config = ProviderConfig::new("test-key");

        state.record_request(100);

        assert_eq!(state.requests_in_window, 1);
        assert_eq!(state.tokens_in_window, 100);
    }

    #[test]
    fn rate_limiter_blocks_when_limited() {
        let mut state = RateLimiterState::default();
        let config = ProviderConfig::new("test-key");

        state.record_rate_limit(Duration::from_secs(60));

        assert!(!state.can_make_request(&config));
    }

    #[test]
    fn rate_limiter_clears_limit() {
        let mut state = RateLimiterState::default();
        let config = ProviderConfig::new("test-key");

        state.record_rate_limit(Duration::from_secs(60));
        assert!(!state.can_make_request(&config));

        state.clear_rate_limit();
        assert!(state.can_make_request(&config));
    }

    #[test]
    fn estimate_tokens_calculation() {
        use crate::messages::Message;
        use crate::types::{AgentId, CorrelationId};

        let request = LLMRequest {
            correlation_id: CorrelationId::new(),
            agent_id: AgentId::new(),
            messages: vec![Message::user("Hello world")], // 11 chars
            tools: None,
        };

        let tokens = estimate_tokens(&request);
        assert_eq!(tokens, 2); // 11 / 4 = 2
    }

    #[test]
    fn provider_metrics_default() {
        let metrics = ProviderMetrics::default();

        assert_eq!(metrics.requests_total, 0);
        assert_eq!(metrics.requests_success, 0);
        assert_eq!(metrics.requests_failed, 0);
        assert_eq!(metrics._rate_limits_hit, 0);
        assert_eq!(metrics._tokens_used, 0);
    }
}
