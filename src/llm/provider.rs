//! LLM Provider actor implementation.
//!
//! The LLM Provider actor manages API calls to language models with
//! rate limiting, retry logic, and streaming support.

use crate::llm::anthropic::AnthropicClient;
use crate::llm::client::{LLMClient, LLMStreamEvent};
use crate::llm::config::{ProviderConfig, ProviderType, SamplingParams};
use crate::llm::error::LLMError;
use crate::llm::openai::OpenAIClient;
use crate::llm::streaming::StreamAccumulator;
use crate::messages::{
    LLMRequest, LLMResponse, LLMStreamEnd, LLMStreamStart, LLMStreamToken, LLMStreamToolCall,
    StopReason, SystemEvent,
};
use acton_reactive::prelude::*;
use futures::StreamExt;
use std::collections::VecDeque;
use std::sync::Arc;
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

/// Internal message reporting that the provider API refused a request with a
/// rate limit.
///
/// Carries the request back so it can be retried rather than dropped.
#[acton_message]
struct RateLimitReported {
    /// How long the API asked us to wait
    retry_after: Duration,
    /// The request that was refused
    request: LLMRequest,
    /// How many attempts this request has already used
    attempts: u32,
}

/// Internal message reporting the outcome of a dispatched request.
///
/// Sent by the request-processing helpers back to the provider so that success
/// and failure counts reflect what actually happened at the API, rather than
/// only what the handler managed to dispatch.
#[acton_message]
struct RequestOutcome {
    /// Whether the request completed successfully
    succeeded: bool,
}

/// Pending request in the queue.
#[derive(Debug, Clone)]
struct PendingRequest {
    /// The LLM request
    request: LLMRequest,
    /// Number of retry attempts already made for this request
    attempts: u32,
    /// When the request was queued, used to report queue wait time on drain
    queued_at: Instant,
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

/// Length of the rolling rate-limit window.
const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

impl RateLimiterState {
    /// Returns the instant at which the next request may be made, or `None`
    /// if a request may be made right now.
    ///
    /// This is the single source of truth for rate-limit scheduling. Callers
    /// ask it both "may I go now?" (`is_none()`) and "when may I go?", so the
    /// two answers cannot drift apart.
    ///
    /// The two ways to be blocked are handled in order of precedence:
    ///
    /// 1. An explicit rate limit reported by the API (`rate_limited_until`).
    /// 2. The local per-minute request budget being spent for the current window.
    fn next_available_at(&self, config: &ProviderConfig, now: Instant) -> Option<Instant> {
        if let Some(until) = self.rate_limited_until {
            if now < until {
                return Some(until);
            }
        }

        let Some(start) = self.window_start else {
            return None; // No window started yet
        };

        let window_ends_at = start + RATE_LIMIT_WINDOW;

        if now >= window_ends_at {
            return None; // Window expired, counters will be reset on next record
        }

        if self.requests_in_window < config.rate_limit.requests_per_minute {
            return None;
        }

        Some(window_ends_at)
    }

    /// Records a request being made.
    fn record_request(&mut self, estimated_tokens: u32) {
        let now = Instant::now();

        // Reset window if expired
        if let Some(start) = self.window_start {
            if start.elapsed() >= RATE_LIMIT_WINDOW {
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

    /// Records a rate limit reported by the provider API.
    fn record_rate_limit(&mut self, retry_after: Duration) {
        self.rate_limited_until = Some(Instant::now() + retry_after);
    }

    /// Drops an explicit rate limit whose deadline has passed.
    ///
    /// Called before draining the queue so that a stale deadline does not keep
    /// re-arming the drain timer after the provider is free again.
    fn clear_expired_rate_limit(&mut self) {
        if let Some(until) = self.rate_limited_until {
            if Instant::now() >= until {
                self.rate_limited_until = None;
            }
        }
    }
}

/// Returns whether another request fits in the queue.
///
/// `max_queue_size == 0` means unlimited, as documented on
/// [`RateLimitConfig::max_queue_size`](crate::llm::config::RateLimitConfig).
const fn queue_has_room(queue_len: usize, max_queue_size: usize) -> bool {
    max_queue_size == 0 || queue_len < max_queue_size
}

/// The LLM Provider actor state.
///
/// Manages API calls to language models with rate limiting and retries.
#[acton_actor]
pub struct LLMProvider {
    /// Configuration
    pub config: Option<ProviderConfig>,
    /// The LLM client (trait object for provider abstraction)
    client: Option<Arc<dyn LLMClient>>,
    /// Request queue
    queue: VecDeque<PendingRequest>,
    /// Rate limiter state
    rate_limiter: RateLimiterState,
    /// Active streams (for future correlation-based stream management)
    _streams: StreamAccumulator,
    /// Whether the provider is shutting down
    shutting_down: bool,
    /// Whether a queue-drain timer is already armed.
    ///
    /// Guarantees at most one outstanding timer no matter how many requests
    /// pile up while the provider is rate limited.
    drain_scheduled: bool,
    /// Metrics
    metrics: ProviderMetrics,
}

/// Metrics for the LLM Provider.
#[derive(Debug, Clone, Default)]
pub struct ProviderMetrics {
    /// Total requests dispatched to the provider API
    pub requests_total: u64,
    /// Requests that completed successfully
    pub requests_success: u64,
    /// Requests that failed
    pub requests_failed: u64,
    /// Requests that were refused by the provider with a rate limit
    pub rate_limits_hit: u64,
    /// Running total of estimated input tokens.
    ///
    /// This is the same rough estimate the rate limiter budgets against
    /// (roughly four characters per token), not a figure reported by the
    /// provider API.
    pub tokens_estimated: u64,
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
        handle
            .send(InitLLMProvider {
                config: provider_config,
            })
            .await;

        handle
    }
}

/// Configures message handlers for the LLM Provider actor.
fn configure_handlers(builder: &mut ManagedActor<Idle, LLMProvider>) {
    // Handle initialization
    builder.mutate_on::<InitLLMProvider>(|actor, envelope| {
        let config = envelope.message().config.clone();

        // Create the appropriate client based on provider type
        let client_result: Result<Arc<dyn LLMClient>, crate::llm::error::LLMError> =
            match &config.provider_type {
                ProviderType::Anthropic => {
                    AnthropicClient::new(config.clone()).map(|c| Arc::new(c) as Arc<dyn LLMClient>)
                }
                ProviderType::OpenAI { base_url } => OpenAIClient::new(base_url.clone(), &config)
                    .map(|c| Arc::new(c) as Arc<dyn LLMClient>),
            };

        match client_result {
            Ok(client) => {
                let provider_name = client.provider_name();
                actor.model.client = Some(client);
                actor.model.config = Some(config);
                actor.model.shutting_down = false;
                tracing::info!(provider = %provider_name, "LLM Provider configured");
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to create LLM client");
            }
        }

        Reply::ready()
    });

    // Handle LLM requests with fallible handler pattern
    builder
        .try_mutate_on::<LLMRequest, (), crate::llm::error::LLMError>(|actor, envelope| {
            let request = envelope.message().clone();
            let correlation_id = request.correlation_id.clone();

            if actor.model.shutting_down {
                tracing::warn!(
                    correlation_id = %correlation_id,
                    "Rejecting request - provider is shutting down"
                );
                return Reply::try_err(crate::llm::error::LLMError::shutting_down());
            }

            let Some(ref config) = actor.model.config else {
                return Reply::try_err(crate::llm::error::LLMError::invalid_config(
                    "provider",
                    "Provider not configured",
                ));
            };

            // Check rate limits
            if let Some(available_at) = actor
                .model
                .rate_limiter
                .next_available_at(config, Instant::now())
            {
                if !config.rate_limit.queue_when_limited {
                    return Reply::try_err(crate::llm::error::LLMError::rate_limited(
                        available_at.saturating_duration_since(Instant::now()),
                    ));
                }

                if !queue_has_room(actor.model.queue.len(), config.rate_limit.max_queue_size) {
                    tracing::warn!(
                        correlation_id = %correlation_id,
                        queue_size = actor.model.queue.len(),
                        "Request queue full, rejecting request"
                    );
                    return Reply::try_err(crate::llm::error::LLMError::api_error(
                        429,
                        "Request queue full",
                        Some("queue_full".to_string()),
                    ));
                }

                actor.model.queue.push_back(PendingRequest {
                    request,
                    attempts: 0,
                    queued_at: Instant::now(),
                });

                tracing::debug!(
                    correlation_id = %correlation_id,
                    queue_size = actor.model.queue.len(),
                    "Request queued due to rate limit"
                );

                // Without this the request would sit in the queue forever:
                // nothing else ever sends `ProcessQueue`.
                arm_drain_timer(actor, available_at);

                return Reply::try_ok(());
            }

            let sampling = merge_sampling(&config.sampling, request.sampling.as_ref());
            let estimated = estimate_tokens(&request);

            actor.model.rate_limiter.record_request(estimated);
            actor.model.metrics.requests_total += 1;
            actor.model.metrics.tokens_estimated += u64::from(estimated);

            dispatch_request(actor, request, sampling, 0);

            Reply::try_ok(())
        })
        .on_error::<LLMRequest, crate::llm::error::LLMError>(|actor, envelope, error| {
            let correlation_id = &envelope.message().correlation_id;

            tracing::error!(
                correlation_id = %correlation_id,
                error = %error,
                "LLM request failed"
            );

            // Update metrics
            actor.model.metrics.requests_failed += 1;

            // Broadcast rate limit event if applicable
            if let Some(retry_after) = error.retry_after() {
                let broker = actor.broker().clone();
                let provider_name = actor
                    .model
                    .client
                    .as_ref()
                    .map(|c| c.provider_name().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                return Box::pin(async move {
                    broker
                        .broadcast(SystemEvent::RateLimitHit {
                            provider: provider_name,
                            retry_after_secs: retry_after.as_secs(),
                        })
                        .await;
                });
            }

            Box::pin(async {})
        });

    // Handle queue processing
    builder.mutate_on::<ProcessQueue>(|actor, _envelope| {
        // The timer that delivered this message has fired; a later queue
        // insertion is free to arm a new one.
        actor.model.drain_scheduled = false;
        actor.model.rate_limiter.clear_expired_rate_limit();

        drain_queue(actor);

        Reply::ready()
    });

    // Handle a rate limit reported by the provider API
    builder.mutate_on::<RateLimitReported>(|actor, envelope| {
        let msg = envelope.message();
        let request = msg.request.clone();
        let attempts = msg.attempts;

        actor.model.metrics.rate_limits_hit += 1;
        actor.model.rate_limiter.record_rate_limit(msg.retry_after);

        let Some(config) = actor.model.config.clone() else {
            return Reply::ready();
        };

        if attempts >= config.retry.max_retries {
            tracing::warn!(
                correlation_id = %request.correlation_id,
                attempts,
                "Max retries exceeded; dropping request"
            );
            return Reply::ready();
        }

        if !queue_has_room(actor.model.queue.len(), config.rate_limit.max_queue_size) {
            tracing::warn!(
                correlation_id = %request.correlation_id,
                queue_size = actor.model.queue.len(),
                "Request queue full; dropping rate-limited request"
            );
            return Reply::ready();
        }

        // Ahead of newer work: this request was accepted first.
        actor.model.queue.push_front(PendingRequest {
            request,
            attempts: attempts + 1,
            queued_at: Instant::now(),
        });

        match actor
            .model
            .rate_limiter
            .next_available_at(&config, Instant::now())
        {
            Some(available_at) => arm_drain_timer(actor, available_at),
            None => drain_queue(actor),
        }

        Reply::ready()
    });

    // Handle request outcome reports from the processing helpers
    builder.mutate_on::<RequestOutcome>(|actor, envelope| {
        if envelope.message().succeeded {
            actor.model.metrics.requests_success += 1;
        } else {
            actor.model.metrics.requests_failed += 1;
        }

        Reply::ready()
    });
}

/// Drains as many queued requests as the rate limiter currently permits.
///
/// Re-arms the drain timer if requests remain, so a partially drained queue
/// still makes progress.
fn drain_queue(actor: &mut ManagedActor<Started, LLMProvider>) {
    let Some(config) = actor.model.config.clone() else {
        return;
    };

    while actor
        .model
        .rate_limiter
        .next_available_at(&config, Instant::now())
        .is_none()
    {
        let Some(pending) = actor.model.queue.pop_front() else {
            return; // Queue drained completely; no timer needed.
        };

        let sampling = merge_sampling(&config.sampling, pending.request.sampling.as_ref());
        let estimated = estimate_tokens(&pending.request);

        actor.model.rate_limiter.record_request(estimated);
        actor.model.metrics.requests_total += 1;
        actor.model.metrics.tokens_estimated += u64::from(estimated);

        tracing::debug!(
            correlation_id = %pending.request.correlation_id,
            waited_ms = pending.queued_at.elapsed().as_millis(),
            attempts = pending.attempts,
            "Dispatching queued request"
        );

        dispatch_request(actor, pending.request, sampling, pending.attempts);
    }

    // Still rate limited with work outstanding: come back when it clears.
    if !actor.model.queue.is_empty() {
        if let Some(available_at) = actor
            .model
            .rate_limiter
            .next_available_at(&config, Instant::now())
        {
            arm_drain_timer(actor, available_at);
        }
    }
}

/// Arms a one-shot timer that sends `ProcessQueue` once `at` has passed.
///
/// Does nothing if a timer is already armed.
///
/// acton-reactive 9.x has no scheduled-send facility, and a `sleep` inside a
/// read-only handler's future would be worse than a detached task: those
/// futures are drained *to completion* at the message loop's next flush point,
/// so the sleep would stall the whole loop. A one-shot task whose only effect
/// is a single `send` is the smallest correct option — it is inert if the
/// provider has already stopped, and `drain_scheduled` bounds it to one at a
/// time.
fn arm_drain_timer(actor: &mut ManagedActor<Started, LLMProvider>, at: Instant) {
    if actor.model.drain_scheduled {
        return;
    }
    actor.model.drain_scheduled = true;

    let handle = actor.handle().clone();
    tokio::spawn(async move {
        tokio::time::sleep_until(at.into()).await;
        handle.send(ProcessQueue).await;
    });
}

/// Sends a request to the provider API on a detached task.
///
/// The work runs on a task rather than in the reply future because the
/// underlying HTTP client's response stream is not `Sync`, while
/// acton-reactive's `FutureBox` requires `Send + Sync`.
fn dispatch_request(
    actor: &ManagedActor<Started, LLMProvider>,
    request: LLMRequest,
    sampling: Option<SamplingParams>,
    attempts: u32,
) {
    let Some(client) = actor.model.client.clone() else {
        tracing::error!(
            correlation_id = %request.correlation_id,
            "Cannot dispatch request: provider has no client"
        );
        return;
    };

    let streaming = actor
        .model
        .config
        .as_ref()
        .is_none_or(|config| config.streaming);

    let ctx = DispatchContext {
        provider_name: client.provider_name().to_string(),
        client,
        broker: actor.broker().clone(),
        provider: actor.handle().clone(),
        sampling,
        attempts,
    };

    tokio::spawn(async move {
        if streaming {
            process_streaming_request(&ctx, &request).await;
        } else {
            process_non_streaming_request(&ctx, &request).await;
        }
    });
}

/// Merges provider-level sampling defaults with per-request overrides.
///
/// Returns `None` when neither side asks for anything, so the client can omit
/// the sampling block entirely.
fn merge_sampling(
    base: &SamplingParams,
    overrides: Option<&SamplingParams>,
) -> Option<SamplingParams> {
    match overrides {
        Some(overrides) => Some(base.merge_with(overrides)),
        None if !base.is_empty() => Some(base.clone()),
        None => None,
    }
}

/// Reports a client error back to the provider actor.
///
/// Records the failure, and when the provider reported a rate limit, tells the
/// limiter about it and asks for the request to be retried so it is not lost.
async fn report_failure(ctx: &DispatchContext, request: &LLMRequest, error: &LLMError) {
    ctx.provider.send(RequestOutcome { succeeded: false }).await;

    let Some(retry_after) = error.retry_after() else {
        return;
    };

    ctx.broker
        .broadcast(SystemEvent::RateLimitHit {
            provider: ctx.provider_name.clone(),
            retry_after_secs: retry_after.as_secs(),
        })
        .await;

    ctx.provider
        .send(RateLimitReported {
            retry_after,
            request: request.clone(),
            attempts: ctx.attempts,
        })
        .await;
}

/// Everything a dispatched request needs to report back to the actor system.
///
/// Bundled into one struct so the processing helpers stay within clippy's
/// argument-count limit without losing any of the context they need.
struct DispatchContext {
    /// The client that performs the API call
    client: Arc<dyn LLMClient>,
    /// Broker used to publish stream events and system events
    broker: ActorHandle,
    /// Handle to the provider actor that dispatched this request
    provider: ActorHandle,
    /// Name of the provider, for logging and system events
    provider_name: String,
    /// Merged sampling parameters, if any apply
    sampling: Option<SamplingParams>,
    /// How many attempts this request has already used
    attempts: u32,
}

/// Processes a streaming request using the unified LLMClient trait.
async fn process_streaming_request(ctx: &DispatchContext, request: &LLMRequest) {
    let correlation_id = &request.correlation_id;
    let DispatchContext { client, broker, .. } = ctx;
    let sampling = ctx.sampling.as_ref();

    // Convert tools if present
    let tools = request.tools.as_deref();

    // Send stream start
    broker
        .broadcast(LLMStreamStart {
            correlation_id: correlation_id.clone(),
        })
        .await;

    // Start streaming request
    match client
        .send_streaming_request(&request.messages, tools, sampling)
        .await
    {
        Ok(mut stream) => {
            let mut accumulated_text = String::new();
            let mut tool_calls = Vec::new();
            let mut stop_reason = StopReason::EndTurn;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(event) => {
                        match event {
                            LLMStreamEvent::Start { .. } => {
                                // Stream already started, no action needed
                            }
                            LLMStreamEvent::Token { text } => {
                                accumulated_text.push_str(&text);

                                // Broadcast token
                                broker
                                    .broadcast(LLMStreamToken {
                                        correlation_id: correlation_id.clone(),
                                        token: text,
                                    })
                                    .await;
                            }
                            LLMStreamEvent::ToolCall { tool_call } => {
                                // Broadcast tool call
                                broker
                                    .broadcast(LLMStreamToolCall {
                                        correlation_id: correlation_id.clone(),
                                        tool_call: tool_call.clone(),
                                    })
                                    .await;

                                tool_calls.push(tool_call);
                            }
                            LLMStreamEvent::End {
                                stop_reason: reason,
                            } => {
                                stop_reason = reason;
                                break;
                            }
                            LLMStreamEvent::Error {
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

            ctx.provider.send(RequestOutcome { succeeded: true }).await;
        }
        Err(e) => {
            tracing::error!(
                correlation_id = %correlation_id,
                error = %e,
                "Failed to start streaming request"
            );

            report_failure(ctx, request, &e).await;

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

/// Processes a non-streaming request using the unified LLMClient trait.
async fn process_non_streaming_request(ctx: &DispatchContext, request: &LLMRequest) {
    let correlation_id = &request.correlation_id;
    let DispatchContext { client, broker, .. } = ctx;

    // Convert tools if present
    let tools = request.tools.as_deref();

    match client
        .send_request(&request.messages, tools, ctx.sampling.as_ref())
        .await
    {
        Ok(response) => {
            broker
                .broadcast(LLMResponse {
                    correlation_id: correlation_id.clone(),
                    content: response.content,
                    tool_calls: if response.tool_calls.is_empty() {
                        None
                    } else {
                        Some(response.tool_calls)
                    },
                    stop_reason: response.stop_reason,
                })
                .await;

            ctx.provider.send(RequestOutcome { succeeded: true }).await;
        }
        Err(e) => {
            tracing::error!(
                correlation_id = %correlation_id,
                error = %e,
                "Non-streaming request failed"
            );

            report_failure(ctx, request, &e).await;
        }
    }
}

/// Estimates the number of tokens in a request.
fn estimate_tokens(request: &LLMRequest) -> u32 {
    // Rough estimate: 4 characters per token
    let char_count: usize = request.messages.iter().map(|m| m.content.len()).sum();

    u32::try_from(char_count / 4).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::config::RateLimitConfig;

    /// A config whose per-minute budget is small enough to exhaust in a test.
    fn config_with_request_budget(requests_per_minute: u32) -> ProviderConfig {
        ProviderConfig::new("test-key")
            .with_rate_limit(RateLimitConfig::new(requests_per_minute, 1_000_000))
    }

    #[test]
    fn fresh_limiter_is_available_immediately() {
        let state = RateLimiterState::default();
        let config = ProviderConfig::new("test-key");

        assert_eq!(state.next_available_at(&config, Instant::now()), None);
    }

    #[test]
    fn rate_limiter_tracks_requests() {
        let mut state = RateLimiterState::default();

        state.record_request(100);

        assert_eq!(state.requests_in_window, 1);
        assert_eq!(state.tokens_in_window, 100);
    }

    #[test]
    fn reported_rate_limit_defers_until_its_deadline() {
        let mut state = RateLimiterState::default();
        let config = ProviderConfig::new("test-key");
        let now = Instant::now();

        state.record_rate_limit(Duration::from_secs(60));

        let available_at = state
            .next_available_at(&config, now)
            .expect("a reported rate limit must block the provider");
        assert!(available_at > now);
    }

    #[test]
    fn expired_rate_limit_no_longer_blocks() {
        let config = ProviderConfig::new("test-key");

        // A deadline that has already passed must not defer anything.
        let state = RateLimiterState {
            rate_limited_until: Some(Instant::now() - Duration::from_secs(1)),
            ..RateLimiterState::default()
        };

        assert_eq!(state.next_available_at(&config, Instant::now()), None);
    }

    #[test]
    fn expired_rate_limit_is_pruned() {
        let mut state = RateLimiterState {
            rate_limited_until: Some(Instant::now() - Duration::from_secs(1)),
            ..RateLimiterState::default()
        };

        state.clear_expired_rate_limit();

        assert_eq!(state.rate_limited_until, None);
    }

    #[test]
    fn live_rate_limit_survives_pruning() {
        let mut state = RateLimiterState::default();
        state.record_rate_limit(Duration::from_secs(60));

        state.clear_expired_rate_limit();

        assert!(state.rate_limited_until.is_some());
    }

    #[test]
    fn spent_request_budget_defers_to_end_of_window() {
        let mut state = RateLimiterState::default();
        let config = config_with_request_budget(2);
        let now = Instant::now();

        state.record_request(10);
        state.record_request(10);

        let window_start = state.window_start.expect("recording opens a window");
        assert_eq!(
            state.next_available_at(&config, now),
            Some(window_start + RATE_LIMIT_WINDOW)
        );
    }

    #[test]
    fn unspent_request_budget_stays_available() {
        let mut state = RateLimiterState::default();
        let config = config_with_request_budget(2);

        state.record_request(10);

        assert_eq!(state.next_available_at(&config, Instant::now()), None);
    }

    #[test]
    fn budget_frees_up_once_the_window_has_elapsed() {
        let mut state = RateLimiterState::default();
        let config = config_with_request_budget(1);

        state.record_request(10);
        let window_start = state.window_start.expect("recording opens a window");

        assert!(state.next_available_at(&config, window_start).is_some());
        assert_eq!(
            state.next_available_at(&config, window_start + RATE_LIMIT_WINDOW),
            None
        );
    }

    #[test]
    fn zero_max_queue_size_means_unlimited() {
        assert!(queue_has_room(0, 0));
        assert!(queue_has_room(10_000, 0));
    }

    #[test]
    fn positive_max_queue_size_is_enforced() {
        assert!(queue_has_room(0, 2));
        assert!(queue_has_room(1, 2));
        assert!(!queue_has_room(2, 2));
        assert!(!queue_has_room(3, 2));
    }

    #[test]
    fn merge_sampling_returns_none_when_nothing_is_set() {
        assert_eq!(merge_sampling(&SamplingParams::default(), None), None);
    }

    #[test]
    fn merge_sampling_falls_back_to_provider_defaults() {
        let base = SamplingParams::new().with_temperature(0.2);

        let merged = merge_sampling(&base, None).expect("provider defaults must be applied");

        assert_eq!(merged.temperature, Some(0.2));
    }

    #[test]
    fn merge_sampling_lets_the_request_win() {
        let base = SamplingParams::new().with_temperature(0.2).with_top_k(10);
        let overrides = SamplingParams::new().with_temperature(0.9);

        let merged = merge_sampling(&base, Some(&overrides)).expect("overrides must be applied");

        assert_eq!(merged.temperature, Some(0.9));
        assert_eq!(merged.top_k, Some(10), "unset overrides keep the default");
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
            sampling: None,
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
        assert_eq!(metrics.rate_limits_hit, 0);
        assert_eq!(metrics.tokens_estimated, 0);
    }

    #[test]
    fn streaming_is_on_by_default() {
        assert!(ProviderConfig::new("test-key").streaming);
        assert!(ProviderConfig::ollama("qwen2.5:7b").streaming);
    }

    #[test]
    fn disabling_queueing_leaves_streaming_alone() {
        // Streaming used to be derived from `queue_when_limited`, so opting out
        // of queueing silently turned streaming off as well.
        let config =
            ProviderConfig::new("test-key").with_rate_limit(RateLimitConfig::default().without_queueing());

        assert!(config.streaming);
    }
}
