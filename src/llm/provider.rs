//! LLM Provider actor implementation.
//!
//! The LLM Provider actor manages API calls to language models with
//! rate limiting, retry logic, and streaming support.

use crate::llm::anthropic::AnthropicClient;
use crate::llm::client::{LLMClient, LLMStreamEvent};
use crate::llm::config::{ProviderConfig, ProviderType, SamplingParams};
use crate::llm::error::LLMError;
use crate::llm::failover::{
    next_state, phase, BreakerPhase, BreakerState, CheckHealth, FailoverEvent, ProviderHealth,
};
use crate::llm::openai::OpenAIClient;
use crate::llm::streaming::StreamAccumulator;
use crate::messages::{
    LLMRequest, LLMResponse, LLMStreamEnd, LLMStreamStart, LLMStreamToken, LLMStreamToolCall,
    StopReason, SystemEvent, Usage, UsageReport,
};
use acton_reactive::prelude::*;
use futures::StreamExt;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Message to initialize the LLM Provider with configuration.
#[acton_message]
pub struct InitLLMProvider {
    /// The name this provider is registered under in the runtime.
    ///
    /// This is the **configured** name (the `[providers.NAME]` key), not the
    /// vendor — usage reports are keyed by it so a runtime with two Anthropic
    /// providers can tell them apart.
    pub name: String,
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
    /// The configured name this provider answers to, used to key usage
    /// reports. Empty until [`InitLLMProvider`] lands.
    pub name: String,
    /// Configuration
    pub config: Option<ProviderConfig>,
    /// The LLM client (trait object for provider abstraction)
    client: Option<Arc<dyn LLMClient>>,
    /// A second client bound to `config.fallback_model`, built at init.
    ///
    /// The model is fixed when a client is constructed, so degrading to a
    /// cheaper model means having a second client rather than threading a
    /// per-request model override through the whole `LLMClient` trait and
    /// both of its implementations. `None` unless `fallback_model` is set.
    fallback_client: Option<Arc<dyn LLMClient>>,
    /// This provider's circuit breaker.
    ///
    /// State per actor, never a shared registry behind a lock: the actor that
    /// observes the failures is the one that owns the consequence.
    breaker: BreakerState,
    /// Request queue
    queue: VecDeque<PendingRequest>,
    /// Rate limiter state
    rate_limiter: RateLimiterState,
    /// Active streams (for future correlation-based stream management)
    _streams: StreamAccumulator,
    /// Whether the provider is shutting down
    shutting_down: bool,
    /// The queue-drain timer currently armed, if any.
    ///
    /// Guarantees at most one outstanding timer no matter how many requests
    /// pile up while the provider is rate limited. The schedule ends with the
    /// actor, so it never outlives the provider.
    drain_timer: Option<ScheduledSend>,
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
    /// * `name` - The name this provider is registered under. Usage reports
    ///   are keyed by it, so pass the configured provider name (the
    ///   `[providers.NAME]` key), not the vendor.
    /// * `config` - Provider configuration
    ///
    /// # Returns
    ///
    /// The ActorHandle for the started provider.
    pub async fn spawn(
        runtime: &mut ActorRuntime,
        name: impl Into<String>,
        config: ProviderConfig,
    ) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<LLMProvider>("llm_provider".to_string());

        // Store config for initialization
        let provider_name = name.into();
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
                name: provider_name,
                config: provider_config,
            })
            .await;

        handle
    }
}

/// Builds the API client a provider config asks for.
///
/// Split out because a provider with a `fallback_model` builds two of them —
/// the same config with a different model — and they must be constructed the
/// same way or the degraded path would silently differ from the primary one.
fn build_client(config: &ProviderConfig) -> Result<Arc<dyn LLMClient>, LLMError> {
    match &config.provider_type {
        ProviderType::Anthropic => {
            AnthropicClient::new(config.clone()).map(|c| Arc::new(c) as Arc<dyn LLMClient>)
        }
        ProviderType::OpenAI { base_url } => {
            OpenAIClient::new(base_url.clone(), config).map(|c| Arc::new(c) as Arc<dyn LLMClient>)
        }
    }
}

/// Configures message handlers for the LLM Provider actor.
fn configure_handlers(builder: &mut ManagedActor<Idle, LLMProvider>) {
    // Handle initialization
    builder.mutate_on::<InitLLMProvider>(|actor, envelope| {
        let config = envelope.message().config.clone();
        let name = envelope.message().name.clone();

        // A second client for the degraded model, built here so the rate-limit
        // path can switch to it without doing any construction work while a
        // request is waiting.
        let fallback_client = config.fallback_model.as_ref().and_then(|model| {
            let degraded = config.clone().with_model(model);
            match build_client(&degraded) {
                Ok(client) => Some(client),
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        fallback_model = %model,
                        "Failed to create the fallback-model client; rate limits will queue \
                         instead of degrading"
                    );
                    None
                }
            }
        });

        match build_client(&config) {
            Ok(client) => {
                let provider_name = client.provider_name();
                actor.model.client = Some(client);
                actor.model.fallback_client = fallback_client;
                actor.model.name = name;
                actor.model.config = Some(config);
                actor.model.shutting_down = false;
                actor.model.breaker = BreakerState::default();
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

            let Some(config) = actor.model.config.clone() else {
                return Reply::try_err(crate::llm::error::LLMError::invalid_config(
                    "provider",
                    "Provider not configured",
                ));
            };

            let now = Instant::now();

            // A tripped circuit refuses without touching the wire. This is
            // the whole point of the breaker: five failures in a row means
            // the next request is very unlikely to be the one that works, and
            // sending it anyway costs latency the caller pays for.
            //
            // `consecutive_failures` is reported as the threshold because
            // that is what the circuit tripped at — the open state does not
            // carry a running count, since nothing is counting while it is
            // open.
            if let ProviderHealth::Open { remaining } =
                ProviderHealth::from_state(actor.model.breaker, now)
            {
                tracing::warn!(
                    correlation_id = %correlation_id,
                    provider = %actor.model.name,
                    remaining_secs = remaining.as_secs_f64(),
                    "Refusing request - circuit breaker is open"
                );
                return Reply::try_err(crate::llm::error::LLMError::circuit_open(
                    remaining,
                    config.circuit_breaker.failure_threshold,
                ));
            }

            // Check rate limits
            if let Some(available_at) = actor.model.rate_limiter.next_available_at(&config, now) {
                // Degrading to a cheaper model is tried before queueing:
                // waiting out a rate limit costs the caller the whole window,
                // and a configured fallback model is an explicit statement
                // that a lesser answer now beats a better one later.
                if let Some(retry_after) = degradation_window(
                    actor.model.rate_limiter.rate_limited_until,
                    actor.model.fallback_client.as_ref(),
                    now,
                ) {
                    let sampling = merge_sampling(&config.sampling, request.sampling.as_ref());
                    let estimated = estimate_tokens(&request);

                    actor.model.rate_limiter.record_request(estimated);
                    actor.model.metrics.requests_total += 1;
                    actor.model.metrics.tokens_estimated += u64::from(estimated);

                    dispatch_request(actor, request, sampling, 0, Some(retry_after));

                    return Reply::try_ok(());
                }

                if !config.rate_limit.queue_when_limited {
                    return Reply::try_err(crate::llm::error::LLMError::rate_limited(
                        available_at.saturating_duration_since(now),
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

            dispatch_request(actor, request, sampling, 0, None);

            Reply::try_ok(())
        })
        .on_error::<LLMRequest, crate::llm::error::LLMError>(|actor, envelope, error| {
            let correlation_id = envelope.message().correlation_id.clone();

            tracing::error!(
                correlation_id = %correlation_id,
                error = %error,
                "LLM request failed"
            );

            // Update metrics
            actor.model.metrics.requests_failed += 1;

            let broker = actor.broker().clone();
            let model = actor
                .model
                .config
                .as_ref()
                .map(|config| config.model.clone())
                .unwrap_or_default();
            let rate_limit = error.retry_after().map(|retry_after| {
                (
                    actor
                        .model
                        .client
                        .as_ref()
                        .map(|c| c.provider_name().to_string())
                        .unwrap_or_else(|| "unknown".to_string()),
                    retry_after,
                )
            });

            Box::pin(async move {
                if let Some((provider_name, retry_after)) = rate_limit {
                    broker
                        .broadcast(SystemEvent::RateLimitHit {
                            provider: provider_name,
                            retry_after_secs: retry_after.as_secs(),
                        })
                        .await;
                }

                // Nothing was dispatched, so nothing else will ever close
                // this round — and its caller is blocked on the terminal
                // event. Refusing a request is only fail-fast if the caller
                // finds out fast.
                broker
                    .broadcast(LLMStreamEnd {
                        correlation_id,
                        stop_reason: StopReason::Error,
                        usage: Usage::default(),
                        model,
                    })
                    .await;
            })
        });

    // Handle queue processing
    builder.mutate_on::<ProcessQueue>(|actor, _envelope| {
        // The timer that delivered this message has fired; a later queue
        // insertion is free to arm a new one.
        actor.model.drain_timer = None;
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

        let now = Instant::now();
        let drop_reason = match actor.model.config.clone() {
            None => Some("provider not configured"),
            Some(config) if attempts >= config.retry.max_retries => Some("max retries exceeded"),
            Some(config) => {
                // Degrading beats queueing for the same reason as on the
                // inbound path: the caller asked for an answer, and a
                // configured fallback model says a cheaper one now is worth
                // more than the good one after the window clears.
                if let Some(retry_after) = degradation_window(
                    actor.model.rate_limiter.rate_limited_until,
                    actor.model.fallback_client.as_ref(),
                    now,
                ) {
                    let sampling = merge_sampling(&config.sampling, request.sampling.as_ref());

                    actor.model.metrics.requests_total += 1;
                    dispatch_request(
                        actor,
                        request.clone(),
                        sampling,
                        attempts + 1,
                        Some(retry_after),
                    );

                    None
                } else if !queue_has_room(actor.model.queue.len(), config.rate_limit.max_queue_size)
                {
                    Some("request queue full")
                } else {
                    // Ahead of newer work: this request was accepted first.
                    actor.model.queue.push_front(PendingRequest {
                        request: request.clone(),
                        attempts: attempts + 1,
                        queued_at: now,
                    });

                    match actor.model.rate_limiter.next_available_at(&config, now) {
                        Some(available_at) => arm_drain_timer(actor, available_at),
                        None => drain_queue(actor),
                    }

                    None
                }
            }
        };

        let Some(reason) = drop_reason else {
            return Reply::ready();
        };

        tracing::warn!(
            correlation_id = %request.correlation_id,
            attempts,
            queue_size = actor.model.queue.len(),
            reason,
            "Dropping rate-limited request"
        );

        // The dropped request's caller is still waiting on its stream. The
        // dispatch path suppressed the terminal event because a retry was
        // expected, so this handler owns closing the round; without it the
        // caller waits forever.
        let broker = actor.broker().clone();
        let correlation_id = request.correlation_id;
        let model = actor
            .model
            .config
            .as_ref()
            .map(|config| config.model.clone())
            .unwrap_or_default();
        Reply::pending(async move {
            broker
                .broadcast(LLMStreamEnd {
                    correlation_id,
                    stop_reason: StopReason::Error,
                    // A request that never reached the API spent nothing.
                    usage: Usage::default(),
                    model,
                })
                .await;
        })
    });

    // Handle request outcome reports from the processing helpers
    builder.mutate_on::<RequestOutcome>(|actor, envelope| {
        let succeeded = envelope.message().succeeded;
        if succeeded {
            actor.model.metrics.requests_success += 1;
        } else {
            actor.model.metrics.requests_failed += 1;
        }

        let config = actor
            .model
            .config
            .as_ref()
            .map(|config| config.circuit_breaker)
            .unwrap_or_default();

        // One `now` for the transition and both phase readings. Three separate
        // clock reads would let a short cooldown expire *between* them, and
        // the edge would be computed against a state nobody was ever in.
        let now = Instant::now();
        let before = actor.model.breaker;
        let after = next_state(before, &config, succeeded, now);
        actor.model.breaker = after;

        // Only the edges are worth telling anyone about: a circuit that stays
        // closed through a successful request is the ordinary case, and an
        // event per request would drown the two that matter.
        let event = match (phase(before, now), phase(after, now)) {
            (BreakerPhase::Closed, BreakerPhase::Open) => Some(FailoverEvent::CircuitOpened {
                provider: actor.model.name.clone(),
                consecutive_failures: config.failure_threshold,
                cooldown_secs: config.cooldown.as_secs(),
            }),
            (BreakerPhase::HalfOpen | BreakerPhase::Open, BreakerPhase::Closed) => {
                Some(FailoverEvent::CircuitClosed {
                    provider: actor.model.name.clone(),
                })
            }
            _ => None,
        };

        let Some(event) = event else {
            return Reply::ready();
        };

        tracing::info!(event = %event, "Circuit breaker state changed");

        let broker = actor.broker().clone();
        Reply::pending(async move {
            broker.broadcast(event).await;
        })
    });

    // Answer health queries from the prompt loop's failover walk.
    //
    // `act_on`, not `mutate_on`: reading the breaker changes nothing, and
    // several candidates can be asked at once. Mailboxes are FIFO, so an
    // answer here also proves every outcome reported before the ask has
    // already been folded in.
    builder.act_on::<CheckHealth>(|actor, envelope| {
        let health = ProviderHealth::from_state(actor.model.breaker, Instant::now());
        let reply_to = envelope.reply_envelope();
        Reply::pending(async move {
            reply_to.send(health).await;
        })
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

        dispatch_request(actor, pending.request, sampling, pending.attempts, None);
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

/// Arms a one-shot scheduled send of `ProcessQueue` once `at` has passed.
///
/// Does nothing if a timer is already armed: `drain_timer` bounds this to one
/// outstanding schedule no matter how many requests queue up meanwhile.
fn arm_drain_timer(actor: &mut ManagedActor<Started, LLMProvider>, at: Instant) {
    if actor.model.drain_timer.is_some() {
        return;
    }
    actor.model.drain_timer = Some(actor.handle().send_at(ProcessQueue, at));
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
    degraded_for: Option<Duration>,
) {
    let primary_model = actor
        .model
        .config
        .as_ref()
        .map(|config| config.model.clone())
        .unwrap_or_default();

    // `degraded_for` is only ever `Some` where a fallback client was proved
    // to exist, so falling back to the primary here is belt and braces
    // rather than a routing decision.
    let degraded = match (
        degraded_for,
        actor.model.fallback_client.clone(),
        actor
            .model
            .config
            .as_ref()
            .and_then(|config| config.fallback_model.clone()),
    ) {
        (Some(retry_after), Some(client), Some(model)) => Some((client, model, retry_after)),
        _ => None,
    };

    let (client, model, pending_event) = match degraded {
        Some((client, model, retry_after)) => {
            let event = FailoverEvent::ModelDegraded {
                provider: actor.model.name.clone(),
                from_model: primary_model,
                to_model: model.clone(),
                retry_after_secs: retry_after.as_secs(),
            };
            tracing::warn!(event = %event, "Degrading to the fallback model");
            (client, model, Some(event))
        }
        None => {
            let Some(client) = actor.model.client.clone() else {
                tracing::error!(
                    correlation_id = %request.correlation_id,
                    "Cannot dispatch request: provider has no client"
                );
                return;
            };
            (client, primary_model, None)
        }
    };

    let streaming = actor
        .model
        .config
        .as_ref()
        .is_none_or(|config| config.streaming);

    let ctx = DispatchContext {
        provider_name: client.provider_name().to_string(),
        configured_name: actor.model.name.clone(),
        model,
        client,
        broker: actor.broker().clone(),
        provider: actor.handle().clone(),
        sampling,
        attempts,
        pending_event,
    };

    tokio::spawn(async move {
        // Published from inside the dispatch task, before the request goes
        // out, so it is on the broker strictly before this round's terminal
        // event — a subscriber that sees the round finish has necessarily
        // already been sent the degradation notice.
        if let Some(event) = ctx.pending_event.clone() {
            ctx.broker.broadcast(event).await;
        }

        if streaming {
            process_streaming_request(&ctx, &request).await;
        } else {
            process_non_streaming_request(&ctx, &request).await;
        }
    });
}

/// How long the API-reported rate limit still has to run, when degrading to
/// the fallback model is possible right now.
///
/// Pure, and deliberately narrow. `None` means dispatch normally:
///
/// - no fallback client was built, so there is nothing to degrade to;
/// - or the provider is not under an API-reported rate limit. A spent *local*
///   per-minute budget is our own bookkeeping, not the API refusing us, and
///   switching models would not make it go away.
fn degradation_window(
    rate_limited_until: Option<Instant>,
    fallback_client: Option<&Arc<dyn LLMClient>>,
    now: Instant,
) -> Option<Duration> {
    fallback_client?;
    let until = rate_limited_until?;
    (until > now).then(|| until.saturating_duration_since(now))
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

/// Broadcasts what a completed request cost.
///
/// Published unconditionally: the message is tiny, and with no subscriber a
/// broadcast is a no-op. Whether anything tallies it is decided by the
/// facade's `usage_tracking` toggle, which governs only whether the
/// accountant actor exists — the provider has no opinion and holds no handle
/// to it.
async fn report_usage(ctx: &DispatchContext, request: &LLMRequest, usage: Usage) {
    ctx.broker
        .broadcast(UsageReport {
            provider: ctx.configured_name.clone(),
            model: ctx.model.clone(),
            correlation_id: request.correlation_id.clone(),
            agent_id: request.agent_id.clone(),
            usage,
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
    /// Vendor name of the provider ("anthropic" / "openai"), for logging and
    /// system events.
    provider_name: String,
    /// The runtime-configured name this provider is registered under. Usage
    /// reports are keyed by this, never by the vendor.
    configured_name: String,
    /// The model serving **this dispatch**, which is the fallback model when
    /// the provider degraded. Carried on usage reports and on the round's
    /// terminal event, so what gets billed and what gets charted are the same
    /// model that actually answered.
    model: String,
    /// Merged sampling parameters, if any apply
    sampling: Option<SamplingParams>,
    /// How many attempts this request has already used
    attempts: u32,
    /// An event to publish before the request goes out, if this dispatch is
    /// itself news — currently only model degradation.
    pending_event: Option<FailoverEvent>,
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

    // Time to first token is measured from here rather than from the first
    // byte off the socket: what a caller waits through includes establishing
    // the request and the model's own prefill, and a metric that excluded
    // them would report a latency nobody experiences.
    let stream_started = std::time::Instant::now();

    // Start streaming request
    match client
        .send_streaming_request(
            &request.messages,
            tools,
            sampling,
            request.tool_choice.as_ref(),
        )
        .await
    {
        Ok(mut stream) => {
            let mut accumulated_text = String::new();
            let mut tool_calls = Vec::new();
            let mut stop_reason = StopReason::EndTurn;
            let mut usage = Usage::default();

            // Taken once per round and cleared by the first token, which is
            // what keeps a long stream from re-recording on every chunk.
            let mut awaiting_first_token = Some(stream_started);

            while let Some(result) = stream.next().await {
                match result {
                    Ok(event) => {
                        match event {
                            LLMStreamEvent::Start { .. } => {
                                // Stream already started, no action needed
                            }
                            LLMStreamEvent::Token { text } => {
                                if let Some(started) = awaiting_first_token.take() {
                                    crate::telemetry::metrics::record_time_to_first_token(
                                        &ctx.configured_name,
                                        &ctx.model,
                                        started.elapsed().as_secs_f64(),
                                    );
                                }
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
                                usage: reported,
                            } => {
                                stop_reason = reason;
                                usage = reported;
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
                    usage,
                    model: ctx.model.clone(),
                })
                .await;

            report_usage(ctx, request, usage).await;

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

            let will_retry = e.retry_after().is_some();
            report_failure(ctx, request, &e).await;

            // A retryable failure keeps the round open: the retry runs under
            // the same correlation ID, so ending the round here would both
            // complete the caller's turn early and orphan the retry as a
            // duplicate request nobody is collecting. If the provider drops
            // the request instead of retrying, the `RateLimitReported`
            // handler closes the round.
            if !will_retry {
                broker
                    .broadcast(LLMStreamEnd {
                        correlation_id: correlation_id.clone(),
                        stop_reason: StopReason::Error,
                        usage: Usage::default(),
                        model: ctx.model.clone(),
                    })
                    .await;
            }
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
        .send_request(
            &request.messages,
            tools,
            ctx.sampling.as_ref(),
            request.tool_choice.as_ref(),
        )
        .await
    {
        Ok(response) => {
            let usage = response.usage;

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

            report_usage(ctx, request, usage).await;

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

    /// A stand-in client, only ever used to prove `degradation_window` looks
    /// at whether a fallback client *exists*.
    fn some_client() -> Arc<dyn LLMClient> {
        Arc::new(
            OpenAIClient::new(
                "http://localhost:1/v1".to_string(),
                &ProviderConfig::openai_compatible("http://localhost:1/v1", "fallback-model"),
            )
            .expect("constructing a client against a local URL must succeed"),
        )
    }

    #[test]
    fn without_a_fallback_client_nothing_degrades() {
        let now = Instant::now();

        assert_eq!(
            degradation_window(Some(now + Duration::from_secs(30)), None, now),
            None,
            "a provider with no fallback model must queue as it always has"
        );
    }

    #[test]
    fn a_live_api_rate_limit_opens_the_degradation_window() {
        let now = Instant::now();
        let client = some_client();

        assert_eq!(
            degradation_window(Some(now + Duration::from_secs(30)), Some(&client), now),
            Some(Duration::from_secs(30))
        );
    }

    #[test]
    fn a_spent_local_budget_is_not_a_reason_to_change_models() {
        let now = Instant::now();
        let client = some_client();

        // `rate_limited_until` is only ever set from an API-reported 429. A
        // provider blocked purely by its own per-minute budget has none.
        assert_eq!(degradation_window(None, Some(&client), now), None);
    }

    #[test]
    fn an_expired_rate_limit_closes_the_degradation_window() {
        let now = Instant::now();
        let client = some_client();

        assert_eq!(
            degradation_window(Some(now), Some(&client), now),
            None,
            "degradation is self-healing: once the window passes, the primary model serves again"
        );
        assert_eq!(
            degradation_window(Some(now - Duration::from_secs(1)), Some(&client), now),
            None
        );
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
            tool_choice: None,
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
        let config = ProviderConfig::new("test-key")
            .with_rate_limit(RateLimitConfig::default().without_queueing());

        assert!(config.streaming);
    }

    /// Observes `LLMStreamEnd` broadcasts so tests can assert a round closed.
    #[acton_actor]
    struct EndProbe;

    #[tokio::test]
    async fn dropped_rate_limited_request_closes_its_stream() {
        use crate::llm::config::RetryConfig;
        use crate::messages::Message;
        use crate::types::{AgentId, CorrelationId};

        let mut runtime = ActonApp::launch_async().await;

        let (end_tx, mut end_rx) = tokio::sync::mpsc::channel::<(CorrelationId, StopReason)>(1);
        let mut probe = runtime.new_actor::<EndProbe>();
        probe.act_on::<LLMStreamEnd>(move |_actor, ctx| {
            let tx = end_tx.clone();
            let correlation_id = ctx.message().correlation_id.clone();
            let stop_reason = ctx.message().stop_reason;
            Reply::pending(async move {
                let _ = tx.send((correlation_id, stop_reason)).await;
            })
        });
        probe.handle().subscribe::<LLMStreamEnd>().await;
        let _probe = probe.start().await;

        let provider = LLMProvider::spawn(
            &mut runtime,
            "test-provider",
            ProviderConfig::new("test-key").with_retry(RetryConfig::no_retries()),
        )
        .await;

        let request = LLMRequest {
            correlation_id: CorrelationId::new(),
            agent_id: AgentId::new(),
            messages: vec![Message::user("hello")],
            tools: None,
            sampling: None,
            tool_choice: None,
        };
        let correlation_id = request.correlation_id.clone();

        // With retries disabled the provider must drop this request — and a
        // dropped request's caller is waiting on the stream, so the drop must
        // broadcast the round's terminal event.
        provider
            .send(RateLimitReported {
                retry_after: Duration::from_secs(60),
                request,
                attempts: 0,
            })
            .await;

        let (ended, stop_reason) = tokio::time::timeout(Duration::from_secs(5), end_rx.recv())
            .await
            .expect("dropping a rate-limited request must close its stream")
            .expect("probe channel closed before delivering the stream end");
        assert_eq!(ended, correlation_id);
        assert_eq!(stop_reason, StopReason::Error);

        runtime
            .shutdown_all()
            .await
            .expect("runtime should shut down cleanly");
    }
}
