//! Metric instruments and the thin recording helpers call sites use.
//!
//! # Shape
//!
//! Every public function here is unconditional and takes plain Rust types, so
//! instrumentation sites in the prompt loop and the provider actor carry no
//! `#[cfg]` of their own. With the `otel` feature off, each one compiles to an
//! empty body the optimiser deletes.
//!
//! Instruments are built once behind a [`OnceLock`](std::sync::OnceLock) and
//! reused: creating one per call would allocate on every token counted.
//!
//! # Attribute discipline
//!
//! Metric attributes are low-cardinality by construction — provider, model,
//! tool name, outcome. Correlation and agent IDs are deliberately absent;
//! they are unique per request, so as attributes they would create one time
//! series per request and melt the backend. IDs belong on spans, which is
//! where [`crate::telemetry::spans`] puts them.

/// Instrument names, shared with the tests that assert on them so a rename
/// cannot pass silently.
///
/// Feature-gated with the instruments they name: with `otel` off there is no
/// instrument to name and nothing reads these.
#[cfg(feature = "otel")]
pub(crate) mod names {
    /// Tokens consumed, by kind.
    pub const TOKENS: &str = "acton_ai.tokens";
    /// Completed provider requests.
    pub const REQUESTS: &str = "acton_ai.llm.requests";
    /// Rate limits reported by providers.
    pub const RATE_LIMITS: &str = "acton_ai.llm.rate_limits";
    /// Budget threshold crossings and denials.
    pub const BUDGET_EVENTS: &str = "acton_ai.budget.events";
    /// Circuit-breaker transitions, chain failovers, and model degradations.
    pub const FAILOVER_EVENTS: &str = "acton_ai.failover.events";
    /// Wall time of one provider round.
    pub const REQUEST_DURATION: &str = "acton_ai.llm.request.duration";
    /// Stream start to first token.
    pub const TIME_TO_FIRST_TOKEN: &str = "acton_ai.llm.time_to_first_token";
    /// Wall time of one tool execution.
    pub const TOOL_DURATION: &str = "acton_ai.tool.duration";
}

/// Outcome attribute value for a successful operation.
pub(crate) const OUTCOME_OK: &str = "ok";
/// Outcome attribute value for a failed operation.
pub(crate) const OUTCOME_ERROR: &str = "error";

#[cfg(feature = "otel")]
mod imp {
    use super::names;
    use opentelemetry::metrics::{Counter, Histogram};
    use opentelemetry::{global, KeyValue};
    use std::sync::OnceLock;

    /// Instrumentation scope name reported alongside every instrument.
    const SCOPE: &str = "acton-ai";

    /// Bucket boundaries, in seconds, for every duration histogram.
    ///
    /// The SDK's default boundaries start `[0, 5, 10, 25, …]` and run to
    /// 10_000 — sensible for milliseconds and useless for seconds, which is
    /// the unit everything here records. Nearly every LLM round, tool call,
    /// and first token would land in one bucket and no percentile would mean
    /// anything. These span 5ms to two minutes, which is the range this crate
    /// actually observes.
    const DURATION_BUCKETS_SECONDS: &[f64] = &[
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
    ];

    /// Every instrument the crate records, built once.
    struct Instruments {
        tokens: Counter<u64>,
        requests: Counter<u64>,
        rate_limits: Counter<u64>,
        budget_events: Counter<u64>,
        failover_events: Counter<u64>,
        request_duration: Histogram<f64>,
        time_to_first_token: Histogram<f64>,
        tool_duration: Histogram<f64>,
    }

    /// Resolves the instrument set, building it on first use.
    ///
    /// With no meter provider installed the global meter is a no-op, so the
    /// instruments built here are themselves no-ops and recording through
    /// them costs a call and nothing else.
    fn instruments() -> &'static Instruments {
        static INSTRUMENTS: OnceLock<Instruments> = OnceLock::new();
        INSTRUMENTS.get_or_init(|| {
            let meter = global::meter(SCOPE);
            Instruments {
                tokens: meter
                    .u64_counter(names::TOKENS)
                    .with_description("Tokens consumed, by kind")
                    .with_unit("{token}")
                    .build(),
                requests: meter
                    .u64_counter(names::REQUESTS)
                    .with_description("Completed LLM provider requests")
                    .with_unit("{request}")
                    .build(),
                rate_limits: meter
                    .u64_counter(names::RATE_LIMITS)
                    .with_description("Rate limits reported by LLM providers")
                    .with_unit("{event}")
                    .build(),
                budget_events: meter
                    .u64_counter(names::BUDGET_EVENTS)
                    .with_description("Budget thresholds crossed and caps exceeded")
                    .with_unit("{event}")
                    .build(),
                failover_events: meter
                    .u64_counter(names::FAILOVER_EVENTS)
                    .with_description(
                        "Circuit breaker transitions, chain failovers, and model degradations",
                    )
                    .with_unit("{event}")
                    .build(),
                request_duration: meter
                    .f64_histogram(names::REQUEST_DURATION)
                    .with_description("Wall time of one LLM provider round")
                    .with_unit("s")
                    .with_boundaries(DURATION_BUCKETS_SECONDS.to_vec())
                    .build(),
                time_to_first_token: meter
                    .f64_histogram(names::TIME_TO_FIRST_TOKEN)
                    .with_description("Stream start to first streamed token")
                    .with_unit("s")
                    .with_boundaries(DURATION_BUCKETS_SECONDS.to_vec())
                    .build(),
                tool_duration: meter
                    .f64_histogram(names::TOOL_DURATION)
                    .with_description("Wall time of one tool execution")
                    .with_unit("s")
                    .with_boundaries(DURATION_BUCKETS_SECONDS.to_vec())
                    .build(),
            }
        })
    }

    pub(super) fn record_tokens(provider: &str, model: &str, usage: &crate::messages::Usage) {
        let counter = &instruments().tokens;
        // Four counters would be four instruments to name, chart, and keep in
        // step. One counter split by `kind` sums to total tokens for free and
        // still breaks down per kind when asked.
        for (kind, value) in [
            ("input", usage.input_tokens),
            ("output", usage.output_tokens),
            ("cache_read", usage.cache_read_tokens),
            ("cache_creation", usage.cache_creation_tokens),
        ] {
            // Zero is the common case for the cache counters; recording it
            // would create time series that only ever say nothing happened.
            if value == 0 {
                continue;
            }
            counter.add(
                value,
                &[
                    KeyValue::new("provider", provider.to_string()),
                    KeyValue::new("model", model.to_string()),
                    KeyValue::new("kind", kind),
                ],
            );
        }
    }

    pub(super) fn record_request(provider: &str, model: &str) {
        instruments().requests.add(
            1,
            &[
                KeyValue::new("provider", provider.to_string()),
                KeyValue::new("model", model.to_string()),
            ],
        );
    }

    pub(super) fn record_rate_limit(provider: &str) {
        instruments()
            .rate_limits
            .add(1, &[KeyValue::new("provider", provider.to_string())]);
    }

    pub(super) fn record_budget_event(kind: &'static str, scope: &'static str) {
        instruments().budget_events.add(
            1,
            &[KeyValue::new("kind", kind), KeyValue::new("scope", scope)],
        );
    }

    pub(super) fn record_failover_event(kind: &'static str, provider: &str) {
        instruments().failover_events.add(
            1,
            &[
                KeyValue::new("kind", kind),
                KeyValue::new("provider", provider.to_string()),
            ],
        );
    }

    pub(super) fn record_request_duration(
        provider: &str,
        model: &str,
        outcome: &'static str,
        seconds: f64,
    ) {
        instruments().request_duration.record(
            seconds,
            &[
                KeyValue::new("provider", provider.to_string()),
                KeyValue::new("model", model.to_string()),
                KeyValue::new("outcome", outcome),
            ],
        );
    }

    pub(super) fn record_time_to_first_token(provider: &str, model: &str, seconds: f64) {
        instruments().time_to_first_token.record(
            seconds,
            &[
                KeyValue::new("provider", provider.to_string()),
                KeyValue::new("model", model.to_string()),
            ],
        );
    }

    pub(super) fn record_tool_duration(tool: &str, outcome: &'static str, seconds: f64) {
        instruments().tool_duration.record(
            seconds,
            &[
                KeyValue::new("tool", tool.to_string()),
                KeyValue::new("outcome", outcome),
            ],
        );
    }
}

/// Adds one round's reported usage to the token counter, split by kind.
///
/// Recorded only by the telemetry actor, which exists only with the
/// `otel` feature, so this is gated alongside it.
#[cfg(feature = "otel")]
pub(crate) fn record_tokens(provider: &str, model: &str, usage: &crate::messages::Usage) {
    #[cfg(feature = "otel")]
    imp::record_tokens(provider, model, usage);
    #[cfg(not(feature = "otel"))]
    {
        let _ = (provider, model, usage);
    }
}

/// Counts one completed provider request.
///
/// Recorded only by the telemetry actor, which exists only with the
/// `otel` feature, so this is gated alongside it.
#[cfg(feature = "otel")]
pub(crate) fn record_request(provider: &str, model: &str) {
    #[cfg(feature = "otel")]
    imp::record_request(provider, model);
    #[cfg(not(feature = "otel"))]
    {
        let _ = (provider, model);
    }
}

/// Counts one rate limit reported by a provider.
///
/// Recorded only by the telemetry actor, which exists only with the
/// `otel` feature, so this is gated alongside it.
#[cfg(feature = "otel")]
pub(crate) fn record_rate_limit(provider: &str) {
    #[cfg(feature = "otel")]
    imp::record_rate_limit(provider);
    #[cfg(not(feature = "otel"))]
    {
        let _ = provider;
    }
}

/// Counts one budget event. `kind` is `threshold_crossed` or `exceeded`;
/// `scope` is `total` or `provider`.
///
/// Recorded only by the telemetry actor, which exists only with the
/// `otel` feature, so this is gated alongside it.
#[cfg(feature = "otel")]
pub(crate) fn record_budget_event(kind: &'static str, scope: &'static str) {
    #[cfg(feature = "otel")]
    imp::record_budget_event(kind, scope);
    #[cfg(not(feature = "otel"))]
    {
        let _ = (kind, scope);
    }
}

/// Counts one failover event. `kind` is one of `circuit_opened`,
/// `circuit_closed`, `failed_over`, or `model_degraded`.
///
/// Unlike the budget counter, this one *is* labelled with the provider name:
/// which provider is failing is the whole question an operator brings to this
/// metric, and the cardinality is bounded by the number of configured
/// providers.
///
/// Recorded only by the telemetry actor, which exists only with the
/// `otel` feature, so this is gated alongside it.
#[cfg(feature = "otel")]
pub(crate) fn record_failover_event(kind: &'static str, provider: &str) {
    #[cfg(feature = "otel")]
    imp::record_failover_event(kind, provider);
    #[cfg(not(feature = "otel"))]
    {
        let _ = (kind, provider);
    }
}

/// Records how long one provider round took.
pub(crate) fn record_request_duration(
    provider: &str,
    model: &str,
    outcome: &'static str,
    seconds: f64,
) {
    #[cfg(feature = "otel")]
    imp::record_request_duration(provider, model, outcome, seconds);
    #[cfg(not(feature = "otel"))]
    {
        let _ = (provider, model, outcome, seconds);
    }
}

/// Records the gap between a stream starting and its first token arriving.
pub(crate) fn record_time_to_first_token(provider: &str, model: &str, seconds: f64) {
    #[cfg(feature = "otel")]
    imp::record_time_to_first_token(provider, model, seconds);
    #[cfg(not(feature = "otel"))]
    {
        let _ = (provider, model, seconds);
    }
}

/// Records how long one tool execution took.
pub(crate) fn record_tool_duration(tool: &str, outcome: &'static str, seconds: f64) {
    #[cfg(feature = "otel")]
    imp::record_tool_duration(tool, outcome, seconds);
    #[cfg(not(feature = "otel"))]
    {
        let _ = (tool, outcome, seconds);
    }
}
