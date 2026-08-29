//! OpenTelemetry export: traces over the prompt loop, metrics for tokens,
//! latency, and reliability.
//!
//! Two audiences, the same wiring. Operators get Grafana-ready signals off an
//! OTLP collector; library users get a one-liner, TOML parity, and nothing to
//! pay when telemetry is off.
//!
//! ```rust,ignore
//! let ai = ActonAI::builder()
//!     .anthropic_from_env()
//!     .telemetry_otlp("http://localhost:4318")
//!     .launch()
//!     .await?;
//! ```
//!
//! # What is exported
//!
//! Traces are created in the prompt loop, which already brackets every
//! provider dispatch and tool execution in time:
//!
//! ```text
//! acton_ai.turn                 one per collect()/extract()
//! ├── acton_ai.llm_round        one per provider dispatch
//! ├── acton_ai.tool             one per tool execution
//! └── acton_ai.llm_round
//! ```
//!
//! Metrics come from two places: durations are recorded inline at the site
//! that measures them, and token/request counters are folded by a
//! `TelemetryActor` subscribed to the same
//! [`UsageReport`](crate::messages::UsageReport) broadcast the cost
//! accountant reads. One broadcast, two independent subscribers.
//!
//! # Deliberate omissions
//!
//! - **Tool arguments and results are never recorded.** They are user data
//!   and unbounded in size; a span is the wrong place for either.
//! - **IDs live on spans, never on metrics.** Correlation and agent IDs are
//!   unique per request, so as metric attributes they would produce one time
//!   series per request.
//! - **No context propagation into provider internals.** Every span is
//!   created in the prompt loop. Retry attempts and rate-limit waits inside
//!   the provider actor are therefore not their own spans; giving
//!   [`LLMRequest`](crate::messages::LLMRequest) a `traceparent` field is the
//!   documented way to add them later.
//!
//! # Feature gating
//!
//! Everything here except [`TelemetryConfig`] and its validation sits behind
//! the `otel` feature, which is on by default. A build without it still
//! parses `[telemetry]` from a config file and then **fails the launch**
//! naming the missing feature — see `unsupported_error`. An operator whose
//! telemetry config was silently ignored would go looking for a broken
//! collector; this way they are told the truth immediately.

pub mod config;

pub use config::{TelemetryConfig, DEFAULT_METRICS_INTERVAL_SECS, DEFAULT_SERVICE_NAME};

#[cfg(feature = "otel")]
pub use config::Telemetry;

#[cfg(feature = "otel")]
mod actor;
#[cfg(feature = "otel")]
mod init;

pub mod metrics;
pub mod spans;

#[cfg(feature = "otel")]
pub(crate) use actor::TelemetryActor;
#[cfg(feature = "otel")]
pub(crate) use init::init_telemetry;

#[cfg(feature = "otel")]
pub use init::{install_with_exporters, TelemetryGuard};

/// The error a build without the `otel` feature raises when a runtime is
/// nonetheless configured for telemetry.
///
/// Kept here rather than inline at the call site so the message stays in the
/// module that owns the feature, and so the feature-off test can assert on
/// exactly what an operator will read.
#[cfg(not(feature = "otel"))]
pub(crate) fn unsupported_error() -> crate::error::ActonAIError {
    crate::error::ActonAIError::configuration(
        "telemetry",
        "telemetry is configured, but this binary was built without the `otel` feature, so \
         nothing would be exported; rebuild with the `otel` feature (it is on by default) or \
         remove the [telemetry] section",
    )
}
