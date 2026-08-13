//! The [`TelemetryActor`]: broadcasts in, counters out.
//!
//! # Why an actor
//!
//! Counters need to see events that are published to the broker, and the
//! broker delivers to actors. This is the same shape
//! [`CostAccountant`](crate::accounting::CostAccountant) has, and
//! deliberately a *separate* subscriber to the same
//! [`UsageReport`](crate::messages::UsageReport) broadcast rather than a hook
//! bolted onto the accountant: one publisher, two independent consumers, each
//! able to be switched off without disturbing the other.
//!
//! # Handler discipline
//!
//! Every handler is `act_on`, never `mutate_on`: recording into an
//! OpenTelemetry instrument mutates the SDK's own aggregation state, not the
//! actor's model, so there is nothing here to serialise and several events
//! can be recorded concurrently. Handlers do no IO and never `ask` — the
//! recording call returns immediately and the SDK batches on its own thread.

use crate::accounting::{BudgetEvent, BudgetScope};
use crate::llm::FailoverEvent;
use crate::messages::{SystemEvent, UsageReport};
use crate::telemetry::metrics;
use acton_reactive::prelude::*;

/// Records broker traffic into OpenTelemetry instruments.
///
/// Stateless: the aggregation lives in the SDK, so the model is empty and
/// every handler is a read.
#[acton_actor]
pub(crate) struct TelemetryActor;

impl TelemetryActor {
    /// Spawns the actor, subscribed **on the builder** to every broadcast it
    /// records.
    ///
    /// Subscribing after `start()` is silently ignored by acton-reactive,
    /// which would leave an actor that runs happily and counts nothing.
    pub(crate) async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<TelemetryActor>("telemetry".to_string());

        // The same report the accountant folds into its tallies. Tokens and
        // request counts are the two things it carries that a dashboard wants.
        builder.act_on::<UsageReport>(|_actor, envelope| {
            let report = envelope.message();
            metrics::record_tokens(&report.provider, &report.model, &report.usage);
            metrics::record_request(&report.provider, &report.model);
            Reply::ready()
        });

        // Only rate limits are interesting here; the other system events are
        // lifecycle noise that would add series nobody charts.
        builder.act_on::<SystemEvent>(|_actor, envelope| {
            if let SystemEvent::RateLimitHit { provider, .. } = envelope.message() {
                metrics::record_rate_limit(provider);
            }
            Reply::ready()
        });

        // Counted as events, not gauges. Where spending actually stands is a
        // question for `usage()`, which prices the tallies on demand; a gauge
        // derived from event traffic would drift from it.
        builder.act_on::<BudgetEvent>(|_actor, envelope| {
            let event = envelope.message();
            let kind = match event {
                BudgetEvent::ThresholdCrossed { .. } => "threshold_crossed",
                BudgetEvent::Exceeded { .. } => "exceeded",
            };
            // The provider's *name* is deliberately not an attribute: caps are
            // few, but this keeps the series count fixed at two by two however
            // many providers are configured.
            let scope = match event.scope() {
                BudgetScope::Total => "total",
                BudgetScope::Provider(_) => "provider",
            };
            metrics::record_budget_event(kind, scope);
            Reply::ready()
        });

        // Where requests went, and why. The provider name *is* an attribute
        // here — unlike the budget counter, "which provider is failing" is
        // the entire question this metric answers, and the series count is
        // bounded by however many providers are configured.
        builder.act_on::<FailoverEvent>(|_actor, envelope| {
            let event = envelope.message();
            metrics::record_failover_event(event.kind(), event.provider());
            Reply::ready()
        });

        builder.handle().subscribe::<UsageReport>().await;
        builder.handle().subscribe::<SystemEvent>().await;
        builder.handle().subscribe::<BudgetEvent>().await;
        builder.handle().subscribe::<FailoverEvent>().await;

        builder.start().await
    }
}
