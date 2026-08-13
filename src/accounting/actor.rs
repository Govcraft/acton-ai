//! The [`CostAccountant`] actor and the snapshot it answers with.
//!
//! # Why an actor
//!
//! Running totals are mutable state that several producers write to. Giving
//! them one owning actor is what makes the tallies correct without a lock:
//! the message loop *is* the mutual exclusion, and every provider in the
//! runtime can broadcast concurrently without contending.
//!
//! # Shape
//!
//! - [`UsageReport`] folds in through `mutate_on`. The handler is pure — it
//!   touches nothing but `actor.model` and returns `Reply::ready()`. No IO,
//!   no `ask`, no spawning.
//! - [`GetUsage`] is answered from `act_on` through the envelope's reply
//!   address, so reads run concurrently with each other and never block the
//!   fold.
//! - Costs are computed **at snapshot time**, not as reports arrive, so
//!   re-pricing is a config change rather than a data migration.
//!
//! The accountant is a plain top-level actor, deliberately **not**
//! supervised: it performs no IO and holds no connection, so it has no
//! failure mode a restart would repair — and a restart would silently zero
//! the very totals it exists to keep.

use crate::accounting::pricing::{CostBreakdown, PricingTable};
use crate::messages::{Usage, UsageReport};
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Asks the accountant for a snapshot of everything tallied so far.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct GetUsage;

impl Request for GetUsage {
    type Response = UsageSnapshot;
}

/// A point-in-time view of everything the accountant has tallied.
///
/// Plain data: taking one neither resets the totals nor blocks further
/// tallying.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct UsageSnapshot {
    /// Grand total across every provider.
    pub totals: Usage,
    /// How many requests were reported in total.
    pub requests: u64,
    /// Grand-total cost, or `None`.
    ///
    /// `None` means at least one provider with recorded usage has no
    /// configured pricing, so no honest grand total exists — a sum over just
    /// the priced providers would read as the whole bill while silently
    /// omitting part of it. Per-provider costs below are still populated
    /// wherever they are known.
    pub cost: Option<CostBreakdown>,
    /// Per-provider tallies, keyed by configured provider name.
    pub providers: BTreeMap<String, ProviderUsage>,
}

impl UsageSnapshot {
    /// The tally for one configured provider, if it recorded anything.
    #[must_use]
    pub fn provider(&self, name: &str) -> Option<&ProviderUsage> {
        self.providers.get(name)
    }

    /// Grand total cost in dollars, when one is known.
    #[must_use]
    pub fn total_usd(&self) -> Option<f64> {
        self.cost.map(|cost| cost.total_usd())
    }
}

/// One provider's slice of a [`UsageSnapshot`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderUsage {
    /// Total usage across every model this provider served.
    pub usage: Usage,
    /// Requests this provider reported.
    pub requests: u64,
    /// Cost, or `None` when this provider has no configured pricing.
    pub cost: Option<CostBreakdown>,
    /// Per-model breakdown, keyed by model name.
    pub models: BTreeMap<String, ModelUsage>,
}

impl ProviderUsage {
    /// The tally for one model this provider served.
    #[must_use]
    pub fn model(&self, name: &str) -> Option<&ModelUsage> {
        self.models.get(name)
    }
}

/// One model's slice of a [`ProviderUsage`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelUsage {
    /// Usage attributed to this model.
    pub usage: Usage,
    /// Requests attributed to this model.
    pub requests: u64,
    /// Cost, or `None` when the owning provider has no configured pricing.
    pub cost: Option<CostBreakdown>,
}

/// Running tally for one provider. The actor's internal shape; the snapshot
/// types above are what callers see.
#[derive(Debug, Clone, Default)]
struct ProviderTally {
    usage: Usage,
    requests: u64,
    by_model: HashMap<String, ModelTally>,
}

/// Running tally for one model within a provider.
#[derive(Debug, Clone, Default)]
struct ModelTally {
    usage: Usage,
    requests: u64,
}

/// State owned by the cost accountant.
#[acton_actor]
pub struct CostAccountant {
    /// Grand total across every provider.
    totals: Usage,
    /// Total requests reported.
    requests: u64,
    /// Per-provider tallies, keyed by configured provider name.
    by_provider: HashMap<String, ProviderTally>,
    /// Rates installed at spawn from configuration. Read only at snapshot
    /// time, so it never sits on the fold path.
    pricing: PricingTable,
}

impl CostAccountant {
    /// Spawns the accountant and subscribes it to [`UsageReport`].
    ///
    /// The subscription is registered on the **builder**, before `start()`:
    /// subscribing afterwards is silently ignored, which would leave an
    /// accountant that runs happily and tallies nothing.
    pub async fn spawn(runtime: &mut ActorRuntime, pricing: PricingTable) -> ActorHandle {
        let mut builder =
            runtime.new_actor_with_name::<CostAccountant>("cost_accountant".to_string());

        // Installed directly on the idle builder rather than sent as a
        // message, so pricing is in place before the actor can receive
        // anything at all.
        builder.model.pricing = pricing;

        configure_handlers(&mut builder);

        builder.handle().subscribe::<UsageReport>().await;

        builder.start().await
    }

    /// Folds one report into the tallies. Pure: total in, totals updated.
    fn record(&mut self, report: &UsageReport) {
        self.totals += report.usage;
        self.requests += 1;

        let provider = self.by_provider.entry(report.provider.clone()).or_default();
        provider.usage += report.usage;
        provider.requests += 1;

        let model = provider.by_model.entry(report.model.clone()).or_default();
        model.usage += report.usage;
        model.requests += 1;
    }

    /// Renders the current tallies, pricing each entry as it goes.
    fn snapshot(&self) -> UsageSnapshot {
        let mut providers = BTreeMap::new();
        // A grand total is only honest when every provider that spent
        // anything can be priced.
        let mut everything_priced = true;
        let mut grand_cost = CostBreakdown::default();

        for (name, tally) in &self.by_provider {
            let cost = self.pricing.cost_for(name, &tally.usage);
            match cost {
                Some(cost) => grand_cost += cost,
                None => everything_priced = false,
            }

            let models = tally
                .by_model
                .iter()
                .map(|(model, model_tally)| {
                    (
                        model.clone(),
                        ModelUsage {
                            usage: model_tally.usage,
                            requests: model_tally.requests,
                            cost: self.pricing.cost_for(name, &model_tally.usage),
                        },
                    )
                })
                .collect();

            providers.insert(
                name.clone(),
                ProviderUsage {
                    usage: tally.usage,
                    requests: tally.requests,
                    cost,
                    models,
                },
            );
        }

        UsageSnapshot {
            totals: self.totals,
            requests: self.requests,
            cost: everything_priced.then_some(grand_cost),
            providers,
        }
    }
}

/// Wires the accountant's two handlers.
fn configure_handlers(builder: &mut ManagedActor<Idle, CostAccountant>) {
    // Folding a report is pure bookkeeping: no IO, no async work, nothing to
    // await. `Reply::ready()` keeps the message loop moving.
    builder.mutate_on::<UsageReport>(|actor, envelope| {
        actor.model.record(envelope.message());
        Reply::ready()
    });

    // Reads answer through the reply envelope from `act_on`, so several
    // snapshots can be served at once and none of them blocks the fold.
    builder.act_on::<GetUsage>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let snapshot = actor.model.snapshot();
        Reply::pending(async move {
            reply.send(snapshot).await;
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accounting::pricing::ModelPricing;
    use crate::types::{AgentId, CorrelationId};

    fn report(provider: &str, model: &str, input: u64, output: u64) -> UsageReport {
        UsageReport {
            provider: provider.to_string(),
            model: model.to_string(),
            correlation_id: CorrelationId::new(),
            agent_id: AgentId::new(),
            usage: Usage {
                input_tokens: input,
                output_tokens: output,
                ..Usage::default()
            },
        }
    }

    /// The fold is a pure method, so it can be exercised without a runtime.
    fn tallied(reports: &[UsageReport], pricing: PricingTable) -> UsageSnapshot {
        let mut accountant = CostAccountant {
            pricing,
            ..CostAccountant::default()
        };
        for report in reports {
            accountant.record(report);
        }
        accountant.snapshot()
    }

    #[test]
    fn an_untouched_accountant_reports_nothing_spent() {
        let snapshot = tallied(&[], PricingTable::new());

        assert_eq!(snapshot.requests, 0);
        assert_eq!(snapshot.totals, Usage::default());
        assert!(snapshot.providers.is_empty());
    }

    #[test]
    fn reports_accumulate_into_the_grand_total() {
        let snapshot = tallied(
            &[
                report("claude", "sonnet", 100, 10),
                report("claude", "sonnet", 200, 20),
            ],
            PricingTable::new(),
        );

        assert_eq!(snapshot.requests, 2);
        assert_eq!(snapshot.totals.input_tokens, 300);
        assert_eq!(snapshot.totals.output_tokens, 30);
    }

    #[test]
    fn tallies_are_kept_separately_per_provider() {
        let snapshot = tallied(
            &[
                report("claude", "sonnet", 100, 10),
                report("local", "qwen", 5, 1),
            ],
            PricingTable::new(),
        );

        assert_eq!(snapshot.provider("claude").unwrap().usage.input_tokens, 100);
        assert_eq!(snapshot.provider("local").unwrap().usage.input_tokens, 5);
        assert_eq!(snapshot.provider("claude").unwrap().requests, 1);
        assert_eq!(snapshot.totals.input_tokens, 105);
    }

    #[test]
    fn tallies_are_kept_separately_per_model_within_a_provider() {
        let snapshot = tallied(
            &[
                report("gateway", "sonnet", 100, 10),
                report("gateway", "haiku", 7, 2),
            ],
            PricingTable::new(),
        );

        let provider = snapshot.provider("gateway").unwrap();
        assert_eq!(provider.requests, 2);
        assert_eq!(provider.model("sonnet").unwrap().usage.input_tokens, 100);
        assert_eq!(provider.model("haiku").unwrap().usage.input_tokens, 7);
    }

    #[test]
    fn a_request_reporting_no_usage_still_counts_as_a_request() {
        // A provider that reports nothing has still spent money; dropping the
        // request would understate how many calls were made.
        let snapshot = tallied(&[report("local", "qwen", 0, 0)], PricingTable::new());

        assert_eq!(snapshot.requests, 1);
        assert_eq!(snapshot.provider("local").unwrap().requests, 1);
        assert!(snapshot.totals.is_empty());
    }

    #[test]
    fn costs_are_priced_per_provider() {
        let mut pricing = PricingTable::new();
        pricing.insert("claude", ModelPricing::from_dollars_per_mtok(3.0, 15.0));

        let snapshot = tallied(&[report("claude", "sonnet", 1_000_000, 1_000_000)], pricing);

        let cost = snapshot.provider("claude").unwrap().cost.unwrap();
        assert_eq!(cost.total_microusd, 18_000_000);
        assert_eq!(snapshot.cost.unwrap().total_microusd, 18_000_000);
    }

    #[test]
    fn an_unpriced_provider_tallies_tokens_but_reports_no_cost() {
        let snapshot = tallied(&[report("local", "qwen", 500, 100)], PricingTable::new());

        let provider = snapshot.provider("local").unwrap();
        assert_eq!(provider.usage.input_tokens, 500);
        assert!(
            provider.cost.is_none(),
            "an unpriced provider must not fabricate a $0.00 cost"
        );
    }

    #[test]
    fn a_partially_priced_run_withholds_the_grand_total() {
        // Summing only the priced provider would present part of the bill as
        // the whole of it.
        let mut pricing = PricingTable::new();
        pricing.insert("claude", ModelPricing::from_dollars_per_mtok(3.0, 15.0));

        let snapshot = tallied(
            &[
                report("claude", "sonnet", 1_000_000, 0),
                report("local", "qwen", 1_000_000, 0),
            ],
            pricing,
        );

        assert!(snapshot.provider("claude").unwrap().cost.is_some());
        assert!(snapshot.provider("local").unwrap().cost.is_none());
        assert!(
            snapshot.cost.is_none(),
            "a grand total that silently omits a provider is worse than none"
        );
    }

    #[test]
    fn per_model_costs_use_the_owning_providers_rates() {
        let mut pricing = PricingTable::new();
        pricing.insert("gateway", ModelPricing::from_dollars_per_mtok(3.0, 15.0));

        let snapshot = tallied(
            &[
                report("gateway", "sonnet", 1_000_000, 0),
                report("gateway", "haiku", 2_000_000, 0),
            ],
            pricing,
        );

        let provider = snapshot.provider("gateway").unwrap();
        assert_eq!(
            provider
                .model("sonnet")
                .unwrap()
                .cost
                .unwrap()
                .total_microusd,
            3_000_000
        );
        assert_eq!(
            provider
                .model("haiku")
                .unwrap()
                .cost
                .unwrap()
                .total_microusd,
            6_000_000
        );
    }

    #[test]
    fn snapshotting_does_not_reset_the_tallies() {
        let mut accountant = CostAccountant::default();
        accountant.record(&report("claude", "sonnet", 100, 10));

        let first = accountant.snapshot();
        let second = accountant.snapshot();

        assert_eq!(first.totals, second.totals);
        assert_eq!(second.requests, 1);
    }
}
