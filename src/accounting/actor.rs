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

use crate::accounting::budget::{
    self, BudgetConfig, BudgetDecision, BudgetEvent, BudgetStatus, CheckBudget, Spend,
};
use crate::accounting::pricing::{CostBreakdown, PricingTable};
use crate::messages::{Usage, UsageReport};
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

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
    /// Where spending stands against the configured caps.
    ///
    /// `None` means no budget is configured — never "a budget with nothing
    /// spent against it".
    pub budget: Option<BudgetStatus>,
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
    /// The caps to enforce, installed at spawn. `None` disables enforcement
    /// entirely: [`CheckBudget`] is never asked, and the fold does no extra
    /// work.
    budget: Option<BudgetConfig>,
}

impl CostAccountant {
    /// Spawns the accountant and subscribes it to [`UsageReport`].
    ///
    /// The subscription is registered on the **builder**, before `start()`:
    /// subscribing afterwards is silently ignored, which would leave an
    /// accountant that runs happily and tallies nothing.
    ///
    /// `budget` is `None` when no caps are configured, which is the only
    /// thing that switches enforcement off — there is no separate flag.
    pub async fn spawn(
        runtime: &mut ActorRuntime,
        pricing: PricingTable,
        budget: Option<BudgetConfig>,
    ) -> ActorHandle {
        let mut builder =
            runtime.new_actor_with_name::<CostAccountant>("cost_accountant".to_string());

        // Installed directly on the idle builder rather than sent as a
        // message, so pricing is in place before the actor can receive
        // anything at all.
        builder.model.pricing = pricing;
        builder.model.budget = budget;

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

    /// Folds one report in and reports which caps it pushed spending past.
    ///
    /// The before/after pricing pass only happens when a budget exists, so
    /// the unbudgeted fold stays exactly as cheap as it was.
    fn record_and_detect(&mut self, report: &UsageReport) -> Vec<BudgetEvent> {
        let Some(budget) = self.budget.clone() else {
            self.record(report);
            return Vec::new();
        };

        let before = self.spend();
        self.record(report);
        let after = self.spend();

        budget::crossings(&budget, &before, &after)
    }

    /// Prices the tallies into spend per scope.
    ///
    /// Uses the same snapshot-time pricing [`Self::snapshot`] does, so a cap
    /// is compared against exactly the figure `usage()` reports.
    fn spend(&self) -> Spend {
        let mut spend = Spend::default();

        for (name, tally) in &self.by_provider {
            match self.pricing.cost_for(name, &tally.usage) {
                Some(cost) => {
                    spend.by_provider.insert(name.clone(), cost.total_microusd);
                    spend.total_microusd = spend.total_microusd.saturating_add(cost.total_microusd);
                }
                // Tokens with no rate are a blind spot; zero tokens are not,
                // since no rate could make them cost anything.
                None if !tally.usage.is_empty() => spend.unpriced.push(name.clone()),
                None => {}
            }
        }

        // `by_provider` is a HashMap, so without this the reason a request was
        // refused would name its providers in a different order each run.
        spend.unpriced.sort_unstable();
        spend
    }

    /// Decides whether one more request to `provider` may go out.
    fn check(&self, provider: &str) -> BudgetDecision {
        self.budget
            .as_ref()
            .map_or(BudgetDecision::Allowed, |budget| {
                budget::decide(budget, &self.spend(), provider)
            })
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
            budget: self
                .budget
                .as_ref()
                .map(|budget| budget::status(budget, &self.spend())),
        }
    }
}

/// Wires the accountant's three handlers.
fn configure_handlers(builder: &mut ManagedActor<Idle, CostAccountant>) {
    // Folding a report is bookkeeping: no IO, nothing to await. The one
    // asynchronous thing it can produce is a broadcast, which is a *send* —
    // asking anything from `mutate_on` would deadlock the message loop.
    builder.mutate_on::<UsageReport>(|actor, envelope| {
        let events = actor.model.record_and_detect(envelope.message());
        if events.is_empty() {
            return Reply::ready();
        }

        // Cloned out of the actor before the async block: the closure is
        // `Fn`, so it cannot hold a borrow of the model across the await.
        let broker = actor.broker().clone();
        Reply::pending(async move {
            for event in events {
                tracing::warn!(%event, "budget event");
                broker.broadcast(event).await;
            }
        })
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

    // Pre-flight enforcement. Also a read, so a check never queues behind
    // another check — only behind the folds already ahead of it, which is
    // exactly the ordering that makes the answer current.
    builder.act_on::<CheckBudget>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let decision = actor.model.check(&envelope.message().provider);
        Reply::pending(async move {
            reply.send(decision).await;
        })
    });
}

// =============================================================================
// Budget-event callback subscriber
// =============================================================================

/// A caller-supplied [`BudgetEvent`] handler.
///
/// Wrapped because `#[acton_actor]` derives `Debug` on the model that holds
/// it, and a boxed closure has none of its own.
#[derive(Clone, Default)]
pub(crate) struct BudgetCallback(pub(crate) Option<Arc<dyn Fn(BudgetEvent) + Send + Sync>>);

impl std::fmt::Debug for BudgetCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BudgetCallback")
            .field(&self.0.is_some())
            .finish()
    }
}

/// Runs a caller's callback for every [`BudgetEvent`] on the broker.
///
/// A whole actor for one closure buys the caller something they cannot get
/// otherwise: broker events are delivered to actors, and the callback has to
/// live somewhere that a subscription can reach.
#[acton_actor]
pub(crate) struct BudgetEventListener {
    callback: BudgetCallback,
}

impl BudgetEventListener {
    /// Spawns the listener, subscribed to [`BudgetEvent`] **on the builder**.
    ///
    /// Subscribing after `start()` is silently ignored, which would leave a
    /// listener that runs happily and never fires.
    pub(crate) async fn spawn(
        runtime: &mut ActorRuntime,
        callback: Arc<dyn Fn(BudgetEvent) + Send + Sync>,
    ) -> ActorHandle {
        let mut builder =
            runtime.new_actor_with_name::<BudgetEventListener>("budget_listener".to_string());

        builder.model.callback = BudgetCallback(Some(callback));

        // `act_on`, not `mutate_on`: the callback reads the event and touches
        // no state, so several events can run through it concurrently. It is
        // caller code on the actor's thread, so it must be cheap and must not
        // block.
        builder.act_on::<BudgetEvent>(|actor, envelope| {
            if let Some(ref callback) = actor.model.callback.0 {
                callback(envelope.message().clone());
            }
            Reply::ready()
        });

        builder.handle().subscribe::<BudgetEvent>().await;

        builder.start().await
    }
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

    /// A table pricing `claude` at $3/$15 per MTok, so 1 MTok of input is
    /// exactly $3.00 and the arithmetic in these tests stays legible.
    fn claude_pricing() -> PricingTable {
        let mut pricing = PricingTable::new();
        pricing.insert("claude", ModelPricing::from_dollars_per_mtok(3.0, 15.0));
        pricing
    }

    /// An accountant wired with pricing and caps, without a runtime.
    fn budgeted(pricing: PricingTable, budget: crate::accounting::Budget) -> CostAccountant {
        CostAccountant {
            pricing,
            budget: Some(budget.to_config().expect("a valid budget")),
            ..CostAccountant::default()
        }
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

    // -------------------------------------------------------------------
    // Budget enforcement
    // -------------------------------------------------------------------

    #[test]
    fn an_unbudgeted_accountant_allows_everything_and_reports_no_status() {
        let mut accountant = CostAccountant {
            pricing: claude_pricing(),
            ..CostAccountant::default()
        };
        // A billion tokens is still allowed when nothing capped it.
        accountant.record(&report("claude", "sonnet", 1_000_000_000, 0));

        assert_eq!(accountant.check("claude"), BudgetDecision::Allowed);
        assert!(
            accountant.snapshot().budget.is_none(),
            "no budget must read as absent, not as a budget with room left"
        );
    }

    #[test]
    fn spending_past_the_cap_denies_the_next_request() {
        let mut accountant = budgeted(claude_pricing(), crate::accounting::Budget::usd(2.00));

        // 1 MTok of input at $3/MTok is $3.00, past a $2.00 cap.
        accountant.record(&report("claude", "sonnet", 1_000_000, 0));

        assert_eq!(
            accountant.check("claude"),
            BudgetDecision::Denied {
                scope: crate::accounting::BudgetScope::Total,
                limit_microusd: 2_000_000,
                spent_microusd: 3_000_000,
            }
        );
    }

    #[test]
    fn a_provider_cap_leaves_other_providers_alone() {
        let mut pricing = claude_pricing();
        pricing.insert("local", ModelPricing::from_dollars_per_mtok(1.0, 1.0));
        let mut accountant = budgeted(
            pricing,
            crate::accounting::Budget::for_provider("claude", 2.00),
        );

        accountant.record(&report("claude", "sonnet", 1_000_000, 0));

        assert!(matches!(
            accountant.check("claude"),
            BudgetDecision::Denied { .. }
        ));
        assert_eq!(accountant.check("local"), BudgetDecision::Allowed);
    }

    #[test]
    fn usage_that_cannot_be_priced_denies_by_default() {
        // `local` has no rates, so its spending is invisible — a cap over it
        // would never be reached, which is the case that has to fail closed.
        let mut accountant = budgeted(claude_pricing(), crate::accounting::Budget::usd(5.00));

        accountant.record(&report("local", "qwen", 500, 100));

        assert_eq!(
            accountant.check("claude"),
            BudgetDecision::Unpriced {
                providers: vec!["local".to_string()],
            }
        );
    }

    #[test]
    fn a_zero_usage_report_from_an_unpriced_provider_is_not_a_blind_spot() {
        // No rate can make zero tokens cost anything, so this must not
        // refuse every subsequent request.
        let mut accountant = budgeted(claude_pricing(), crate::accounting::Budget::usd(5.00));

        accountant.record(&report("local", "qwen", 0, 0));

        assert_eq!(accountant.check("claude"), BudgetDecision::Allowed);
    }

    #[test]
    fn allow_unpriced_counts_invisible_usage_as_zero() {
        let mut accountant = budgeted(
            claude_pricing(),
            crate::accounting::Budget::usd(5.00).allow_unpriced(),
        );

        accountant.record(&report("local", "qwen", 500, 100));

        assert_eq!(accountant.check("claude"), BudgetDecision::Allowed);
    }

    #[test]
    fn folding_a_report_reports_the_thresholds_it_crossed() {
        let mut accountant = budgeted(claude_pricing(), crate::accounting::Budget::usd(4.00));

        // $2.70 of $4.00 is 67% — under the default 80% warning.
        let quiet = accountant.record_and_detect(&report("claude", "sonnet", 900_000, 0));
        assert!(quiet.is_empty(), "events = {quiet:?}");

        // $3.30 total is 82%: the warning fires, the cap does not.
        let warned = accountant.record_and_detect(&report("claude", "sonnet", 200_000, 0));
        assert_eq!(warned.len(), 1, "events = {warned:?}");
        assert!(matches!(
            warned[0],
            BudgetEvent::ThresholdCrossed {
                percent_used: 82,
                ..
            }
        ));

        // Past the cap: exceeded fires, the warning does not fire twice.
        let exceeded = accountant.record_and_detect(&report("claude", "sonnet", 900_000, 0));
        assert_eq!(exceeded.len(), 1, "events = {exceeded:?}");
        assert!(matches!(exceeded[0], BudgetEvent::Exceeded { .. }));

        // And nothing fires again however much more is spent.
        let quiet_again = accountant.record_and_detect(&report("claude", "sonnet", 900_000, 0));
        assert!(quiet_again.is_empty(), "events = {quiet_again:?}");
    }

    #[test]
    fn an_unbudgeted_fold_detects_nothing_and_still_tallies() {
        let mut accountant = CostAccountant {
            pricing: claude_pricing(),
            ..CostAccountant::default()
        };

        let events = accountant.record_and_detect(&report("claude", "sonnet", 1_000_000, 0));

        assert!(events.is_empty());
        assert_eq!(accountant.snapshot().totals.input_tokens, 1_000_000);
    }

    #[test]
    fn the_snapshot_carries_budget_standing() {
        let mut accountant = budgeted(
            claude_pricing(),
            crate::accounting::Budget::usd(4.00).provider("claude", 3.50),
        );

        accountant.record(&report("claude", "sonnet", 1_000_000, 0));

        let status = accountant
            .snapshot()
            .budget
            .expect("a configured budget must appear in the snapshot");
        let total = status.total.expect("a total cap was configured");
        assert_eq!(total.spent_microusd, 3_000_000);
        assert_eq!(total.remaining_microusd(), 1_000_000);
        assert_eq!(total.percent_used(), 75);
        assert_eq!(
            status.provider("claude").unwrap().remaining_microusd(),
            500_000
        );
    }

    #[test]
    fn budget_spend_matches_the_cost_the_snapshot_reports() {
        // The cap has to be compared against the same figure `usage()` shows,
        // or an operator cannot reconcile a refusal with what they were told
        // they had spent.
        let mut accountant = budgeted(claude_pricing(), crate::accounting::Budget::usd(10.00));
        accountant.record(&report("claude", "sonnet", 1_234_567, 89_012));

        let snapshot = accountant.snapshot();
        let reported = snapshot.cost.expect("everything is priced").total_microusd;
        let against_cap = snapshot.budget.unwrap().total.unwrap().spent_microusd;

        assert_eq!(reported, against_cap);
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
