//! Token and cost accounting.
//!
//! Providers broadcast a [`UsageReport`](crate::messages::UsageReport) after
//! every request. When usage tracking is on — it is by default — the facade
//! spawns one [`CostAccountant`] subscribed to that broadcast, and
//! [`ActonAI::usage`](crate::facade::ActonAI::usage) asks it for a snapshot.
//!
//! Nothing holds a handle to the accountant except the facade: providers
//! publish to the broker and are indifferent to whether anyone is listening,
//! which is what lets the toggle be nothing more than "spawn it, or don't".
//!
//! ```rust,ignore
//! let ai = ActonAI::builder()
//!     .app_name("my-app")
//!     .ollama("qwen2.5:7b")
//!     .launch()
//!     .await?;
//!
//! ai.prompt("Hello!").collect().await?;
//!
//! let usage = ai.usage().await?;
//! println!("{} tokens over {} requests",
//!          usage.totals.total_tokens(), usage.requests);
//! ```
//!
//! # Cost
//!
//! Costs need a pricing table, which is configured per provider — there is no
//! bundled price list, because shipping vendor prices means shipping silently
//! wrong ones the day they change. Absent pricing yields `None`, never
//! `$0.00`. See [`pricing`] for the money representation.
//!
//! # Budgets
//!
//! A [`Budget`] turns those tallies into enforcement. Set one and every
//! provider dispatch is preceded by a [`CheckBudget`] ask; once a cap is
//! reached the prompt fails with
//! [`ActonAIErrorKind::BudgetExceeded`](crate::error::ActonAIErrorKind::BudgetExceeded)
//! instead of spending more.
//!
//! ```rust,ignore
//! let ai = ActonAI::builder()
//!     .anthropic_from_env()
//!     .budget_usd(5.00)
//!     .launch()
//!     .await?;
//! ```
//!
//! Two limits are worth stating plainly:
//!
//! - **A cap is a circuit breaker, not an exact meter.** The check is
//!   pre-flight, so a request already in flight when the ceiling is crossed
//!   still completes. Worst-case overshoot is one request per concurrent
//!   prompt loop.
//! - **Only the prompt loop is guarded.** Low-level callers that send an
//!   [`LLMRequest`](crate::messages::LLMRequest) straight to a provider actor
//!   bypass the check entirely. They can ask the accountant themselves with
//!   [`CheckBudget`]; nothing in the provider enforces it for them.
//!
//! See [`budget`] for the fail-closed rules around unpriced usage.

mod actor;
pub mod budget;
pub mod pricing;

pub(crate) use actor::BudgetEventListener;
pub use actor::{CostAccountant, GetUsage, ModelUsage, ProviderUsage, UsageSnapshot};
pub use budget::{
    Budget, BudgetConfig, BudgetDecision, BudgetEvent, BudgetScope, BudgetStatus, CheckBudget,
    ScopeStatus, DEFAULT_WARN_AT_PERCENT,
};
pub use pricing::{
    dollars_per_mtok_to_microusd, dollars_to_microusd, CostBreakdown, ModelPricing, PricingTable,
};
