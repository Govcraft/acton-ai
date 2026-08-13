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

mod actor;
pub mod pricing;

pub use actor::{CostAccountant, GetUsage, ModelUsage, ProviderUsage, UsageSnapshot};
pub use pricing::{dollars_per_mtok_to_microusd, CostBreakdown, ModelPricing, PricingTable};
