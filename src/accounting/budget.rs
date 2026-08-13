//! Spending caps: the types an operator configures, and the pure decisions
//! the accountant makes from them.
//!
//! # A cap is a circuit breaker, not an exact meter
//!
//! Enforcement is **pre-flight**: before every provider dispatch the prompt
//! loop asks the accountant whether the next request may go out, and a `Denied`
//! answer becomes an
//! [`ActonAIErrorKind::BudgetExceeded`](crate::error::ActonAIErrorKind::BudgetExceeded).
//! A request already in flight when the ceiling is crossed still completes, so
//! the worst-case overshoot is one request per concurrent prompt loop. Nothing
//! here can cancel spending that has already happened; it can only refuse the
//! next call.
//!
//! # Money is an integer
//!
//! Caps are compared in the same integer micro-USD the rest of
//! [`accounting`](crate::accounting) computes in, priced by the same
//! [`PricingTable`](crate::accounting::PricingTable) at the same snapshot time.
//! Dollars (`f64`) exist only where a human types them — the builder and the
//! `[budget]` TOML section — and are converted exactly once by
//! [`dollars_to_microusd`](crate::accounting::dollars_to_microusd).
//!
//! # Unpriced usage fails closed
//!
//! A budget over a provider whose tokens cannot be priced is not a budget: the
//! ceiling would never be reached no matter how much was spent. So a
//! configured provider with no `[providers.<name>.pricing]` is a **launch
//! error** while a budget is set, and usage the accountant cannot price at
//! runtime **denies** the next request. [`Budget::allow_unpriced`] opts out,
//! counting unpriced usage as `$0` — an explicit acceptance of a blind spot.

use crate::accounting::pricing::{dollars_to_microusd, microusd_to_usd};
use crate::error::ActonAIError;
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// The warning threshold applied when none is configured, as a percentage of
/// the cap.
pub const DEFAULT_WARN_AT_PERCENT: u8 = 80;

/// Renders micro-USD as dollars, for messages. Lossy; display only.
pub(crate) fn usd(microusd: u64) -> f64 {
    microusd_to_usd(microusd)
}

// =============================================================================
// Scope
// =============================================================================

/// Which ceiling a decision or event is about.
///
/// ```rust
/// use acton_ai::prelude::BudgetScope;
///
/// assert_eq!(BudgetScope::Total.to_string(), "total");
/// assert_eq!(
///     BudgetScope::Provider("claude".to_string()).to_string(),
///     "provider `claude`"
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BudgetScope {
    /// The process-wide cap, across every provider.
    Total,
    /// The cap on one configured provider, named as it is in config.
    Provider(String),
}

impl fmt::Display for BudgetScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Total => write!(f, "total"),
            Self::Provider(name) => write!(f, "provider `{name}`"),
        }
    }
}

// =============================================================================
// Budget — the fluent, user-facing builder
// =============================================================================

/// A spending ceiling, in dollars, as a caller writes it.
///
/// There is no way to build an empty `Budget`: every constructor installs a
/// cap, so "a budget that limits nothing" is unrepresentable rather than
/// something to discover at launch.
///
/// ```rust
/// use acton_ai::prelude::Budget;
///
/// // Process-wide cap, warn at 80%, refuse at the cap.
/// let simple = Budget::usd(5.00);
///
/// // Total plus per-provider caps, warning earlier.
/// let detailed = Budget::usd(5.00)
///     .provider("claude", 2.00)
///     .provider("local", 0.50)
///     .warn_at_percent(50);
///
/// // Per-provider caps only, with no process-wide ceiling.
/// let per_provider = Budget::for_provider("claude", 2.00);
/// # let _ = (simple, detailed, per_provider);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
    total_usd: Option<f64>,
    providers: BTreeMap<String, f64>,
    warn_at_percent: u8,
    allow_unpriced: bool,
}

impl Budget {
    /// A process-wide cap of `dollars`, across every provider.
    ///
    /// ```rust
    /// use acton_ai::prelude::Budget;
    ///
    /// let budget = Budget::usd(5.00);
    /// assert_eq!(budget.to_config().unwrap().total_microusd(), Some(5_000_000));
    /// ```
    #[must_use]
    pub fn usd(dollars: f64) -> Self {
        Self {
            total_usd: Some(dollars),
            providers: BTreeMap::new(),
            warn_at_percent: DEFAULT_WARN_AT_PERCENT,
            allow_unpriced: false,
        }
    }

    /// A cap on one provider, with no process-wide ceiling.
    ///
    /// Use this — rather than a `Budget::new()` that limits nothing — when
    /// only individual providers need bounding.
    ///
    /// ```rust
    /// use acton_ai::prelude::Budget;
    ///
    /// let budget = Budget::for_provider("claude", 2.00).provider("local", 0.25);
    /// let config = budget.to_config().unwrap();
    ///
    /// assert_eq!(config.total_microusd(), None);
    /// assert_eq!(config.provider_limit("claude"), Some(2_000_000));
    /// ```
    #[must_use]
    pub fn for_provider(name: impl Into<String>, dollars: f64) -> Self {
        let mut providers = BTreeMap::new();
        providers.insert(name.into(), dollars);
        Self {
            total_usd: None,
            providers,
            warn_at_percent: DEFAULT_WARN_AT_PERCENT,
            allow_unpriced: false,
        }
    }

    /// Sets (or replaces) the process-wide cap.
    ///
    /// ```rust
    /// use acton_ai::prelude::Budget;
    ///
    /// let budget = Budget::for_provider("claude", 2.00).total_usd(5.00);
    /// assert_eq!(budget.to_config().unwrap().total_microusd(), Some(5_000_000));
    /// ```
    #[must_use]
    pub fn total_usd(mut self, dollars: f64) -> Self {
        self.total_usd = Some(dollars);
        self
    }

    /// Adds a cap on one configured provider. Repeatable; the last value for
    /// a given name wins.
    ///
    /// The name is the **configured provider name** — the key under
    /// `[providers.<name>]`, or whatever was passed to
    /// [`provider_named`](crate::facade::ActonAIBuilder::provider_named) —
    /// not a vendor or model name. A cap naming a provider that was never
    /// configured fails the launch rather than sitting there unenforced.
    ///
    /// ```rust
    /// use acton_ai::prelude::Budget;
    ///
    /// let budget = Budget::usd(5.00).provider("claude", 2.00);
    /// assert_eq!(
    ///     budget.to_config().unwrap().provider_limit("claude"),
    ///     Some(2_000_000)
    /// );
    /// ```
    #[must_use]
    pub fn provider(mut self, name: impl Into<String>, dollars: f64) -> Self {
        self.providers.insert(name.into(), dollars);
        self
    }

    /// Warns once per scope when spending crosses this percentage of a cap.
    ///
    /// Defaults to [`DEFAULT_WARN_AT_PERCENT`]. `0` disables warnings without
    /// disabling enforcement.
    ///
    /// ```rust
    /// use acton_ai::prelude::Budget;
    ///
    /// let quiet = Budget::usd(5.00).warn_at_percent(0);
    /// assert_eq!(quiet.to_config().unwrap().warn_at_percent(), 0);
    /// ```
    #[must_use]
    pub fn warn_at_percent(mut self, percent: u8) -> Self {
        self.warn_at_percent = percent;
        self
    }

    /// Counts usage that cannot be priced as `$0` instead of refusing it.
    ///
    /// Without this, a budget alongside a provider that has no configured
    /// pricing is a launch error, because such a budget could never be
    /// reached. Setting it is an explicit statement that the blind spot is
    /// acceptable — a local model with genuinely no marginal cost is the case
    /// it exists for.
    ///
    /// ```rust
    /// use acton_ai::prelude::Budget;
    ///
    /// let budget = Budget::usd(5.00).allow_unpriced();
    /// assert!(budget.to_config().unwrap().allow_unpriced());
    /// ```
    #[must_use]
    pub fn allow_unpriced(mut self) -> Self {
        self.allow_unpriced = true;
        self
    }

    /// Validates the caps and converts dollars into integer micro-USD.
    ///
    /// This is the one place budget dollars cross from floating point to
    /// integer; every comparison downstream is integer.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when a cap is negative or not finite,
    /// when the warning threshold exceeds 100, or when no cap is set at all.
    ///
    /// ```rust
    /// use acton_ai::prelude::Budget;
    ///
    /// assert!(Budget::usd(5.00).to_config().is_ok());
    /// assert!(Budget::usd(f64::NAN).to_config().is_err());
    /// assert!(Budget::usd(5.00).warn_at_percent(120).to_config().is_err());
    /// ```
    pub fn to_config(&self) -> Result<BudgetConfig, ActonAIError> {
        if self.warn_at_percent > 100 {
            return Err(ActonAIError::configuration(
                "budget.warn_at_percent",
                format!(
                    "warning threshold {} is above 100%; pass a percentage of the cap (0 disables \
                     warnings)",
                    self.warn_at_percent
                ),
            ));
        }

        let total_microusd = match self.total_usd {
            Some(dollars) => Some(validated_microusd("budget.total_usd", dollars)?),
            None => None,
        };

        let mut providers = BTreeMap::new();
        for (name, dollars) in &self.providers {
            let field = format!("budget.providers.{name}");
            providers.insert(name.clone(), validated_microusd(&field, *dollars)?);
        }

        if total_microusd.is_none() && providers.is_empty() {
            return Err(ActonAIError::configuration(
                "budget",
                "a budget with no cap limits nothing; set total_usd, or at least one \
                 per-provider cap",
            ));
        }

        Ok(BudgetConfig {
            total_microusd,
            providers,
            warn_at_percent: self.warn_at_percent,
            allow_unpriced: self.allow_unpriced,
        })
    }
}

/// Rejects a nonsensical dollar amount rather than clamping it.
///
/// [`dollars_to_microusd`] clamps negatives and NaN to `0`, which is right for
/// a price (free) and wrong for a cap: a `$0` ceiling refuses every request,
/// so a typo would silently produce a runtime that does nothing.
fn validated_microusd(field: &str, dollars: f64) -> Result<u64, ActonAIError> {
    if !dollars.is_finite() {
        return Err(ActonAIError::configuration(
            field,
            format!("{dollars} is not a finite dollar amount"),
        ));
    }
    if dollars < 0.0 {
        return Err(ActonAIError::configuration(
            field,
            format!("a cap cannot be negative (got {dollars})"),
        ));
    }
    Ok(dollars_to_microusd(dollars))
}

// =============================================================================
// BudgetConfig — the validated, integer form the actor stores
// =============================================================================

/// A validated budget in integer micro-USD: what the accountant enforces.
///
/// Produced by [`Budget::to_config`] or by the `[budget]` section of a config
/// file. Construct it through those; the fields are deliberately private so a
/// cap cannot be assembled without passing validation.
///
/// ```rust
/// use acton_ai::prelude::Budget;
///
/// let config = Budget::usd(5.00).provider("claude", 2.00).to_config().unwrap();
///
/// assert_eq!(config.total_microusd(), Some(5_000_000));
/// assert_eq!(config.provider_limit("claude"), Some(2_000_000));
/// assert_eq!(config.provider_limit("nobody"), None);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetConfig {
    total_microusd: Option<u64>,
    providers: BTreeMap<String, u64>,
    warn_at_percent: u8,
    allow_unpriced: bool,
}

impl BudgetConfig {
    /// The process-wide cap, when one is set.
    #[must_use]
    pub fn total_microusd(&self) -> Option<u64> {
        self.total_microusd
    }

    /// The cap on one configured provider, when one is set.
    #[must_use]
    pub fn provider_limit(&self, provider: &str) -> Option<u64> {
        self.providers.get(provider).copied()
    }

    /// Every provider name this budget caps, in sorted order.
    pub fn capped_providers(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    /// The warning threshold, as a percentage of each cap. `0` disables
    /// warnings.
    #[must_use]
    pub fn warn_at_percent(&self) -> u8 {
        self.warn_at_percent
    }

    /// Whether usage that cannot be priced counts as `$0` instead of refusing
    /// the next request.
    #[must_use]
    pub fn allow_unpriced(&self) -> bool {
        self.allow_unpriced
    }
}

// =============================================================================
// Spend — priced tallies, split by scope
// =============================================================================

/// What has been spent so far, priced and split by scope.
///
/// Internal: the accountant computes one of these from its tallies and the
/// pricing table, and every decision below is a pure function of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Spend {
    /// Priced spend across every provider that could be priced.
    pub(crate) total_microusd: u64,
    /// Priced spend per configured provider name.
    pub(crate) by_provider: BTreeMap<String, u64>,
    /// Providers that recorded usage no pricing could be found for, sorted.
    pub(crate) unpriced: Vec<String>,
}

impl Spend {
    /// Spend attributed to one provider, `0` when it has recorded nothing.
    fn for_provider(&self, provider: &str) -> u64 {
        self.by_provider.get(provider).copied().unwrap_or(0)
    }
}

// =============================================================================
// Decisions
// =============================================================================

/// Asks the accountant whether one more request to `provider` may go out.
///
/// Sent by the prompt loop before every provider dispatch when a budget is
/// configured. Low-level callers that send
/// [`LLMRequest`](crate::messages::LLMRequest) straight to a provider actor
/// bypass the loop, and therefore this check; they can ask the accountant
/// themselves.
///
/// ```rust,ignore
/// let decision = accountant.ask(CheckBudget::for_provider("claude")).await?;
/// ```
#[acton_message]
pub struct CheckBudget {
    /// The configured provider name the next request would go to.
    pub provider: String,
}

impl CheckBudget {
    /// Asks about one configured provider by name.
    #[must_use]
    pub fn for_provider(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
        }
    }
}

impl Request for CheckBudget {
    type Response = BudgetDecision;
}

/// The accountant's answer to [`CheckBudget`].
#[acton_message]
#[derive(PartialEq, Eq)]
pub enum BudgetDecision {
    /// Nothing is over its cap; the request may go out.
    Allowed,
    /// A cap has been reached. `spent_microusd` is the spend in the violated
    /// scope, not the grand total.
    Denied {
        /// Which ceiling was hit.
        scope: BudgetScope,
        /// The ceiling, in micro-USD.
        limit_microusd: u64,
        /// Spend in that scope, in micro-USD.
        spent_microusd: u64,
    },
    /// Usage exists that cannot be priced, so no cap can be trusted.
    ///
    /// Fail-closed belt and braces for reports from providers absent from
    /// config; the launch-time check catches the ordinary case. Silenced by
    /// [`Budget::allow_unpriced`].
    Unpriced {
        /// The providers whose usage could not be priced, sorted.
        providers: Vec<String>,
    },
}

/// Decides whether one more request to `provider` may go out.
///
/// Pure: every input is an argument. Precedence is most-specific-first — when
/// both a provider cap and the total cap are breached, the provider one is
/// reported, because that is the knob that has to change.
pub(crate) fn decide(config: &BudgetConfig, spend: &Spend, provider: &str) -> BudgetDecision {
    if !config.allow_unpriced && !spend.unpriced.is_empty() {
        return BudgetDecision::Unpriced {
            providers: spend.unpriced.clone(),
        };
    }

    if let Some(limit) = config.provider_limit(provider) {
        let spent = spend.for_provider(provider);
        if spent >= limit {
            return BudgetDecision::Denied {
                scope: BudgetScope::Provider(provider.to_string()),
                limit_microusd: limit,
                spent_microusd: spent,
            };
        }
    }

    if let Some(limit) = config.total_microusd {
        if spend.total_microusd >= limit {
            return BudgetDecision::Denied {
                scope: BudgetScope::Total,
                limit_microusd: limit,
                spent_microusd: spend.total_microusd,
            };
        }
    }

    BudgetDecision::Allowed
}

// =============================================================================
// Events
// =============================================================================

/// Broadcast when spending crosses a threshold on the way up.
///
/// Published on the broker, so anything in the runtime can subscribe;
/// [`on_budget_event`](crate::facade::ActonAIBuilder::on_budget_event) is the
/// convenience wrapper.
///
/// Each variant fires **once per scope**: crossings are computed from the
/// spend before and after each report is folded in, and spend only ever
/// grows, so a threshold can be crossed exactly once.
///
/// ```rust
/// use acton_ai::prelude::{BudgetEvent, BudgetScope};
///
/// let event = BudgetEvent::Exceeded {
///     scope: BudgetScope::Total,
///     limit_microusd: 5_000_000,
///     spent_microusd: 5_400_000,
/// };
///
/// assert!(event.to_string().contains("total"));
/// assert_eq!(event.scope(), &BudgetScope::Total);
/// ```
#[acton_message]
#[derive(PartialEq, Eq)]
#[non_exhaustive]
pub enum BudgetEvent {
    /// Spending crossed the configured warning threshold for a scope.
    ThresholdCrossed {
        /// Which ceiling the warning is about.
        scope: BudgetScope,
        /// Percentage of the cap now used, floored. Can exceed 100 when a
        /// single report jumps past both the warning and the cap.
        percent_used: u32,
        /// The ceiling, in micro-USD.
        limit_microusd: u64,
        /// Spend in that scope, in micro-USD.
        spent_microusd: u64,
    },
    /// Spending reached or passed a cap. Further requests in that scope are
    /// refused.
    Exceeded {
        /// Which ceiling was reached.
        scope: BudgetScope,
        /// The ceiling, in micro-USD.
        limit_microusd: u64,
        /// Spend in that scope, in micro-USD.
        spent_microusd: u64,
    },
}

impl BudgetEvent {
    /// Which ceiling this event concerns.
    #[must_use]
    pub fn scope(&self) -> &BudgetScope {
        match self {
            Self::ThresholdCrossed { scope, .. } | Self::Exceeded { scope, .. } => scope,
        }
    }
}

impl fmt::Display for BudgetEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThresholdCrossed {
                scope,
                percent_used,
                limit_microusd,
                spent_microusd,
            } => write!(
                f,
                "{scope} budget at {percent_used}%: ${:.4} of ${:.4}",
                microusd_to_usd(*spent_microusd),
                microusd_to_usd(*limit_microusd)
            ),
            Self::Exceeded {
                scope,
                limit_microusd,
                spent_microusd,
            } => write!(
                f,
                "{scope} budget exceeded: ${:.4} of ${:.4}",
                microusd_to_usd(*spent_microusd),
                microusd_to_usd(*limit_microusd)
            ),
        }
    }
}

/// Percentage of `limit` that `spent` represents, floored.
///
/// A zero cap has no meaningful ratio: it reads as `0` while nothing has been
/// spent and `100` the moment anything has, which is what makes a `$0` budget
/// emit its crossing exactly once like any other.
fn percent_used(spent: u64, limit: u64) -> u32 {
    if limit == 0 {
        return u32::from(spent > 0) * 100;
    }
    let percent = u128::from(spent) * 100 / u128::from(limit);
    u32::try_from(percent).unwrap_or(u32::MAX)
}

/// Detects the thresholds crossed between two spend states.
///
/// Pure, and free of any "already fired" bookkeeping: spend is monotonic, so
/// asking "was it below and is it now at or above" answers "is this the
/// crossing" exactly once per scope, however many reports arrive afterwards.
pub(crate) fn crossings(config: &BudgetConfig, before: &Spend, after: &Spend) -> Vec<BudgetEvent> {
    let mut events = Vec::new();

    if let Some(limit) = config.total_microusd {
        scope_crossings(
            &BudgetScope::Total,
            limit,
            before.total_microusd,
            after.total_microusd,
            config.warn_at_percent,
            &mut events,
        );
    }

    for (name, limit) in &config.providers {
        scope_crossings(
            &BudgetScope::Provider(name.clone()),
            *limit,
            before.for_provider(name),
            after.for_provider(name),
            config.warn_at_percent,
            &mut events,
        );
    }

    events
}

/// Appends the events one scope crossed between `before` and `after`.
fn scope_crossings(
    scope: &BudgetScope,
    limit: u64,
    before: u64,
    after: u64,
    warn_at_percent: u8,
    events: &mut Vec<BudgetEvent>,
) {
    let before_percent = percent_used(before, limit);
    let after_percent = percent_used(after, limit);

    let warn = u32::from(warn_at_percent);
    if warn > 0 && before_percent < warn && after_percent >= warn {
        events.push(BudgetEvent::ThresholdCrossed {
            scope: scope.clone(),
            percent_used: after_percent,
            limit_microusd: limit,
            spent_microusd: after,
        });
    }

    if before_percent < 100 && after_percent >= 100 {
        events.push(BudgetEvent::Exceeded {
            scope: scope.clone(),
            limit_microusd: limit,
            spent_microusd: after,
        });
    }
}

// =============================================================================
// Status
// =============================================================================

/// How much of one cap has been used.
///
/// ```rust
/// use acton_ai::prelude::ScopeStatus;
///
/// let status = ScopeStatus {
///     limit_microusd: 5_000_000,
///     spent_microusd: 1_250_000,
/// };
///
/// assert_eq!(status.percent_used(), 25);
/// assert_eq!(status.remaining_microusd(), 3_750_000);
/// assert!((status.remaining_usd() - 3.75).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeStatus {
    /// The ceiling, in micro-USD.
    pub limit_microusd: u64,
    /// Spend against it, in micro-USD.
    pub spent_microusd: u64,
}

impl ScopeStatus {
    /// Headroom left, saturating at zero once the cap is passed.
    #[must_use]
    pub fn remaining_microusd(&self) -> u64 {
        self.limit_microusd.saturating_sub(self.spent_microusd)
    }

    /// Percentage of the cap used, floored. Can exceed 100.
    #[must_use]
    pub fn percent_used(&self) -> u32 {
        percent_used(self.spent_microusd, self.limit_microusd)
    }

    /// Whether the cap has been reached, and further requests refused.
    #[must_use]
    pub fn is_exceeded(&self) -> bool {
        self.spent_microusd >= self.limit_microusd
    }

    /// The ceiling in dollars, for display.
    #[must_use]
    pub fn limit_usd(&self) -> f64 {
        microusd_to_usd(self.limit_microusd)
    }

    /// Spend in dollars, for display.
    #[must_use]
    pub fn spent_usd(&self) -> f64 {
        microusd_to_usd(self.spent_microusd)
    }

    /// Headroom in dollars, for display.
    #[must_use]
    pub fn remaining_usd(&self) -> f64 {
        microusd_to_usd(self.remaining_microusd())
    }
}

/// Budget standing at snapshot time, carried on
/// [`UsageSnapshot::budget`](crate::accounting::UsageSnapshot::budget).
///
/// `None` on the snapshot means no budget was configured — never "a budget
/// with nothing spent".
///
/// ```rust,ignore
/// let usage = ai.usage().await?;
/// if let Some(budget) = &usage.budget {
///     println!("${:.2} left of the cap", budget.remaining_usd().unwrap_or(0.0));
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetStatus {
    /// The process-wide cap and its usage, when one is configured.
    pub total: Option<ScopeStatus>,
    /// Per-provider caps and their usage, keyed by configured provider name.
    pub providers: BTreeMap<String, ScopeStatus>,
    /// The warning threshold in force, as a percentage. `0` means warnings
    /// are disabled.
    pub warn_at_percent: u8,
    /// Whether unpriced usage is being counted as `$0`.
    pub allow_unpriced: bool,
}

impl BudgetStatus {
    /// One provider's cap and its usage.
    #[must_use]
    pub fn provider(&self, name: &str) -> Option<&ScopeStatus> {
        self.providers.get(name)
    }

    /// Headroom under the process-wide cap, in dollars.
    ///
    /// `None` when the budget sets only per-provider caps.
    #[must_use]
    pub fn remaining_usd(&self) -> Option<f64> {
        self.total.map(|total| total.remaining_usd())
    }

    /// Percentage of the process-wide cap used, floored.
    ///
    /// `None` when the budget sets only per-provider caps.
    #[must_use]
    pub fn percent_used(&self) -> Option<u32> {
        self.total.map(|total| total.percent_used())
    }

    /// Whether any cap — total or per-provider — has been reached.
    #[must_use]
    pub fn is_exceeded(&self) -> bool {
        self.total.is_some_and(|total| total.is_exceeded())
            || self.providers.values().any(ScopeStatus::is_exceeded)
    }
}

/// Renders the current standing of every configured cap.
pub(crate) fn status(config: &BudgetConfig, spend: &Spend) -> BudgetStatus {
    BudgetStatus {
        total: config.total_microusd.map(|limit| ScopeStatus {
            limit_microusd: limit,
            spent_microusd: spend.total_microusd,
        }),
        providers: config
            .providers
            .iter()
            .map(|(name, limit)| {
                (
                    name.clone(),
                    ScopeStatus {
                        limit_microusd: *limit,
                        spent_microusd: spend.for_provider(name),
                    },
                )
            })
            .collect(),
        warn_at_percent: config.warn_at_percent,
        allow_unpriced: config.allow_unpriced,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(
        total: Option<u64>,
        providers: &[(&str, u64)],
        warn: u8,
        allow: bool,
    ) -> BudgetConfig {
        BudgetConfig {
            total_microusd: total,
            providers: providers
                .iter()
                .map(|(name, limit)| ((*name).to_string(), *limit))
                .collect(),
            warn_at_percent: warn,
            allow_unpriced: allow,
        }
    }

    fn spend(total: u64, providers: &[(&str, u64)], unpriced: &[&str]) -> Spend {
        Spend {
            total_microusd: total,
            by_provider: providers
                .iter()
                .map(|(name, spent)| ((*name).to_string(), *spent))
                .collect(),
            unpriced: unpriced.iter().map(|name| (*name).to_string()).collect(),
        }
    }

    // -------------------------------------------------------------------
    // Builder → config validation
    // -------------------------------------------------------------------

    #[test]
    fn a_total_cap_converts_dollars_once_into_microusd() {
        let config = Budget::usd(5.55).to_config().expect("a valid budget");

        assert_eq!(config.total_microusd(), Some(5_550_000));
        assert_eq!(config.warn_at_percent(), DEFAULT_WARN_AT_PERCENT);
        assert!(!config.allow_unpriced());
    }

    #[test]
    fn per_provider_caps_survive_the_conversion() {
        let config = Budget::for_provider("claude", 2.0)
            .provider("local", 0.5)
            .to_config()
            .expect("a valid budget");

        assert_eq!(config.total_microusd(), None);
        assert_eq!(config.provider_limit("claude"), Some(2_000_000));
        assert_eq!(config.provider_limit("local"), Some(500_000));
        assert_eq!(
            config.capped_providers().collect::<Vec<_>>(),
            vec!["claude", "local"]
        );
    }

    #[test]
    fn a_warning_threshold_above_one_hundred_is_rejected() {
        let err = Budget::usd(5.0)
            .warn_at_percent(120)
            .to_config()
            .expect_err("120% of a cap is not a threshold");

        assert!(err.is_configuration());
        assert!(err.to_string().contains("warn_at_percent"), "err = {err}");
    }

    #[test]
    fn a_negative_cap_is_rejected_rather_than_clamped_to_zero() {
        // Clamping would produce a $0 cap, which refuses every request — a
        // typo would turn into a runtime that silently does nothing.
        let err = Budget::usd(-1.0)
            .to_config()
            .expect_err("a negative cap is nonsense");

        assert!(err.to_string().contains("negative"), "err = {err}");
    }

    #[test]
    fn a_non_finite_cap_is_rejected() {
        assert!(Budget::usd(f64::NAN).to_config().is_err());
        assert!(Budget::usd(f64::INFINITY).to_config().is_err());
        assert!(Budget::for_provider("claude", f64::NAN)
            .to_config()
            .is_err());
    }

    #[test]
    fn a_provider_cap_names_the_offending_provider_when_it_is_invalid() {
        let err = Budget::for_provider("claude", -2.0)
            .to_config()
            .expect_err("a negative cap is nonsense");

        assert!(err.to_string().contains("claude"), "err = {err}");
    }

    #[test]
    fn a_zero_cap_is_a_valid_refuse_everything_budget() {
        let config = Budget::usd(0.0).to_config().expect("zero is a real cap");

        assert_eq!(config.total_microusd(), Some(0));
        assert!(matches!(
            decide(&config, &Spend::default(), "claude"),
            BudgetDecision::Denied { .. }
        ));
    }

    // -------------------------------------------------------------------
    // decide()
    // -------------------------------------------------------------------

    #[test]
    fn spending_below_every_cap_is_allowed() {
        let config = config(Some(5_000_000), &[("claude", 2_000_000)], 80, false);
        let spend = spend(1_000_000, &[("claude", 1_000_000)], &[]);

        assert_eq!(decide(&config, &spend, "claude"), BudgetDecision::Allowed);
    }

    #[test]
    fn reaching_the_total_cap_denies_the_next_request() {
        let config = config(Some(5_000_000), &[], 80, false);
        let spend = spend(5_000_000, &[("claude", 5_000_000)], &[]);

        assert_eq!(
            decide(&config, &spend, "claude"),
            BudgetDecision::Denied {
                scope: BudgetScope::Total,
                limit_microusd: 5_000_000,
                spent_microusd: 5_000_000,
            },
            "a cap is reached at equality, not one micro-dollar past it"
        );
    }

    #[test]
    fn a_provider_cap_only_denies_that_provider() {
        let config = config(
            Some(50_000_000),
            &[("claude", 2_000_000), ("local", 9_000_000)],
            80,
            false,
        );
        let spend = spend(3_000_000, &[("claude", 2_500_000), ("local", 500_000)], &[]);

        assert!(matches!(
            decide(&config, &spend, "claude"),
            BudgetDecision::Denied {
                scope: BudgetScope::Provider(ref name),
                ..
            } if name == "claude"
        ));
        assert_eq!(decide(&config, &spend, "local"), BudgetDecision::Allowed);
    }

    #[test]
    fn when_both_caps_are_breached_the_provider_one_is_reported() {
        // The provider cap is the knob that has to change, so naming the
        // total would send the operator to the wrong setting.
        let config = config(Some(1_000_000), &[("claude", 500_000)], 80, false);
        let spend = spend(2_000_000, &[("claude", 2_000_000)], &[]);

        assert!(matches!(
            decide(&config, &spend, "claude"),
            BudgetDecision::Denied {
                scope: BudgetScope::Provider(_),
                limit_microusd: 500_000,
                ..
            }
        ));
    }

    #[test]
    fn an_uncapped_provider_still_answers_to_the_total_cap() {
        let config = config(Some(1_000_000), &[("claude", 9_000_000)], 80, false);
        let spend = spend(1_500_000, &[("local", 1_500_000)], &[]);

        assert!(matches!(
            decide(&config, &spend, "local"),
            BudgetDecision::Denied {
                scope: BudgetScope::Total,
                ..
            }
        ));
    }

    #[test]
    fn unpriced_usage_denies_everything_by_default() {
        let config = config(Some(5_000_000), &[], 80, false);
        let spend = spend(0, &[], &["local"]);

        assert_eq!(
            decide(&config, &spend, "claude"),
            BudgetDecision::Unpriced {
                providers: vec!["local".to_string()],
            },
            "a cap that cannot see part of the spend is not a cap"
        );
    }

    #[test]
    fn allow_unpriced_counts_unpriced_usage_as_zero() {
        let config = config(Some(5_000_000), &[], 80, true);
        let spend = spend(1_000_000, &[("claude", 1_000_000)], &["local"]);

        assert_eq!(decide(&config, &spend, "claude"), BudgetDecision::Allowed);
    }

    // -------------------------------------------------------------------
    // crossings()
    // -------------------------------------------------------------------

    #[test]
    fn crossing_the_warning_threshold_fires_once() {
        let config = config(Some(1_000_000), &[], 80, false);

        let first = crossings(
            &config,
            &spend(700_000, &[], &[]),
            &spend(850_000, &[], &[]),
        );
        assert_eq!(first.len(), 1);
        assert!(matches!(
            first[0],
            BudgetEvent::ThresholdCrossed {
                scope: BudgetScope::Total,
                percent_used: 85,
                ..
            }
        ));

        // A later report, still under the cap, is not a second crossing.
        let second = crossings(
            &config,
            &spend(850_000, &[], &[]),
            &spend(900_000, &[], &[]),
        );
        assert!(second.is_empty(), "events = {second:?}");
    }

    #[test]
    fn a_zero_threshold_disables_warnings_without_disabling_enforcement() {
        let config = config(Some(1_000_000), &[], 0, false);

        let events = crossings(&config, &spend(0, &[], &[]), &spend(900_000, &[], &[]));
        assert!(events.is_empty(), "events = {events:?}");

        let exceeded = crossings(
            &config,
            &spend(900_000, &[], &[]),
            &spend(1_000_000, &[], &[]),
        );
        assert_eq!(exceeded.len(), 1, "the cap itself still reports");
        assert!(matches!(exceeded[0], BudgetEvent::Exceeded { .. }));
    }

    #[test]
    fn one_report_can_cross_the_warning_and_the_cap_together() {
        let config = config(Some(1_000_000), &[], 80, false);

        let events = crossings(&config, &spend(0, &[], &[]), &spend(1_500_000, &[], &[]));

        assert_eq!(events.len(), 2, "events = {events:?}");
        assert!(matches!(
            events[0],
            BudgetEvent::ThresholdCrossed {
                percent_used: 150,
                ..
            }
        ));
        assert!(matches!(events[1], BudgetEvent::Exceeded { .. }));
    }

    #[test]
    fn exceeding_a_cap_fires_once_however_far_past_it_goes() {
        let config = config(Some(1_000_000), &[], 80, false);

        let past = crossings(
            &config,
            &spend(1_200_000, &[], &[]),
            &spend(9_000_000, &[], &[]),
        );

        assert!(past.is_empty(), "events = {past:?}");
    }

    #[test]
    fn provider_scopes_cross_independently_of_the_total() {
        let config = config(Some(10_000_000), &[("claude", 1_000_000)], 80, false);

        let events = crossings(
            &config,
            &spend(0, &[], &[]),
            &spend(900_000, &[("claude", 900_000)], &[]),
        );

        assert_eq!(events.len(), 1, "events = {events:?}");
        assert_eq!(
            events[0].scope(),
            &BudgetScope::Provider("claude".to_string())
        );
    }

    // -------------------------------------------------------------------
    // percent + status
    // -------------------------------------------------------------------

    #[test]
    fn percentages_floor_rather_than_round() {
        assert_eq!(percent_used(899_999, 1_000_000), 89);
        assert_eq!(percent_used(1, 1_000_000), 0);
    }

    #[test]
    fn a_zero_cap_reads_as_fully_used_the_moment_anything_is_spent() {
        assert_eq!(percent_used(0, 0), 0);
        assert_eq!(percent_used(1, 0), 100);
    }

    #[test]
    fn status_reports_remaining_headroom_and_saturates_past_the_cap() {
        let config = config(Some(1_000_000), &[("claude", 400_000)], 80, false);
        let status = status(&config, &spend(1_250_000, &[("claude", 100_000)], &[]));

        let total = status.total.expect("a total cap was configured");
        assert_eq!(total.percent_used(), 125);
        assert_eq!(
            total.remaining_microusd(),
            0,
            "headroom saturates rather than wrapping"
        );
        assert!(total.is_exceeded());

        let claude = status.provider("claude").expect("a provider cap");
        assert_eq!(claude.remaining_microusd(), 300_000);
        assert!(!claude.is_exceeded());
        assert!(status.is_exceeded(), "the total scope is over");
    }

    #[test]
    fn status_without_a_total_cap_reports_no_process_wide_remainder() {
        let config = config(None, &[("claude", 400_000)], 80, false);
        let status = status(&config, &spend(100_000, &[("claude", 100_000)], &[]));

        assert!(status.total.is_none());
        assert!(status.remaining_usd().is_none());
        assert!(status.percent_used().is_none());
    }

    #[test]
    fn scope_display_names_the_knob_to_change() {
        assert_eq!(BudgetScope::Total.to_string(), "total");
        assert_eq!(
            BudgetScope::Provider("claude".to_string()).to_string(),
            "provider `claude`"
        );
    }
}
