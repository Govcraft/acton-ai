//! Pricing tables and cost arithmetic.
//!
//! # Money is an integer
//!
//! Every computed figure in this module is an integer count of **micro-USD**
//! (millionths of a dollar). Floating point appears exactly twice, both at the
//! edges: once when a TOML file is read — pricing pages print
//! dollars-per-million-tokens, so that is what the config accepts — and once
//! in [`CostBreakdown::total_usd`], for display. Everything between is `u64`
//! and `u128`, so no accumulated rounding drift is possible.
//!
//! # There is no bundled price list
//!
//! Shipping hardcoded vendor prices would mean silently wrong costs the day a
//! vendor changes them, with nothing to signal the staleness. A model with no
//! configured pricing therefore yields `None` — never a fabricated `$0.00` —
//! while its tokens are still counted.

use crate::messages::Usage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Micro-USD in one US dollar.
const MICRO_USD_PER_USD: u64 = 1_000_000;

/// Tokens in one "MTok", the unit vendors quote prices in.
const TOKENS_PER_MTOK: u128 = 1_000_000;

/// Converts a vendor-quoted price in dollars per million tokens into the
/// integer micro-USD-per-MTok this module works in.
///
/// This is the *only* place a price crosses from floating point to integer.
/// The result is **rounded to the nearest micro-USD**, not truncated: `3.3`
/// dollars is `3_300_000` micro-USD, where truncating the inexact
/// `3.2999999999999998` binary representation would lose a micro-dollar on
/// every MTok forever.
///
/// Negative or non-finite input clamps to `0` — a nonsensical price should
/// read as "free", never wrap into an enormous positive rate.
#[must_use]
pub fn dollars_per_mtok_to_microusd(dollars: f64) -> u64 {
    dollars_to_microusd(dollars)
}

/// Converts a dollar amount into integer micro-USD.
///
/// The same conversion [`dollars_per_mtok_to_microusd`] performs — the two
/// differ only in what the number *means* (a rate there, an amount here), and
/// sharing one implementation is what keeps a budget cap and the prices it is
/// compared against rounding identically.
///
/// **Rounded to the nearest micro-USD**, not truncated: `5.55` dollars is
/// `5_550_000` micro-USD, where truncating the inexact binary representation
/// would lose a micro-dollar. Negative or non-finite input clamps to `0` —
/// callers that must reject nonsense (budgets do) validate before converting.
///
/// ```rust
/// use acton_ai::prelude::dollars_to_microusd;
///
/// assert_eq!(dollars_to_microusd(5.0), 5_000_000);
/// assert_eq!(dollars_to_microusd(0.25), 250_000);
/// assert_eq!(dollars_to_microusd(-1.0), 0);
/// ```
#[must_use]
pub fn dollars_to_microusd(dollars: f64) -> u64 {
    if !dollars.is_finite() || dollars <= 0.0 {
        return 0;
    }
    // Rounded, then clamped: `as u64` on an out-of-range float saturates in
    // Rust, but being explicit documents the intent.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let micro = (dollars * MICRO_USD_PER_USD as f64).round() as u128;
    u64::try_from(micro).unwrap_or(u64::MAX)
}

/// Renders integer micro-USD as dollars, for display only.
///
/// Lossy by construction, exactly as [`CostBreakdown::total_usd`] is: render
/// with it, never accumulate with it.
#[must_use]
pub(crate) fn microusd_to_usd(microusd: u64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        microusd as f64 / MICRO_USD_PER_USD as f64
    }
}

/// What one model costs, in integer micro-USD per million tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Price per MTok of uncached input.
    pub input_per_mtok_microusd: u64,
    /// Price per MTok of generated output.
    pub output_per_mtok_microusd: u64,
    /// Price per MTok of input served from cache. Usually far below the input
    /// rate; `0` when the vendor does not charge for reads.
    pub cache_read_per_mtok_microusd: u64,
    /// Price per MTok of input written to cache. Usually above the input rate.
    pub cache_creation_per_mtok_microusd: u64,
}

impl ModelPricing {
    /// Builds a pricing entry from vendor-quoted dollars per million tokens.
    ///
    /// Cache rates default to `0`, which prices cache traffic as free rather
    /// than guessing a multiple of the input rate.
    #[must_use]
    pub fn from_dollars_per_mtok(input: f64, output: f64) -> Self {
        Self {
            input_per_mtok_microusd: dollars_per_mtok_to_microusd(input),
            output_per_mtok_microusd: dollars_per_mtok_to_microusd(output),
            cache_read_per_mtok_microusd: 0,
            cache_creation_per_mtok_microusd: 0,
        }
    }

    /// Sets the cache-read rate, in dollars per million tokens.
    #[must_use]
    pub fn with_cache_read_per_mtok(mut self, dollars: f64) -> Self {
        self.cache_read_per_mtok_microusd = dollars_per_mtok_to_microusd(dollars);
        self
    }

    /// Sets the cache-write rate, in dollars per million tokens.
    #[must_use]
    pub fn with_cache_creation_per_mtok(mut self, dollars: f64) -> Self {
        self.cache_creation_per_mtok_microusd = dollars_per_mtok_to_microusd(dollars);
        self
    }

    /// Prices one bucket of tokens at one rate.
    ///
    /// **Rounding is floor.** `tokens × rate ÷ 1_000_000` is computed in
    /// `u128` (so a large token count times a large rate cannot overflow
    /// mid-way) and truncated. Under-charging by at most one micro-dollar per
    /// bucket is the right direction for a figure a user might act on, and it
    /// makes the arithmetic exactly reproducible.
    #[must_use]
    fn price(tokens: u64, rate_per_mtok_microusd: u64) -> u64 {
        let micro = u128::from(tokens) * u128::from(rate_per_mtok_microusd) / TOKENS_PER_MTOK;
        u64::try_from(micro).unwrap_or(u64::MAX)
    }

    /// Prices a usage record against this model's rates.
    #[must_use]
    pub fn cost_of(&self, usage: &Usage) -> CostBreakdown {
        let input_microusd = Self::price(usage.input_tokens, self.input_per_mtok_microusd);
        let output_microusd = Self::price(usage.output_tokens, self.output_per_mtok_microusd);
        let cache_read_microusd =
            Self::price(usage.cache_read_tokens, self.cache_read_per_mtok_microusd);
        let cache_creation_microusd = Self::price(
            usage.cache_creation_tokens,
            self.cache_creation_per_mtok_microusd,
        );

        CostBreakdown {
            input_microusd,
            output_microusd,
            cache_read_microusd,
            cache_creation_microusd,
            total_microusd: input_microusd
                .saturating_add(output_microusd)
                .saturating_add(cache_read_microusd)
                .saturating_add(cache_creation_microusd),
        }
    }
}

/// A priced usage record, in integer micro-USD.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Cost of uncached input tokens.
    pub input_microusd: u64,
    /// Cost of generated output tokens.
    pub output_microusd: u64,
    /// Cost of cache-read input tokens.
    pub cache_read_microusd: u64,
    /// Cost of cache-written input tokens.
    pub cache_creation_microusd: u64,
    /// Sum of the four components above.
    pub total_microusd: u64,
}

impl CostBreakdown {
    /// The total as dollars, for display.
    ///
    /// Lossy by construction — `f64` cannot represent every micro-dollar
    /// count exactly. Render with it; never accumulate with it. Keep summing
    /// [`Self::total_microusd`] instead.
    #[must_use]
    pub fn total_usd(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        {
            self.total_microusd as f64 / MICRO_USD_PER_USD as f64
        }
    }
}

impl core::ops::Add for CostBreakdown {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            input_microusd: self.input_microusd.saturating_add(rhs.input_microusd),
            output_microusd: self.output_microusd.saturating_add(rhs.output_microusd),
            cache_read_microusd: self
                .cache_read_microusd
                .saturating_add(rhs.cache_read_microusd),
            cache_creation_microusd: self
                .cache_creation_microusd
                .saturating_add(rhs.cache_creation_microusd),
            total_microusd: self.total_microusd.saturating_add(rhs.total_microusd),
        }
    }
}

impl core::ops::AddAssign for CostBreakdown {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Pricing for every configured provider.
///
/// Keyed by **provider name**, not by model: a provider entry names exactly
/// one model, so the provider name is already the finest key that exists. A
/// provider absent from this table prices to `None`.
#[derive(Debug, Clone, Default)]
pub struct PricingTable {
    by_provider: HashMap<String, ModelPricing>,
}

impl PricingTable {
    /// An empty table. Everything prices to `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records pricing for one configured provider.
    pub fn insert(&mut self, provider: impl Into<String>, pricing: ModelPricing) {
        self.by_provider.insert(provider.into(), pricing);
    }

    /// Returns true when no provider has pricing configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_provider.is_empty()
    }

    /// The rates configured for `provider`, if any.
    #[must_use]
    pub fn get(&self, provider: &str) -> Option<&ModelPricing> {
        self.by_provider.get(provider)
    }

    /// Prices `usage` against `provider`'s rates.
    ///
    /// `None` means no pricing is configured for that provider — a caller
    /// must render that as "unknown", never as zero.
    #[must_use]
    pub fn cost_for(&self, provider: &str, usage: &Usage) -> Option<CostBreakdown> {
        self.by_provider
            .get(provider)
            .map(|pricing| pricing.cost_of(usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u64, output: u64, read: u64, creation: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: read,
            cache_creation_tokens: creation,
        }
    }

    // -------------------------------------------------------------------
    // Dollars → micro-USD conversion
    // -------------------------------------------------------------------

    #[test]
    fn whole_dollars_convert_exactly() {
        assert_eq!(dollars_per_mtok_to_microusd(3.0), 3_000_000);
        assert_eq!(dollars_per_mtok_to_microusd(15.0), 15_000_000);
    }

    #[test]
    fn fractional_dollars_round_to_the_nearest_microusd() {
        // 3.3 is not representable in binary; truncation would yield 3_299_999.
        assert_eq!(dollars_per_mtok_to_microusd(3.3), 3_300_000);
        assert_eq!(dollars_per_mtok_to_microusd(0.25), 250_000);
        assert_eq!(dollars_per_mtok_to_microusd(0.0000005), 1);
    }

    #[test]
    fn sub_microdollar_prices_round_to_zero_or_one() {
        assert_eq!(dollars_per_mtok_to_microusd(0.0000004), 0);
        assert_eq!(dollars_per_mtok_to_microusd(0.0000006), 1);
    }

    #[test]
    fn nonsensical_prices_clamp_to_zero_rather_than_wrapping() {
        assert_eq!(dollars_per_mtok_to_microusd(-5.0), 0);
        assert_eq!(dollars_per_mtok_to_microusd(f64::NAN), 0);
        assert_eq!(dollars_per_mtok_to_microusd(f64::INFINITY), 0);
        assert_eq!(dollars_per_mtok_to_microusd(0.0), 0);
    }

    // -------------------------------------------------------------------
    // Cost arithmetic
    // -------------------------------------------------------------------

    #[test]
    fn one_mtok_costs_exactly_the_quoted_rate() {
        let pricing = ModelPricing::from_dollars_per_mtok(3.0, 15.0);

        let cost = pricing.cost_of(&usage(1_000_000, 1_000_000, 0, 0));

        assert_eq!(cost.input_microusd, 3_000_000);
        assert_eq!(cost.output_microusd, 15_000_000);
        assert_eq!(cost.total_microusd, 18_000_000);
    }

    #[test]
    fn cost_scales_linearly_below_one_mtok() {
        let pricing = ModelPricing::from_dollars_per_mtok(3.0, 15.0);

        // 1_000 input @ $3/MTok = 3_000 micro-USD; 500 output @ $15/MTok = 7_500.
        let cost = pricing.cost_of(&usage(1_000, 500, 0, 0));

        assert_eq!(cost.input_microusd, 3_000);
        assert_eq!(cost.output_microusd, 7_500);
        assert_eq!(cost.total_microusd, 10_500);
    }

    #[test]
    fn cost_arithmetic_floors_the_remainder() {
        // 1 token @ $3/MTok is 3 micro-USD; 1 token @ $0.4/MTok is 0.4
        // micro-USD, which floors to 0 rather than rounding up to 1.
        let pricing = ModelPricing {
            input_per_mtok_microusd: 3_000_000,
            output_per_mtok_microusd: 400_000,
            ..ModelPricing::default()
        };

        let cost = pricing.cost_of(&usage(1, 1, 0, 0));

        assert_eq!(cost.input_microusd, 3);
        assert_eq!(
            cost.output_microusd, 0,
            "the remainder floors, never rounds up"
        );
        assert_eq!(cost.total_microusd, 3);
    }

    #[test]
    fn cache_rates_are_priced_separately_from_input() {
        let pricing = ModelPricing::from_dollars_per_mtok(3.0, 15.0)
            .with_cache_read_per_mtok(0.3)
            .with_cache_creation_per_mtok(3.75);

        let cost = pricing.cost_of(&usage(1_000_000, 0, 1_000_000, 1_000_000));

        assert_eq!(cost.input_microusd, 3_000_000);
        assert_eq!(cost.cache_read_microusd, 300_000);
        assert_eq!(cost.cache_creation_microusd, 3_750_000);
        assert_eq!(cost.total_microusd, 7_050_000);
    }

    #[test]
    fn unconfigured_cache_rates_price_cache_traffic_as_free() {
        let pricing = ModelPricing::from_dollars_per_mtok(3.0, 15.0);

        let cost = pricing.cost_of(&usage(0, 0, 5_000_000, 5_000_000));

        assert_eq!(cost.total_microusd, 0);
    }

    #[test]
    fn huge_token_counts_do_not_overflow() {
        let pricing = ModelPricing {
            input_per_mtok_microusd: u64::MAX,
            ..ModelPricing::default()
        };

        // Computed in u128 and saturated, not wrapped.
        let cost = pricing.cost_of(&usage(u64::MAX, 0, 0, 0));

        assert_eq!(cost.input_microusd, u64::MAX);
    }

    #[test]
    fn total_usd_renders_micro_usd_as_dollars() {
        let cost = CostBreakdown {
            total_microusd: 1_234_500,
            ..CostBreakdown::default()
        };

        assert!((cost.total_usd() - 1.2345).abs() < f64::EPSILON);
    }

    #[test]
    fn cost_breakdowns_add_component_wise() {
        let a = CostBreakdown {
            input_microusd: 1,
            output_microusd: 2,
            cache_read_microusd: 3,
            cache_creation_microusd: 4,
            total_microusd: 10,
        };

        let sum = a + a;

        assert_eq!(sum.input_microusd, 2);
        assert_eq!(sum.output_microusd, 4);
        assert_eq!(sum.cache_read_microusd, 6);
        assert_eq!(sum.cache_creation_microusd, 8);
        assert_eq!(sum.total_microusd, 20);
    }

    // -------------------------------------------------------------------
    // Table lookup
    // -------------------------------------------------------------------

    #[test]
    fn an_unpriced_provider_yields_none_not_zero() {
        let mut table = PricingTable::new();
        table.insert("claude", ModelPricing::from_dollars_per_mtok(3.0, 15.0));

        assert!(table.cost_for("claude", &usage(1_000, 0, 0, 0)).is_some());
        assert!(
            table.cost_for("local", &usage(1_000, 0, 0, 0)).is_none(),
            "an unconfigured provider must not price to $0.00"
        );
    }

    #[test]
    fn an_empty_table_prices_nothing() {
        let table = PricingTable::new();

        assert!(table.is_empty());
        assert!(table.cost_for("anything", &usage(1, 1, 1, 1)).is_none());
    }
}
