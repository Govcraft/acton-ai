//! Example: Spending Budgets
//!
//! Shows the three things an operator needs from a cap: setting one, being
//! told when it is approached, and handling the refusal when it is reached.
//!
//! # Setup
//!
//! Budgets need pricing — a cap over a provider whose tokens cannot be priced
//! is not a cap, so the framework refuses to launch with one. This example
//! uses a local Ollama model and prices it, which is what
//! [`Budget::allow_unpriced`] exists to avoid having to do for genuinely free
//! providers.
//!
//! ```bash
//! ollama serve
//! ollama pull qwen2.5:7b
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example budget
//! ```

use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    // ---------------------------------------------------------------------
    // The one-liner: a process-wide cap, warning at 80%, refusing at the cap.
    //
    //     ActonAI::builder()
    //         .anthropic_from_env()
    //         .budget_usd(5.00)
    //         .launch()
    //         .await?;
    //
    // The full form below is the same thing with the knobs exposed.
    // ---------------------------------------------------------------------

    // Deliberately tiny, so this example actually hits its ceiling.
    let ai = ActonAI::builder()
        .app_name("budget-example")
        .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
        // Pretend rates, so there is something for the cap to measure. The
        // TOML twin of this is `[providers.local.pricing]`.
        .pricing("local", ModelPricing::from_dollars_per_mtok(3.0, 15.0))
        .budget(
            Budget::usd(0.01) // total cap
                .provider("local", 0.005) // per-provider cap, repeatable
                .warn_at_percent(50), // default is 80; 0 disables warnings
        )
        .on_budget_event(|event| eprintln!("!! {event}"))
        .launch()
        .await?;

    // Prompt until the cap refuses one. Every dispatch is checked first, so
    // the refusal costs nothing — no request goes out.
    for round in 1..=20 {
        match ai
            .prompt("Write one sentence about actors.")
            .collect()
            .await
        {
            Ok(response) => println!("round {round}: {}", response.text.trim()),
            Err(err) if err.is_budget_exceeded() => {
                println!("\nrefused on round {round}: {err}");
                break;
            }
            Err(err) => return Err(err),
        }
    }

    // Where the caps stand. `budget` is `None` when none is configured —
    // never a budget with nothing spent against it.
    let usage = ai.usage().await?;
    if let Some(budget) = &usage.budget {
        if let Some(total) = budget.total {
            println!(
                "\ntotal: ${:.4} of ${:.4} ({}%), ${:.4} left",
                total.spent_usd(),
                total.limit_usd(),
                total.percent_used(),
                total.remaining_usd(),
            );
        }
        for (name, scope) in &budget.providers {
            println!(
                "provider `{name}`: ${:.4} of ${:.4} ({}%)",
                scope.spent_usd(),
                scope.limit_usd(),
                scope.percent_used(),
            );
        }
    }

    ai.shutdown().await?;
    Ok(())
}
