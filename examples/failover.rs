//! Example: Failover chains, circuit breaking, and model degradation
//!
//! Shows the three things an operator needs when a provider dies: somewhere
//! else for the traffic to go, a breaker that stops hammering a corpse, and a
//! cheaper model to fall back to when a vendor is merely throttling rather
//! than down.
//!
//! # Setup
//!
//! Two Ollama models stand in for two vendors. Nothing here needs a cloud key,
//! and the `broken` provider deliberately points at a port nothing is
//! listening on — that is what makes the failure real rather than simulated.
//!
//! ```bash
//! ollama serve
//! ollama pull qwen2.5:7b
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example failover
//! ```
//!
//! # What you should see
//!
//! The first prompt goes to `broken`, fails, and is re-dispatched to `local`
//! within the same round — the caller gets an answer and never learns the
//! difference except from the events. After the second failure the circuit
//! opens, and from then on `broken` is skipped without a request going out at
//! all. Thirty seconds later one real request is allowed through as a probe.

use acton_ai::prelude::*;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    // ---------------------------------------------------------------------
    // The one-liner: a chain is one call on the provider that owns it.
    //
    //     ProviderConfig::anthropic(key).with_failover(["backup"])
    //
    // The full form below adds the breaker and the cheaper model.
    // ---------------------------------------------------------------------

    let ai = ActonAI::builder()
        .app_name("failover-example")
        .provider_named(
            "broken",
            // Nothing is listening here. Every dispatch to it fails, which is
            // the point.
            ProviderConfig::openai_compatible("http://127.0.0.1:1/v1", "qwen2.5:7b")
                // Tried in order when `broken` cannot serve a round. Only this
                // provider's chain is consulted — fallbacks do not chain on to
                // fallbacks, so there are no cycles to reason about.
                .with_failover(["local"])
                // Two consecutive failures open the circuit for 30 seconds.
                // The default is 5 and 30s; `without_circuit_breaker()` opts
                // out entirely.
                .with_circuit_breaker(CircuitBreakerConfig::new(2, Duration::from_secs(30)))
                // A *rate limit* is not a death: rather than moving vendors,
                // the provider re-dispatches to this cheaper model on its own
                // endpoint and returns to the primary model once the limit
                // clears. Nothing to reset by hand.
                .with_fallback_model("qwen2.5:0.5b"),
        )
        .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
        .default_provider("broken")
        // Every routing decision is a broadcast, so an operator never has to
        // infer a failover from a latency graph.
        .on_failover_event(|event| eprintln!("!! {event}"))
        .launch()
        .await?;

    // Three prompts against a provider that is never coming back.
    //
    // Rounds 1 and 2 dispatch to `broken`, fail, and fail over to `local`;
    // round 2's failure trips the breaker. Round 3 skips `broken` outright —
    // watch the timing, not just the answer: an open circuit costs nothing.
    for round in 1..=3 {
        match ai.prompt("Reply with one short sentence.").collect().await {
            Ok(response) => println!("round {round}: {}", response.text.trim()),
            // Every candidate refused. This error is the whole story: each
            // provider that was tried, and why it did not serve.
            Err(err) if err.is_all_providers_failed() => {
                println!("\nround {round} had nowhere to go:");
                for attempt in err.provider_attempts().unwrap_or_default() {
                    println!("  {attempt}");
                }
                break;
            }
            Err(err) => return Err(err),
        }
    }

    // Billing follows the provider that actually served, not the one that was
    // asked. Nothing had to be wired up for that: each provider stamps its own
    // name on the usage it reports.
    let usage = ai.usage().await?;
    for (name, provider) in &usage.providers {
        println!(
            "\nprovider `{name}`: {} requests, {} input tokens",
            provider.requests, provider.usage.input_tokens,
        );
    }

    // ---------------------------------------------------------------------
    // The same setup in TOML:
    //
    //     [providers.broken]
    //     type = "openai"
    //     model = "qwen2.5:7b"
    //     base_url = "http://127.0.0.1:1/v1"
    //     failover = ["local"]
    //     fallback_model = "qwen2.5:0.5b"
    //
    //     [providers.broken.circuit_breaker]
    //     failure_threshold = 2
    //     cooldown_secs = 30
    //     enabled = true
    //
    // `acton-ai config` renders the resolved values, including the breaker
    // defaults that apply when the table is absent.
    // ---------------------------------------------------------------------

    ai.shutdown().await?;
    Ok(())
}
