//! Example: OpenTelemetry Export
//!
//! Traces every prompt loop as `turn → llm_round → tool`, and exports metrics
//! for tokens, latency, and reliability, over OTLP to any collector.
//!
//! # See the data
//!
//! Jaeger ships an all-in-one image that accepts OTLP directly and serves a
//! trace UI, which is the shortest path to actually looking at a trace:
//!
//! ```bash
//! docker run --rm \
//!     -p 4318:4318 \
//!     -p 16686:16686 \
//!     jaegertracing/jaeger:latest
//! ```
//!
//! Then run this example and browse <http://localhost:16686>, picking the
//! `telemetry-example` service. Each prompt is one `acton_ai.turn` trace with
//! a child span per provider round and per tool call.
//!
//! Jaeger stores traces, not metrics. To see the metrics as well, point the
//! endpoint at an OpenTelemetry Collector configured with a Prometheus
//! exporter — the `[telemetry]` block is identical either way, which is the
//! whole point of exporting OTLP rather than a vendor format.
//!
//! # Setup
//!
//! ```bash
//! ollama serve
//! ollama pull qwen2.5:7b
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example telemetry
//! ```

use acton_ai::prelude::*;
use acton_ai::telemetry::Telemetry;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    // ---------------------------------------------------------------------
    // The one-liner. Everything else takes its default: service name
    // "acton-ai", metrics exported every 60s, no headers.
    //
    //     ActonAI::builder()
    //         .anthropic_from_env()
    //         .telemetry_otlp("http://localhost:4318")
    //         .launch()
    //         .await?;
    //
    // The full form below is the same thing with the knobs exposed.
    // ---------------------------------------------------------------------

    let ai = ActonAI::builder()
        .app_name("telemetry-example")
        .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
        // `.header(name, value)` is repeatable, for collectors behind bearer
        // auth: `.header("authorization", "Bearer ...")`.
        .telemetry(
            Telemetry::otlp("http://localhost:4318")
                // How traces and metrics are grouped in the backend.
                .service_name("telemetry-example")
                // Default is 60s; shortened so this example's metrics show up
                // while you are still watching.
                .metrics_interval_secs(5),
        )
        .launch()
        .await?;

    // A plain prompt: one `acton_ai.turn` span with a single child round.
    println!("-- plain prompt --");
    let response = ai
        .prompt("In one sentence, what is a trace span?")
        .on_token(|token| print!("{token}"))
        .collect()
        .await?;
    println!("\n   ({} tokens)\n", response.token_count);

    // A prompt with a tool: the turn now has two rounds and a tool span
    // between them. The tool span records the tool's *name* and whether it
    // succeeded — never its arguments or its result, which are user data.
    println!("-- prompt with a tool --");
    let response = ai
        .prompt("What is 17 * 42? Use the calculator.")
        .tool(
            "calculator",
            "Evaluates an arithmetic expression",
            serde_json::json!({
                "type": "object",
                "properties": { "expression": { "type": "string" } },
                "required": ["expression"]
            }),
            |args| async move {
                let expression = args["expression"].as_str().unwrap_or_default();
                Ok(serde_json::json!({ "result": format!("{expression} = 714") }))
            },
        )
        .collect()
        .await?;
    println!("{}\n", response.text);

    // Shutting down flushes telemetry *after* the actors stop, so the final
    // spans and the last token counts are exported rather than dropped on
    // the way out. Skipping this is how the end of a run goes missing.
    ai.shutdown().await?;

    println!("Traces sent. Browse http://localhost:16686 and pick the");
    println!("`telemetry-example` service.");

    Ok(())
}
