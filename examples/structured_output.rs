//! Example: Typed Structured Output
//!
//! This example demonstrates `extract::<T>()`, which returns a typed,
//! schema-validated Rust value instead of prose. It shows:
//!
//! 1. Deriving `Deserialize` + `JsonSchema` on the target type
//! 2. Extracting a nested struct from unstructured text
//! 3. Combining extraction with a real tool, which runs first
//!
//! Under the hood acton-ai appends a synthetic `structured_output` tool whose
//! input schema is the schema of your type, then constrains the request so
//! the model has to call it. If the arguments don't deserialize, the serde
//! error goes back to the model and it is asked to correct itself.
//!
//! # Configuration
//!
//! Create an `acton-ai.toml` file in the project root or at
//! `~/.config/acton-ai/config.toml`:
//!
//! ```toml
//! default_provider = "ollama"
//!
//! [providers.ollama]
//! type = "ollama"
//! model = "qwen2.5:7b"
//! base_url = "http://localhost:11434/v1"
//! ```
//!
//! Pick a model that supports tool calling — extraction is built on it.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example structured_output
//! ```

use acton_ai::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// A single billed line on an invoice.
#[derive(Debug, Deserialize, JsonSchema)]
struct LineItem {
    /// What was billed.
    description: String,
    /// Amount in cents, to keep the money integral.
    cents: u64,
}

/// An invoice pulled out of an email.
#[derive(Debug, Deserialize, JsonSchema)]
struct Invoice {
    /// Who sent the invoice.
    vendor: String,
    /// Invoice total in cents.
    total_cents: u64,
    /// Every billed line.
    line_items: Vec<LineItem>,
}

/// How risky the reviewer considers a purchase.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Risk {
    Low,
    Medium,
    High,
}

/// A review decision about an invoice.
#[derive(Debug, Deserialize, JsonSchema)]
struct Review {
    /// The reviewer's risk call.
    risk: Risk,
    /// One sentence explaining the call.
    rationale: String,
}

const EMAIL: &str = "\
Hi — attached is our invoice for last month.

Acme Supplies
  Widget assembly ................ $40.00
  Expedited shipping .............  $2.50
  Total .......................... $42.50

Payment due in 30 days. Thanks!";

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    let runtime = ActonAI::builder()
        .app_name("structured-output-example")
        .from_config()?
        .launch()
        .await?;

    // --- 1. Plain extraction -------------------------------------------
    // No tools, so the model is told to record its answer straight away.

    println!("Extracting the invoice...\n");

    let invoice: Invoice = runtime
        .prompt(format!("Extract the invoice from this email:\n\n{EMAIL}"))
        .system("Extract exactly what the email states. Amounts are in cents.")
        .extract::<Invoice>()
        .await?;

    println!("Vendor:  {}", invoice.vendor);
    println!("Total:   {} cents", invoice.total_cents);
    for item in &invoice.line_items {
        println!("  - {:<24} {:>7} cents", item.description, item.cents);
    }

    // --- 2. Extraction alongside a real tool ----------------------------
    // The model may call `vendor_history` first; extraction is the terminal
    // step either way.

    println!("\nReviewing the invoice...\n");

    let review: Review = runtime
        .prompt(format!(
            "Review this invoice for risk, checking the vendor's history first:\n\n{invoice:#?}"
        ))
        .system("Use vendor_history before deciding. Be brief.")
        .tool(
            "vendor_history",
            "Returns how long a vendor has been a customer and any disputes",
            json!({
                "type": "object",
                "properties": {
                    "vendor": {"type": "string", "description": "The vendor's name"}
                },
                "required": ["vendor"],
            }),
            |args| async move {
                let vendor = args["vendor"].as_str().unwrap_or("unknown");
                println!("  [tool] looking up {vendor}");
                Ok(json!({
                    "vendor": vendor,
                    "years_active": 6,
                    "open_disputes": 0,
                }))
            },
        )
        .extract::<Review>()
        .await?;

    println!("\nRisk:      {:?}", review.risk);
    println!("Rationale: {}", review.rationale);

    runtime.shutdown().await?;
    Ok(())
}
