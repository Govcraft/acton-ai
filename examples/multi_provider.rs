//! Example: Multi-Provider Configuration
//!
//! This example demonstrates how to configure multiple LLM providers via
//! a config file and select which one to use per-prompt.
//!
//! # Setup
//!
//! Copy the example config to your project root or XDG config directory:
//!
//! ```bash
//! # Option 1: Project-local (checked first)
//! cp examples/acton-ai.toml ./acton-ai.toml
//!
//! # Option 2: XDG config directory
//! mkdir -p ~/.config/acton-ai
//! cp examples/acton-ai.toml ~/.config/acton-ai/config.toml
//! ```
//!
//! Edit the config file to match your setup (Ollama URL, API keys, etc.)
//!
//! # Usage
//!
//! ```bash
//! cargo run --example multi_provider
//! ```

use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    // Load providers from config file
    // Searches: ./acton-ai.toml then ~/.config/acton-ai/config.toml
    let runtime = ActonAI::builder()
        .app_name("multi-provider-example")
        .from_config()?
        .launch()
        .await?;

    // Show available providers
    let providers: Vec<_> = runtime.provider_names().collect();
    eprintln!("Available providers: {}", providers.join(", "));
    eprintln!("Default provider: {}", runtime.default_provider_name());
    eprintln!();

    // Use the default provider
    eprintln!("=== Using default provider ({}) ===", runtime.default_provider_name());
    let response = runtime
        .prompt("What is 2 + 2? Answer in one word.")
        .on_token(|t| print!("{t}"))
        .collect()
        .await?;
    println!("\n[Tokens: {}]\n", response.token_count);

    // Use a specific provider if available
    if runtime.has_provider("claude") {
        eprintln!("=== Using claude provider ===");
        let response = runtime
            .prompt("What is the capital of France? Answer in one word.")
            .provider("claude")
            .on_token(|t| print!("{t}"))
            .collect()
            .await?;
        println!("\n[Tokens: {}]\n", response.token_count);
    }

    // Demonstrate error handling for non-existent provider
    eprintln!("=== Attempting to use non-existent provider ===");
    match runtime
        .prompt("Test")
        .provider("nonexistent")
        .collect()
        .await
    {
        Ok(_) => eprintln!("Unexpected success"),
        Err(e) => eprintln!("Expected error: {e}"),
    }

    Ok(())
}
