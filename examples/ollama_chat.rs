//! Example: Simple Chat with Ollama
//!
//! This example demonstrates the high-level ActonAI API for chat applications.
//! All the complexity of actors, subscriptions, and message handling is hidden.
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
//! # Usage
//!
//! ```bash
//! cargo run --example ollama_chat
//! ```

use acton_ai::prelude::*;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    eprintln!("Loading configuration from acton-ai.toml...");

    // Launch ActonAI with configuration from file
    let runtime = ActonAI::builder()
        .app_name("ollama-chat")
        .from_config()?
        .launch()
        .await?;

    // Send a prompt and stream the response
    let response = runtime
        .prompt("What is the capital of France? Answer in one sentence.")
        .system("You are a helpful assistant. Be concise.")
        .on_token(|token| {
            print!("{token}");
            std::io::stdout().flush().ok();
        })
        .collect()
        .await?;

    println!();
    println!(
        "[{} tokens, {:?}]",
        response.token_count, response.stop_reason
    );

    // Graceful shutdown
    runtime.shutdown().await?;

    Ok(())
}
