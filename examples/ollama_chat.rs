//! Example: Simple Chat with Ollama
//!
//! This example demonstrates the high-level ActonAI API for chat applications.
//! All the complexity of actors, subscriptions, and message handling is hidden.
//!
//! # Configuration
//!
//! Set environment variables or create a `.env` file:
//!
//! ```bash
//! export OLLAMA_URL="http://localhost:11434/v1"
//! export OLLAMA_MODEL="qwen2.5:7b"
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example ollama_chat
//! ```

use acton_ai::prelude::*;
use std::io::Write;

/// Load environment variables from .env file if present
fn load_dotenv() {
    if let Ok(contents) = std::fs::read_to_string(".env") {
        for line in contents.lines() {
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"');
                if !key.is_empty() && !key.starts_with('#') {
                    std::env::set_var(key, value);
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    load_dotenv();

    // Get configuration from environment
    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());

    eprintln!("Connecting to Ollama at {ollama_url} with model {model}");

    // Launch ActonAI with Ollama
    let runtime = ActonAI::builder()
        .app_name("ollama-chat")
        .ollama_at(&ollama_url, &model)
        .launch()
        .await?;

    // Send a prompt and stream the response
    print!("Response: ");
    std::io::stdout().flush().ok();

    let response = runtime
        .prompt("What is the capital of France? Answer in one paragraph.")
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
