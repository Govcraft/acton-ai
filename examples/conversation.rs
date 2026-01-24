//! Example: Multi-Turn Conversation with Tools
//!
//! This example demonstrates the minimal API for building a terminal chat
//! using the `run_chat()` method.
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
//! cargo run --example conversation
//! ```

use acton_ai::prelude::*;

/// Load environment variables from .env file if present.
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

    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());

    eprintln!("Connecting to Ollama at {ollama_url} with model {model}");
    eprintln!("Say goodbye to end the conversation.\n");

    // Minimal chat application - just 5 lines!
    // - with_builtins() gives access to bash, read_file, etc.
    // - run_chat() handles input/output, streaming, history, and exit detection
    // - Default system prompt is used automatically
    ActonAI::builder()
        .app_name("conversation-example")
        .ollama_at(&ollama_url, &model)
        .with_builtins()
        .launch()
        .await?
        .conversation()
        .run_chat()
        .await
}
