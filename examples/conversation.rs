//! Example: Multi-Turn Conversation with Tools
//!
//! This example demonstrates the minimal API for building a terminal chat
//! using the `run_chat()` method.
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
//! cargo run --example conversation
//! ```

use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    eprintln!("Loading configuration from acton-ai.toml...");
    eprintln!("Say goodbye to end the conversation.\n");

    // Minimal chat application - just 5 lines!
    // - from_config() loads provider settings from acton-ai.toml
    // - with_builtins() gives access to bash, read_file, etc.
    // - run_chat() handles input/output, streaming, history, and exit detection
    ActonAI::builder()
        .app_name("conversation-example")
        .from_config()?
        .with_builtins()
        .launch()
        .await?
        .conversation()
        .run_chat()
        .await
}
