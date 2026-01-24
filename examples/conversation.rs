//! Example: Multi-Turn Conversation with Tools
//!
//! This example demonstrates multi-turn conversation using the `Conversation`
//! abstraction which provides automatic history management and built-in exit tool.
//!
//! ## Features Demonstrated
//!
//! 1. Setting up ActonAI runtime with Ollama and built-in tools
//! 2. Using the `Conversation` API for automatic history management
//! 3. Streaming responses with `send_streaming()`
//! 4. Built-in tools (bash, read_file, etc.) for agentic capabilities
//! 5. Built-in exit tool with `with_exit_tool()` for graceful conversation termination
//!
//! ## Key Benefits of Conversation API
//!
//! - **No manual history tracking**: Messages are automatically added to history
//! - **No cloning**: History is managed internally, no `.clone()` needed
//! - **No empty strings**: No confusing `.prompt("")` for continuations
//! - **Auto-builtins**: When configured with `with_builtins()`, tools are available
//! - **Built-in exit tool**: No manual `Arc<AtomicBool>` or tool definition needed
//!
//! # Configuration
//!
//! Set environment variables or create a `.env` file:
//!
//! ```bash
//! export OLLAMA_URL="http://localhost:11434/v1"
//! export OLLAMA_MODEL="qwen2.5:7b"  # 7b+ recommended for tool calling
//! ```
//!
//! **Note:** This example uses tool calling for exit detection. Smaller models
//! (3b and below) may not handle tool calling reliably. Use 7b+ models for
//! best results.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example conversation
//! ```

use acton_ai::prelude::*;
use std::io::Write;

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

/// Read a line of input from the user.
///
/// Returns `None` if EOF is reached or an error occurs.
fn read_user_input(prompt: &str) -> Option<String> {
    print!("{prompt}");
    std::io::stdout().flush().ok();

    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        Ok(0) => None, // EOF
        Ok(_) => Some(input.trim().to_string()),
        Err(_) => None,
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
    eprintln!("Say goodbye to end the conversation.\n");

    // Launch ActonAI runtime with built-in tools
    // Note: with_builtins() auto-enables builtins on all prompts/conversations
    let runtime = ActonAI::builder()
        .app_name("conversation-example")
        .ollama_at(&ollama_url, &model)
        .with_builtins() // Auto-enables built-in tools on every prompt
        .launch()
        .await?;

    // Create a managed conversation with system prompt and built-in exit tool
    // No manual Arc<AtomicBool> or exit tool definition needed!
    let mut conv = runtime
        .conversation()
        .system(
            "You are a helpful assistant with access to tools including bash, \
             read_file, write_file, glob, grep, and more. Be conversational and \
             remember our previous exchanges. Use tools when appropriate to help \
             the user. When the user wants to leave, use the exit_conversation tool.",
        )
        .with_exit_tool() // Enable built-in exit tool
        .build();

    // Main conversation loop
    loop {
        let Some(input) = read_user_input("You: ") else {
            break; // EOF
        };

        if input.is_empty() {
            continue;
        }

        print!("Assistant: ");

        // Simply use send_streaming - exit tool is automatically injected
        // History is automatically managed - no need to push/clone messages!
        let _response = conv
            .send_streaming(&input, |token| {
                print!("{token}");
                std::io::stdout().flush().ok();
            })
            .await?;
        println!();

        // Check if exit was triggered - simple and clean!
        if conv.should_exit() {
            break;
        }
    }

    // Graceful shutdown
    runtime.shutdown().await?;

    Ok(())
}
