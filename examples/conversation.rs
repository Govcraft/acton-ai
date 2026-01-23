//! Example: Multi-Turn Conversation
//!
//! This example demonstrates multi-turn conversation with history management
//! using the high-level ActonAI `PromptBuilder` API. It showcases:
//!
//! 1. Setting up ActonAI runtime with Ollama
//! 2. Using the fluent `.messages()` API for conversation history
//! 3. Streaming responses with `.on_token()` callback
//! 4. Clean conversation loop without any mutexes or low-level actor boilerplate
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

/// Check if the user wants to quit the conversation.
fn should_quit(input: &str) -> bool {
    let lower = input.to_lowercase();
    matches!(lower.as_str(), "quit" | "exit" | "/quit" | "/exit")
}

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    load_dotenv();

    // Get configuration from environment
    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());

    eprintln!("Connecting to Ollama at {ollama_url} with model {model}");
    eprintln!("Type 'quit' or 'exit' to end the conversation.\n");

    // Launch ActonAI runtime
    let runtime = ActonAI::builder()
        .app_name("conversation-example")
        .ollama_at(&ollama_url, &model)
        .launch()
        .await?;

    // System prompt for the assistant
    let system_prompt =
        "You are a helpful assistant. Be conversational and remember our previous exchanges.";

    // Conversation history - simple Vec, no mutexes needed!
    let mut history: Vec<Message> = Vec::new();

    // Main conversation loop
    loop {
        let Some(input) = read_user_input("You: ") else {
            break; // EOF
        };

        if input.is_empty() {
            continue;
        }

        if should_quit(&input) {
            eprintln!("\nGoodbye!");
            break;
        }

        // Add user message to history
        history.push(Message::user(&input));

        // Send request with full history using the clean PromptBuilder API
        print!("Assistant: ");
        let response = runtime
            .prompt("") // Ignored when .messages() is used
            .system(system_prompt)
            .messages(history.clone())
            .on_token(|token| {
                print!("{token}");
                std::io::stdout().flush().ok();
            })
            .collect()
            .await?;
        println!(); // New line after streaming

        // Add assistant response to history
        history.push(Message::assistant(&response.text));
    }

    // Graceful shutdown
    runtime.shutdown().await?;

    Ok(())
}
