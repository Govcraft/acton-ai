//! Example: Multi-Turn Conversation
//!
//! This example demonstrates multi-turn conversation with history management
//! using the acton-ai API. It shows:
//!
//! 1. Setting up ActonAI runtime with Ollama
//! 2. Reading user input from stdin
//! 3. Maintaining conversation history across turns
//! 4. Streaming responses with token callbacks
//! 5. Graceful exit handling
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
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

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

/// Build an LLM request with the full conversation history.
fn build_request(system_prompt: &str, history: &[Message]) -> LLMRequest {
    let mut builder = LLMRequest::builder().system(system_prompt);

    for msg in history {
        builder = builder.message(msg.clone());
    }

    builder.build()
}

/// Simple actor for collecting stream tokens.
#[acton_actor]
struct StreamCollector;

/// Send a request and stream the response, returning the full text.
async fn send_and_stream(runtime: &ActonAI, request: LLMRequest) -> Result<String, ActonAIError> {
    let stream_done = Arc::new(Notify::new());
    let stream_done_signal = stream_done.clone();

    let buffer = Arc::new(Mutex::new(String::new()));
    let buffer_clone = buffer.clone();

    let mut actor_runtime = runtime.runtime().clone();
    let mut collector = actor_runtime.new_actor::<StreamCollector>();

    let correlation_id = request.correlation_id.clone();
    let expected_id = correlation_id.clone();

    // Handle tokens - print and collect
    collector.mutate_on::<LLMStreamToken>(move |_actor, envelope| {
        if envelope.message().correlation_id == expected_id {
            let token = &envelope.message().token;
            if let Ok(mut buf) = buffer_clone.try_lock() {
                buf.push_str(token);
            }
            print!("{token}");
            std::io::stdout().flush().ok();
        }
        Reply::ready()
    });

    // Handle stream end - signal completion
    let expected_id = correlation_id.clone();
    collector.mutate_on::<LLMStreamEnd>(move |_actor, envelope| {
        if envelope.message().correlation_id == expected_id {
            stream_done_signal.notify_one();
        }
        Reply::ready()
    });

    // Subscribe and start
    collector.handle().subscribe::<LLMStreamToken>().await;
    collector.handle().subscribe::<LLMStreamEnd>().await;
    let collector_handle = collector.start().await;

    // Send request
    runtime.provider_handle().send(request).await;

    // Wait for completion
    stream_done.notified().await;

    // Stop collector
    let _ = collector_handle.stop().await;

    let text = buffer.lock().await.clone();
    Ok(text)
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

    // Conversation history
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

        // Build request with full history
        let request = build_request(system_prompt, &history);

        // Send request and stream response
        print!("Assistant: ");
        let response_text = send_and_stream(&runtime, request).await?;
        println!(); // New line after streaming

        // Add assistant response to history
        history.push(Message::assistant(&response_text));
    }

    // Graceful shutdown
    runtime.shutdown().await?;

    Ok(())
}
