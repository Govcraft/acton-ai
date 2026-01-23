//! Example: Advanced Chat with Ollama (Low-Level API)
//!
//! This example demonstrates using acton-ai's low-level actor API for full
//! control over the streaming response handling. Use this approach when you
//! need custom actor state or complex stream processing.
//!
//! For simpler use cases, see the `ollama_chat.rs` example which uses the
//! high-level `ActonAI` facade.
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
//! cargo run --example ollama_chat_advanced
//! ```

use acton_ai::prelude::*;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

/// Spinner frames for the thinking animation
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// A custom response collector that subscribes to LLM events.
/// Uses actor state to collect tokens and drive the spinner animation.
#[acton_actor]
struct ResponseCollector {
    /// Buffer to collect streamed tokens
    buffer: String,
    /// Current spinner frame (advanced by each token)
    frame: usize,
    /// Word count for statistics
    word_count: usize,
}

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
async fn main() -> anyhow::Result<()> {
    // Load .env file if present
    load_dotenv();

    // Launch the actor runtime
    let mut runtime = ActonApp::launch_async().await;

    // Spawn the kernel with custom app name for logs
    let kernel_config = KernelConfig::default().with_app_name("ollama-chat-advanced");
    let _kernel = Kernel::spawn_with_config(&mut runtime, kernel_config).await;

    // Get and display the log directory
    let log_config = LoggingConfig::default().with_app_name("ollama-chat-advanced");
    if let Ok(log_dir) = get_log_dir(&log_config) {
        eprintln!("Logs being written to: {}", log_dir.display());
    }

    tracing::info!("Starting advanced Ollama chat example...");

    // Configure for Ollama - URL and model from environment
    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());

    let provider_config = ProviderConfig::openai_compatible(&ollama_url, &model)
        .with_timeout(Duration::from_secs(120))
        .with_max_tokens(256);

    tracing::info!(
        "Connecting to Ollama at {} with model {}",
        ollama_url,
        model
    );

    // Notify to signal stream completion
    let stream_done = Arc::new(Notify::new());
    let stream_done_signal = stream_done.clone();

    // Create the response collector actor with custom state
    let mut collector = runtime.new_actor::<ResponseCollector>();

    // Use lifecycle hook to print the collected response when actor stops
    collector.after_stop(|actor| {
        // Clear the spinner line and print the response
        print!("\r\x1b[K"); // Carriage return + clear line
        if actor.model.buffer.is_empty() {
            println!("No response received!");
        } else {
            println!("Response: {}", actor.model.buffer);
            println!(
                "[{} tokens, {} words]",
                actor.model.frame, actor.model.word_count
            );
        }
        std::io::stdout().flush().ok();
        Reply::ready()
    });

    // Handle stream start - show initial spinner
    collector.mutate_on::<LLMStreamStart>(|actor, envelope| {
        tracing::info!("[Stream started: {}]", envelope.message().correlation_id);
        print!(
            "\r{} Thinking...",
            SPINNER[actor.model.frame % SPINNER.len()]
        );
        std::io::stdout().flush().ok();
        Reply::ready()
    });

    // Handle streaming tokens - advance spinner and collect
    collector.mutate_on::<LLMStreamToken>(|actor, envelope| {
        let token = &envelope.message().token;
        actor.model.buffer.push_str(token);
        actor.model.frame += 1;
        // Count words as they arrive
        actor.model.word_count += token.split_whitespace().count();
        print!(
            "\r{} Thinking...",
            SPINNER[actor.model.frame % SPINNER.len()]
        );
        std::io::stdout().flush().ok();
        Reply::ready()
    });

    // Handle stream end - signal completion
    collector.mutate_on::<LLMStreamEnd>(move |_actor, envelope| {
        tracing::info!("[Stream ended: {:?}]", envelope.message().stop_reason);
        stream_done_signal.notify_one();
        Reply::ready()
    });

    // Subscribe to streaming events BEFORE starting
    collector.handle().subscribe::<LLMStreamToken>().await;
    collector.handle().subscribe::<LLMStreamEnd>().await;
    collector.handle().subscribe::<LLMStreamStart>().await;

    let _collector_handle = collector.start().await;

    // Create and start the LLM provider
    let provider_handle = LLMProvider::spawn(&mut runtime, provider_config).await;

    tracing::info!("Sending prompt...");

    // Create the request using the new builder API
    let request = LLMRequest::builder()
        .system("You are a helpful assistant. Be concise.")
        .user("What is the capital of France? Answer in one paragraph.")
        .build();

    // Send the request
    provider_handle.send(request).await;

    // Wait for stream completion (driven by token messages)
    stream_done.notified().await;

    // Graceful shutdown - this triggers after_stop on the collector
    tracing::info!("Shutting down...");
    runtime.shutdown_all().await?;

    Ok(())
}
