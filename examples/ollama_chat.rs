//! Example: Chat with Ollama on a remote server
//!
//! This example demonstrates using acton-ai to connect to Ollama
//! running on a remote server via network.
//!
//! File logging is automatically initialized by the kernel and writes to
//! `~/.local/share/acton/logs/`. No manual logging setup is required.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example ollama_chat
//! ```

use acton_ai::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

/// A simple response collector that subscribes to LLM events.
/// Uses actor state to collect tokens - no mutex needed!
#[acton_actor]
struct ResponseCollector {
    /// Buffer to collect streamed tokens
    buffer: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Launch the actor runtime
    let mut runtime = ActonApp::launch_async().await;

    // Spawn the kernel with custom app name for logs
    // This automatically initializes file logging to ~/.local/share/acton/logs/
    let kernel_config = KernelConfig::default().with_app_name("ollama-chat");
    let _kernel = Kernel::spawn_with_config(&mut runtime, kernel_config).await;

    // Get and display the log directory
    let log_config = LoggingConfig::default().with_app_name("ollama-chat");
    if let Ok(log_dir) = get_log_dir(&log_config) {
        eprintln!("Logs being written to: {}", log_dir.display());
    }

    tracing::info!("Starting Ollama chat example...");

    // Configure for Ollama on server (network IP)
    let ollama_url = "http://localhost:11434/v1";
    let model = "qwen2.5:3b";

    let provider_config = ProviderConfig::openai_compatible(ollama_url, model)
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

    // Create the response collector actor
    let mut collector = runtime.new_actor::<ResponseCollector>();

    // Use lifecycle hook to print the collected response when actor stops
    collector.after_stop(|actor| {
        if actor.model.buffer.is_empty() {
            eprintln!("\nNo response received!");
        } else {
            println!("\n\nFull response: {}", actor.model.buffer);
        }
        Reply::ready()
    });

    // Handle streaming tokens - append to actor's buffer
    collector.mutate_on::<LLMStreamToken>(|actor, envelope| {
        let token = &envelope.message().token;
        print!("{}", token);
        use std::io::Write;
        std::io::stdout().flush().ok();
        actor.model.buffer.push_str(token);
        Reply::ready()
    });

    // Handle stream end - signal completion
    collector.mutate_on::<LLMStreamEnd>(move |_actor, envelope| {
        tracing::info!("\n[Stream ended: {:?}]", envelope.message().stop_reason);
        stream_done_signal.notify_one();
        Reply::ready()
    });

    // Handle stream start
    collector.act_on::<LLMStreamStart>(|_actor, envelope| {
        tracing::info!("[Stream started: {}]", envelope.message().correlation_id);
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

    // Create the request
    let request = LLMRequest {
        correlation_id: CorrelationId::new(),
        agent_id: AgentId::new(),
        messages: vec![
            Message::system("You are a helpful assistant. Be concise."),
            Message::user("What is the capital of France? Answer in one sentence."),
        ],
        tools: None,
    };

    // Send the request
    print!("Response: ");
    use std::io::Write;
    std::io::stdout().flush().ok();

    provider_handle.send(request).await;

    // Wait for stream completion (no arbitrary sleep!)
    stream_done.notified().await;

    // Graceful shutdown - this triggers after_stop on the collector
    tracing::info!("Shutting down...");
    provider_handle.stop().await?;
    runtime.shutdown_all().await?;

    Ok(())
}
