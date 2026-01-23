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
use std::time::Duration;

/// A simple response collector that subscribes to LLM events.
/// Uses actor state to collect tokens - no mutex needed!
#[acton_actor]
struct ResponseCollector {
    /// Buffer to collect streamed tokens
    buffer: String,
    /// Whether the stream has completed
    done: bool,
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

    // Handle streaming tokens - append to actor's buffer (no mutex!)
    collector.mutate_on::<LLMStreamToken>(|actor, envelope| {
        let token = &envelope.message().token;
        // Print token as it arrives
        print!("{}", token);
        use std::io::Write;
        std::io::stdout().flush().ok();
        // Append to actor's own buffer
        actor.model.buffer.push_str(token);
        Reply::ready()
    });

    // Handle stream end
    collector.mutate_on::<LLMStreamEnd>(|actor, envelope| {
        tracing::info!("\n[Stream ended: {:?}]", envelope.message().stop_reason);
        actor.model.done = true;
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

    // Wait for provider to initialize
    tokio::time::sleep(Duration::from_millis(200)).await;

    tracing::info!("LLM Provider ready, sending prompt...");

    // Create the request
    let correlation_id = CorrelationId::new();
    let agent_id = AgentId::new();

    let request = LLMRequest {
        correlation_id: correlation_id.clone(),
        agent_id,
        messages: vec![
            Message::system("You are a helpful assistant. Be concise."),
            Message::user("What is the capital of France? Answer in three sentences."),
        ],
        tools: None,
    };

    // Send the request
    provider_handle.send(request).await;

    tracing::info!("Request sent, waiting for response...\n");
    print!("Response: ");
    use std::io::Write;
    std::io::stdout().flush().ok();

    // Wait for response to complete
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Graceful shutdown - this triggers after_stop on the collector
    tracing::info!("Shutting down...");
    runtime.shutdown_all().await?;

    Ok(())
}
