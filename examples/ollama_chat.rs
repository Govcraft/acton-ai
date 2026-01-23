//! Example: Chat with Ollama on a remote server
//!
//! This example demonstrates using acton-ai to connect to Ollama
//! running on a remote server via network.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example ollama_chat
//! ```

use acton_ai::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

/// A simple response collector that subscribes to LLM events
#[acton_actor]
struct ResponseCollector {
    done: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for logs
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

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

    // Launch the actor runtime
    let mut runtime = ActonApp::launch_async().await;

    // Create a shared buffer to collect the response
    let response_buffer: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let buffer_clone = response_buffer.clone();

    // Create a simple collector actor
    let mut collector = runtime.new_actor::<ResponseCollector>();

    // Handle streaming tokens
    let buffer_for_token = buffer_clone.clone();
    collector.mutate_on::<LLMStreamToken>(move |_actor, envelope| {
        let token = envelope.message().token.clone();
        let buffer = buffer_for_token.clone();
        print!("{}", token); // Print token as it arrives
        use std::io::Write;
        std::io::stdout().flush().ok();
        Reply::pending(async move {
            buffer.lock().await.push_str(&token);
        })
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
            Message::user("What is the capital of France? Answer in one sentence."),
        ],
        tools: None,
    };

    // Send the request
    provider_handle.send(request).await;

    tracing::info!("Request sent, waiting for response...\n");
    print!("Response: ");
    use std::io::Write;
    std::io::stdout().flush().ok();

    // Wait for response
    tokio::time::sleep(Duration::from_secs(30)).await;

    // Print final response
    let final_response = response_buffer.lock().await;
    if final_response.is_empty() {
        tracing::warn!("No response received!");
    } else {
        println!("\n\nFull response: {}", *final_response);
    }

    // Graceful shutdown
    tracing::info!("Shutting down...");
    runtime.shutdown_all().await?;

    Ok(())
}
