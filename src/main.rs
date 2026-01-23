//! Acton-AI CLI entry point
//!
//! This binary provides a command-line interface for the Acton-AI framework.
//! For library usage, see the `acton_ai` crate documentation.

use acton_ai::prelude::*;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[acton_main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "acton_ai=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Acton-AI framework");

    let mut runtime = ActonApp::launch_async().await;

    // Spawn the kernel
    let kernel_handle = Kernel::spawn(&mut runtime).await;
    tracing::info!("Kernel started");

    // Example: spawn an agent
    let agent_config = AgentConfig::new("You are a helpful assistant.");
    kernel_handle.send(SpawnAgent { config: agent_config }).await;

    // Keep running until interrupted
    tracing::info!("Acton-AI is running. Press Ctrl+C to shutdown.");
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for ctrl-c");

    tracing::info!("Shutting down...");
    runtime.shutdown_all().await.expect("Shutdown failed");
    tracing::info!("Shutdown complete");
}
