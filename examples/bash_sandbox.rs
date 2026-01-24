//! Example: Bash Tool with Hyperlight Sandbox
//!
//! This example demonstrates how to use the built-in bash tool with optional
//! Hyperlight micro-VM sandboxing for hardware-isolated command execution.
//!
//! # Features
//!
//! - **Without `hyperlight` feature**: Bash commands execute directly on the host
//! - **With `hyperlight` feature**: Bash commands execute inside a Hyperlight
//!   micro-VM with hardware isolation (requires KVM on Linux, Hyper-V on Windows)
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
//! Run without sandbox (direct execution):
//! ```bash
//! cargo run --example bash_sandbox
//! ```
//!
//! Run with Hyperlight sandbox (requires hypervisor):
//! ```bash
//! cargo run --example bash_sandbox --features hyperlight
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    load_dotenv();

    // Get configuration from environment
    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());

    eprintln!("Connecting to Ollama at {ollama_url} with model {model}");

    // Build the runtime with bash tool enabled
    // The bash tool is marked as `sandboxed: true` by default
    let builder = ActonAI::builder()
        .app_name("bash-sandbox-example")
        .provider(
            ProviderConfig::openai_compatible(&ollama_url, &model)
                .with_timeout(std::time::Duration::from_secs(120))
                .with_max_tokens(512),
        )
        .with_builtin_tools(&["bash"]); // Enable only the bash tool

    // Configure sandbox based on feature flag
    #[cfg(feature = "hyperlight")]
    let builder = {
        eprintln!("Hyperlight sandbox ENABLED - commands execute in micro-VM");
        builder.with_hyperlight_sandbox()
    };

    #[cfg(not(feature = "hyperlight"))]
    eprintln!("Hyperlight sandbox DISABLED - commands execute directly on host");

    let runtime = builder.launch().await?;

    eprintln!("\nAsking LLM to run a simple bash command...\n");

    // Ask the LLM to run a command - it will use the bash tool
    let response = runtime
        .prompt("What is the current date and time?")
        .system(
            "You are a helpful assistant with access to a bash tool. \
             When asked to run commands like `date` or `ls`, use the bash tool. \
             Report the command output clearly.",
        )
        .use_builtins() // Make built-in tools available to the LLM
        .on_token(|token| {
            print!("{token}");
            std::io::stdout().flush().ok();
        })
        .collect()
        .await?;

    println!("\n");
    println!("--- Response Stats ---");
    println!("Total tokens: {}", response.token_count);
    println!("Stop reason: {:?}", response.stop_reason);
    println!("Tool calls: {}", response.tool_calls.len());
    for tc in &response.tool_calls {
        let cmd = tc
            .arguments
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        println!("  - {} `{}`", tc.name, cmd);
        println!("    {:?}", tc.result);
    }

    // Demonstrate a more complex command
    eprintln!("\n--- Second prompt: listing files ---\n");

    let response = runtime
        .prompt("What is the current working directory?")
        .system(
            "You are a helpful assistant with access to a bash tool. \
             When asked to run commands like `date` or `ls`, use the bash tool. \
             Report the command output clearly.",
        )
        .use_builtins() // Make built-in tools available to the LLM
        .on_token(|token| {
            print!("{token}");
            std::io::stdout().flush().ok();
        })
        .collect()
        .await?;

    println!("\n");
    println!("--- Response Stats ---");
    println!("Total tokens: {}", response.token_count);
    println!("Stop reason: {:?}", response.stop_reason);
    println!("Tool calls: {}", response.tool_calls.len());
    for tc in &response.tool_calls {
        let cmd = tc
            .arguments
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        println!("  - {} `{}`", tc.name, cmd);
        println!("    {:?}", tc.result);
    }

    // Shutdown cleanly
    runtime.shutdown().await?;

    Ok(())
}
