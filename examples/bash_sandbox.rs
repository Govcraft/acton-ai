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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    eprintln!("Loading configuration from acton-ai.toml...");

    // Build the runtime with bash tool enabled from config file
    // The bash tool is marked as `sandboxed: true` by default
    let builder = ActonAI::builder()
        .app_name("bash-sandbox-example")
        .from_config()?
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
