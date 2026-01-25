//! Example: Bash Tool with Hyperlight Sandbox
//!
//! This example demonstrates how to use the built-in bash tool with
//! Hyperlight micro-VM sandboxing for hardware-isolated command execution.
//!
//! # Features
//!
//! - Bash commands execute inside a Hyperlight micro-VM with hardware isolation
//! - Requires KVM on Linux or Hyper-V on Windows
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
//! ```bash
//! cargo run --example bash_sandbox
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

    // Configure Hyperlight sandbox for command execution
    eprintln!("Hyperlight sandbox ENABLED - commands execute in micro-VM");
    let builder = builder.with_hyperlight_sandbox();

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
