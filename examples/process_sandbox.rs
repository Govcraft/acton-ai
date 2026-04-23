//! Example: bash tool routed through the portable ProcessSandbox.
//!
//! This example shows how to enable the [`ProcessSandbox`] for sandboxed
//! builtin tool execution. Each bash invocation re-execs the current binary
//! as a child process, applies rlimits (address space, CPU, file size), and
//! on Linux kernels 5.13+ installs best-effort landlock + seccomp filters
//! before the tool runs. The parent enforces a wall-clock timeout and kills
//! the child's process group on overrun.
//!
//! This is NOT hardware isolation — there is no VM or hypervisor. The aim
//! is pragmatic: a small, auditable, cross-platform fence that catches the
//! realistic failure modes (runaway processes, accidental /etc writes,
//! syscall-based escape tricks) without pinning the crate to a single host
//! architecture.
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
//! Optional sandbox tuning:
//!
//! ```toml
//! [sandbox]
//! hardening = "besteffort"    # "off" | "besteffort" | "enforce"
//!
//! [sandbox.limits]
//! max_execution_ms = 30000
//! max_memory_mb = 256
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example process_sandbox
//! ```
//!
//! [`ProcessSandbox`]: acton_ai::tools::sandbox::ProcessSandbox

use acton_ai::prelude::*;
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    eprintln!("Loading configuration from acton-ai.toml...");

    // Build the runtime with bash tool enabled from config file.
    // The bash tool is marked as `sandboxed: true` by default, so with a
    // ProcessSandbox factory registered the bash invocations below are
    // routed through a subprocess instead of running in-process.
    let builder = ActonAI::builder()
        .app_name("process-sandbox-example")
        .from_config()?
        .with_builtin_tools(&["bash"]) // Enable only the bash tool
        .with_process_sandbox();

    eprintln!(
        "ProcessSandbox ENABLED - bash commands execute in a re-execed child \
         with rlimits + (on Linux) landlock/seccomp"
    );

    let runtime = builder.launch().await?;

    eprintln!("\nAsking LLM to run a simple bash command...\n");

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

    eprintln!("\n--- Second prompt: current working directory ---\n");

    let response = runtime
        .prompt("What is the current working directory?")
        .system(
            "You are a helpful assistant with access to a bash tool. \
             When asked to run commands like `date` or `ls`, use the bash tool. \
             Report the command output clearly.",
        )
        .use_builtins()
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

    runtime.shutdown().await?;

    Ok(())
}
