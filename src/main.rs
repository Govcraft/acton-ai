//! Acton-AI CLI entry point.
//!
//! This binary provides a command-line interface for the Acton-AI framework.
//! For library usage, see the `acton_ai` crate documentation.
//!
//! The same binary is also re-execed by the process sandbox as a child
//! worker. When the environment variable `ACTON_AI_SANDBOX_RUNNER=1` is set,
//! the entry point detours into [`acton_ai::tools::sandbox::process::runner::main`]
//! *before* any clap parsing or actor-runtime setup.

use acton_ai::cli::Cli;
use clap::Parser;

fn main() {
    // Sandbox runner mode: re-execed child from ProcessSandbox. Never returns.
    // Must run before any actor/tokio runtime construction so the child
    // stays a minimal one-shot executor.
    if is_sandbox_runner() {
        acton_ai::tools::sandbox::process::runner::main();
    }

    cli_main();
}

/// Returns `true` when the binary was re-execed as a sandbox child.
fn is_sandbox_runner() -> bool {
    std::env::var(acton_ai::tools::sandbox::process::runner::env_vars::RUNNER)
        .ok()
        .as_deref()
        == Some("1")
}

/// Regular CLI entry point.
///
/// The `#[acton_main]` proc-macro hard-codes `fn main`, so we inline the
/// tokio runtime setup it would have generated and invoke our async body
/// manually.
fn cli_main() {
    let runtime = acton_reactive::prelude::tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build acton tokio runtime");
    let exit_code = runtime.block_on(async {
        let cli = Cli::parse();
        acton_ai::cli::run(cli).await
    });
    std::process::exit(exit_code);
}
