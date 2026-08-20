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
    // Sandbox runner mode: re-execed child from ProcessSandbox. Never returns
    // when the runner env var is set. Must run before any actor/tokio runtime
    // construction so the child stays a minimal one-shot executor. This is
    // the same guard embedder binaries are told to install, so the CLI and
    // the documented contract cannot drift apart.
    acton_ai::tools::sandbox::process::runner::run_if_sandbox_child();

    // Before clap, before the tokio runtime, before anything that could open a
    // socket: rustls honours only the first crypto provider installed in a
    // process, so a FIPS build has exactly one chance to install the right
    // one and this is it.
    if let Err(error) = acton_ai::fips::install_crypto_provider() {
        eprintln!("error: {error}");
        std::process::exit(acton_ai::cli::error::exit_code::RUNTIME_ERROR);
    }

    cli_main();
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
