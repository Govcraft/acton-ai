//! Acton-AI CLI entry point.
//!
//! This binary provides a command-line interface for the Acton-AI framework.
//! For library usage, see the `acton_ai` crate documentation.

use acton_ai::cli::Cli;
use acton_ai::prelude::*;
use clap::Parser;

#[acton_main]
async fn main() {
    let cli = Cli::parse();
    let exit_code = acton_ai::cli::run(cli).await;
    std::process::exit(exit_code);
}
