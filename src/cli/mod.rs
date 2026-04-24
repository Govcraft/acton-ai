//! Command-line interface for acton-ai.
//!
//! Provides a scriptable CLI with session management, autonomous heartbeat,
//! chat, and job execution capabilities.

pub mod chat_ui;
pub mod commands;
pub mod error;
pub mod output;
pub mod runtime;

use clap::{Parser, Subcommand};
use error::exit_code;
use output::{OutputMode, OutputWriter};
use std::path::PathBuf;

/// Acton-AI: An agentic AI framework built on the actor model.
#[derive(Parser, Debug)]
#[command(
    name = "acton-ai",
    version,
    about = "An agentic AI framework built on the actor model",
    long_about = "Acton-AI provides scriptable AI agents with persistent sessions,\n\
                  autonomous task execution, and tool-using capabilities."
)]
pub struct Cli {
    /// Output in JSON format (machine-readable).
    #[arg(long, global = true)]
    pub json: bool,

    /// Path to configuration file (overrides default search paths).
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Override the default LLM provider.
    #[arg(long, global = true)]
    pub provider: Option<String>,

    /// Increase verbosity (-v info, -vv debug, -vvv trace).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Suppress all stderr output.
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// The command to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI commands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Autonomous wake-up cycle — review and execute due heartbeat entries.
    Heartbeat(commands::heartbeat::HeartbeatArgs),

    /// Send a message or start an interactive chat session.
    Chat(commands::chat::ChatArgs),

    /// Execute a named job from the configuration file.
    #[command(name = "run-job")]
    RunJob(commands::run_job::RunJobArgs),

    /// Manage persistent sessions (list, show, delete).
    Session(commands::session::SessionArgs),
}

/// Run the CLI with the parsed arguments.
///
/// Returns the process exit code.
pub async fn run(cli: Cli) -> i32 {
    runtime::init_tracing(cli.verbose, cli.quiet);

    let mode = if cli.json {
        OutputMode::Json
    } else {
        OutputMode::Plain
    };
    let output = OutputWriter::new(mode);

    let config_path = cli.config.as_ref();
    let provider = cli.provider.as_deref();

    let result = match &cli.command {
        Commands::Heartbeat(args) => {
            commands::heartbeat::execute(args, &output, config_path, provider).await
        }
        Commands::Chat(args) => commands::chat::execute(args, &output, config_path, provider).await,
        Commands::RunJob(args) => {
            commands::run_job::execute(args, &output, config_path, provider).await
        }
        Commands::Session(args) => {
            commands::session::execute(args, &output, config_path, provider).await
        }
    };

    match result {
        Ok(()) => exit_code::SUCCESS,
        Err(err) => {
            let _ = output.error(&err.to_string());
            err.exit_code()
        }
    }
}
