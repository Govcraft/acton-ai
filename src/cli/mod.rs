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
    #[command(
        long_about = "Send a single message, pipe input, or open an interactive \
                            REPL with persistent per-session history.\n\n\
                            EXAMPLES:\n  \
                              Single-shot:    acton-ai chat -m \"what is rust?\"\n  \
                              From file:      acton-ai chat < question.txt\n  \
                              From pipe:      git log | acton-ai chat -m \"summarize\"\n  \
                              Interactive:    acton-ai chat\n  \
                              Resume session: acton-ai chat --session work\n\n\
                            JSON OUTPUT:\n  \
                              With --json, single-shot responses are one line:\n    \
                              {\"schemaVersion\":1,\"session\":\"main\",\"role\":\"assistant\",\n     \
                               \"text\":\"...\",\"tokenCount\":42}\n  \
                              Schema is versioned — consumers should branch on schemaVersion."
    )]
    Chat(commands::chat::ChatArgs),

    /// Execute a named job from the configuration file.
    #[command(name = "run-job")]
    RunJob(commands::run_job::RunJobArgs),

    /// Manage persistent sessions (list, show, delete).
    Session(commands::session::SessionArgs),

    /// Show the resolved configuration file and effective values.
    #[command(long_about = "Display which configuration file the CLI loaded (or \
                            was overridden to load via --config), the full search \
                            order with a marker on the matching path, the database \
                            path, and every effective value after merging config \
                            and CLI overrides. API keys set directly in the TOML \
                            are redacted; keys sourced from environment variables \
                            show only the variable name and whether it's set.")]
    Config(commands::config::ConfigArgs),

    /// Ask a running acton-ai process what it is doing.
    #[command(
        long_about = "Connect to a running process over its introspection socket \
                            and print what it is doing right now: admission state, \
                            turns and tool calls in flight, provider circuit-breaker \
                            health, MCP server generations, and usage totals.\n\n\
                            The socket is found from --socket, then the `[introspection] \
                            socket_path` config key, then the default runtime directory. \
                            A process only listens when it has been configured to."
    )]
    Status(commands::introspect::StatusArgs),

    /// Tell a running process to stop admitting new turns.
    #[command(
        long_about = "Stop a running process admitting new turns. Turns already \
                            running are never interrupted, and callers that try to start \
                            one get a refusal naming `acton-ai resume` as the way back."
    )]
    Pause(commands::introspect::PauseArgs),

    /// Tell a paused or draining process to admit turns again.
    Resume(commands::introspect::ResumeArgs),

    /// Stop admitting new turns and report when the last one finishes.
    #[command(
        long_about = "Stop a running process admitting new turns and report how \
                            many are still running. With --wait, keep reporting until \
                            the last one finishes, which is what makes this safe to put \
                            in a systemd ExecStop or a deploy script."
    )]
    Drain(commands::introspect::DrainArgs),
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
    let output = OutputWriter::new(mode).with_quiet(cli.quiet);

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
        Commands::Config(args) => commands::config::execute(args, &output, config_path, provider),
        Commands::Status(args) => {
            commands::introspect::execute_status(args, &output, config_path).await
        }
        Commands::Pause(args) => {
            commands::introspect::execute_pause(args, &output, config_path).await
        }
        Commands::Resume(args) => {
            commands::introspect::execute_resume(args, &output, config_path).await
        }
        Commands::Drain(args) => {
            commands::introspect::execute_drain(args, &output, config_path).await
        }
    };

    match result {
        Ok(()) => exit_code::SUCCESS,
        Err(err) => {
            let hint = err.hint();
            let _ = output.error_with_hint(&err.to_string(), hint.as_deref());
            err.exit_code()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_tree_is_internally_consistent() {
        // clap's own audit: duplicate long flags, conflicting short flags, and
        // ill-formed `global = true` propagation are all compile-clean and only
        // panic at parse time, so this runs the check up front.
        Cli::command().debug_assert();
    }

    #[test]
    fn the_introspection_commands_are_reachable_by_their_documented_names() {
        // These four strings appear in the module docs, the README, and every
        // systemd unit anyone writes. Renaming a subcommand is a breaking
        // change to an interface with no compiler to catch it.
        for name in ["status", "pause", "resume", "drain"] {
            let cli = Cli::try_parse_from(["acton-ai", name]).expect("{name} should parse");
            let parsed = matches!(
                (&cli.command, name),
                (Commands::Status(_), "status")
                    | (Commands::Pause(_), "pause")
                    | (Commands::Resume(_), "resume")
                    | (Commands::Drain(_), "drain")
            );
            assert!(parsed, "`acton-ai {name}` dispatched to the wrong command");
        }
    }

    #[test]
    fn the_socket_flag_is_accepted_by_every_introspection_command() {
        for name in ["status", "pause", "resume", "drain"] {
            Cli::try_parse_from(["acton-ai", name, "--socket", "/run/a.sock"])
                .unwrap_or_else(|e| panic!("`{name} --socket` must parse: {e}"));
        }
    }

    #[test]
    fn drain_defaults_to_not_waiting() {
        let cli = Cli::try_parse_from(["acton-ai", "drain"]).expect("parses");
        let Commands::Drain(args) = cli.command else {
            panic!("expected drain");
        };
        // The default has to be the non-blocking one: a bare `drain` in an
        // ExecStop that silently blocked for five minutes would be a very
        // surprising deploy.
        assert!(!args.wait);
        assert_eq!(
            args.timeout,
            commands::introspect::DEFAULT_DRAIN_TIMEOUT_SECS
        );
    }

    #[test]
    fn drain_accepts_an_explicit_wait_and_timeout() {
        let cli =
            Cli::try_parse_from(["acton-ai", "drain", "--wait", "--timeout", "0"]).expect("parses");
        let Commands::Drain(args) = cli.command else {
            panic!("expected drain");
        };
        assert!(args.wait);
        // Zero is the documented "wait indefinitely" value, so it must survive
        // parsing rather than being rejected as out of range.
        assert_eq!(args.timeout, 0);
    }

    #[test]
    fn the_global_json_flag_reaches_the_introspection_commands() {
        // `--json` is declared once, globally. A subcommand added without
        // thought could shadow it, and scripted `status --json` is the main
        // consumer of this whole surface.
        let cli = Cli::try_parse_from(["acton-ai", "status", "--json"]).expect("parses");
        assert!(cli.json);
    }
}
