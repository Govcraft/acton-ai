//! The `chat` command — send messages and manage conversations.
//!
//! Supports single-shot messages (`--message`), stdin piping, and
//! interactive terminal chat with session persistence. Accepts a repeatable
//! `--skill-dir` flag that loads agent skills into the runtime; the model
//! can then discover them via `list_skills` / `activate_skill`.

use crate::cli::error::CliError;
use crate::cli::output::{OutputMode, OutputWriter};
use crate::cli::runtime::CliRuntime;
use crate::conversation::DEFAULT_SYSTEM_PROMPT;
use crate::memory::persistence;
use crate::messages::Message;
use crate::types::AgentId;
use serde::Serialize;
use std::io::{self, Read as _};
use std::path::PathBuf;

/// Options for the chat command.
#[derive(Debug, clap::Args)]
pub struct ChatArgs {
    /// Session name to use (default: "main").
    #[arg(long, env = "ACTON_SESSION")]
    pub session: Option<String>,

    /// Message to send (reads from stdin if omitted and not a TTY).
    #[arg(short, long)]
    pub message: Option<String>,

    /// System prompt override.
    #[arg(long)]
    pub system: Option<String>,

    /// Create the session if it doesn't exist.
    #[arg(long)]
    pub create: bool,

    /// Disable streaming output (collect full response before printing).
    #[arg(long)]
    pub no_stream: bool,

    /// Skill file or directory to load (repeatable).
    ///
    /// Each occurrence appends one path. Paths may point at a single `.md`
    /// skill or a directory that's scanned recursively. Appends to any
    /// `[skills] paths = [...]` entries in the config file.
    #[arg(long = "skill-dir", short = 's', value_name = "PATH")]
    pub skill_dirs: Vec<PathBuf>,
}

/// JSON response envelope for `--json` mode.
#[derive(Serialize)]
struct ChatResponse {
    session: String,
    role: String,
    text: String,
    token_count: usize,
}

/// Execute the chat command.
pub async fn execute(
    args: &ChatArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
    provider_override: Option<&str>,
) -> Result<(), CliError> {
    // Pre-validate skill paths so a typo surfaces as a friendly CliError
    // instead of the framework's generic Configuration error at launch.
    for path in &args.skill_dirs {
        if !path.exists() {
            return Err(CliError::configuration(format!(
                "--skill-dir path not found: {}",
                path.display()
            )));
        }
    }

    let rt = CliRuntime::new(config_path, provider_override, &args.skill_dirs).await?;

    // Determine session name
    let session_name = args.session.clone().unwrap_or_else(|| "main".to_string());

    // Resolve or create session
    let conn = rt.connection().await?;
    let session = persistence::resolve_session(&conn, &session_name).await?;

    let (conversation_id, system_prompt, session_origin) = if let Some(info) = session {
        (info.conversation_id, info.system_prompt, "resumed")
    } else if args.create || session_name == "main" {
        // Auto-create for "main" or when --create is specified
        let agent_id = AgentId::new();
        let system = args.system.as_deref().unwrap_or(DEFAULT_SYSTEM_PROMPT);
        let conv_id =
            persistence::create_session(&conn, &session_name, &agent_id, Some(system)).await?;
        (conv_id, Some(system.to_string()), "created")
    } else {
        return Err(CliError::session_not_found(&session_name));
    };

    // Load conversation history
    let history = persistence::load_conversation_messages(&conn, &conversation_id).await?;

    // Determine system prompt — CLI flag wins, then whatever was persisted
    // on the session, then the library-canonical default.
    let (system, system_source) = if let Some(s) = args.system.clone() {
        (s, "cli-flag")
    } else if let Some(s) = system_prompt {
        (s, "session-persisted")
    } else {
        (DEFAULT_SYSTEM_PROMPT.to_string(), "default")
    };

    tracing::info!(
        session = %session_name,
        origin = %session_origin,
        conversation_id = %conversation_id,
        history_len = history.len(),
        "chat session resolved",
    );
    tracing::info!(
        source = %system_source,
        len = system.len(),
        prompt = %system,
        "system prompt",
    );

    // Build the Conversation. The exit tool is always enabled for the CLI
    // because the interactive REPL below depends on the model being able to
    // terminate the session via the `exit_conversation` tool.
    let conv = rt
        .ai
        .conversation()
        .system(&system)
        .restore(history)
        .with_exit_tool()
        .build()
        .await;

    // Determine message source
    let message = resolve_message(args)?;

    match message {
        Some(msg) => {
            tracing::info!(
                mode = "single-shot",
                len = msg.len(),
                preview = %preview(&msg, 120),
                "sending message",
            );
            // Single-shot mode: send one message and output response
            let response = conv.send(&msg).await?;

            // Persist user message and assistant response
            let user_msg = Message::user(&msg);
            let assistant_msg = Message::assistant(&response.text);
            persistence::save_message(&conn, &conversation_id, &user_msg).await?;
            persistence::save_message(&conn, &conversation_id, &assistant_msg).await?;
            persistence::touch_session(&conn, &session_name).await?;

            // Output
            match output.mode() {
                OutputMode::Json => {
                    output.write_json(&ChatResponse {
                        session: session_name,
                        role: "assistant".to_string(),
                        text: response.text,
                        token_count: response.token_count,
                    })?;
                }
                OutputMode::Plain => {
                    output.write_line(&response.text)?;
                }
            }
        }
        None => {
            // Interactive mode — only valid when stdin is a TTY
            if !OutputWriter::stdin_is_tty() {
                return Err(CliError::no_input());
            }

            tracing::info!(mode = "interactive", "entering chat loop");
            // Drive the richer CLI REPL: reedline line editor, persistent
            // per-session history, and slash commands.
            let history_path = crate::cli::chat_ui::history_path_for_session(&session_name);
            crate::cli::chat_ui::run(&conv, &rt.ai, history_path).await?;

            // After interactive chat, persist any new messages
            let current_history = conv.history();
            for msg in current_history {
                persistence::save_message(&conn, &conversation_id, &msg).await?;
            }
            persistence::touch_session(&conn, &session_name).await?;
        }
    }

    rt.shutdown().await?;
    Ok(())
}

/// Truncate `s` to `max` chars for log previews, appending `…` when cut.
fn preview(s: &str, max: usize) -> String {
    let compact = s.replace('\n', " ");
    if compact.chars().count() <= max {
        compact
    } else {
        let cut: String = compact.chars().take(max).collect();
        format!("{cut}…")
    }
}

/// Resolve the user's message from --message flag or stdin.
///
/// Returns:
/// - `Some(msg)` if a message was provided via flag or stdin pipe
/// - `None` if no message and stdin is a TTY (interactive mode)
fn resolve_message(args: &ChatArgs) -> Result<Option<String>, CliError> {
    // Explicit --message flag takes priority
    if let Some(ref msg) = args.message {
        return Ok(Some(msg.clone()));
    }

    // If stdin is not a TTY, read from it (piped input)
    if !OutputWriter::stdin_is_tty() {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(CliError::no_input());
        }
        return Ok(Some(trimmed.to_string()));
    }

    // stdin is a TTY and no --message → interactive mode
    Ok(None)
}
