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
#[command(
    about = "Send a message or start an interactive chat session.",
    long_about = "Send a single message, pipe input, or open an interactive \
                  REPL with persistent per-session history.\n\n\
                  \x20 EXAMPLES\n\
                  \x20   Single-shot:    acton-ai chat -m \"what is rust?\"\n\
                  \x20   From file:      acton-ai chat < question.txt\n\
                  \x20   From pipe:      git log | acton-ai chat -m \"summarize\"\n\
                  \x20   Interactive:    acton-ai chat\n\
                  \x20   Resume session: acton-ai chat --session work\n\n\
                  \x20 JSON OUTPUT\n\
                  \x20   With --json, single-shot responses are emitted as one line:\n\
                  \x20     {\"schemaVersion\":1,\"session\":\"main\",\"role\":\"assistant\",\n\
                  \x20      \"text\":\"...\",\"tokenCount\":42}\n\
                  \x20   Schema is versioned — consumers should branch on schemaVersion."
)]
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

/// Versioned JSON response envelope for `--json` mode. Bumping
/// `schema_version` is the signal to consumers that the shape has changed.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatResponse {
    /// Schema version for this envelope. Start at 1 — bump on breaking
    /// changes (renamed fields, removed fields, changed types).
    schema_version: u32,
    session: String,
    role: String,
    text: String,
    token_count: usize,
}

/// Current `--json` envelope version.
const CHAT_RESPONSE_SCHEMA_VERSION: u32 = 1;

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
        let available = persistence::list_sessions(&conn)
            .await
            .map(|rows| rows.into_iter().map(|s| s.name).collect())
            .unwrap_or_default();
        return Err(CliError::session_not_found(&session_name, available));
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
                        schema_version: CHAT_RESPONSE_SCHEMA_VERSION,
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
            // per-session history, slash commands, styled labels + spinner.
            let history_path = crate::cli::chat_ui::history_path_for_session(&session_name);
            let theme = crate::cli::chat_ui::style::Theme::resolve();
            let provider_name = rt.ai.default_provider_name().to_string();
            let provider_count = rt.ai.provider_names().count();
            let banner_info = crate::cli::chat_ui::banner::BannerInfo {
                session: &session_name,
                origin: session_origin,
                provider: &provider_name,
                history_len: conv.len(),
                provider_count,
            };
            crate::cli::chat_ui::banner::print_startup(&banner_info, output, &theme);

            // Per-turn persistence: each finished turn (and its tool-result
            // messages) is flushed inside the REPL. A Ctrl+C or crash
            // therefore loses at most the current in-flight turn.
            let persist = crate::cli::chat_ui::PersistCtx {
                conn: &conn,
                conversation_id: &conversation_id,
                session_name: &session_name,
            };

            let started_at = std::time::Instant::now();
            let run_result =
                crate::cli::chat_ui::run(&conv, &rt.ai, history_path, Some(persist)).await;

            // The REPL flushes every turn as it completes, so by the time we
            // land here the DB is already up to date. `touch_session` is
            // idempotent and cheap — call it once more to bump the
            // last-used timestamp on clean exit.
            persistence::touch_session(&conn, &session_name).await?;
            let history_len = conv.history().len();

            if run_result.is_ok() {
                crate::cli::chat_ui::banner::print_exit_summary(
                    &session_name,
                    history_len,
                    started_at.elapsed(),
                    output,
                );
            }
            run_result?;
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
