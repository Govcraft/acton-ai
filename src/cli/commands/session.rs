//! The `session` command — manage persistent sessions.
//!
//! Subcommands for listing, inspecting, and deleting sessions.

use crate::cli::error::CliError;
use crate::cli::output::{OutputMode, OutputWriter};
use crate::cli::runtime::resolve_db_path;
use crate::memory::persistence::{
    self, initialize_schema, list_sessions, load_conversation_messages, open_database,
    resolve_session, SessionInfo,
};
use crate::memory::PersistenceConfig;
use crate::messages::Message;
use libsql::Connection;
use serde::Serialize;
use std::path::PathBuf;

/// Maximum number of recent messages to display in `session show`.
const SHOW_MESSAGE_LIMIT: usize = 10;

/// Session management subcommands.
#[derive(Debug, clap::Args)]
pub struct SessionArgs {
    #[command(subcommand)]
    pub command: SessionCommand,
}

/// Available session subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum SessionCommand {
    /// List all sessions.
    List,
    /// Show details for a session.
    Show {
        /// Session name.
        name: String,
    },
    /// Delete a session and its conversation history.
    Delete {
        /// Session name.
        name: String,
        /// Skip confirmation prompt.
        #[arg(long)]
        force: bool,
    },
}

/// JSON envelope for `session show`.
#[derive(Serialize)]
struct SessionShowResponse {
    #[serde(flatten)]
    session: SessionInfo,
    recent_messages: Vec<MessageSummary>,
}

/// Compact message representation for JSON output.
#[derive(Serialize)]
struct MessageSummary {
    role: String,
    content: String,
}

impl From<&Message> for MessageSummary {
    fn from(msg: &Message) -> Self {
        Self {
            role: msg.role.to_string(),
            content: msg.content.clone(),
        }
    }
}

/// Execute the session command.
pub async fn execute(
    args: &SessionArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
    _provider_override: Option<&str>,
) -> Result<(), CliError> {
    let conn = open_session_db(config_path).await?;

    match &args.command {
        SessionCommand::List => execute_list(&conn, output).await,
        SessionCommand::Show { name } => execute_show(&conn, output, name).await,
        SessionCommand::Delete { name, force } => execute_delete(&conn, output, name, *force).await,
    }
}

/// Open a database connection without bootstrapping the full LLM runtime.
async fn open_session_db(config_path: Option<&PathBuf>) -> Result<Connection, CliError> {
    let db_path = resolve_db_path(config_path);
    let config = PersistenceConfig::new(&db_path);
    let db = open_database(&config).await?;
    let conn = db
        .connect()
        .map_err(|e| CliError::configuration(format!("failed to connect to database: {e}")))?;
    initialize_schema(&conn).await?;
    Ok(conn)
}

/// List all sessions.
async fn execute_list(conn: &Connection, output: &OutputWriter) -> Result<(), CliError> {
    let sessions = list_sessions(conn).await?;

    if sessions.is_empty() {
        output.status("No sessions found.")?;
        return Ok(());
    }

    match output.mode() {
        OutputMode::Json => {
            output.write_json(&sessions)?;
        }
        OutputMode::Plain => {
            output.write_line(&format!(
                "{:<20} {:>8}  {:<20} {:<20}",
                "NAME", "MESSAGES", "CREATED", "LAST ACTIVE"
            ))?;
            output.write_line(&"-".repeat(72))?;
            for s in &sessions {
                output.write_line(&format!(
                    "{:<20} {:>8}  {:<20} {:<20}",
                    s.name, s.message_count, s.created_at, s.last_active
                ))?;
            }
        }
    }

    Ok(())
}

/// Show details for a single session.
async fn execute_show(
    conn: &Connection,
    output: &OutputWriter,
    name: &str,
) -> Result<(), CliError> {
    let session = resolve_session(conn, name)
        .await?
        .ok_or_else(|| {
            // `ok_or_else` is sync, so we can't run the async list_sessions
            // here. Pass an empty available list — the hint still tells the
            // user how to list sessions.
            CliError::session_not_found(name, Vec::new())
        })?;

    // Load recent messages
    let all_messages = load_conversation_messages(conn, &session.conversation_id).await?;
    let recent: Vec<&Message> = all_messages
        .iter()
        .rev()
        .take(SHOW_MESSAGE_LIMIT)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    match output.mode() {
        OutputMode::Json => {
            let response = SessionShowResponse {
                session,
                recent_messages: recent.iter().map(|m| MessageSummary::from(*m)).collect(),
            };
            output.write_json(&response)?;
        }
        OutputMode::Plain => {
            output.write_line(&format!("Session:      {}", session.name))?;
            output.write_line(&format!("Agent ID:     {}", session.agent_id))?;
            output.write_line(&format!("Conversation: {}", session.conversation_id))?;
            if let Some(ref prompt) = session.system_prompt {
                output.write_line(&format!("System:       {prompt}"))?;
            }
            output.write_line(&format!("Created:      {}", session.created_at))?;
            output.write_line(&format!("Last active:  {}", session.last_active))?;
            output.write_line(&format!("Messages:     {}", session.message_count))?;

            if !recent.is_empty() {
                output.write_line("")?;
                output.write_line(&format!("--- Last {} messages ---", recent.len()))?;
                for msg in &recent {
                    let role = msg.role.to_string();
                    // Truncate long content for display
                    let content = truncate_for_display(&msg.content, 120);
                    output.write_line(&format!("[{role}] {content}"))?;
                }
            }
        }
    }

    Ok(())
}

/// Delete a session after verifying it exists.
async fn execute_delete(
    conn: &Connection,
    output: &OutputWriter,
    name: &str,
    force: bool,
) -> Result<(), CliError> {
    // Verify the session exists
    let _session = resolve_session(conn, name)
        .await?
        .ok_or_else(|| {
            // `ok_or_else` is sync, so we can't run the async list_sessions
            // here. Pass an empty available list — the hint still tells the
            // user how to list sessions.
            CliError::session_not_found(name, Vec::new())
        })?;

    if !force {
        output.error(&format!(
            "refusing to delete session '{name}' without --force"
        ))?;
        return Err(CliError::configuration(format!(
            "use --force to delete session '{name}'"
        )));
    }

    persistence::delete_session(conn, name).await?;
    output.status(&format!("Deleted session '{name}'."))?;

    Ok(())
}

/// Truncate a string for terminal display, appending "..." if truncated.
fn truncate_for_display(s: &str, max_chars: usize) -> String {
    // Replace newlines with spaces for single-line display
    let single_line: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    if single_line.chars().count() <= max_chars {
        single_line
    } else {
        let mut truncated: String = single_line.chars().take(max_chars).collect();
        truncated.push_str("...");
        truncated
    }
}
