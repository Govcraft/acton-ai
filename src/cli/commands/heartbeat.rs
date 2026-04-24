//! The `heartbeat` command — autonomous wake-up cycle.
//!
//! Triggered by a systemd timer (or cron), the heartbeat reviews stored
//! heartbeat entries, references session context, executes due tasks
//! autonomously, and outputs a JSON activity report.

use crate::cli::error::CliError;
use crate::cli::output::OutputWriter;
use crate::cli::runtime::CliRuntime;
use crate::memory::persistence;
use serde::Serialize;
use std::path::PathBuf;

/// Options for the heartbeat command.
#[derive(Debug, clap::Args)]
pub struct HeartbeatArgs {
    /// Only process entries for this session.
    #[arg(long)]
    pub session: Option<String>,
}

/// Result of executing a single heartbeat entry.
#[derive(Debug, Serialize)]
struct EntryResult {
    /// The heartbeat entry ID that was processed.
    entry_id: String,
    /// The session this entry belongs to.
    session_name: String,
    /// The task summary from the entry.
    summary: String,
    /// Whether execution succeeded: `"ok"` or `"error"`.
    status: String,
    /// The LLM response text (if successful).
    #[serde(skip_serializing_if = "Option::is_none")]
    response_text: Option<String>,
    /// Number of tokens consumed.
    token_count: usize,
    /// Error message (if execution failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Top-level JSON report emitted by the heartbeat command.
#[derive(Debug, Serialize)]
struct HeartbeatReport {
    /// ISO 8601 timestamp of when the heartbeat ran.
    timestamp: String,
    /// How many entries were processed.
    entries_processed: usize,
    /// Per-entry results.
    results: Vec<EntryResult>,
}

/// Execute the heartbeat command.
pub async fn execute(
    args: &HeartbeatArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
    provider_override: Option<&str>,
) -> Result<(), CliError> {
    let rt = CliRuntime::new(config_path, provider_override, &[]).await?;
    let conn = rt.connection().await?;

    // Query due entries — either all or filtered by session.
    let entries = if let Some(ref session_name) = args.session {
        persistence::list_entries_for_session(&conn, session_name).await?
    } else {
        persistence::list_due_entries(&conn).await?
    };

    let _ = output.status(&format!("heartbeat: {} due entries found", entries.len()));

    let mut results = Vec::with_capacity(entries.len());

    for entry in &entries {
        let result = process_entry(&rt, &conn, entry, output).await;
        results.push(result);
    }

    let report = HeartbeatReport {
        timestamp: chrono::Utc::now().to_rfc3339(),
        entries_processed: results.len(),
        results,
    };

    output.write_json(&report)?;

    rt.shutdown().await?;
    Ok(())
}

/// Build the system prompt for a heartbeat task execution.
fn build_system_prompt(entry_summary: &str, history_context: &str) -> String {
    format!(
        "You are an autonomous agent executing a scheduled heartbeat task.\n\
         \n\
         ## Task\n\
         {entry_summary}\n\
         \n\
         ## Conversation Context\n\
         {history_context}\n\
         \n\
         ## Instructions\n\
         - Execute the task described above using the tools available to you.\n\
         - Be concise and report what you did.\n\
         - If the task cannot be completed, explain why."
    )
}

/// Format conversation history into a context string for the system prompt.
fn format_history_context(messages: &[crate::messages::Message]) -> String {
    if messages.is_empty() {
        return "(no prior conversation history)".to_string();
    }

    // Include the most recent messages to stay within reasonable context limits.
    const MAX_HISTORY_MESSAGES: usize = 20;
    let start = messages.len().saturating_sub(MAX_HISTORY_MESSAGES);
    let recent = &messages[start..];

    let mut context = String::new();
    for msg in recent {
        let role = match msg.role {
            crate::messages::MessageRole::System => "system",
            crate::messages::MessageRole::User => "user",
            crate::messages::MessageRole::Assistant => "assistant",
            crate::messages::MessageRole::Tool => "tool",
        };
        context.push_str(&format!("[{role}]: {}\n", msg.content));
    }
    context
}

/// Process a single heartbeat entry: resolve session, load context, execute via LLM,
/// and update the database.
async fn process_entry(
    rt: &CliRuntime,
    conn: &libsql::Connection,
    entry: &persistence::HeartbeatEntry,
    output: &OutputWriter,
) -> EntryResult {
    let _ = output.status(&format!(
        "  processing entry {} (session: {})",
        entry.id, entry.session_name
    ));

    // Attempt the full execution pipeline; capture errors as entry-level failures.
    match execute_entry(rt, conn, entry).await {
        Ok((response_text, token_count)) => {
            // Update the database based on the schedule.
            let db_result = update_entry_status(conn, entry).await;
            if let Err(e) = db_result {
                let _ = output.error(&format!(
                    "failed to update entry {} after run: {e}",
                    entry.id
                ));
            }

            EntryResult {
                entry_id: entry.id.clone(),
                session_name: entry.session_name.clone(),
                summary: entry.summary.clone(),
                status: "ok".to_string(),
                response_text: Some(response_text),
                token_count,
                error: None,
            }
        }
        Err(e) => EntryResult {
            entry_id: entry.id.clone(),
            session_name: entry.session_name.clone(),
            summary: entry.summary.clone(),
            status: "error".to_string(),
            response_text: None,
            token_count: 0,
            error: Some(e.to_string()),
        },
    }
}

/// Execute the LLM prompt for a single heartbeat entry.
///
/// Returns the response text and token count on success.
async fn execute_entry(
    rt: &CliRuntime,
    conn: &libsql::Connection,
    entry: &persistence::HeartbeatEntry,
) -> Result<(String, usize), CliError> {
    // Resolve session to get the conversation ID.
    let session_info = match persistence::resolve_session(conn, &entry.session_name).await? {
        Some(info) => info,
        None => {
            let available = persistence::list_sessions(conn)
                .await
                .map(|rows| rows.into_iter().map(|s| s.name).collect())
                .unwrap_or_default();
            return Err(CliError::session_not_found(
                &entry.session_name,
                available,
            ));
        }
    };

    // Load conversation history for context.
    let messages =
        persistence::load_conversation_messages(conn, &session_info.conversation_id).await?;

    let history_context = format_history_context(&messages);
    let system_prompt = build_system_prompt(&entry.summary, &history_context);

    // The user-facing message is the entry summary itself.
    let response = rt
        .ai
        .prompt(&entry.summary)
        .system(system_prompt)
        .collect()
        .await?;

    Ok((response.text, response.token_count))
}

/// Update the heartbeat entry in the database after a successful run.
async fn update_entry_status(
    conn: &libsql::Connection,
    entry: &persistence::HeartbeatEntry,
) -> Result<(), CliError> {
    let is_once = entry
        .schedule
        .as_deref()
        .is_some_and(|s| s.eq_ignore_ascii_case("once"));

    if is_once {
        persistence::complete_entry(conn, &entry.id).await?;
    } else {
        // Recurring: mark as run, clear next_due so it becomes due again next cycle.
        persistence::update_entry_after_run(conn, &entry.id, None).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{Message, MessageRole};

    #[test]
    fn build_system_prompt_contains_task_and_context() {
        let prompt = build_system_prompt("Check disk usage", "user: hi\nassistant: hello");
        assert!(prompt.contains("Check disk usage"));
        assert!(prompt.contains("user: hi"));
        assert!(prompt.contains("autonomous agent"));
    }

    #[test]
    fn format_history_context_empty() {
        let context = format_history_context(&[]);
        assert_eq!(context, "(no prior conversation history)");
    }

    #[test]
    fn format_history_context_formats_roles() {
        let messages = vec![
            Message {
                role: MessageRole::User,
                content: "hello".to_string(),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: MessageRole::Assistant,
                content: "hi there".to_string(),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let context = format_history_context(&messages);
        assert!(context.contains("[user]: hello"));
        assert!(context.contains("[assistant]: hi there"));
    }

    #[test]
    fn format_history_context_truncates_long_history() {
        let messages: Vec<Message> = (0..30)
            .map(|i| Message {
                role: MessageRole::User,
                content: format!("message {i}"),
                tool_calls: None,
                tool_call_id: None,
            })
            .collect();

        let context = format_history_context(&messages);
        // Should not contain early messages (0-9), should contain recent ones (10-29).
        assert!(!context.contains("message 0\n"));
        assert!(context.contains("message 29"));
    }

    #[test]
    fn heartbeat_report_serializes_to_json() {
        let report = HeartbeatReport {
            timestamp: "2026-03-18T12:00:00+00:00".to_string(),
            entries_processed: 1,
            results: vec![EntryResult {
                entry_id: "hb_123".to_string(),
                session_name: "main".to_string(),
                summary: "Check disk".to_string(),
                status: "ok".to_string(),
                response_text: Some("Disk is fine.".to_string()),
                token_count: 42,
                error: None,
            }],
        };

        let json = serde_json::to_string(&report).expect("serialization should succeed");
        assert!(json.contains("\"entries_processed\":1"));
        assert!(json.contains("\"status\":\"ok\""));
        // error field should be absent due to skip_serializing_if
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn entry_result_error_variant_includes_error_field() {
        let result = EntryResult {
            entry_id: "hb_456".to_string(),
            session_name: "test".to_string(),
            summary: "Run backup".to_string(),
            status: "error".to_string(),
            response_text: None,
            token_count: 0,
            error: Some("session not found".to_string()),
        };

        let json = serde_json::to_string(&result).expect("serialization should succeed");
        assert!(json.contains("\"error\":\"session not found\""));
        assert!(!json.contains("\"response_text\""));
    }
}
