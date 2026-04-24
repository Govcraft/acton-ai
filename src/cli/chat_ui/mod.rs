//! Interactive terminal UX for `acton-ai chat`.
//!
//! Owns the REPL: line editing + persistent history (via `reedline`), slash
//! commands, and streaming display. The library-level
//! [`Conversation::run_chat_with`] remains a thin stdin/stdout loop for
//! programmatic use; the CLI drives this richer path via the public
//! [`Conversation::send_streaming`] API.

pub mod slash;

use std::io::Write;
use std::path::{Path, PathBuf};

use acton_reactive::prelude::*;
use reedline::{DefaultPrompt, DefaultPromptSegment, FileBackedHistory, Reedline, Signal};

use crate::conversation::{Conversation, StreamToken};
use crate::error::ActonAIError;
use crate::facade::ActonAI;
use crate::messages::Message;

/// Number of lines kept in the persistent reedline history file.
const HISTORY_SIZE: usize = 1000;

/// Resolve the on-disk reedline history file for the given session, creating
/// parent directories if necessary. Returns `None` when the platform has no
/// stable data directory.
#[must_use]
pub fn history_path_for_session(session: &str) -> Option<PathBuf> {
    let data_dir = dirs::data_dir()?;
    let dir = data_dir.join("acton-ai").join("history");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(format!("{}.txt", sanitize_session_for_path(session))))
}

/// Replace any character that is not alphanumeric, dash, or underscore with
/// `_` so a session name is safe to use as a file-name fragment.
fn sanitize_session_for_path(session: &str) -> String {
    session
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Token-rendering actor used by the CLI chat loop. Currently just writes each
/// token to stdout; follow-up PRs extend it with spinner signalling, tool-call
/// visibility, and markdown rendering.
#[derive(Default, Debug)]
struct ChatUiActor;

/// Run the interactive chat REPL.
///
/// Drives a reedline-based input loop, dispatches slash commands, and streams
/// responses through the supplied [`Conversation`]. Returns when the user
/// exits via `/exit`, `Ctrl+D`, or the model calls the exit tool.
///
/// # Errors
///
/// Returns any error raised while streaming an LLM response or while driving
/// the reedline read-loop.
pub async fn run(
    conv: &Conversation,
    ai: &ActonAI,
    history_path: Option<PathBuf>,
) -> Result<(), ActonAIError> {
    // Token-rendering actor. Same pattern as the previous inline printer in
    // `Conversation::run_chat_with`, lifted here so follow-up PRs can hang
    // spinner / tool-render hooks off the same actor without touching
    // library code.
    let mut actor_runtime = ai.runtime().clone();
    let mut token_actor = actor_runtime.new_actor::<ChatUiActor>();
    token_actor.mutate_on::<StreamToken>(|_actor, ctx| {
        print!("{}", ctx.message().text);
        std::io::stdout().flush().ok();
        Reply::ready()
    });
    let token_handle = token_actor.start().await;

    let mut editor = build_editor(history_path.as_deref());
    let left_prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic("You".to_string()),
        DefaultPromptSegment::Empty,
    );

    let mut last_user_message: Option<String> = None;

    let result = loop {
        // reedline's `read_line` is blocking — move it to a dedicated thread
        // so it doesn't pin a tokio worker. We pass ownership in and back so
        // history state persists across turns.
        let prompt = left_prompt.clone();
        let read = tokio::task::spawn_blocking(move || {
            let sig = editor.read_line(&prompt);
            (editor, sig)
        })
        .await;

        let (returned, sig) = match read {
            Ok(pair) => pair,
            Err(join_err) => {
                break Err(ActonAIError::prompt_failed(format!(
                    "reedline read thread panicked: {join_err}"
                )));
            }
        };
        editor = returned;

        let line = match sig {
            Ok(Signal::Success(buffer)) => buffer,
            Ok(Signal::CtrlC) => {
                // At the prompt, Ctrl+C is not "exit" — hint and continue.
                eprintln!("(type /exit or press Ctrl+D to quit)");
                continue;
            }
            Ok(Signal::CtrlD) => break Ok(()),
            Ok(other) => {
                // `Signal` is `#[non_exhaustive]`; future variants land here.
                tracing::debug!(?other, "reedline returned unhandled signal — continuing");
                continue;
            }
            Err(e) => {
                break Err(ActonAIError::prompt_failed(format!(
                    "reedline read_line failed: {e}"
                )));
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match slash::parse(trimmed) {
            slash::SlashAction::NotSlash => {
                last_user_message = Some(trimmed.to_string());
                send_turn(conv, &token_handle, trimmed).await?;
                if conv.should_exit() {
                    break Ok(());
                }
            }
            slash::SlashAction::Help => println!("{}", slash::HELP_TEXT),
            slash::SlashAction::Clear => {
                conv.clear();
                last_user_message = None;
                println!("(history cleared)");
            }
            slash::SlashAction::History => print_history(&conv.history()),
            slash::SlashAction::Exit => break Ok(()),
            slash::SlashAction::Retry => match last_user_message.clone() {
                Some(prev) => {
                    send_turn(conv, &token_handle, &prev).await?;
                    if conv.should_exit() {
                        break Ok(());
                    }
                }
                None => println!("(no previous message to retry)"),
            },
            slash::SlashAction::Unknown(name) => {
                println!("unknown command: {name}. Type /help for a list.");
            }
        }
    };

    let _ = token_handle.stop().await;
    result
}

/// Print the assistant prefix, run one streaming turn, close with a newline.
async fn send_turn(
    conv: &Conversation,
    token_handle: &ActorHandle,
    content: &str,
) -> Result<(), ActonAIError> {
    print!("Assistant: ");
    std::io::stdout().flush().ok();
    conv.send_streaming(content, token_handle).await?;
    println!();
    Ok(())
}

fn build_editor(history_path: Option<&Path>) -> Reedline {
    let editor = Reedline::create();
    let Some(path) = history_path else {
        return editor;
    };
    match FileBackedHistory::with_file(HISTORY_SIZE, path.to_path_buf()) {
        Ok(history) => editor.with_history(Box::new(history)),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "failed to open reedline history file — continuing without persistence",
            );
            editor
        }
    }
}

fn print_history(messages: &[Message]) {
    if messages.is_empty() {
        println!("(no messages yet)");
        return;
    }
    for (i, msg) in messages.iter().enumerate() {
        println!("  [{i}] {}: {}", msg.role, preview(&msg.content, 400));
    }
}

fn preview(s: &str, max: usize) -> String {
    let compact: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    if compact.chars().count() <= max {
        compact
    } else {
        let cut: String = compact.chars().take(max).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_session_rejects_path_separators() {
        assert_eq!(sanitize_session_for_path("main"), "main");
        assert_eq!(sanitize_session_for_path("foo/bar"), "foo_bar");
        assert_eq!(sanitize_session_for_path("../etc"), "___etc");
        assert_eq!(sanitize_session_for_path("work-1_2"), "work-1_2");
    }

    #[test]
    fn preview_truncates_and_flattens_newlines() {
        assert_eq!(preview("hello\nworld", 80), "hello world");
        let long: String = "a".repeat(500);
        let got = preview(&long, 10);
        assert_eq!(got.chars().count(), 11); // 10 chars + ellipsis
        assert!(got.ends_with('…'));
    }
}
