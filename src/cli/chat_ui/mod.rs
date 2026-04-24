//! Interactive terminal UX for `acton-ai chat`.
//!
//! Owns the REPL: line editing + persistent history (via `reedline`), slash
//! commands, streaming display, spinner, and (styled) speaker labels. The
//! library-level [`Conversation::run_chat_with`] remains a thin stdin/stdout
//! loop for programmatic use; the CLI drives this richer path via the public
//! [`Conversation::send_streaming`] API.

pub mod banner;
pub mod slash;
pub mod spinner;
pub mod style;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use acton_reactive::prelude::*;
use indicatif::ProgressBar;
use reedline::{DefaultPrompt, DefaultPromptSegment, FileBackedHistory, Reedline, Signal};

use crate::conversation::{Conversation, StreamToken};
use crate::error::ActonAIError;
use crate::facade::ActonAI;
use crate::messages::Message;

use self::style::Theme;

/// Number of lines kept in the persistent reedline history file.
const HISTORY_SIZE: usize = 1000;

/// Shared handle used to hand a live spinner to the token actor for the
/// current turn. The actor takes the spinner out on the first streamed token
/// and clears it; the run loop takes it out on stream completion (covering
/// tools-only / empty-response turns).
type SpinnerSlot = Arc<Mutex<Option<ProgressBar>>>;

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

/// Token-rendering actor state. Holds the slot the spinner is parked in so
/// the token handler can dismiss it on the first streamed token.
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
    let theme = Theme::resolve();
    let spinner_slot: SpinnerSlot = Arc::new(Mutex::new(None));

    // Token-rendering actor. Same pattern as the previous inline printer in
    // `Conversation::run_chat_with`, lifted here so the spinner can be
    // dismissed the moment the first token arrives.
    let mut actor_runtime = ai.runtime().clone();
    let mut token_actor = actor_runtime.new_actor::<ChatUiActor>();
    let slot_for_actor = spinner_slot.clone();
    token_actor.mutate_on::<StreamToken>(move |_actor, ctx| {
        // Mutex contention is limited to the first token per turn — on
        // subsequent tokens `.take()` returns None and the lock is released
        // immediately.
        if let Some(pb) = take_spinner(&slot_for_actor) {
            pb.finish_and_clear();
        }
        print!("{}", ctx.message().text);
        std::io::stdout().flush().ok();
        Reply::ready()
    });
    let token_handle = token_actor.start().await;

    let mut editor = build_editor(history_path.as_deref());
    let left_prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic(theme.user_prompt_label.clone()),
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
                eprintln!(
                    "{}(type /exit or press Ctrl+D to quit){}",
                    theme.dim_open, theme.dim_close
                );
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
                send_turn(conv, &token_handle, trimmed, &theme, &spinner_slot).await?;
                if conv.should_exit() {
                    break Ok(());
                }
            }
            slash::SlashAction::Help => println!("{}", slash::HELP_TEXT),
            slash::SlashAction::Clear => {
                conv.clear();
                last_user_message = None;
                println!(
                    "{}(history cleared){}",
                    theme.dim_open, theme.dim_close
                );
            }
            slash::SlashAction::History => print_history(&conv.history(), &theme),
            slash::SlashAction::Exit => break Ok(()),
            slash::SlashAction::Retry => match last_user_message.clone() {
                Some(prev) => {
                    send_turn(conv, &token_handle, &prev, &theme, &spinner_slot).await?;
                    if conv.should_exit() {
                        break Ok(());
                    }
                }
                None => println!(
                    "{}(no previous message to retry){}",
                    theme.dim_open, theme.dim_close
                ),
            },
            slash::SlashAction::Unknown(name) => {
                println!(
                    "{warn}unknown command: {name}{reset}. Type /help for a list.",
                    warn = theme.warn_open,
                    reset = if theme.colors_enabled { "\x1b[0m" } else { "" },
                );
            }
        }
    };

    let _ = token_handle.stop().await;
    result
}

/// Print the assistant prefix, show a spinner until the first token arrives,
/// stream the response, and close with a newline.
async fn send_turn(
    conv: &Conversation,
    token_handle: &ActorHandle,
    content: &str,
    theme: &Theme,
    spinner_slot: &SpinnerSlot,
) -> Result<(), ActonAIError> {
    print!("{}", theme.assistant_label);
    std::io::stdout().flush().ok();

    // Park a spinner in the shared slot so the token handler can dismiss it
    // the moment the first token lands.
    let pb = spinner::build("thinking…");
    if let Ok(mut slot) = spinner_slot.lock() {
        *slot = Some(pb.clone());
    }

    let result = conv.send_streaming(content, token_handle).await;

    // Empty responses (errors, tools-only turns) never emit a `StreamToken`
    // so the slot can still be occupied. Clear any leftover spinner here.
    if let Some(pb) = take_spinner(spinner_slot) {
        pb.finish_and_clear();
    }

    result?;
    println!();
    Ok(())
}

fn take_spinner(slot: &SpinnerSlot) -> Option<ProgressBar> {
    // Recover from a poisoned lock rather than panicking — the data it
    // protects is an `Option<ProgressBar>` and no invariants can be broken.
    match slot.lock() {
        Ok(mut guard) => guard.take(),
        Err(poisoned) => poisoned.into_inner().take(),
    }
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

fn print_history(messages: &[Message], theme: &Theme) {
    if messages.is_empty() {
        println!(
            "{}(no messages yet){}",
            theme.dim_open, theme.dim_close
        );
        return;
    }
    for (i, msg) in messages.iter().enumerate() {
        println!(
            "  {dim}[{i}] {role}:{reset} {text}",
            dim = theme.dim_open,
            reset = theme.dim_close,
            role = msg.role,
            text = preview(&msg.content, 400),
        );
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
