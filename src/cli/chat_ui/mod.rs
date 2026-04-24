//! Interactive terminal UX for `acton-ai chat`.
//!
//! Owns the REPL: line editing + persistent history (via `reedline`), slash
//! commands, streaming display, spinner, and (styled) speaker labels. The
//! library-level [`Conversation::run_chat_with`] remains a thin stdin/stdout
//! loop for programmatic use; the CLI drives this richer path via the public
//! [`Conversation::send_streaming`] API.

pub mod banner;
pub mod render;
pub mod slash;
pub mod spinner;
pub mod style;
pub mod tool_render;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use acton_reactive::prelude::*;
use indicatif::ProgressBar;
use libsql::Connection;
use reedline::{DefaultPrompt, DefaultPromptSegment, FileBackedHistory, Reedline, Signal};

use crate::conversation::{Conversation, StreamToken};
use crate::error::ActonAIError;
use crate::facade::ActonAI;
use crate::memory::persistence;
use crate::messages::{LLMStreamToolCall, Message};
use crate::types::ConversationId;

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

/// Per-session persistence handles passed to [`run`]. When present, each turn
/// is flushed to the database as it completes so a Ctrl+C or crash loses at
/// most the in-flight turn.
pub struct PersistCtx<'a> {
    pub conn: &'a Connection,
    pub conversation_id: &'a ConversationId,
    pub session_name: &'a str,
}

/// Options bag for [`run`]. Passing a struct keeps the signature stable as
/// new toggles land in follow-up PRs.
pub struct RunOptions<'a> {
    pub history_path: Option<PathBuf>,
    pub persist: Option<PersistCtx<'a>>,
    /// When true, buffer each turn's response and render it as markdown
    /// once streaming completes. Token-by-token display is suppressed —
    /// the spinner carries the "still working" signal instead.
    pub render_markdown: bool,
}

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
    options: RunOptions<'_>,
) -> Result<(), ActonAIError> {
    let RunOptions {
        history_path,
        persist,
        render_markdown,
    } = options;

    let theme = Theme::resolve();
    let spinner_slot: SpinnerSlot = Arc::new(Mutex::new(None));
    // When the user cancels a turn with Ctrl+C we flip this to silence the
    // remaining token and tool-call renders from the still-running stream.
    // The bool is cleared at the top of each turn.
    let muted: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    // Flipped true the first time a token lands for a turn so the "Assistant:"
    // label is printed exactly once — and only when the model actually
    // produced text. Empty / errored turns leave no orphan label.
    let label_printed: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

    // Chat-UI actor.
    //
    // Receives three kinds of events:
    //   * `StreamToken` (sent directly by the conversation's send_streaming
    //     path) — writes the token to stdout and dismisses any active
    //     spinner on first arrival.
    //   * `LLMStreamToolCall` (broadcast by the LLM provider) — renders a
    //     dim bracketed line so the user sees which tools ran.
    let mut actor_runtime = ai.runtime().clone();
    let mut token_actor = actor_runtime.new_actor::<ChatUiActor>();

    let slot_for_actor = spinner_slot.clone();
    let muted_for_token = muted.clone();
    let label_for_actor = theme.assistant_label.clone();
    let label_printed_for_actor = label_printed.clone();
    token_actor.mutate_on::<StreamToken>(move |_actor, ctx| {
        if muted_for_token.load(Ordering::Relaxed) {
            return Reply::ready();
        }
        // Mutex contention is limited to the first token per turn — on
        // subsequent tokens `.take()` returns None and the lock is released
        // immediately.
        if let Some(pb) = take_spinner(&slot_for_actor) {
            pb.finish_and_clear();
        }
        // Print the "Assistant:" label once per turn, right before the first
        // token. Deferring the label keeps the spinner on a clean line and
        // avoids an orphan label when a turn ends without any tokens.
        if !label_printed_for_actor.swap(true, Ordering::Relaxed) {
            print!("{label_for_actor}");
        }
        print!("{}", ctx.message().text);
        std::io::stdout().flush().ok();
        Reply::ready()
    });

    let theme_for_tool = theme.clone();
    let slot_for_tool = spinner_slot.clone();
    let muted_for_tool = muted.clone();
    token_actor.mutate_on::<LLMStreamToolCall>(move |_actor, ctx| {
        if muted_for_tool.load(Ordering::Relaxed) {
            return Reply::ready();
        }
        // Stop the spinner (tools fire after initial tokens may have streamed
        // or, for tools-first turns, before any tokens) so the inline line
        // isn't overlapped by the spinner animation.
        if let Some(pb) = take_spinner(&slot_for_tool) {
            pb.finish_and_clear();
        }
        tool_render::render_tool_call(&ctx.message().tool_call, &theme_for_tool);
        Reply::ready()
    });

    // Subscribe to broadcast events BEFORE starting. `StreamToken` is sent
    // point-to-point via the handle returned below, so it doesn't need a
    // broadcast subscription.
    token_actor
        .handle()
        .subscribe::<LLMStreamToolCall>()
        .await;

    let token_handle = token_actor.start().await;

    let mut editor = build_editor(history_path.as_deref());
    let left_prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic(theme.user_prompt_label.clone()),
        DefaultPromptSegment::Empty,
    );

    let mut last_user_message: Option<String> = None;
    // Index of the next history message to flush. Reset to 0 on /clear.
    let mut persist_cursor: usize = conv.len();

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

        let turn_ctx = TurnContext {
            conv,
            token_handle: &token_handle,
            theme: &theme,
            spinner_slot: &spinner_slot,
            muted: &muted,
            label_printed: &label_printed,
            render_markdown,
        };

        match slash::parse(trimmed) {
            slash::SlashAction::NotSlash => {
                last_user_message = Some(trimmed.to_string());
                let outcome = send_turn(&turn_ctx, trimmed).await?;
                flush_new_messages(conv, &persist, &mut persist_cursor).await?;
                if matches!(outcome, TurnOutcome::Canceled) {
                    // Stream aborted by Ctrl+C — loop back to the prompt.
                    continue;
                }
                if conv.should_exit() {
                    break Ok(());
                }
            }
            slash::SlashAction::Help => println!("{}", slash::HELP_TEXT),
            slash::SlashAction::Clear => {
                conv.clear();
                last_user_message = None;
                // The DB still has historical rows; truncating would be a
                // destructive surprise. New messages simply start fresh.
                persist_cursor = 0;
                println!(
                    "{}(history cleared){}",
                    theme.dim_open, theme.dim_close
                );
            }
            slash::SlashAction::History => print_history(&conv.history(), &theme),
            slash::SlashAction::Exit => break Ok(()),
            slash::SlashAction::Retry => match last_user_message.clone() {
                Some(prev) => {
                    let outcome = send_turn(&turn_ctx, &prev).await?;
                    flush_new_messages(conv, &persist, &mut persist_cursor).await?;
                    if matches!(outcome, TurnOutcome::Canceled) {
                        continue;
                    }
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

/// Outcome of a single chat turn, distinguishing normal completion from a
/// user-initiated cancel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnOutcome {
    Completed,
    Canceled,
}

/// Print the assistant prefix, show a spinner until the first token arrives,
/// stream the response (racing against `Ctrl+C`), and close with a newline.
///
/// In `render_markdown` mode, tokens aren't displayed as they arrive — the
/// spinner carries the waiting signal, and the full response is rendered
/// through `termimad` when the stream ends.
///
/// Returns whether the turn completed normally or was canceled mid-stream.
/// All per-turn state that `send_turn` needs. Bundled into one struct so the
/// signature stays tidy as follow-up UX toggles land.
struct TurnContext<'a> {
    conv: &'a Conversation,
    token_handle: &'a ActorHandle,
    theme: &'a Theme,
    spinner_slot: &'a SpinnerSlot,
    muted: &'a Arc<AtomicBool>,
    label_printed: &'a Arc<AtomicBool>,
    render_markdown: bool,
}

async fn send_turn(
    ctx: &TurnContext<'_>,
    content: &str,
) -> Result<TurnOutcome, ActonAIError> {
    // Reset per-turn signals so the token handler emits the label again and
    // any previous cancel no longer silences output.
    ctx.muted.store(false, Ordering::Relaxed);
    ctx.label_printed.store(false, Ordering::Relaxed);

    // Park a spinner in the shared slot. In streaming mode the token handler
    // dismisses it on first token (and prints the "Assistant:" label); in
    // render mode it stays up until we clear it ourselves post-stream.
    let pb = spinner::build("thinking…");
    if let Ok(mut slot) = ctx.spinner_slot.lock() {
        *slot = Some(pb.clone());
    }

    // Race the stream against a SIGINT. Reedline releases raw-mode at the
    // prompt, so Ctrl+C during streaming delivers a real SIGINT that
    // `tokio::signal::ctrl_c()` resolves on.
    //
    // In render mode we use the non-streaming `send` path so tokens never
    // reach the actor (nothing to display live). Tool-call events still
    // broadcast and render inline via the existing subscriber.
    let outcome = if ctx.render_markdown {
        tokio::select! {
            res = ctx.conv.send(content) => TurnSelectResult::Done(res),
            _ = tokio::signal::ctrl_c() => {
                ctx.muted.store(true, Ordering::Relaxed);
                TurnSelectResult::Canceled
            }
        }
    } else {
        tokio::select! {
            res = ctx.conv.send_streaming(content, ctx.token_handle) => TurnSelectResult::Done(res),
            _ = tokio::signal::ctrl_c() => {
                ctx.muted.store(true, Ordering::Relaxed);
                TurnSelectResult::Canceled
            }
        }
    };

    // Empty responses (errors, tools-only turns, canceled) may not emit a
    // `StreamToken` so the slot can still be occupied. Clear any leftover.
    if let Some(pb) = take_spinner(ctx.spinner_slot) {
        pb.finish_and_clear();
    }

    match outcome {
        TurnSelectResult::Done(res) => {
            let response = res?;
            if ctx.render_markdown {
                // Render header + markdown as one block. Same bold-magenta
                // label as streaming mode keeps the UX recognisable.
                print!("{}", ctx.theme.assistant_label);
                std::io::stdout().flush().ok();
                println!();
                render::render_to_stdout(&response.text);
            } else {
                println!();
            }
            Ok(TurnOutcome::Completed)
        }
        TurnSelectResult::Canceled => {
            eprintln!(
                "\n{}^C canceled{}",
                ctx.theme.warn_open,
                if ctx.theme.colors_enabled { "\x1b[0m" } else { "" },
            );
            Ok(TurnOutcome::Canceled)
        }
    }
}

/// Intermediate enum used inside `send_turn` — kept private so the public
/// surface only exposes the broader `TurnOutcome`.
enum TurnSelectResult {
    Done(Result<crate::stream::CollectedResponse, ActonAIError>),
    Canceled,
}

/// Persist messages added since the last flush. No-op when no persist
/// context was supplied (e.g. tests driving the loop directly).
async fn flush_new_messages(
    conv: &Conversation,
    persist: &Option<PersistCtx<'_>>,
    cursor: &mut usize,
) -> Result<(), ActonAIError> {
    let Some(ctx) = persist.as_ref() else {
        return Ok(());
    };
    let history = conv.history();
    if history.len() <= *cursor {
        return Ok(());
    }
    for msg in &history[*cursor..] {
        persistence::save_message(ctx.conn, ctx.conversation_id, msg)
            .await
            .map_err(|e| ActonAIError::prompt_failed(format!("failed to save message: {e}")))?;
    }
    persistence::touch_session(ctx.conn, ctx.session_name)
        .await
        .map_err(|e| ActonAIError::prompt_failed(format!("failed to touch session: {e}")))?;
    *cursor = history.len();
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
