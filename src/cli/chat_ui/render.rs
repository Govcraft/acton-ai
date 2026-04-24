//! Buffered markdown rendering for `--render` mode.
//!
//! Streaming tokens one at a time makes inline markdown rendering (code
//! fences, lists, headings) nearly impossible without heavy buffering. The
//! pragmatic compromise here: when `--render` is enabled, the chat loop
//! buffers the full response and renders it via [`termimad`] once the stream
//! ends. Users see the spinner throughout the turn, then the formatted
//! result. Tool-call lines still render inline via the existing
//! `LLMStreamToolCall` handler, so long tool-heavy turns don't look hung.
//!
//! When stdout is piped or colours are disabled the renderer falls back to
//! plain text, so scripts consuming `acton-ai chat --render > file.md` still
//! get raw markdown on disk.

use crate::cli::output::OutputWriter;

/// Render markdown text to stdout. Falls back to writing the text verbatim
/// when stdout isn't a terminal or colours are off, so pipes and
/// `NO_COLOR=1` keep clean output.
pub fn render_to_stdout(markdown: &str) {
    if !OutputWriter::stdout_is_tty() || !OutputWriter::use_colors() {
        // Plain write — no ANSI, no reflow. Matches what a user piping to
        // `less -R` or a file expects.
        println!("{markdown}");
        return;
    }

    let skin = termimad::MadSkin::default_dark();
    skin.print_text(markdown);
}
