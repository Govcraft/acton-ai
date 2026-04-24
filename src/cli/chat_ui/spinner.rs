//! Spinner for the "waiting on first token" window of a chat turn.
//!
//! Creates an [`indicatif::ProgressBar`] configured to draw on stderr, so
//! stdout (assistant text) stays pipe-friendly. When colours are disabled or
//! stderr is not a terminal we hand back a hidden progress bar that is cheap
//! to tick and display nothing — the chat loop stays identical either way.

use std::time::Duration;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::cli::output::OutputWriter;

/// Build a spinner for the current turn.
///
/// The spinner draws to stderr so piping `acton-ai chat` keeps stdout clean.
/// When stderr is not a TTY or colours are disabled the returned bar is
/// hidden — the rest of the loop calls `finish_and_clear` unconditionally.
#[must_use]
pub fn build(message: impl Into<String>) -> ProgressBar {
    if !OutputWriter::use_colors() {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr());
    pb.set_style(
        ProgressStyle::with_template("{spinner} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "  "]),
    );
    pb.set_message(message.into());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

/// Stop and fully erase a spinner. Safe to call on a hidden bar.
pub fn clear(pb: &ProgressBar) {
    pb.finish_and_clear();
}
