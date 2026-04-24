//! Resolved terminal theme for the interactive chat REPL.
//!
//! `Theme::resolve()` consults [`OutputWriter::use_colors`] once at REPL
//! startup and pre-renders the ANSI-wrapped label strings the loop needs.
//! When colors are disabled (piped output, `NO_COLOR=1`, non-TTY stderr) the
//! fields collapse to plain text so the chat loop can emit them unconditionally
//! without re-checking the gate on every turn.

use crate::cli::output::OutputWriter;

// ANSI escape constants. Kept private so all styling decisions funnel through
// [`Theme::resolve`] — callers should never hand-concatenate colour codes.
const RESET: &str = "\x1b[0m";
const BOLD_MAGENTA: &str = "\x1b[1;35m";
const BOLD_CYAN: &str = "\x1b[1;36m";
const BOLD_RED: &str = "\x1b[1;31m";
const DIM: &str = "\x1b[2m";

/// Pre-rendered label strings used by the chat REPL.
#[derive(Debug, Clone)]
pub struct Theme {
    /// "Assistant: " printed before a streamed response.
    pub assistant_label: String,
    /// "You" reedline prompt segment (used in [`DefaultPromptSegment::Basic`]).
    pub user_prompt_label: String,
    /// Opening escape for dim-gray inline accents. Empty when colours are off.
    pub dim_open: &'static str,
    /// Closing reset. Empty when colours are off.
    pub dim_close: &'static str,
    /// Opening escape for bold-cyan inline accents. Empty when colours are off.
    pub accent_open: &'static str,
    /// Opening escape for bold-red. Empty when colours are off.
    pub warn_open: &'static str,
    /// Whether colours are enabled — useful for the spinner gate.
    pub colors_enabled: bool,
}

impl Theme {
    /// Resolve theme from the ambient environment (TTY + `NO_COLOR`).
    #[must_use]
    pub fn resolve() -> Self {
        if OutputWriter::use_colors() {
            Self {
                assistant_label: format!("{BOLD_MAGENTA}Assistant:{RESET} "),
                // reedline draws the left prompt itself, so we keep it plain
                // for now — reedline's own styling layer handles decoration.
                user_prompt_label: "You".to_string(),
                dim_open: DIM,
                dim_close: RESET,
                accent_open: BOLD_CYAN,
                warn_open: BOLD_RED,
                colors_enabled: true,
            }
        } else {
            Self::plain()
        }
    }

    /// Plain-text theme used for piped output, `NO_COLOR`, and tests.
    #[must_use]
    pub fn plain() -> Self {
        Self {
            assistant_label: "Assistant: ".to_string(),
            user_prompt_label: "You".to_string(),
            dim_open: "",
            dim_close: "",
            accent_open: "",
            warn_open: "",
            colors_enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_theme_has_no_escape_codes() {
        let t = Theme::plain();
        assert_eq!(t.assistant_label, "Assistant: ");
        assert_eq!(t.dim_open, "");
        assert_eq!(t.dim_close, "");
        assert!(!t.colors_enabled);
    }
}
