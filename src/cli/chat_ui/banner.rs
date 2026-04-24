//! Startup banner and exit summary for the interactive chat REPL.
//!
//! Output goes to stderr so stdout remains a clean stream of assistant text
//! suitable for piping. Both helpers no-op when stderr is not a terminal or
//! colours are disabled via `NO_COLOR` — matching the ambient gating the rest
//! of the chat UX uses.

use std::time::Duration;

use crate::cli::chat_ui::style::Theme;
use crate::cli::output::OutputWriter;

/// Compact package version pulled from the crate so the banner stays accurate
/// without touching build scripts.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Information the banner needs from the caller. Kept as plain data so the
/// banner remains trivially testable and free of runtime dependencies.
#[derive(Debug, Clone)]
pub struct BannerInfo<'a> {
    pub session: &'a str,
    /// "created" or "resumed" — matches the log origin used at
    /// `src/cli/commands/chat.rs:87-96`.
    pub origin: &'a str,
    pub provider: &'a str,
    /// Number of messages already in the conversation history.
    pub history_len: usize,
    /// Total number of providers configured (1 when only the default is set).
    pub provider_count: usize,
}

/// Print the banner on stderr. Suppressed when stderr is not a TTY, `NO_COLOR`
/// is set, or the caller has asked for quiet output.
pub fn print_startup(info: &BannerInfo<'_>, output: &OutputWriter, theme: &Theme) {
    if output.is_quiet() || !OutputWriter::use_colors() {
        return;
    }

    let accent = theme.accent_open;
    let dim_on = theme.dim_open;
    let dim_off = theme.dim_close;
    let reset = if theme.colors_enabled { "\x1b[0m" } else { "" };

    eprintln!(
        "{accent}acton-ai {VERSION}{reset}  {dim_on}|{dim_off}  \
         session={bold}{session}{reset} ({origin}, {len} msg){dim_on}  |  {dim_off}\
         provider={bold}{provider}{reset}{extra}",
        bold = accent,
        session = info.session,
        origin = info.origin,
        len = info.history_len,
        provider = info.provider,
        extra = if info.provider_count > 1 {
            format!("{dim_on} (+{} more){dim_off}", info.provider_count - 1)
        } else {
            String::new()
        },
    );
    eprintln!(
        "{dim_on}Type {dim_off}/help{dim_on} for commands, Ctrl+D to exit.{dim_off}"
    );
}

/// Print the exit summary on stderr. Shown on clean exit, suppressed when
/// quiet or stderr is not a TTY.
pub fn print_exit_summary(
    session: &str,
    history_len: usize,
    elapsed: Duration,
    output: &OutputWriter,
) {
    if output.is_quiet() || !OutputWriter::use_colors() {
        return;
    }

    eprintln!(
        "\x1b[2mSession '{session}' saved ({history_len} messages, {dur}). \
         Resume with: acton-ai chat --session {session}\x1b[0m",
        dur = format_duration(elapsed),
    );
}

/// Human-friendly duration rendering for the exit summary.
fn format_duration(d: Duration) -> String {
    let total = d.as_secs();
    if total < 60 {
        format!("{total}s")
    } else if total < 3600 {
        format!("{}m{:02}s", total / 60, total % 60)
    } else {
        format!("{}h{:02}m", total / 3600, (total % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_under_a_minute_renders_seconds() {
        assert_eq!(format_duration(Duration::from_secs(0)), "0s");
        assert_eq!(format_duration(Duration::from_secs(42)), "42s");
    }

    #[test]
    fn duration_minutes_zero_pad_seconds() {
        assert_eq!(format_duration(Duration::from_secs(60)), "1m00s");
        assert_eq!(format_duration(Duration::from_secs(135)), "2m15s");
    }

    #[test]
    fn duration_hours_are_rendered() {
        assert_eq!(format_duration(Duration::from_secs(3600)), "1h00m");
        assert_eq!(format_duration(Duration::from_secs(3720)), "1h02m");
    }
}
