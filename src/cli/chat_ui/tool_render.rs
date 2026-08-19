//! Inline rendering of tool calls during a streaming chat turn.
//!
//! The LLM provider broadcasts `LLMStreamToolCall` events via the acton
//! broker as the model decides to call a tool. The chat-UI actor subscribes
//! to these and prints a dim, bracketed line so users can see exactly which
//! tools ran on their behalf — a clig.dev "transparency" win and a
//! safety-relevant signal for destructive tools (bash, write_file, edit_file).

use crate::cli::chat_ui::style::Theme;
use crate::messages::ToolCall;

/// Maximum printed width of the args preview line before truncation.
const ARGS_PREVIEW_MAX: usize = 120;

/// Render a single tool-call event to stdout as a dim-bracketed line.
///
/// Inserts a leading newline so the line visually separates from any token
/// text already on the current line.
pub fn render_tool_call(tool: &ToolCall, theme: &Theme) {
    let args_preview = args_preview(&tool.arguments);
    println!(
        "\n{dim}[{name}]{reset} {dim}{args}{reset}",
        dim = theme.dim_open,
        reset = theme.dim_close,
        name = sanitize_line(&tool.name),
        args = args_preview,
    );
}

/// Render the outcome of a tool call (success or error) as an indented
/// follow-up line to the preceding `[tool]` invocation. Successes render
/// dim; errors render in the theme's warn colour so they stand out.
pub fn render_tool_result(tool_name: &str, success: bool, summary: &str, theme: &Theme) {
    let tool_name = sanitize_line(tool_name);
    let summary = sanitize_line(summary);
    if success {
        println!(
            "  {dim}↳ ok{reset}  {dim}{summary}{reset}",
            dim = theme.dim_open,
            reset = theme.dim_close,
            summary = summary,
        );
    } else {
        let reset = if theme.colors_enabled { "\x1b[0m" } else { "" };
        println!(
            "  {warn}↳ error{reset}  {dim}{tool_name}: {summary}{dim_close}",
            warn = theme.warn_open,
            reset = reset,
            dim = theme.dim_open,
            dim_close = theme.dim_close,
            tool_name = tool_name,
            summary = summary,
        );
    }
}

/// Produce a one-line preview of tool arguments for inline rendering.
///
/// For the common cases (`{ command: "..." }`, `{ path: "..." }`) this picks
/// the single most informative field. Falls back to a truncated JSON
/// rendering for less-structured argument payloads.
fn args_preview(args: &serde_json::Value) -> String {
    let object = match args.as_object() {
        Some(obj) => obj,
        None => return truncate(&args.to_string(), ARGS_PREVIEW_MAX),
    };

    // Common single-field shortcuts — show just the salient value.
    for key in ["command", "path", "pattern", "query", "expression", "url"] {
        if let Some(value) = object.get(key).and_then(|v| v.as_str()) {
            return truncate(value, ARGS_PREVIEW_MAX);
        }
    }

    // Generic fallback: compact JSON.
    let rendered = serde_json::to_string(args).unwrap_or_else(|_| "<args>".to_string());
    truncate(&rendered, ARGS_PREVIEW_MAX)
}

/// Neutralise characters that would let tool-supplied text rewrite or spoof
/// this line. These lines are the user's only in-band view of which tools ran
/// with which arguments, so a `\r` or an ANSI escape smuggled through a tool
/// name, argument, or result summary could display a command other than the
/// one that executed. C0/C1 controls become U+FFFD; Unicode bidi overrides,
/// which can visually reorder a command, are treated the same way.
fn sanitize_line(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\n' | '\t' => ' ',
            c if c.is_control() => char::REPLACEMENT_CHARACTER,
            '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' => {
                char::REPLACEMENT_CHARACTER
            }
            c => c,
        })
        .collect()
}

fn truncate(s: &str, max: usize) -> String {
    let flat = sanitize_line(s);
    if flat.chars().count() <= max {
        flat
    } else {
        let cut: String = flat.chars().take(max).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn preview_prefers_command_field() {
        assert_eq!(
            args_preview(&json!({ "command": "wc -l src/**/*.rs" })),
            "wc -l src/**/*.rs"
        );
    }

    #[test]
    fn preview_prefers_path_field() {
        assert_eq!(
            args_preview(&json!({ "path": "src/main.rs" })),
            "src/main.rs"
        );
    }

    #[test]
    fn preview_falls_back_to_compact_json() {
        let got = args_preview(&json!({ "foo": 1, "bar": "baz" }));
        assert!(got.contains("foo"));
        assert!(got.contains("bar"));
    }

    #[test]
    fn preview_truncates_long_values() {
        let long = "a".repeat(500);
        let got = args_preview(&json!({ "command": long }));
        assert!(got.chars().count() <= ARGS_PREVIEW_MAX + 1); // +1 for ellipsis
        assert!(got.ends_with('…'));
    }

    #[test]
    fn preview_flattens_newlines() {
        let got = args_preview(&json!({ "command": "line1\nline2" }));
        assert_eq!(got, "line1 line2");
    }

    #[test]
    fn preview_neutralises_ansi_escapes() {
        let got = args_preview(&json!({ "command": "safe\u{001b}[2K\rrm -rf /" }));
        assert!(!got.contains('\u{1b}'));
        assert!(!got.contains('\r'));
        assert_eq!(got, "safe\u{fffd}[2K\u{fffd}rm -rf /");
    }

    #[test]
    fn sanitize_neutralises_c1_and_bidi_overrides() {
        // C1 CSI (0x9B) is interpreted as an escape introducer by some
        // terminals; U+202E reverses the visual order of what follows.
        let got = sanitize_line("a\u{9b}b\u{202E}c");
        assert_eq!(got, "a\u{fffd}b\u{fffd}c");
    }

    #[test]
    fn sanitize_preserves_plain_unicode() {
        assert_eq!(sanitize_line("café → 東京"), "café → 東京");
    }
}
