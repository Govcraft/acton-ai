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
        name = tool.name,
        args = args_preview,
    );
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

fn truncate(s: &str, max: usize) -> String {
    // Flatten embedded newlines so a single line of args stays a single line.
    let flat: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
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
}
