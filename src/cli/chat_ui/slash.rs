//! Slash-command parsing for the interactive chat REPL.
//!
//! `parse` is a pure function — easy to unit test — that classifies a user
//! input line as either a regular prompt (`SlashAction::NotSlash`) or one of
//! the built-in commands. The REPL loop is responsible for applying the
//! side effect; this module only decides.

/// The action a slash command maps to.
#[derive(Debug, PartialEq, Eq)]
pub enum SlashAction<'a> {
    /// Line is not a slash command — treat as a normal prompt.
    NotSlash,
    /// `/help` or `/?`.
    Help,
    /// `/clear` — clear conversation history.
    Clear,
    /// `/history` — list recent messages.
    History,
    /// `/retry` — resend the previous user message.
    Retry,
    /// `/exit`, `/quit`, or `/q`.
    Exit,
    /// Unrecognized `/something` — name is the raw first token.
    Unknown(&'a str),
}

/// Classify a user-entered line as a slash command or a regular message.
///
/// Leading whitespace is tolerated. Anything not starting with `/` returns
/// `NotSlash`. The first whitespace-delimited token decides the action.
pub fn parse(input: &str) -> SlashAction<'_> {
    let trimmed = input.trim_start();
    if !trimmed.starts_with('/') {
        return SlashAction::NotSlash;
    }
    let cmd = trimmed.split_whitespace().next().unwrap_or("/");
    match cmd {
        "/help" | "/?" => SlashAction::Help,
        "/clear" => SlashAction::Clear,
        "/history" => SlashAction::History,
        "/retry" => SlashAction::Retry,
        "/exit" | "/quit" | "/q" => SlashAction::Exit,
        other => SlashAction::Unknown(other),
    }
}

/// User-facing help text for the slash command set.
pub const HELP_TEXT: &str = "Commands:\n  \
     /help, /?         show this help\n  \
     /clear            clear the conversation history\n  \
     /history          list messages in the current session\n  \
     /retry            resend the previous user message\n  \
     /exit, /quit, /q  end the session\n\n\
     Tip: press Ctrl+D at the prompt to exit.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_slash_line_is_treated_as_message() {
        assert_eq!(parse("hello"), SlashAction::NotSlash);
        assert_eq!(parse(""), SlashAction::NotSlash);
        assert_eq!(parse("what is /tmp?"), SlashAction::NotSlash);
    }

    #[test]
    fn parses_basic_commands() {
        assert_eq!(parse("/help"), SlashAction::Help);
        assert_eq!(parse("/?"), SlashAction::Help);
        assert_eq!(parse("/clear"), SlashAction::Clear);
        assert_eq!(parse("/history"), SlashAction::History);
        assert_eq!(parse("/retry"), SlashAction::Retry);
        assert_eq!(parse("/exit"), SlashAction::Exit);
        assert_eq!(parse("/quit"), SlashAction::Exit);
        assert_eq!(parse("/q"), SlashAction::Exit);
    }

    #[test]
    fn tolerates_leading_whitespace() {
        assert_eq!(parse("   /help"), SlashAction::Help);
    }

    #[test]
    fn extra_args_are_ignored_at_parse_time() {
        // `/clear --force` still resolves to Clear; the dispatcher decides
        // whether to consume the args.
        assert_eq!(parse("/clear --force"), SlashAction::Clear);
        assert_eq!(parse("/history 10"), SlashAction::History);
    }

    #[test]
    fn unknown_slash_returns_unknown_with_command_name() {
        match parse("/foo bar") {
            SlashAction::Unknown(name) => assert_eq!(name, "/foo"),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }
}
