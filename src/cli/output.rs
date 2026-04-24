//! Output formatting for the CLI.
//!
//! Handles JSON vs plain text output, NO_COLOR support, and stdout/stderr separation.

use serde::Serialize;
use std::io::{self, IsTerminal, Write};

/// Output mode for CLI responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Human-readable plain text output.
    Plain,
    /// Machine-readable JSON output (one JSON object per line).
    Json,
}

/// Writer that respects output mode and stream separation.
///
/// - LLM response text goes to stdout.
/// - All diagnostics and progress go to stderr.
pub struct OutputWriter {
    mode: OutputMode,
    quiet: bool,
}

impl OutputWriter {
    /// Creates a new output writer with the given mode. Non-quiet by default.
    #[must_use]
    pub fn new(mode: OutputMode) -> Self {
        Self {
            mode,
            quiet: false,
        }
    }

    /// Returns the output mode.
    #[must_use]
    pub fn mode(&self) -> OutputMode {
        self.mode
    }

    /// Mark this writer as quiet — commands should suppress banners,
    /// spinners, and other chrome on stderr. The `error()` channel still
    /// surfaces (failures are never silent).
    #[must_use]
    pub fn with_quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    /// Whether quiet mode is in effect.
    #[must_use]
    pub fn is_quiet(&self) -> bool {
        self.quiet
    }

    /// Returns true if stdout is a terminal.
    #[must_use]
    pub fn stdout_is_tty() -> bool {
        io::stdout().is_terminal()
    }

    /// Returns true if stdin is a terminal.
    #[must_use]
    pub fn stdin_is_tty() -> bool {
        io::stdin().is_terminal()
    }

    /// Returns true if colors should be used (respects NO_COLOR env var).
    #[must_use]
    pub fn use_colors() -> bool {
        std::env::var("NO_COLOR").is_err() && io::stderr().is_terminal()
    }

    /// Write a response to stdout.
    ///
    /// In JSON mode, serializes the value as a JSON line.
    /// In plain mode, writes the text directly.
    pub fn write_response(&self, text: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        match self.mode {
            OutputMode::Plain => {
                write!(stdout, "{text}")?;
                stdout.flush()
            }
            OutputMode::Json => {
                // In JSON mode, response text is part of a larger envelope
                // written via write_json
                write!(stdout, "{text}")?;
                stdout.flush()
            }
        }
    }

    /// Write a complete line to stdout (adds newline).
    pub fn write_line(&self, text: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{text}")?;
        stdout.flush()
    }

    /// Write a JSON value to stdout as a single line.
    pub fn write_json<T: Serialize>(&self, value: &T) -> io::Result<()> {
        let json = serde_json::to_string(value)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{json}")?;
        stdout.flush()
    }

    /// Write a status/diagnostic message to stderr.
    pub fn status(&self, msg: &str) -> io::Result<()> {
        let mut stderr = io::stderr().lock();
        writeln!(stderr, "{msg}")?;
        stderr.flush()
    }

    /// Write an error message to stderr.
    pub fn error(&self, msg: &str) -> io::Result<()> {
        let mut stderr = io::stderr().lock();
        writeln!(stderr, "error: {msg}")?;
        stderr.flush()
    }
}
