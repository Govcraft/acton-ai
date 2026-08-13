//! Error types for MCP (Model Context Protocol) client operations.
//!
//! Follows the repo-wide custom-error pattern (see [`crate::error`]): a struct
//! carrying a `kind` enum, `Display` written for a human reading a log line,
//! and no dependency on `anyhow` or `thiserror`.

use crate::error::{ActonAIError, ActonAIErrorKind};
use crate::tools::ToolError;
use std::fmt;
use std::time::Duration;

/// Errors raised while talking to an MCP server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpError {
    /// The specific error that occurred.
    kind: Box<McpErrorKind>,
}

/// Specific MCP error cases.
///
/// Marked `#[non_exhaustive]` so new failure kinds can be added without
/// breaking downstream `match`es.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum McpErrorKind {
    /// The transport could not be established, or the MCP handshake failed.
    ConnectFailed {
        /// Configured name of the server.
        server: String,
        /// Why the connection failed.
        reason: String,
    },
    /// `tools/list` failed.
    ToolDiscoveryFailed {
        /// Configured name of the server.
        server: String,
        /// Why discovery failed.
        reason: String,
    },
    /// `tools/call` failed at the protocol or transport level.
    CallFailed {
        /// Configured name of the server.
        server: String,
        /// Remote (unprefixed) tool name.
        tool: String,
        /// Why the call failed.
        reason: String,
    },
    /// The server definition in configuration is not usable.
    InvalidConfig {
        /// Configured name of the server.
        server: String,
        /// What is wrong with it.
        reason: String,
    },
    /// A call exceeded its configured timeout.
    Timeout {
        /// Configured name of the server.
        server: String,
        /// Remote (unprefixed) tool name.
        tool: String,
        /// The deadline that elapsed.
        duration: Duration,
    },
    /// The server actor has no live connection right now.
    ///
    /// Produced when a tool call arrives while the connection is being
    /// (re-)established — for example in the window between a supervised
    /// restart and the restarted incarnation completing its handshake.
    NotConnected {
        /// Configured name of the server.
        server: String,
    },
}

impl McpError {
    /// Creates a new error with the given kind.
    #[must_use]
    pub fn new(kind: McpErrorKind) -> Self {
        Self {
            kind: Box::new(kind),
        }
    }

    /// Returns a reference to the error kind.
    #[must_use]
    pub fn kind(&self) -> &McpErrorKind {
        &self.kind
    }

    /// Creates a connect-failed error.
    #[must_use]
    pub fn connect_failed(server: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(McpErrorKind::ConnectFailed {
            server: server.into(),
            reason: reason.into(),
        })
    }

    /// Creates a tool-discovery-failed error.
    #[must_use]
    pub fn tool_discovery_failed(server: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(McpErrorKind::ToolDiscoveryFailed {
            server: server.into(),
            reason: reason.into(),
        })
    }

    /// Creates a call-failed error.
    #[must_use]
    pub fn call_failed(
        server: impl Into<String>,
        tool: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::new(McpErrorKind::CallFailed {
            server: server.into(),
            tool: tool.into(),
            reason: reason.into(),
        })
    }

    /// Creates an invalid-config error.
    #[must_use]
    pub fn invalid_config(server: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(McpErrorKind::InvalidConfig {
            server: server.into(),
            reason: reason.into(),
        })
    }

    /// Creates a timeout error.
    #[must_use]
    pub fn timeout(server: impl Into<String>, tool: impl Into<String>, duration: Duration) -> Self {
        Self::new(McpErrorKind::Timeout {
            server: server.into(),
            tool: tool.into(),
            duration,
        })
    }

    /// Creates a not-connected error.
    #[must_use]
    pub fn not_connected(server: impl Into<String>) -> Self {
        Self::new(McpErrorKind::NotConnected {
            server: server.into(),
        })
    }

    /// The configured name of the server this error concerns.
    #[must_use]
    pub fn server(&self) -> &str {
        match self.kind.as_ref() {
            McpErrorKind::ConnectFailed { server, .. }
            | McpErrorKind::ToolDiscoveryFailed { server, .. }
            | McpErrorKind::CallFailed { server, .. }
            | McpErrorKind::InvalidConfig { server, .. }
            | McpErrorKind::Timeout { server, .. }
            | McpErrorKind::NotConnected { server } => server,
        }
    }

    /// Returns true when the error means the connection is unusable and the
    /// owning actor should be recycled by its supervisor.
    ///
    /// A protocol-level `CallFailed` from a live connection is *not* fatal —
    /// only losing the connection is.
    #[must_use]
    pub fn is_connection_fatal(&self) -> bool {
        matches!(
            self.kind.as_ref(),
            McpErrorKind::ConnectFailed { .. } | McpErrorKind::NotConnected { .. }
        )
    }
}

impl fmt::Display for McpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind.as_ref() {
            McpErrorKind::ConnectFailed { server, reason } => {
                write!(f, "MCP server '{server}' failed to connect: {reason}")
            }
            McpErrorKind::ToolDiscoveryFailed { server, reason } => {
                write!(f, "MCP server '{server}' tool discovery failed: {reason}")
            }
            McpErrorKind::CallFailed {
                server,
                tool,
                reason,
            } => {
                write!(f, "MCP server '{server}' tool '{tool}' failed: {reason}")
            }
            McpErrorKind::InvalidConfig { server, reason } => {
                write!(f, "invalid MCP server configuration '{server}': {reason}")
            }
            McpErrorKind::Timeout {
                server,
                tool,
                duration,
            } => {
                write!(
                    f,
                    "MCP server '{server}' tool '{tool}' timed out after {} seconds",
                    duration.as_secs()
                )
            }
            McpErrorKind::NotConnected { server } => {
                write!(
                    f,
                    "MCP server '{server}' is not connected; the connection is being re-established"
                )
            }
        }
    }
}

impl std::error::Error for McpError {}

impl From<McpError> for ActonAIError {
    fn from(err: McpError) -> Self {
        Self::new(ActonAIErrorKind::Mcp {
            server: err.server().to_string(),
            reason: err.to_string(),
        })
    }
}

impl From<McpError> for ToolError {
    fn from(err: McpError) -> Self {
        match err.kind() {
            McpErrorKind::Timeout {
                tool,
                duration,
                server,
            } => Self::timeout(format!("mcp__{server}__{tool}"), *duration),
            _ => Self::execution_failed(format!("mcp:{}", err.server()), err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolErrorKind;

    #[test]
    fn connect_failed_display_names_the_server() {
        let err = McpError::connect_failed("filesystem", "spawn failed: no such file");
        let message = err.to_string();
        assert!(message.contains("filesystem"));
        assert!(message.contains("no such file"));
    }

    #[test]
    fn call_failed_display_names_tool_and_server() {
        let err = McpError::call_failed("linear", "create_issue", "transport closed");
        let message = err.to_string();
        assert!(message.contains("linear"));
        assert!(message.contains("create_issue"));
        assert!(message.contains("transport closed"));
    }

    #[test]
    fn timeout_display_mentions_seconds() {
        let err = McpError::timeout("fs", "read", Duration::from_secs(45));
        assert!(err.to_string().contains("45"));
    }

    #[test]
    fn server_accessor_covers_every_kind() {
        assert_eq!(McpError::connect_failed("a", "x").server(), "a");
        assert_eq!(McpError::tool_discovery_failed("b", "x").server(), "b");
        assert_eq!(McpError::call_failed("c", "t", "x").server(), "c");
        assert_eq!(McpError::invalid_config("d", "x").server(), "d");
        assert_eq!(
            McpError::timeout("e", "t", Duration::from_secs(1)).server(),
            "e"
        );
        assert_eq!(McpError::not_connected("f").server(), "f");
    }

    #[test]
    fn only_connection_loss_is_fatal() {
        assert!(McpError::connect_failed("a", "x").is_connection_fatal());
        assert!(McpError::not_connected("a").is_connection_fatal());
        assert!(!McpError::call_failed("a", "t", "x").is_connection_fatal());
        assert!(!McpError::invalid_config("a", "x").is_connection_fatal());
    }

    #[test]
    fn converts_into_acton_ai_error() {
        let err: ActonAIError = McpError::connect_failed("fs", "boom").into();
        assert!(matches!(err.kind, ActonAIErrorKind::Mcp { .. }));
        assert!(err.to_string().contains("fs"));
    }

    #[test]
    fn timeout_converts_into_tool_timeout() {
        let err: ToolError = McpError::timeout("fs", "read", Duration::from_secs(3)).into();
        match err.kind() {
            ToolErrorKind::Timeout {
                tool_name,
                duration,
            } => {
                assert_eq!(tool_name, "mcp__fs__read");
                assert_eq!(*duration, Duration::from_secs(3));
            }
            other => panic!("expected timeout, got {other:?}"),
        }
    }

    #[test]
    fn other_errors_convert_into_tool_execution_failure() {
        let err: ToolError = McpError::call_failed("fs", "read", "nope").into();
        assert!(matches!(err.kind(), ToolErrorKind::ExecutionFailed { .. }));
    }
}
