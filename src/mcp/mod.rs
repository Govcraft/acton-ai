//! MCP (Model Context Protocol) client support.
//!
//! External MCP servers — stdio child processes or streamable-HTTP endpoints —
//! contribute their tools to acton-ai prompts exactly like built-in tools do,
//! under the name `mcp__{server}__{tool}`.
//!
//! ## Shape
//!
//! Each configured server becomes one **supervised actor** owning its
//! connection ([`McpServerState`]). The connection is fallible external state,
//! so recovery is delegated to acton-reactive's restart engine rather than
//! hand-rolled: a dead transport stops the actor, the supervisor rebuilds it
//! from its blueprint, and the new incarnation reconnects on `after_start`.
//!
//! Tool executors hold a [`SupervisedChild`](acton_reactive::prelude::SupervisedChild)
//! and resolve the current incarnation per call, so a restart is invisible to
//! everything downstream.
//!
//! ## Configuration
//!
//! ```toml
//! [mcp_servers.filesystem]
//! command = "npx"
//! args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
//!
//! [mcp_servers.linear]
//! url = "https://mcp.linear.app/mcp"
//! auth_token_env = "LINEAR_MCP_TOKEN"
//! ```
//!
//! or programmatically:
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! let runtime = ActonAI::builder()
//!     .app_name("my-app")
//!     .ollama("qwen2.5:7b")
//!     .with_mcp_server("filesystem", McpServerConfig::stdio("npx")
//!         .with_args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]))
//!     .launch()
//!     .await?;
//! ```
//!
//! ## Shutdown
//!
//! [`ActonAI::shutdown`](crate::facade::ActonAI::shutdown) stops the
//! supervising actor, which cascades to each server actor. Their `before_stop`
//! cancels the rmcp session, closing the transport; rmcp then kills the stdio
//! child process. Dropping `ActonAI` **without** calling `shutdown()` leaves
//! that to rmcp's own drop guard, which spawns the kill on the Tokio runtime —
//! so it happens only if the runtime outlives the drop. Call `shutdown()`.

pub mod actor;
pub mod connector;
pub mod error;
pub mod launch;
pub mod tools;

pub use actor::{
    build_server_actor, AwaitReady, CallMcpTool, Connect, ConnectionStatus, DiscoverTools,
    DiscoveredTool, DiscoveredTools, McpCallOutcome, McpServerState, ReadyStatus,
};
pub use connector::{
    client_info, plan_transport, validate_transport_shape, ConfigConnector, ConnectFuture,
    McpConnector, McpService, TransportPlan,
};
pub use error::{McpError, McpErrorKind};
pub use launch::{launch_mcp_servers, spawn_mcp_server, MCP_SUPERVISOR_NAME};
pub use tools::{
    call_result_to_value, prefixed_tool_name, tool_definition, McpToolExecutor, McpTools,
};
