//! Establishing MCP client connections.
//!
//! The actor never builds a transport itself. It owns an [`McpConnector`],
//! which the supervision blueprint installs on every incarnation, so a
//! restarted actor reconnects by calling the same object again. Tests swap in
//! their own connector to drive the production actor over an in-memory
//! transport.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use acton_reactive::prelude::tokio;
use rmcp::model::{ClientCapabilities, ClientInfo, Implementation};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};

use crate::config::McpServerConfig;
use crate::mcp::error::McpError;

/// A live MCP client session.
pub type McpService = RunningService<RoleClient, ClientInfo>;

/// A future that yields a connection attempt's outcome.
///
/// `Sync` is required because acton-reactive's `FutureBox` is
/// `Send + Sync + 'static`; see [`detached`] for how that is guaranteed.
pub type ConnectFuture =
    Pin<Box<dyn Future<Output = Result<McpService, McpError>> + Send + Sync + 'static>>;

/// Establishes a connection to one MCP server.
///
/// Implementations must be callable repeatedly: the supervisor rebuilds the
/// owning actor on failure and the new incarnation connects again.
pub trait McpConnector: fmt::Debug + Send + Sync + 'static {
    /// The configured name of the server, used in errors and log lines.
    fn server_name(&self) -> &str;

    /// Opens a fresh connection.
    fn connect(&self) -> ConnectFuture;
}

/// Runs `fut` on its own task and yields its output through a oneshot channel.
///
/// Every future acton-reactive accepts from a handler must be `Sync`, and the
/// futures rmcp hands back are only guaranteed `Send`. Awaiting a
/// [`tokio::sync::oneshot::Receiver`] — which *is* `Sync` whenever its payload
/// is `Send` — bridges the gap without constraining rmcp.
///
/// `None` means the task was cancelled or panicked before producing a value.
pub fn detached<F, T>(fut: F) -> impl Future<Output = Option<T>> + Send + Sync + 'static
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let _ = tx.send(fut.await);
    });
    async move { rx.await.ok() }
}

/// The client identity presented to every MCP server acton-ai talks to.
#[must_use]
pub fn client_info() -> ClientInfo {
    ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("acton-ai", env!("CARGO_PKG_VERSION")),
    )
}

/// A validated, ready-to-build transport description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportPlan {
    /// Spawn a child process and speak MCP over its stdio.
    Stdio {
        /// Program to execute.
        command: String,
        /// Arguments passed to the program.
        args: Vec<String>,
        /// Additional environment variables for the child.
        env: HashMap<String, String>,
        /// Working directory for the child.
        cwd: Option<PathBuf>,
    },
    /// Speak MCP over a streamable-HTTP endpoint.
    Http {
        /// Endpoint URL.
        url: String,
        /// Bearer token, already resolved from the environment.
        ///
        /// Stored without the `Bearer ` prefix: rmcp's reqwest client applies
        /// `bearer_auth`, which adds the scheme itself.
        token: Option<String>,
    },
}

/// Rejects a server definition that names neither or both transports.
///
/// Pure; called at launch so a malformed `[mcp_servers.*]` block fails fast
/// with the offending name rather than at first use.
///
/// # Errors
///
/// [`McpError::invalid_config`] when `command` and `url` are both set or both
/// absent, or when a set value is blank.
pub fn validate_transport_shape(server: &str, config: &McpServerConfig) -> Result<(), McpError> {
    let command = config
        .command
        .as_deref()
        .map(str::trim)
        .filter(|c| !c.is_empty());
    let url = config
        .url
        .as_deref()
        .map(str::trim)
        .filter(|u| !u.is_empty());

    match (command, url) {
        (Some(_), None) | (None, Some(_)) => Ok(()),
        (Some(_), Some(_)) => Err(McpError::invalid_config(
            server,
            "set either `command` (stdio) or `url` (streamable HTTP), not both",
        )),
        (None, None) => Err(McpError::invalid_config(
            server,
            "set either `command` (stdio) or `url` (streamable HTTP)",
        )),
    }
}

/// Resolves a server definition into a [`TransportPlan`].
///
/// Pure with respect to `env_lookup`, which supplies environment variables so
/// the resolution — including the "named but unset" rejection — is unit
/// testable without touching the process environment.
///
/// # Errors
///
/// - [`McpError::invalid_config`] for a malformed transport selection.
/// - [`McpError::invalid_config`] when `auth_token_env` names a variable that
///   is unset or empty; connecting unauthenticated instead would silently
///   downgrade the caller's intent.
pub fn plan_transport(
    server: &str,
    config: &McpServerConfig,
    env_lookup: impl Fn(&str) -> Option<String>,
) -> Result<TransportPlan, McpError> {
    validate_transport_shape(server, config)?;

    if let Some(command) = config.command.as_deref() {
        return Ok(TransportPlan::Stdio {
            command: command.trim().to_string(),
            args: config.args.clone(),
            env: config.env.clone(),
            cwd: config.cwd.clone(),
        });
    }

    let url = config
        .url
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .to_string();

    let token = match config.auth_token_env.as_deref() {
        None => None,
        Some(var) => {
            let value = env_lookup(var).unwrap_or_default();
            if value.trim().is_empty() {
                return Err(McpError::invalid_config(
                    server,
                    format!("auth_token_env names '{var}', which is unset or empty"),
                ));
            }
            Some(value)
        }
    };

    Ok(TransportPlan::Http { url, token })
}

/// The production connector: builds a transport from `[mcp_servers.*]`
/// configuration and performs the MCP handshake.
#[derive(Debug, Clone)]
pub struct ConfigConnector {
    server_name: String,
    config: McpServerConfig,
}

impl ConfigConnector {
    /// Creates a connector for a configured server.
    #[must_use]
    pub fn new(server_name: impl Into<String>, config: McpServerConfig) -> Self {
        Self {
            server_name: server_name.into(),
            config,
        }
    }

    /// Wraps this connector for installation on an actor.
    #[must_use]
    pub fn shared(self) -> Arc<dyn McpConnector> {
        Arc::new(self)
    }
}

impl McpConnector for ConfigConnector {
    fn server_name(&self) -> &str {
        &self.server_name
    }

    fn connect(&self) -> ConnectFuture {
        let server = self.server_name.clone();
        // Planned per attempt, not once at launch, so a reconnect picks up a
        // token that was rotated in the environment since the last one.
        let plan = plan_transport(&server, &self.config, |var| std::env::var(var).ok());

        Box::pin(async move {
            let plan = plan?;
            let owned_server = server.clone();
            let attempt = async move { open(&owned_server, plan).await };
            detached(attempt)
                .await
                .unwrap_or_else(|| Err(McpError::connect_failed(&server, "connect task cancelled")))
        })
    }
}

/// Builds the transport described by `plan` and completes the MCP handshake.
async fn open(server: &str, plan: TransportPlan) -> Result<McpService, McpError> {
    match plan {
        TransportPlan::Stdio {
            command,
            args,
            env,
            cwd,
        } => {
            let mut cmd = tokio::process::Command::new(&command);
            cmd.args(&args);
            for (key, value) in &env {
                cmd.env(key, value);
            }
            if let Some(dir) = &cwd {
                cmd.current_dir(dir);
            }

            let transport = TokioChildProcess::new(cmd).map_err(|e| {
                McpError::connect_failed(server, format!("failed to spawn '{command}': {e}"))
            })?;

            client_info()
                .serve(transport)
                .await
                .map_err(|e| McpError::connect_failed(server, e.to_string()))
        }
        TransportPlan::Http { url, token } => {
            let mut http_config = StreamableHttpClientTransportConfig::with_uri(url.clone());
            http_config.auth_header = token;
            let transport = StreamableHttpClientTransport::from_config(http_config);

            client_info()
                .serve(transport)
                .await
                .map_err(|e| McpError::connect_failed(server, format!("{url}: {e}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn stdio_config_plans_a_child_process() {
        let config = McpServerConfig::stdio("npx")
            .with_args(["-y", "server"])
            .with_env_var("FOO", "bar")
            .with_cwd("/tmp");

        let plan = plan_transport("fs", &config, no_env).unwrap();
        match plan {
            TransportPlan::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                assert_eq!(command, "npx");
                assert_eq!(args, vec!["-y".to_string(), "server".to_string()]);
                assert_eq!(env.get("FOO").map(String::as_str), Some("bar"));
                assert_eq!(cwd, Some(PathBuf::from("/tmp")));
            }
            other => panic!("expected stdio, got {other:?}"),
        }
    }

    #[test]
    fn http_config_without_auth_plans_a_tokenless_endpoint() {
        let config = McpServerConfig::http("https://example.test/mcp");
        let plan = plan_transport("remote", &config, no_env).unwrap();
        assert_eq!(
            plan,
            TransportPlan::Http {
                url: "https://example.test/mcp".to_string(),
                token: None,
            }
        );
    }

    #[test]
    fn auth_token_env_is_resolved_without_a_bearer_prefix() {
        let config =
            McpServerConfig::http("https://example.test/mcp").with_auth_token_env("MCP_TOKEN");

        let plan = plan_transport("remote", &config, |var| {
            (var == "MCP_TOKEN").then(|| "secret-value".to_string())
        })
        .unwrap();

        assert_eq!(
            plan,
            TransportPlan::Http {
                url: "https://example.test/mcp".to_string(),
                token: Some("secret-value".to_string()),
            }
        );
    }

    #[test]
    fn missing_auth_token_env_is_an_error_not_an_anonymous_connection() {
        let config =
            McpServerConfig::http("https://example.test/mcp").with_auth_token_env("MCP_TOKEN");

        let err = plan_transport("remote", &config, no_env).unwrap_err();
        assert_eq!(err.server(), "remote");
        assert!(err.to_string().contains("MCP_TOKEN"));
    }

    #[test]
    fn blank_auth_token_env_is_an_error() {
        let config =
            McpServerConfig::http("https://example.test/mcp").with_auth_token_env("MCP_TOKEN");

        let err = plan_transport("remote", &config, |_| Some("   ".to_string())).unwrap_err();
        assert!(err.to_string().contains("unset or empty"));
    }

    #[test]
    fn both_transports_is_rejected() {
        let mut config = McpServerConfig::stdio("npx");
        config.url = Some("https://example.test/mcp".to_string());

        let err = validate_transport_shape("both", &config).unwrap_err();
        assert!(err.to_string().contains("not both"));
        assert!(plan_transport("both", &config, no_env).is_err());
    }

    #[test]
    fn neither_transport_is_rejected() {
        let config = McpServerConfig::default();
        let err = validate_transport_shape("empty", &config).unwrap_err();
        assert!(err.to_string().contains("either"));
    }

    #[test]
    fn blank_command_counts_as_absent() {
        let config = McpServerConfig::stdio("   ");
        assert!(validate_transport_shape("blank", &config).is_err());
    }

    #[tokio::test]
    async fn detached_forwards_the_inner_output() {
        let value = detached(async { 41_u32 + 1 }).await;
        assert_eq!(value, Some(42));
    }

    #[tokio::test]
    async fn detached_reports_none_when_the_task_panics() {
        let value = detached(async { panic!("boom") }).await;
        assert_eq!(value, None::<u32>);
    }
}
