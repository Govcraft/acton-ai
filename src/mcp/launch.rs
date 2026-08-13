//! Bringing MCP servers up under supervision.
//!
//! One supervisor actor owns every server actor, so `shutdown_all()` reaches
//! them through the normal cascade and their `before_stop` hooks close the
//! sessions. Servers are brought up concurrently; any failure names the
//! server that caused it.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use acton_reactive::prelude::*;
use futures::future::join_all;

use crate::config::McpServerConfig;
use crate::mcp::actor::{
    build_server_actor, AwaitReady, DiscoverTools, DiscoveredTool, McpServerState,
};
use crate::mcp::connector::{validate_transport_shape, ConfigConnector, McpConnector};
use crate::mcp::error::McpError;
use crate::mcp::tools::{McpToolExecutor, McpTools};

/// Name of the actor supervising every MCP server connection.
pub const MCP_SUPERVISOR_NAME: &str = "mcp-supervisor";

/// How many restarts a single server may take inside the window before the
/// supervisor gives up on it.
const MAX_RESTARTS: u32 = 10;

/// The window restart counts are measured over.
const RESTART_WINDOW_SECS: u64 = 60;

/// Delay before the first reconnect attempt, doubling up to [`MAX_BACKOFF_MS`].
const INITIAL_BACKOFF_MS: u64 = 100;

/// Ceiling on the reconnect backoff.
const MAX_BACKOFF_MS: u64 = 5_000;

/// State of the supervising actor. It holds nothing: its whole job is to own
/// the children so the restart engine has somewhere to run.
#[acton_actor]
pub struct McpSupervisorState;

/// Everything needed to bring up one MCP server.
///
/// The connector is injectable so tests drive the production actor and
/// supervision path over an in-memory transport.
#[derive(Debug, Clone)]
pub struct McpServerSpec {
    /// Configured name of the server; the `{server}` in `mcp__{server}__{tool}`.
    pub name: String,
    /// How to establish (and re-establish) the connection.
    pub connector: Arc<dyn McpConnector>,
    /// Allowlist of remote tool names; `None` exposes everything.
    pub enabled_tools: Option<Vec<String>>,
    /// Per-call deadline.
    pub tool_timeout: Duration,
}

impl McpServerSpec {
    /// Builds a spec that connects using `[mcp_servers.*]` configuration.
    ///
    /// # Errors
    ///
    /// [`McpError::invalid_config`] when the entry names neither or both
    /// transports.
    pub fn from_config(name: &str, config: &McpServerConfig) -> Result<Self, McpError> {
        validate_transport_shape(name, config)?;
        Ok(Self {
            name: name.to_string(),
            enabled_tools: config.enabled_tools.clone(),
            tool_timeout: config.tool_timeout(),
            connector: ConfigConnector::new(name, config.clone()).shared(),
        })
    }

    /// Builds a spec around an already-constructed connector.
    #[must_use]
    pub fn with_connector(name: impl Into<String>, connector: Arc<dyn McpConnector>) -> Self {
        Self {
            name: name.into(),
            connector,
            enabled_tools: None,
            tool_timeout: Duration::from_secs(crate::config::DEFAULT_MCP_TOOL_TIMEOUT_SECS),
        }
    }

    /// Restricts the exposed tools to the given remote names.
    #[must_use]
    pub fn with_enabled_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.enabled_tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the per-call deadline.
    #[must_use]
    pub fn with_tool_timeout(mut self, timeout: Duration) -> Self {
        self.tool_timeout = timeout;
        self
    }
}

/// Converts a `[mcp_servers]` table into specs, rejecting malformed entries.
///
/// Sorted by name so launch order — and therefore log output — is stable.
///
/// # Errors
///
/// The first [`McpError::invalid_config`] encountered, naming its server.
pub fn specs_from_config(
    servers: &HashMap<String, McpServerConfig>,
) -> Result<Vec<McpServerSpec>, McpError> {
    let mut names: Vec<&String> = servers.keys().collect();
    names.sort();

    names
        .into_iter()
        .map(|name| McpServerSpec::from_config(name, &servers[name]))
        .collect()
}

/// Starts the actor that supervises every MCP server connection.
///
/// # Errors
///
/// [`McpError::invalid_config`] if the supervisor's identifier cannot be built.
pub async fn spawn_mcp_supervisor(runtime: &mut ActorRuntime) -> Result<ActorHandle, McpError> {
    let config = ActorConfig::new_with_name(MCP_SUPERVISOR_NAME)
        .map_err(|e| McpError::invalid_config(MCP_SUPERVISOR_NAME, e.to_string()))?
        .with_supervision_strategy(SupervisionStrategy::OneForOne);

    Ok(runtime
        .new_actor_with_config::<McpSupervisorState>(config)
        .start()
        .await)
}

/// Registers one server actor under `supervisor` and returns its restart-stable
/// reference.
///
/// The child is `Permanent` so a self-stop — the actor's way of reporting a
/// dead transport — brings it back, while a parent-initiated shutdown does not.
///
/// # Errors
///
/// [`McpError::connect_failed`] if registration fails.
pub async fn spawn_mcp_server(
    runtime: &ActorRuntime,
    supervisor: &ActorHandle,
    spec: &McpServerSpec,
) -> Result<SupervisedChild, McpError> {
    // `for_supervised_child`, not `new_with_name`: the parent link lives in the
    // config the spawner replays, and a child built without it never reports
    // its termination — so it would stop and simply stay down.
    let config = ActorConfig::for_supervised_child(
        format!("mcp-{}", spec.name),
        supervisor.clone(),
        Some(runtime.broker()),
    )
    .map_err(|e| McpError::invalid_config(&spec.name, e.to_string()))?
    .with_restart_policy(RestartPolicy::Permanent)
    .with_restart_limiter(RestartLimiterConfig {
        enabled: true,
        max_restarts: MAX_RESTARTS,
        window_secs: RESTART_WINDOW_SECS,
        initial_backoff_ms: INITIAL_BACKOFF_MS,
        max_backoff_ms: MAX_BACKOFF_MS,
        ..Default::default()
    });

    let blueprint = build_server_actor(
        spec.name.clone(),
        Arc::clone(&spec.connector),
        spec.enabled_tools.clone(),
        spec.tool_timeout,
    );

    supervisor
        .supervise_with::<McpServerState>(runtime, config, blueprint)
        .await
        .map_err(|e| McpError::connect_failed(&spec.name, format!("supervision failed: {e}")))
}

/// Waits for one server to finish connecting, then discovers its tools.
async fn bring_up(
    name: &str,
    child: &mut SupervisedChild,
) -> Result<Vec<DiscoveredTool>, McpError> {
    let handle = child
        .wait_running()
        .await
        .map_err(|e| McpError::connect_failed(name, format!("actor did not start: {e}")))?;

    // Not a sleep: the actor parks this reply until its connection attempt
    // resolves, so completing it proves the handshake finished.
    let status = handle
        .ask(AwaitReady)
        .await
        .map_err(|e| McpError::connect_failed(name, format!("readiness check failed: {e}")))?;

    if let Some(reason) = status.error {
        return Err(McpError::connect_failed(name, reason));
    }

    let discovered = handle
        .ask(DiscoverTools)
        .await
        .map_err(|e| McpError::tool_discovery_failed(name, e.to_string()))?;

    if let Some(reason) = discovered.error {
        return Err(McpError::tool_discovery_failed(name, reason));
    }

    Ok(discovered.tools)
}

/// Brings every configured MCP server up and returns their tools.
///
/// Servers connect concurrently; the whole launch fails on the first server
/// that cannot connect or list its tools, naming it.
///
/// # Errors
///
/// [`McpError`] from whichever server failed.
pub async fn launch_mcp_servers(
    runtime: &mut ActorRuntime,
    specs: &[McpServerSpec],
) -> Result<McpTools, McpError> {
    let mut tools = McpTools::new();
    if specs.is_empty() {
        return Ok(tools);
    }

    let supervisor = spawn_mcp_supervisor(runtime).await?;

    let mut children = Vec::with_capacity(specs.len());
    for spec in specs {
        let child = spawn_mcp_server(runtime, &supervisor, spec).await?;
        children.push(child);
    }

    let brought_up = join_all(
        specs
            .iter()
            .zip(children.iter_mut())
            .map(|(spec, child)| async move { bring_up(&spec.name, child).await }),
    )
    .await;

    for ((spec, child), discovered) in specs.iter().zip(children).zip(brought_up) {
        let discovered = discovered?;

        tracing::info!(
            server = %spec.name,
            count = discovered.len(),
            tools = %discovered
                .iter()
                .map(|t| t.definition.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            "MCP server ready"
        );

        tools.add_server(&spec.name, child.clone());
        for tool in discovered {
            let executor = McpToolExecutor::new(
                &spec.name,
                &tool.remote_name,
                child.clone(),
                spec.tool_timeout,
            );
            tools.add_tool(tool.definition, executor);
        }
    }

    Ok(tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_are_built_in_name_order() {
        let mut servers = HashMap::new();
        servers.insert("zebra".to_string(), McpServerConfig::stdio("z"));
        servers.insert("alpha".to_string(), McpServerConfig::stdio("a"));

        let specs = specs_from_config(&servers).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "alpha");
        assert_eq!(specs[1].name, "zebra");
    }

    #[test]
    fn a_malformed_entry_fails_the_whole_table_naming_itself() {
        let mut servers = HashMap::new();
        servers.insert("good".to_string(), McpServerConfig::stdio("a"));
        servers.insert("broken".to_string(), McpServerConfig::default());

        let err = specs_from_config(&servers).unwrap_err();
        assert_eq!(err.server(), "broken");
    }

    #[test]
    fn spec_carries_the_configured_timeout_and_allowlist() {
        let config = McpServerConfig::stdio("npx")
            .with_tool_timeout_secs(7)
            .with_enabled_tools(["read"]);

        let spec = McpServerSpec::from_config("fs", &config).unwrap();
        assert_eq!(spec.tool_timeout, Duration::from_secs(7));
        assert_eq!(
            spec.enabled_tools.as_deref(),
            Some(["read".to_string()].as_slice())
        );
    }

    #[tokio::test]
    async fn no_servers_means_no_supervisor_and_no_tools() {
        let mut runtime = ActonApp::launch_async().await;
        let before = runtime.actor_count();

        let tools = launch_mcp_servers(&mut runtime, &[]).await.unwrap();

        assert!(tools.is_empty());
        assert_eq!(runtime.actor_count(), before, "no actors should be spawned");
        runtime.shutdown_all().await.unwrap();
    }
}
