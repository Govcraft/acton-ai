//! End-to-end tests for MCP client support.
//!
//! These drive the *production* actor, supervision, and executor path. The only
//! substitution is the transport: instead of spawning a child process or
//! opening an HTTP connection, the connector wires an in-process rmcp server to
//! the client over a `tokio::io::duplex` pair. Everything downstream of
//! `serve()` — the supervised actor, its blueprint, restart-on-failure, tool
//! discovery, and `McpToolExecutor` — is the code that ships.
//!
//! Nothing here synchronizes with `sleep`. Barriers are `ask` round-trips and
//! `SupervisedChild::wait_generation`.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use acton_ai::mcp::connector::{client_info, detached, ConnectFuture, McpConnector};
use acton_ai::mcp::launch::{launch_mcp_servers, McpServerSpec};
use acton_ai::mcp::McpError;
use acton_ai::tools::ToolErrorKind;
use acton_reactive::prelude::*;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RunningService};
use rmcp::{ErrorData, RoleServer, ServiceExt};
use serde_json::{json, Map, Value};
use tokio::sync::Mutex;

// =============================================================================
// An in-process MCP server
// =============================================================================

/// A three-tool MCP server: one that echoes, one that computes, and one that
/// reports a tool-level failure.
#[derive(Debug, Clone, Default)]
struct TestServer;

fn schema(value: Value) -> Arc<Map<String, Value>> {
    Arc::new(value.as_object().cloned().expect("schema is an object"))
}

impl TestServer {
    fn tools() -> Vec<Tool> {
        vec![
            Tool::new(
                "echo",
                "Echoes the given message back",
                schema(json!({
                    "type": "object",
                    "properties": { "message": { "type": "string" } },
                    "required": ["message"]
                })),
            ),
            Tool::new(
                "add",
                "Adds two numbers",
                schema(json!({
                    "type": "object",
                    "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
                    "required": ["a", "b"]
                })),
            ),
            Tool::new(
                "boom",
                "Always reports a tool-level error",
                schema(json!({ "type": "object", "properties": {} })),
            ),
        ]
    }
}

impl ServerHandler for TestServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }

    fn list_tools(
        &self,
        _params: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListToolsResult, ErrorData>> + Send + '_ {
        std::future::ready(Ok(ListToolsResult::with_all_items(Self::tools())))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<rmcp::model::CallToolResponse, ErrorData>> + Send + '_ {
        let args = request.arguments.unwrap_or_default();
        let result = match request.name.as_ref() {
            "echo" => {
                let message = args
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                    "echo: {message}"
                ))]))
            }
            "add" => {
                let a = args.get("a").and_then(Value::as_f64).unwrap_or_default();
                let b = args.get("b").and_then(Value::as_f64).unwrap_or_default();
                let mut result = CallToolResult::success(vec![]);
                result.structured_content = Some(json!({ "sum": a + b }));
                Ok(result)
            }
            "boom" => Ok(CallToolResult::error(vec![ContentBlock::text(
                "the tool exploded",
            )])),
            other => Err(ErrorData::invalid_params(
                format!("unknown tool: {other}"),
                None,
            )),
        };

        std::future::ready(result.map(Into::into))
    }
}

// =============================================================================
// A connector that hands out a fresh in-memory transport per attempt
// =============================================================================

/// Connects the production actor to an in-process [`TestServer`].
///
/// Each `connect()` builds a brand-new duplex pair and a brand-new server, so a
/// reconnect after a supervised restart behaves the way a real reconnect does:
/// the old session is gone and a fresh one takes its place.
#[derive(Clone)]
struct DuplexConnector {
    name: String,
    attempts: Arc<AtomicUsize>,
    /// 1-based attempt numbers that fail before any transport is built.
    fail_on: Vec<usize>,
    /// The live server session, kept so a test can drop it and sever the link.
    server: Arc<Mutex<Option<RunningService<RoleServer, TestServer>>>>,
}

impl std::fmt::Debug for DuplexConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuplexConnector")
            .field("name", &self.name)
            .field("attempts", &self.attempts.load(Ordering::SeqCst))
            .finish()
    }
}

impl DuplexConnector {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            attempts: Arc::new(AtomicUsize::new(0)),
            fail_on: Vec::new(),
            server: Arc::new(Mutex::new(None)),
        }
    }

    /// A connector whose listed attempts (1-based) fail with a connect error.
    fn failing_on(name: &str, fail_on: Vec<usize>) -> Self {
        Self {
            fail_on,
            ..Self::new(name)
        }
    }

    fn attempts(&self) -> usize {
        self.attempts.load(Ordering::SeqCst)
    }

    /// Severs the link by dropping the server session.
    ///
    /// rmcp's drop guard cancels the session, which closes its half of the
    /// duplex; the client then sees the transport go away. This is the
    /// in-memory equivalent of a stdio child process dying.
    async fn sever(&self) {
        let taken = self.server.lock().await.take();
        assert!(taken.is_some(), "no live server session to sever");
        drop(taken);
    }
}

impl McpConnector for DuplexConnector {
    fn server_name(&self) -> &str {
        &self.name
    }

    fn connect(&self) -> ConnectFuture {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        let name = self.name.clone();

        if self.fail_on.contains(&attempt) {
            return Box::pin(async move {
                Err(McpError::connect_failed(
                    &name,
                    format!("injected failure on attempt {attempt}"),
                ))
            });
        }

        let slot = Arc::clone(&self.server);

        Box::pin(async move {
            let attempt = async move {
                let (client_side, server_side) = tokio::io::duplex(8 * 1024);

                // The server half of `initialize` waits for the client's
                // request, so the two handshakes must run concurrently.
                let serving = tokio::spawn(async move { TestServer.serve(server_side).await });

                let client = client_info()
                    .serve(client_side)
                    .await
                    .map_err(|e| McpError::connect_failed(&name, format!("client: {e}")))?;

                let server = serving
                    .await
                    .map_err(|e| McpError::connect_failed(&name, format!("server task: {e}")))?
                    .map_err(|e| McpError::connect_failed(&name, format!("server: {e}")))?;

                *slot.lock().await = Some(server);
                Ok(client)
            };

            detached(attempt)
                .await
                .unwrap_or_else(|| Err(McpError::connect_failed("duplex", "task cancelled")))
        })
    }
}

/// A connector that always fails, for the launch-failure path.
#[derive(Debug, Clone)]
struct BrokenConnector;

impl McpConnector for BrokenConnector {
    fn server_name(&self) -> &str {
        "broken"
    }

    fn connect(&self) -> ConnectFuture {
        Box::pin(async {
            Err(McpError::connect_failed(
                "broken",
                "the executable does not exist",
            ))
        })
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn spec(connector: &DuplexConnector) -> McpServerSpec {
    McpServerSpec::with_connector(&connector.name, Arc::new(connector.clone()))
        .with_tool_timeout(Duration::from_secs(5))
}

async fn call(
    tools: &acton_ai::mcp::McpTools,
    name: &str,
    args: Value,
) -> Result<Value, acton_ai::tools::ToolError> {
    let executor = tools
        .get_executor(name)
        .unwrap_or_else(|| panic!("no executor registered for {name}"));
    executor.execute(args).await
}

// =============================================================================
// Discovery
// =============================================================================

#[tokio::test]
async fn discovery_exposes_every_tool_under_a_prefixed_name() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let connector = DuplexConnector::new("testfs");

    let tools = launch_mcp_servers(&mut runtime, &[spec(&connector)]).await?;

    assert_eq!(tools.len(), 3);
    for expected in ["mcp__testfs__echo", "mcp__testfs__add", "mcp__testfs__boom"] {
        assert!(
            tools.get_config(expected).is_some(),
            "missing tool {expected}"
        );
    }

    let echo = tools.get_config("mcp__testfs__echo").unwrap();
    assert_eq!(echo.definition.description, "Echoes the given message back");
    assert_eq!(echo.definition.input_schema["required"], json!(["message"]));
    assert!(!echo.sandboxed);
    assert_eq!(echo.timeout, Duration::from_secs(5));

    runtime.shutdown_all().await?;
    Ok(())
}

#[tokio::test]
async fn the_allowlist_filters_discovered_tools() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let connector = DuplexConnector::new("testfs");

    // "teleport" is not offered by the server: an unknown allowlist entry must
    // warn rather than fail the launch.
    let filtered = spec(&connector).with_enabled_tools(["echo", "teleport"]);
    let tools = launch_mcp_servers(&mut runtime, &[filtered]).await?;

    assert_eq!(tools.len(), 1);
    assert!(tools.get_config("mcp__testfs__echo").is_some());
    assert!(tools.get_config("mcp__testfs__add").is_none());

    runtime.shutdown_all().await?;
    Ok(())
}

#[tokio::test]
async fn several_servers_connect_and_namespace_separately() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let one = DuplexConnector::new("alpha");
    let two = DuplexConnector::new("beta");

    let tools = launch_mcp_servers(&mut runtime, &[spec(&one), spec(&two)]).await?;

    assert_eq!(tools.len(), 6);
    assert!(tools.get_config("mcp__alpha__echo").is_some());
    assert!(tools.get_config("mcp__beta__echo").is_some());
    assert!(tools.server("alpha").is_some());
    assert!(tools.server("beta").is_some());

    runtime.shutdown_all().await?;
    Ok(())
}

// =============================================================================
// Execution
// =============================================================================

#[tokio::test]
async fn the_executor_round_trips_arguments_and_text_results() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let connector = DuplexConnector::new("testfs");
    let tools = launch_mcp_servers(&mut runtime, &[spec(&connector)]).await?;

    let value = call(&tools, "mcp__testfs__echo", json!({"message": "hello"})).await?;
    assert_eq!(value, json!("echo: hello"));

    runtime.shutdown_all().await?;
    Ok(())
}

#[tokio::test]
async fn structured_content_reaches_the_caller_verbatim() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let connector = DuplexConnector::new("testfs");
    let tools = launch_mcp_servers(&mut runtime, &[spec(&connector)]).await?;

    let value = call(&tools, "mcp__testfs__add", json!({"a": 2, "b": 3})).await?;
    assert_eq!(value, json!({"sum": 5.0}));

    runtime.shutdown_all().await?;
    Ok(())
}

#[tokio::test]
async fn a_tool_level_error_surfaces_as_a_tool_error() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let connector = DuplexConnector::new("testfs");
    let tools = launch_mcp_servers(&mut runtime, &[spec(&connector)]).await?;

    let err = call(&tools, "mcp__testfs__boom", json!({}))
        .await
        .expect_err("boom must fail");

    assert!(matches!(err.kind(), ToolErrorKind::ExecutionFailed { .. }));
    assert!(err.to_string().contains("the tool exploded"));

    runtime.shutdown_all().await?;
    Ok(())
}

#[tokio::test]
async fn non_object_arguments_are_rejected_before_the_call() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let connector = DuplexConnector::new("testfs");
    let tools = launch_mcp_servers(&mut runtime, &[spec(&connector)]).await?;

    let err = call(&tools, "mcp__testfs__echo", json!("not an object"))
        .await
        .expect_err("a string is not valid MCP arguments");

    assert!(matches!(err.kind(), ToolErrorKind::ValidationFailed { .. }));

    runtime.shutdown_all().await?;
    Ok(())
}

// =============================================================================
// Launch failure
// =============================================================================

#[tokio::test]
async fn a_server_that_cannot_connect_fails_the_launch_by_name() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let broken = McpServerSpec::with_connector("broken", Arc::new(BrokenConnector));

    let err = launch_mcp_servers(&mut runtime, &[broken])
        .await
        .expect_err("a server that cannot connect must fail the launch");

    assert_eq!(err.server(), "broken");
    assert!(err.to_string().contains("does not exist"));

    runtime.shutdown_all().await?;
    Ok(())
}

// =============================================================================
// Supervision
// =============================================================================

/// The point of the supervised design.
///
/// Severing the connection must (a) fail the in-flight call cleanly rather than
/// hanging, (b) get the actor restarted by its supervisor, and (c) leave the
/// *same* executor working afterwards, because it resolves the current
/// incarnation per call instead of caching a handle.
///
/// This test discriminates: removing the escalation in the actor's call handler
/// leaves the child at generation 0 and `wait_generation` never resolves.
#[tokio::test]
async fn a_severed_connection_is_restarted_and_the_same_executor_keeps_working(
) -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let connector = DuplexConnector::new("testfs");

    let tools = launch_mcp_servers(&mut runtime, &[spec(&connector)]).await?;
    assert_eq!(connector.attempts(), 1, "one connection at launch");

    // Baseline: the tool works before anything is broken.
    let before = call(&tools, "mcp__testfs__echo", json!({"message": "first"})).await?;
    assert_eq!(before, json!("echo: first"));

    let mut child = tools.server("testfs").expect("server actor").clone();
    assert_eq!(
        child.status().generation(),
        RestartGeneration::FIRST,
        "the child starts as its first incarnation"
    );

    // Kill the server out from under the actor.
    connector.sever().await;

    // The in-flight call fails rather than hanging or silently succeeding.
    let during = call(&tools, "mcp__testfs__echo", json!({"message": "second"})).await;
    assert!(
        during.is_err(),
        "a call over a severed transport must fail, got {during:?}"
    );

    // Not a sleep: this resolves only once the supervisor has actually
    // rebuilt the child as a later incarnation.
    // The barrier is `wait_generation`; the deadline only turns "the restart
    // never happened" into a failure instead of a hang.
    let restarted = tokio::time::timeout(
        Duration::from_secs(10),
        child.wait_generation(RestartGeneration::FIRST.next()),
    )
    .await
    .expect("the supervisor must restart the child after the transport died")?;
    assert!(
        child.status().generation().get() >= 1,
        "the supervisor must have rebuilt the child at least once"
    );

    // And only once that incarnation has finished its own handshake.
    let status = restarted.ask(acton_ai::mcp::AwaitReady).await?;
    assert_eq!(status.error, None, "the restarted actor must reconnect");
    assert_eq!(
        connector.attempts(),
        2,
        "the restarted incarnation must open a fresh connection"
    );

    // The executor was built before the restart and never rebuilt.
    let after = call(&tools, "mcp__testfs__echo", json!({"message": "third"})).await?;
    assert_eq!(after, json!("echo: third"));

    runtime.shutdown_all().await?;
    Ok(())
}

/// A reconnect attempt that fails must not strand the actor.
///
/// The supervisor keeps retrying (with backoff) until a later attempt
/// succeeds. This discriminates the `ConnectionFailed` escalation: without the
/// self-terminate in that handler, the second incarnation parks in `Failed`
/// forever, attempt 3 never happens, and `wait_generation` times out.
#[tokio::test]
async fn a_failed_reconnect_keeps_retrying_until_the_server_returns() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    // Attempt 1 (launch) succeeds, attempt 2 (first reconnect) fails,
    // attempt 3 succeeds.
    let connector = DuplexConnector::failing_on("testfs", vec![2]);

    let tools = launch_mcp_servers(&mut runtime, &[spec(&connector)]).await?;
    let before = call(&tools, "mcp__testfs__echo", json!({"message": "first"})).await?;
    assert_eq!(before, json!("echo: first"));

    let mut child = tools.server("testfs").expect("server actor").clone();

    // Kill the server, then trip the escalation with a failing call.
    connector.sever().await;
    let during = call(&tools, "mcp__testfs__echo", json!({"message": "second"})).await;
    assert!(during.is_err(), "a severed transport must fail the call");

    // Generation 1's reconnect is the injected failure; reaching generation 2
    // proves the failed attempt escalated instead of stranding the actor.
    let restarted = tokio::time::timeout(
        Duration::from_secs(10),
        child.wait_generation(RestartGeneration::FIRST.next().next()),
    )
    .await
    .expect("the supervisor must keep restarting past a failed reconnect")?;

    let status = restarted.ask(acton_ai::mcp::AwaitReady).await?;
    assert_eq!(status.error, None, "attempt 3 must reconnect");
    assert_eq!(
        connector.attempts(),
        3,
        "launch + failed reconnect + successful reconnect"
    );

    // The executor from before the outage still works.
    let after = call(&tools, "mcp__testfs__echo", json!({"message": "third"})).await?;
    assert_eq!(after, json!("echo: third"));

    runtime.shutdown_all().await?;
    Ok(())
}

#[tokio::test]
async fn shutdown_stops_the_server_actors() -> anyhow::Result<()> {
    let mut runtime = ActonApp::launch_async().await;
    let connector = DuplexConnector::new("testfs");
    let tools = launch_mcp_servers(&mut runtime, &[spec(&connector)]).await?;

    let child = tools.server("testfs").expect("server actor").clone();
    assert!(child.current().is_some(), "running before shutdown");

    runtime.shutdown_all().await?;

    // A shutdown is a parent cascade, never a restart: the child must stay
    // down rather than being rebuilt while the runtime is winding up.
    assert_eq!(
        connector.attempts(),
        1,
        "shutdown must not trigger a reconnect"
    );
    Ok(())
}
