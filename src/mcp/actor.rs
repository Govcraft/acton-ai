//! The supervised actor that owns one MCP server connection.
//!
//! An MCP connection is fallible external state: a stdio child process can
//! die, an HTTP transport can drop. Wrapping it in an actor whose supervisor
//! restarts it means recovery is the framework's job rather than a hand-rolled
//! reconnect loop, and the whole recipe for rebuilding the connection lives in
//! one place — the [blueprint](build_server_actor) the supervisor replays.
//!
//! ## Lifecycle
//!
//! ```text
//! start ──after_start──> Connect ──connect().await──┬─> ConnectionReady ─> Ready
//!                                                    └─> ConnectionFailed ─> Failed
//! ```
//!
//! [`AwaitReady`] is the launch-time barrier. It resolves as soon as the
//! outcome is known, and callers that ask before the handshake finishes have
//! their reply envelope stashed rather than being answered with a premature
//! "not connected" — the deferred-reply pattern, not a sleep.
//!
//! ## Failure escalation
//!
//! A tool call that fails because the transport is gone replies the error to
//! the caller and then stops the actor. A failed *reconnect* does the same once
//! its waiters are answered, so a restarted incarnation whose reconnect fails
//! is retried by the supervisor (with backoff) rather than stranded in
//! `Failed`.
//!
//! The first incarnation is asymmetric on purpose: it does **not** stop itself,
//! because [`launch_mcp_servers`](crate::mcp::launch::launch_mcp_servers) is
//! holding a readiness `ask` against its mailbox, and closing that mailbox
//! first replaces the real reason with an undeliverable-mailbox error. It can
//! afford to stay up — a launch-time failure fails the launch, so the runtime
//! is discarded rather than used. `McpServerState::is_reconnect` is what tells
//! the two apart. A clean `stop()` records
//! [`TerminationReason::Normal`](acton_reactive::prelude::TerminationReason),
//! which a `Permanent` restart policy treats as restartable, so the supervisor
//! rebuilds the actor from the blueprint and the new incarnation reconnects.
//! A parent-initiated shutdown records `ParentShutdown` instead and is never
//! restarted, so `shutdown_all()` does not fight the supervisor.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use acton_reactive::prelude::*;
use rmcp::model::CallToolRequestParams;
use rmcp::service::ServiceError;
use serde_json::{Map, Value};

use crate::mcp::connector::{detached, McpConnector, McpService};
use crate::mcp::error::McpError;
use crate::mcp::tools::{call_result_to_value, prefixed_tool_name, tool_definition};
use crate::messages::ToolDefinition;
use crate::tools::ToolError;

/// A live connection, wrapped so the actor state stays `Debug` without
/// printing the whole rmcp service graph on every trace line.
#[derive(Clone)]
pub struct Connection(Arc<McpService>);

impl Connection {
    /// Wraps an established session.
    #[must_use]
    pub fn new(service: McpService) -> Self {
        Self(Arc::new(service))
    }

    /// The underlying session.
    #[must_use]
    pub fn service(&self) -> &McpService {
        &self.0
    }
}

impl fmt::Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Connection(<mcp session>)")
    }
}

/// Where the connection currently stands.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// The handshake has not finished (or has not started) yet.
    #[default]
    Pending,
    /// A session is established.
    Ready,
    /// The last attempt failed; the message says why.
    Failed(String),
}

// =============================================================================
// Messages
// =============================================================================

/// Asks the actor to establish its connection.
///
/// Self-sent from `after_start`, so a restarted incarnation reconnects without
/// any launch-time code noticing the restart happened.
#[acton_message]
pub struct Connect;

/// Delivers an established session into actor state.
#[acton_message]
pub struct ConnectionReady {
    /// The session that was just established.
    pub connection: Connection,
}

/// Records a failed connection attempt.
#[acton_message]
pub struct ConnectionFailed {
    /// Rendered [`McpError`].
    pub reason: String,
}

/// Waits until the connection attempt has resolved, one way or the other.
#[acton_message]
pub struct AwaitReady;

/// The resolved outcome of the connection attempt.
#[acton_message]
pub struct ReadyStatus {
    /// `None` when the session is live; otherwise why it is not.
    pub error: Option<String>,
}

impl Request for AwaitReady {
    type Response = ReadyStatus;
}

/// Asks the actor for the server's tool list.
#[acton_message]
pub struct DiscoverTools;

/// One tool the server offered.
#[acton_message]
pub struct DiscoveredTool {
    /// The name the server knows the tool by.
    pub remote_name: String,
    /// The definition sent to the LLM, named `mcp__{server}__{tool}`.
    pub definition: ToolDefinition,
}

/// The result of a discovery round.
#[acton_message]
pub struct DiscoveredTools {
    /// Tools that passed the `enabled_tools` allowlist.
    pub tools: Vec<DiscoveredTool>,
    /// `Some` when discovery failed; `tools` is then empty.
    pub error: Option<String>,
}

impl Request for DiscoverTools {
    type Response = DiscoveredTools;
}

/// Asks the actor to invoke one remote tool.
#[acton_message]
pub struct CallMcpTool {
    /// The name the server knows the tool by (unprefixed).
    pub remote_name: String,
    /// Arguments, already validated to be a JSON object.
    pub arguments: Map<String, Value>,
}

/// The outcome of a tool call, shaped for the prompt loop.
#[acton_message]
pub struct McpCallOutcome {
    /// The tool's value, or the error the LLM should see.
    pub outcome: Result<Value, ToolError>,
}

impl Request for CallMcpTool {
    type Response = McpCallOutcome;
}

// =============================================================================
// State
// =============================================================================

/// State of the actor that owns one MCP server connection.
#[acton_actor]
pub struct McpServerState {
    /// Configured name of the server.
    pub server_name: String,
    /// How to (re-)establish the connection. Installed by the blueprint.
    pub connector: Option<Arc<dyn McpConnector>>,
    /// The live session, once the handshake completes.
    pub connection: Option<Connection>,
    /// Where the connection attempt stands.
    pub status: ConnectionStatus,
    /// Allowlist of remote tool names; `None` exposes everything.
    pub enabled_tools: Option<Vec<String>>,
    /// Per-call deadline.
    pub tool_timeout: Duration,
    /// Reply envelopes parked until the connection attempt resolves.
    pub waiting: Vec<OutboundEnvelope>,
    /// Whether this incarnation is a supervised restart rather than the
    /// original launch.
    ///
    /// Set by the blueprint from a counter that outlives any one incarnation.
    /// It decides whether a failed connection attempt escalates: a reconnect
    /// must (nothing else would ever retry it), while the first incarnation
    /// must not, because `launch_mcp_servers` is holding an `ask` against this
    /// mailbox and needs the reason.
    pub is_reconnect: bool,
}

/// Classifies an rmcp error as "the connection is gone".
///
/// Pure, so the escalation rule is testable without a transport. A protocol
/// error from a live session is the server saying no; only these mean the
/// session itself is unusable and the actor should be recycled.
#[must_use]
pub fn is_transport_dead(error: &ServiceError) -> bool {
    matches!(
        error,
        ServiceError::TransportClosed
            | ServiceError::TransportSend(_)
            | ServiceError::Cancelled { .. }
    )
}

/// Applies the `enabled_tools` allowlist, warning about names the server never
/// offered.
///
/// Unknown allowlist entries are a warning rather than an error: a server can
/// legitimately drop a tool between releases, and refusing to start over it
/// would make an unrelated upgrade break the whole runtime.
fn filter_enabled(
    server_name: &str,
    allowlist: Option<&Vec<String>>,
    discovered: Vec<DiscoveredTool>,
) -> Vec<DiscoveredTool> {
    let Some(allowed) = allowlist else {
        return discovered;
    };

    for wanted in allowed {
        if !discovered.iter().any(|t| &t.remote_name == wanted) {
            tracing::warn!(
                server = %server_name,
                tool = %wanted,
                "enabled_tools names a tool this MCP server does not offer"
            );
        }
    }

    discovered
        .into_iter()
        .filter(|t| allowed.contains(&t.remote_name))
        .collect()
}

/// Builds the supervision blueprint for one MCP server.
///
/// The returned closure is the *entire* definition of the child: the
/// supervisor replays it on every restart, so everything the actor needs —
/// the connector, the allowlist, the timeout, the handlers, and the
/// `after_start` reconnect — is established inside it.
pub fn build_server_actor(
    server_name: String,
    connector: Arc<dyn McpConnector>,
    enabled_tools: Option<Vec<String>>,
    tool_timeout: Duration,
) -> impl Fn(&mut ManagedActor<Idle, McpServerState>) + Send + Sync + 'static {
    // The one piece of state that must outlive an incarnation. Actor state is
    // rebuilt from `Default` on every restart, so a counter kept here — in the
    // blueprint the supervisor replays — is how an incarnation learns it is a
    // restart rather than the original.
    let incarnations = Arc::new(AtomicU64::new(0));

    move |actor: &mut ManagedActor<Idle, McpServerState>| {
        actor.model.server_name = server_name.clone();
        actor.model.connector = Some(Arc::clone(&connector));
        actor.model.enabled_tools = enabled_tools.clone();
        actor.model.tool_timeout = tool_timeout;
        actor.model.is_reconnect = incarnations.fetch_add(1, Ordering::SeqCst) > 0;

        register_handlers(actor);

        // Every incarnation connects itself. Nothing outside has to notice
        // that a restart happened.
        actor.after_start(|actor| {
            let handle = actor.handle().clone();
            async move {
                handle.send(Connect).await;
            }
        });

        // Cancelling the session closes the transport, which for a stdio
        // server is what makes rmcp reap the child process.
        actor.before_stop(|actor| {
            let server = actor.model.server_name.clone();
            let token = actor
                .model
                .connection
                .as_ref()
                .map(|c| c.service().cancellation_token());
            async move {
                if let Some(token) = token {
                    tracing::debug!(server = %server, "cancelling MCP session");
                    token.cancel();
                }
            }
        });
    }
}

fn register_handlers(actor: &mut ManagedActor<Idle, McpServerState>) {
    actor.mutate_on::<Connect>(|actor, _ctx| {
        if actor.model.status == ConnectionStatus::Ready {
            return Reply::ready();
        }

        let server = actor.model.server_name.clone();
        let connector = actor.model.connector.clone();
        let handle = actor.handle().clone();

        Reply::pending(async move {
            let Some(connector) = connector else {
                handle
                    .send(ConnectionFailed {
                        reason: McpError::connect_failed(
                            &server,
                            "no connector installed on this actor",
                        )
                        .to_string(),
                    })
                    .await;
                return;
            };

            match connector.connect().await {
                Ok(service) => {
                    tracing::info!(server = %server, "MCP session established");
                    handle
                        .send(ConnectionReady {
                            connection: Connection::new(service),
                        })
                        .await;
                }
                Err(err) => {
                    tracing::warn!(server = %server, error = %err, "MCP connection failed");
                    handle
                        .send(ConnectionFailed {
                            reason: err.to_string(),
                        })
                        .await;
                }
            }
        })
    });

    actor.mutate_on::<ConnectionReady>(|actor, ctx| {
        actor.model.connection = Some(ctx.message().connection.clone());
        actor.model.status = ConnectionStatus::Ready;
        let waiting = std::mem::take(&mut actor.model.waiting);

        Reply::pending(async move {
            for envelope in waiting {
                envelope.send(ReadyStatus { error: None }).await;
            }
        })
    });

    actor.mutate_on::<ConnectionFailed>(|actor, ctx| {
        let reason = ctx.message().reason.clone();
        let server = actor.model.server_name.clone();
        let is_reconnect = actor.model.is_reconnect;
        actor.model.connection = None;
        actor.model.status = ConnectionStatus::Failed(reason.clone());
        let waiting = std::mem::take(&mut actor.model.waiting);
        let handle = actor.handle().clone();

        Reply::pending(async move {
            for envelope in waiting {
                envelope
                    .send(ReadyStatus {
                        error: Some(reason.clone()),
                    })
                    .await;
            }

            // A failed *reconnect* must not strand this incarnation: without an
            // escalation here, a restarted actor whose reconnect fails would
            // park in `Failed` forever — tool calls hit the no-connection
            // branch, which deliberately does not re-escalate, so the
            // supervisor's backoff engine never gets another chance.
            //
            // The first incarnation is the exception, and deliberately so. It
            // is the one `launch_mcp_servers` is holding a readiness barrier
            // against, and that barrier is an `ask` to *this* mailbox. Stopping
            // here would close the inbox before the ask arrives whenever the
            // connect attempt loses its race with the launch task, turning a
            // precise "failed to spawn 'npx': No such file" into an
            // undeliverable-mailbox error — and starting a restart storm behind
            // a launch that is about to fail anyway. Nothing is stranded by
            // staying up: a first-incarnation failure fails the launch, so the
            // runtime is discarded rather than used.
            if is_reconnect {
                tracing::warn!(
                    server = %server,
                    reason = %reason,
                    "MCP reconnect failed; stopping so the supervisor retries with backoff"
                );
                handle.send(SystemSignal::Terminate).await;
            }
        })
    });

    actor.mutate_on::<AwaitReady>(|actor, ctx| match actor.model.status.clone() {
        ConnectionStatus::Ready => {
            let reply = ctx.reply_envelope();
            Reply::pending(async move {
                reply.send(ReadyStatus { error: None }).await;
            })
        }
        ConnectionStatus::Failed(reason) => {
            let reply = ctx.reply_envelope();
            Reply::pending(async move {
                reply
                    .send(ReadyStatus {
                        error: Some(reason),
                    })
                    .await;
            })
        }
        // Not answerable yet. Park the envelope; the caller waits on `ask`,
        // not on a timer.
        ConnectionStatus::Pending => {
            actor.model.waiting.push(ctx.reply_envelope());
            Reply::ready()
        }
    });

    actor.act_on::<DiscoverTools>(|actor, ctx| {
        let server = actor.model.server_name.clone();
        let allowlist = actor.model.enabled_tools.clone();
        let connection = actor.model.connection.clone();
        let reply = ctx.reply_envelope();

        Reply::pending(async move {
            let response = discover(&server, allowlist.as_ref(), connection).await;
            reply.send(response).await;
        })
    });

    actor.act_on::<CallMcpTool>(|actor, ctx| {
        let server = actor.model.server_name.clone();
        let timeout = actor.model.tool_timeout;
        let connection = actor.model.connection.clone();
        let handle = actor.handle().clone();
        let reply = ctx.reply_envelope();
        let remote_name = ctx.message().remote_name.clone();
        let arguments = ctx.message().arguments.clone();

        Reply::pending(async move {
            let (outcome, fatal) =
                call_tool(&server, connection, &remote_name, arguments, timeout).await;

            reply.send(McpCallOutcome { outcome }).await;

            if fatal {
                // Stop cleanly rather than panicking: a `Permanent` child is
                // restarted from a normal termination, so this is the
                // escalation path, and it keeps the mailbox drain intact so
                // sibling calls still get their replies.
                //
                // The signal is *sent*, not awaited through `stop()`. This
                // future is drained by the actor's own message loop, and
                // `stop()` waits for that loop's task to finish — awaiting it
                // here would have the actor wait for itself.
                tracing::warn!(
                    server = %server,
                    "MCP transport is gone; stopping so the supervisor reconnects"
                );
                handle.send(SystemSignal::Terminate).await;
            }
        })
    });
}

/// Lists every tool the server offers and converts them for the LLM.
async fn discover(
    server: &str,
    allowlist: Option<&Vec<String>>,
    connection: Option<Connection>,
) -> DiscoveredTools {
    let Some(connection) = connection else {
        return DiscoveredTools {
            tools: Vec::new(),
            error: Some(McpError::not_connected(server).to_string()),
        };
    };

    // `list_all_tools` runs the `next_cursor` pagination loop for us.
    let listing = detached(async move { connection.service().list_all_tools().await }).await;

    match listing {
        Some(Ok(tools)) => {
            let discovered = tools
                .iter()
                .map(|tool| DiscoveredTool {
                    remote_name: tool.name.to_string(),
                    definition: tool_definition(server, tool),
                })
                .collect();

            DiscoveredTools {
                tools: filter_enabled(server, allowlist, discovered),
                error: None,
            }
        }
        Some(Err(err)) => DiscoveredTools {
            tools: Vec::new(),
            error: Some(McpError::tool_discovery_failed(server, err.to_string()).to_string()),
        },
        None => DiscoveredTools {
            tools: Vec::new(),
            error: Some(
                McpError::tool_discovery_failed(server, "discovery task cancelled").to_string(),
            ),
        },
    }
}

/// Invokes one remote tool, returning the value for the LLM and whether the
/// failure means the connection is gone.
async fn call_tool(
    server: &str,
    connection: Option<Connection>,
    remote_name: &str,
    arguments: Map<String, Value>,
    timeout: Duration,
) -> (Result<Value, ToolError>, bool) {
    let display_name = prefixed_tool_name(server, remote_name);

    let Some(connection) = connection else {
        // Already disconnected: the supervisor is either restarting this
        // actor or about to. Escalating again would just churn.
        return (Err(McpError::not_connected(server).into()), false);
    };

    let mut params = CallToolRequestParams::new(remote_name.to_string());
    params.arguments = Some(arguments);

    // Kept out of the call future so the transport can be probed afterwards:
    // classifying the error alone loses the race where the client has not yet
    // noticed the peer went away and reports a timeout instead.
    let probe = connection.clone();
    let call =
        async move { tokio::time::timeout(timeout, connection.service().call_tool(params)).await };

    match detached(call).await {
        Some(Ok(Ok(result))) => (call_result_to_value(&display_name, result), false),
        Some(Ok(Err(err))) => {
            let fatal = is_transport_dead(&err) || probe.service().is_transport_closed();
            (
                Err(McpError::call_failed(server, remote_name, err.to_string()).into()),
                fatal,
            )
        }
        Some(Err(_elapsed)) => (
            Err(McpError::timeout(server, remote_name, timeout).into()),
            probe.service().is_transport_closed(),
        ),
        None => (
            Err(McpError::call_failed(server, remote_name, "call task cancelled").into()),
            false,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(remote: &str) -> DiscoveredTool {
        DiscoveredTool {
            remote_name: remote.to_string(),
            definition: ToolDefinition {
                name: prefixed_tool_name("fs", remote),
                description: String::new(),
                input_schema: json!({"type": "object"}),
            },
        }
    }

    #[test]
    fn no_allowlist_keeps_everything() {
        let kept = filter_enabled("fs", None, vec![tool("read"), tool("write")]);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn allowlist_keeps_only_named_tools() {
        let allowed = vec!["read".to_string()];
        let kept = filter_enabled("fs", Some(&allowed), vec![tool("read"), tool("write")]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].remote_name, "read");
    }

    #[test]
    fn empty_allowlist_keeps_nothing() {
        let allowed: Vec<String> = Vec::new();
        let kept = filter_enabled("fs", Some(&allowed), vec![tool("read")]);
        assert!(kept.is_empty());
    }

    #[test]
    fn allowlist_naming_an_unknown_tool_does_not_fail() {
        let allowed = vec!["read".to_string(), "teleport".to_string()];
        let kept = filter_enabled("fs", Some(&allowed), vec![tool("read")]);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn transport_errors_are_fatal() {
        assert!(is_transport_dead(&ServiceError::TransportClosed));
        assert!(is_transport_dead(&ServiceError::Cancelled { reason: None }));
    }

    #[test]
    fn protocol_errors_are_not_fatal() {
        assert!(!is_transport_dead(&ServiceError::UnexpectedResponse));
        assert!(!is_transport_dead(&ServiceError::Timeout {
            timeout: Duration::from_secs(1)
        }));
    }

    #[test]
    fn default_status_is_pending() {
        assert_eq!(ConnectionStatus::default(), ConnectionStatus::Pending);
    }

    #[tokio::test]
    async fn calling_without_a_connection_is_an_error_but_not_an_escalation() {
        let (outcome, fatal) =
            call_tool("fs", None, "read", Map::new(), Duration::from_secs(1)).await;

        assert!(outcome.is_err());
        assert!(!fatal, "a missing connection must not re-escalate");
    }

    #[tokio::test]
    async fn discovery_without_a_connection_reports_an_error() {
        let result = discover("fs", None, None).await;
        assert!(result.tools.is_empty());
        assert!(result.error.is_some());
    }
}
