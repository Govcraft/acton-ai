//! The introspection wire protocol: one definition, both ends.
//!
//! The server actor and the `acton-ai status` client are compiled from these
//! same types, so the two can never drift into disagreeing about the format.
//! They are also what `--json` prints, which makes the JSON rendering of a
//! status the wire format rather than a parallel structure that has to be
//! kept in step by hand.
//!
//! # Compatibility policy
//!
//! Every response carries [`SCHEMA_VERSION`]. Evolution is **additive only**:
//!
//! - New fields may be added; they must be `Option` or carry `#[serde(default)]`
//!   so an older client keeps deserializing a newer server's reply.
//! - Fields are never removed, renamed, or retyped, and the meaning of an
//!   existing field never changes.
//! - Anything that cannot be done additively is a new `SCHEMA_VERSION` and a
//!   deliberate break, announced in the changelog.
//!
//! A client should branch on `schema_version` rather than assume, because the
//! process it is talking to is a *different binary* that was quite possibly
//! installed at a different time.
//!
//! # Message type names
//!
//! Every [`RemoteRequest::MESSAGE_TYPE`] is prefixed `acton_ai.`. The IPC
//! registry is a per-process namespace shared with whatever the embedding
//! application registers, and a bare `GetStatus` is a name somebody else will
//! plausibly want.

use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};

/// The version of every payload in this module.
///
/// See the module docs for what a client may assume when it reads this.
pub const SCHEMA_VERSION: u32 = 1;

/// The name the introspection actor is exposed under on the IPC socket.
///
/// Stable: it is half of the address every `acton-ai status` invocation uses.
pub const INTROSPECTION_ACTOR_NAME: &str = "introspection";

// =============================================================================
// Requests
// =============================================================================

/// Asks a running process for a full [`StatusReport`].
///
/// Read-only: it never changes what the process is doing.
#[acton_message(ipc)]
pub struct GetStatus;

impl Request for GetStatus {
    type Response = StatusReport;
}

impl RemoteRequest for GetStatus {
    const MESSAGE_TYPE: &'static str = "acton_ai.GetStatus";
}

/// Stops the process admitting new turns, pending a [`Resume`].
///
/// In-flight turns are never interrupted.
#[acton_message(ipc)]
pub struct Pause;

impl Request for Pause {
    type Response = AdmissionAck;
}

impl RemoteRequest for Pause {
    const MESSAGE_TYPE: &'static str = "acton_ai.Pause";
}

/// Admits new turns again, from either refusing state.
#[acton_message(ipc)]
pub struct Resume;

impl Request for Resume {
    type Response = AdmissionAck;
}

impl RemoteRequest for Resume {
    const MESSAGE_TYPE: &'static str = "acton_ai.Resume";
}

/// Stops admitting new turns and reports how much work is still in flight.
///
/// The drain is complete when [`AdmissionAck::in_flight_turns`] reaches zero.
/// A caller that wants to *wait* for that re-asks; see
/// [`crate::introspection`] for why there is no push notification in v1.
#[acton_message(ipc)]
pub struct Drain;

impl Request for Drain {
    type Response = AdmissionAck;
}

impl RemoteRequest for Drain {
    const MESSAGE_TYPE: &'static str = "acton_ai.Drain";
}

// =============================================================================
// Responses
// =============================================================================

/// The answer to [`Pause`], [`Resume`], and [`Drain`].
///
/// Carries the resulting state rather than a bare acknowledgement so a caller
/// never has to follow a command with a query to find out what it achieved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionAck {
    /// See the module docs on compatibility.
    pub schema_version: u32,
    /// The admission state now in force, as
    /// [`AdmissionState::as_str`](crate::introspection::AdmissionState::as_str).
    pub admission: String,
    /// Turns still running when the command was applied.
    ///
    /// Zero in the reply to a [`Drain`] means the drain is already complete.
    pub in_flight_turns: u64,
}

impl AdmissionAck {
    /// Builds an ack at the current [`SCHEMA_VERSION`].
    #[must_use]
    pub fn new(admission: impl Into<String>, in_flight_turns: u64) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            admission: admission.into(),
            in_flight_turns,
        }
    }

    /// Whether a drain that produced this ack has already finished.
    #[must_use]
    pub fn is_drained(&self) -> bool {
        self.in_flight_turns == 0
    }
}

/// Everything a running process will say about itself.
///
/// Assembled on demand: the process is asked, and it asks its own providers
/// and accountant. Nothing here is cached, so nothing here can be stale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusReport {
    /// See the module docs on compatibility.
    pub schema_version: u32,
    /// The `app_name` the runtime was launched with.
    pub app_name: String,
    /// The version of the `acton-ai` crate the process is running.
    ///
    /// The status of a long-lived process is one of the few places the
    /// deployed version is genuinely in question.
    pub crate_version: String,
    /// Process ID, so an operator can go from a status to `kill`, `gdb`, or
    /// a journal query without a second lookup.
    pub pid: u32,
    /// Seconds since the runtime finished launching.
    pub uptime_secs: u64,
    /// Admission state, as
    /// [`AdmissionState::as_str`](crate::introspection::AdmissionState::as_str).
    pub admission: String,
    /// Turns running right now.
    pub active_turns: u64,
    /// Tool calls executing right now, across every active turn.
    pub in_flight_tool_calls: u64,
    /// Turns that have started since launch, including the active ones.
    pub turns_started: u64,
    /// Turns refused because admission was closed.
    ///
    /// Nonzero here with `admission: "running"` is the signature of a pause
    /// that was resumed after callers had already been turned away.
    pub turns_refused: u64,
    /// Histories compacted since launch, across every turn.
    ///
    /// Nonzero means the runtime has been rewriting histories mid-turn to
    /// keep them inside the context window — the first thing to check when a
    /// model appears to have forgotten something it was told.
    ///
    /// Added after `SCHEMA_VERSION` 1, so it carries `#[serde(default)]`: an
    /// older server's reply deserializes with a zero here.
    #[serde(default)]
    pub turns_compacted: u64,
    /// Every configured provider, in name order.
    pub providers: Vec<ProviderStatus>,
    /// Every configured MCP server, in name order. Empty when none.
    pub mcp_servers: Vec<McpServerStatus>,
    /// Usage totals, or `None` when usage tracking is switched off.
    ///
    /// `None` means "nobody is counting", never "nothing was spent" — zeros
    /// would read as the latter.
    pub usage: Option<UsageSummary>,
}

/// One provider's line in a [`StatusReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderStatus {
    /// The configured name — the runtime's key for it, not the vendor.
    pub name: String,
    /// The model this provider serves.
    pub model: String,
    /// Circuit-breaker health, rendered from
    /// [`ProviderHealth`](crate::llm::ProviderHealth).
    pub health: String,
    /// Whether the breaker is open, so a caller can filter on the one bit
    /// that matters without parsing `health`.
    pub circuit_open: bool,
    /// The configured failover chain, in order. Empty when none.
    pub failover: Vec<String>,
}

/// One MCP server's line in a [`StatusReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerStatus {
    /// The configured server name.
    pub name: String,
    /// Whether the supervised connection currently resolves to a live actor.
    pub up: bool,
    /// The supervisor's lifecycle state — `running`, `restarting`, `down`,
    /// and so on.
    pub state: String,
    /// Which incarnation is current. Starts at 0; every supervised restart
    /// increments it, so a rising number is a server that keeps dying.
    pub generation: u64,
    /// Restarts the supervisor has recorded inside its current window.
    pub restarts_in_window: u64,
}

/// Spend and token totals in a [`StatusReport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageSummary {
    /// Input tokens billed across every provider.
    pub input_tokens: u64,
    /// Output tokens billed across every provider.
    pub output_tokens: u64,
    /// Provider requests counted.
    pub requests: u64,
    /// Grand-total cost in dollars, or `None`.
    ///
    /// `None` means at least one provider with recorded usage has no
    /// configured pricing, so no honest total exists — the same rule
    /// [`UsageSnapshot::cost`](crate::accounting::UsageSnapshot::cost)
    /// follows.
    pub total_cost_usd: Option<f64>,
    /// Where spending stands against the configured caps, or `None` when no
    /// budget is configured.
    pub budget: Option<BudgetStanding>,
}

/// Where spending stands against one configured cap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetStanding {
    /// The process-wide ceiling in integer micro-USD, when one is set.
    pub total_limit_microusd: Option<u64>,
    /// Spend against that ceiling, in micro-USD.
    pub total_spent_microusd: u64,
    /// Whether any configured cap has been reached, in which case the
    /// runtime is already refusing requests pre-flight.
    pub exceeded: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> StatusReport {
        StatusReport {
            schema_version: SCHEMA_VERSION,
            app_name: "my-agent".to_string(),
            crate_version: "0.30.0".to_string(),
            pid: 4321,
            uptime_secs: 90,
            admission: "running".to_string(),
            active_turns: 2,
            in_flight_tool_calls: 1,
            turns_started: 17,
            turns_refused: 3,
            turns_compacted: 2,
            providers: vec![ProviderStatus {
                name: "claude".to_string(),
                model: "claude-sonnet-4".to_string(),
                health: "closed".to_string(),
                circuit_open: false,
                failover: vec!["local".to_string()],
            }],
            mcp_servers: vec![McpServerStatus {
                name: "files".to_string(),
                up: true,
                state: "running".to_string(),
                generation: 2,
                restarts_in_window: 1,
            }],
            usage: Some(UsageSummary {
                input_tokens: 1_000,
                output_tokens: 250,
                requests: 4,
                total_cost_usd: Some(0.0125),
                budget: Some(BudgetStanding {
                    total_limit_microusd: Some(5_000_000),
                    total_spent_microusd: 12_500,
                    exceeded: false,
                }),
            }),
        }
    }

    #[test]
    fn a_status_report_round_trips_through_json() {
        let json = serde_json::to_string(&report()).expect("serializes");
        let decoded: StatusReport = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(decoded, report());
    }

    #[test]
    fn a_status_report_carries_its_schema_version_on_the_wire() {
        let value = serde_json::to_value(report()).expect("serializes");
        // A client's very first act is to branch on this field, so its
        // presence under this exact name is part of the contract.
        assert_eq!(value["schema_version"], serde_json::json!(1));
    }

    #[test]
    fn an_admission_ack_round_trips_through_json() {
        let ack = AdmissionAck::new("draining", 2);
        let json = serde_json::to_string(&ack).expect("serializes");
        assert_eq!(
            serde_json::from_str::<AdmissionAck>(&json).expect("deserializes"),
            ack
        );
    }

    #[test]
    fn a_new_ack_stamps_the_current_schema_version() {
        assert_eq!(AdmissionAck::new("running", 0).schema_version, 1);
    }

    #[test]
    fn an_ack_is_drained_only_at_zero_in_flight() {
        assert!(AdmissionAck::new("draining", 0).is_drained());
        // One turn still running must not read as a completed drain: that is
        // precisely the bug that would let systemd kill live work.
        assert!(!AdmissionAck::new("draining", 1).is_drained());
    }

    #[test]
    fn untracked_usage_is_absent_rather_than_zero() {
        let mut without = report();
        without.usage = None;
        let value = serde_json::to_value(&without).expect("serializes");
        // Null, not a zeroed summary: "nobody counted" and "nothing was
        // spent" are different answers and must not be conflated.
        assert!(value["usage"].is_null());
    }

    #[test]
    fn the_message_types_are_namespaced_and_distinct() {
        let names = [
            GetStatus::MESSAGE_TYPE,
            Pause::MESSAGE_TYPE,
            Resume::MESSAGE_TYPE,
            Drain::MESSAGE_TYPE,
        ];
        for name in names {
            assert!(
                name.starts_with("acton_ai."),
                "{name} would collide in an embedder's registry"
            );
        }
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(
            unique.len(),
            names.len(),
            "message type names must be unique"
        );
    }

    #[test]
    fn the_exposed_actor_name_is_the_documented_one() {
        // The CLI addresses this string; changing it breaks every installed
        // client against every running server.
        assert_eq!(INTROSPECTION_ACTOR_NAME, "introspection");
    }
}
