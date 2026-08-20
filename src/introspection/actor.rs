//! The [`IntrospectionActor`]: passive observer in, status out.
//!
//! # Why an actor
//!
//! Two reasons, and they are different reasons.
//!
//! *Observation* needs to be an actor because the in-flight picture is built
//! from broker broadcasts, and the broker delivers to actors. Deriving it
//! passively — rather than having the prompt loop report *to* a status object
//! — is what keeps introspection off the turn's critical path entirely. The
//! prompt loop publishes a [`TurnLifecycle`] and moves on; whether anyone is
//! listening is not its problem.
//!
//! *Serving* needs to be an actor because acton exposes actors over IPC. The
//! same handle that answers a local `ask` answers a remote one.
//!
//! # Handler discipline
//!
//! - The four [`TurnLifecycle`] handlers are `mutate_on`: they are the only
//!   writers of the in-flight sets, and the sets must be updated one at a
//!   time or a start and a finish can interleave into a lost count.
//! - [`GetStatus`] is `act_on`: it reads the model and then does IO — asking
//!   each provider for its health, asking the accountant for usage. Several
//!   statuses can be assembled concurrently, and, decisively, **an `ask` from
//!   `mutate_on` would deadlock**: the handler holds the actor's turn while
//!   waiting for a reply that may need this same actor to make progress.
//!   Everything the reply future needs is cloned out of the model first.
//! - [`Pause`]/[`Resume`]/[`Drain`] are `act_on` as well: they write through
//!   the [`AdmissionGate`], whose own atomic does the serialising, and read
//!   the in-flight count for their ack.

use std::collections::HashMap;
use std::time::Instant;

use acton_reactive::prelude::*;

use crate::accounting::{GetUsage, UsageSnapshot};
use crate::introspection::wire::{
    AdmissionAck, BudgetStanding, Drain, GetStatus, McpServerStatus, Pause, ProviderStatus, Resume,
    StatusReport, UsageSummary, SCHEMA_VERSION,
};
use crate::introspection::AdmissionGate;
use crate::llm::{CheckHealth, ProviderHealth};
use crate::messages::TurnLifecycle;
use crate::types::TurnId;

/// How long a status assembly waits on any one collaborator.
///
/// The whole point of `acton-ai status` is to work when something is wrong,
/// so a provider actor that is wedged must degrade the report rather than
/// hang it. A caller that gets no answer learns nothing; a caller that gets
/// "this provider did not answer in time" has learned the most important
/// thing in the process.
const COLLABORATOR_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// What a status assembly needs to ask other actors for.
///
/// Cloned out of the model before any await, because a handler must not hold
/// a borrow of the model across one.
#[derive(Debug, Clone, Default)]
pub(crate) struct StatusSources {
    /// Provider handles keyed by configured name.
    pub(crate) providers: HashMap<String, ActorHandle>,
    /// The model each provider serves, keyed by the same name.
    pub(crate) provider_models: HashMap<String, String>,
    /// Failover chains, keyed by the same name. Absent means none configured.
    pub(crate) provider_failover: HashMap<String, Vec<String>>,
    /// The cost accountant, when usage tracking is on.
    pub(crate) accountant: Option<ActorHandle>,
    /// MCP servers under supervision, keyed by configured name.
    ///
    /// The supervision handles are copied out of
    /// [`McpTools`](crate::mcp::McpTools) rather than the whole registry being
    /// held: the tool definitions and executors in there are large, are not
    /// `Clone`, and say nothing a status report needs. A `SupervisedChild` is
    /// a cheap handle over a watch channel, which is all this reads.
    pub(crate) mcp: HashMap<String, SupervisedChild>,
    /// The runtime's `app_name`.
    pub(crate) app_name: String,
}

/// Observes turn lifecycle broadcasts and answers introspection requests.
#[acton_actor]
pub struct IntrospectionActor {
    /// Turns currently running.
    ///
    /// A set rather than a counter so a duplicate or out-of-order broadcast
    /// corrects itself instead of permanently skewing the count. An operator
    /// who cannot trust the in-flight number cannot trust a drain.
    active_turns: std::collections::HashSet<TurnId>,
    /// Tool calls currently executing, keyed by provider-assigned call ID.
    in_flight_tools: std::collections::HashSet<String>,
    /// Turns started since launch.
    turns_started: u64,
    /// Turns refused because admission was closed.
    turns_refused: u64,
    /// When the runtime finished launching.
    ///
    /// `Option` because an actor model must be `Default` and `Instant` is
    /// not. Seeded on the builder before `start()`, so it is always `Some`
    /// by the time a handler runs.
    started_at: Option<Instant>,
    /// Shared with the prompt loop; see [`AdmissionGate`] for why this one
    /// piece of state is shared rather than messaged.
    gate: AdmissionGate,
    /// Everything a status needs to ask for.
    sources: StatusSources,
}

impl IntrospectionActor {
    /// Spawns the actor, subscribed **on the builder** to [`TurnLifecycle`].
    ///
    /// Subscribing after `start()` is silently ignored by acton-reactive,
    /// which would leave an actor that runs happily and reports every process
    /// as idle — the most dangerous possible lie for a drain to tell.
    pub(crate) async fn spawn(
        runtime: &mut ActorRuntime,
        gate: AdmissionGate,
        sources: StatusSources,
    ) -> ActorHandle {
        let mut builder =
            runtime.new_actor_with_name::<IntrospectionActor>("introspection".to_string());

        builder.model.gate = gate;
        builder.model.sources = sources;
        builder.model.started_at = Some(Instant::now());

        // The in-flight picture. `mutate_on`: one writer, applied in order.
        builder.mutate_on::<TurnLifecycle>(|actor, envelope| {
            match envelope.message() {
                TurnLifecycle::TurnStarted { turn_id } => {
                    if actor.model.active_turns.insert(turn_id.clone()) {
                        actor.model.turns_started = actor.model.turns_started.saturating_add(1);
                    }
                }
                TurnLifecycle::TurnFinished { turn_id } => {
                    actor.model.active_turns.remove(turn_id);
                    // Belt and braces: a turn that ended while a tool was
                    // still recorded would otherwise leak that tool forever.
                    // Unreachable through the prompt loop, which brackets
                    // every tool it starts, but the cost of being wrong here
                    // is a drain that never completes.
                    actor
                        .model
                        .in_flight_tools
                        .retain(|id| !id.starts_with(&tool_scope(turn_id)));
                }
                TurnLifecycle::ToolStarted {
                    turn_id,
                    tool_call_id,
                    ..
                } => {
                    actor
                        .model
                        .in_flight_tools
                        .insert(scoped_tool_id(turn_id, tool_call_id));
                }
                TurnLifecycle::ToolFinished {
                    turn_id,
                    tool_call_id,
                    ..
                } => {
                    actor
                        .model
                        .in_flight_tools
                        .remove(&scoped_tool_id(turn_id, tool_call_id));
                }
                TurnLifecycle::TurnRefused => {
                    actor.model.turns_refused = actor.model.turns_refused.saturating_add(1);
                }
            }
            Reply::ready()
        });

        // Status assembly. `act_on` and never `mutate_on`: this handler asks
        // other actors, and an ask from a mutating handler deadlocks.
        builder.act_on::<GetStatus>(|actor, envelope| {
            let reply = envelope.reply_envelope();
            // Everything the future needs, cloned out before the await.
            let snapshot = ModelSnapshot::of(&actor.model);
            let sources = actor.model.sources.clone();

            Reply::pending(async move {
                let report = assemble(snapshot, sources).await;
                reply.send(report).await;
            })
        });

        builder.act_on::<Pause>(|actor, envelope| {
            let reply = envelope.reply_envelope();
            let state = actor.model.gate.pause();
            let in_flight = actor.model.active_turns.len() as u64;
            tracing::info!(in_flight, "turn admission paused over introspection");
            Reply::pending(async move {
                reply
                    .send(AdmissionAck::new(state.as_str(), in_flight))
                    .await;
            })
        });

        builder.act_on::<Resume>(|actor, envelope| {
            let reply = envelope.reply_envelope();
            let state = actor.model.gate.resume();
            let in_flight = actor.model.active_turns.len() as u64;
            tracing::info!(in_flight, "turn admission resumed over introspection");
            Reply::pending(async move {
                reply
                    .send(AdmissionAck::new(state.as_str(), in_flight))
                    .await;
            })
        });

        builder.act_on::<Drain>(|actor, envelope| {
            let reply = envelope.reply_envelope();
            let state = actor.model.gate.drain();
            let in_flight = actor.model.active_turns.len() as u64;
            tracing::info!(in_flight, "drain started over introspection");
            Reply::pending(async move {
                reply
                    .send(AdmissionAck::new(state.as_str(), in_flight))
                    .await;
            })
        });

        builder.handle().subscribe::<TurnLifecycle>().await;

        builder.start().await
    }
}

/// The prefix every tool of `turn_id` is recorded under.
fn tool_scope(turn_id: &TurnId) -> String {
    format!("{turn_id}\u{1f}")
}

/// Tool call IDs are only unique within the turn that issued them, so the
/// in-flight set keys on both. The separator is ASCII unit separator, which
/// cannot occur in a TypeID.
fn scoped_tool_id(turn_id: &TurnId, tool_call_id: &str) -> String {
    format!("{turn_id}\u{1f}{tool_call_id}")
}

/// The parts of the model a status needs, copied out before any await.
#[derive(Debug, Clone)]
struct ModelSnapshot {
    active_turns: u64,
    in_flight_tool_calls: u64,
    turns_started: u64,
    turns_refused: u64,
    uptime_secs: u64,
    admission: String,
}

impl ModelSnapshot {
    fn of(model: &IntrospectionActor) -> Self {
        Self {
            active_turns: model.active_turns.len() as u64,
            in_flight_tool_calls: model.in_flight_tools.len() as u64,
            turns_started: model.turns_started,
            turns_refused: model.turns_refused,
            uptime_secs: model
                .started_at
                .map_or(0, |started| started.elapsed().as_secs()),
            admission: model.gate.state().as_str().to_string(),
        }
    }
}

/// Builds the report: local state, then everything that must be asked for.
async fn assemble(snapshot: ModelSnapshot, sources: StatusSources) -> StatusReport {
    StatusReport {
        schema_version: SCHEMA_VERSION,
        app_name: sources.app_name.clone(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        pid: std::process::id(),
        uptime_secs: snapshot.uptime_secs,
        admission: snapshot.admission,
        active_turns: snapshot.active_turns,
        in_flight_tool_calls: snapshot.in_flight_tool_calls,
        turns_started: snapshot.turns_started,
        turns_refused: snapshot.turns_refused,
        providers: provider_statuses(&sources).await,
        mcp_servers: mcp_statuses(&sources),
        usage: usage_summary(&sources).await,
    }
}

/// Asks every provider for its breaker health, in name order.
async fn provider_statuses(sources: &StatusSources) -> Vec<ProviderStatus> {
    let mut names: Vec<&String> = sources.providers.keys().collect();
    names.sort();

    let mut statuses = Vec::with_capacity(names.len());
    for name in names {
        let health: Option<ProviderHealth> = match sources.providers.get(name) {
            Some(handle) => handle
                .ask_with_timeout(CheckHealth, COLLABORATOR_TIMEOUT)
                .await
                .ok(),
            None => None,
        };
        statuses.push(ProviderStatus {
            name: name.clone(),
            model: sources
                .provider_models
                .get(name)
                .cloned()
                .unwrap_or_default(),
            health: health.as_ref().map_or_else(
                // Naming the silence beats an empty string: a provider that
                // will not answer a health ask is itself the finding.
                || "unreachable (no answer within 2s)".to_string(),
                ProviderHealth::to_string,
            ),
            circuit_open: health.is_some_and(|health| health.is_open()),
            failover: sources
                .provider_failover
                .get(name)
                .cloned()
                .unwrap_or_default(),
        });
    }
    statuses
}

/// Reads each MCP server's supervision snapshot, in name order.
///
/// Free of message passing: a supervisor publishes its status to a watch
/// channel, so reading one costs neither a round trip nor a block on the
/// supervisor's own task.
fn mcp_statuses(sources: &StatusSources) -> Vec<McpServerStatus> {
    let mut statuses: Vec<McpServerStatus> = sources
        .mcp
        .iter()
        .map(|(name, child)| {
            let status = child.status();
            McpServerStatus {
                name: name.clone(),
                // The question an operator is asking is "can a tool call
                // reach it right now", and that is exactly what resolving
                // the current incarnation answers.
                up: child.current().is_some(),
                state: status.state().to_string(),
                generation: status.generation().get(),
                restarts_in_window: status.restarts_in_window() as u64,
            }
        })
        .collect();
    statuses.sort_by(|a, b| a.name.cmp(&b.name));
    statuses
}

/// Asks the accountant for usage, when there is one.
async fn usage_summary(sources: &StatusSources) -> Option<UsageSummary> {
    let accountant = sources.accountant.as_ref()?;
    let snapshot: UsageSnapshot = accountant
        .ask_with_timeout(GetUsage, COLLABORATOR_TIMEOUT)
        .await
        .ok()?;

    Some(UsageSummary {
        input_tokens: snapshot.totals.input_tokens,
        output_tokens: snapshot.totals.output_tokens,
        requests: snapshot.requests,
        total_cost_usd: snapshot.total_usd(),
        budget: snapshot.budget.as_ref().map(|budget| BudgetStanding {
            total_limit_microusd: budget.total.map(|total| total.limit_microusd),
            total_spent_microusd: budget.total.map_or(0, |total| total.spent_microusd),
            exceeded: budget.is_exceeded(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_ids_are_scoped_to_their_turn() {
        let one = TurnId::new();
        let two = TurnId::new();

        // Providers only promise call IDs unique within a response, so two
        // concurrent turns can legitimately both be running "call_1". Keying
        // on the bare ID would let one turn's finish clear the other's tool.
        assert_ne!(
            scoped_tool_id(&one, "call_1"),
            scoped_tool_id(&two, "call_1")
        );
    }

    #[test]
    fn a_scoped_tool_id_starts_with_its_turn_scope() {
        let turn = TurnId::new();
        // The sweep on TurnFinished matches by this prefix, so the two must
        // agree or the sweep silently clears nothing.
        assert!(scoped_tool_id(&turn, "call_1").starts_with(&tool_scope(&turn)));
    }

    #[test]
    fn one_turns_scope_does_not_match_anothers_tools() {
        let one = TurnId::new();
        let two = TurnId::new();
        assert!(!scoped_tool_id(&two, "call_1").starts_with(&tool_scope(&one)));
    }
}
