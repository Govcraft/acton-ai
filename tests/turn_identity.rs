//! End-to-end tests for turn identity and the enriched lifecycle events.
//!
//! These drive the real stack — facade, prompt loop, provider actor, policy
//! gate, audit actor — against the scripted server in [`mock_llm`], with a
//! test-owned subscriber actor observing the broker. What they pin down is
//! the contract an external embedder (an ACP agent, say) builds on:
//!
//! 1. The turn's [`TurnId`] is caller-suppliable and is reported back on
//!    [`CollectedResponse`], so no claim/bind side table is ever needed.
//! 2. The tool bracket is total: every `ToolFinished` and every
//!    [`LLMStreamToolResult`] is preceded by exactly one `ToolStarted` with
//!    the same `tool_call_id` — **including calls the policy gate refused**.
//! 3. The events carry what a client renders: the proposed arguments on
//!    `ToolStarted`, the verdict and summary on `ToolFinished`, the turn on
//!    the result event.
//! 4. The audit trail records the same `tool_call_id` the live events
//!    carried, so a trail read later reconciles with a session watched live.
//!
//! # Determinism
//!
//! Nothing sleeps. `FlushBroadcasts` is the broker-side barrier and the `ask`
//! to the observer is the subscriber-side one: the observer's mailbox is FIFO,
//! so its answer cannot arrive before every event delivered ahead of the ask
//! has been folded in.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{MockServer, Round};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

// =============================================================================
// The observer
// =============================================================================

/// One event the observer saw, reduced to the fields these tests assert on.
#[derive(Debug, Clone)]
enum Observed {
    TurnStarted {
        turn_id: TurnId,
    },
    ToolStarted {
        turn_id: TurnId,
        tool_call_id: String,
        tool_name: String,
        arguments: Value,
    },
    ToolFinished {
        turn_id: TurnId,
        tool_call_id: String,
        success: bool,
        summary: String,
    },
    ToolResult {
        turn_id: TurnId,
        tool_call_id: String,
        success: bool,
    },
    TurnFinished {
        turn_id: TurnId,
    },
}

/// A test-owned subscriber for [`TurnLifecycle`] and [`LLMStreamToolResult`].
///
/// Test-owned is the point: it proves the enriched events are broadcast for
/// anyone on the broker, not just for a consumer wired in by the framework.
#[acton_actor]
struct Observer {
    events: Vec<Observed>,
}

#[acton_message]
struct GetObserved;

impl Request for GetObserved {
    type Response = ObservedEvents;
}

#[acton_message]
struct ObservedEvents {
    events: Vec<Observed>,
}

async fn spawn_observer(runtime: &mut ActorRuntime) -> ActorHandle {
    let mut builder = runtime.new_actor_with_name::<Observer>("observer".to_string());

    builder.mutate_on::<TurnLifecycle>(|actor, envelope| {
        // The enum and its variants are `#[non_exhaustive]`, which is why the
        // patterns carry `..` and the match a wildcard: this is exactly the
        // shape a downstream consumer writes.
        match envelope.message() {
            TurnLifecycle::TurnStarted { turn_id, .. } => {
                actor.model.events.push(Observed::TurnStarted {
                    turn_id: turn_id.clone(),
                });
            }
            TurnLifecycle::TurnFinished { turn_id, .. } => {
                actor.model.events.push(Observed::TurnFinished {
                    turn_id: turn_id.clone(),
                });
            }
            TurnLifecycle::ToolStarted {
                turn_id,
                tool_call_id,
                tool_name,
                arguments,
                ..
            } => {
                actor.model.events.push(Observed::ToolStarted {
                    turn_id: turn_id.clone(),
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                });
            }
            TurnLifecycle::ToolFinished {
                turn_id,
                tool_call_id,
                success,
                summary,
                ..
            } => {
                actor.model.events.push(Observed::ToolFinished {
                    turn_id: turn_id.clone(),
                    tool_call_id: tool_call_id.clone(),
                    success: *success,
                    summary: summary.clone(),
                });
            }
            _ => {}
        }
        Reply::ready()
    });

    builder.mutate_on::<LLMStreamToolResult>(|actor, envelope| {
        let message = envelope.message();
        actor.model.events.push(Observed::ToolResult {
            turn_id: message.turn_id.clone(),
            tool_call_id: message.tool_call_id.clone(),
            success: message.success,
        });
        Reply::ready()
    });

    builder.act_on::<GetObserved>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let events = actor.model.events.clone();
        Reply::pending(async move {
            reply.send(ObservedEvents { events }).await;
        })
    });

    // On the builder, before start: a subscription registered afterwards is
    // silently ignored.
    builder.handle().subscribe::<TurnLifecycle>().await;
    builder.handle().subscribe::<LLMStreamToolResult>().await;

    builder.start().await
}

/// Broker-side barrier: returns once everything broadcast before it has been
/// forwarded to every subscriber's inbox.
async fn flush_broadcasts(ai: &ActonAI) {
    ai.runtime()
        .broker()
        .ask(FlushBroadcasts)
        .await
        .expect("the broker must answer a flush");
}

/// Both barriers, then the observer's log.
async fn observed(ai: &ActonAI, observer: &ActorHandle) -> Vec<Observed> {
    flush_broadcasts(ai).await;
    observer
        .ask(GetObserved)
        .await
        .expect("the observer must answer")
        .events
}

/// A tool the scripted rounds can call.
fn tool_definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: "Echoes its argument back.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }),
    }
}

/// Asserts the bracket invariant for one call id over an event log: exactly
/// one `ToolStarted`, and every `ToolFinished`/`LLMStreamToolResult` for the
/// id comes after it.
fn assert_bracket(events: &[Observed], call_id: &str) {
    let started: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match event {
            Observed::ToolStarted { tool_call_id, .. } if tool_call_id == call_id => Some(index),
            _ => None,
        })
        .collect();
    assert_eq!(
        started.len(),
        1,
        "exactly one ToolStarted for {call_id}: {events:?}"
    );
    let opened_at = started[0];

    let finished: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match event {
            Observed::ToolFinished { tool_call_id, .. } if tool_call_id == call_id => Some(index),
            _ => None,
        })
        .collect();
    assert_eq!(
        finished.len(),
        1,
        "exactly one ToolFinished for {call_id}: {events:?}"
    );
    assert!(
        finished[0] > opened_at,
        "ToolFinished for {call_id} must come after its ToolStarted"
    );

    let results: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match event {
            Observed::ToolResult { tool_call_id, .. } if tool_call_id == call_id => Some(index),
            _ => None,
        })
        .collect();
    assert_eq!(
        results.len(),
        1,
        "exactly one LLMStreamToolResult for {call_id}: {events:?}"
    );
    assert!(
        results[0] > opened_at,
        "the result event for {call_id} must come after its ToolStarted"
    );
}

// =============================================================================
// 1. A caller-supplied TurnId round-trips through every event and the response
// =============================================================================

#[tokio::test]
async fn a_caller_supplied_turn_id_reaches_every_event_and_the_response() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})),
        Round::text("done"),
    ])
    .await;
    let mut ai = ActonAI::builder()
        .app_name("turn-id-supplied")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .launch()
        .await
        .expect("launching the runtime must succeed");
    let observer = spawn_observer(ai.runtime_mut()).await;

    let turn_id = TurnId::new();
    let response = ai
        .prompt("echo")
        .turn_id(turn_id.clone())
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(
        response.turn_id, turn_id,
        "the response must report the id the caller supplied"
    );

    let events = observed(&ai, &observer).await;

    // Every event of the turn carries the caller's id, so an embedder that
    // announced the turn before calling collect() needs no claim/bind table.
    for event in &events {
        let event_turn = match event {
            Observed::TurnStarted { turn_id }
            | Observed::TurnFinished { turn_id }
            | Observed::ToolStarted { turn_id, .. }
            | Observed::ToolFinished { turn_id, .. }
            | Observed::ToolResult { turn_id, .. } => turn_id,
        };
        assert_eq!(event_turn, &turn_id, "every event carries the caller's id");
    }

    // The start event carries what the model proposed, verbatim: this is the
    // `raw_input` an ACP `tool_call` event renders.
    let arguments = events
        .iter()
        .find_map(|event| match event {
            Observed::ToolStarted { arguments, .. } => Some(arguments.clone()),
            _ => None,
        })
        .expect("the tool call was announced");
    assert_eq!(arguments, json!({"value": "hi"}));

    // The finish event carries the verdict, so a consumer never has to join
    // it against the result broadcast to learn how the call ended.
    let (success, summary) = events
        .iter()
        .find_map(|event| match event {
            Observed::ToolFinished {
                success, summary, ..
            } => Some((*success, summary.clone())),
            _ => None,
        })
        .expect("the tool call was closed");
    assert!(success, "the echo tool ran and succeeded");
    assert!(!summary.is_empty(), "the summary previews the result");

    assert_bracket(&events, "call_1");

    ai.shutdown().await.expect("shutdown");
}

// =============================================================================
// 2. An unsupplied TurnId is minted and still reported
// =============================================================================

#[tokio::test]
async fn a_minted_turn_id_is_reported_and_matches_the_events() {
    let server = MockServer::start(vec![Round::text("ok")]).await;
    let mut ai = ActonAI::builder()
        .app_name("turn-id-minted")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .launch()
        .await
        .expect("launching the runtime must succeed");
    let observer = spawn_observer(ai.runtime_mut()).await;

    let response = ai
        .prompt("hello")
        .collect()
        .await
        .expect("the turn must complete");

    let events = observed(&ai, &observer).await;
    let started = events
        .iter()
        .find_map(|event| match event {
            Observed::TurnStarted { turn_id } => Some(turn_id.clone()),
            _ => None,
        })
        .expect("the turn was announced");

    assert_eq!(
        response.turn_id, started,
        "the response reports the same id the lifecycle events carried"
    );

    ai.shutdown().await.expect("shutdown");
}

// =============================================================================
// 3. A denied call still gets a full bracket, and the bracket never leaks
// =============================================================================

#[tokio::test]
async fn a_denied_call_gets_a_start_event_and_a_failed_finish() {
    // One round proposes two calls: one the policy refuses, one it admits.
    // The bracket invariant must hold for both — the denied call is the one
    // an ACP embedder previously had to synthesize a start event for.
    let server = MockServer::start(vec![
        Round::tool_call("call_deny", "bash", json!({"value": "rm -rf /"}))
            .with_tool_call("call_ok", "echo", json!({"value": "hi"})),
        Round::text("understood"),
    ])
    .await;
    let mut ai = ActonAI::builder()
        .app_name("turn-id-denied")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .tool_policy(ToolPolicy::new().deny(["bash"]))
        .launch()
        .await
        .expect("launching the runtime must succeed");
    let observer = spawn_observer(ai.runtime_mut()).await;

    let turn_id = TurnId::new();
    ai.prompt("clean up")
        .turn_id(turn_id.clone())
        .with_tool(tool_definition("bash"), |args| async move { Ok(args) })
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("a denial must not fail the turn");

    let events = observed(&ai, &observer).await;

    // The invariant, on both the refused and the executed call.
    assert_bracket(&events, "call_deny");
    assert_bracket(&events, "call_ok");

    // The refused call was announced with the arguments the model proposed —
    // published before the gate ran, so the gate's answer cannot erase it.
    let denied_start = events
        .iter()
        .find_map(|event| match event {
            Observed::ToolStarted {
                tool_call_id,
                arguments,
                turn_id,
                ..
            } if tool_call_id == "call_deny" => Some((arguments.clone(), turn_id.clone())),
            _ => None,
        })
        .expect("the denied call must still be announced");
    assert_eq!(denied_start.0, json!({"value": "rm -rf /"}));
    assert_eq!(denied_start.1, turn_id);

    // And closed as a failure whose summary names the refusal.
    let (denied_success, denied_summary) = events
        .iter()
        .find_map(|event| match event {
            Observed::ToolFinished {
                tool_call_id,
                success,
                summary,
                ..
            } if tool_call_id == "call_deny" => Some((*success, summary.clone())),
            _ => None,
        })
        .expect("the denied call must be closed");
    assert!(!denied_success, "a refusal is not a success");
    assert!(
        denied_summary.contains("bash"),
        "the summary names what was refused: {denied_summary}"
    );

    // The result event agrees, and is attributable to the turn.
    let denied_result = events
        .iter()
        .find_map(|event| match event {
            Observed::ToolResult {
                tool_call_id,
                success,
                turn_id,
            } if tool_call_id == "call_deny" => Some((*success, turn_id.clone())),
            _ => None,
        })
        .expect("the denied call must broadcast a result");
    assert!(!denied_result.0);
    assert_eq!(denied_result.1, turn_id);

    // The admitted sibling ran and closed as a success.
    let ok_finish = events.iter().find_map(|event| match event {
        Observed::ToolFinished {
            tool_call_id,
            success,
            ..
        } if tool_call_id == "call_ok" => Some(*success),
        _ => None,
    });
    assert_eq!(ok_finish, Some(true));

    ai.shutdown().await.expect("shutdown");
}

// =============================================================================
// 4. The audit trail carries the same call id the events carried
// =============================================================================

#[tokio::test]
async fn the_audit_entry_carries_the_call_id_the_events_carried() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let server = MockServer::start(vec![
        Round::tool_call("call_deny", "bash", json!({"value": "rm -rf /"}))
            .with_tool_call("call_ok", "echo", json!({"value": "hi"})),
        Round::text("understood"),
    ])
    .await;
    let mut ai = ActonAI::builder()
        .app_name("turn-id-audit")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .tool_policy(ToolPolicy::new().deny(["bash"]))
        .audit_to(&path)
        .launch()
        .await
        .expect("launching the runtime must succeed");
    let observer = spawn_observer(ai.runtime_mut()).await;

    let turn_id = TurnId::new();
    ai.prompt("clean up")
        .turn_id(turn_id.clone())
        .with_tool(tool_definition("bash"), |args| async move { Ok(args) })
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("a denial must not fail the turn");

    // The audit barrier: the head cannot answer before both entries are
    // sealed and written.
    let head = ai.audit_head().await.expect("the trail must report a head");
    assert_eq!(head.entries, 2, "one denied and one executed call");

    let events = observed(&ai, &observer).await;
    let contents = std::fs::read_to_string(&path).expect("the trail must exist");
    let entries = acton_ai::audit::parse_entries(&contents).expect("the trail must parse");

    // Each entry's call id is the one the live events carried for that tool,
    // which is what lets an investigator join the trail to a watched session.
    for entry in &entries {
        let event_call_id = events
            .iter()
            .find_map(|event| match event {
                Observed::ToolStarted {
                    tool_call_id,
                    tool_name,
                    ..
                } if tool_name == &entry.tool_name => Some(tool_call_id.clone()),
                _ => None,
            })
            .expect("every audited tool was announced");
        assert_eq!(
            entry.tool_call_id, event_call_id,
            "the trail and the events must name the same call"
        );
        assert_eq!(
            entry.turn_id, turn_id,
            "the trail records the caller's turn id"
        );
    }

    let denied = entries
        .iter()
        .find(|entry| entry.tool_name == "bash")
        .expect("the denied call is recorded");
    assert_eq!(denied.tool_call_id, "call_deny");
    assert!(matches!(denied.outcome, AuditOutcome::Denied { .. }));

    ai.shutdown().await.expect("shutdown");
}

// =============================================================================
// 5. Stream callbacks receive the turn and round identity
// =============================================================================

#[tokio::test]
async fn stream_callbacks_receive_the_turn_and_round_identity() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})),
        Round::text("done"),
    ])
    .await;
    let ai = ActonAI::builder()
        .app_name("turn-id-callbacks")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let starts: Arc<Mutex<Vec<StreamContext>>> = Arc::new(Mutex::new(Vec::new()));
    let ends: Arc<Mutex<Vec<StreamContext>>> = Arc::new(Mutex::new(Vec::new()));
    let starts_sink = Arc::clone(&starts);
    let ends_sink = Arc::clone(&ends);

    let turn_id = TurnId::new();
    ai.prompt("echo")
        .turn_id(turn_id.clone())
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .on_start(move |context| {
            starts_sink
                .lock()
                .expect("no other holder panics")
                .push(context.clone());
        })
        .on_end(move |context, _reason| {
            ends_sink
                .lock()
                .expect("no other holder panics")
                .push(context.clone());
        })
        .collect()
        .await
        .expect("the turn must complete");

    let starts = starts.lock().expect("the turn is over").clone();
    let ends = ends.lock().expect("the turn is over").clone();

    // A tool turn is two provider rounds, so each callback fired twice.
    assert_eq!(starts.len(), 2, "one start per round: {starts:?}");
    assert_eq!(ends.len(), 2, "one end per round: {ends:?}");

    // Both rounds belong to the caller's turn…
    for context in starts.iter().chain(ends.iter()) {
        assert_eq!(context.turn_id, turn_id);
    }
    // …under distinct round identities, and the end of a round names the
    // same round its start did.
    assert_ne!(
        starts[0].correlation_id, starts[1].correlation_id,
        "each round has its own correlation id"
    );
    assert_eq!(starts[0], ends[0]);
    assert_eq!(starts[1], ends[1]);

    ai.shutdown().await.expect("shutdown");
}
