//! End-to-end tests for turn identity and the tool observation bracket.
//!
//! These exist because of what a downstream consumer has to do without them.
//! An ACP daemon driving an IDE client maps our events onto a protocol where
//! a tool result may only follow a tool call it already announced, and where
//! everything must be attributed to a session. If `collect()` did not return
//! the turn, that daemon would need a claim/bind router to guess which turn a
//! response belonged to; if a policy-refused call published a result with no
//! preceding start, the daemon would have to synthesize a start it never saw.
//! Both are the kind of workaround that is invisible until it is wrong.
//!
//! So the invariant under test is total, not typical:
//!
//! > Every `ToolFinished` and every `LLMStreamToolResult` is preceded by
//! > exactly one `ToolStarted` carrying the same `turn_id` and the same
//! > `tool_call_id` — **including the calls policy refused.**
//!
//! # Determinism
//!
//! Nothing sleeps. Two barriers do all the waiting:
//!
//! - A **completed `collect()`**. The prompt loop cannot send round N+1
//!   before round N's tool results exist, so a returned response proves the
//!   whole turn ran.
//! - **`broker.ask(FlushBroadcasts)`**. `broadcast().await` only proves a
//!   message reached the *broker*, not that a subscriber handled it. The
//!   broker's reply to `FlushBroadcasts` cannot arrive until every earlier
//!   broadcast is sitting in each subscriber's inbox, and because mailboxes
//!   are FIFO the `ask` that reads the recorder is necessarily processed
//!   behind it.
//!
//! Without the second barrier these would pass on an idle machine and flake
//! on a loaded one.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use serde_json::{json, Value};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// =============================================================================
// Fixtures
// =============================================================================

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

/// Launches a runtime pointed at `server` with `policy` in force.
///
/// The no-policy case goes through [`mock_llm::runtime_pointed_at`] instead,
/// so the two differ in exactly one thing.
async fn runtime_with_policy(server: &MockServer, app_name: &str, policy: ToolPolicy) -> ActonAI {
    ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .tool_policy(policy)
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

// =============================================================================
// The recorder
// =============================================================================

/// One observed event, flattened into owned data.
///
/// Flattened rather than stored as the message itself because the assertions
/// are about *relationships between* events — ordering, and which ids pair up
/// — and comparing whole messages would drag in fields no test cares about.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Seen {
    TurnStarted {
        turn_id: String,
    },
    TurnFinished {
        turn_id: String,
    },
    ToolStarted {
        turn_id: String,
        tool_call_id: String,
        tool_name: String,
        arguments: String,
    },
    ToolFinished {
        turn_id: String,
        tool_call_id: String,
        success: bool,
        summary: String,
    },
    ToolResult {
        turn_id: String,
        tool_call_id: String,
        success: bool,
    },
}

impl Seen {
    /// The turn every event belongs to, for the tests that assert one turn.
    fn turn_id(&self) -> &str {
        match self {
            Self::TurnStarted { turn_id }
            | Self::TurnFinished { turn_id }
            | Self::ToolStarted { turn_id, .. }
            | Self::ToolFinished { turn_id, .. }
            | Self::ToolResult { turn_id, .. } => turn_id,
        }
    }
}

/// Records every turn-scoped event on the broker, in arrival order.
///
/// A test-owned subscriber, which is the point twice over: it proves the
/// events are broadcast for *anyone* rather than only reaching the facade's
/// own bookkeeping, and it is the barrier, since the `ask` that reads it
/// cannot be answered before the events ahead of it in the same FIFO inbox
/// have been handled.
#[acton_actor]
struct EventRecorder {
    seen: Vec<Seen>,
}

#[acton_message]
struct GetSeen;

impl Request for GetSeen {
    type Response = SeenEvents;
}

#[acton_message]
struct SeenEvents {
    events: Vec<Seen>,
}

async fn spawn_recorder(runtime: &mut ActorRuntime) -> ActorHandle {
    let mut builder = runtime.new_actor_with_name::<EventRecorder>("event_recorder".to_string());

    builder.mutate_on::<TurnLifecycle>(|actor, envelope| {
        // `TurnLifecycle` and its variants are `#[non_exhaustive]`, so these
        // arms bind by name and ignore the rest on purpose: a variant added
        // later must not break this recorder.
        let recorded = match envelope.message() {
            TurnLifecycle::TurnStarted { turn_id, .. } => Some(Seen::TurnStarted {
                turn_id: turn_id.to_string(),
            }),
            TurnLifecycle::TurnFinished { turn_id, .. } => Some(Seen::TurnFinished {
                turn_id: turn_id.to_string(),
            }),
            TurnLifecycle::ToolStarted {
                turn_id,
                tool_call_id,
                tool_name,
                arguments,
                ..
            } => Some(Seen::ToolStarted {
                turn_id: turn_id.to_string(),
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                arguments: arguments.to_string(),
            }),
            TurnLifecycle::ToolFinished {
                turn_id,
                tool_call_id,
                success,
                summary,
                ..
            } => Some(Seen::ToolFinished {
                turn_id: turn_id.to_string(),
                tool_call_id: tool_call_id.clone(),
                success: *success,
                summary: summary.clone(),
            }),
            _ => None,
        };
        if let Some(event) = recorded {
            actor.model.seen.push(event);
        }
        Reply::ready()
    });

    builder.mutate_on::<LLMStreamToolResult>(|actor, envelope| {
        let message = envelope.message();
        actor.model.seen.push(Seen::ToolResult {
            turn_id: message.turn_id.to_string(),
            tool_call_id: message.tool_call_id.clone(),
            success: message.success,
        });
        Reply::ready()
    });

    builder.act_on::<GetSeen>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let events = actor.model.seen.clone();
        Reply::pending(async move {
            reply.send(SeenEvents { events }).await;
        })
    });

    // On the builder, before start: a subscription registered after `start()`
    // is silently ignored.
    builder.handle().subscribe::<TurnLifecycle>().await;
    builder.handle().subscribe::<LLMStreamToolResult>().await;

    builder.start().await
}

/// Drains the broker, then reads everything the recorder saw.
async fn recorded(ai: &ActonAI, recorder: &ActorHandle) -> Vec<Seen> {
    ai.runtime()
        .broker()
        .ask(FlushBroadcasts)
        .await
        .expect("the broker must answer a flush");
    recorder
        .ask(GetSeen)
        .await
        .expect("the recorder must answer GetSeen")
        .events
}

/// Pairs every `ToolStarted` with the events that close it.
///
/// Returns, for each `tool_call_id` in the order it was started, how many
/// starts, finishes, and results carried it. The whole invariant is a
/// statement about these counts.
fn bracket_counts(events: &[Seen]) -> Vec<(String, usize, usize, usize)> {
    let mut order: Vec<String> = Vec::new();
    for event in events {
        if let Seen::ToolStarted { tool_call_id, .. } = event {
            if !order.contains(tool_call_id) {
                order.push(tool_call_id.clone());
            }
        }
    }
    order
        .into_iter()
        .map(|id| {
            let starts = events
                .iter()
                .filter(
                    |e| matches!(e, Seen::ToolStarted { tool_call_id, .. } if *tool_call_id == id),
                )
                .count();
            let finishes = events
                .iter()
                .filter(
                    |e| matches!(e, Seen::ToolFinished { tool_call_id, .. } if *tool_call_id == id),
                )
                .count();
            let results = events
                .iter()
                .filter(
                    |e| matches!(e, Seen::ToolResult { tool_call_id, .. } if *tool_call_id == id),
                )
                .count();
            (id, starts, finishes, results)
        })
        .collect()
}

/// The index of the first event matching `predicate`.
fn position_of(events: &[Seen], predicate: impl Fn(&Seen) -> bool) -> Option<usize> {
    events.iter().position(predicate)
}

// =============================================================================
// 1. A caller-supplied turn id is honoured everywhere
// =============================================================================

#[tokio::test]
async fn a_caller_supplied_turn_id_reaches_the_response_and_every_event() {
    // The gap this closes: an embedder that announced a turn to its own
    // client before calling us needs OUR events to carry ITS id, or it has to
    // maintain a mapping table between two identity schemes.
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})),
        Round::text("done"),
    ])
    .await;
    let mut ai = runtime_pointed_at(&server, "turn-supplied").await;
    let recorder = spawn_recorder(ai.runtime_mut()).await;

    let turn_id = TurnId::new();
    let response = ai
        .prompt("go")
        .turn_id(turn_id.clone())
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must succeed");

    assert_eq!(
        response.turn_id, turn_id,
        "collect() must return the id the caller supplied, not a fresh one"
    );

    let events = recorded(&ai, &recorder).await;
    assert!(
        !events.is_empty(),
        "a turn running a tool must publish events"
    );
    let expected = turn_id.to_string();
    for event in &events {
        assert_eq!(
            event.turn_id(),
            expected,
            "every event of this turn must carry the caller's id: {event:?}"
        );
    }
}

// =============================================================================
// 2. A minted turn id is still reported back
// =============================================================================

#[tokio::test]
async fn a_turn_id_is_minted_and_still_returned_when_the_caller_supplies_none() {
    // Without this, a caller who did not supply an id has no way to correlate
    // the response with the events it just watched go by — which was the
    // original defect: `collect()` simply never told anyone.
    let server = MockServer::start(vec![Round::text("done")]).await;
    let mut ai = runtime_pointed_at(&server, "turn-minted").await;
    let recorder = spawn_recorder(ai.runtime_mut()).await;

    let response = ai.prompt("go").collect().await.expect("the turn succeeds");

    let events = recorded(&ai, &recorder).await;
    let started = events
        .iter()
        .find(|e| matches!(e, Seen::TurnStarted { .. }))
        .expect("a turn that ran must publish TurnStarted");

    assert_eq!(
        response.turn_id.to_string(),
        started.turn_id(),
        "the minted id on the response must be the one the events carried"
    );
}

// =============================================================================
// 3. A refused call is bracketed exactly like one that ran
// =============================================================================

#[tokio::test]
async fn a_denied_call_is_bracketed_exactly_like_one_that_ran() {
    // The headline gap. Before this, a denial published a result with no
    // preceding start, so a consumer had to invent one.
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "bash", json!({"command": "rm -rf /"})),
        Round::text("understood"),
    ])
    .await;
    let mut ai =
        runtime_with_policy(&server, "turn-denied", ToolPolicy::new().deny(["bash"])).await;
    let recorder = spawn_recorder(ai.runtime_mut()).await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();
    ai.prompt("clean up")
        .with_tool(tool_definition("bash"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .collect()
        .await
        .expect("a denial must not fail the turn");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        0,
        "the tool must not have run; otherwise this tests the wrong path"
    );

    let events = recorded(&ai, &recorder).await;
    assert_eq!(
        bracket_counts(&events),
        vec![("call_1".to_string(), 1, 1, 1)],
        "a refused call must be started once, finished once, and reported once"
    );

    let start = position_of(&events, |e| matches!(e, Seen::ToolStarted { .. }))
        .expect("a refused call must still be announced");
    let finish = position_of(&events, |e| matches!(e, Seen::ToolFinished { .. }))
        .expect("a refused call must still be closed");
    assert!(
        start < finish,
        "the start must precede the finish even when the gate refused: {events:?}"
    );

    let Some(Seen::ToolFinished {
        success, summary, ..
    }) = events.get(finish)
    else {
        unreachable!("index came from a ToolFinished match")
    };
    assert!(!success, "a refused call did not succeed");
    assert!(
        !summary.is_empty(),
        "a refusal must say why, or the client has nothing to render"
    );
}

// =============================================================================
// 4. The invariant itself, over a turn that mixes both paths
// =============================================================================

#[tokio::test]
async fn every_tool_result_was_preceded_by_exactly_one_start() {
    // One round proposing two calls, one allowed and one denied, so the
    // allowed and refused paths are exercised inside a single turn and any
    // divergence between them shows up as an asymmetry here.
    let server = MockServer::start(vec![
        Round::tool_call("call_ok", "echo", json!({"value": "fine"})).with_tool_call(
            "call_no",
            "bash",
            json!({"command": "rm -rf /"}),
        ),
        Round::text("done"),
    ])
    .await;
    let mut ai = runtime_with_policy(&server, "turn-mixed", ToolPolicy::new().deny(["bash"])).await;
    let recorder = spawn_recorder(ai.runtime_mut()).await;

    ai.prompt("do both")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .with_tool(tool_definition("bash"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("a mixed turn must succeed");

    let events = recorded(&ai, &recorder).await;

    let mut counts = bracket_counts(&events);
    counts.sort();
    assert_eq!(
        counts,
        vec![
            ("call_no".to_string(), 1, 1, 1),
            ("call_ok".to_string(), 1, 1, 1),
        ],
        "both the allowed and the refused call must be bracketed identically"
    );

    // Ordering, not just counts: a start that arrives after its own result is
    // useless to a consumer that must announce before it reports.
    for (id, ..) in &counts {
        let start = position_of(
            &events,
            |e| matches!(e, Seen::ToolStarted { tool_call_id, .. } if tool_call_id == id),
        )
        .expect("every call is started");
        let finish = position_of(
            &events,
            |e| matches!(e, Seen::ToolFinished { tool_call_id, .. } if tool_call_id == id),
        )
        .expect("every call is finished");
        let result = position_of(
            &events,
            |e| matches!(e, Seen::ToolResult { tool_call_id, .. } if tool_call_id == id),
        )
        .expect("every call reports a result");

        assert!(start < finish, "{id}: start must precede finish");
        assert!(start < result, "{id}: start must precede result");
    }
}

// =============================================================================
// 5. The start carries what the model asked for
// =============================================================================

#[tokio::test]
async fn tool_started_carries_the_arguments_the_model_proposed() {
    // A client renders the arguments next to the tool name. Before this it
    // had to dig them out of the follow-up request's message history.
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "the proposed one"})),
        Round::text("done"),
    ])
    .await;
    let mut ai = runtime_pointed_at(&server, "turn-args").await;
    let recorder = spawn_recorder(ai.runtime_mut()).await;

    ai.prompt("go")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must succeed");

    let events = recorded(&ai, &recorder).await;
    let started = events
        .iter()
        .find_map(|e| match e {
            Seen::ToolStarted {
                tool_name,
                arguments,
                ..
            } => Some((tool_name.clone(), arguments.clone())),
            _ => None,
        })
        .expect("a tool call must be announced");

    assert_eq!(started.0, "echo");
    let parsed: Value =
        serde_json::from_str(&started.1).expect("the arguments must be carried as real JSON");
    assert_eq!(
        parsed,
        json!({"value": "the proposed one"}),
        "the announced arguments must be verbatim what the model proposed"
    );

    // "What the model proposed" is only meaningful against what the model was
    // offered, so pin that too: the announced call must name a tool that
    // actually went out on the wire, described by a self-contained schema.
    let first = server.requests().remove(0);
    let offered = tool_named(&first, "echo").expect("the tool must be advertised to the model");
    assert!(
        !contains_ref(&offered["function"]["parameters"]),
        "provider support for $ref inside a tool schema is inconsistent, so it must be inlined"
    );
}

// =============================================================================
// 5b. The announced arguments cross the same redaction boundary as the trail
// =============================================================================

#[tokio::test]
async fn tool_started_arguments_are_redacted_when_a_trail_is_configured() {
    // The audit file redacts at the boundary so a secret never reaches the
    // audit actor's mailbox. `ToolStarted` fans out to *every* lifecycle
    // subscriber's mailbox — the introspection actor, an embedder's
    // forwarder rendering arguments in a client UI — so the same boundary
    // must apply, or the redaction config keeps secrets out of the file
    // while broadcasting them everywhere else.
    let dir = tempfile::tempdir().expect("a temp dir");
    let server = MockServer::start(vec![
        Round::tool_call(
            "call_1",
            "echo",
            json!({"api_key": "sk-live-do-not-broadcast-me", "value": "safe"}),
        ),
        Round::text("done"),
    ])
    .await;

    let mut ai = ActonAI::builder()
        .app_name("turn-args-redacted")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .audit_to(dir.path().join("audit.jsonl"))
        .launch()
        .await
        .expect("launching the runtime must succeed");
    let recorder = spawn_recorder(ai.runtime_mut()).await;

    ai.prompt("go")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must succeed");

    let events = recorded(&ai, &recorder).await;
    let arguments = events
        .iter()
        .find_map(|e| match e {
            Seen::ToolStarted { arguments, .. } => Some(arguments.clone()),
            _ => None,
        })
        .expect("a tool call must be announced");

    assert!(
        !arguments.contains("sk-live-do-not-broadcast-me"),
        "a secret must never ride a lifecycle broadcast: {arguments}"
    );
    let parsed: Value =
        serde_json::from_str(&arguments).expect("the arguments must be carried as real JSON");
    assert_eq!(parsed["api_key"], json!("[redacted]"));
    assert_eq!(
        parsed["value"],
        json!("safe"),
        "redaction must be surgical, not wholesale"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 5c. A turn dropped on a foreign thread still balances its start
// =============================================================================

#[tokio::test]
async fn a_turn_dropped_on_a_non_tokio_thread_still_publishes_its_finish() {
    // The `Send + Sync` `collect()` future exists so an embedder can store a
    // turn in its own session table; the thread that later drops that entry
    // — a UI thread, a C-FFI callback, a watchdog `std::thread` — has no
    // Tokio context. The drop guard must still publish the balancing
    // `TurnFinished`, or the introspection actor counts the turn in-flight
    // forever and `acton-ai drain --wait` wedges.
    let server = MockServer::start(vec![Round::text("never read")]).await;
    let mut ai = runtime_pointed_at(&server, "turn-drop-off-runtime").await;
    let recorder = spawn_recorder(ai.runtime_mut()).await;

    // Drive the future by hand so it can be abandoned mid-turn: poll until
    // the recorder has seen `TurnStarted`, then never poll it again. The
    // future cannot advance past our last poll, so it is still in flight
    // when it is dropped. Bounded loop, no sleeps: `recorded` is the
    // flush-barrier, `yield_now` lets the actors run.
    let mut fut = Box::pin(ai.prompt("go").collect());
    let waker = futures::task::noop_waker();
    let mut started = false;
    for _ in 0..10_000 {
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(
            fut.as_mut().poll(&mut cx).is_pending(),
            "the turn must still be in flight when it is abandoned"
        );
        if recorded(&ai, &recorder)
            .await
            .iter()
            .any(|e| matches!(e, Seen::TurnStarted { .. }))
        {
            started = true;
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(started, "the turn must announce itself before the drop");

    std::thread::spawn(move || {
        assert!(
            tokio::runtime::Handle::try_current().is_err(),
            "this thread must have no ambient runtime, or the test proves nothing"
        );
        drop(fut);
    })
    .join()
    .expect("the dropping thread must not panic");

    // The guard spawned the balancing broadcast onto the runtime handle it
    // captured at construction; yield until it lands.
    let mut finished = false;
    for _ in 0..10_000 {
        if recorded(&ai, &recorder)
            .await
            .iter()
            .any(|e| matches!(e, Seen::TurnFinished { .. }))
        {
            finished = true;
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        finished,
        "a turn dropped on a non-Tokio thread must still balance its start"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 6. The finish carries the outcome
// =============================================================================

#[tokio::test]
async fn a_finished_tool_reports_its_outcome() {
    // Previously `ToolFinished` carried only ids, so a consumer could tell
    // that a call ended but not whether it worked.
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "result text"})),
        Round::text("done"),
    ])
    .await;
    let mut ai = runtime_pointed_at(&server, "turn-outcome").await;
    let recorder = spawn_recorder(ai.runtime_mut()).await;

    ai.prompt("go")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must succeed");

    let events = recorded(&ai, &recorder).await;
    let (success, summary) = events
        .iter()
        .find_map(|e| match e {
            Seen::ToolFinished {
                success, summary, ..
            } => Some((*success, summary.clone())),
            _ => None,
        })
        .expect("a tool call must be closed");

    assert!(
        success,
        "a tool that returned Ok must be reported as success"
    );
    assert!(
        !summary.is_empty(),
        "a successful call must still preview its result"
    );

    // The pair must agree; an observer watching only one of them has to reach
    // the same conclusion as one watching the other.
    let result_success = events
        .iter()
        .find_map(|e| match e {
            Seen::ToolResult { success, .. } => Some(*success),
            _ => None,
        })
        .expect("a tool call must report a result");
    assert_eq!(
        success, result_success,
        "ToolFinished and LLMStreamToolResult must agree about the same call"
    );
}

// =============================================================================
// 7. The streaming callbacks know who they belong to
// =============================================================================

#[tokio::test]
async fn on_start_receives_the_turn_and_round_identity() {
    // Gap 8: `on_start` used to take no arguments at all, so a caller
    // multiplexing concurrent prompts could not tell which one had begun.
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})),
        Round::text("done"),
    ])
    .await;
    let ai = runtime_pointed_at(&server, "turn-callbacks").await;

    let turn_id = TurnId::new();
    let starts: Arc<std::sync::Mutex<Vec<(String, String)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let ends: Arc<std::sync::Mutex<Vec<(String, String)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));

    let seen_starts = starts.clone();
    let seen_ends = ends.clone();

    let response = ai
        .prompt("go")
        .turn_id(turn_id.clone())
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .on_start(move |ctx| {
            if let Ok(mut seen) = seen_starts.lock() {
                seen.push((ctx.turn_id.to_string(), ctx.correlation_id.to_string()));
            }
        })
        .on_end(move |ctx, _reason| {
            if let Ok(mut seen) = seen_ends.lock() {
                seen.push((ctx.turn_id.to_string(), ctx.correlation_id.to_string()));
            }
        })
        .collect()
        .await
        .expect("the turn must succeed");

    let starts = starts.lock().expect("the callback must not poison").clone();
    let ends = ends.lock().expect("the callback must not poison").clone();
    let expected_turn = turn_id.to_string();

    // Two rounds: the tool-call round and the round after it. This is the
    // distinction the context exists to express — one turn, several rounds.
    assert_eq!(
        server.request_count(),
        2,
        "the turn must really have driven two rounds, or the rest proves nothing"
    );
    assert_eq!(starts.len(), 2, "a two-round turn starts two streams");
    assert_eq!(ends.len(), 2, "a two-round turn ends two streams");
    assert_eq!(response.turn_id, turn_id);

    for (turn, _) in starts.iter().chain(ends.iter()) {
        assert_eq!(
            *turn, expected_turn,
            "every round of the turn reports the same turn id"
        );
    }

    let rounds: Vec<&String> = starts.iter().map(|(_, round)| round).collect();
    assert_ne!(
        rounds[0], rounds[1],
        "each round must carry its own correlation id, or the context cannot distinguish them"
    );

    let end_rounds: Vec<&String> = ends.iter().map(|(_, round)| round).collect();
    assert_eq!(
        rounds, end_rounds,
        "a round's end must report the same identity its start did"
    );
}

// =============================================================================
// 8. The start is published before the gate deliberates
// =============================================================================

#[tokio::test]
async fn the_start_is_published_before_the_gate_deliberates() {
    // The bracket tests above prove a refused call is *eventually* announced,
    // but they still pass if the announcement is moved to after the gate —
    // the refusal path closes what it opens either way. This one pins the
    // ordering the design actually calls for, and it is the ordering a human
    // approval hook depends on: while a person is deciding, the call is
    // already in the client's status line, because the model proposed it and
    // something is deliberating about it.
    //
    // Determinism: the hook flushes the broker before it looks. The flush
    // cannot answer until every broadcast already handed to the broker is in
    // the recorder's inbox, so "not yet visible after a flush" means "not yet
    // published" rather than "published but still in flight".
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "bash", json!({"command": "rm -rf /"})),
        Round::text("understood"),
    ])
    .await;

    // Populated after launch. The hook cannot run before `collect()`, which
    // is long after this is set, so the `get()` inside it always succeeds —
    // and if it ever did not, `saw_start` would stay false and the assertion
    // would fail loudly rather than skip.
    let wiring = Arc::new(std::sync::OnceLock::new());
    let saw_start = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let hook_wiring = wiring.clone();
    let hook_saw_start = saw_start.clone();

    let policy = ToolPolicy::new().on_approval(move |invocation: ToolInvocation| {
        let wiring = hook_wiring.clone();
        let saw_start = hook_saw_start.clone();
        async move {
            if let Some((broker, recorder)) = wiring.get() {
                let _: Result<_, _> = ActorHandle::ask(broker, FlushBroadcasts).await;
                if let Ok(seen) = ActorHandle::ask(recorder, GetSeen).await {
                    let announced = seen.events.iter().any(|event| {
                        matches!(event, Seen::ToolStarted { tool_call_id, .. }
                            if *tool_call_id == invocation.tool_call_id)
                    });
                    saw_start.store(announced, Ordering::SeqCst);
                }
            }
            ApprovalDecision::deny("a human said no")
        }
    });

    let mut ai = runtime_with_policy(&server, "turn-gate-order", policy).await;
    let recorder = spawn_recorder(ai.runtime_mut()).await;
    let _ = wiring.set((ai.runtime().broker().clone(), recorder.clone()));

    ai.prompt("clean up")
        .with_tool(tool_definition("bash"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("a denial must not fail the turn");

    assert!(
        saw_start.load(Ordering::SeqCst),
        "the gate must deliberate on a call that has already been announced"
    );

    // And the bracket still closes exactly once, so the ordering above was
    // not bought by publishing a second start.
    let events = recorded(&ai, &recorder).await;
    assert_eq!(
        bracket_counts(&events),
        vec![("call_1".to_string(), 1, 1, 1)],
        "announcing before the gate must not duplicate the announcement"
    );
}

// =============================================================================
// 9. The audit trail names the same call the events did
// =============================================================================

#[tokio::test]
async fn the_audit_entry_carries_the_call_id_the_events_carried() {
    // The bracket tests prove a live watcher sees a coherent story; the audit
    // tests elsewhere prove the trail records *a* call id. What neither pins
    // is the join: an investigator reading the trail after the fact must be
    // able to line its entries up with what a session watcher saw live, and
    // that join is `tool_call_id` plus `turn_id`. So this test runs both
    // observers at once — the recorder on the broker and the sealed trail on
    // disk — and asserts they name the same calls in the same turn, on an
    // executed call and a refused one alike.
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let server = MockServer::start(vec![
        Round::tool_call("call_deny", "bash", json!({"value": "rm -rf /"})).with_tool_call(
            "call_ok",
            "echo",
            json!({"value": "hi"}),
        ),
        Round::text("understood"),
    ])
    .await;
    let mut ai = ActonAI::builder()
        .app_name("turn-events-audit")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .tool_policy(ToolPolicy::new().deny(["bash"]))
        .audit_to(&path)
        .launch()
        .await
        .expect("launching the runtime must succeed");
    let recorder = spawn_recorder(ai.runtime_mut()).await;

    let turn_id = TurnId::new();
    ai.prompt("clean up")
        .turn_id(turn_id.clone())
        .with_tool(tool_definition("bash"), |args| async move { Ok(args) })
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("a denial must not fail the turn");

    // The audit-side barrier: the head cannot answer before both entries are
    // sealed and written.
    let head = ai.audit_head().await.expect("the trail must report a head");
    assert_eq!(head.entries, 2, "one denied and one executed call");

    let events = recorded(&ai, &recorder).await;
    let contents = std::fs::read_to_string(&path).expect("the trail must exist");
    let entries = acton_ai::audit::parse_entries(&contents).expect("the trail must parse");

    // Each entry names the call the live events named for that tool, in the
    // turn the caller supplied.
    for entry in &entries {
        let event_call_id = events
            .iter()
            .find_map(|event| match event {
                Seen::ToolStarted {
                    tool_call_id,
                    tool_name,
                    ..
                } if *tool_name == entry.tool_name => Some(tool_call_id.clone()),
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
