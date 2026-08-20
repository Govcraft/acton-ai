//! End-to-end tests for live introspection over the control socket.
//!
//! These drive the whole path a real `acton-ai status` takes: a runtime
//! launched with `introspection_at`, a real `IpcClient` connected over a real
//! Unix socket, the shared [`wire`](acton_ai::introspection::wire) types
//! encoded and decoded by acton-reactive's IPC layer, and an
//! `IntrospectionActor` assembling its answer from live providers and a live
//! cost accountant. Nothing here is mocked except the LLM itself.
//!
//! # Determinism
//!
//! Nothing sleeps. Two barriers do all the waiting:
//!
//! - A **gated tool**. The prompt loop executes tools between rounds, so a
//!   tool that signals and then blocks parks a turn mid-flight at a point the
//!   test chooses, and releases it when the test says so. That is how the
//!   in-flight assertions get something to observe.
//! - **`FlushBroadcasts`**. Turn tracking is passive: the prompt loop
//!   broadcasts [`TurnLifecycle`], and `broadcast().await` only proves the
//!   message reached the *broker*, not the introspection actor. The broker's
//!   reply to `FlushBroadcasts` cannot arrive until every earlier broadcast is
//!   sitting in each subscriber's inbox, and because mailboxes are FIFO the
//!   `GetStatus` issued afterwards is necessarily processed behind it.
//!
//! Without the second barrier these tests would pass on a fast machine and
//! flake on a loaded one, which is the worst of both worlds.

#![cfg(feature = "ipc")]

mod mock_llm;

use acton_ai::introspection::wire::{
    AdmissionAck, Drain, GetStatus, Pause, Resume, StatusReport, INTROSPECTION_ACTOR_NAME,
    SCHEMA_VERSION,
};
use acton_ai::prelude::*;
use acton_reactive::ipc::IpcClient;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Notify;

// =============================================================================
// Fixtures
// =============================================================================

/// Launches a runtime listening on `socket`, with `server` as its only
/// provider.
async fn runtime_listening_at(
    server: &MockServer,
    app_name: &str,
    socket: &std::path::Path,
) -> ActonAI {
    ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .introspection_at(socket)
        .launch()
        .await
        .expect("launching a runtime with introspection must succeed")
}

/// Connects a client to a listening runtime.
async fn connect(socket: &std::path::Path) -> IpcClient {
    IpcClient::connect(socket)
        .await
        .expect("the socket must accept a client while the runtime is up")
}

/// Asks for a status report over the wire.
async fn ask_status(client: &IpcClient) -> StatusReport {
    client
        .actor(INTROSPECTION_ACTOR_NAME)
        .ask(GetStatus)
        .await
        .expect("a running process must answer GetStatus")
}

/// Sends one of the three admission commands and returns its acknowledgement.
macro_rules! ask_admission {
    ($client:expr, $request:expr) => {
        $client
            .actor(INTROSPECTION_ACTOR_NAME)
            .ask($request)
            .await
            .expect("a running process must answer an admission command")
    };
}

/// Waits until every broadcast issued so far is in every subscriber's inbox.
///
/// See the module docs: this is what makes the turn-count assertions
/// deterministic without a sleep.
async fn flush_broadcasts(ai: &ActonAI) {
    ai.runtime()
        .broker()
        .ask(FlushBroadcasts)
        .await
        .expect("the broker must answer a flush");
}

/// A tool that parks the turn executing it until the test lets it go.
struct GatedTool {
    /// Notified once the tool body has begun.
    started: Arc<Notify>,
    /// Notified by the test to let the tool return.
    release: Arc<Notify>,
}

impl GatedTool {
    fn new() -> Self {
        Self {
            started: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        }
    }
}

// =============================================================================
// 1. Status
// =============================================================================

#[tokio::test]
async fn a_running_process_describes_itself_over_its_socket() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("status.sock");
    let server = MockServer::start(vec![Round::text("hello").with_usage(11, 7)]).await;
    let ai = runtime_listening_at(&server, "status-agent", &socket).await;

    ai.prompt("say hello").collect().await.expect("the prompt");
    flush_broadcasts(&ai).await;

    let report = ask_status(&connect(&socket).await).await;

    // Identity, so an operator holding a status report knows which process on
    // which host produced it, and which build.
    assert_eq!(report.schema_version, SCHEMA_VERSION);
    assert_eq!(report.app_name, "status-agent");
    assert_eq!(report.pid, std::process::id());
    assert_eq!(report.crate_version, env!("CARGO_PKG_VERSION"));

    // Turn tracking, the whole point of the passive subscription.
    assert_eq!(report.admission, "running");
    assert_eq!(report.turns_started, 1);
    assert_eq!(report.turns_refused, 0);
    assert_eq!(
        report.active_turns, 0,
        "a finished turn must not still be counted as active"
    );
    assert_eq!(report.in_flight_tool_calls, 0);

    // Assembled from the live provider, not from the config: this is what
    // makes the report worth asking for during an incident.
    assert_eq!(report.providers.len(), 1);
    assert_eq!(report.providers[0].model, "mock-model");
    assert!(!report.providers[0].circuit_open);

    let usage = report.usage.expect("usage tracking is on by default");
    assert_eq!(usage.input_tokens, 11);
    assert_eq!(usage.output_tokens, 7);
    assert_eq!(usage.requests, 1);

    ai.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn successive_turns_accumulate_rather_than_replace() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("counting.sock");
    let server = MockServer::start(vec![
        Round::text("one").with_usage(3, 1),
        Round::text("two").with_usage(4, 2),
    ])
    .await;
    let ai = runtime_listening_at(&server, "counting-agent", &socket).await;

    ai.prompt("first").collect().await.expect("first prompt");
    ai.prompt("second").collect().await.expect("second prompt");
    flush_broadcasts(&ai).await;

    let report = ask_status(&connect(&socket).await).await;

    // A counter that reset per connection, or per turn, would make "how busy
    // has this process been" unanswerable.
    assert_eq!(report.turns_started, 2);
    assert_eq!(report.active_turns, 0);
    let usage = report.usage.expect("usage");
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.input_tokens, 7);

    ai.shutdown().await.expect("shutdown");
}

// =============================================================================
// 2. Pause and resume
// =============================================================================

#[tokio::test]
async fn pausing_over_the_socket_refuses_the_next_turn_and_resuming_takes_it_back() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("pause.sock");
    let server = MockServer::start(vec![Round::text("after the pause")]).await;
    let ai = runtime_listening_at(&server, "pause-agent", &socket).await;
    let client = connect(&socket).await;

    let ack: AdmissionAck = ask_admission!(client, Pause);
    assert_eq!(ack.admission, "paused");
    assert_eq!(
        ack.in_flight_turns, 0,
        "nothing was running, so the pause is already fully in effect"
    );

    // The refusal has to be a distinguishable error, not a generic failure:
    // a caller retrying a paused runtime forever is the failure mode this
    // whole surface exists to avoid.
    let refused = ai
        .prompt("this must not reach the provider")
        .collect()
        .await
        .expect_err("a paused runtime refuses new turns");
    assert!(refused.is_turns_not_admitted(), "{refused}");
    assert!(
        refused.to_string().contains("acton-ai resume"),
        "the refusal must name the way out: {refused}"
    );

    // Refused before anything went out: the provider never heard about it.
    assert_eq!(
        server.request_count(),
        0,
        "a refused turn must cost nothing"
    );

    flush_broadcasts(&ai).await;
    let report = ask_status(&client).await;
    assert_eq!(report.admission, "paused");
    assert_eq!(report.turns_refused, 1);
    assert_eq!(
        report.turns_started, 0,
        "a refused turn never started, so it must not inflate the started count"
    );

    let ack: AdmissionAck = ask_admission!(client, Resume);
    assert_eq!(ack.admission, "running");

    let response = ai
        .prompt("now it should work")
        .collect()
        .await
        .expect("a resumed runtime takes turns again");
    assert_eq!(response.text, "after the pause");
    assert_eq!(server.request_count(), 1);

    ai.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn resume_recovers_from_draining_as_well_as_from_paused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("undrain.sock");
    let server = MockServer::start(vec![Round::text("back in service")]).await;
    let ai = runtime_listening_at(&server, "undrain-agent", &socket).await;
    let client = connect(&socket).await;

    let ack: AdmissionAck = ask_admission!(client, Drain);
    assert_eq!(ack.admission, "draining");

    // Draining is not a one-way door. An operator who drained the wrong
    // process needs to undo it without a restart, which is the only reason
    // `resume` accepts a draining runtime at all.
    let ack: AdmissionAck = ask_admission!(client, Resume);
    assert_eq!(ack.admission, "running");

    ai.prompt("hello again")
        .collect()
        .await
        .expect("an un-drained runtime takes turns");

    ai.shutdown().await.expect("shutdown");
}

// =============================================================================
// 3. Drain with work in flight
// =============================================================================

#[tokio::test]
async fn a_drain_reports_the_turn_still_running_and_completes_once_it_finishes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("drain.sock");
    let server = MockServer::start(vec![
        Round::tool_call("call_gate", "gate", json!({})),
        Round::text("finished after the drain"),
    ])
    .await;
    let ai = runtime_listening_at(&server, "drain-agent", &socket).await;
    let client = connect(&socket).await;

    let gate = GatedTool::new();
    let started = Arc::clone(&gate.started);
    let release = Arc::clone(&gate.release);

    let turn = tokio::spawn({
        let ai = ai.clone();
        async move {
            ai.prompt("run the gated tool")
                .tool(
                    "gate",
                    "Blocks until the test releases it",
                    json!({"type": "object", "properties": {}}),
                    move |_args| {
                        let started = Arc::clone(&gate.started);
                        let release = Arc::clone(&gate.release);
                        async move {
                            started.notify_one();
                            release.notified().await;
                            Ok(json!({"released": true}))
                        }
                    },
                )
                .collect()
                .await
        }
    });

    // Barrier one: the tool body is running, so a turn is genuinely mid-flight.
    started.notified().await;

    // The tool the status report is about is the tool the provider was
    // actually offered — otherwise the in-flight count below could be counting
    // something else entirely.
    let request = server
        .requests()
        .into_iter()
        .next()
        .expect("the first round went out");
    let gate = tool_named(&request, "gate").expect("the gate tool was advertised");
    assert!(
        !contains_ref(&gate["function"]["parameters"]),
        "provider support for $ref inside a tool schema is inconsistent: {gate}"
    );

    // Barrier two: TurnStarted and ToolStarted have reached the actor's inbox.
    flush_broadcasts(&ai).await;

    let report = ask_status(&client).await;
    assert_eq!(report.active_turns, 1);
    assert_eq!(
        report.in_flight_tool_calls, 1,
        "the tool the turn is blocked in must be visible, since that is what \
         an operator is trying to identify"
    );

    let ack: AdmissionAck = ask_admission!(client, Drain);
    assert_eq!(ack.admission, "draining");
    assert_eq!(ack.in_flight_turns, 1);
    assert!(
        !ack.is_drained(),
        "reporting a completed drain here would let a supervisor kill live work"
    );

    // Draining never interrupts: the turn that was already running finishes,
    // tool and all.
    release.notify_one();
    let response = turn
        .await
        .expect("the turn task must not panic")
        .expect("a turn already running is unaffected by a drain");
    assert_eq!(response.text, "finished after the drain");

    flush_broadcasts(&ai).await;
    let ack: AdmissionAck = ask_admission!(client, Drain);
    assert!(
        ack.is_drained(),
        "with the last turn finished the drain is complete: {ack:?}"
    );
    assert_eq!(ack.admission, "draining");

    // And no new turn is taken, which is the other half of what drain means.
    let refused = ai
        .prompt("too late")
        .collect()
        .await
        .expect_err("a drained runtime takes nothing new");
    assert!(refused.is_turns_not_admitted(), "{refused}");

    ai.shutdown().await.expect("shutdown");
}

// =============================================================================
// 4. The socket itself
// =============================================================================

#[tokio::test]
async fn the_live_socket_is_owner_only_and_is_gone_after_shutdown() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("perms.sock");
    let server = MockServer::start(vec![Round::text("ok")]).await;
    let ai = runtime_listening_at(&server, "perms-agent", &socket).await;

    // Asserted on a socket that has demonstrably served a request, so this is
    // the mode of a *working* socket rather than of a file that happens to
    // exist.
    let _ = ask_status(&connect(&socket).await).await;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&socket)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        // `pause` and `drain` are levers over this process. The permission
        // bits are the whole of the access control, because acton-reactive's
        // IPC layer exposes no peer-credential hook to a message handler.
        assert_eq!(mode, 0o600, "{mode:#o}");
    }

    ai.shutdown().await.expect("shutdown");
    assert!(
        !socket.exists(),
        "a socket left behind makes the next launch treat a dead address as \
         possibly-live"
    );
}

#[tokio::test]
async fn a_second_runtime_cannot_take_over_a_socket_that_is_answering() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("contested.sock");
    let server = MockServer::start(vec![Round::text("ok")]).await;
    let first = runtime_listening_at(&server, "first-agent", &socket).await;

    let report = ask_status(&connect(&socket).await).await;
    assert_eq!(report.app_name, "first-agent");

    let second = ActonAI::builder()
        .app_name("second-agent")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .introspection_at(&socket)
        .launch()
        .await;

    let error = second
        .expect_err("a live socket must not be stolen")
        .to_string();
    assert!(error.contains(&socket.display().to_string()), "{error}");
    // Not just that it failed: the only part of this an operator can act on is
    // being told how to find out who is already holding the socket.
    assert!(error.contains("acton-ai status"), "{error}");

    // The incumbent is untouched: a failed second launch must not have taken
    // the first process's control socket down with it.
    let report = ask_status(&connect(&socket).await).await;
    assert_eq!(report.app_name, "first-agent");

    first.shutdown().await.expect("shutdown");
}

// =============================================================================
// 5. Configuration
// =============================================================================

#[tokio::test]
async fn a_socket_configured_in_toml_is_the_one_that_gets_bound() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("from-toml.sock");
    let server = MockServer::start(vec![Round::text("ok")]).await;

    let toml = format!(
        "{}\n[introspection]\nsocket_path = \"{}\"\n",
        mock_llm::provider_toml("local", &server, "mock-model"),
        socket.display(),
    );
    let config = acton_ai::config::from_str(&toml).expect("the config must parse");

    let ai = ActonAI::builder()
        .app_name("toml-agent")
        .apply_config(config)
        .expect("the config must apply")
        .launch()
        .await
        .expect("launch");

    assert_eq!(ai.introspection_socket(), Some(socket.as_path()));
    let report = ask_status(&connect(&socket).await).await;
    assert_eq!(report.app_name, "toml-agent");

    ai.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn introspection_disabled_in_toml_binds_nothing() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("never-bound.sock");
    let server = MockServer::start(vec![Round::text("ok")]).await;

    let toml = format!(
        "{}\n[introspection]\nenabled = false\nsocket_path = \"{}\"\n",
        mock_llm::provider_toml("local", &server, "mock-model"),
        socket.display(),
    );
    let config = acton_ai::config::from_str(&toml).expect("the config must parse");

    let ai = ActonAI::builder()
        .app_name("disabled-agent")
        .apply_config(config)
        .expect("the config must apply")
        .launch()
        .await
        .expect("a disabled section must not fail the launch");

    // `enabled = false` with a path still present is the shape a config ends
    // up in when someone switches introspection off temporarily. Binding it
    // anyway would make the switch a lie.
    assert_eq!(ai.introspection_socket(), None);
    assert!(!socket.exists());

    // Admission control is in-process state and keeps working regardless.
    assert!(ai.admission_state().admits());
    ai.pause();
    assert!(!ai.admission_state().admits());

    ai.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn no_introspection_section_means_no_socket_at_all() {
    let server = MockServer::start(vec![Round::text("ok")]).await;
    // Built exactly the way every other suite builds a runtime, so this is the
    // ordinary path and not a specially-configured one.
    let ai = runtime_pointed_at(&server, "quiet-agent").await;

    // Compiling the feature in must not open a control socket. A library whose
    // default behaviour is to listen would be a surprising one to depend on.
    assert_eq!(ai.introspection_socket(), None);

    ai.shutdown().await.expect("shutdown");
}

// =============================================================================
// 6. Interrupted turns
// =============================================================================

/// A test-owned subscriber recording every turn start and finish it sees.
///
/// Both the observation and the barrier: an `ask` answered by this actor
/// proves every lifecycle broadcast ahead of it in the same FIFO inbox has
/// been folded in.
#[acton_actor]
struct LifecycleSpy {
    started: Vec<String>,
    finished: Vec<(String, String)>,
}

#[acton_message]
struct GetLifecycle;

impl Request for GetLifecycle {
    type Response = LifecycleSeen;
}

#[acton_message]
struct LifecycleSeen {
    started: Vec<String>,
    finished: Vec<(String, String)>,
}

async fn spawn_lifecycle_spy(runtime: &mut ActorRuntime) -> ActorHandle {
    let mut builder = runtime.new_actor_with_name::<LifecycleSpy>("lifecycle_spy".to_string());

    builder.mutate_on::<TurnLifecycle>(|actor, envelope| {
        match envelope.message() {
            TurnLifecycle::TurnStarted { turn_id, .. } => {
                actor.model.started.push(turn_id.to_string());
            }
            TurnLifecycle::TurnFinished {
                turn_id, outcome, ..
            } => {
                actor
                    .model
                    .finished
                    .push((turn_id.to_string(), outcome.to_string()));
            }
            _ => {}
        }
        Reply::ready()
    });

    builder.act_on::<GetLifecycle>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let started = actor.model.started.clone();
        let finished = actor.model.finished.clone();
        Reply::pending(async move {
            reply.send(LifecycleSeen { started, finished }).await;
        })
    });

    // On the builder, before start: a subscription registered afterwards is
    // silently ignored.
    builder.handle().subscribe::<TurnLifecycle>().await;

    builder.start().await
}

/// Asks for status until the in-flight turn count reaches zero.
///
/// The interrupted `TurnFinished` is published from a task the drop guard
/// spawned, so no future the test holds can be awaited for it; this is the
/// same loop a real `acton-ai drain --wait` runs against the status surface.
/// Bounded, and paced by asks and yields rather than by the clock.
async fn status_once_idle(client: &IpcClient, ai: &ActonAI) -> StatusReport {
    for _ in 0..10_000 {
        flush_broadcasts(ai).await;
        let report = ask_status(client).await;
        if report.active_turns == 0 {
            return report;
        }
        tokio::task::yield_now().await;
    }
    panic!("the interrupted turn was never finished; a drain would hang forever");
}

#[tokio::test]
async fn a_dropped_turn_future_still_finishes_its_lifecycle_and_a_drain_completes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let socket = dir.path().join("dropped.sock");
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "gate", json!({})),
        Round::text("never reached"),
    ])
    .await;
    let mut ai = runtime_listening_at(&server, "dropped-turn-agent", &socket).await;
    let spy = spawn_lifecycle_spy(ai.runtime_mut()).await;

    let gate = GatedTool::new();
    let started = Arc::clone(&gate.started);

    // Held locally, not spawned: the whole point is to drop it.
    let mut turn = Box::pin(
        ai.prompt("run the gated tool")
            .tool(
                "gate",
                "Blocks until the test releases it",
                json!({"type": "object", "properties": {}}),
                move |_args| {
                    let started = Arc::clone(&gate.started);
                    let release = Arc::clone(&gate.release);
                    async move {
                        started.notify_one();
                        release.notified().await;
                        Ok(json!({"released": true}))
                    }
                },
            )
            .collect(),
    );

    // Drive the turn until the tool body is running, then drop it mid-tool —
    // the exact shape of a user cancelling from a select! arm.
    tokio::select! {
        _ = &mut turn => panic!("the gated tool never returns, so the turn cannot complete"),
        () = started.notified() => {}
    }
    drop(turn);

    // The drop guard's TurnFinished settles the in-flight count to zero.
    let client = connect(&socket).await;
    let report = status_once_idle(&client, &ai).await;
    assert_eq!(report.turns_started, 1);
    assert_eq!(
        report.in_flight_tool_calls, 0,
        "the TurnFinished sweep must reap the tool the dropped turn was parked in"
    );

    // Which is precisely what lets a drain complete instead of hanging on a
    // turn that no longer exists.
    let ack: AdmissionAck = ask_admission!(client, Drain);
    assert!(
        ack.is_drained(),
        "a dropped turn must not hold a drain open: {ack:?}"
    );

    // The pair is balanced and the finish is distinguishable from success.
    flush_broadcasts(&ai).await;
    let seen: LifecycleSeen = spy.ask(GetLifecycle).await.expect("the spy answers");
    assert_eq!(seen.started.len(), 1);
    assert_eq!(
        seen.finished,
        vec![(seen.started[0].clone(), "interrupted".to_string())],
        "exactly one finish, for the started turn, marked interrupted"
    );

    ai.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn a_completed_turn_finishes_exactly_once_as_completed() {
    let server = MockServer::start(vec![Round::text("done")]).await;
    let mut ai = runtime_pointed_at(&server, "completed-turn-agent").await;
    let spy = spawn_lifecycle_spy(ai.runtime_mut()).await;

    ai.prompt("say done").collect().await.expect("the prompt");

    // A guard that also fired from Drop on the normal path would show up
    // here as a second finish for the same turn.
    flush_broadcasts(&ai).await;
    let seen: LifecycleSeen = spy.ask(GetLifecycle).await.expect("the spy answers");
    assert_eq!(seen.started.len(), 1);
    assert_eq!(
        seen.finished,
        vec![(seen.started[0].clone(), "completed".to_string())]
    );

    ai.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn a_failed_turn_finishes_exactly_once_as_failed() {
    // One scripted failure and no failover chain: the turn returns an error.
    let server = MockServer::start(vec![Round::server_error()]).await;
    let mut ai = runtime_pointed_at(&server, "failed-turn-agent").await;
    let spy = spawn_lifecycle_spy(ai.runtime_mut()).await;

    ai.prompt("try anyway")
        .collect()
        .await
        .expect_err("the scripted round fails");

    flush_broadcasts(&ai).await;
    let seen: LifecycleSeen = spy.ask(GetLifecycle).await.expect("the spy answers");
    assert_eq!(seen.started.len(), 1);
    assert_eq!(
        seen.finished,
        vec![(seen.started[0].clone(), "failed".to_string())],
        "an error is a finished turn, distinguishable from success and from interruption"
    );

    ai.shutdown().await.expect("shutdown");
}
