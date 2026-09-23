//! End-to-end tests for the clock: ticks in, one timeline out.
//!
//! These launch real runtimes with a driven [`TimeSource`], so every time a
//! test asserts on is one the test chose. Nothing sleeps: ticks are awaited on
//! a channel fed by `on_tick`, and a socket tick is an ordinary request /
//! response over a real Unix socket.
//!
//! No test here talks to a model. The provider points at a port nothing
//! listens on, which launch never contacts.

use std::path::Path;
use std::time::Duration;

use acton_ai::clock::{read_journal, EventTime, TickOrigin, Ticked, TimeSource};
use acton_ai::prelude::*;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

// =============================================================================
// Fixtures
// =============================================================================

fn at(text: &str) -> EventTime {
    EventTime::parse_rfc3339(text).expect("a valid timestamp")
}

/// A builder with a provider that is never called.
fn builder() -> ActonAIBuilder {
    ActonAI::builder()
        .app_name("clock-test")
        .provider(ProviderConfig::openai_compatible(
            "http://127.0.0.1:9",
            "never-called",
        ))
}

/// Subscribes to ticks, returning the builder and the receiving end.
fn with_tick_channel(builder: ActonAIBuilder) -> (ActonAIBuilder, UnboundedReceiver<Ticked>) {
    let (sender, receiver) = unbounded_channel();
    let builder = builder.on_tick(move |tick| {
        let _ = sender.send(tick);
    });
    (builder, receiver)
}

/// The next tick the runtime broadcasts.
async fn next(ticks: &mut UnboundedReceiver<Ticked>) -> Ticked {
    tokio::time::timeout(Duration::from_secs(5), ticks.recv())
        .await
        .expect("a tick is broadcast")
        .expect("the tick channel is open")
}

// =============================================================================
// In-process ticks
// =============================================================================

#[tokio::test]
async fn a_tick_is_stamped_from_the_runtime_time_source_and_broadcast() {
    let time = TimeSource::driven(at("2026-09-23T03:00:00Z"));
    let (builder, mut ticks) = with_tick_channel(builder().time_source(time.clone()));
    let ai = builder.launch().await.expect("launch");

    assert_eq!(ai.now(), at("2026-09-23T03:00:00Z"));

    let first = ai.tick().await.expect("tick");
    time.advance_to(at("2026-09-23T03:30:00Z")).unwrap();
    let second = ai.tick().await.expect("tick");

    assert_eq!((first.seq, first.at), (1, at("2026-09-23T03:00:00Z")));
    assert_eq!((second.seq, second.at), (2, at("2026-09-23T03:30:00Z")));
    assert_eq!(first.origin, TickOrigin::InProcess);

    let broadcast: Vec<u64> = vec![next(&mut ticks).await.seq, next(&mut ticks).await.seq];
    assert_eq!(
        broadcast,
        vec![1, 2],
        "listeners see ticks in timeline order"
    );

    ai.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_journal_holds_one_timeline_across_restarts() {
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("ticks.jsonl");
    let time = TimeSource::driven(at("2026-09-23T03:00:00Z"));

    let ai = builder()
        .time_source(time.clone())
        .tick_journal(&journal)
        .launch()
        .await
        .expect("launch");
    ai.tick().await.unwrap();
    ai.shutdown().await.unwrap();

    time.advance_to(at("2026-09-23T04:00:00Z")).unwrap();
    let ai = builder()
        .time_source(time)
        .tick_journal(&journal)
        .launch()
        .await
        .expect("relaunch");
    assert_eq!(ai.tick().await.unwrap().seq, 2, "the sequence continues");
    ai.shutdown().await.unwrap();

    let recorded: Vec<(u64, EventTime)> = read_journal(&journal)
        .unwrap()
        .into_iter()
        .map(|entry| (entry.seq, entry.at))
        .collect();
    assert_eq!(
        recorded,
        vec![
            (1, at("2026-09-23T03:00:00Z")),
            (2, at("2026-09-23T04:00:00Z"))
        ]
    );
}

#[tokio::test]
async fn an_unreadable_journal_fails_the_launch() {
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("ticks.jsonl");
    std::fs::write(&journal, "not json\n").unwrap();

    let error = builder()
        .tick_journal(&journal)
        .launch()
        .await
        .expect_err("a corrupt journal must stop the launch");
    assert!(error.is_configuration(), "{error}");
    assert!(error.to_string().contains("line 1"), "{error}");
}

// =============================================================================
// Replay
// =============================================================================

/// Records a three-tick journal a week apart, the way a live process would.
async fn record_a_week(journal: &Path) {
    let time = TimeSource::driven(at("2026-09-01T03:00:00Z"));
    let ai = builder()
        .time_source(time.clone())
        .tick_journal(journal)
        .launch()
        .await
        .expect("launch");
    ai.tick().await.unwrap();
    time.advance_to(at("2026-09-08T03:00:00Z")).unwrap();
    ai.tick().await.unwrap();
    time.advance_to(at("2026-09-15T03:00:00Z")).unwrap();
    ai.tick().await.unwrap();
    ai.shutdown().await.unwrap();
}

#[tokio::test]
async fn replay_reproduces_the_recorded_timeline() {
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("ticks.jsonl");
    record_a_week(&journal).await;

    let time = TimeSource::driven(at("2026-01-01T00:00:00Z"));
    let (builder, mut ticks) = with_tick_channel(builder().time_source(time));
    let ai = builder.launch().await.expect("launch");

    assert_eq!(ai.replay_journal(&journal).await.expect("replay"), 3);
    assert_eq!(
        ai.now(),
        at("2026-09-15T03:00:00Z"),
        "time ends where the journal ended, not today"
    );

    let mut replayed = Vec::new();
    for _ in 0..3 {
        let tick = next(&mut ticks).await;
        assert!(tick.replayed);
        replayed.push((tick.seq, tick.at));
    }
    assert_eq!(
        replayed,
        vec![
            (1, at("2026-09-01T03:00:00Z")),
            (2, at("2026-09-08T03:00:00Z")),
            (3, at("2026-09-15T03:00:00Z")),
        ]
    );
    ai.shutdown().await.unwrap();
}

#[tokio::test]
async fn replay_is_refused_on_the_wall_clock() {
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("ticks.jsonl");
    record_a_week(&journal).await;

    let ai = builder().launch().await.expect("launch");
    let error = ai
        .replay_journal(&journal)
        .await
        .expect_err("the wall clock cannot be replayed into");
    assert!(error.is_clock(), "{error}");
    ai.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_recording_runtime_refuses_to_replay() {
    let dir = tempfile::tempdir().unwrap();
    let recorded = dir.path().join("recorded.jsonl");
    record_a_week(&recorded).await;

    let ai = builder()
        .time_source(TimeSource::driven(at("2026-01-01T00:00:00Z")))
        .tick_journal(dir.path().join("live.jsonl"))
        .launch()
        .await
        .expect("launch");
    let error = ai
        .replay_journal(&recorded)
        .await
        .expect_err("replay would be written back as new history");
    assert!(error.is_clock(), "{error}");
    ai.shutdown().await.unwrap();
}

// =============================================================================
// Ticks over the socket
// =============================================================================

#[cfg(all(feature = "ipc", unix))]
mod socket {
    use super::*;
    use acton_ai::introspection::wire::{Tick, TickAck, INTROSPECTION_ACTOR_NAME};
    use acton_reactive::ipc::IpcClient;

    async fn ask_tick(socket: &Path) -> TickAck {
        IpcClient::connect(socket)
            .await
            .expect("the socket accepts a client")
            .actor(INTROSPECTION_ACTOR_NAME)
            .ask(Tick)
            .await
            .expect("a running process answers Tick")
    }

    #[tokio::test]
    async fn a_socket_tick_is_stamped_by_the_process_not_the_client() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("agent.sock");
        let journal = dir.path().join("ticks.jsonl");
        let time = TimeSource::driven(at("2026-09-23T03:00:00Z"));

        let (builder, mut ticks) = with_tick_channel(
            builder()
                .time_source(time)
                .tick_journal(&journal)
                .introspection_at(&socket),
        );
        let ai = builder.launch().await.expect("launch");

        let ack = ask_tick(&socket).await;
        assert!(ack.is_recorded(), "{ack:?}");
        assert_eq!(ack.seq, Some(1));
        assert_eq!(ack.at.as_deref(), Some("2026-09-23T03:00:00.000000Z"));
        assert_eq!(ack.admission, "running");

        let tick = next(&mut ticks).await;
        assert_eq!((tick.seq, tick.origin), (1, TickOrigin::Socket));
        assert_eq!(read_journal(&journal).unwrap().len(), 1);

        ai.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn a_paused_process_still_records_that_time_passed() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("agent.sock");
        let ai = builder()
            .time_source(TimeSource::driven(at("2026-09-23T03:00:00Z")))
            .introspection_at(&socket)
            .launch()
            .await
            .expect("launch");

        ai.pause();
        let ack = ask_tick(&socket).await;
        assert!(ack.is_recorded(), "{ack:?}");
        assert_eq!(
            ack.admission, "paused",
            "the ack says the work is held back"
        );

        ai.shutdown().await.unwrap();
    }
}
