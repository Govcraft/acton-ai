//! The [`ClockActor`]: the one place a tick enters the runtime.
//!
//! # Handler discipline
//!
//! Both handlers are `mutate_on`, so ticks are sequenced one at a time in
//! mailbox order: the sequence number, the timestamp, and the journal line of
//! a tick are all settled before the next tick is looked at. That order is the
//! timeline, and it is the order a replay reproduces.
//!
//! The journal write happens inside the handler, synchronously, before the
//! model moves on. Ticks arrive minutes apart, so blocking for one `fsync` is
//! nothing; what it buys is that a tick is either recorded *and* broadcast or
//! neither. A failed write leaves the timeline where it was, and the caller is
//! told — a tick acknowledged but missing from the journal would make every
//! replay of that journal a lie about what happened.

use std::path::Path;
use std::sync::Arc;

use acton_reactive::prelude::*;

use crate::clock::journal::{check_follows, JournalWriter};
use crate::clock::{ClockError, EventTime, JournalEntry, JournalEvent, TickOrigin, TimeSource};

/// The name the clock actor is spawned under.
pub const CLOCK_ACTOR_NAME: &str = "clock";

/// Asks the clock to record a tick now.
///
/// Carries no timestamp on purpose. The clock stamps a tick with its own
/// [`TimeSource`], so nothing that can send a tick — including any client of
/// the introspection socket — can choose what time the runtime believes it is.
#[acton_message]
pub struct RecordTick {
    /// Who is asking.
    pub origin: TickOrigin,
}

/// The clock's answer to [`RecordTick`].
#[acton_message]
pub struct TickRecorded {
    /// The tick as recorded and broadcast, or why it was not.
    pub result: Result<Ticked, ClockError>,
}

impl Request for RecordTick {
    type Response = TickRecorded;
}

/// Feeds one recorded journal entry back through the clock.
///
/// Accepted only by a clock with a driven [`TimeSource`] that is not itself
/// writing a journal, and only in order: the entry must be the one that comes
/// next on the timeline.
#[acton_message]
pub struct ReplayEntry {
    /// The entry, as [`read_journal`](crate::clock::read_journal) returned it.
    pub entry: JournalEntry,
}

/// The clock's answer to [`ReplayEntry`].
#[acton_message]
pub struct EntryReplayed {
    /// The entry once replayed, or why it was refused.
    pub result: Result<JournalEntry, ClockError>,
}

impl Request for ReplayEntry {
    type Response = EntryReplayed;
}

/// Broadcast for every tick the clock accepts, live or replayed.
///
/// This is what anything running on rhythm subscribes to. `at` is the time to
/// act on: a subscriber that compares `next_due` against `at`, rather than
/// counting ticks, does the same thing whether ticks were late, missed, or
/// replayed a week later in a few milliseconds.
#[acton_message]
pub struct Ticked {
    /// Position on the timeline, from 1.
    pub seq: u64,
    /// When the tick happened.
    pub at: EventTime,
    /// Who asked for it.
    pub origin: TickOrigin,
    /// Whether this tick came from a journal rather than from now.
    pub replayed: bool,
}

impl Ticked {
    fn of(entry: &JournalEntry, replayed: bool) -> Self {
        let JournalEvent::Tick { origin } = entry.event;
        Self {
            seq: entry.seq,
            at: entry.at,
            origin,
            replayed,
        }
    }
}

/// Sequences, stamps, journals, and broadcasts ticks.
#[acton_actor]
pub struct ClockActor {
    /// Where "now" comes from.
    time: TimeSource,
    /// The journal being written, when one was configured.
    writer: Option<JournalWriter>,
    /// The latest entry on the timeline, whether written here or replayed.
    last: Option<JournalEntry>,
}

impl ClockActor {
    /// Spawns the clock.
    ///
    /// With a `journal`, the file is opened (or created) and read first, and
    /// the timeline continues from its last entry: a restarted process picks
    /// up the sequence where the previous one left it, so one journal holds
    /// one timeline across restarts.
    ///
    /// # Errors
    ///
    /// Any error opening or validating the journal. A journal that cannot be
    /// read is a launch failure, not something to discover on the first tick.
    pub async fn spawn(
        runtime: &mut ActorRuntime,
        time: TimeSource,
        journal: Option<&Path>,
    ) -> Result<ActorHandle, ClockError> {
        let (writer, last) = match journal {
            Some(path) => {
                let (writer, existing) = JournalWriter::open(path)?;
                (Some(writer), existing.last().cloned())
            }
            None => (None, None),
        };

        let mut builder = runtime.new_actor_with_name::<ClockActor>(CLOCK_ACTOR_NAME.to_string());
        builder.model.time = time;
        builder.model.writer = writer;
        builder.model.last = last;

        builder.mutate_on::<RecordTick>(|actor, envelope| {
            let reply = envelope.reply_envelope();
            let broker = actor.broker().clone();
            let result = actor
                .model
                .record(envelope.message().origin)
                .map(|entry| Ticked::of(&entry, false));
            Reply::pending(async move {
                if let Ok(ref ticked) = result {
                    broker.broadcast(ticked.clone()).await;
                }
                reply.send(TickRecorded { result }).await;
            })
        });

        builder.mutate_on::<ReplayEntry>(|actor, envelope| {
            let reply = envelope.reply_envelope();
            let broker = actor.broker().clone();
            let result = actor.model.replay(envelope.message().entry.clone());
            let ticked = result.as_ref().ok().map(|entry| Ticked::of(entry, true));
            Reply::pending(async move {
                if let Some(ticked) = ticked {
                    broker.broadcast(ticked).await;
                }
                reply.send(EntryReplayed { result }).await;
            })
        });

        Ok(builder.start().await)
    }

    /// Stamps, journals, and adopts one live tick.
    fn record(&mut self, origin: TickOrigin) -> Result<JournalEntry, ClockError> {
        let mut at = self.time.now();
        if let Some(last) = &self.last {
            // The wall clock can step backwards (NTP, a VM restored from a
            // snapshot). The timeline cannot, so the tick is pinned to the
            // last time instead: late is recoverable, backwards is not.
            if at < last.at {
                tracing::warn!(
                    wall_clock = %at,
                    timeline = %last.at,
                    "the wall clock is behind the timeline; stamping the tick with the \
                     timeline's time",
                );
                at = last.at;
            }
        }

        let entry = JournalEntry {
            seq: self.last.as_ref().map_or(1, |last| last.seq + 1),
            at,
            event: JournalEvent::Tick { origin },
        };

        if let Some(writer) = &self.writer {
            writer.append(&entry)?;
        }
        tracing::debug!(seq = entry.seq, at = %entry.at, origin = origin.as_str(), "tick");
        self.last = Some(entry.clone());
        Ok(entry)
    }

    /// Checks and adopts one replayed entry.
    fn replay(&mut self, entry: JournalEntry) -> Result<JournalEntry, ClockError> {
        if let Some(writer) = &self.writer {
            return Err(ClockError::replay_refused(format!(
                "this clock is writing the journal at {}; a process that replays must not \
                 also record, or the replay would be written back as new history",
                writer.path().display()
            )));
        }
        if !self.time.is_driven() {
            return Err(ClockError::replay_refused(
                "replay moves time to each recorded tick, so the runtime needs a driven time \
                 source; build it with .time_source(TimeSource::driven(..))",
            ));
        }
        check_follows(self.last.as_ref(), &entry)?;
        self.time.advance_to(entry.at)?;
        self.last = Some(entry.clone());
        Ok(entry)
    }
}

/// A caller's tick callback, as the listener holds it.
#[derive(Clone, Default)]
pub(crate) struct TickCallback(pub(crate) Option<Arc<dyn Fn(Ticked) + Send + Sync>>);

impl std::fmt::Debug for TickCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("TickCallback")
            .field(&self.0.is_some())
            .finish()
    }
}

/// Runs a caller's callback for every [`Ticked`] on the broker.
#[acton_actor]
pub(crate) struct TickListener {
    callback: TickCallback,
}

impl TickListener {
    /// Spawns the listener, subscribed to [`Ticked`] **on the builder**.
    ///
    /// Subscribing after `start()` is silently ignored, which would leave a
    /// listener that runs happily and never fires.
    pub(crate) async fn spawn(
        runtime: &mut ActorRuntime,
        callback: Arc<dyn Fn(Ticked) + Send + Sync>,
    ) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<TickListener>("tick_listener".to_string());
        builder.model.callback = TickCallback(Some(callback));

        // `mutate_on`, unlike the budget listener's `act_on`: a caller that
        // runs work per tick is entitled to see ticks in timeline order, and
        // concurrent `act_on` dispatch would not promise that.
        builder.mutate_on::<Ticked>(|actor, envelope| {
            if let Some(ref callback) = actor.model.callback.0 {
                callback(envelope.message().clone());
            }
            Reply::ready()
        });

        builder.handle().subscribe::<Ticked>().await;
        builder.start().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{read_journal, ClockErrorKind};

    fn at(text: &str) -> EventTime {
        EventTime::parse_rfc3339(text).unwrap()
    }

    fn tick_entry(seq: u64, time: &str) -> JournalEntry {
        JournalEntry {
            seq,
            at: at(time),
            event: JournalEvent::Tick {
                origin: TickOrigin::Socket,
            },
        }
    }

    async fn record(clock: &ActorHandle) -> Result<Ticked, ClockError> {
        clock
            .ask(RecordTick {
                origin: TickOrigin::Socket,
            })
            .await
            .expect("clock answers")
            .result
    }

    async fn replay(clock: &ActorHandle, entry: JournalEntry) -> Result<JournalEntry, ClockError> {
        clock
            .ask(ReplayEntry { entry })
            .await
            .expect("clock answers")
            .result
    }

    #[tokio::test]
    async fn ticks_are_numbered_and_stamped_from_the_time_source() {
        let mut runtime = ActonApp::launch_async().await;
        let time = TimeSource::driven(at("2026-09-23T03:00:00Z"));
        let clock = ClockActor::spawn(&mut runtime, time.clone(), None)
            .await
            .unwrap();

        let first = record(&clock).await.unwrap();
        time.advance_to(at("2026-09-23T03:30:00Z")).unwrap();
        let second = record(&clock).await.unwrap();

        assert_eq!((first.seq, first.at), (1, at("2026-09-23T03:00:00Z")));
        assert_eq!((second.seq, second.at), (2, at("2026-09-23T03:30:00Z")));
        assert!(!second.replayed);
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn a_journaled_timeline_continues_across_restarts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ticks.jsonl");
        let time = TimeSource::driven(at("2026-09-23T03:00:00Z"));

        let mut runtime = ActonApp::launch_async().await;
        let clock = ClockActor::spawn(&mut runtime, time.clone(), Some(&path))
            .await
            .unwrap();
        record(&clock).await.unwrap();
        record(&clock).await.unwrap();
        runtime.shutdown_all().await.unwrap();

        let mut runtime = ActonApp::launch_async().await;
        let clock = ClockActor::spawn(&mut runtime, time, Some(&path))
            .await
            .unwrap();
        assert_eq!(record(&clock).await.unwrap().seq, 3);
        runtime.shutdown_all().await.unwrap();

        let seqs: Vec<u64> = read_journal(&path).unwrap().iter().map(|e| e.seq).collect();
        assert_eq!(seqs, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn a_clock_behind_the_timeline_is_pinned_rather_than_run_backwards() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ticks.jsonl");
        std::fs::write(
            &path,
            format!(
                "{}\n",
                serde_json::to_string(&tick_entry(1, "2026-09-23T05:00:00Z")).unwrap()
            ),
        )
        .unwrap();

        let mut runtime = ActonApp::launch_async().await;
        let behind = TimeSource::driven(at("2026-09-23T04:00:00Z"));
        let clock = ClockActor::spawn(&mut runtime, behind, Some(&path))
            .await
            .unwrap();

        let ticked = record(&clock).await.unwrap();
        assert_eq!((ticked.seq, ticked.at), (2, at("2026-09-23T05:00:00Z")));
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn an_unwritable_journal_refuses_the_tick_and_keeps_the_timeline() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ticks.jsonl");

        let mut runtime = ActonApp::launch_async().await;
        let clock = ClockActor::spawn(&mut runtime, TimeSource::system(), Some(&path))
            .await
            .unwrap();
        // Replacing the file with a directory makes every append fail.
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();

        let error = record(&clock).await.unwrap_err();
        assert!(error.is_journal_error(), "{error}");

        std::fs::remove_dir(&path).unwrap();
        std::fs::write(&path, "").unwrap();
        assert_eq!(
            record(&clock).await.unwrap().seq,
            1,
            "the refused tick did not consume a sequence number"
        );
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn replay_moves_time_to_each_recorded_tick() {
        let mut runtime = ActonApp::launch_async().await;
        let time = TimeSource::driven(at("2026-09-23T00:00:00Z"));
        let clock = ClockActor::spawn(&mut runtime, time.clone(), None)
            .await
            .unwrap();

        replay(&clock, tick_entry(1, "2026-09-23T03:00:00Z"))
            .await
            .unwrap();
        assert_eq!(time.now(), at("2026-09-23T03:00:00Z"));
        replay(&clock, tick_entry(2, "2026-09-30T03:00:00Z"))
            .await
            .unwrap();
        assert_eq!(time.now(), at("2026-09-30T03:00:00Z"));
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn replay_refuses_entries_out_of_order() {
        let mut runtime = ActonApp::launch_async().await;
        let time = TimeSource::driven(at("2026-09-23T00:00:00Z"));
        let clock = ClockActor::spawn(&mut runtime, time, None).await.unwrap();

        let error = replay(&clock, tick_entry(2, "2026-09-23T03:00:00Z"))
            .await
            .unwrap_err();
        assert!(matches!(
            error.kind(),
            ClockErrorKind::OutOfSequence {
                expected: 1,
                found: 2
            }
        ));
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn replay_needs_a_driven_clock_that_is_not_recording() {
        let dir = tempfile::tempdir().unwrap();
        let mut runtime = ActonApp::launch_async().await;

        let wall = ClockActor::spawn(&mut runtime, TimeSource::system(), None)
            .await
            .unwrap();
        let error = replay(&wall, tick_entry(1, "2026-09-23T03:00:00Z"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("driven"), "{error}");

        let recording = ClockActor::spawn(
            &mut runtime,
            TimeSource::driven(at("2026-09-23T00:00:00Z")),
            Some(&dir.path().join("ticks.jsonl")),
        )
        .await
        .unwrap();
        let error = replay(&recording, tick_entry(1, "2026-09-23T03:00:00Z"))
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("must not also record"),
            "{error}"
        );
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn listeners_see_every_tick_in_order() {
        let mut runtime = ActonApp::launch_async().await;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        TickListener::spawn(
            &mut runtime,
            Arc::new(move |ticked: Ticked| {
                let _ = sender.send((ticked.seq, ticked.replayed));
            }),
        )
        .await;

        let time = TimeSource::driven(at("2026-09-23T00:00:00Z"));
        let clock = ClockActor::spawn(&mut runtime, time, None).await.unwrap();
        for seq in 1..=3 {
            replay(&clock, tick_entry(seq, "2026-09-23T03:00:00Z"))
                .await
                .unwrap();
        }

        let mut seen = Vec::new();
        for _ in 0..3 {
            let next = tokio::time::timeout(std::time::Duration::from_secs(5), receiver.recv())
                .await
                .expect("the listener fires")
                .expect("the channel is open");
            seen.push(next);
        }
        assert_eq!(seen, vec![(1, true), (2, true), (3, true)]);
        runtime.shutdown_all().await.unwrap();
    }
}
