//! Time as an input: one timeline, stamped at the edge, replayable.
//!
//! A process that reads the wall clock wherever it needs the time cannot be
//! replayed: run it again tomorrow and every timestamp, every "is this due
//! yet", comes out different. This module moves time to the edge of the
//! system. Time is read from a [`TimeSource`] that the embedder chooses, and
//! the rhythm that drives scheduled work arrives as ticks through a single
//! actor that numbers, stamps, and journals each one before anything else
//! sees it.
//!
//! ## Architecture
//!
//! ```text
//!  systemd timer ──► acton-ai tick ──► introspection socket ─┐
//!                                                            ├─► ClockActor ──► Ticked (broadcast)
//!  ActonAI::tick() ──────────────────────────────────────────┘      │
//!                                                                   ├─► journal (append, fsync)
//!  journal file ──► ActonAI::replay_journal() ──► ReplayEntry ──────┘
//! ```
//!
//! - [`EventTime`] is a UTC instant to the microsecond, rendered the same way
//!   everywhere it is written.
//! - [`TimeSource`] is where the runtime reads "now": the wall clock in a live
//!   process, or a driven timeline that only moves when told to, in a replay
//!   or a test.
//! - [`ClockActor`] is the one entry point for ticks. It gives each one the
//!   next sequence number and the time source's current time, appends it to
//!   the journal, and only then broadcasts [`Ticked`]. A tick request carries
//!   no timestamp, so whoever can send one — including any client of the
//!   introspection socket — cannot choose what time the process believes it
//!   is.
//! - The journal ([`read_journal`], [`JournalWriter`]) is one JSON object per
//!   line, sequence-numbered from 1 with no gaps and times that never go
//!   backwards. Replay feeds it back through the same actor, which moves a
//!   driven time source to each entry's time in turn.
//!
//! ## What is and is not on the timeline
//!
//! Calendar time — timestamps in records, `next_due` comparisons, the time a
//! heartbeat ran — comes from the [`TimeSource`]. Durations do not: timeouts,
//! retry backoff, rate limits, and circuit-breaker cooldowns measure elapsed
//! time with a monotonic [`Instant`](std::time::Instant), and replaying them
//! would be meaningless. They belong to the calls whose *outcomes* a later
//! journal will record.
//!
//! Ticks are the only entries the journal holds today. Deterministic replay of
//! a whole process also needs every model completion and tool result on the
//! same timeline; the journal's `kind` tag is where those will go.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//! use acton_ai::clock::{EventTime, TimeSource};
//!
//! // Live: wall-clock time, every tick journaled.
//! let ai = ActonAI::builder()
//!     .ollama("qwen2.5:7b")
//!     .tick_journal("/var/lib/my-agent/ticks.jsonl")
//!     .on_tick(|tick| tracing::info!(seq = tick.seq, at = %tick.at, "tick"))
//!     .launch()
//!     .await?;
//! ai.tick().await?;
//!
//! // Replay: time moves only to the times the journal recorded.
//! let start = EventTime::parse_rfc3339("2026-01-01T00:00:00Z")?;
//! let ai = ActonAI::builder()
//!     .ollama("qwen2.5:7b")
//!     .time_source(TimeSource::driven(start))
//!     .launch()
//!     .await?;
//! let replayed = ai.replay_journal("/var/lib/my-agent/ticks.jsonl").await?;
//! ```

mod actor;
mod error;
mod journal;
mod time;

pub(crate) use actor::TickListener;
pub use actor::{
    ClockActor, EntryReplayed, RecordTick, ReplayEntry, TickRecorded, Ticked, CLOCK_ACTOR_NAME,
};
pub use error::{ClockError, ClockErrorKind};
pub use journal::{
    read_journal, JournalEntry, JournalEvent, JournalWriter, TickOrigin, JOURNAL_MODE,
};
pub use time::{EventTime, TimeSource};
