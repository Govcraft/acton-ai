//! Checkpoint and resume for the prompt loop.
//!
//! A turn can run for many provider rounds and execute many tools before it
//! produces an answer. Without a checkpoint, a process that dies in the middle
//! of one loses all of it: the rounds are re-dispatched, the tools re-executed,
//! and the caller pays twice. This module records how far a turn got, and
//! decides what a later run may do with that record.
//!
//! ## The three pieces
//!
//! - [`CheckpointRecord`] — the saved progress: the conversation as the next
//!   round would send it, the rounds already spent, the tools already
//!   executed, and a [`TurnFingerprint`] of the inputs it belongs to.
//! - [`plan_resume`] — the decision, as a pure function. Start fresh, resume
//!   from round K, or replay a finished answer, with every refusal spelled out
//!   as a [`CheckpointError`].
//! - [`CheckpointSink`] — the only impure piece: an opt-in handle that loads
//!   and saves records through the [`MemoryStore`](crate::memory::MemoryStore)
//!   actor. A prompt with no checkpoint configured holds an empty sink, and
//!   every call on it is a no-op.
//!
//! ## Why the fingerprint
//!
//! A checkpoint is keyed by a [`CheckpointId`](crate::types::CheckpointId) the
//! caller chooses, which is what lets a resume find it after a restart. But an
//! ID alone cannot say whether the saved conversation belongs to the turn
//! being started now. Resuming a record written for a different prompt, a
//! different tool set, or a different provider would splice two turns together
//! and bill the caller for a conversation nobody asked for, so the planner
//! refuses it. That is a deliberate error, not a silent fresh start: silently
//! discarding saved progress is exactly the outcome a checkpoint exists to
//! prevent.
//!
//! ## Using it
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//! use acton_ai::types::CheckpointId;
//!
//! # async fn run(runtime: ActonAI, store: ActorHandle) -> Result<(), ActonAIError> {
//! // Persist this ID alongside whatever work the turn belongs to; passing it
//! // again is what makes the second attempt a resume rather than a rerun.
//! let checkpoint = CheckpointId::new();
//!
//! let answer = runtime
//!     .prompt("Summarize every .rs file under src/")
//!     .use_builtins()
//!     .checkpoint(store, checkpoint)
//!     .collect()
//!     .await?;
//! # Ok(())
//! # }
//! ```

mod error;
mod plan;
mod policy;
mod record;
mod sink;

pub use error::{CheckpointError, CheckpointErrorKind};
pub use plan::{
    abandon, advance, complete, fail, plan_from_record, plan_resume, resolve_pending_call,
    uncertain_feedback, FinalAnswer, PendingCallAction, ResumePlan, RoundProgress,
};
pub use policy::{CheckpointConfig, ResumePolicy};
pub use record::{
    decode_row, encode_record, CheckpointColumns, CheckpointRecord, CheckpointStatus,
    PendingCallState, PendingRound, PendingToolCall, TurnFingerprint, TurnInputs,
    CHECKPOINT_FORMAT_VERSION,
};
pub use sink::CheckpointSink;
