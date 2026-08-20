//! The checkpoint sink: the only part of this module that talks to an actor.
//!
//! Everything the prompt loop needs is behind three methods, and all three are
//! no-ops on a sink that was never configured. That is what keeps checkpoint
//! support out of the way of a prompt that did not ask for it: the loop calls
//! the same four places either way, and an empty sink answers
//! [`ResumePlan::Start`] and returns without sending anything.

use crate::checkpoint::error::CheckpointError;
use crate::checkpoint::plan::{self, FinalAnswer, ResumePlan, RoundProgress};
use crate::checkpoint::record::{CheckpointRecord, TurnFingerprint, TurnInputs};
use crate::error::ActonAIError;
use crate::memory::{
    CheckpointClaimed, CheckpointLoaded, CheckpointReleased, CheckpointSaved, ClaimCheckpoint,
    LoadCheckpoint, ReleaseCheckpoint, SaveCheckpoint,
};
use crate::types::{CheckpointId, ConversationId};
use acton_reactive::prelude::*;

/// Loads and saves one turn's checkpoint through the
/// [`MemoryStore`](crate::memory::MemoryStore) actor.
///
/// Constructed empty by default. [`PromptBuilder::checkpoint`](crate::prompt::PromptBuilder::checkpoint)
/// is what fills it in.
#[derive(Debug, Clone, Default)]
pub struct CheckpointSink {
    target: Option<Target>,
}

/// Where a configured sink writes, and what it last wrote.
#[derive(Debug, Clone)]
struct Target {
    store: ActorHandle,
    id: CheckpointId,
}

impl CheckpointSink {
    /// A sink that records nothing. Every method on it is a no-op.
    #[must_use]
    pub fn disabled() -> Self {
        Self { target: None }
    }

    /// A sink that records this turn under `id`, through `store`.
    #[must_use]
    pub fn to_store(store: ActorHandle, id: CheckpointId) -> Self {
        Self {
            target: Some(Target { store, id }),
        }
    }

    /// Whether this sink records anything.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.target.is_some()
    }

    /// The checkpoint this sink writes under, if it writes at all.
    #[must_use]
    pub fn id(&self) -> Option<&CheckpointId> {
        self.target.as_ref().map(|t| &t.id)
    }

    /// Loads the stored checkpoint, if any, and decides what to do with it.
    ///
    /// A disabled sink answers [`ResumePlan::Start`] without a round trip.
    ///
    /// # Errors
    ///
    /// Returns an error when the store cannot be reached, when the lookup
    /// fails, or when the stored record does not entitle this turn to resume —
    /// see [`plan_resume`](crate::checkpoint::plan_resume) for the refusals.
    pub async fn plan(&self, inputs: &TurnInputs<'_>) -> Result<ResumePlan, ActonAIError> {
        let Some(ref target) = self.target else {
            return Ok(ResumePlan::Start);
        };

        let record = self.load(target).await?;

        plan::plan_resume(record.as_ref(), inputs)
            .map_err(|e| ActonAIError::from(e.with_checkpoint(target.id.clone())))
    }

    /// Records the turn's state after one more round.
    ///
    /// A disabled sink sends nothing.
    ///
    /// # Errors
    ///
    /// Returns an error when the store cannot be reached or the write fails.
    /// A checkpoint that cannot be written is worth failing the turn over: a
    /// turn that believes it is checkpointed and is not will silently redo
    /// everything on the next attempt, which is the outcome the caller asked
    /// for a checkpoint to avoid.
    pub async fn record_progress(
        &self,
        conversation_id: Option<&ConversationId>,
        fingerprint: &TurnFingerprint,
        progress: RoundProgress,
    ) -> Result<(), ActonAIError> {
        let Some(ref target) = self.target else {
            return Ok(());
        };

        let record = plan::advance(
            target.id.clone(),
            conversation_id.cloned(),
            fingerprint.clone(),
            progress,
        );
        self.write(target, record).await
    }

    /// Records the finished answer, so a later run replays it instead of
    /// paying for the turn again.
    ///
    /// # Errors
    ///
    /// Returns an error when the store cannot be reached or the write fails.
    pub async fn record_completion(
        &self,
        conversation_id: Option<&ConversationId>,
        fingerprint: &TurnFingerprint,
        progress: RoundProgress,
        answer: FinalAnswer,
    ) -> Result<(), ActonAIError> {
        let Some(ref target) = self.target else {
            return Ok(());
        };

        let record = plan::complete(
            plan::advance(
                target.id.clone(),
                conversation_id.cloned(),
                fingerprint.clone(),
                progress,
            ),
            answer,
        );
        self.write(target, record).await
    }

    /// Marks the stored checkpoint as belonging to a turn that ended in an
    /// error.
    ///
    /// The saved progress is untouched — a failed turn is still resumable, and
    /// where it got to is the point of the record. All this changes is what an
    /// operator listing checkpoints sees, which is why the prompt loop calls
    /// it on the way out of a failed turn rather than leaving the record
    /// looking like a turn still running.
    ///
    /// A record already in a terminal state — `Completed` or `Abandoned` — is
    /// left exactly as it is. A finished answer must stay replayable and an
    /// operator's abandonment must stay closed, whatever error a later attempt
    /// against the same ID hit on its way out.
    ///
    /// Does nothing when the sink is disabled or nothing was ever written.
    ///
    /// # Errors
    ///
    /// Returns an error when the store cannot be reached or either the read or
    /// the write fails.
    pub async fn mark_failed(&self) -> Result<(), ActonAIError> {
        let Some(ref target) = self.target else {
            return Ok(());
        };

        let Some(record) = self.load(target).await? else {
            return Ok(());
        };

        let failed = plan::fail(record);
        if failed.status != crate::checkpoint::CheckpointStatus::Failed {
            // `fail` declined to touch a terminal record; there is nothing
            // worth writing back.
            return Ok(());
        }

        self.write(target, failed).await
    }

    /// Claims this sink's checkpoint ID for the turn about to run under it.
    ///
    /// A checkpoint has exactly one live owner per process. The prompt loop
    /// claims before it plans, so a second loop aiming at the same ID — an
    /// operator resuming a turn that is still in flight, or a caller's retry
    /// racing the `resume_auto` background task — is refused up front instead
    /// of double-executing the turn's pending tool calls and interleaving
    /// checkpoint writes with the live owner.
    ///
    /// A disabled sink claims nothing and succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointErrorKind::AlreadyRunning`](crate::checkpoint::CheckpointErrorKind::AlreadyRunning)
    /// when another turn in this process holds the ID, and a checkpoint error
    /// when the store cannot be reached.
    pub async fn claim(&self) -> Result<(), ActonAIError> {
        let Some(ref target) = self.target else {
            return Ok(());
        };

        let claimed: CheckpointClaimed = target
            .store
            .ask(ClaimCheckpoint {
                id: target.id.clone(),
            })
            .await
            .map_err(|e| {
                ActonAIError::checkpoint(
                    target.id.clone(),
                    format!("the memory store did not answer the checkpoint claim: {e}"),
                )
            })?;

        if claimed.granted {
            Ok(())
        } else {
            Err(ActonAIError::from(
                CheckpointError::already_running().with_checkpoint(target.id.clone()),
            ))
        }
    }

    /// Releases the claim taken by [`claim`](Self::claim).
    ///
    /// Best-effort: a release that cannot land is logged, because by the time
    /// it runs the caller already holds the turn's real outcome and must not
    /// trade it for a bookkeeping error. An unreachable store here means the
    /// runtime is shutting down, which releases every claim anyway.
    pub async fn release(&self) {
        let Some(ref target) = self.target else {
            return;
        };

        let released: Result<CheckpointReleased, _> = target
            .store
            .ask(ReleaseCheckpoint {
                id: target.id.clone(),
            })
            .await;
        if let Err(error) = released {
            tracing::warn!(
                checkpoint = %target.id,
                %error,
                "could not release the checkpoint claim",
            );
        }
    }

    /// Reads the stored record, if there is one.
    async fn load(&self, target: &Target) -> Result<Option<CheckpointRecord>, ActonAIError> {
        let loaded: CheckpointLoaded = target
            .store
            .ask(LoadCheckpoint {
                id: target.id.clone(),
            })
            .await
            .map_err(|e| {
                ActonAIError::checkpoint(
                    target.id.clone(),
                    format!("the memory store did not answer the checkpoint lookup: {e}"),
                )
            })?;

        loaded.result.map_err(|e| {
            ActonAIError::checkpoint(target.id.clone(), format!("could not load it: {e}"))
        })
    }

    /// Sends one record to the store and waits for it to land.
    async fn write(&self, target: &Target, record: CheckpointRecord) -> Result<(), ActonAIError> {
        let saved: CheckpointSaved =
            target
                .store
                .ask(SaveCheckpoint { record })
                .await
                .map_err(|e| {
                    ActonAIError::checkpoint(
                        target.id.clone(),
                        format!("the memory store did not answer the checkpoint write: {e}"),
                    )
                })?;

        saved.result.map_err(|e| {
            ActonAIError::checkpoint(target.id.clone(), format!("could not save it: {e}"))
        })
    }
}

impl From<CheckpointError> for ActonAIError {
    fn from(error: CheckpointError) -> Self {
        match error.checkpoint_id() {
            Some(id) => Self::checkpoint(id.clone(), error.kind().to_string()),
            None => Self::checkpoint_without_id(error.kind().to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{Message, StopReason, Usage};

    fn tools() -> Vec<String> {
        vec!["read_file".to_string()]
    }

    fn progress() -> RoundProgress {
        RoundProgress {
            rounds_completed: 1,
            messages: vec![Message::user("hello")],
            tool_calls: Vec::new(),
            token_count: 0,
            usage: Usage::default(),
            pending_round: None,
            resume_attempts: 0,
        }
    }

    fn inputs(tools: &[String]) -> TurnInputs<'_> {
        TurnInputs {
            system_prompt: None,
            user_content: "hello",
            tool_names: tools,
            provider: "claude",
            max_tool_rounds: 8,
            structured_schema: None,
        }
    }

    #[tokio::test]
    async fn a_disabled_sink_plans_a_fresh_start() {
        let sink = CheckpointSink::disabled();
        let tools = tools();

        assert_eq!(sink.plan(&inputs(&tools)).await.unwrap(), ResumePlan::Start);
        assert!(!sink.is_enabled());
        assert!(sink.id().is_none());
    }

    #[tokio::test]
    async fn a_disabled_sink_records_nothing() {
        let sink = CheckpointSink::disabled();
        let tools = tools();
        let fingerprint = TurnFingerprint::of(&inputs(&tools));

        sink.record_progress(None, &fingerprint, progress())
            .await
            .unwrap();

        sink.record_completion(
            None,
            &fingerprint,
            progress(),
            FinalAnswer {
                text: "hi".to_string(),
                stop_reason: StopReason::EndTurn,
                structured_output: None,
            },
        )
        .await
        .unwrap();

        sink.mark_failed().await.unwrap();
        sink.claim().await.unwrap();
        sink.release().await;
    }

    #[test]
    fn a_checkpoint_error_carrying_an_id_keeps_it_through_the_conversion() {
        let id = CheckpointId::new();
        let error =
            ActonAIError::from(CheckpointError::rounds_exhausted(4, 4).with_checkpoint(id.clone()));
        assert!(error.to_string().contains(&id.to_string()), "{error}");
    }

    #[test]
    fn a_checkpoint_error_without_an_id_still_converts() {
        let error = ActonAIError::from(CheckpointError::corrupt("messages", "empty"));
        assert!(error.to_string().contains("messages"), "{error}");
    }
}
