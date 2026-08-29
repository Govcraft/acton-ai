//! The resume planner.
//!
//! Every decision about what a saved checkpoint entitles a new run to do lives
//! here, as pure functions over values. Nothing in this module opens a
//! connection, reads a clock, or sends a message — which is what makes the
//! interesting cases (a changed prompt, an exhausted round budget, a record
//! from a future build) testable without a database or a model.

use crate::checkpoint::error::CheckpointError;
use crate::checkpoint::record::{
    CheckpointRecord, CheckpointStatus, PendingCallState, PendingRound, TurnFingerprint,
    TurnInputs, CHECKPOINT_FORMAT_VERSION,
};
use crate::messages::{Message, StopReason, Usage};
use crate::stream::{CollectedResponse, ExecutedToolCall};
use crate::types::{CheckpointId, ConversationId};

/// What a run is allowed to do about a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumePlan {
    /// Nothing usable is saved. Run the turn from the beginning.
    Start,
    /// Pick the turn up where it stopped.
    Resume {
        /// The conversation as the next round should send it.
        messages: Vec<Message>,
        /// Rounds already spent, which count against `max_tool_rounds`.
        rounds_completed: usize,
        /// Tool calls already executed, carried into the final response so the
        /// caller sees the whole turn and not just its tail.
        tool_calls: Vec<ExecutedToolCall>,
        /// Streamed token events already observed.
        token_count: usize,
        /// Provider-reported usage already accrued.
        usage: Usage,
        /// The mid-round state, when the turn died while executing tools.
        /// The loop settles these calls — reusing finished results, running
        /// unstarted calls, surfacing uncertainty for the rest — before it
        /// dispatches anything to the model.
        pending_round: Option<PendingRound>,
        /// Attempts at this turn that already ended in an error, carried so
        /// the resumed loop's own progress writes preserve the count.
        resume_attempts: u32,
    },
    /// The turn already finished. Hand back what it produced without
    /// dispatching anything.
    Replay {
        /// The answer the finished turn produced.
        response: Box<CollectedResponse>,
        /// The validated `structured_output` arguments, on an extracting turn.
        structured_output: Option<serde_json::Value>,
    },
}

/// A turn's whole state after one more round completed.
///
/// Every field is cumulative, not per-round: the prompt loop already
/// accumulates all of this as it goes, and having the checkpoint mirror what
/// the loop holds means neither side has to reconstruct the other's bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundProgress {
    /// Loop iterations spent so far, which is what `max_tool_rounds` bounds.
    pub rounds_completed: usize,
    /// The conversation as the next round would send it, including any tool
    /// results appended to it.
    pub messages: Vec<Message>,
    /// Every tool call executed so far, in order.
    pub tool_calls: Vec<ExecutedToolCall>,
    /// Streamed token events observed across the turn so far.
    pub token_count: usize,
    /// Provider-reported usage across the turn so far.
    pub usage: Usage,
    /// The mid-round state, when the write happens between a round's tool
    /// calls rather than at its boundary. `None` at every boundary.
    pub pending_round: Option<PendingRound>,
    /// Attempts at this turn that already ended in an error. Zero on a fresh
    /// turn; a resume carries the stored count forward so its own writes do
    /// not reset it.
    pub resume_attempts: u32,
}

/// Decides what a run may do with the checkpoint it found.
///
/// The refusals matter more than the acceptances: resuming a checkpoint
/// written for a different prompt would splice two turns together and bill the
/// caller for a conversation nobody asked for, so a mismatch is an error
/// rather than a silent fresh start.
///
/// # Errors
///
/// - [`InputsChanged`](crate::checkpoint::CheckpointErrorKind::InputsChanged)
///   when the turn's fingerprint differs from the stored one.
/// - [`RoundsExhausted`](crate::checkpoint::CheckpointErrorKind::RoundsExhausted)
///   when the checkpoint already spent the round budget it is resuming under.
/// - [`VersionMismatch`](crate::checkpoint::CheckpointErrorKind::VersionMismatch)
///   when the record was written in a format this build does not read.
/// - [`Corrupt`](crate::checkpoint::CheckpointErrorKind::Corrupt) when a
///   resumable record carries no messages to resume from.
/// - [`IncompleteCompletion`](crate::checkpoint::CheckpointErrorKind::IncompleteCompletion)
///   when a finished record is missing the answer it claims to hold.
pub fn plan_resume(
    checkpoint: Option<&CheckpointRecord>,
    inputs: &TurnInputs<'_>,
) -> Result<ResumePlan, CheckpointError> {
    let Some(record) = checkpoint else {
        return Ok(ResumePlan::Start);
    };

    let refuse = |error: CheckpointError| error.with_checkpoint(record.id.clone());

    if record.format_version != CHECKPOINT_FORMAT_VERSION {
        return Err(refuse(CheckpointError::version_mismatch(
            record.format_version,
            CHECKPOINT_FORMAT_VERSION,
        )));
    }

    let fingerprint = TurnFingerprint::of(inputs);
    if fingerprint != record.fingerprint {
        return Err(refuse(CheckpointError::inputs_changed(
            record.fingerprint.as_str(),
            fingerprint.as_str(),
        )));
    }

    plan_from_record(record, inputs.max_tool_rounds)
}

/// Decides what a run may do with a record it holds on its own authority.
///
/// This is [`plan_resume`] without the fingerprint check. It exists for the
/// operator-driven path — [`ActonAI::resume_turn`](crate::facade::ActonAI::resume_turn) hands
/// the loop a record it just listed, and the record's own messages *are* the
/// turn, so there are no separate inputs to fingerprint against. Everything
/// else still applies: the version, the status, the round budget, and the
/// messages are all still validated.
///
/// # Errors
///
/// The same refusals as [`plan_resume`], minus
/// [`InputsChanged`](crate::checkpoint::CheckpointErrorKind::InputsChanged).
pub fn plan_from_record(
    record: &CheckpointRecord,
    max_tool_rounds: usize,
) -> Result<ResumePlan, CheckpointError> {
    let refuse = |error: CheckpointError| error.with_checkpoint(record.id.clone());

    if record.format_version != CHECKPOINT_FORMAT_VERSION {
        return Err(refuse(CheckpointError::version_mismatch(
            record.format_version,
            CHECKPOINT_FORMAT_VERSION,
        )));
    }

    if record.status == CheckpointStatus::Completed {
        return replay(record).map_err(refuse);
    }

    if record.status == CheckpointStatus::Abandoned {
        return Err(refuse(CheckpointError::abandoned()));
    }

    if record.rounds_completed >= max_tool_rounds {
        return Err(refuse(CheckpointError::rounds_exhausted(
            record.rounds_completed,
            max_tool_rounds,
        )));
    }

    if record.messages.is_empty() {
        return Err(refuse(CheckpointError::corrupt(
            "messages",
            "a resumable checkpoint holds no conversation to resume from",
        )));
    }

    Ok(ResumePlan::Resume {
        messages: record.messages.clone(),
        rounds_completed: record.rounds_completed,
        tool_calls: record.tool_calls.clone(),
        token_count: record.token_count,
        usage: record.usage,
        pending_round: record.pending_round.clone(),
        resume_attempts: record.resume_attempts,
    })
}

/// Rebuilds the finished answer a `Completed` record holds.
fn replay(record: &CheckpointRecord) -> Result<ResumePlan, CheckpointError> {
    let Some(ref text) = record.final_text else {
        return Err(CheckpointError::incomplete_completion("final_text"));
    };
    let Some(stop_reason) = record.stop_reason else {
        return Err(CheckpointError::incomplete_completion("stop_reason"));
    };

    let response = CollectedResponse::with_tool_calls(
        text.clone(),
        stop_reason,
        record.token_count,
        record.tool_calls.clone(),
    )
    .with_usage(record.usage);

    Ok(ResumePlan::Replay {
        response: Box::new(response),
        structured_output: record.structured_output.clone(),
    })
}

/// Builds the in-progress record that describes a turn's state right now.
///
/// A resumed turn keeps counting from where the previous attempt stopped
/// because `progress.rounds_completed` is the loop's own iteration counter,
/// which a resume seeds from the checkpoint it picked up.
#[must_use]
pub fn advance(
    id: CheckpointId,
    conversation_id: Option<ConversationId>,
    fingerprint: TurnFingerprint,
    progress: RoundProgress,
) -> CheckpointRecord {
    CheckpointRecord {
        id,
        conversation_id,
        fingerprint,
        format_version: CHECKPOINT_FORMAT_VERSION,
        status: CheckpointStatus::InProgress,
        rounds_completed: progress.rounds_completed,
        token_count: progress.token_count,
        usage: progress.usage,
        messages: progress.messages,
        tool_calls: progress.tool_calls,
        final_text: None,
        stop_reason: None,
        structured_output: None,
        pending_round: progress.pending_round,
        resume_attempts: progress.resume_attempts,
    }
}

/// Marks a record finished and records the answer to replay.
#[must_use]
pub fn complete(mut record: CheckpointRecord, answer: FinalAnswer) -> CheckpointRecord {
    record.status = CheckpointStatus::Completed;
    record.final_text = Some(answer.text);
    record.stop_reason = Some(answer.stop_reason);
    record.structured_output = answer.structured_output;
    // A finished turn has no calls in flight; a stale pending round on a
    // completed record would be a contradiction nobody should have to read.
    record.pending_round = None;
    record
}

/// What a finished turn produced, over and above the progress already
/// recorded.
///
/// Token counts and usage are not here: they belong to [`RoundProgress`],
/// which the completion is layered on top of, and duplicating them would give
/// two places to disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalAnswer {
    /// The answer text.
    pub text: String,
    /// Why the turn stopped.
    pub stop_reason: StopReason,
    /// The validated `structured_output` arguments, on an extracting turn.
    pub structured_output: Option<serde_json::Value>,
}

/// Marks a record as belonging to a turn that ended in an error.
///
/// The messages and rounds are left exactly as they were: a failed turn is
/// resumable, and where it got to is the point of the record. What changes is
/// only what an operator sees — [`ListCheckpoints`](crate::memory::ListCheckpoints)
/// can pick failed turns out, which an in-progress marking would hide among
/// the turns still running. The failure also counts: `resume_attempts` goes
/// up by one, which is what lets an unattended resume loop stop re-paying for
/// a turn that fails the same way on every restart.
///
/// A terminal record — `Completed` or `Abandoned` — comes back unchanged.
/// Those statuses are outcomes, not progress: a finished answer must stay
/// replayable and an operator's abandonment must stay closed, no matter what
/// error a later attempt against the same ID ran into on its way out.
#[must_use]
pub fn fail(mut record: CheckpointRecord) -> CheckpointRecord {
    if matches!(
        record.status,
        CheckpointStatus::Completed | CheckpointStatus::Abandoned
    ) {
        return record;
    }
    record.status = CheckpointStatus::Failed;
    record.resume_attempts = record.resume_attempts.saturating_add(1);
    record
}

/// Marks a record as belonging to a turn an operator policy declined to
/// resume.
///
/// Everything the turn had done is kept: the record is the evidence of the
/// outcome, and evidence with the progress stripped out would say only that
/// something was abandoned, not what.
#[must_use]
pub fn abandon(mut record: CheckpointRecord) -> CheckpointRecord {
    record.status = CheckpointStatus::Abandoned;
    record
}

/// What a resumed loop does about one call from an interrupted round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingCallAction {
    /// The call finished before the interruption. Feed its recorded result
    /// back to the model; nothing runs again.
    UseStored {
        /// The tool result exactly as the first execution produced it.
        result: String,
    },
    /// The call is safe to run: it never started, or it is idempotent and
    /// running it twice observes the same world.
    Execute,
    /// The call may have run, and the tool is not idempotent, so it is NOT
    /// re-run. The uncertainty itself becomes the tool result.
    Uncertain {
        /// The tool result telling the model what is and is not known.
        feedback: String,
    },
}

/// Decides what to do about one call from an interrupted round.
///
/// Pure, and deliberately conservative: the only state that ever re-runs a
/// non-idempotent tool is `Pending`, which means execution provably never
/// began. A call that had `Started` when the process died may or may not have
/// had its effect, and `idempotent` is the tool author's declaration of
/// whether running it again on a "maybe" is safe.
#[must_use]
pub fn resolve_pending_call(
    state: &PendingCallState,
    tool_name: &str,
    idempotent: bool,
) -> PendingCallAction {
    match state {
        PendingCallState::Completed { result } => PendingCallAction::UseStored {
            result: result.clone(),
        },
        PendingCallState::Pending => PendingCallAction::Execute,
        PendingCallState::Started if idempotent => PendingCallAction::Execute,
        PendingCallState::Started => PendingCallAction::Uncertain {
            feedback: uncertain_feedback(tool_name),
        },
    }
}

/// The tool result handed to the model for a call whose first execution is
/// uncertain.
///
/// This is the model's only account of what happened, so it says all three
/// things the model needs: the call may have run, it was NOT re-run, and
/// verifying the effect is the way forward.
#[must_use]
pub fn uncertain_feedback(tool_name: &str) -> String {
    format!(
        "A previous attempt at this turn was interrupted while `{tool_name}` was executing. \
         Whether it completed is unknown, and `{tool_name}` is not idempotent, so it was NOT \
         re-run. If you need its effect, verify the current state first (for example with a \
         read-only tool) before deciding whether to repeat it."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::error::CheckpointErrorKind;
    use crate::checkpoint::record::PendingToolCall;

    const TOOLS: &[&str] = &["read_file"];

    fn tool_names() -> Vec<String> {
        TOOLS.iter().map(|s| (*s).to_string()).collect()
    }

    fn inputs<'a>(tools: &'a [String]) -> TurnInputs<'a> {
        TurnInputs {
            system_prompt: Some("be brief"),
            user_content: "summarize a.txt",
            tool_names: tools,
            provider: "claude",
            max_tool_rounds: 8,
            structured_schema: None,
        }
    }

    fn in_progress(tools: &[String]) -> CheckpointRecord {
        CheckpointRecord {
            id: CheckpointId::new(),
            conversation_id: None,
            fingerprint: TurnFingerprint::of(&inputs(tools)),
            format_version: CHECKPOINT_FORMAT_VERSION,
            status: CheckpointStatus::InProgress,
            rounds_completed: 2,
            token_count: 30,
            usage: Usage {
                input_tokens: 100,
                output_tokens: 20,
                ..Usage::default()
            },
            messages: vec![Message::user("summarize a.txt")],
            tool_calls: vec![ExecutedToolCall::success(
                "call_1",
                "read_file",
                serde_json::json!({}),
                serde_json::json!("contents"),
            )],
            final_text: None,
            stop_reason: None,
            structured_output: None,
            pending_round: None,
            resume_attempts: 0,
        }
    }

    /// A plain (non-extracting) finished answer.
    fn answer(text: &str, stop_reason: StopReason) -> FinalAnswer {
        FinalAnswer {
            text: text.to_string(),
            stop_reason,
            structured_output: None,
        }
    }

    #[test]
    fn no_checkpoint_starts_fresh() {
        let tools = tool_names();
        assert_eq!(
            plan_resume(None, &inputs(&tools)).unwrap(),
            ResumePlan::Start
        );
    }

    #[test]
    fn matching_in_progress_checkpoint_resumes_from_its_messages() {
        let tools = tool_names();
        let record = in_progress(&tools);

        let plan = plan_resume(Some(&record), &inputs(&tools)).unwrap();

        let ResumePlan::Resume {
            messages,
            rounds_completed,
            tool_calls,
            token_count,
            usage,
            pending_round,
            ..
        } = plan
        else {
            panic!("expected a resume, got {plan:?}");
        };
        assert_eq!(messages, record.messages);
        assert_eq!(rounds_completed, 2);
        assert_eq!(tool_calls, record.tool_calls);
        assert_eq!(token_count, 30);
        assert_eq!(usage, record.usage);
        assert_eq!(pending_round, None);
    }

    #[test]
    fn failed_checkpoint_is_resumable() {
        let tools = tool_names();
        let record = fail(in_progress(&tools));

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&tools)).unwrap(),
            ResumePlan::Resume { .. }
        ));
    }

    #[test]
    fn completed_checkpoint_replays_without_a_round() {
        let tools = tool_names();
        let mut base = in_progress(&tools);
        base.token_count = 42;
        base.usage = Usage {
            input_tokens: 100,
            output_tokens: 25,
            ..Usage::default()
        };
        let record = complete(base, answer("the file says hello", StopReason::EndTurn));

        let plan = plan_resume(Some(&record), &inputs(&tools)).unwrap();
        let ResumePlan::Replay { response, .. } = plan else {
            panic!("expected a replay, got {plan:?}");
        };
        assert_eq!(response.text, "the file says hello");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.token_count, 42);
        assert_eq!(response.usage.output_tokens, 25);
        assert_eq!(response.tool_calls, record.tool_calls);
    }

    #[test]
    fn a_completed_checkpoint_replays_even_with_its_rounds_spent() {
        // The round budget bounds work still to do. A finished turn has none,
        // so exhaustion must not stand between the caller and their answer.
        let tools = tool_names();
        let mut record = complete(in_progress(&tools), answer("done", StopReason::EndTurn));
        record.rounds_completed = 8;

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&tools)).unwrap(),
            ResumePlan::Replay { .. }
        ));
    }

    #[test]
    fn completed_checkpoint_missing_its_text_is_refused() {
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.status = CheckpointStatus::Completed;
        record.stop_reason = Some(StopReason::EndTurn);

        let error = plan_resume(Some(&record), &inputs(&tools)).unwrap_err();
        assert!(matches!(
            error.kind(),
            CheckpointErrorKind::IncompleteCompletion {
                missing: "final_text"
            }
        ));
        assert_eq!(error.checkpoint_id(), Some(&record.id));
    }

    #[test]
    fn completed_checkpoint_missing_its_stop_reason_is_refused() {
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.status = CheckpointStatus::Completed;
        record.final_text = Some("done".to_string());

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&tools))
                .unwrap_err()
                .kind(),
            CheckpointErrorKind::IncompleteCompletion {
                missing: "stop_reason"
            }
        ));
    }

    #[test]
    fn changed_user_content_refuses_the_resume() {
        let tools = tool_names();
        let record = in_progress(&tools);
        let mut changed = inputs(&tools);
        changed.user_content = "summarize b.txt";

        assert!(matches!(
            plan_resume(Some(&record), &changed).unwrap_err().kind(),
            CheckpointErrorKind::InputsChanged { .. }
        ));
    }

    #[test]
    fn changed_tool_set_refuses_the_resume() {
        let tools = tool_names();
        let record = in_progress(&tools);
        let more = vec!["read_file".to_string(), "write_file".to_string()];

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&more))
                .unwrap_err()
                .kind(),
            CheckpointErrorKind::InputsChanged { .. }
        ));
    }

    #[test]
    fn a_reordered_tool_set_still_resumes() {
        let tools = vec!["alpha".to_string(), "beta".to_string()];
        let record = in_progress(&tools);
        let reordered = vec!["beta".to_string(), "alpha".to_string()];

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&reordered)).unwrap(),
            ResumePlan::Resume { .. }
        ));
    }

    #[test]
    fn exhausted_rounds_refuse_the_resume() {
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.rounds_completed = 8;

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&tools))
                .unwrap_err()
                .kind(),
            CheckpointErrorKind::RoundsExhausted {
                rounds_completed: 8,
                max_tool_rounds: 8,
            }
        ));
    }

    #[test]
    fn a_changed_round_ceiling_refuses_before_it_can_look_exhausted() {
        // `max_tool_rounds` is part of the fingerprint, so raising it to make
        // room is refused as an input change rather than silently honoured.
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.rounds_completed = 8;
        let mut roomier = inputs(&tools);
        roomier.max_tool_rounds = 16;

        assert!(matches!(
            plan_resume(Some(&record), &roomier).unwrap_err().kind(),
            CheckpointErrorKind::InputsChanged { .. }
        ));
    }

    #[test]
    fn unknown_format_version_refuses_the_resume() {
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.format_version = CHECKPOINT_FORMAT_VERSION + 1;

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&tools))
                .unwrap_err()
                .kind(),
            CheckpointErrorKind::VersionMismatch { .. }
        ));
    }

    #[test]
    fn a_version_mismatch_is_reported_before_a_fingerprint_mismatch() {
        // A record this build cannot read must not be described by its
        // contents, which this build has no basis to interpret.
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.format_version = CHECKPOINT_FORMAT_VERSION + 1;
        record.fingerprint = TurnFingerprint::from_hex("something else");

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&tools))
                .unwrap_err()
                .kind(),
            CheckpointErrorKind::VersionMismatch { .. }
        ));
    }

    #[test]
    fn empty_messages_are_reported_as_corrupt() {
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.messages.clear();

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&tools))
                .unwrap_err()
                .kind(),
            CheckpointErrorKind::Corrupt {
                field: "messages",
                ..
            }
        ));
    }

    #[test]
    fn advance_records_the_progress_it_was_given() {
        let tools = tool_names();
        let id = CheckpointId::new();
        let record = advance(
            id.clone(),
            None,
            TurnFingerprint::of(&inputs(&tools)),
            RoundProgress {
                rounds_completed: 3,
                messages: vec![Message::user("summarize a.txt"), Message::assistant("ok")],
                tool_calls: vec![ExecutedToolCall::error(
                    "call_2",
                    "read_file",
                    serde_json::json!({}),
                    "no such file",
                )],
                token_count: 44,
                usage: Usage {
                    input_tokens: 200,
                    output_tokens: 30,
                    ..Usage::default()
                },
                pending_round: None,
                resume_attempts: 0,
            },
        );

        assert_eq!(record.id, id);
        assert_eq!(record.rounds_completed, 3);
        assert_eq!(record.status, CheckpointStatus::InProgress);
        assert_eq!(record.messages.len(), 2);
        assert_eq!(record.tool_calls.len(), 1);
        assert_eq!(record.token_count, 44);
        assert_eq!(record.usage.input_tokens, 200);
        assert!(record.final_text.is_none());
    }

    #[test]
    fn a_record_advanced_then_resumed_reports_the_rounds_it_had_spent() {
        // The round count has to survive the write/read cycle intact, because
        // it is what a resumed loop seeds its iteration counter from.
        let tools = tool_names();
        let record = advance(
            CheckpointId::new(),
            None,
            TurnFingerprint::of(&inputs(&tools)),
            RoundProgress {
                rounds_completed: 5,
                messages: vec![Message::user("summarize a.txt")],
                tool_calls: Vec::new(),
                token_count: 0,
                usage: Usage::default(),
                pending_round: None,
                resume_attempts: 0,
            },
        );

        let plan = plan_resume(Some(&record), &inputs(&tools)).unwrap();
        let ResumePlan::Resume {
            rounds_completed, ..
        } = plan
        else {
            panic!("expected a resume, got {plan:?}");
        };
        assert_eq!(rounds_completed, 5);
    }

    #[test]
    fn advance_carries_the_conversation_it_was_given() {
        let tools = tool_names();
        let conversation = ConversationId::new();
        let record = advance(
            CheckpointId::new(),
            Some(conversation.clone()),
            TurnFingerprint::of(&inputs(&tools)),
            RoundProgress {
                rounds_completed: 1,
                messages: vec![Message::user("summarize a.txt")],
                tool_calls: Vec::new(),
                token_count: 0,
                usage: Usage::default(),
                pending_round: None,
                resume_attempts: 0,
            },
        );

        assert_eq!(record.conversation_id, Some(conversation));
    }

    #[test]
    fn complete_records_the_text_and_stop_reason() {
        let tools = tool_names();
        let record = complete(in_progress(&tools), answer("answer", StopReason::MaxTokens));

        assert_eq!(record.status, CheckpointStatus::Completed);
        assert_eq!(record.final_text.as_deref(), Some("answer"));
        assert_eq!(record.stop_reason, Some(StopReason::MaxTokens));
        // The progress the record already carried is left alone.
        assert_eq!(record.token_count, 30);
        assert_eq!(record.usage.output_tokens, 20);
    }

    #[test]
    fn a_completed_extraction_replays_its_recorded_value() {
        let tools = tool_names();
        let record = complete(
            in_progress(&tools),
            FinalAnswer {
                text: String::new(),
                stop_reason: StopReason::ToolUse,
                structured_output: Some(serde_json::json!({ "vendor": "Acme" })),
            },
        );

        let plan = plan_resume(Some(&record), &inputs(&tools)).unwrap();
        let ResumePlan::Replay {
            structured_output, ..
        } = plan
        else {
            panic!("expected a replay, got {plan:?}");
        };
        assert_eq!(
            structured_output,
            Some(serde_json::json!({ "vendor": "Acme" }))
        );
    }

    #[test]
    fn an_extracting_turn_does_not_resume_a_plain_checkpoint() {
        // The two turns share a prompt and a tool set but not a schema, so the
        // fingerprint separates them and the plain record is refused.
        let tools = tool_names();
        let record = in_progress(&tools);
        let mut extracting = inputs(&tools);
        extracting.structured_schema = Some(r#"{"type":"object"}"#);

        assert!(matches!(
            plan_resume(Some(&record), &extracting).unwrap_err().kind(),
            CheckpointErrorKind::InputsChanged { .. }
        ));
    }

    #[test]
    fn fail_keeps_the_progress_it_was_given() {
        let tools = tool_names();
        let record = fail(in_progress(&tools));

        assert_eq!(record.status, CheckpointStatus::Failed);
        assert_eq!(record.rounds_completed, 2);
        assert_eq!(record.messages.len(), 1);
    }

    #[test]
    fn fail_counts_the_attempt() {
        let tools = tool_names();
        let record = fail(in_progress(&tools));
        assert_eq!(record.resume_attempts, 1);

        // A failed resume of the failed record counts again.
        let record = fail(record);
        assert_eq!(record.resume_attempts, 2);
        assert_eq!(record.status, CheckpointStatus::Failed);
    }

    #[test]
    fn fail_leaves_a_completed_record_untouched() {
        // A pre-flight refusal — changed inputs, an exhausted budget — errors
        // out of the loop before anything runs, and the loop marks the record
        // failed on the way out. That marking must not downgrade a finished
        // answer: the record stays Completed, still replayable, and the
        // attempt is not counted against it.
        let tools = tool_names();
        let completed = complete(in_progress(&tools), answer("answer", StopReason::EndTurn));

        let record = fail(completed.clone());

        assert_eq!(record, completed);
        assert_eq!(record.status, CheckpointStatus::Completed);
        assert_eq!(record.resume_attempts, 0);
    }

    #[test]
    fn fail_leaves_an_abandoned_record_untouched() {
        // Same for an operator's abandonment: resuming an abandoned record is
        // refused, and that refusal must not quietly reopen the record by
        // flipping it to Failed — Failed is resumable, Abandoned is closed.
        let tools = tool_names();
        let abandoned = abandon(in_progress(&tools));

        let record = fail(abandoned.clone());

        assert_eq!(record, abandoned);
        assert_eq!(record.status, CheckpointStatus::Abandoned);
    }

    #[test]
    fn a_completed_record_planned_twice_replays_the_same_answer() {
        let tools = tool_names();
        let record = complete(in_progress(&tools), answer("answer", StopReason::EndTurn));

        let first = plan_resume(Some(&record), &inputs(&tools)).unwrap();
        let second = plan_resume(Some(&record), &inputs(&tools)).unwrap();

        // Everything the record holds replays identically. The one exception
        // is `turn_id`: the record does not store one — the original turn's
        // lifecycle concluded when it completed — so each replayed response
        // mints a fresh id, which by design matches no lifecycle events.
        let (
            ResumePlan::Replay {
                response: mut a,
                structured_output: sa,
            },
            ResumePlan::Replay {
                response: b,
                structured_output: sb,
            },
        ) = (first, second)
        else {
            panic!("a completed record must plan a replay");
        };
        a.turn_id = b.turn_id.clone();
        assert_eq!(a, b);
        assert_eq!(sa, sb);
    }

    #[test]
    fn an_abandoned_checkpoint_refuses_the_resume() {
        let tools = tool_names();
        let record = abandon(in_progress(&tools));

        assert!(matches!(
            plan_resume(Some(&record), &inputs(&tools))
                .unwrap_err()
                .kind(),
            CheckpointErrorKind::Abandoned
        ));
    }

    #[test]
    fn abandon_keeps_the_progress_it_was_given() {
        let tools = tool_names();
        let record = abandon(in_progress(&tools));

        assert_eq!(record.status, CheckpointStatus::Abandoned);
        assert!(!record.status.is_resumable());
        assert_eq!(record.rounds_completed, 2);
        assert_eq!(record.messages.len(), 1);
        assert_eq!(record.tool_calls.len(), 1);
    }

    #[test]
    fn plan_from_record_resumes_without_a_fingerprint_to_match() {
        // The operator-driven path holds only the record, so the plan must
        // not depend on reconstructing inputs that were never stored.
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.fingerprint = TurnFingerprint::from_hex("not any turn's fingerprint");

        assert!(matches!(
            plan_from_record(&record, 8).unwrap(),
            ResumePlan::Resume { .. }
        ));
    }

    #[test]
    fn plan_from_record_still_refuses_an_abandoned_record() {
        let tools = tool_names();
        let record = abandon(in_progress(&tools));

        assert!(matches!(
            plan_from_record(&record, 8).unwrap_err().kind(),
            CheckpointErrorKind::Abandoned
        ));
    }

    #[test]
    fn plan_from_record_still_enforces_the_round_budget() {
        let tools = tool_names();
        let record = in_progress(&tools);

        assert!(matches!(
            plan_from_record(&record, 2).unwrap_err().kind(),
            CheckpointErrorKind::RoundsExhausted { .. }
        ));
    }

    #[test]
    fn a_resume_carries_the_pending_round_it_found() {
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.pending_round = Some(PendingRound {
            assistant_text: "reading the file".to_string(),
            calls: vec![PendingToolCall {
                call: crate::messages::ToolCall {
                    id: "call_9".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({ "path": "a.txt" }),
                },
                state: PendingCallState::Started,
            }],
        });

        let plan = plan_resume(Some(&record), &inputs(&tools)).unwrap();
        let ResumePlan::Resume { pending_round, .. } = plan else {
            panic!("expected a resume, got {plan:?}");
        };
        assert_eq!(pending_round, record.pending_round);
    }

    #[test]
    fn a_completed_call_is_never_re_run() {
        let action = resolve_pending_call(
            &PendingCallState::Completed {
                result: "42".to_string(),
            },
            "bash",
            false,
        );
        assert_eq!(
            action,
            PendingCallAction::UseStored {
                result: "42".to_string()
            }
        );
    }

    #[test]
    fn an_unstarted_call_runs_whatever_the_tool_declares() {
        assert_eq!(
            resolve_pending_call(&PendingCallState::Pending, "bash", false),
            PendingCallAction::Execute
        );
        assert_eq!(
            resolve_pending_call(&PendingCallState::Pending, "read_file", true),
            PendingCallAction::Execute
        );
    }

    #[test]
    fn a_started_idempotent_call_re_runs() {
        assert_eq!(
            resolve_pending_call(&PendingCallState::Started, "read_file", true),
            PendingCallAction::Execute
        );
    }

    #[test]
    fn a_started_non_idempotent_call_surfaces_its_uncertainty() {
        let action = resolve_pending_call(&PendingCallState::Started, "bash", false);
        let PendingCallAction::Uncertain { feedback } = action else {
            panic!("expected uncertainty, got {action:?}");
        };
        assert!(feedback.contains("NOT re-run"), "{feedback}");
        assert!(feedback.contains("bash"), "{feedback}");
    }

    #[test]
    fn complete_clears_any_pending_round() {
        let tools = tool_names();
        let mut record = in_progress(&tools);
        record.pending_round = Some(PendingRound {
            assistant_text: String::new(),
            calls: Vec::new(),
        });

        let record = complete(record, answer("done", StopReason::EndTurn));
        assert_eq!(record.pending_round, None);
    }
}
