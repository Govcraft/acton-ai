//! The checkpoint record, its fingerprint, and the pure codec that moves it
//! in and out of a database row.
//!
//! Nothing in this module touches a connection. `CheckpointRecord` goes to
//! [`CheckpointColumns`] and back through two pure functions, which is what
//! keeps the SQL in `memory::persistence` free of decoding logic and lets the
//! decoding be tested without a database.

use crate::checkpoint::error::CheckpointError;
use crate::messages::{Message, StopReason, ToolCall, Usage};
use crate::stream::ExecutedToolCall;
use crate::types::{CheckpointId, ConversationId};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// The checkpoint format this build reads and writes.
///
/// Bumped whenever a stored record's meaning changes in a way an older build
/// would misread. A record carrying any other version is refused rather than
/// guessed at.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

/// How far a checkpointed turn got.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointStatus {
    /// The turn is mid-flight: rounds have been spent and more are expected.
    InProgress,
    /// The turn produced its answer. Resuming replays that answer.
    Completed,
    /// The turn ended in an error. Resumable — the rounds already spent still
    /// count, and the recorded messages are still the place to pick up from.
    Failed,
    /// An operator policy declined to resume the interrupted turn. Terminal:
    /// the record is kept as evidence of the outcome, never picked up again.
    Abandoned,
}

impl CheckpointStatus {
    /// The stable text written to the `status` column.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Abandoned => "abandoned",
        }
    }

    /// Whether a turn in this state still has work left to do.
    #[must_use]
    pub const fn is_resumable(&self) -> bool {
        matches!(self, Self::InProgress | Self::Failed)
    }
}

impl fmt::Display for CheckpointStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for CheckpointStatus {
    type Err = CheckpointError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "in_progress" => Ok(Self::InProgress),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "abandoned" => Ok(Self::Abandoned),
            other => Err(CheckpointError::corrupt(
                "status",
                format!(
                    "expected one of `in_progress`, `completed`, `failed`, `abandoned`; \
                     found `{other}`"
                ),
            )),
        }
    }
}

/// Everything about a turn that decides whether a checkpoint describes *it*.
///
/// Borrowed rather than owned: this is built at the top of the prompt loop out
/// of values the loop already holds, and only ever read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnInputs<'a> {
    /// The system prompt, if the turn has one.
    pub system_prompt: Option<&'a str>,
    /// The user content that opened the turn.
    pub user_content: &'a str,
    /// The names of every tool offered to the model. Order does not matter;
    /// the fingerprint sorts them.
    pub tool_names: &'a [String],
    /// The configured provider name the turn bills to.
    pub provider: &'a str,
    /// The round ceiling the turn runs under.
    pub max_tool_rounds: usize,
    /// The JSON Schema of the extraction target, when the turn is extracting.
    ///
    /// A turn that must record a typed answer is a different turn from one
    /// free to answer in prose, and one extracting into a different shape is
    /// different again. Without this, a checkpoint written by `collect()`
    /// would look resumable to an `extract()` of the same prompt, and the
    /// replayed answer would carry no recorded value at all.
    pub structured_schema: Option<&'a str>,
}

/// A content hash over the inputs that define a turn.
///
/// Two turns with the same fingerprint are the same question asked of the same
/// model with the same tools. That is exactly the condition under which
/// resuming a checkpoint is sound, and the planner refuses to resume without
/// it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TurnFingerprint(String);

impl TurnFingerprint {
    /// Computes the fingerprint of a turn's inputs.
    ///
    /// Pure: the same inputs always hash to the same value, in any process, on
    /// any host. Tool names are sorted first, so offering the same tools in a
    /// different order is the same turn.
    #[must_use]
    pub fn of(inputs: &TurnInputs<'_>) -> Self {
        let mut hasher = blake3::Hasher::new();

        // Every field is length-prefixed, so no concatenation of two fields
        // can collide with a different split of the same bytes. Counts go in
        // as fixed-width `u64` rather than `usize`, so the same turn
        // fingerprints identically on a 32-bit and a 64-bit host.
        update_field(
            &mut hasher,
            inputs.system_prompt.unwrap_or_default().as_bytes(),
        );
        update_field(&mut hasher, inputs.user_content.as_bytes());
        update_field(&mut hasher, inputs.provider.as_bytes());
        update_field(&mut hasher, &widen(inputs.max_tool_rounds).to_le_bytes());
        update_field(
            &mut hasher,
            inputs.structured_schema.unwrap_or_default().as_bytes(),
        );

        let mut names: Vec<&str> = inputs.tool_names.iter().map(String::as_str).collect();
        names.sort_unstable();
        update_field(&mut hasher, &widen(names.len()).to_le_bytes());
        for name in names {
            update_field(&mut hasher, name.as_bytes());
        }

        Self(hasher.finalize().to_hex().to_string())
    }

    /// Adopts an already-computed fingerprint, such as one read back from the
    /// database.
    #[must_use]
    pub fn from_hex(hex: impl Into<String>) -> Self {
        Self(hex.into())
    }

    /// The fingerprint as stored.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TurnFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Feeds one length-prefixed field into a fingerprint hasher.
fn update_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&widen(bytes.len()).to_le_bytes());
    hasher.update(bytes);
}

/// Widens a count to the fixed width the fingerprint hashes.
///
/// Saturating rather than wrapping: a saturated count is already far past
/// anything a real turn produces, and a wrap would let two different turns
/// fingerprint alike.
fn widen(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// How far one tool call inside an interrupted round got.
///
/// The distinction between `Pending` and `Started` is the whole reason this
/// type exists: a call that never began is always safe to run, while a call
/// that began and never finished may or may not have had its effect — and
/// only an idempotent tool is safe to run again on a "maybe".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum PendingCallState {
    /// The call was requested by the model but execution never began.
    Pending,
    /// Execution began; whether it finished is unknown to a later run.
    Started,
    /// The call finished, and this is the tool result the model would have
    /// been given.
    Completed {
        /// The tool result exactly as it would go back to the model.
        result: String,
    },
}

/// One tool call inside an interrupted round, with how far it got.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingToolCall {
    /// The call as the model requested it.
    pub call: ToolCall,
    /// How far execution got before the record was written.
    pub state: PendingCallState,
}

/// The mid-round state of a turn interrupted while executing tools.
///
/// Written before the first tool of a round runs and rewritten as each call
/// starts and finishes, all into the same row as the rest of the record — one
/// upsert per change, so the pending state and the progress it belongs to can
/// never disagree. `None` on a record written at a round boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingRound {
    /// The assistant text that accompanied the round's tool calls.
    pub assistant_text: String,
    /// Every call the round asked for, in order, each with how far it got.
    pub calls: Vec<PendingToolCall>,
}

/// One turn's saved progress.
///
/// Written after every round that completes, so a process that dies mid-turn
/// leaves behind the messages, the rounds already spent, and the tools already
/// executed. See [`plan_resume`](crate::checkpoint::plan_resume) for what a
/// later run is allowed to do with it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointRecord {
    /// The caller-chosen key this turn resumes under.
    pub id: CheckpointId,
    /// The conversation the turn belongs to, when it belongs to one.
    pub conversation_id: Option<ConversationId>,
    /// Fingerprint of the turn's inputs at the time it was written.
    pub fingerprint: TurnFingerprint,
    /// The checkpoint format the record was written in.
    pub format_version: u32,
    /// How far the turn got.
    pub status: CheckpointStatus,
    /// Loop iterations already spent, which is what `max_tool_rounds` bounds.
    pub rounds_completed: usize,
    /// Streamed token events observed so far. Transport-level, not billed.
    pub token_count: usize,
    /// Provider-reported usage summed over the rounds already spent.
    pub usage: Usage,
    /// The conversation as the next round would send it.
    pub messages: Vec<Message>,
    /// Every tool call executed so far, in order.
    pub tool_calls: Vec<ExecutedToolCall>,
    /// The finished answer, on a `Completed` record.
    pub final_text: Option<String>,
    /// Why the finished turn stopped, on a `Completed` record.
    pub stop_reason: Option<StopReason>,
    /// The validated `structured_output` arguments, on a `Completed` record of
    /// an extracting turn. `None` on every other record.
    pub structured_output: Option<serde_json::Value>,
    /// The mid-round state, when the turn was interrupted while executing
    /// tools. `None` on a record written at a round boundary.
    pub pending_round: Option<PendingRound>,
    /// How many attempts at this turn have ended in an error.
    ///
    /// Incremented by [`fail`](crate::checkpoint::fail) each time the record
    /// is marked failed, and carried forward by every progress write, so a
    /// turn that fails for the same reason on every restart accumulates a
    /// count an unattended resume loop can bound itself on — see
    /// [`CheckpointConfig::max_resume_attempts`](crate::checkpoint::CheckpointConfig::max_resume_attempts).
    pub resume_attempts: u32,
}

impl CheckpointRecord {
    /// Creates the first record of a turn: nothing spent, nothing executed.
    #[must_use]
    pub fn opening(
        id: CheckpointId,
        conversation_id: Option<ConversationId>,
        fingerprint: TurnFingerprint,
        messages: Vec<Message>,
    ) -> Self {
        Self {
            id,
            conversation_id,
            fingerprint,
            format_version: CHECKPOINT_FORMAT_VERSION,
            status: CheckpointStatus::InProgress,
            rounds_completed: 0,
            token_count: 0,
            usage: Usage::default(),
            messages,
            tool_calls: Vec::new(),
            final_text: None,
            stop_reason: None,
            structured_output: None,
            pending_round: None,
            resume_attempts: 0,
        }
    }
}

/// A checkpoint as it sits in a database row.
///
/// The JSON-bearing columns are `String`s here and structured values on
/// [`CheckpointRecord`]; [`encode_record`] and [`decode_row`] are the only
/// crossing points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointColumns {
    /// `id`
    pub id: String,
    /// `conversation_id`
    pub conversation_id: Option<String>,
    /// `fingerprint`
    pub fingerprint: String,
    /// `format_version`
    pub format_version: u32,
    /// `status`
    pub status: String,
    /// `rounds_completed`
    pub rounds_completed: u32,
    /// `token_count`
    pub token_count: u32,
    /// `usage`, JSON
    pub usage: String,
    /// `messages`, JSON array
    pub messages: String,
    /// `tool_calls`, JSON array
    pub tool_calls: String,
    /// `final_text`
    pub final_text: Option<String>,
    /// `stop_reason`
    pub stop_reason: Option<String>,
    /// `structured_output`, JSON
    pub structured_output: Option<String>,
    /// `pending_round`, JSON
    pub pending_round: Option<String>,
    /// `resume_attempts`
    pub resume_attempts: u32,
}

/// Renders a record into the columns that store it.
///
/// Pure. Returns [`CheckpointError`] only when a payload cannot be serialized,
/// which in practice means a tool result holding a non-finite float.
///
/// # Errors
///
/// Returns [`CheckpointErrorKind::Corrupt`](crate::checkpoint::CheckpointErrorKind::Corrupt)
/// naming the payload that would not serialize.
pub fn encode_record(record: &CheckpointRecord) -> Result<CheckpointColumns, CheckpointError> {
    Ok(CheckpointColumns {
        id: record.id.to_string(),
        conversation_id: record.conversation_id.as_ref().map(ToString::to_string),
        fingerprint: record.fingerprint.as_str().to_string(),
        format_version: record.format_version,
        status: record.status.as_str().to_string(),
        rounds_completed: clamp_to_u32(record.rounds_completed),
        token_count: clamp_to_u32(record.token_count),
        usage: encode_payload("usage", &record.id, &record.usage)?,
        messages: encode_payload("messages", &record.id, &record.messages)?,
        tool_calls: encode_payload("tool_calls", &record.id, &record.tool_calls)?,
        final_text: record.final_text.clone(),
        structured_output: record
            .structured_output
            .as_ref()
            .map(|value| encode_payload("structured_output", &record.id, value))
            .transpose()?,
        pending_round: record
            .pending_round
            .as_ref()
            .map(|value| encode_payload("pending_round", &record.id, value))
            .transpose()?,
        stop_reason: record.stop_reason.map(|reason| {
            match reason {
                StopReason::EndTurn => "end_turn",
                StopReason::MaxTokens => "max_tokens",
                StopReason::ToolUse => "tool_use",
                StopReason::StopSequence => "stop_sequence",
                StopReason::Error => "error",
            }
            .to_string()
        }),
        resume_attempts: record.resume_attempts,
    })
}

/// Rebuilds a record from the columns that store it.
///
/// Pure, and the only place a stored checkpoint is trusted — every column that
/// could have been written by another build, or edited by hand, is validated
/// here rather than deeper in.
///
/// # Errors
///
/// Returns [`CheckpointErrorKind::Corrupt`](crate::checkpoint::CheckpointErrorKind::Corrupt)
/// naming the first column that would not decode.
pub fn decode_row(columns: &CheckpointColumns) -> Result<CheckpointRecord, CheckpointError> {
    let id = CheckpointId::parse(&columns.id)
        .map_err(|e| CheckpointError::corrupt("id", e.to_string()))?;

    let conversation_id = match columns.conversation_id {
        Some(ref raw) => Some(
            ConversationId::parse(raw)
                .map_err(|e| CheckpointError::corrupt("conversation_id", e.to_string()))
                .map_err(|e| e.with_checkpoint(id.clone()))?,
        ),
        None => None,
    };

    let status =
        CheckpointStatus::from_str(&columns.status).map_err(|e| e.with_checkpoint(id.clone()))?;

    let usage: Usage = serde_json::from_str(&columns.usage).map_err(|e| {
        CheckpointError::corrupt("usage", e.to_string()).with_checkpoint(id.clone())
    })?;

    let messages: Vec<Message> = serde_json::from_str(&columns.messages).map_err(|e| {
        CheckpointError::corrupt("messages", e.to_string()).with_checkpoint(id.clone())
    })?;

    let tool_calls: Vec<ExecutedToolCall> =
        serde_json::from_str(&columns.tool_calls).map_err(|e| {
            CheckpointError::corrupt("tool_calls", e.to_string()).with_checkpoint(id.clone())
        })?;

    let stop_reason = match columns.stop_reason {
        Some(ref raw) => Some(decode_stop_reason(raw).map_err(|e| e.with_checkpoint(id.clone()))?),
        None => None,
    };

    let structured_output = match columns.structured_output {
        Some(ref raw) => Some(serde_json::from_str(raw).map_err(|e| {
            CheckpointError::corrupt("structured_output", e.to_string()).with_checkpoint(id.clone())
        })?),
        None => None,
    };

    let pending_round: Option<PendingRound> = match columns.pending_round {
        Some(ref raw) => Some(serde_json::from_str(raw).map_err(|e| {
            CheckpointError::corrupt("pending_round", e.to_string()).with_checkpoint(id.clone())
        })?),
        None => None,
    };

    Ok(CheckpointRecord {
        id,
        conversation_id,
        fingerprint: TurnFingerprint::from_hex(columns.fingerprint.clone()),
        format_version: columns.format_version,
        status,
        rounds_completed: narrow(columns.rounds_completed),
        token_count: narrow(columns.token_count),
        usage,
        messages,
        tool_calls,
        final_text: columns.final_text.clone(),
        stop_reason,
        structured_output,
        pending_round,
        resume_attempts: columns.resume_attempts,
    })
}

/// Parses the `stop_reason` column.
fn decode_stop_reason(raw: &str) -> Result<StopReason, CheckpointError> {
    match raw {
        "end_turn" => Ok(StopReason::EndTurn),
        "max_tokens" => Ok(StopReason::MaxTokens),
        "tool_use" => Ok(StopReason::ToolUse),
        "stop_sequence" => Ok(StopReason::StopSequence),
        "error" => Ok(StopReason::Error),
        other => Err(CheckpointError::corrupt(
            "stop_reason",
            format!("unknown stop reason `{other}`"),
        )),
    }
}

/// Narrows a count to what the database column holds.
///
/// A turn cannot plausibly run four billion rounds, so saturating here loses
/// nothing real while keeping the cast free of a silent wrap.
fn clamp_to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Widens a stored count back to the in-memory one.
///
/// Infallible on every target this crate builds for; saturating keeps it free
/// of a panic path on any that is not.
fn narrow(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// Serializes one payload column, naming the column in any failure.
fn encode_payload<T: Serialize>(
    field: &'static str,
    id: &CheckpointId,
    value: &T,
) -> Result<String, CheckpointError> {
    serde_json::to_string(value)
        .map_err(|e| CheckpointError::corrupt(field, e.to_string()).with_checkpoint(id.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::error::CheckpointErrorKind;
    use crate::messages::ToolCall;

    fn inputs<'a>(user: &'a str, tools: &'a [String]) -> TurnInputs<'a> {
        TurnInputs {
            system_prompt: Some("be brief"),
            user_content: user,
            tool_names: tools,
            provider: "claude",
            max_tool_rounds: 8,
            structured_schema: None,
        }
    }

    fn a_record() -> CheckpointRecord {
        let tools = vec!["read_file".to_string()];
        CheckpointRecord {
            id: CheckpointId::new(),
            conversation_id: Some(ConversationId::new()),
            fingerprint: TurnFingerprint::of(&inputs("hello", &tools)),
            format_version: CHECKPOINT_FORMAT_VERSION,
            status: CheckpointStatus::InProgress,
            rounds_completed: 2,
            token_count: 41,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            },
            messages: vec![
                Message::system("be brief"),
                Message::user("hello"),
                Message::assistant_with_tools(
                    "",
                    vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "read_file".to_string(),
                        arguments: serde_json::json!({ "path": "a.txt" }),
                    }],
                ),
                Message::tool("call_1", "contents"),
            ],
            tool_calls: vec![ExecutedToolCall::success(
                "call_1",
                "read_file",
                serde_json::json!({ "path": "a.txt" }),
                serde_json::json!("contents"),
            )],
            final_text: None,
            stop_reason: None,
            structured_output: None,
            pending_round: None,
            resume_attempts: 3,
        }
    }

    #[test]
    fn record_round_trips_through_its_columns() {
        let record = a_record();
        let columns = encode_record(&record).unwrap();
        assert_eq!(decode_row(&columns).unwrap(), record);
    }

    #[test]
    fn a_completed_record_round_trips_its_answer() {
        let mut record = a_record();
        record.status = CheckpointStatus::Completed;
        record.final_text = Some("done".to_string());
        record.stop_reason = Some(StopReason::EndTurn);

        let decoded = decode_row(&encode_record(&record).unwrap()).unwrap();
        assert_eq!(decoded.final_text.as_deref(), Some("done"));
        assert_eq!(decoded.stop_reason, Some(StopReason::EndTurn));
    }

    #[test]
    fn every_stop_reason_survives_the_column() {
        for reason in [
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::ToolUse,
            StopReason::StopSequence,
            StopReason::Error,
        ] {
            let mut record = a_record();
            record.stop_reason = Some(reason);
            let decoded = decode_row(&encode_record(&record).unwrap()).unwrap();
            assert_eq!(decoded.stop_reason, Some(reason), "{reason:?}");
        }
    }

    #[test]
    fn unknown_status_text_is_reported_as_corrupt() {
        let mut columns = encode_record(&a_record()).unwrap();
        columns.status = "half_way".to_string();

        let error = decode_row(&columns).unwrap_err();
        assert!(matches!(
            error.kind(),
            CheckpointErrorKind::Corrupt {
                field: "status",
                ..
            }
        ));
        assert!(error.checkpoint_id().is_some());
    }

    #[test]
    fn bad_messages_json_is_reported_as_corrupt() {
        let mut columns = encode_record(&a_record()).unwrap();
        columns.messages = "[{".to_string();

        assert!(matches!(
            decode_row(&columns).unwrap_err().kind(),
            CheckpointErrorKind::Corrupt {
                field: "messages",
                ..
            }
        ));
    }

    #[test]
    fn bad_tool_calls_json_is_reported_as_corrupt() {
        let mut columns = encode_record(&a_record()).unwrap();
        columns.tool_calls = "not json".to_string();

        assert!(matches!(
            decode_row(&columns).unwrap_err().kind(),
            CheckpointErrorKind::Corrupt {
                field: "tool_calls",
                ..
            }
        ));
    }

    #[test]
    fn an_unparseable_id_is_reported_as_corrupt() {
        let mut columns = encode_record(&a_record()).unwrap();
        columns.id = "turn_01h455vb4pex5vsknk084sn02q".to_string();

        assert!(matches!(
            decode_row(&columns).unwrap_err().kind(),
            CheckpointErrorKind::Corrupt { field: "id", .. }
        ));
    }

    #[test]
    fn an_unknown_stop_reason_is_reported_as_corrupt() {
        let mut columns = encode_record(&a_record()).unwrap();
        columns.stop_reason = Some("gave_up".to_string());

        assert!(matches!(
            decode_row(&columns).unwrap_err().kind(),
            CheckpointErrorKind::Corrupt {
                field: "stop_reason",
                ..
            }
        ));
    }

    #[test]
    fn tool_order_does_not_change_the_fingerprint() {
        let forwards = vec!["alpha".to_string(), "beta".to_string()];
        let backwards = vec!["beta".to_string(), "alpha".to_string()];

        assert_eq!(
            TurnFingerprint::of(&inputs("hello", &forwards)),
            TurnFingerprint::of(&inputs("hello", &backwards)),
        );
    }

    #[test]
    fn a_different_user_message_changes_the_fingerprint() {
        let tools = Vec::new();
        assert_ne!(
            TurnFingerprint::of(&inputs("hello", &tools)),
            TurnFingerprint::of(&inputs("goodbye", &tools)),
        );
    }

    #[test]
    fn a_different_tool_set_changes_the_fingerprint() {
        let one = vec!["read_file".to_string()];
        let two = vec!["read_file".to_string(), "write_file".to_string()];
        assert_ne!(
            TurnFingerprint::of(&inputs("hello", &one)),
            TurnFingerprint::of(&inputs("hello", &two)),
        );
    }

    #[test]
    fn a_different_provider_changes_the_fingerprint() {
        let tools = Vec::new();
        let mut other = inputs("hello", &tools);
        other.provider = "ollama";
        assert_ne!(
            TurnFingerprint::of(&inputs("hello", &tools)),
            TurnFingerprint::of(&other),
        );
    }

    #[test]
    fn a_different_round_ceiling_changes_the_fingerprint() {
        let tools = Vec::new();
        let mut other = inputs("hello", &tools);
        other.max_tool_rounds = 9;
        assert_ne!(
            TurnFingerprint::of(&inputs("hello", &tools)),
            TurnFingerprint::of(&other),
        );
    }

    #[test]
    fn field_boundaries_are_not_confusable() {
        // "ab" + "c" must not hash the same as "a" + "bc".
        let tools = Vec::new();
        let mut left = inputs("hello", &tools);
        left.system_prompt = Some("ab");
        left.user_content = "c";
        let mut right = left;
        right.system_prompt = Some("a");
        right.user_content = "bc";

        assert_ne!(TurnFingerprint::of(&left), TurnFingerprint::of(&right));
    }

    #[test]
    fn an_absent_system_prompt_is_not_the_empty_one_by_accident() {
        // Both hash the same by construction; the test pins that down so a
        // future change to `unwrap_or_default` is a deliberate one.
        let tools = Vec::new();
        let mut none = inputs("hello", &tools);
        none.system_prompt = None;
        let mut empty = none;
        empty.system_prompt = Some("");

        assert_eq!(TurnFingerprint::of(&none), TurnFingerprint::of(&empty));
    }

    #[test]
    fn a_structured_answer_round_trips_through_its_column() {
        let mut record = a_record();
        record.status = CheckpointStatus::Completed;
        record.final_text = String::new().into();
        record.stop_reason = Some(StopReason::EndTurn);
        record.structured_output = Some(serde_json::json!({ "vendor": "Acme" }));

        let decoded = decode_row(&encode_record(&record).unwrap()).unwrap();
        assert_eq!(
            decoded.structured_output,
            Some(serde_json::json!({ "vendor": "Acme" }))
        );
    }

    #[test]
    fn bad_structured_output_json_is_reported_as_corrupt() {
        let mut columns = encode_record(&a_record()).unwrap();
        columns.structured_output = Some("{".to_string());

        assert!(matches!(
            decode_row(&columns).unwrap_err().kind(),
            CheckpointErrorKind::Corrupt {
                field: "structured_output",
                ..
            }
        ));
    }

    #[test]
    fn an_extraction_schema_changes_the_fingerprint() {
        let tools = Vec::new();
        let plain = inputs("hello", &tools);
        let mut extracting = plain;
        extracting.structured_schema = Some(r#"{"type":"object"}"#);

        assert_ne!(
            TurnFingerprint::of(&plain),
            TurnFingerprint::of(&extracting)
        );
    }

    #[test]
    fn a_different_extraction_schema_changes_the_fingerprint() {
        let tools = Vec::new();
        let mut one = inputs("hello", &tools);
        one.structured_schema = Some(r#"{"type":"object"}"#);
        let mut two = one;
        two.structured_schema = Some(r#"{"type":"array"}"#);

        assert_ne!(TurnFingerprint::of(&one), TurnFingerprint::of(&two));
    }

    #[test]
    fn status_text_round_trips() {
        for status in [
            CheckpointStatus::InProgress,
            CheckpointStatus::Completed,
            CheckpointStatus::Failed,
        ] {
            assert_eq!(CheckpointStatus::from_str(status.as_str()).unwrap(), status);
        }
    }

    #[test]
    fn only_unfinished_states_are_resumable() {
        assert!(CheckpointStatus::InProgress.is_resumable());
        assert!(CheckpointStatus::Failed.is_resumable());
        assert!(!CheckpointStatus::Completed.is_resumable());
    }

    #[test]
    fn an_opening_record_has_spent_nothing() {
        let tools = Vec::new();
        let record = CheckpointRecord::opening(
            CheckpointId::new(),
            None,
            TurnFingerprint::of(&inputs("hello", &tools)),
            vec![Message::user("hello")],
        );

        assert_eq!(record.rounds_completed, 0);
        assert_eq!(record.status, CheckpointStatus::InProgress);
        assert!(record.tool_calls.is_empty());
        assert_eq!(record.format_version, CHECKPOINT_FORMAT_VERSION);
    }
}
