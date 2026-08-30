//! Audit entries and the hash that chains them.
//!
//! Everything in this module is pure. An entry is sealed by hashing its
//! contents together with the hash of the entry before it, which is what makes
//! an edit anywhere in the file detectable: changing one entry changes its
//! hash, and every hash after it was computed from the old value.

use crate::messages::Usage;
use crate::policy::Decider;
use crate::types::{ConversationId, CorrelationId, TrailId, TurnId};
use serde::{Deserialize, Serialize};

/// The `prev_hash` of the first entry in a chain.
///
/// A fixed, well-known value rather than an absent field, so the first entry
/// is hashed by exactly the same rule as every other one and verification has
/// no special case to get wrong.
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// How a tool invocation ended.
///
/// Denied and errored invocations are recorded exactly like successful ones —
/// an audit trail that only recorded what succeeded would answer the wrong
/// question.
///
/// Marked `#[non_exhaustive]` so new outcomes can be added without breaking
/// downstream `match`es.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
#[non_exhaustive]
pub enum AuditOutcome {
    /// The tool ran and returned a value.
    Success {
        /// A bounded summary of the result, never the whole value.
        summary: String,
    },
    /// The tool ran and failed.
    Error {
        /// The error, truncated to a bounded length.
        message: String,
    },
    /// The gate refused the call, so the tool never ran.
    Denied {
        /// The rendered denial reason.
        reason: String,
    },
    /// A crash left the call's outcome unknowable, and the resume settlement
    /// declined to run it again.
    ///
    /// Written by a resumed turn for a call the interrupted process had
    /// `Started` on a non-idempotent tool: it may or may not have had its
    /// effect, and re-running it on a "maybe" is exactly what the tool's own
    /// declaration forbids. The entry is the trail's only trace of the call —
    /// the first attempt died before it could write one.
    Uncertain {
        /// The feedback handed to the model in place of a tool result.
        message: String,
    },
}

/// The gate's verdict, and who reached it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditDecision {
    /// Whether the call was allowed to run.
    pub approved: bool,
    /// Which rule or party decided.
    pub decided_by: Decider,
}

impl AuditDecision {
    /// A call that was allowed to run.
    #[must_use]
    pub fn approved(decided_by: Decider) -> Self {
        Self {
            approved: true,
            decided_by,
        }
    }

    /// A call that was refused.
    #[must_use]
    pub fn refused(decided_by: Decider) -> Self {
        Self {
            approved: false,
            decided_by,
        }
    }
}

/// Everything about one invocation except its position in the chain.
///
/// The prompt loop builds this; the audit actor turns it into a sealed
/// [`AuditEntry`] by adding the sequence number and the previous hash, which
/// are the two things only the single writer knows.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InvocationRecord {
    /// When the invocation finished, RFC 3339 in UTC.
    pub timestamp: String,
    /// The round that requested the call.
    pub correlation_id: CorrelationId,
    /// The conversation, when the call happened inside one.
    pub conversation_id: Option<ConversationId>,
    /// The principal on whose behalf the call ran, when configured.
    pub user: Option<String>,
    /// The turn the call belongs to.
    pub turn_id: TurnId,
    /// The provider's ID for this particular call.
    ///
    /// Together with `turn_id` this is what joins the entry to the
    /// `TurnLifecycle::ToolStarted` and `ToolFinished` a live observer saw,
    /// so a recorded trail and a watched session can be reconciled after
    /// the fact.
    pub tool_call_id: String,
    /// The tool name as the model saw it.
    pub tool_name: String,
    /// The arguments, already redacted.
    pub arguments: serde_json::Value,
    /// How it ended.
    pub outcome: AuditOutcome,
    /// What the gate decided.
    pub decision: AuditDecision,
    /// How long the call took, in milliseconds. Zero for a refused call.
    pub duration_ms: u64,
    /// Size in bytes of the complete serialized tool response.
    ///
    /// Absent when the tool produced no response, such as a refused call.
    pub response_size_bytes: Option<u64>,
    /// Whether the call ran inside a turn resumed from a checkpoint.
    ///
    /// `false` for every first-run call. A restarted process finishing an
    /// interrupted turn stamps `true` on everything it executes, so the trail
    /// shows which effects came from crash recovery.
    pub resumed: bool,
}

/// How an attempted model turn ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
#[non_exhaustive]
pub enum TurnAuditOutcome {
    /// The model produced a final answer.
    Completed,
    /// The turn ended with an error.
    Failed,
    /// The caller dropped the turn future before it finished.
    Interrupted,
    /// Admission control refused the turn before any provider request.
    Refused {
        /// The stable admission state (`paused` or `draining`).
        decision: String,
        /// The rendered refusal handed to the caller.
        reason: String,
    },
}

/// Metadata for one attempted model turn, excluding its chain position.
///
/// Prompt and response content are deliberately absent. Their byte counts
/// make activity and response length auditable without copying user data into
/// a long-lived SIEM trail.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TurnRecord {
    /// When the turn ended, RFC 3339 in UTC.
    pub timestamp: String,
    /// Correlation identity for this turn-level event.
    pub correlation_id: CorrelationId,
    /// The conversation, when the turn happened inside one.
    pub conversation_id: Option<ConversationId>,
    /// The principal on whose behalf the turn ran, when configured.
    pub user: Option<String>,
    /// The attempted turn.
    pub turn_id: TurnId,
    /// How the turn ended.
    pub outcome: TurnAuditOutcome,
    /// Size in bytes of the user's prompt for this turn.
    pub prompt_size_bytes: u64,
    /// Size in bytes of the final response; zero when none was produced.
    pub response_size_bytes: u64,
    /// Configured provider name billed for the turn.
    pub provider: String,
    /// Configured model name.
    pub model: String,
    /// Usage summed across all provider rounds.
    pub usage: Usage,
}

/// An unsealed event accepted by the audit writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AuditRecord {
    /// A tool invocation.
    Invocation(InvocationRecord),
    /// A model turn.
    Turn(TurnRecord),
}

/// The kind of event represented by an [`AuditEntry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEntryKind {
    /// One tool invocation. Omitted on disk for backward compatibility.
    Invocation,
    /// One attempted model turn.
    Turn,
}

/// One sealed, chained record of a model turn or tool invocation.
///
/// Serialized as a single JSON object per line (JSONL), which is what makes
/// the file both append-only and directly ingestible by a SIEM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Position in the chain, starting at 1.
    pub sequence: u64,
    /// When the invocation finished, RFC 3339 in UTC.
    pub timestamp: String,
    /// The round that requested the call.
    pub correlation_id: CorrelationId,
    /// The conversation, when the call happened inside one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<ConversationId>,
    /// The principal on whose behalf the call ran, when configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// The turn the call belongs to.
    pub turn_id: TurnId,
    /// Entry discriminator. Absent on legacy invocation entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_kind: Option<AuditEntryKind>,
    /// The provider's ID for this particular call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// The tool name as the model saw it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// The arguments, already redacted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
    /// How it ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<AuditOutcome>,
    /// What the gate decided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<AuditDecision>,
    /// How long the call took, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Size in bytes of the complete serialized tool response, when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_size_bytes: Option<u64>,
    /// How a turn ended. Present only on turn entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_outcome: Option<TurnAuditOutcome>,
    /// User prompt length. Present only on turn entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_size_bytes: Option<u64>,
    /// Configured provider. Present only on turn entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Configured model. Present only on turn entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Input tokens summed across the turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Output tokens summed across the turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Whether the call ran inside a turn resumed from a checkpoint.
    ///
    /// Serialized only when `true`, so entries written before this field
    /// existed — and every ordinary first-run entry since — keep the exact
    /// bytes their hashes were computed over.
    #[serde(default, skip_serializing_if = "is_false")]
    pub resumed: bool,
    /// The identity of the trail this entry was sealed into.
    ///
    /// Covered by the hash, so a sealed entry cannot be relabelled as
    /// belonging to a different trail without breaking its own hash. Absent
    /// on entries written before trails had an identity; those keep the exact
    /// bytes their hashes were computed over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trail_id: Option<TrailId>,
    /// The hash of the entry before this one, or [`GENESIS_HASH`].
    pub prev_hash: String,
    /// This entry's hash, covering every field above.
    pub hash: String,
}

/// The fields covered by an entry's hash, in a fixed order.
///
/// A dedicated struct rather than hashing the serialized entry, so that
/// `hash` is excluded by construction and the field order is a deliberate,
/// stable decision rather than whatever the entry's layout happens to be.
#[derive(Serialize)]
struct HashPreimage<'a> {
    sequence: u64,
    timestamp: &'a str,
    correlation_id: &'a CorrelationId,
    conversation_id: Option<&'a ConversationId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
    turn_id: &'a TurnId,
    #[serde(skip_serializing_if = "Option::is_none")]
    entry_kind: Option<AuditEntryKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<&'a AuditOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decision: Option<&'a AuditDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    turn_outcome: Option<&'a TurnAuditOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_tokens: Option<u64>,
    /// `Some(true)` only on a resumed call; skipped when absent. A field that
    /// serialized `false` on every pre-existing entry would change their
    /// pre-image bytes and break verification of every chain already written.
    #[serde(skip_serializing_if = "Option::is_none")]
    resumed: Option<bool>,
    /// Skipped when absent for the same reason as `resumed`: a legacy entry
    /// must hash to what it hashed to when it was written.
    #[serde(skip_serializing_if = "Option::is_none")]
    trail_id: Option<&'a TrailId>,
    prev_hash: &'a str,
}

/// Whether a bool is `false`, for `skip_serializing_if` — which requires the
/// reference-taking signature.
fn is_false(value: &bool) -> bool {
    !*value
}

/// Hashes a pre-image with BLAKE3, hex-encoded.
///
/// The previous hash is fed in as bytes ahead of the pre-image as well as
/// being one of its fields. Belt and braces: the link survives even if a
/// future change to the pre-image layout drops the field.
fn hash_preimage(preimage: &HashPreimage<'_>) -> String {
    let encoded = serde_json::to_vec(preimage)
        .expect("an audit pre-image is plain data and always serializes");

    let mut hasher = blake3::Hasher::new();
    hasher.update(preimage.prev_hash.as_bytes());
    hasher.update(b"\x00");
    hasher.update(&encoded);
    hasher.finalize().to_hex().to_string()
}

impl AuditEntry {
    /// Seals a record into the chain behind `prev_hash`, under `trail_id`.
    ///
    /// Pure: the same record at the same position behind the same predecessor
    /// in the same trail always produces the same entry, which is what lets
    /// verification recompute it later. `None` for the trail is the legacy
    /// form — what every entry looked like before trails had an identity —
    /// kept so old fixtures can still be reproduced; a running audit log
    /// always passes its identity.
    #[must_use]
    pub fn seal(
        record: InvocationRecord,
        sequence: u64,
        prev_hash: &str,
        trail_id: Option<&TrailId>,
    ) -> Self {
        let InvocationRecord {
            timestamp,
            correlation_id,
            conversation_id,
            user,
            turn_id,
            tool_call_id,
            tool_name,
            arguments,
            outcome,
            decision,
            duration_ms,
            response_size_bytes,
            resumed,
        } = record;

        let hash = hash_preimage(&HashPreimage {
            sequence,
            timestamp: &timestamp,
            correlation_id: &correlation_id,
            conversation_id: conversation_id.as_ref(),
            user: user.as_deref(),
            turn_id: &turn_id,
            entry_kind: None,
            tool_call_id: Some(&tool_call_id),
            tool_name: Some(&tool_name),
            arguments: Some(&arguments),
            outcome: Some(&outcome),
            decision: Some(&decision),
            duration_ms: Some(duration_ms),
            response_size_bytes,
            turn_outcome: None,
            prompt_size_bytes: None,
            provider: None,
            model: None,
            input_tokens: None,
            output_tokens: None,
            resumed: resumed.then_some(true),
            trail_id,
            prev_hash,
        });

        Self {
            sequence,
            timestamp,
            correlation_id,
            conversation_id,
            user,
            turn_id,
            entry_kind: None,
            tool_call_id: Some(tool_call_id),
            tool_name: Some(tool_name),
            arguments: Some(arguments),
            outcome: Some(outcome),
            decision: Some(decision),
            duration_ms: Some(duration_ms),
            response_size_bytes,
            turn_outcome: None,
            prompt_size_bytes: None,
            provider: None,
            model: None,
            input_tokens: None,
            output_tokens: None,
            resumed,
            trail_id: trail_id.cloned(),
            prev_hash: prev_hash.to_string(),
            hash,
        }
    }

    /// Seals a turn-level metadata record into the chain.
    #[must_use]
    pub fn seal_turn(
        record: TurnRecord,
        sequence: u64,
        prev_hash: &str,
        trail_id: Option<&TrailId>,
    ) -> Self {
        let TurnRecord {
            timestamp,
            correlation_id,
            conversation_id,
            user,
            turn_id,
            outcome,
            prompt_size_bytes,
            response_size_bytes,
            provider,
            model,
            usage,
        } = record;
        let kind = AuditEntryKind::Turn;
        let hash = hash_preimage(&HashPreimage {
            sequence,
            timestamp: &timestamp,
            correlation_id: &correlation_id,
            conversation_id: conversation_id.as_ref(),
            user: user.as_deref(),
            turn_id: &turn_id,
            entry_kind: Some(kind),
            tool_call_id: None,
            tool_name: None,
            arguments: None,
            outcome: None,
            decision: None,
            duration_ms: None,
            response_size_bytes: Some(response_size_bytes),
            turn_outcome: Some(&outcome),
            prompt_size_bytes: Some(prompt_size_bytes),
            provider: Some(&provider),
            model: Some(&model),
            input_tokens: Some(
                usage
                    .input_tokens
                    .saturating_add(usage.cache_read_tokens)
                    .saturating_add(usage.cache_creation_tokens),
            ),
            output_tokens: Some(usage.output_tokens),
            resumed: None,
            trail_id,
            prev_hash,
        });
        Self {
            sequence,
            timestamp,
            correlation_id,
            conversation_id,
            user,
            turn_id,
            entry_kind: Some(kind),
            tool_call_id: None,
            tool_name: None,
            arguments: None,
            outcome: None,
            decision: None,
            duration_ms: None,
            response_size_bytes: Some(response_size_bytes),
            turn_outcome: Some(outcome),
            prompt_size_bytes: Some(prompt_size_bytes),
            provider: Some(provider),
            model: Some(model),
            input_tokens: Some(
                usage
                    .input_tokens
                    .saturating_add(usage.cache_read_tokens)
                    .saturating_add(usage.cache_creation_tokens),
            ),
            output_tokens: Some(usage.output_tokens),
            resumed: false,
            trail_id: trail_id.cloned(),
            prev_hash: prev_hash.to_string(),
            hash,
        }
    }

    /// Returns the logical entry kind, treating an absent discriminator as a
    /// legacy invocation.
    #[must_use]
    pub fn kind(&self) -> AuditEntryKind {
        self.entry_kind.unwrap_or(AuditEntryKind::Invocation)
    }

    /// Recomputes what this entry's hash should be from its own contents.
    ///
    /// Verification compares this against the stored [`hash`](Self::hash);
    /// any edit to a covered field makes the two disagree.
    #[must_use]
    pub fn recompute_hash(&self) -> String {
        hash_preimage(&HashPreimage {
            sequence: self.sequence,
            timestamp: &self.timestamp,
            correlation_id: &self.correlation_id,
            conversation_id: self.conversation_id.as_ref(),
            user: self.user.as_deref(),
            turn_id: &self.turn_id,
            entry_kind: self.entry_kind,
            tool_call_id: self.tool_call_id.as_deref(),
            tool_name: self.tool_name.as_deref(),
            arguments: self.arguments.as_ref(),
            outcome: self.outcome.as_ref(),
            decision: self.decision.as_ref(),
            duration_ms: self.duration_ms,
            response_size_bytes: self.response_size_bytes,
            turn_outcome: self.turn_outcome.as_ref(),
            prompt_size_bytes: self.prompt_size_bytes,
            provider: self.provider.as_deref(),
            model: self.model.as_deref(),
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            resumed: self.resumed.then_some(true),
            trail_id: self.trail_id.as_ref(),
            prev_hash: &self.prev_hash,
        })
    }

    /// Serializes the entry as one JSONL line, without the newline.
    ///
    /// # Errors
    ///
    /// Returns the underlying `serde_json` error if the entry cannot be
    /// serialized, which in practice means an argument value that is not
    /// valid JSON.
    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn record(tool_name: &str) -> InvocationRecord {
        InvocationRecord {
            timestamp: "2026-08-19T12:00:00Z".to_string(),
            correlation_id: CorrelationId::new(),
            conversation_id: None,
            user: None,
            turn_id: TurnId::new(),
            tool_call_id: "toolu_01".to_string(),
            tool_name: tool_name.to_string(),
            arguments: json!({"value": 1}),
            outcome: AuditOutcome::Success {
                summary: "ok".to_string(),
            },
            decision: AuditDecision::approved(Decider::NoPolicy),
            duration_ms: 5,
            response_size_bytes: Some(4),
            resumed: false,
        }
    }

    #[test]
    fn sealing_is_deterministic() {
        let first = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);
        let second = AuditEntry::seal(
            InvocationRecord {
                correlation_id: first.correlation_id.clone(),
                turn_id: first.turn_id.clone(),
                ..record("bash")
            },
            1,
            GENESIS_HASH,
            None,
        );

        assert_eq!(first.hash, second.hash);
    }

    #[test]
    fn a_sealed_entry_verifies_against_itself() {
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);
        assert_eq!(entry.hash, entry.recompute_hash());
    }

    #[test]
    fn editing_any_covered_field_changes_the_hash() {
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);

        let mut tampered = entry.clone();
        tampered.arguments = Some(json!({"value": 2}));
        assert_ne!(tampered.hash, tampered.recompute_hash());

        let mut renamed = entry.clone();
        renamed.tool_name = Some("read_file".to_string());
        assert_ne!(renamed.hash, renamed.recompute_hash());

        let mut reversed = entry.clone();
        reversed.decision = Some(AuditDecision::refused(Decider::Denylist));
        assert_ne!(reversed.hash, reversed.recompute_hash());

        let mut reattributed = entry.clone();
        reattributed.user = Some("acct:mallory".to_string());
        assert_ne!(reattributed.hash, reattributed.recompute_hash());

        let mut resized = entry.clone();
        resized.response_size_bytes = Some(999);
        assert_ne!(resized.hash, resized.recompute_hash());

        // Repointing an entry at a different call would let one invocation's
        // recorded outcome be passed off as another's.
        let mut recalled = entry;
        recalled.tool_call_id = Some("toolu_02".to_string());
        assert_ne!(recalled.hash, recalled.recompute_hash());
    }

    #[test]
    fn the_tool_call_id_survives_a_seal_and_verify_round_trip() {
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);
        assert_eq!(entry.tool_call_id.as_deref(), Some("toolu_01"));

        let line = entry.to_jsonl().expect("an entry must serialize");
        let parsed: AuditEntry = serde_json::from_str(&line).expect("an entry must parse back");

        assert_eq!(parsed.tool_call_id.as_deref(), Some("toolu_01"));
        assert_eq!(parsed.recompute_hash(), entry.hash);
    }

    #[test]
    fn position_in_the_chain_is_part_of_the_hash() {
        let first = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);
        let moved = AuditEntry::seal(record("bash"), 2, GENESIS_HASH, None);
        assert_ne!(first.hash, moved.hash);
    }

    #[test]
    fn the_predecessor_is_part_of_the_hash() {
        let behind_genesis = AuditEntry::seal(record("bash"), 2, GENESIS_HASH, None);
        let behind_other = AuditEntry::seal(record("bash"), 2, &"a".repeat(64), None);
        assert_ne!(behind_genesis.hash, behind_other.hash);
    }

    #[test]
    fn an_entry_survives_a_jsonl_round_trip_with_its_hash_intact() {
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);
        let line = entry.to_jsonl().expect("an entry must serialize");

        assert!(!line.contains('\n'), "a JSONL line must be one line");

        let parsed: AuditEntry = serde_json::from_str(&line).expect("an entry must parse back");
        assert_eq!(parsed, entry);
        assert_eq!(
            parsed.recompute_hash(),
            entry.hash,
            "the hash must survive serialization, or verification of a written file is meaningless"
        );
    }

    #[test]
    fn a_first_run_entry_serializes_without_the_resumed_field() {
        // The marker must not change the bytes of ordinary entries: every
        // chain written before the field existed has to keep verifying.
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);
        let line = entry.to_jsonl().unwrap();
        assert!(!line.contains("resumed"), "{line}");

        // A line written by a build that predates the field reads back and
        // verifies, because absence and `false` are the same thing.
        let parsed: AuditEntry = serde_json::from_str(&line).unwrap();
        assert!(!parsed.resumed);
        assert_eq!(parsed.recompute_hash(), entry.hash);
    }

    #[test]
    fn an_entry_written_by_v0_33_still_verifies() {
        let line = r#"{"sequence":1,"timestamp":"2026-08-19T12:00:00Z","correlation_id":"corr_01h455vb4pex5vsknk084sn02q","turn_id":"turn_01h455vb4pex5vsknk084sn02q","tool_call_id":"toolu_01","tool_name":"bash","arguments":{"command":"ls"},"outcome":{"kind":"success","summary":"ok"},"decision":{"approved":true,"decided_by":"no_policy"},"duration_ms":3,"prev_hash":"0000000000000000000000000000000000000000000000000000000000000000","hash":"2de3ebea3a7de4cbcef1f69cdcf2c7ebfa2f77a961439609db1d6315d16d602b"}"#;
        let entry: AuditEntry = serde_json::from_str(line).expect("the v0.33 fixture must parse");

        assert_eq!(entry.recompute_hash(), entry.hash);
    }

    #[test]
    fn a_resumed_entry_is_marked_and_hashes_differently() {
        let plain = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);
        let resumed = AuditEntry::seal(
            InvocationRecord {
                resumed: true,
                correlation_id: plain.correlation_id.clone(),
                turn_id: plain.turn_id.clone(),
                ..record("bash")
            },
            1,
            GENESIS_HASH,
            None,
        );

        assert!(resumed.resumed);
        assert_ne!(
            plain.hash, resumed.hash,
            "the marker must be covered by the hash, or it could be stripped undetected"
        );

        let line = resumed.to_jsonl().unwrap();
        assert!(line.contains("\"resumed\":true"), "{line}");
        let parsed: AuditEntry = serde_json::from_str(&line).unwrap();
        assert!(parsed.resumed);
        assert_eq!(parsed.recompute_hash(), resumed.hash);
    }

    #[test]
    fn a_denied_entry_records_the_reason_and_no_duration() {
        let denied = InvocationRecord {
            outcome: AuditOutcome::Denied {
                reason: "denied by policy".to_string(),
            },
            decision: AuditDecision::refused(Decider::Denylist),
            duration_ms: 0,
            ..record("bash")
        };
        let entry = AuditEntry::seal(denied, 1, GENESIS_HASH, None);

        assert!(!entry.decision.as_ref().unwrap().approved);
        assert_eq!(
            entry.decision.as_ref().unwrap().decided_by,
            Decider::Denylist
        );
        assert_eq!(entry.hash, entry.recompute_hash());
    }

    #[test]
    fn an_identified_entry_carries_its_trail_and_hashes_differently() {
        let trail = TrailId::new();
        let plain = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);
        let identified = AuditEntry::seal(
            InvocationRecord {
                correlation_id: plain.correlation_id.clone(),
                turn_id: plain.turn_id.clone(),
                ..record("bash")
            },
            1,
            GENESIS_HASH,
            Some(&trail),
        );

        assert_eq!(identified.trail_id.as_ref(), Some(&trail));
        assert_ne!(
            plain.hash, identified.hash,
            "the identity must be covered by the hash, or it could be stripped undetected"
        );
        assert_eq!(identified.recompute_hash(), identified.hash);

        let line = identified.to_jsonl().unwrap();
        assert!(
            line.contains(&format!("\"trail_id\":\"{trail}\"")),
            "{line}"
        );
        let parsed: AuditEntry = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed, identified);
        assert_eq!(parsed.recompute_hash(), identified.hash);
    }

    #[test]
    fn relabelling_a_sealed_entry_to_another_trail_breaks_its_hash() {
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, Some(&TrailId::new()));

        let mut relabelled = entry.clone();
        relabelled.trail_id = Some(TrailId::new());
        assert_ne!(relabelled.hash, relabelled.recompute_hash());

        let mut stripped = entry;
        stripped.trail_id = None;
        assert_ne!(
            stripped.hash,
            stripped.recompute_hash(),
            "removing the identity must be as detectable as changing it"
        );
    }

    #[test]
    fn an_unidentified_entry_serializes_without_the_trail_field() {
        // Byte-identical to what a build before trail identity wrote, so every
        // chain already on disk keeps verifying.
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH, None);
        let line = entry.to_jsonl().unwrap();
        assert!(!line.contains("trail_id"), "{line}");

        let parsed: AuditEntry = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed.trail_id, None);
        assert_eq!(parsed.recompute_hash(), entry.hash);
    }
}
