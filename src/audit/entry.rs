//! Audit entries and the hash that chains them.
//!
//! Everything in this module is pure. An entry is sealed by hashing its
//! contents together with the hash of the entry before it, which is what makes
//! an edit anywhere in the file detectable: changing one entry changes its
//! hash, and every hash after it was computed from the old value.

use crate::policy::Decider;
use crate::types::{ConversationId, CorrelationId, TurnId};
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
pub struct InvocationRecord {
    /// When the invocation finished, RFC 3339 in UTC.
    pub timestamp: String,
    /// The round that requested the call.
    pub correlation_id: CorrelationId,
    /// The conversation, when the call happened inside one.
    pub conversation_id: Option<ConversationId>,
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
    /// Whether the call ran inside a turn resumed from a checkpoint.
    ///
    /// `false` for every first-run call. A restarted process finishing an
    /// interrupted turn stamps `true` on everything it executes, so the trail
    /// shows which effects came from crash recovery.
    pub resumed: bool,
}

/// One sealed, chained record of a tool invocation.
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
    /// The turn the call belongs to.
    pub turn_id: TurnId,
    /// The provider's ID for this particular call.
    pub tool_call_id: String,
    /// The tool name as the model saw it.
    pub tool_name: String,
    /// The arguments, already redacted.
    pub arguments: serde_json::Value,
    /// How it ended.
    pub outcome: AuditOutcome,
    /// What the gate decided.
    pub decision: AuditDecision,
    /// How long the call took, in milliseconds.
    pub duration_ms: u64,
    /// Whether the call ran inside a turn resumed from a checkpoint.
    ///
    /// Serialized only when `true`, so entries written before this field
    /// existed — and every ordinary first-run entry since — keep the exact
    /// bytes their hashes were computed over.
    #[serde(default, skip_serializing_if = "is_false")]
    pub resumed: bool,
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
    turn_id: &'a TurnId,
    tool_call_id: &'a str,
    tool_name: &'a str,
    arguments: &'a serde_json::Value,
    outcome: &'a AuditOutcome,
    decision: &'a AuditDecision,
    duration_ms: u64,
    /// `Some(true)` only on a resumed call; skipped when absent. A field that
    /// serialized `false` on every pre-existing entry would change their
    /// pre-image bytes and break verification of every chain already written.
    #[serde(skip_serializing_if = "Option::is_none")]
    resumed: Option<bool>,
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
    /// Seals a record into the chain behind `prev_hash`.
    ///
    /// Pure: the same record at the same position behind the same predecessor
    /// always produces the same entry, which is what lets verification
    /// recompute it later.
    #[must_use]
    pub fn seal(record: InvocationRecord, sequence: u64, prev_hash: &str) -> Self {
        let InvocationRecord {
            timestamp,
            correlation_id,
            conversation_id,
            turn_id,
            tool_call_id,
            tool_name,
            arguments,
            outcome,
            decision,
            duration_ms,
            resumed,
        } = record;

        let hash = hash_preimage(&HashPreimage {
            sequence,
            timestamp: &timestamp,
            correlation_id: &correlation_id,
            conversation_id: conversation_id.as_ref(),
            turn_id: &turn_id,
            tool_call_id: &tool_call_id,
            tool_name: &tool_name,
            arguments: &arguments,
            outcome: &outcome,
            decision: &decision,
            duration_ms,
            resumed: resumed.then_some(true),
            prev_hash,
        });

        Self {
            sequence,
            timestamp,
            correlation_id,
            conversation_id,
            turn_id,
            tool_call_id,
            tool_name,
            arguments,
            outcome,
            decision,
            duration_ms,
            resumed,
            prev_hash: prev_hash.to_string(),
            hash,
        }
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
            turn_id: &self.turn_id,
            tool_call_id: &self.tool_call_id,
            tool_name: &self.tool_name,
            arguments: &self.arguments,
            outcome: &self.outcome,
            decision: &self.decision,
            duration_ms: self.duration_ms,
            resumed: self.resumed.then_some(true),
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
            turn_id: TurnId::new(),
            tool_call_id: "toolu_01".to_string(),
            tool_name: tool_name.to_string(),
            arguments: json!({"value": 1}),
            outcome: AuditOutcome::Success {
                summary: "ok".to_string(),
            },
            decision: AuditDecision::approved(Decider::NoPolicy),
            duration_ms: 5,
            resumed: false,
        }
    }

    #[test]
    fn sealing_is_deterministic() {
        let first = AuditEntry::seal(record("bash"), 1, GENESIS_HASH);
        let second = AuditEntry::seal(
            InvocationRecord {
                correlation_id: first.correlation_id.clone(),
                turn_id: first.turn_id.clone(),
                ..record("bash")
            },
            1,
            GENESIS_HASH,
        );

        assert_eq!(first.hash, second.hash);
    }

    #[test]
    fn a_sealed_entry_verifies_against_itself() {
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH);
        assert_eq!(entry.hash, entry.recompute_hash());
    }

    #[test]
    fn editing_any_covered_field_changes_the_hash() {
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH);

        let mut tampered = entry.clone();
        tampered.arguments = json!({"value": 2});
        assert_ne!(tampered.hash, tampered.recompute_hash());

        let mut renamed = entry.clone();
        renamed.tool_name = "read_file".to_string();
        assert_ne!(renamed.hash, renamed.recompute_hash());

        let mut reversed = entry.clone();
        reversed.decision = AuditDecision::refused(Decider::Denylist);
        assert_ne!(reversed.hash, reversed.recompute_hash());

        // Repointing an entry at a different call would let one invocation's
        // recorded outcome be passed off as another's.
        let mut recalled = entry;
        recalled.tool_call_id = "toolu_02".to_string();
        assert_ne!(recalled.hash, recalled.recompute_hash());
    }

    #[test]
    fn the_tool_call_id_survives_a_seal_and_verify_round_trip() {
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH);
        assert_eq!(entry.tool_call_id, "toolu_01");

        let line = entry.to_jsonl().expect("an entry must serialize");
        let parsed: AuditEntry = serde_json::from_str(&line).expect("an entry must parse back");

        assert_eq!(parsed.tool_call_id, "toolu_01");
        assert_eq!(parsed.recompute_hash(), entry.hash);
    }

    #[test]
    fn position_in_the_chain_is_part_of_the_hash() {
        let first = AuditEntry::seal(record("bash"), 1, GENESIS_HASH);
        let moved = AuditEntry::seal(record("bash"), 2, GENESIS_HASH);
        assert_ne!(first.hash, moved.hash);
    }

    #[test]
    fn the_predecessor_is_part_of_the_hash() {
        let behind_genesis = AuditEntry::seal(record("bash"), 2, GENESIS_HASH);
        let behind_other = AuditEntry::seal(record("bash"), 2, &"a".repeat(64));
        assert_ne!(behind_genesis.hash, behind_other.hash);
    }

    #[test]
    fn an_entry_survives_a_jsonl_round_trip_with_its_hash_intact() {
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH);
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
        let entry = AuditEntry::seal(record("bash"), 1, GENESIS_HASH);
        let line = entry.to_jsonl().unwrap();
        assert!(!line.contains("resumed"), "{line}");

        // A line written by a build that predates the field reads back and
        // verifies, because absence and `false` are the same thing.
        let parsed: AuditEntry = serde_json::from_str(&line).unwrap();
        assert!(!parsed.resumed);
        assert_eq!(parsed.recompute_hash(), entry.hash);
    }

    #[test]
    fn a_resumed_entry_is_marked_and_hashes_differently() {
        let plain = AuditEntry::seal(record("bash"), 1, GENESIS_HASH);
        let resumed = AuditEntry::seal(
            InvocationRecord {
                resumed: true,
                correlation_id: plain.correlation_id.clone(),
                turn_id: plain.turn_id.clone(),
                ..record("bash")
            },
            1,
            GENESIS_HASH,
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
        let entry = AuditEntry::seal(denied, 1, GENESIS_HASH);

        assert!(!entry.decision.approved);
        assert_eq!(entry.decision.decided_by, Decider::Denylist);
        assert_eq!(entry.hash, entry.recompute_hash());
    }
}
