//! Walking a chain and reporting where it breaks.
//!
//! Pure: entries in, verdict out. The CLI reads the file and hands the parsed
//! entries here, which is what lets the interesting cases — a tampered middle
//! entry, a reordering, a truncation — be tested without touching a disk.

use crate::audit::entry::{AuditEntry, GENESIS_HASH};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Where a verified chain ends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainHead {
    /// The sequence number of the last entry, or 0 for an empty chain.
    pub sequence: u64,
    /// The hash of the last entry, or [`GENESIS_HASH`] for an empty chain.
    pub hash: String,
    /// How many entries the chain holds.
    pub entries: u64,
}

impl ChainHead {
    /// The head of a chain with nothing in it yet.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            sequence: 0,
            hash: GENESIS_HASH.to_string(),
            entries: 0,
        }
    }
}

/// What is wrong at the first broken link.
///
/// Marked `#[non_exhaustive]` so further integrity checks can be added without
/// breaking downstream `match`es.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
#[non_exhaustive]
pub enum ChainBreakKind {
    /// The entry's contents do not hash to the hash it carries: it was edited.
    HashMismatch {
        /// What the contents actually hash to.
        expected: String,
        /// What the entry claims.
        found: String,
    },
    /// The entry does not point at its predecessor: one was removed,
    /// reordered, or spliced in.
    PrevHashMismatch {
        /// The predecessor's real hash.
        expected: String,
        /// What this entry points at.
        found: String,
    },
    /// Sequence numbers are not consecutive: an entry was removed.
    SequenceGap {
        /// The sequence number this position should hold.
        expected: u64,
        /// What it holds.
        found: u64,
    },
}

impl fmt::Display for ChainBreakKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HashMismatch { expected, found } => write!(
                f,
                "entry contents hash to {expected} but the entry carries {found} \
                 — the entry was modified",
            ),
            Self::PrevHashMismatch { expected, found } => write!(
                f,
                "entry points at predecessor {found} but the preceding entry hashes to \
                 {expected} — an entry was removed, reordered, or inserted",
            ),
            Self::SequenceGap { expected, found } => write!(
                f,
                "expected sequence {expected} but found {found} — an entry is missing",
            ),
        }
    }
}

/// The first broken link in a chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainBreak {
    /// The 1-based line the broken entry sits on in the file.
    pub line: usize,
    /// The sequence number the broken entry carries.
    pub sequence: u64,
    /// What is wrong.
    pub kind: ChainBreakKind,
}

impl fmt::Display for ChainBreak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "line {}, sequence {}: {}",
            self.line, self.sequence, self.kind
        )
    }
}

/// Walks a chain from the genesis hash and reports the first break.
///
/// Three things are checked per entry, in the order that gives the most
/// specific diagnosis first: that sequence numbers are consecutive, that the
/// entry points at the entry before it, and that the entry's contents still
/// hash to the hash it carries.
///
/// # Errors
///
/// Returns the first [`ChainBreak`] found. An intact chain returns its
/// [`ChainHead`].
///
/// # Note on truncation
///
/// Removing entries from the *end* leaves a chain that is internally
/// consistent — every remaining link still verifies. That is inherent to
/// hash chaining: detecting it needs the head compared against a value kept
/// somewhere the writer cannot reach. [`ChainHead`] is what to compare.
pub fn verify_chain(entries: &[AuditEntry]) -> Result<ChainHead, ChainBreak> {
    let mut prev_hash = GENESIS_HASH.to_string();
    let mut expected_sequence: u64 = 1;

    for (index, entry) in entries.iter().enumerate() {
        let line = index + 1;

        if entry.sequence != expected_sequence {
            return Err(ChainBreak {
                line,
                sequence: entry.sequence,
                kind: ChainBreakKind::SequenceGap {
                    expected: expected_sequence,
                    found: entry.sequence,
                },
            });
        }

        if entry.prev_hash != prev_hash {
            return Err(ChainBreak {
                line,
                sequence: entry.sequence,
                kind: ChainBreakKind::PrevHashMismatch {
                    expected: prev_hash,
                    found: entry.prev_hash.clone(),
                },
            });
        }

        let recomputed = entry.recompute_hash();
        if recomputed != entry.hash {
            return Err(ChainBreak {
                line,
                sequence: entry.sequence,
                kind: ChainBreakKind::HashMismatch {
                    expected: recomputed,
                    found: entry.hash.clone(),
                },
            });
        }

        prev_hash = entry.hash.clone();
        expected_sequence = expected_sequence.saturating_add(1);
    }

    Ok(ChainHead {
        sequence: entries.last().map_or(0, |entry| entry.sequence),
        hash: prev_hash,
        entries: entries.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::entry::{AuditDecision, AuditOutcome, InvocationRecord};
    use crate::policy::Decider;
    use crate::types::{CorrelationId, TurnId};
    use serde_json::json;

    fn record(index: u64) -> InvocationRecord {
        InvocationRecord {
            timestamp: format!("2026-08-19T12:00:0{index}Z"),
            correlation_id: CorrelationId::new(),
            conversation_id: None,
            turn_id: TurnId::new(),
            tool_call_id: format!("toolu_{index}"),
            tool_name: format!("tool_{index}"),
            arguments: json!({"index": index}),
            outcome: AuditOutcome::Success {
                summary: "ok".to_string(),
            },
            decision: AuditDecision::approved(Decider::NoPolicy),
            duration_ms: index,
        }
    }

    /// Builds an intact chain of `count` entries.
    fn chain(count: u64) -> Vec<AuditEntry> {
        let mut prev = GENESIS_HASH.to_string();
        let mut entries = Vec::new();
        for index in 1..=count {
            let entry = AuditEntry::seal(record(index), index, &prev);
            prev.clone_from(&entry.hash);
            entries.push(entry);
        }
        entries
    }

    #[test]
    fn an_empty_chain_verifies_to_the_genesis_head() {
        let head = verify_chain(&[]).expect("an empty chain is intact");
        assert_eq!(head, ChainHead::empty());
    }

    #[test]
    fn an_intact_chain_verifies() {
        let entries = chain(5);
        let head = verify_chain(&entries).expect("an untouched chain must verify");

        assert_eq!(head.entries, 5);
        assert_eq!(head.sequence, 5);
        assert_eq!(head.hash, entries[4].hash);
    }

    #[test]
    fn the_first_entry_links_to_the_genesis_hash() {
        let entries = chain(1);
        assert_eq!(entries[0].prev_hash, GENESIS_HASH);
        assert!(verify_chain(&entries).is_ok());
    }

    #[test]
    fn a_tampered_middle_entry_is_reported_at_that_entry() {
        let mut entries = chain(5);
        entries[2].arguments = json!({"index": "tampered"});

        let broken = verify_chain(&entries).expect_err("an edited entry must break the chain");

        assert_eq!(broken.line, 3, "the third line is the edited one");
        assert_eq!(broken.sequence, 3);
        assert!(
            matches!(broken.kind, ChainBreakKind::HashMismatch { .. }),
            "an edit is a hash mismatch, got {:?}",
            broken.kind
        );
    }

    #[test]
    fn rewriting_an_entry_and_its_own_hash_still_breaks_the_next_link() {
        // The thorough forger: edit the entry, then reseal it so it hashes to
        // itself. The chain still breaks, because the *next* entry was
        // computed from the old hash. This is the property the chain buys.
        let mut entries = chain(5);
        let resealed = AuditEntry::seal(
            InvocationRecord {
                arguments: json!({"index": "tampered"}),
                ..record(3)
            },
            3,
            &entries[1].hash,
        );
        entries[2] = resealed;

        let broken = verify_chain(&entries).expect_err("a resealed entry must still be caught");

        assert_eq!(broken.line, 4, "the break surfaces at the following entry");
        assert!(
            matches!(broken.kind, ChainBreakKind::PrevHashMismatch { .. }),
            "got {:?}",
            broken.kind
        );
    }

    #[test]
    fn a_removed_entry_is_reported_as_a_sequence_gap() {
        let mut entries = chain(5);
        entries.remove(2);

        let broken = verify_chain(&entries).expect_err("a removed entry must break the chain");

        assert_eq!(broken.line, 3);
        assert_eq!(
            broken.kind,
            ChainBreakKind::SequenceGap {
                expected: 3,
                found: 4
            }
        );
    }

    #[test]
    fn reordered_entries_break_the_chain() {
        let mut entries = chain(5);
        entries.swap(1, 3);

        let broken = verify_chain(&entries).expect_err("reordering must break the chain");
        assert_eq!(broken.line, 2);
    }

    #[test]
    fn a_truncated_tail_still_verifies_but_moves_the_head() {
        // Documented, deliberate limitation: only the head reveals truncation.
        let full = chain(5);
        let truncated = &full[..3];

        let head = verify_chain(truncated).expect("a truncated prefix is internally consistent");

        assert_eq!(head.entries, 3);
        assert_ne!(
            head.hash, full[4].hash,
            "the head is what reveals the truncation"
        );
    }

    #[test]
    fn only_the_first_break_is_reported() {
        let mut entries = chain(5);
        entries[1].tool_name = "edited".to_string();
        entries[3].tool_name = "also_edited".to_string();

        let broken = verify_chain(&entries).expect_err("the chain is broken");
        assert_eq!(broken.line, 2, "the earliest break is the one reported");
    }
}
