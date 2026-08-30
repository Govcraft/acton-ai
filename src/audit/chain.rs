//! Walking a chain and reporting where it breaks.
//!
//! Pure: entries in, verdict out. The CLI reads the file and hands the parsed
//! entries here, which is what lets the interesting cases — a tampered middle
//! entry, a reordering, a truncation, a relabelled trail — be tested without
//! touching a disk.
//!
//! The walk is one step, [`verify_next`], folded over the entries. Exposing
//! the step is deliberate: a verifier that holds the head somewhere else — a
//! control plane checking entries as they arrive, one at a time — applies
//! exactly the rule the file walk applies, rather than a reimplementation
//! that could drift.

use crate::audit::entry::{AuditEntry, GENESIS_HASH};
use crate::types::TrailId;
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
    /// The identity the chain is sealed under, once one has been seen.
    ///
    /// `None` for an empty chain and for a chain written entirely before
    /// trails had an identity. Once an entry carries an identity, every
    /// later entry must carry the same one, and the head reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trail_id: Option<TrailId>,
}

impl ChainHead {
    /// The head of a chain with nothing in it yet.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            sequence: 0,
            hash: GENESIS_HASH.to_string(),
            entries: 0,
            trail_id: None,
        }
    }

    /// Whether the chain holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries == 0
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
    /// The entry names a different trail than the chain it sits in, or no
    /// trail at all where the chain already has one: an entry from another
    /// trail was spliced in, or the trail was relabelled.
    TrailMismatch {
        /// The identity the chain is sealed under, rendered; `None` never
        /// occurs today (a chain without an identity accepts anything) but
        /// is kept so the two sides read alike.
        expected: Option<String>,
        /// The identity this entry carries, rendered; `None` for an entry
        /// with no identity at all.
        found: Option<String>,
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
            Self::TrailMismatch { expected, found } => write!(
                f,
                "the chain is sealed under trail {} but the entry carries {} \
                 — an entry from another trail was spliced in, or the trail was relabelled",
                expected.as_deref().unwrap_or("unidentified"),
                found.as_deref().unwrap_or("unidentified"),
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

/// Checks that `entry` is the next link after `head`, and returns the head it
/// advances to.
///
/// Four things are checked, in the order that gives the most specific
/// diagnosis first: that the sequence number is the next one, that the entry
/// points at the entry before it, that the entry's contents still hash to the
/// hash it carries, and that the entry belongs to the trail the chain is
/// sealed under.
///
/// The identity rule: an entry without an identity is accepted only while
/// the chain has not seen one — the prefix written before trails had
/// identities. Once an identity has been seen, every later entry must carry
/// exactly that identity. A chain can gain an identity once and never change
/// or lose it.
///
/// `line` is the 1-based position of the entry in the file, reported on a
/// break; a verifier without a file passes the sequence number.
///
/// # Errors
///
/// Returns the [`ChainBreak`] describing why `entry` does not follow `head`.
pub fn verify_next(
    head: &ChainHead,
    entry: &AuditEntry,
    line: usize,
) -> Result<ChainHead, ChainBreak> {
    let broken = |kind| ChainBreak {
        line,
        sequence: entry.sequence,
        kind,
    };

    let expected_sequence = head.sequence.saturating_add(1);
    if entry.sequence != expected_sequence {
        return Err(broken(ChainBreakKind::SequenceGap {
            expected: expected_sequence,
            found: entry.sequence,
        }));
    }

    if entry.prev_hash != head.hash {
        return Err(broken(ChainBreakKind::PrevHashMismatch {
            expected: head.hash.clone(),
            found: entry.prev_hash.clone(),
        }));
    }

    let recomputed = entry.recompute_hash();
    if recomputed != entry.hash {
        return Err(broken(ChainBreakKind::HashMismatch {
            expected: recomputed,
            found: entry.hash.clone(),
        }));
    }

    let trail_id = match (&head.trail_id, &entry.trail_id) {
        // Legacy prefix: nothing identified yet, and this entry does not
        // change that.
        (None, None) => None,
        // The first identified entry: the chain takes on its identity.
        (None, Some(found)) => Some(found.clone()),
        (Some(expected), Some(found)) if expected == found => Some(expected.clone()),
        (Some(expected), found) => {
            return Err(broken(ChainBreakKind::TrailMismatch {
                expected: Some(expected.to_string()),
                found: found.as_ref().map(ToString::to_string),
            }));
        }
    };

    Ok(ChainHead {
        sequence: entry.sequence,
        hash: entry.hash.clone(),
        entries: head.entries.saturating_add(1),
        trail_id,
    })
}

/// Walks a chain from the genesis hash and reports the first break.
///
/// A fold of [`verify_next`] over the entries, starting from
/// [`ChainHead::empty`]; see there for what is checked.
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
    entries
        .iter()
        .enumerate()
        .try_fold(ChainHead::empty(), |head, (index, entry)| {
            verify_next(&head, entry, index + 1)
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
            user: None,
            turn_id: TurnId::new(),
            tool_call_id: format!("toolu_{index}"),
            tool_name: format!("tool_{index}"),
            arguments: json!({"index": index}),
            outcome: AuditOutcome::Success {
                summary: "ok".to_string(),
            },
            decision: AuditDecision::approved(Decider::NoPolicy),
            duration_ms: index,
            response_size_bytes: Some(4),
            resumed: false,
        }
    }

    /// Builds an intact chain of `count` legacy (unidentified) entries.
    fn chain(count: u64) -> Vec<AuditEntry> {
        chain_under(count, &[])
    }

    /// Builds an intact chain of `count` entries, where entry `i` (1-based)
    /// carries `trails[i - 1]` — `None` past the end of the slice.
    fn chain_under(count: u64, trails: &[Option<&TrailId>]) -> Vec<AuditEntry> {
        let mut prev = GENESIS_HASH.to_string();
        let mut entries = Vec::new();
        for index in 1..=count {
            let trail = trails.get(index as usize - 1).copied().flatten();
            let entry = AuditEntry::seal(record(index), index, &prev, trail);
            prev.clone_from(&entry.hash);
            entries.push(entry);
        }
        entries
    }

    #[test]
    fn an_empty_chain_verifies_to_the_genesis_head() {
        let head = verify_chain(&[]).expect("an empty chain is intact");
        assert_eq!(head, ChainHead::empty());
        assert!(head.is_empty());
        assert_eq!(head.trail_id, None);
    }

    #[test]
    fn an_intact_chain_verifies() {
        let entries = chain(5);
        let head = verify_chain(&entries).expect("an untouched chain must verify");

        assert_eq!(head.entries, 5);
        assert_eq!(head.sequence, 5);
        assert_eq!(head.hash, entries[4].hash);
        assert!(!head.is_empty());
        assert_eq!(head.trail_id, None, "a legacy chain has no identity");
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
        entries[2].arguments = Some(json!({"index": "tampered"}));

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
            None,
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
        entries[1].tool_name = Some("edited".to_string());
        entries[3].tool_name = Some("also_edited".to_string());

        let broken = verify_chain(&entries).expect_err("the chain is broken");
        assert_eq!(broken.line, 2, "the earliest break is the one reported");
    }

    #[test]
    fn verifying_one_step_at_a_time_reaches_the_same_head_as_the_walk() {
        let trail = TrailId::new();
        let entries = chain_under(4, &[Some(&trail); 4]);

        let mut head = ChainHead::empty();
        for (index, entry) in entries.iter().enumerate() {
            head = verify_next(&head, entry, index + 1).expect("each link holds");
            assert_eq!(head.sequence, entry.sequence);
            assert_eq!(head.entries, entry.sequence);
            assert_eq!(head.trail_id.as_ref(), Some(&trail));
        }

        assert_eq!(head, verify_chain(&entries).unwrap());
    }

    #[test]
    fn an_identified_chain_reports_its_trail_in_the_head() {
        let trail = TrailId::new();
        let entries = chain_under(3, &[Some(&trail); 3]);

        let head = verify_chain(&entries).expect("an identified chain verifies");
        assert_eq!(head.trail_id, Some(trail));
    }

    #[test]
    fn a_legacy_prefix_followed_by_identified_entries_verifies() {
        // A trail written before identities existed, appended to by a build
        // that has one: the chain takes on the identity where it first
        // appears and keeps verifying.
        let trail = TrailId::new();
        let entries = chain_under(4, &[None, None, Some(&trail), Some(&trail)]);

        let head = verify_chain(&entries).expect("gaining an identity is allowed once");
        assert_eq!(head.trail_id, Some(trail));
        assert_eq!(head.entries, 4);
    }

    #[test]
    fn an_identified_chain_cannot_lose_its_identity() {
        let trail = TrailId::new();
        let entries = chain_under(3, &[Some(&trail), Some(&trail), None]);

        let broken =
            verify_chain(&entries).expect_err("dropping the identity must break the chain");

        assert_eq!(broken.line, 3);
        assert_eq!(
            broken.kind,
            ChainBreakKind::TrailMismatch {
                expected: Some(trail.to_string()),
                found: None,
            }
        );
        assert!(broken.to_string().contains("unidentified"), "{broken}");
    }

    #[test]
    fn two_different_identities_break_the_chain() {
        let ours = TrailId::new();
        let theirs = TrailId::new();
        let entries = chain_under(3, &[Some(&ours), Some(&ours), Some(&theirs)]);

        let broken = verify_chain(&entries).expect_err("a second identity must break the chain");

        assert_eq!(broken.line, 3);
        assert_eq!(
            broken.kind,
            ChainBreakKind::TrailMismatch {
                expected: Some(ours.to_string()),
                found: Some(theirs.to_string()),
            }
        );
        let text = broken.to_string();
        assert!(text.contains(&ours.to_string()), "{text}");
        assert!(text.contains(&theirs.to_string()), "{text}");
    }

    #[test]
    fn relabelling_a_whole_trail_is_caught_at_the_first_entry() {
        // The forger who rewrites every entry's trail_id to pass one trail's
        // evidence off as another's: the identity is in the hash, so the very
        // first entry no longer hashes to itself.
        let mut entries = chain_under(3, &[Some(&TrailId::new()); 3]);
        let other = TrailId::new();
        for entry in &mut entries {
            entry.trail_id = Some(other.clone());
        }

        let broken = verify_chain(&entries).expect_err("a relabelled trail must not verify");
        assert_eq!(broken.line, 1);
        assert!(matches!(broken.kind, ChainBreakKind::HashMismatch { .. }));
    }

    #[test]
    fn the_head_serializes_without_an_absent_identity() {
        // Consumers reading the head as JSON — the CLI report, an embedder's
        // status endpoint — see the same shape they did before identities.
        let json = serde_json::to_value(ChainHead::empty()).unwrap();
        assert!(json.get("trail_id").is_none(), "{json}");

        let trail = TrailId::new();
        let head = ChainHead {
            trail_id: Some(trail.clone()),
            ..ChainHead::empty()
        };
        let json = serde_json::to_value(&head).unwrap();
        assert_eq!(json["trail_id"], json!(trail.to_string()));
        assert_eq!(serde_json::from_value::<ChainHead>(json).unwrap(), head);
    }
}
