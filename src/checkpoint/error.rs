//! Checkpoint error types.
//!
//! Custom error types for checkpoint decoding and resume planning. Storage
//! failures are not here — those stay
//! [`PersistenceError`](crate::memory::PersistenceError), because "the
//! database refused the write" and "this checkpoint does not describe the turn
//! you are asking to resume" are different problems with different fixes.

use crate::types::CheckpointId;
use std::fmt;

/// An error raised while decoding a checkpoint or planning a resume from one.
///
/// Carries the checkpoint it concerns whenever one is known, because the first
/// thing an operator does with a refused resume is go and look at the record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointError {
    /// The checkpoint the error concerns, when the caller named one.
    checkpoint_id: Option<CheckpointId>,
    /// The specific failure (boxed for size efficiency).
    kind: Box<CheckpointErrorKind>,
}

/// Specific checkpoint failure modes.
///
/// Marked `#[non_exhaustive]`: the planner gains refusal reasons as it learns
/// to reject more, and a downstream `match` should not break each time it
/// does. Match on the variants you handle and add a `_` arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CheckpointErrorKind {
    /// The turn's inputs no longer match the ones the checkpoint was written
    /// for, so resuming would splice two different turns together.
    InputsChanged {
        /// The fingerprint stored on the checkpoint.
        expected: String,
        /// The fingerprint of the turn being started now.
        actual: String,
    },
    /// The checkpoint already used every round the turn is allowed.
    RoundsExhausted {
        /// Rounds already spent according to the checkpoint.
        rounds_completed: usize,
        /// The ceiling the resuming turn was configured with.
        max_tool_rounds: usize,
    },
    /// The record was written by a different, unsupported checkpoint format.
    VersionMismatch {
        /// The format version found on the record.
        found: u32,
        /// The format version this build understands.
        supported: u32,
    },
    /// A stored column could not be decoded into the value it should hold.
    Corrupt {
        /// The column or field that could not be decoded.
        field: &'static str,
        /// What went wrong.
        message: String,
    },
    /// The record claims the turn finished but is missing what it finished
    /// with, so there is nothing to replay.
    IncompleteCompletion {
        /// The part of the finished answer that is absent.
        missing: &'static str,
    },
    /// An operator policy already declined to resume this turn. The record is
    /// kept as evidence of that outcome and is never picked up again.
    Abandoned,
}

impl CheckpointError {
    /// Creates an error that does not name a particular checkpoint.
    #[must_use]
    pub fn new(kind: CheckpointErrorKind) -> Self {
        Self {
            checkpoint_id: None,
            kind: Box::new(kind),
        }
    }

    /// Creates an error against a specific checkpoint.
    #[must_use]
    pub fn for_checkpoint(id: CheckpointId, kind: CheckpointErrorKind) -> Self {
        Self {
            checkpoint_id: Some(id),
            kind: Box::new(kind),
        }
    }

    /// Attaches a checkpoint ID to an error raised before one was known.
    #[must_use]
    pub fn with_checkpoint(mut self, id: CheckpointId) -> Self {
        self.checkpoint_id = Some(id);
        self
    }

    /// The checkpoint this error concerns, if one was named.
    #[must_use]
    pub fn checkpoint_id(&self) -> Option<&CheckpointId> {
        self.checkpoint_id.as_ref()
    }

    /// The specific failure.
    #[must_use]
    pub fn kind(&self) -> &CheckpointErrorKind {
        &self.kind
    }

    /// Creates an [`CheckpointErrorKind::InputsChanged`] error.
    #[must_use]
    pub fn inputs_changed(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::new(CheckpointErrorKind::InputsChanged {
            expected: expected.into(),
            actual: actual.into(),
        })
    }

    /// Creates a [`CheckpointErrorKind::RoundsExhausted`] error.
    #[must_use]
    pub fn rounds_exhausted(rounds_completed: usize, max_tool_rounds: usize) -> Self {
        Self::new(CheckpointErrorKind::RoundsExhausted {
            rounds_completed,
            max_tool_rounds,
        })
    }

    /// Creates a [`CheckpointErrorKind::VersionMismatch`] error.
    #[must_use]
    pub fn version_mismatch(found: u32, supported: u32) -> Self {
        Self::new(CheckpointErrorKind::VersionMismatch { found, supported })
    }

    /// Creates a [`CheckpointErrorKind::Corrupt`] error.
    #[must_use]
    pub fn corrupt(field: &'static str, message: impl Into<String>) -> Self {
        Self::new(CheckpointErrorKind::Corrupt {
            field,
            message: message.into(),
        })
    }

    /// Creates a [`CheckpointErrorKind::IncompleteCompletion`] error.
    #[must_use]
    pub fn incomplete_completion(missing: &'static str) -> Self {
        Self::new(CheckpointErrorKind::IncompleteCompletion { missing })
    }

    /// Creates a [`CheckpointErrorKind::Abandoned`] error.
    #[must_use]
    pub fn abandoned() -> Self {
        Self::new(CheckpointErrorKind::Abandoned)
    }
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref id) = self.checkpoint_id {
            write!(f, "checkpoint {id}: ")?;
        }
        write!(f, "{}", self.kind)
    }
}

impl fmt::Display for CheckpointErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputsChanged { expected, actual } => write!(
                f,
                "the turn's inputs changed since the checkpoint was written \
                 (checkpoint {expected}, this turn {actual}); resume with the \
                 same prompt, tools, and provider, or start a new checkpoint"
            ),
            Self::RoundsExhausted {
                rounds_completed,
                max_tool_rounds,
            } => write!(
                f,
                "the checkpoint already spent {rounds_completed} of {max_tool_rounds} \
                 tool rounds; raise max_tool_rounds or start a new checkpoint"
            ),
            Self::VersionMismatch { found, supported } => write!(
                f,
                "the checkpoint was written in format version {found}, but this build \
                 understands version {supported}; start a new checkpoint"
            ),
            Self::Corrupt { field, message } => {
                write!(f, "the checkpoint's `{field}` could not be read: {message}")
            }
            Self::IncompleteCompletion { missing } => write!(
                f,
                "the checkpoint is marked finished but has no `{missing}` to replay; \
                 start a new checkpoint"
            ),
            Self::Abandoned => write!(
                f,
                "the turn this checkpoint belongs to was abandoned by operator policy; \
                 the record is kept as evidence, so start a new checkpoint"
            ),
        }
    }
}

impl std::error::Error for CheckpointError {}
impl std::error::Error for CheckpointErrorKind {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inputs_changed_display_names_both_fingerprints() {
        let text = CheckpointError::inputs_changed("aaaa", "bbbb").to_string();
        assert!(text.contains("aaaa"), "{text}");
        assert!(text.contains("bbbb"), "{text}");
    }

    #[test]
    fn display_leads_with_the_checkpoint_when_one_is_known() {
        let id = CheckpointId::new();
        let error = CheckpointError::rounds_exhausted(8, 8).with_checkpoint(id.clone());
        let text = error.to_string();
        assert!(text.starts_with(&format!("checkpoint {id}: ")), "{text}");
    }

    #[test]
    fn display_omits_the_prefix_when_no_checkpoint_is_named() {
        let text = CheckpointError::corrupt("messages", "trailing comma").to_string();
        assert!(!text.starts_with("checkpoint "), "{text}");
        assert!(text.contains("messages"), "{text}");
    }

    #[test]
    fn rounds_exhausted_suggests_a_way_forward() {
        let text = CheckpointError::rounds_exhausted(4, 4).to_string();
        assert!(text.contains("max_tool_rounds"), "{text}");
    }

    #[test]
    fn version_mismatch_reports_both_versions() {
        let text = CheckpointError::version_mismatch(7, 1).to_string();
        assert!(text.contains('7'), "{text}");
        assert!(text.contains('1'), "{text}");
    }

    #[test]
    fn an_abandoned_error_says_the_record_is_evidence() {
        let text = CheckpointError::abandoned().to_string();
        assert!(text.contains("abandoned"), "{text}");
        assert!(text.contains("new checkpoint"), "{text}");
    }

    #[test]
    fn incomplete_completion_names_the_missing_part() {
        let text = CheckpointError::incomplete_completion("final_text").to_string();
        assert!(text.contains("final_text"), "{text}");
    }

    #[test]
    fn kind_is_readable_without_the_wrapper() {
        let error = CheckpointError::for_checkpoint(
            CheckpointId::new(),
            CheckpointErrorKind::Corrupt {
                field: "status",
                message: "unknown".to_string(),
            },
        );
        assert!(matches!(
            error.kind(),
            CheckpointErrorKind::Corrupt {
                field: "status",
                ..
            }
        ));
        assert!(error.checkpoint_id().is_some());
    }
}
