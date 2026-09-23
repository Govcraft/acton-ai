//! Clock error types.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::clock::EventTime;

/// An error raised by the clock, its journal, or a replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClockError {
    /// The specific failure (boxed for size efficiency).
    kind: Box<ClockErrorKind>,
}

/// Specific clock failure modes.
///
/// Marked `#[non_exhaustive]`: the journal will record more than ticks, and
/// each new kind of entry brings new ways for a replay to disagree with it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClockErrorKind {
    /// Text that should have been an RFC 3339 timestamp was not.
    InvalidTime {
        /// The text as found.
        text: String,
        /// What the parser said.
        message: String,
    },
    /// Something tried to move the wall clock.
    NotDriven,
    /// A timeline was asked to move to an earlier time.
    TimeWentBackwards {
        /// Where the timeline stood.
        current: EventTime,
        /// Where it was asked to go.
        requested: EventTime,
    },
    /// The journal could not be opened, read, or written.
    JournalIo {
        /// The journal file.
        path: PathBuf,
        /// What the operating system said.
        message: String,
    },
    /// A journal line could not be decoded.
    JournalCorrupt {
        /// The journal file.
        path: PathBuf,
        /// The 1-based line number.
        line: usize,
        /// What was wrong with it.
        message: String,
    },
    /// An entry's sequence number is not the one the timeline expected next.
    OutOfSequence {
        /// The sequence number the timeline expected.
        expected: u64,
        /// The one the entry carried.
        found: u64,
    },
    /// A replay was attempted on a clock that cannot take one.
    ReplayRefused {
        /// Why.
        reason: String,
    },
}

impl ClockError {
    /// Creates an error from its kind.
    #[must_use]
    pub fn new(kind: ClockErrorKind) -> Self {
        Self {
            kind: Box::new(kind),
        }
    }

    /// The specific failure.
    #[must_use]
    pub fn kind(&self) -> &ClockErrorKind {
        &self.kind
    }

    /// Creates a [`ClockErrorKind::InvalidTime`] error.
    #[must_use]
    pub fn invalid_time(text: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ClockErrorKind::InvalidTime {
            text: text.into(),
            message: message.into(),
        })
    }

    /// Creates a [`ClockErrorKind::NotDriven`] error.
    #[must_use]
    pub fn not_driven() -> Self {
        Self::new(ClockErrorKind::NotDriven)
    }

    /// Creates a [`ClockErrorKind::TimeWentBackwards`] error.
    #[must_use]
    pub fn time_went_backwards(current: EventTime, requested: EventTime) -> Self {
        Self::new(ClockErrorKind::TimeWentBackwards { current, requested })
    }

    /// Creates a [`ClockErrorKind::JournalIo`] error.
    #[must_use]
    pub fn journal_io(path: &Path, message: impl Into<String>) -> Self {
        Self::new(ClockErrorKind::JournalIo {
            path: path.to_path_buf(),
            message: message.into(),
        })
    }

    /// Creates a [`ClockErrorKind::JournalCorrupt`] error.
    #[must_use]
    pub fn journal_corrupt(path: &Path, line: usize, message: impl Into<String>) -> Self {
        Self::new(ClockErrorKind::JournalCorrupt {
            path: path.to_path_buf(),
            line,
            message: message.into(),
        })
    }

    /// Creates a [`ClockErrorKind::OutOfSequence`] error.
    #[must_use]
    pub fn out_of_sequence(expected: u64, found: u64) -> Self {
        Self::new(ClockErrorKind::OutOfSequence { expected, found })
    }

    /// Creates a [`ClockErrorKind::ReplayRefused`] error.
    #[must_use]
    pub fn replay_refused(reason: impl Into<String>) -> Self {
        Self::new(ClockErrorKind::ReplayRefused {
            reason: reason.into(),
        })
    }

    /// Whether the journal on disk is at fault, as opposed to the caller.
    #[must_use]
    pub fn is_journal_error(&self) -> bool {
        matches!(
            *self.kind,
            ClockErrorKind::JournalIo { .. } | ClockErrorKind::JournalCorrupt { .. }
        )
    }
}

impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind.as_ref() {
            ClockErrorKind::InvalidTime { text, message } => write!(
                f,
                "'{text}' is not an RFC 3339 timestamp ({message}); write it as \
                 2026-01-01T00:00:00Z"
            ),
            ClockErrorKind::NotDriven => write!(
                f,
                "the wall clock cannot be moved; build the runtime with \
                 TimeSource::driven(..) to control time"
            ),
            ClockErrorKind::TimeWentBackwards { current, requested } => write!(
                f,
                "time cannot go backwards: the timeline is at {current} and was asked to \
                 move to {requested}"
            ),
            ClockErrorKind::JournalIo { path, message } => write!(
                f,
                "could not use the tick journal at {}: {message}; check the path exists and \
                 is writable by this user",
                path.display()
            ),
            ClockErrorKind::JournalCorrupt {
                path,
                line,
                message,
            } => write!(
                f,
                "the tick journal at {} is unreadable at line {line}: {message}; a journal \
                 is append-only, so this usually means it was edited by hand or truncated \
                 mid-write",
                path.display()
            ),
            ClockErrorKind::OutOfSequence { expected, found } => write!(
                f,
                "journal entry {found} arrived where entry {expected} was expected; entries \
                 must be replayed in order, from the start, with none missing"
            ),
            ClockErrorKind::ReplayRefused { reason } => {
                write!(f, "replay refused: {reason}")
            }
        }
    }
}

impl std::error::Error for ClockError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_has_an_actionable_message() {
        let t1 = EventTime::parse_rfc3339("2026-01-02T00:00:00Z").unwrap();
        let t0 = EventTime::parse_rfc3339("2026-01-01T00:00:00Z").unwrap();
        let path = Path::new("/var/lib/agent/ticks.jsonl");

        let cases = [
            (ClockError::invalid_time("soon", "bad"), "RFC 3339"),
            (ClockError::not_driven(), "TimeSource::driven"),
            (ClockError::time_went_backwards(t1, t0), "backwards"),
            (ClockError::journal_io(path, "denied"), "ticks.jsonl"),
            (ClockError::journal_corrupt(path, 7, "eof"), "line 7"),
            (ClockError::out_of_sequence(3, 5), "entry 5"),
            (ClockError::replay_refused("journaling"), "journaling"),
        ];
        for (error, needle) in cases {
            let text = error.to_string();
            assert!(text.contains(needle), "{text:?} should mention {needle:?}");
        }
    }

    #[test]
    fn journal_errors_are_told_apart_from_caller_errors() {
        let path = Path::new("j");
        assert!(ClockError::journal_io(path, "x").is_journal_error());
        assert!(ClockError::journal_corrupt(path, 1, "x").is_journal_error());
        assert!(!ClockError::out_of_sequence(1, 2).is_journal_error());
        assert!(!ClockError::not_driven().is_journal_error());
    }

    #[test]
    fn errors_clone_and_compare() {
        let error = ClockError::out_of_sequence(1, 2);
        assert_eq!(error.clone(), error);
        assert_ne!(error, ClockError::out_of_sequence(1, 3));
    }
}
