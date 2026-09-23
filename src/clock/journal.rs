//! The event journal: one timeline, written once, read back in order.
//!
//! One JSON object per line:
//!
//! ```text
//! {"seq":1,"at":"2026-09-23T03:00:00.000000Z","kind":"tick","origin":"socket"}
//! {"seq":2,"at":"2026-09-23T03:30:00.000000Z","kind":"tick","origin":"socket"}
//! ```
//!
//! Ticks are the only kind of entry today. The `kind` tag is there because
//! they will not stay the only kind: replaying a process deterministically
//! means recording every input it could not have computed for itself — model
//! completions, tool results — on this same timeline, in this same order.
//!
//! # Invariants
//!
//! A journal that [`read_journal`] accepts has sequence numbers starting at 1
//! with no gaps, and timestamps that never decrease. Both are checked on read
//! rather than trusted, because a replay that silently skipped an entry or
//! reordered two would diverge from the run it claims to reproduce, and
//! nothing downstream could tell.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::clock::{ClockError, EventTime};

/// Permission bits for a newly created journal: owner read/write only.
///
/// The journal says when the process woke and why, which is an operator's
/// business and nobody else's.
pub const JOURNAL_MODE: u32 = 0o600;

/// Who asked for a tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TickOrigin {
    /// A client on the introspection socket — normally `acton-ai tick`,
    /// fired by a systemd timer.
    Socket,
    /// Code in this process, through
    /// [`ActonAI::tick`](crate::facade::ActonAI::tick).
    InProcess,
}

impl TickOrigin {
    /// The name the journal and the wire use.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Socket => "socket",
            Self::InProcess => "in_process",
        }
    }
}

/// What happened at one point on the timeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum JournalEvent {
    /// Time advanced, and anything that runs on rhythm should look again.
    Tick {
        /// Who asked for it.
        origin: TickOrigin,
    },
}

/// One line of the journal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Position on the timeline, from 1, with no gaps.
    pub seq: u64,
    /// When it happened. Never earlier than the entry before it.
    pub at: EventTime,
    /// What happened.
    #[serde(flatten)]
    pub event: JournalEvent,
}

/// Reads and validates a whole journal.
///
/// # Errors
///
/// Returns a journal I/O error when the file cannot be read, a corruption
/// error naming the line when one cannot be decoded, an out-of-sequence error
/// on a gap or repeat, and a time-went-backwards error when an entry is
/// earlier than the one before it.
pub fn read_journal(path: &Path) -> Result<Vec<JournalEntry>, ClockError> {
    let file = std::fs::File::open(path)
        .map_err(|error| ClockError::journal_io(path, error.to_string()))?;
    let mut entries: Vec<JournalEntry> = Vec::new();

    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|error| ClockError::journal_io(path, error.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: JournalEntry = serde_json::from_str(&line)
            .map_err(|error| ClockError::journal_corrupt(path, line_number, error.to_string()))?;
        check_follows(entries.last(), &entry)?;
        entries.push(entry);
    }

    Ok(entries)
}

/// Checks that `next` may come straight after `previous` on one timeline.
///
/// # Errors
///
/// Out of sequence on a gap or repeat; time went backwards on an earlier
/// timestamp.
pub(crate) fn check_follows(
    previous: Option<&JournalEntry>,
    next: &JournalEntry,
) -> Result<(), ClockError> {
    let expected = previous.map_or(1, |entry| entry.seq + 1);
    if next.seq != expected {
        return Err(ClockError::out_of_sequence(expected, next.seq));
    }
    if let Some(previous) = previous {
        if next.at < previous.at {
            return Err(ClockError::time_went_backwards(previous.at, next.at));
        }
    }
    Ok(())
}

/// Appends entries to a journal file.
///
/// Each append is written and flushed to disk before it returns. Ticks arrive
/// minutes apart, so paying for an `fsync` on every one is cheap, and it is
/// what makes "the tick was acknowledged" mean "the tick will be replayed".
#[derive(Debug, Clone, Default)]
pub struct JournalWriter {
    path: PathBuf,
}

impl JournalWriter {
    /// Opens `path` for appending, creating it if absent, and returns the
    /// entries already in it so the caller can continue the timeline.
    ///
    /// # Errors
    ///
    /// Anything [`read_journal`] can return for an existing file, or a
    /// journal I/O error when the file cannot be created.
    pub fn open(path: &Path) -> Result<(Self, Vec<JournalEntry>), ClockError> {
        let existing = if path.exists() {
            read_journal(path)?
        } else {
            create_journal(path)?;
            Vec::new()
        };
        Ok((
            Self {
                path: path.to_path_buf(),
            },
            existing,
        ))
    }

    /// Where the journal lives.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes one entry and syncs it to disk.
    ///
    /// The caller is responsible for the entry following the last one: the
    /// next sequence number, and a time no earlier than the last entry's.
    ///
    /// # Errors
    ///
    /// A journal I/O error when the write or the sync fails. The entry must
    /// then be treated as not recorded.
    pub fn append(&self, entry: &JournalEntry) -> Result<(), ClockError> {
        let mut line = serde_json::to_string(entry)
            .map_err(|error| ClockError::journal_io(&self.path, error.to_string()))?;
        line.push('\n');

        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(|error| ClockError::journal_io(&self.path, error.to_string()))?;
        file.write_all(line.as_bytes())
            .and_then(|()| file.sync_data())
            .map_err(|error| ClockError::journal_io(&self.path, error.to_string()))
    }
}

/// Creates an empty journal with owner-only permissions, and its directory.
fn create_journal(path: &Path) -> Result<(), ClockError> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|error| ClockError::journal_io(path, error.to_string()))?;
    }

    let mut options = OpenOptions::new();
    options.append(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(JOURNAL_MODE);
    }
    options
        .open(path)
        .map(drop)
        .map_err(|error| ClockError::journal_io(path, error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ClockErrorKind;

    fn at(text: &str) -> EventTime {
        EventTime::parse_rfc3339(text).unwrap()
    }

    fn tick(seq: u64, time: &str) -> JournalEntry {
        JournalEntry {
            seq,
            at: at(time),
            event: JournalEvent::Tick {
                origin: TickOrigin::Socket,
            },
        }
    }

    #[test]
    fn an_entry_is_one_flat_json_object() {
        let json = serde_json::to_string(&tick(1, "2026-09-23T03:00:00Z")).unwrap();
        assert_eq!(
            json,
            r#"{"seq":1,"at":"2026-09-23T03:00:00.000000Z","kind":"tick","origin":"socket"}"#
        );
    }

    #[test]
    fn appended_entries_read_back_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("ticks.jsonl");

        let (writer, existing) = JournalWriter::open(&path).unwrap();
        assert!(existing.is_empty());
        writer.append(&tick(1, "2026-09-23T03:00:00Z")).unwrap();
        writer.append(&tick(2, "2026-09-23T03:30:00Z")).unwrap();

        let (_, reopened) = JournalWriter::open(&path).unwrap();
        assert_eq!(
            reopened,
            vec![
                tick(1, "2026-09-23T03:00:00Z"),
                tick(2, "2026-09-23T03:30:00Z")
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_new_journal_is_private_to_its_owner() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ticks.jsonl");
        JournalWriter::open(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, JOURNAL_MODE);
    }

    #[test]
    fn a_gap_is_refused() {
        let error = check_follows(
            Some(&tick(1, "2026-09-23T03:00:00Z")),
            &tick(3, "2026-09-23T03:30:00Z"),
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            &ClockErrorKind::OutOfSequence {
                expected: 2,
                found: 3
            }
        );
    }

    #[test]
    fn a_journal_must_start_at_one() {
        let error = check_follows(None, &tick(2, "2026-09-23T03:00:00Z")).unwrap_err();
        assert!(matches!(
            error.kind(),
            ClockErrorKind::OutOfSequence { expected: 1, .. }
        ));
    }

    #[test]
    fn time_running_backwards_is_refused_but_equal_times_are_not() {
        let first = tick(1, "2026-09-23T03:00:00Z");
        assert!(check_follows(Some(&first), &tick(2, "2026-09-23T03:00:00Z")).is_ok());
        let error = check_follows(Some(&first), &tick(2, "2026-09-23T02:00:00Z")).unwrap_err();
        assert!(matches!(
            error.kind(),
            ClockErrorKind::TimeWentBackwards { .. }
        ));
    }

    #[test]
    fn a_corrupt_line_is_named() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ticks.jsonl");
        let good = serde_json::to_string(&tick(1, "2026-09-23T03:00:00Z")).unwrap();
        std::fs::write(&path, format!("{good}\n{{\"seq\":2,\"at\":\n")).unwrap();

        let error = read_journal(&path).unwrap_err();
        assert!(matches!(
            error.kind(),
            ClockErrorKind::JournalCorrupt { line: 2, .. }
        ));
    }

    #[test]
    fn an_unknown_kind_is_corrupt_rather_than_skipped() {
        // Skipping an entry a newer build wrote would replay a different
        // timeline from the one recorded, so it is refused instead.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ticks.jsonl");
        std::fs::write(
            &path,
            "{\"seq\":1,\"at\":\"2026-09-23T03:00:00Z\",\"kind\":\"completion\"}\n",
        )
        .unwrap();
        assert!(read_journal(&path).unwrap_err().is_journal_error());
    }

    #[test]
    fn blank_lines_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ticks.jsonl");
        let good = serde_json::to_string(&tick(1, "2026-09-23T03:00:00Z")).unwrap();
        std::fs::write(&path, format!("\n{good}\n\n")).unwrap();
        assert_eq!(read_journal(&path).unwrap().len(), 1);
    }

    #[test]
    fn a_missing_journal_is_an_io_error() {
        let error = read_journal(Path::new("/nonexistent/ticks.jsonl")).unwrap_err();
        assert!(matches!(error.kind(), ClockErrorKind::JournalIo { .. }));
    }
}
