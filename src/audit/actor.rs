//! The audit actor: the one writer that owns the chain.
//!
//! A hash chain has exactly one piece of mutable state — the head — and it has
//! to be advanced and written in the same order every time. That is a state
//! owner, so it is an actor, and it is the only thing that ever appends to the
//! file. Within the process no lock appears because there is nothing to
//! share: the head has one owner and everyone else sends messages.
//!
//! # One writer per file, across processes too
//!
//! The one lock in this module is about *other processes*. Two processes
//! appending to the same trail would each seal entries against their own idea
//! of the head and interleave lines that both claim the same sequence: a
//! forked chain that `audit verify` reports as broken after the fact. So the
//! trail is claimed with an exclusive advisory lock ([`claim_trail`]) before
//! its head is read, and the lock is held for the actor's lifetime — the
//! locked descriptor lives in [`AuditLog`] itself. A second opener is refused
//! at spawn with [`TrailClaimError::Busy`], naming the holder where the
//! platform can tell us. The kernel drops the lock when the descriptor closes,
//! which covers a clean shutdown and a `SIGKILL` alike; there is no pid file
//! to go stale.
//!
//! The lock conflicts only with other lock takers. The actor's own appends and
//! read-only verification of a live trail are unaffected.

use crate::audit::chain::ChainHead;
use crate::audit::config::AuditConfig;
use crate::audit::entry::{AuditEntry, InvocationRecord};
use acton_reactive::prelude::tokio::io::AsyncWriteExt;
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{File, TryLockError};
use std::path::{Path, PathBuf};

/// Why the audit trail could not be claimed for exclusive writing.
#[derive(Debug)]
#[non_exhaustive]
pub enum TrailClaimError {
    /// Another process holds the trail's exclusive lock.
    Busy {
        /// The trail that is already owned.
        path: PathBuf,
        /// The holder's process id, when the platform exposes it
        /// (`/proc/locks` on Linux); `None` means "another process, unknown
        /// which".
        holder_pid: Option<u32>,
    },
    /// The trail could not be opened or locked at all.
    Io {
        /// The trail that could not be claimed.
        path: PathBuf,
        /// What the operating system said.
        source: std::io::Error,
    },
}

impl TrailClaimError {
    /// The trail this error is about.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            Self::Busy { path, .. } | Self::Io { path, .. } => path,
        }
    }

    /// Which config field to blame when this becomes a configuration error.
    const FIELD: &'static str = "audit.path";
}

impl fmt::Display for TrailClaimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy { path, holder_pid } => {
                write!(
                    f,
                    "the audit trail at {} is already owned by another process ",
                    path.display()
                )?;
                match holder_pid {
                    Some(pid) => write!(f, "(pid {pid} holds its exclusive lock)")?,
                    None => write!(f, "(its exclusive lock is held)")?,
                }
                write!(
                    f,
                    "; a hash chain has exactly one writer. Stop the other process, or point \
                     this one at a different trail"
                )
            }
            Self::Io { path, source } => {
                write!(
                    f,
                    "could not claim the audit trail at {}: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for TrailClaimError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Busy { .. } => None,
            Self::Io { source, .. } => Some(source),
        }
    }
}

impl From<TrailClaimError> for crate::error::ActonAIError {
    fn from(error: TrailClaimError) -> Self {
        Self::configuration(TrailClaimError::FIELD, error.to_string())
    }
}

/// Takes the exclusive advisory lock on the trail.
///
/// The returned handle *is* the lock: dropping it, or the process dying,
/// releases it. Opened in append mode so the claim can never truncate what is
/// already on disk, and created if missing so a first run claims the trail it
/// is about to start.
///
/// # Errors
///
/// [`TrailClaimError::Busy`] if another process already holds the lock;
/// [`TrailClaimError::Io`] if the file cannot be opened or locked.
pub fn claim_trail(path: &Path) -> Result<File, TrailClaimError> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|source| TrailClaimError::Io {
            path: path.to_path_buf(),
            source,
        })?;

    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(TrailClaimError::Busy {
            path: path.to_path_buf(),
            holder_pid: lock_holder_pid(&file),
        }),
        Err(TryLockError::Error(source)) => Err(TrailClaimError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// Best-effort: which process holds the lock on this file.
///
/// Linux publishes every held lock in `/proc/locks`, keyed by device and
/// inode. Anywhere that table does not exist, or the file cannot be stat'ed,
/// the answer is simply "unknown" — the refusal stands either way.
#[cfg(target_os = "linux")]
fn lock_holder_pid(file: &File) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;

    let meta = file.metadata().ok()?;
    let locks = std::fs::read_to_string("/proc/locks").ok()?;
    find_lock_holder(&locks, meta.dev(), meta.ino())
}

#[cfg(not(target_os = "linux"))]
fn lock_holder_pid(_file: &File) -> Option<u32> {
    None
}

/// Finds the pid holding a lock on `(dev, inode)` in `/proc/locks` text.
///
/// Each line reads `N: FLOCK ADVISORY WRITE pid MAJ:MIN:INODE start end`; the
/// device is encoded as major and minor in hex. Pure, so the parser is tested
/// against a captured table rather than a live kernel.
fn find_lock_holder(locks: &str, dev: u64, inode: u64) -> Option<u32> {
    locks.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        let pid = fields.nth(4)?.parse::<u32>().ok()?;
        let mut location = fields.next()?.split(':');
        let major = u64::from_str_radix(location.next()?, 16).ok()?;
        let minor = u64::from_str_radix(location.next()?, 16).ok()?;
        let ino = location.next()?.parse::<u64>().ok()?;
        (ino == inode && device_matches(dev, major, minor)).then_some(pid)
    })
}

/// Whether a `stat` device number names the same device as `/proc/locks`'
/// `major:minor` pair, using the kernel's `makedev` layout.
fn device_matches(dev: u64, major: u64, minor: u64) -> bool {
    let stat_major = ((dev >> 32) & 0xffff_f000) | ((dev >> 8) & 0xfff);
    let stat_minor = ((dev >> 12) & 0xffff_ff00) | (dev & 0xff);
    stat_major == major && stat_minor == minor
}

/// Asks the audit log where its chain currently ends.
///
/// Also the barrier that makes audited flows testable without sleeping:
/// mailboxes are FIFO, so a reply to this proves every [`RecordInvocation`]
/// sent earlier has already been folded and written.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct GetChainHead;

impl Request for GetChainHead {
    type Response = ChainHead;
}

/// One tool invocation to append to the trail.
///
/// The arguments must already be redacted: redaction happens at the boundary
/// so a secret never enters this actor's state in the first place.
#[acton_message]
pub struct RecordInvocation {
    /// Everything about the invocation except its place in the chain.
    pub record: InvocationRecord,
}

impl RecordInvocation {
    /// Wraps a record for sending.
    #[must_use]
    pub fn new(record: InvocationRecord) -> Self {
        Self { record }
    }
}

/// State owned by the audit log.
///
/// `#[acton_actor]` derives `Default`, which would leave `prev_hash` as the
/// empty string — a predecessor that cannot exist. [`AuditLog::spawn`] is the
/// only constructor and always seeds the head, either from the genesis hash or
/// from the end of an existing file.
#[acton_actor]
pub struct AuditLog {
    /// Where entries are appended.
    path: PathBuf,
    /// The hash of the last entry written, or the genesis hash.
    prev_hash: String,
    /// How many entries have been written.
    sequence: u64,
    /// The exclusive lock on the trail. Held, not used: its presence in the
    /// model ties the lock's lifetime to the actor's, so it is released when
    /// the actor stops and never before. `None` only in unit tests that
    /// exercise sealing without a file.
    lock: Option<File>,
}

impl AuditLog {
    /// Spawns the audit log over the configured file.
    ///
    /// Resumes an existing file rather than starting a fresh chain: the head
    /// is read once at spawn so a restarted process appends to the chain it
    /// already has instead of silently starting a second one in the same file.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if another process already owns the
    /// trail (see [`claim_trail`]), if the existing file cannot be read, or if
    /// its chain does not verify — refusing to start is the right answer,
    /// because appending to a forked or broken chain would bury the evidence.
    pub async fn spawn(
        runtime: &mut ActorRuntime,
        config: &AuditConfig,
    ) -> Result<ActorHandle, crate::error::ActonAIError> {
        // Claimed before the head is read, so no other writer can append
        // between the read and the first seal: the head we resume from is
        // exact for as long as we hold the lock.
        let lock = claim_trail(config.path())?;
        let head = read_head(config.path()).await?;

        let mut builder = runtime.new_actor_with_name::<AuditLog>("audit_log".to_string());

        // Installed on the idle builder rather than sent as a message, so the
        // chain head is in place before the actor can receive anything.
        builder.model.path = config.path().to_path_buf();
        builder.model.prev_hash = head.hash;
        builder.model.sequence = head.sequence;
        builder.model.lock = Some(lock);

        configure_handlers(&mut builder);

        Ok(builder.start().await)
    }

    /// Seals the next entry and advances the head. Pure bookkeeping.
    fn seal_next(&mut self, record: InvocationRecord) -> AuditEntry {
        let sequence = self.sequence.saturating_add(1);
        let entry = AuditEntry::seal(record, sequence, &self.prev_hash);

        self.sequence = sequence;
        self.prev_hash.clone_from(&entry.hash);

        entry
    }

    /// The chain head as it currently stands.
    fn head(&self) -> ChainHead {
        ChainHead {
            sequence: self.sequence,
            hash: self.prev_hash.clone(),
            entries: self.sequence,
        }
    }
}

/// Wires the audit log's two handlers.
fn configure_handlers(builder: &mut ManagedActor<Idle, AuditLog>) {
    // `mutate_on`, and the write lives in the returned future deliberately.
    // A mutable handler's future is awaited inline before the actor takes its
    // next message, so appends happen in exactly the order the entries were
    // sealed. For a hash chain that serialization is the requirement, not a
    // cost: two concurrent appends would interleave lines that claim to
    // follow each other.
    builder.mutate_on::<RecordInvocation>(|actor, envelope| {
        let entry = actor.model.seal_next(envelope.message().record.clone());
        // Cloned out of the model before the async block: the closure is `Fn`
        // and cannot hold a borrow of the model across the await.
        let path = actor.model.path.clone();

        Reply::pending(async move {
            if let Err(error) = append_entry(&path, &entry).await {
                // A failed append must not fail the turn — the tool already
                // ran, and refusing to continue would not un-run it. It is
                // logged at error level because a compliance deployment needs
                // to notice, and the chain head still advances so the gap is
                // visible to `audit verify` rather than silently healed.
                tracing::error!(
                    path = %path.display(),
                    sequence = entry.sequence,
                    %error,
                    "could not append to the audit trail",
                );
            }
        })
    });

    // A read, so several asks can be served at once and none of them blocks
    // the appends queued ahead of them.
    builder.act_on::<GetChainHead>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let head = actor.model.head();
        Reply::pending(async move {
            reply.send(head).await;
        })
    });
}

/// Appends one entry as a JSONL line.
///
/// Opens in append mode per write. That costs a syscall and buys the property
/// the file is for: nothing this process holds can rewind or overwrite what is
/// already on disk, and a crash between entries leaves a shorter valid chain
/// rather than a corrupt one.
async fn append_entry(path: &Path, entry: &AuditEntry) -> Result<(), std::io::Error> {
    let mut line = entry
        .to_jsonl()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    line.push('\n');

    let mut file = acton_reactive::prelude::tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;

    file.write_all(line.as_bytes()).await?;
    file.flush().await
}

/// Reads an existing trail and returns where its chain ends.
///
/// A missing file is an empty chain, not an error: that is simply the first
/// run.
async fn read_head(path: &Path) -> Result<ChainHead, crate::error::ActonAIError> {
    let entries = match crate::audit::read_entries(path).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(ChainHead::empty()),
        Err(error) => {
            return Err(crate::error::ActonAIError::configuration(
                "audit.path",
                format!(
                    "could not read the existing audit trail at {}: {error}",
                    path.display()
                ),
            ))
        }
    };

    crate::audit::chain::verify_chain(&entries).map_err(|broken| {
        crate::error::ActonAIError::configuration(
            "audit.path",
            format!(
                "the existing audit trail at {} does not verify ({broken}); refusing to append \
                 to a broken chain — move the file aside after investigating it. Two entries \
                 claiming the same sequence after one hash is the signature of two concurrent \
                 writers; the trail lock now refuses that at startup",
                path.display()
            ),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::entry::{AuditDecision, AuditOutcome, GENESIS_HASH};
    use crate::policy::Decider;
    use crate::types::{CorrelationId, TurnId};
    use serde_json::json;

    fn record(name: &str) -> InvocationRecord {
        InvocationRecord {
            timestamp: "2026-08-19T12:00:00Z".to_string(),
            correlation_id: CorrelationId::new(),
            conversation_id: None,
            user: None,
            turn_id: TurnId::new(),
            tool_call_id: "toolu_01".to_string(),
            tool_name: name.to_string(),
            arguments: json!({"value": 1}),
            outcome: AuditOutcome::Success {
                summary: "ok".to_string(),
            },
            decision: AuditDecision::approved(Decider::NoPolicy),
            duration_ms: 1,
            response_size_bytes: Some(4),
            resumed: false,
        }
    }

    #[test]
    fn sealing_advances_the_head_and_links_each_entry_to_the_last() {
        let mut log = AuditLog {
            path: PathBuf::from("/dev/null"),
            prev_hash: GENESIS_HASH.to_string(),
            sequence: 0,
            lock: None,
        };

        let first = log.seal_next(record("a"));
        assert_eq!(first.sequence, 1);
        assert_eq!(first.prev_hash, GENESIS_HASH);

        let second = log.seal_next(record("b"));
        assert_eq!(second.sequence, 2);
        assert_eq!(
            second.prev_hash, first.hash,
            "each entry must point at the one before it"
        );

        assert_eq!(log.head().sequence, 2);
        assert_eq!(log.head().hash, second.hash);
    }

    fn trail_in(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("audit.jsonl")
    }

    #[test]
    fn claiming_a_trail_twice_is_refused_as_busy() {
        let dir = tempfile::tempdir().unwrap();
        let path = trail_in(&dir);

        let first = claim_trail(&path).expect("an unclaimed trail can be claimed");

        // Two open file descriptions in one process conflict under flock, so
        // this proves the cross-process case without forking.
        match claim_trail(&path) {
            Err(TrailClaimError::Busy { path: busy, .. }) => assert_eq!(busy, path),
            other => panic!("a second claim must be refused as busy, got {other:?}"),
        }

        drop(first);
        claim_trail(&path).expect("dropping the handle releases the lock");
    }

    #[test]
    fn a_busy_refusal_names_the_holder_on_linux() {
        let dir = tempfile::tempdir().unwrap();
        let path = trail_in(&dir);
        let _held = claim_trail(&path).unwrap();

        let Err(error) = claim_trail(&path) else {
            panic!("the second claim must be refused");
        };
        let message = error.to_string();
        assert!(
            message.contains("already owned by another process"),
            "{message}"
        );
        if cfg!(target_os = "linux") {
            assert!(
                matches!(error, TrailClaimError::Busy { holder_pid: Some(pid), .. } if pid == std::process::id()),
                "the holder is this very process, and /proc/locks says so: {error:?}"
            );
            assert!(message.contains(&format!("pid {}", std::process::id())), "{message}");
        }
    }

    #[test]
    fn a_missing_directory_is_an_io_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no-such-dir").join("audit.jsonl");

        match claim_trail(&path) {
            Err(TrailClaimError::Io { path: failed, .. }) => assert_eq!(failed, path),
            other => panic!("an unopenable trail must be an io refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_proc_locks_parser_matches_on_device_and_inode() {
        // makedev(0x103, 0x2): major 0x103 -> bits 8..20, minor 2 -> bits 0..8.
        let dev = (0x103_u64 << 8) | 0x2;
        let locks = "1: FLOCK  ADVISORY  WRITE 4242 103:02:9876 0 EOF\n\
                     2: POSIX  ADVISORY  WRITE 17 08:01:555 0 EOF\n\
                     3: FLOCK  ADVISORY  WRITE 99 103:02:1 0 EOF\n";

        assert_eq!(find_lock_holder(locks, dev, 9876), Some(4242));
        assert_eq!(find_lock_holder(locks, dev, 1), Some(99));
        assert_eq!(find_lock_holder(locks, dev, 555), None, "wrong device");
        assert_eq!(find_lock_holder(locks, dev, 7), None, "no such inode");
        assert_eq!(find_lock_holder("garbage\n\n", dev, 9876), None);
    }

    #[tokio::test]
    async fn a_second_audit_log_over_the_same_trail_refuses_to_spawn() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir));
        let mut runtime = ActonApp::launch_async().await;

        let _first = AuditLog::spawn(&mut runtime, &config)
            .await
            .expect("the first opener owns the trail");

        let error = match AuditLog::spawn(&mut runtime, &config).await {
            Err(error) => error,
            Ok(_) => panic!("a second opener must be refused while the first holds the lock"),
        };
        assert!(error.is_configuration(), "{error:?}");
        assert!(
            error.to_string().contains("already owned by another process"),
            "{error}"
        );

        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn the_lock_is_released_when_the_actor_stops() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir));

        let mut runtime = ActonApp::launch_async().await;
        let _log = AuditLog::spawn(&mut runtime, &config).await.unwrap();
        assert!(
            matches!(claim_trail(config.path()), Err(TrailClaimError::Busy { .. })),
            "the running actor holds the lock"
        );
        runtime.shutdown_all().await.unwrap();

        // Stopping the actor dropped its model, and with it the descriptor.
        let mut runtime = ActonApp::launch_async().await;
        AuditLog::spawn(&mut runtime, &config)
            .await
            .expect("a stopped log releases the trail for the next opener");
        runtime.shutdown_all().await.unwrap();
    }
}
