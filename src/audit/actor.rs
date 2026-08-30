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
//!
//! # Health, and how it stays truthful
//!
//! The actor also owns the writer's [`AuditHealth`]: how many appends reached
//! the disk, how many did not, and the first sequence that is missing. A
//! `mutate_on` handler cannot touch the model after its future has awaited
//! the disk, so each append future ends by sending the actor a private
//! `NoteAppendOutcome`, and the health is folded when that note is
//! processed. For a durable append the note is sent *before* the receipt is
//! delivered, and mailboxes are FIFO, so any [`GetAuditHealth`] a caller
//! sends after receiving the receipt is served after the failure has been
//! folded. That ordering is what makes the strict-mode guard race-free: the
//! prompt loop asks about health only after the previous append has been
//! acknowledged. The healthy-to-degraded transition is broadcast once as
//! [`AuditHealthChanged`].

use crate::audit::chain::ChainHead;
use crate::audit::config::{
    read_trail_id, resolve_trail_id, write_trail_id, AuditConfig, AuditDurability,
};
use crate::audit::entry::{AuditEntry, AuditRecord, InvocationRecord, TurnRecord};
use crate::audit::health::AuditHealth;
use crate::types::TrailId;
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

/// One tool invocation to append to the trail, acknowledged once it is on
/// disk.
///
/// The `ask` form of [`RecordInvocation`]: the reply is an [`AppendReceipt`]
/// that arrives only after the entry has been written and synced — or after
/// the write failed, in which case the receipt says so. The prompt loop uses
/// this under strict durability so a tool's record is durable before the next
/// tool is considered.
#[acton_message]
pub struct RecordInvocationDurably {
    /// Everything about the invocation except its place in the chain.
    pub record: InvocationRecord,
}

/// One completed or refused turn to append to the trail.
#[acton_message]
pub struct RecordTurn {
    /// Turn metadata excluding its place in the chain.
    pub record: TurnRecord,
}

impl RecordTurn {
    /// Wraps a record for sending.
    #[must_use]
    pub fn new(record: TurnRecord) -> Self {
        Self { record }
    }
}

/// A turn record acknowledged once it is on disk.
#[acton_message]
pub struct RecordTurnDurably {
    /// Turn metadata excluding its place in the chain.
    pub record: TurnRecord,
}

impl RecordTurnDurably {
    /// Wraps a record for asking.
    #[must_use]
    pub fn new(record: TurnRecord) -> Self {
        Self { record }
    }
}

impl Request for RecordTurnDurably {
    type Response = AppendReceipt;
}

impl RecordInvocationDurably {
    /// Wraps a record for asking.
    #[must_use]
    pub fn new(record: InvocationRecord) -> Self {
        Self { record }
    }
}

impl Request for RecordInvocationDurably {
    type Response = AppendReceipt;
}

/// What became of one durable append.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
#[non_exhaustive]
pub enum AppendReceipt {
    /// The entry is on disk with the configured durability guarantee.
    Durable {
        /// The entry's place in the chain.
        sequence: u64,
        /// The entry's hash — the new chain head.
        hash: String,
    },
    /// The entry was sealed but never reached the disk.
    Failed {
        /// The sequence the entry was sealed at; the chain has a gap there.
        sequence: u64,
        /// What the operating system said.
        error: String,
    },
}

impl AppendReceipt {
    /// Whether the entry reached the disk.
    #[must_use]
    pub fn is_durable(&self) -> bool {
        matches!(self, Self::Durable { .. })
    }
}

/// Asks the audit log how its writer is doing.
///
/// A read, like [`GetChainHead`], and the same barrier: a reply proves every
/// append outcome noted before it has been folded into the health.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct GetAuditHealth;

impl Request for GetAuditHealth {
    type Response = AuditHealth;
}

/// Broadcast once, on the healthy-to-degraded transition.
///
/// Later failures change the counters, which [`GetAuditHealth`] reports, but
/// they are not a new event: the writer was already degraded.
#[acton_message]
#[derive(Serialize, Deserialize)]
pub struct AuditHealthChanged {
    /// The health as of the first failure.
    pub health: AuditHealth,
}

/// Sent by the actor to itself from an append future.
///
/// A `mutate_on` handler's future cannot touch the model after it has awaited
/// the disk, so the outcome comes back as a message and is folded in order
/// with everything else.
#[acton_message]
struct NoteAppendOutcome {
    /// The head after the entry this note is about.
    head: ChainHead,
    /// What went wrong, if anything.
    error: Option<String>,
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
    /// The identity every entry is sealed under.
    ///
    /// Settled once at spawn from the sidecar and the chain (see
    /// [`AuditLog::spawn`]); the derived `Default` mints a throwaway id that
    /// `spawn` always replaces.
    trail_id: TrailId,
    /// What an append promises before it is acknowledged.
    durability: AuditDurability,
    /// How the writer is doing. Defaults to disabled, which is a lie for a
    /// running actor; [`AuditLog::spawn`] always arms it before starting.
    health: AuditHealth,
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
    /// The trail's identity is settled here too. The sidecar
    /// ([`AuditConfig::trail_id_path`]) and the chain's own entries are both
    /// read; when they agree, or only one of them speaks, that is the
    /// identity, and a missing sidecar is written from it. A trail with no
    /// identity anywhere — a first run, or a trail from before identities —
    /// is given a fresh one. Every entry sealed from now on carries it.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if another process already owns the
    /// trail (see [`claim_trail`]), if the existing file cannot be read, if
    /// its chain does not verify — refusing to start is the right answer,
    /// because appending to a forked or broken chain would bury the evidence
    /// — or if the sidecar and the chain name different trails, which means
    /// one of the two files was moved or copied from somewhere else.
    pub async fn spawn(
        runtime: &mut ActorRuntime,
        config: &AuditConfig,
    ) -> Result<ActorHandle, crate::error::ActonAIError> {
        // Claimed before the head is read, so no other writer can append
        // between the read and the first seal: the head we resume from is
        // exact for as long as we hold the lock.
        let lock = claim_trail(config.path())?;
        let mut head = read_head(config.path()).await?;
        let trail_id = settle_trail_id(config, head.trail_id.as_ref())?;
        head.trail_id = Some(trail_id.clone());

        let mut builder = runtime.new_actor_with_name::<AuditLog>("audit_log".to_string());

        // Installed on the idle builder rather than sent as a message, so the
        // chain head is in place before the actor can receive anything.
        builder.model.path = config.path().to_path_buf();
        builder.model.prev_hash = head.hash.clone();
        builder.model.sequence = head.sequence;
        builder.model.trail_id = trail_id;
        builder.model.durability = config.durability();
        builder.model.health = AuditHealth::armed(head, config.durability());
        builder.model.lock = Some(lock);

        configure_handlers(&mut builder);

        Ok(builder.start().await)
    }

    /// Seals the next entry and hands back what an append future needs.
    fn prepare_append(&mut self, record: AuditRecord) -> PreparedAppend {
        let entry = self.seal_next(record);
        PreparedAppend {
            path: self.path.clone(),
            head: self.head(),
            durability: self.durability,
            entry,
        }
    }

    /// Folds one append outcome into the health. Returns the transition
    /// event, if this was the first failure.
    fn note_outcome(&mut self, note: &NoteAppendOutcome, now: &str) -> Option<AuditHealthChanged> {
        match &note.error {
            None => {
                self.health.note_success(note.head.clone());
                None
            }
            Some(error) => self
                .health
                .note_failure(note.head.clone(), error, now)
                .then(|| AuditHealthChanged {
                    health: self.health.clone(),
                }),
        }
    }

    /// Seals the next entry and advances the head. Pure bookkeeping.
    fn seal_next(&mut self, record: AuditRecord) -> AuditEntry {
        let sequence = self.sequence.saturating_add(1);
        let entry = match record {
            AuditRecord::Invocation(record) => {
                AuditEntry::seal(record, sequence, &self.prev_hash, Some(&self.trail_id))
            }
            AuditRecord::Turn(record) => {
                AuditEntry::seal_turn(record, sequence, &self.prev_hash, Some(&self.trail_id))
            }
        };

        self.sequence = sequence;
        self.prev_hash.clone_from(&entry.hash);

        entry
    }

    /// The chain head as it currently stands.
    ///
    /// Reports the identity the log seals under even before the first
    /// identified entry is written: it is settled and on disk in the sidecar
    /// from spawn.
    fn head(&self) -> ChainHead {
        ChainHead {
            sequence: self.sequence,
            hash: self.prev_hash.clone(),
            entries: self.sequence,
            trail_id: Some(self.trail_id.clone()),
        }
    }
}

/// Settles the trail's identity from its sidecar and its chain, writing the
/// sidecar if it did not exist.
///
/// `chain` is the identity the existing entries are sealed under, `None` for
/// an empty or legacy chain. Runs under the trail lock, so nothing else can
/// write the sidecar between the read and the write.
fn settle_trail_id(
    config: &AuditConfig,
    chain: Option<&TrailId>,
) -> Result<TrailId, crate::error::ActonAIError> {
    let sidecar_path = config.trail_id_path();
    let sidecar = read_trail_id(&sidecar_path).map_err(|error| {
        crate::error::ActonAIError::configuration(
            "audit.path",
            format!(
                "could not read the trail identity at {}: {error}",
                sidecar_path.display()
            ),
        )
    })?;
    let had_sidecar = sidecar.is_some();

    let trail_id = resolve_trail_id(sidecar, chain.cloned()).map_err(|conflict| {
        crate::error::ActonAIError::configuration(
            "audit.path",
            format!(
                "the audit trail at {} carries trail id {} but {} says {}; refusing to start — \
                 a trail cannot be two trails, so one of the files was moved or copied from \
                 elsewhere. Investigate which, and put the pair back together",
                config.path().display(),
                conflict.chain,
                sidecar_path.display(),
                conflict.sidecar,
            ),
        )
    })?;

    if !had_sidecar {
        write_trail_id(&sidecar_path, &trail_id).map_err(|error| {
            crate::error::ActonAIError::configuration(
                "audit.path",
                format!(
                    "could not write the trail identity to {}: {error}",
                    sidecar_path.display()
                ),
            )
        })?;
        tracing::info!(
            trail_id = %trail_id,
            sidecar = %sidecar_path.display(),
            "recorded the audit trail's identity",
        );
    }

    Ok(trail_id)
}

/// Everything an append future needs, cloned out of the model before the
/// async block: the handler closure is `Fn` and cannot hold a borrow of the
/// model across the await.
struct PreparedAppend {
    path: PathBuf,
    head: ChainHead,
    durability: AuditDurability,
    entry: AuditEntry,
}

impl PreparedAppend {
    /// Writes the entry and reports back to the actor.
    ///
    /// The self-note is sent before this returns, so a caller that goes on
    /// to ask [`GetAuditHealth`] after a reply from this append is served
    /// after the outcome has been folded.
    async fn perform(self, self_handle: &ActorHandle) -> Result<(), String> {
        let result = append_entry(&self.path, &self.entry, self.durability)
            .await
            .map_err(|error| error.to_string());

        if let Err(error) = &result {
            // A failed append must not fail the turn — the tool already ran,
            // and refusing to continue would not un-run it. It is logged at
            // error level because a compliance deployment needs to notice,
            // and the chain head still advances so the gap is visible to
            // `audit verify` rather than silently healed. What a strict trail
            // does about the *next* call is the prompt loop's decision, made
            // on the health this note feeds.
            tracing::error!(
                path = %self.path.display(),
                sequence = self.entry.sequence,
                %error,
                "could not append to the audit trail",
            );
        }

        self_handle
            .send(NoteAppendOutcome {
                head: self.head,
                error: result.clone().err(),
            })
            .await;

        result
    }

    /// The receipt a durable append answers with.
    fn receipt(sequence: u64, hash: String, result: Result<(), String>) -> AppendReceipt {
        match result {
            Ok(()) => AppendReceipt::Durable { sequence, hash },
            Err(error) => AppendReceipt::Failed { sequence, error },
        }
    }
}

/// Wires the audit log's handlers.
fn configure_handlers(builder: &mut ManagedActor<Idle, AuditLog>) {
    // The actor's own address, captured by the append futures so they can
    // report their outcome back as a message.
    let self_handle = builder.handle().clone();

    // `mutate_on`, and the write lives in the returned future deliberately.
    // A mutable handler's future is awaited inline before the actor takes its
    // next message, so appends happen in exactly the order the entries were
    // sealed. For a hash chain that serialization is the requirement, not a
    // cost: two concurrent appends would interleave lines that claim to
    // follow each other.
    let handle = self_handle.clone();
    builder.mutate_on::<RecordInvocation>(move |actor, envelope| {
        let prepared = actor
            .model
            .prepare_append(AuditRecord::Invocation(envelope.message().record.clone()));
        let handle = handle.clone();

        Reply::pending(async move {
            // The outcome is already logged and noted; a fire-and-forget
            // sender has nobody to hand it to.
            let _ = prepared.perform(&handle).await;
        })
    });

    // Same write, but the sender is waiting for the receipt. The note to self
    // goes out before the reply does — see the module docs for why that
    // ordering is what makes the strict-mode guard race-free.
    let handle = self_handle.clone();
    builder.mutate_on::<RecordInvocationDurably>(move |actor, envelope| {
        let prepared = actor
            .model
            .prepare_append(AuditRecord::Invocation(envelope.message().record.clone()));
        let reply = envelope.reply_envelope();
        let handle = handle.clone();

        Reply::pending(async move {
            let sequence = prepared.entry.sequence;
            let hash = prepared.entry.hash.clone();
            let result = prepared.perform(&handle).await;
            reply
                .send(PreparedAppend::receipt(sequence, hash, result))
                .await;
        })
    });

    let handle = self_handle.clone();
    builder.mutate_on::<RecordTurn>(move |actor, envelope| {
        let prepared = actor
            .model
            .prepare_append(AuditRecord::Turn(envelope.message().record.clone()));
        let handle = handle.clone();
        Reply::pending(async move {
            let _ = prepared.perform(&handle).await;
        })
    });

    let handle = self_handle.clone();
    builder.mutate_on::<RecordTurnDurably>(move |actor, envelope| {
        let prepared = actor
            .model
            .prepare_append(AuditRecord::Turn(envelope.message().record.clone()));
        let reply = envelope.reply_envelope();
        let handle = handle.clone();
        Reply::pending(async move {
            let sequence = prepared.entry.sequence;
            let hash = prepared.entry.hash.clone();
            let result = prepared.perform(&handle).await;
            reply
                .send(PreparedAppend::receipt(sequence, hash, result))
                .await;
        })
    });

    // Folding an outcome is bookkeeping; the one asynchronous thing it can
    // produce is the transition broadcast, which is a send.
    builder.mutate_on::<NoteAppendOutcome>(|actor, envelope| {
        let now = chrono::Utc::now().to_rfc3339();
        let Some(event) = actor.model.note_outcome(envelope.message(), &now) else {
            return Reply::ready();
        };

        tracing::warn!(
            failures = event.health.failures,
            first_failed_sequence = ?event.health.first_failed_sequence,
            durability = %event.health.durability,
            "the audit writer is degraded",
        );
        let broker = actor.broker().clone();
        Reply::pending(async move {
            broker.broadcast(event).await;
        })
    });

    // Reads, so several asks can be served at once and none of them blocks
    // the appends queued ahead of them.
    builder.act_on::<GetChainHead>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let head = actor.model.head();
        Reply::pending(async move {
            reply.send(head).await;
        })
    });

    builder.act_on::<GetAuditHealth>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let health = actor.model.health.clone();
        Reply::pending(async move {
            reply.send(health).await;
        })
    });
}

/// Appends one entry as a JSONL line.
///
/// Opens in append mode per write. That costs a syscall and buys the property
/// the file is for: nothing this process holds can rewind or overwrite what is
/// already on disk, and a crash between entries leaves a shorter valid chain
/// rather than a corrupt one.
///
/// Under strict durability the data is synced to the device before this
/// returns, so an acknowledged entry survives a power cut, not just a crash.
async fn append_entry(
    path: &Path,
    entry: &AuditEntry,
    durability: AuditDurability,
) -> Result<(), std::io::Error> {
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
    file.flush().await?;
    if durability.is_strict() {
        file.sync_data().await?;
    }
    Ok(())
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
        let trail_id = TrailId::new();
        let mut log = AuditLog {
            path: PathBuf::from("/dev/null"),
            prev_hash: GENESIS_HASH.to_string(),
            sequence: 0,
            trail_id: trail_id.clone(),
            durability: AuditDurability::BestEffort,
            health: AuditHealth::armed(ChainHead::empty(), AuditDurability::BestEffort),
            lock: None,
        };

        let first = log.seal_next(AuditRecord::Invocation(record("a")));
        assert_eq!(first.sequence, 1);
        assert_eq!(first.prev_hash, GENESIS_HASH);
        assert_eq!(first.trail_id.as_ref(), Some(&trail_id));

        let second = log.seal_next(AuditRecord::Invocation(record("b")));
        assert_eq!(second.sequence, 2);
        assert_eq!(
            second.prev_hash, first.hash,
            "each entry must point at the one before it"
        );
        assert_eq!(second.trail_id.as_ref(), Some(&trail_id));

        assert_eq!(log.head().sequence, 2);
        assert_eq!(log.head().hash, second.hash);
        assert_eq!(log.head().trail_id, Some(trail_id));
    }

    #[tokio::test]
    async fn a_first_spawn_mints_an_identity_and_writes_the_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir));
        let mut runtime = ActonApp::launch_async().await;

        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();

        let head = log.ask(GetChainHead).await.unwrap();
        let trail_id = head
            .trail_id
            .expect("a spawned log has an identity from the start");
        assert_eq!(
            read_trail_id(&config.trail_id_path()).unwrap(),
            Some(trail_id.clone()),
            "the sidecar holds the identity before any entry is written"
        );

        log.send(RecordInvocation::new(record("a"))).await;
        log.ask(GetChainHead).await.unwrap();
        let entries = crate::audit::read_entries(config.path()).await.unwrap();
        assert_eq!(entries[0].trail_id, Some(trail_id));
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn a_restart_keeps_the_identity_it_was_given() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir));

        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();
        log.send(RecordInvocation::new(record("a"))).await;
        let first_head = log.ask(GetChainHead).await.unwrap();
        runtime.shutdown_all().await.unwrap();

        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();
        log.send(RecordInvocation::new(record("b"))).await;
        let second_head = log.ask(GetChainHead).await.unwrap();
        runtime.shutdown_all().await.unwrap();

        assert_eq!(second_head.trail_id, first_head.trail_id);
        let entries = crate::audit::read_entries(config.path()).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].trail_id, first_head.trail_id);
        assert_eq!(
            crate::audit::verify_chain(&entries).unwrap().trail_id,
            first_head.trail_id
        );
    }

    #[tokio::test]
    async fn a_legacy_trail_gains_an_identity_without_a_move_aside() {
        // A trail written before identities: no sidecar, entries without a
        // trail_id. It keeps verifying, gets an identity, and every entry
        // from here on carries it.
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir));
        let legacy = AuditEntry::seal(record("old"), 1, GENESIS_HASH, None);
        std::fs::write(config.path(), format!("{}\n", legacy.to_jsonl().unwrap())).unwrap();

        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config)
            .await
            .expect("a legacy trail is still a valid trail");
        log.send(RecordInvocation::new(record("new"))).await;
        let head = log.ask(GetChainHead).await.unwrap();
        runtime.shutdown_all().await.unwrap();

        let trail_id = head.trail_id.expect("the legacy trail now has an identity");
        assert_eq!(
            read_trail_id(&config.trail_id_path()).unwrap(),
            Some(trail_id.clone())
        );
        let entries = crate::audit::read_entries(config.path()).await.unwrap();
        assert_eq!(entries[0].trail_id, None, "history is not rewritten");
        assert_eq!(entries[1].trail_id, Some(trail_id.clone()));
        assert_eq!(
            crate::audit::verify_chain(&entries).unwrap().trail_id,
            Some(trail_id)
        );
    }

    #[tokio::test]
    async fn a_lost_sidecar_is_rebuilt_from_the_chain() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir));

        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();
        log.send(RecordInvocation::new(record("a"))).await;
        let head = log.ask(GetChainHead).await.unwrap();
        runtime.shutdown_all().await.unwrap();

        std::fs::remove_file(config.trail_id_path()).unwrap();

        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config)
            .await
            .expect("the chain still knows who it is");
        assert_eq!(log.ask(GetChainHead).await.unwrap().trail_id, head.trail_id);
        runtime.shutdown_all().await.unwrap();

        assert_eq!(
            read_trail_id(&config.trail_id_path()).unwrap(),
            head.trail_id
        );
    }

    #[tokio::test]
    async fn a_sidecar_that_disagrees_with_the_chain_refuses_to_spawn() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir));

        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();
        log.send(RecordInvocation::new(record("a"))).await;
        let head = log.ask(GetChainHead).await.unwrap();
        runtime.shutdown_all().await.unwrap();

        // Somebody drops another trail's sidecar next to this chain.
        let other = TrailId::new();
        std::fs::remove_file(config.trail_id_path()).unwrap();
        write_trail_id(&config.trail_id_path(), &other).unwrap();

        let mut runtime = ActonApp::launch_async().await;
        let error = match AuditLog::spawn(&mut runtime, &config).await {
            Err(error) => error,
            Ok(_) => panic!("two identities for one trail must be refused"),
        };
        runtime.shutdown_all().await.unwrap();

        assert!(error.is_configuration(), "{error:?}");
        let text = error.to_string();
        assert!(text.contains("refusing to start"), "{text}");
        assert!(text.contains(&other.to_string()), "{text}");
        assert!(
            text.contains(&head.trail_id.unwrap().to_string()),
            "the refusal names both identities: {text}"
        );
    }

    #[tokio::test]
    async fn a_relabelled_trail_refuses_to_spawn_as_broken() {
        // The forger rewrites every entry's trail_id and the sidecar to match:
        // the identity is inside each hash, so the chain no longer verifies
        // and the log refuses to append to it.
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir));

        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();
        log.send(RecordInvocation::new(record("a"))).await;
        log.send(RecordInvocation::new(record("b"))).await;
        log.ask(GetChainHead).await.unwrap();
        runtime.shutdown_all().await.unwrap();

        let other = TrailId::new();
        let mut entries = crate::audit::read_entries(config.path()).await.unwrap();
        for entry in &mut entries {
            entry.trail_id = Some(other.clone());
        }
        let relabelled: String = entries
            .iter()
            .map(|entry| format!("{}\n", entry.to_jsonl().unwrap()))
            .collect();
        std::fs::write(config.path(), relabelled).unwrap();
        std::fs::remove_file(config.trail_id_path()).unwrap();
        write_trail_id(&config.trail_id_path(), &other).unwrap();

        let mut runtime = ActonApp::launch_async().await;
        let error = match AuditLog::spawn(&mut runtime, &config).await {
            Err(error) => error,
            Ok(_) => panic!("a relabelled trail must not be appended to"),
        };
        runtime.shutdown_all().await.unwrap();

        assert!(error.is_configuration(), "{error:?}");
        assert!(
            error.to_string().contains("does not verify"),
            "the relabelling is caught as a broken chain: {error}"
        );
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
            assert!(
                message.contains(&format!("pid {}", std::process::id())),
                "{message}"
            );
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
            error
                .to_string()
                .contains("already owned by another process"),
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
            matches!(
                claim_trail(config.path()),
                Err(TrailClaimError::Busy { .. })
            ),
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

    /// Replaces the trail with a directory of the same name, from under the
    /// running writer: the lock stays on the old inode, the next open by
    /// path fails.
    fn make_unappendable(path: &Path) {
        std::fs::remove_file(path).unwrap();
        std::fs::create_dir(path).unwrap();
    }

    #[tokio::test]
    async fn a_fresh_log_is_healthy_with_nothing_counted() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir)).with_durability(AuditDurability::Strict);
        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();

        let health = log.ask(GetAuditHealth).await.unwrap();

        assert_eq!(health.state, crate::audit::AuditHealthState::Healthy);
        assert_eq!(health.durability, AuditDurability::Strict);
        assert_eq!(health.appended, 0);
        assert_eq!(health.failures, 0);
        assert_eq!(health.head.sequence, 0);
        assert_eq!(health.head.hash, GENESIS_HASH);
        assert!(
            health.head.trail_id.is_some(),
            "even an empty trail knows who it is once spawned"
        );
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn a_durable_append_is_acknowledged_with_the_new_head() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir)).with_durability(AuditDurability::Strict);
        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();

        let receipt = log
            .ask(RecordInvocationDurably::new(record("a")))
            .await
            .unwrap();

        let AppendReceipt::Durable { sequence, hash } = receipt else {
            panic!("a healthy trail acknowledges durably, got {receipt:?}");
        };
        assert_eq!(sequence, 1);

        // The receipt was sent after the self-note, so this ask lands after
        // the outcome has been folded.
        let health = log.ask(GetAuditHealth).await.unwrap();
        assert_eq!(health.appended, 1);
        assert_eq!(health.failures, 0);
        assert_eq!(health.head.hash, hash);
        assert_eq!(health.head.sequence, 1);

        let entries = crate::audit::read_entries(config.path()).await.unwrap();
        assert_eq!(entries.len(), 1, "the entry is on disk before the receipt");
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn a_failed_durable_append_degrades_the_writer_and_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir)).with_durability(AuditDurability::Strict);
        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();
        log.ask(RecordInvocationDurably::new(record("a")))
            .await
            .unwrap();

        make_unappendable(config.path());

        let receipt = log
            .ask(RecordInvocationDurably::new(record("b")))
            .await
            .unwrap();
        let AppendReceipt::Failed { sequence, error } = receipt else {
            panic!("an unappendable trail must report failure, got {receipt:?}");
        };
        assert_eq!(sequence, 2);
        assert!(!error.is_empty());

        let health = log.ask(GetAuditHealth).await.unwrap();
        assert!(health.is_degraded());
        assert_eq!(health.appended, 1);
        assert_eq!(health.failures, 1);
        assert_eq!(health.first_failed_sequence, Some(2));
        assert_eq!(health.last_error.as_deref(), Some(error.as_str()));
        assert!(health.degraded_since.is_some());
        assert_eq!(
            health.head.sequence, 2,
            "the head advances past the gap so `audit verify` can see it"
        );

        // A second failure counts; the first failed sequence does not move.
        let receipt = log
            .ask(RecordInvocationDurably::new(record("c")))
            .await
            .unwrap();
        assert!(!receipt.is_durable());
        let health = log.ask(GetAuditHealth).await.unwrap();
        assert_eq!(health.failures, 2);
        assert_eq!(health.first_failed_sequence, Some(2));
        runtime.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn a_best_effort_failure_is_visible_through_health_after_the_head_barrier() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir));
        let mut runtime = ActonApp::launch_async().await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();

        make_unappendable(config.path());
        log.send(RecordInvocation::new(record("a"))).await;

        // The head ask cannot answer until the append future has finished,
        // and that future queues the outcome note before it finishes; so a
        // health ask sent after the head reply is served after the note.
        let head = log.ask(GetChainHead).await.unwrap();
        assert_eq!(head.sequence, 1);
        let health = log.ask(GetAuditHealth).await.unwrap();

        assert!(health.is_degraded());
        assert_eq!(health.durability, AuditDurability::BestEffort);
        assert_eq!(health.failures, 1);
        assert_eq!(health.first_failed_sequence, Some(1));
        runtime.shutdown_all().await.unwrap();
    }

    #[acton_actor]
    struct HealthSpy {
        seen: Vec<AuditHealth>,
    }

    #[acton_message]
    struct GetSeen;

    #[acton_message]
    struct Seen {
        events: Vec<AuditHealth>,
    }

    impl Request for GetSeen {
        type Response = Seen;
    }

    async fn spawn_health_spy(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder = runtime.new_actor::<HealthSpy>();
        builder.mutate_on::<AuditHealthChanged>(|actor, envelope| {
            actor.model.seen.push(envelope.message().health.clone());
            Reply::ready()
        });
        builder.act_on::<GetSeen>(|actor, envelope| {
            let reply = envelope.reply_envelope();
            let events = actor.model.seen.clone();
            Reply::pending(async move {
                reply.send(Seen { events }).await;
            })
        });
        builder.handle().subscribe::<AuditHealthChanged>().await;
        builder.start().await
    }

    #[tokio::test]
    async fn the_first_failure_is_broadcast_once() {
        let dir = tempfile::tempdir().unwrap();
        let config = AuditConfig::new(trail_in(&dir)).with_durability(AuditDurability::Strict);
        let mut runtime = ActonApp::launch_async().await;
        let spy = spawn_health_spy(&mut runtime).await;
        let log = AuditLog::spawn(&mut runtime, &config).await.unwrap();

        make_unappendable(config.path());
        for name in ["a", "b", "c"] {
            let receipt = log
                .ask(RecordInvocationDurably::new(record(name)))
                .await
                .unwrap();
            assert!(!receipt.is_durable());
        }
        // Both notes are folded once the health answers, so both broadcasts
        // — if there were two — have been handed to the broker by now.
        log.ask(GetAuditHealth).await.unwrap();
        runtime.broker().ask(FlushBroadcasts).await.unwrap();

        let seen = spy.ask(GetSeen).await.unwrap().events;
        assert_eq!(seen.len(), 1, "one transition, one event: {seen:?}");
        assert_eq!(seen[0].failures, 1);
        assert_eq!(seen[0].first_failed_sequence, Some(1));
        assert!(seen[0].is_degraded());
        runtime.shutdown_all().await.unwrap();
    }
}
