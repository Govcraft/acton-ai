//! The audit actor: the one writer that owns the chain.
//!
//! A hash chain has exactly one piece of mutable state — the head — and it has
//! to be advanced and written in the same order every time. That is a state
//! owner, so it is an actor, and it is the only thing that ever appends to the
//! file. No lock appears anywhere in this module because there is nothing to
//! share: the head has one owner and everyone else sends messages.

use crate::audit::chain::ChainHead;
use crate::audit::config::AuditConfig;
use crate::audit::entry::{AuditEntry, InvocationRecord};
use acton_reactive::prelude::tokio::io::AsyncWriteExt;
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
    /// Returns a configuration error if the existing file cannot be read or
    /// its chain does not verify — refusing to start is the right answer,
    /// because appending to a broken chain would bury the evidence.
    pub async fn spawn(
        runtime: &mut ActorRuntime,
        config: &AuditConfig,
    ) -> Result<ActorHandle, crate::error::ActonAIError> {
        let head = read_head(config.path()).await?;

        let mut builder = runtime.new_actor_with_name::<AuditLog>("audit_log".to_string());

        // Installed on the idle builder rather than sent as a message, so the
        // chain head is in place before the actor can receive anything.
        builder.model.path = config.path().to_path_buf();
        builder.model.prev_hash = head.hash;
        builder.model.sequence = head.sequence;

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
                 to a broken chain — move the file aside after investigating it",
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
            turn_id: TurnId::new(),
            tool_name: name.to_string(),
            arguments: json!({"value": 1}),
            outcome: AuditOutcome::Success {
                summary: "ok".to_string(),
            },
            decision: AuditDecision::approved(Decider::NoPolicy),
            duration_ms: 1,
            resumed: false,
        }
    }

    #[test]
    fn sealing_advances_the_head_and_links_each_entry_to_the_last() {
        let mut log = AuditLog {
            path: PathBuf::from("/dev/null"),
            prev_hash: GENESIS_HASH.to_string(),
            sequence: 0,
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
}
