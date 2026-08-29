//! Tamper-evident audit trail for tool invocations.
//!
//! Every tool call the prompt loop dispatches — allowed, refused, or failed —
//! is appended to a JSONL file as one hash-chained entry. Each entry carries
//! the hash of the entry before it, so editing any entry invalidates it, and
//! reselaing that entry invalidates the one after. One writer owns the chain,
//! which is why it is an actor rather than a shared handle.
//!
//! # What an entry records
//!
//! Timestamp, correlation / conversation / turn IDs, tool name, arguments with
//! secrets redacted, a bounded result summary, the approval decision and who
//! made it, and how long the call took. See [`AuditEntry`].
//!
//! # One writer per file
//!
//! The trail is claimed with an exclusive advisory lock when the audit actor
//! spawns, and the lock is held until it stops. A second process opening the
//! same trail fails to launch with a configuration error naming the holder,
//! instead of forking the chain. See [`claim_trail`] and [`TrailClaimError`].
//!
//! # One identity per trail
//!
//! A trail is given a [`TrailId`](crate::types::TrailId) the first time an
//! audit log opens it. The id is kept in a sidecar beside the file
//! (`audit.jsonl.trail`, see [`AuditConfig::trail_id_path`]) and sealed into
//! every entry's hash, so an entry cannot be relabelled as another trail's
//! and a chain cannot change identity part-way: [`verify_chain`] reports
//! [`ChainBreakKind::TrailMismatch`] where it tries. Trails written before
//! identities existed keep verifying — their unidentified prefix is allowed
//! — and gain an identity on the next spawn. A sidecar that names a different
//! trail than the chain is sealed under refuses the spawn: one of the two
//! files was moved or copied from somewhere else. [`ChainHead::trail_id`]
//! reports the identity through [`ActonAI::audit_head`](crate::facade::ActonAI::audit_head)
//! and `audit verify`.
//!
//! # Off by default
//!
//! No `[audit]` section and no `.audit(..)` call means no actor is spawned and
//! nothing is written.
//!
//! # Durability and health
//!
//! By default an append is best effort: a failed write is logged and the
//! turn continues, because the tool already ran. `[audit] durability =
//! "strict"` (or [`AuditConfig::with_durability`]) makes the trail the
//! authority instead. Every entry is fsynced and acknowledged before the
//! prompt loop considers the next tool call, and once one append has failed
//! the loop refuses every non-idempotent tool call — through the ordinary
//! policy path, recorded as denied, with the reason fed back to the model —
//! until the process is restarted over a repaired trail. Tools declared
//! idempotent keep running. An audit actor that cannot answer is treated the
//! same as a degraded one: the guard fails closed.
//!
//! Either way the writer's state is readable as [`AuditHealth`] through
//! [`ActonAI::audit_health`](crate::facade::ActonAI::audit_health) or the
//! [`GetAuditHealth`] request, and the first failure is broadcast once as
//! [`AuditHealthChanged`].
//!
//! # Verifying
//!
//! ```text
//! acton-ai audit verify --file /var/log/acton-ai/audit.jsonl
//! ```
//!
//! reports the first broken link, or the verified head. [`verify_chain`] is
//! the same function, callable directly.
//!
//! # Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! let ai = ActonAI::builder()
//!     .anthropic_from_env()
//!     .audit(
//!         AuditConfig::new("/var/log/acton-ai/audit.jsonl")
//!             .with_user("acct:alice"),
//!     )
//!     .launch()
//!     .await?;
//!
//! ai.prompt("read the config").use_builtins().collect().await?;
//!
//! // `audit_head` doubles as a barrier: it cannot answer until every
//! // invocation sent before it has been written.
//! let head = ai.audit_head().await?;
//! println!("{} entries, head {}", head.entries, head.hash);
//! ```

pub mod actor;
pub mod chain;
pub mod config;
pub mod entry;
pub mod health;
pub mod redact;

pub use actor::{
    claim_trail, AppendReceipt, AuditHealthChanged, AuditLog, GetAuditHealth, GetChainHead,
    RecordInvocation, RecordInvocationDurably, TrailClaimError,
};
pub use chain::{verify_chain, verify_next, ChainBreak, ChainBreakKind, ChainHead};
pub use config::{
    default_audit_path, read_trail_id, resolve_trail_id, write_trail_id, AuditConfig,
    AuditDurability, TrailIdConflict, DEFAULT_AUDIT_FILE, TRAIL_ID_SUFFIX,
};
pub use entry::{AuditDecision, AuditEntry, AuditOutcome, InvocationRecord, GENESIS_HASH};
pub use health::{AuditHealth, AuditHealthState};
pub use redact::{Redactor, DEFAULT_REDACT_PATTERNS, REDACTED};

use std::path::Path;

/// Reads a JSONL trail into entries.
///
/// Blank lines are skipped so a trailing newline is not an error. A line that
/// does not parse is reported as invalid data naming the line number, because
/// "the file is not what it claims to be" is itself an integrity finding and
/// should not be silently skipped.
///
/// # Errors
///
/// Returns the underlying IO error if the file cannot be read — including
/// `NotFound`, which callers distinguish to mean "no trail yet" — or an
/// `InvalidData` error naming the first line that does not parse.
pub async fn read_entries(path: &Path) -> Result<Vec<AuditEntry>, std::io::Error> {
    let contents = acton_reactive::prelude::tokio::fs::read_to_string(path).await?;
    parse_entries(&contents)
}

/// Parses the contents of a JSONL trail.
///
/// Pure, so the interesting failure — a line that is not an entry — is
/// testable without a file.
///
/// # Errors
///
/// Returns an `InvalidData` error naming the first line that does not parse.
pub fn parse_entries(contents: &str) -> Result<Vec<AuditEntry>, std::io::Error> {
    contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str::<AuditEntry>(line).map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("line {} is not a valid audit entry: {error}", index + 1),
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Decider;
    use crate::types::{CorrelationId, TurnId};
    use serde_json::json;

    fn entry(sequence: u64, prev_hash: &str) -> AuditEntry {
        AuditEntry::seal(
            InvocationRecord {
                timestamp: "2026-08-19T12:00:00Z".to_string(),
                correlation_id: CorrelationId::new(),
                conversation_id: None,
                user: None,
                turn_id: TurnId::new(),
                tool_call_id: "toolu_01".to_string(),
                tool_name: "bash".to_string(),
                arguments: json!({"command": "ls"}),
                outcome: AuditOutcome::Success {
                    summary: "ok".to_string(),
                },
                decision: AuditDecision::approved(Decider::NoPolicy),
                duration_ms: 3,
                response_size_bytes: Some(4),
                resumed: false,
            },
            sequence,
            prev_hash,
            None,
        )
    }

    #[test]
    fn parsing_round_trips_a_written_trail() {
        let first = entry(1, GENESIS_HASH);
        let second = entry(2, &first.hash);
        let contents = format!(
            "{}\n{}\n",
            first.to_jsonl().expect("serializes"),
            second.to_jsonl().expect("serializes")
        );

        let parsed = parse_entries(&contents).expect("a written trail must parse back");

        assert_eq!(parsed, vec![first, second]);
        assert!(verify_chain(&parsed).is_ok());
    }

    #[test]
    fn blank_lines_are_skipped() {
        let first = entry(1, GENESIS_HASH);
        let contents = format!("\n{}\n\n", first.to_jsonl().expect("serializes"));

        assert_eq!(parse_entries(&contents).expect("parses").len(), 1);
    }

    #[test]
    fn a_corrupt_line_is_reported_with_its_line_number() {
        let first = entry(1, GENESIS_HASH);
        let contents = format!("{}\nnot json\n", first.to_jsonl().expect("serializes"));

        let error = parse_entries(&contents).expect_err("a corrupt line must be reported");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("line 2"), "{error}");
    }

    #[tokio::test]
    async fn a_missing_file_reports_not_found() {
        let error = read_entries(Path::new("/nonexistent/audit.jsonl"))
            .await
            .expect_err("a missing trail must report NotFound");

        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}
