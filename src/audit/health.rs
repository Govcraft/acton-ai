//! The audit writer's health, as the audit actor reports it.
//!
//! A trail that is configured is not the same as a trail that is being
//! written. Between those two sits every way a disk can fail after launch —
//! filled, unmounted, replaced, permissions changed — and a compliance
//! deployment needs to see that failure as a state, not hunt for it in a log
//! line. [`AuditHealth`] is that state: counters for what reached the disk
//! and what did not, the sequence number of the first entry that failed, and
//! the error the operating system gave.
//!
//! Everything in this module is pure bookkeeping. The actor folds outcomes
//! into it; nothing here opens a file.

use crate::audit::chain::ChainHead;
use crate::audit::config::AuditDurability;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Where the writer stands, in one word.
///
/// A three-way answer rather than a boolean because "no trail is configured"
/// and "the trail is healthy" must never read the same: an unconfigured
/// deployment that reports healthy is the exact failure an audit exists to
/// catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AuditHealthState {
    /// No trail is configured; nothing is being recorded.
    Disabled,
    /// Every append so far reached the disk.
    Healthy,
    /// At least one append failed since this process started.
    Degraded,
}

impl fmt::Display for AuditHealthState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Disabled => "disabled",
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
        };
        f.write_str(name)
    }
}

/// The audit writer's health.
///
/// Counters cover this process only: a restart starts them from zero, which
/// is deliberate — the trail on disk is what survives a restart, and the
/// operator recovers by restarting over a repaired trail. A degraded writer
/// stays degraded for the life of the process for the same reason: an entry
/// that never reached the disk cannot be reached later, so "it works again
/// now" is not the same as "the record is complete".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditHealth {
    /// Where the writer stands.
    pub state: AuditHealthState,
    /// Where the chain ends, as the writer sees it.
    pub head: ChainHead,
    /// What an append promises before it is acknowledged.
    pub durability: AuditDurability,
    /// Entries appended — and, when strict, synced — in this process.
    pub appended: u64,
    /// Appends that failed in this process.
    pub failures: u64,
    /// Sequence number of the first entry that failed to reach the disk.
    pub first_failed_sequence: Option<u64>,
    /// What the operating system said about the most recent failure.
    pub last_error: Option<String>,
    /// When the first failure happened, RFC 3339.
    pub degraded_since: Option<String>,
}

impl Default for AuditHealth {
    /// [`AuditHealth::disabled`]: the truthful answer when nothing has armed
    /// a trail yet.
    fn default() -> Self {
        Self::disabled()
    }
}

impl AuditHealth {
    /// The health of a trail that was just armed: nothing written, nothing
    /// failed.
    #[must_use]
    pub fn armed(head: ChainHead, durability: AuditDurability) -> Self {
        Self {
            state: AuditHealthState::Healthy,
            head,
            durability,
            appended: 0,
            failures: 0,
            first_failed_sequence: None,
            last_error: None,
            degraded_since: None,
        }
    }

    /// The health of a runtime with no trail configured.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            state: AuditHealthState::Disabled,
            head: ChainHead::empty(),
            durability: AuditDurability::default(),
            appended: 0,
            failures: 0,
            first_failed_sequence: None,
            last_error: None,
            degraded_since: None,
        }
    }

    /// Whether at least one append has failed.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.state == AuditHealthState::Degraded
    }

    /// Folds a successful append. `head` is the chain head after it.
    pub(crate) fn note_success(&mut self, head: ChainHead) {
        self.appended = self.appended.saturating_add(1);
        self.head = head;
    }

    /// Folds a failed append.
    ///
    /// Returns `true` exactly once — on the transition from healthy to
    /// degraded — so the caller knows when to announce it. The head still
    /// advances: the actor sealed the entry and the next one links to it, so
    /// the gap stays visible to `audit verify` rather than being healed over.
    pub(crate) fn note_failure(&mut self, head: ChainHead, error: &str, now: &str) -> bool {
        self.failures = self.failures.saturating_add(1);
        self.head = head;
        self.last_error = Some(error.to_string());

        let first = self.state != AuditHealthState::Degraded;
        if first {
            self.state = AuditHealthState::Degraded;
            self.first_failed_sequence = Some(self.head.sequence);
            self.degraded_since = Some(now.to_string());
        }
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(sequence: u64) -> ChainHead {
        ChainHead {
            sequence,
            hash: format!("hash-{sequence}"),
            entries: sequence,
            trail_id: None,
        }
    }

    #[test]
    fn an_armed_trail_is_healthy_with_nothing_counted() {
        let health = AuditHealth::armed(head(0), AuditDurability::Strict);

        assert_eq!(health.state, AuditHealthState::Healthy);
        assert!(!health.is_degraded());
        assert_eq!(health.appended, 0);
        assert_eq!(health.failures, 0);
        assert_eq!(health.first_failed_sequence, None);
        assert_eq!(health.durability, AuditDurability::Strict);
    }

    #[test]
    fn a_disabled_trail_is_neither_healthy_nor_degraded() {
        let health = AuditHealth::disabled();

        assert_eq!(health.state, AuditHealthState::Disabled);
        assert!(!health.is_degraded());
        assert_eq!(health.head, ChainHead::empty());
    }

    #[test]
    fn a_success_counts_and_moves_the_head() {
        let mut health = AuditHealth::armed(head(0), AuditDurability::BestEffort);

        health.note_success(head(1));
        health.note_success(head(2));

        assert_eq!(health.appended, 2);
        assert_eq!(health.head, head(2));
        assert_eq!(health.state, AuditHealthState::Healthy);
    }

    #[test]
    fn the_first_failure_is_the_transition_and_later_ones_are_not() {
        let mut health = AuditHealth::armed(head(2), AuditDurability::Strict);

        assert!(
            health.note_failure(head(3), "No space left on device", "2026-08-29T10:00:00Z"),
            "the first failure is the healthy -> degraded transition"
        );
        assert!(
            !health.note_failure(head(4), "No space left on device", "2026-08-29T10:00:01Z"),
            "a second failure changes counters, not state"
        );

        assert!(health.is_degraded());
        assert_eq!(health.state, AuditHealthState::Degraded);
        assert_eq!(health.failures, 2);
        assert_eq!(health.first_failed_sequence, Some(3));
        assert_eq!(
            health.degraded_since.as_deref(),
            Some("2026-08-29T10:00:00Z")
        );
        assert_eq!(health.head, head(4), "the head still advances past a gap");
    }

    #[test]
    fn a_success_after_a_failure_does_not_heal_the_writer() {
        // The entry that failed is gone; a later success does not bring it
        // back, so the state must not read as healthy again.
        let mut health = AuditHealth::armed(head(0), AuditDurability::Strict);
        health.note_failure(head(1), "Is a directory", "2026-08-29T10:00:00Z");

        health.note_success(head(2));

        assert!(health.is_degraded());
        assert_eq!(health.appended, 1);
        assert_eq!(health.failures, 1);
        assert_eq!(health.first_failed_sequence, Some(1));
        assert_eq!(health.last_error.as_deref(), Some("Is a directory"));
    }

    #[test]
    fn the_last_error_follows_the_most_recent_failure() {
        let mut health = AuditHealth::armed(head(0), AuditDurability::Strict);
        health.note_failure(head(1), "first", "2026-08-29T10:00:00Z");
        health.note_failure(head(2), "second", "2026-08-29T10:00:01Z");

        assert_eq!(health.last_error.as_deref(), Some("second"));
        assert_eq!(
            health.degraded_since.as_deref(),
            Some("2026-08-29T10:00:00Z"),
            "degraded_since is the first failure, not the latest"
        );
    }

    #[test]
    fn health_serializes_with_snake_case_states() {
        let health = AuditHealth::armed(head(0), AuditDurability::BestEffort);
        let json = serde_json::to_value(&health).expect("serializes");

        assert_eq!(json["state"], "healthy");
        assert_eq!(json["durability"], "best_effort");
        assert_eq!(AuditHealthState::Degraded.to_string(), "degraded");
    }
}
