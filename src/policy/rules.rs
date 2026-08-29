//! The rule-based half of the policy gate.
//!
//! Everything here is pure: rules in, verdict out. No IO, no clock, no
//! callback. That is what makes the precedence between a denylist, an
//! allowlist and a per-turn cap testable on its own, without a model or a
//! runtime in the picture.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Why a tool call was refused.
///
/// A denial is an *outcome*, not a failure, so this deliberately does not
/// implement [`std::error::Error`]: it is rendered into the tool result the
/// model reads and recorded in the audit trail, and it never aborts a turn.
///
/// Marked `#[non_exhaustive]` so new rule kinds can be added without breaking
/// downstream `match`es.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DenialReason {
    /// An allowlist was configured and the tool was not on it.
    NotAllowed {
        /// The tool the model asked for.
        tool_name: String,
    },
    /// The tool was named on the denylist.
    Denied {
        /// The tool the model asked for.
        tool_name: String,
    },
    /// The per-turn cap for this tool was already reached.
    CapExceeded {
        /// The tool the model asked for.
        tool_name: String,
        /// The configured cap.
        limit: u32,
        /// How many calls had already been made this turn.
        used: u32,
    },
    /// The approval hook refused it.
    Callback {
        /// The reason the hook gave.
        reason: String,
    },
    /// The audit trail is strict and its writer is degraded: an earlier
    /// entry never reached the disk, so a call that could change the world
    /// is refused rather than run unrecorded.
    AuditDegraded {
        /// How many appends have failed in this process.
        failures: u64,
        /// What the operating system said about the most recent one.
        last_error: Option<String>,
    },
    /// The audit trail is strict and its writer could not be asked at all.
    ///
    /// Distinct from degraded because nothing is known about the trail: the
    /// actor is gone, or did not answer in time. The guard fails closed
    /// either way.
    AuditUnreachable {
        /// Why the writer could not be consulted.
        error: String,
    },
}

impl fmt::Display for DenialReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAllowed { tool_name } => {
                write!(f, "the tool '{tool_name}' is not on this agent's allowlist",)
            }
            Self::Denied { tool_name } => {
                write!(f, "the tool '{tool_name}' is denied by policy")
            }
            Self::CapExceeded {
                tool_name,
                limit,
                used,
            } => write!(
                f,
                "the tool '{tool_name}' may be called at most {limit} time(s) per turn \
                 and has already been called {used} time(s)",
            ),
            Self::Callback { reason } => {
                write!(f, "the approval hook refused this call: {reason}")
            }
            Self::AuditDegraded {
                failures,
                last_error,
            } => {
                write!(
                    f,
                    "the audit trail is degraded ({failures} append failure(s)"
                )?;
                if let Some(error) = last_error {
                    write!(f, ", last error: {error}")?;
                }
                write!(
                    f,
                    "); mutating tools are refused until the trail is repaired and the agent \
                     restarted"
                )
            }
            Self::AuditUnreachable { error } => write!(
                f,
                "the audit trail is strict and its writer could not be consulted ({error}); \
                 mutating tools are refused until it answers"
            ),
        }
    }
}

/// Which rule, or which party, decided a tool call's fate.
///
/// Recorded in every audit entry so a reviewer can tell a blanket denylist
/// from a human pressing "no".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Decider {
    /// No policy was configured; the call ran unexamined.
    NoPolicy,
    /// A policy was in force and its rules raised no objection.
    ///
    /// Distinct from [`NoPolicy`](Self::NoPolicy) on purpose: "nobody was
    /// watching" and "the rules looked and were content" are different
    /// findings, and only the second one is evidence.
    Rules,
    /// An allowlist admitted or refused it.
    Allowlist,
    /// A denylist refused it.
    Denylist,
    /// A per-turn invocation cap refused it.
    PerTurnCap,
    /// The approval hook decided.
    Callback,
    /// The crash-recovery settlement decided.
    ///
    /// A resumed turn found the call mid-flight in a dead process's pending
    /// ledger and resolved it on the tool's idempotency declaration rather
    /// than through the gate — the gate already admitted it once, on the
    /// attempt that died.
    Settlement,
    /// The strict audit trail's guard refused it.
    ///
    /// The writer was degraded or unreachable and the tool is not declared
    /// idempotent, so the call was refused before any policy looked at it.
    AuditGuard,
}

impl fmt::Display for Decider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::NoPolicy => "no_policy",
            Self::Rules => "rules",
            Self::Allowlist => "allowlist",
            Self::Denylist => "denylist",
            Self::PerTurnCap => "per_turn_cap",
            Self::Callback => "callback",
            Self::Settlement => "settlement",
            Self::AuditGuard => "audit_guard",
        };
        f.write_str(name)
    }
}

/// How many times each tool has been invoked so far this turn.
///
/// Owned by the round loop, which is a single plain-async task, so this is a
/// plain map rather than shared state: there is nothing to share it with.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnCounts(BTreeMap<String, u32>);

impl TurnCounts {
    /// Creates an empty set of counts.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many times `tool_name` has been invoked this turn.
    #[must_use]
    pub fn used(&self, tool_name: &str) -> u32 {
        self.0.get(tool_name).copied().unwrap_or(0)
    }

    /// Records one invocation of `tool_name`.
    ///
    /// Saturates rather than wrapping: a turn that somehow reached
    /// `u32::MAX` calls should stay capped, not roll over to zero and start
    /// admitting calls again.
    pub fn record(&mut self, tool_name: &str) {
        let slot = self.0.entry(tool_name.to_string()).or_insert(0);
        *slot = slot.saturating_add(1);
    }
}

/// Matches a tool name against a policy pattern.
///
/// Exact match, or a single trailing `*` acting as a prefix wildcard. That is
/// enough to name a whole MCP server's tools (`mcp__fs__*`) without pulling in
/// a glob dependency or letting a policy file express something subtle enough
/// to be misread.
///
/// # Examples
///
/// ```
/// use acton_ai::policy::name_matches;
///
/// assert!(name_matches("bash", "bash"));
/// assert!(!name_matches("bash", "bash_extra"));
/// assert!(name_matches("mcp__fs__*", "mcp__fs__read_file"));
/// assert!(!name_matches("mcp__fs__*", "mcp__net__fetch"));
/// assert!(name_matches("*", "anything"));
/// ```
#[must_use]
pub fn name_matches(pattern: &str, name: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => pattern == name,
    }
}

/// Applies the rule-based half of the gate.
///
/// Precedence is denylist, then allowlist, then per-turn cap. The denylist
/// wins over the allowlist because when a tool is named by both, refusing is
/// the reading that cannot cause harm.
///
/// The cap compares against `used + 1`, so `limit = 3` admits exactly three
/// calls per turn.
///
/// Returns `Ok(())` when the rules admit the call, or the reason and the rule
/// that refused it.
pub(crate) fn evaluate_rules(
    allow: Option<&[String]>,
    deny: &[String],
    caps: &BTreeMap<String, u32>,
    tool_name: &str,
    counts: &TurnCounts,
) -> Result<(), (DenialReason, Decider)> {
    if deny.iter().any(|pattern| name_matches(pattern, tool_name)) {
        return Err((
            DenialReason::Denied {
                tool_name: tool_name.to_string(),
            },
            Decider::Denylist,
        ));
    }

    if let Some(allow) = allow {
        if !allow.iter().any(|pattern| name_matches(pattern, tool_name)) {
            return Err((
                DenialReason::NotAllowed {
                    tool_name: tool_name.to_string(),
                },
                Decider::Allowlist,
            ));
        }
    }

    // The cap is keyed by pattern too, so `mcp__fs__* = 5` caps a server as a
    // whole. The tightest matching cap wins, which keeps a specific entry
    // meaningful alongside a broad one.
    let cap = caps
        .iter()
        .filter(|(pattern, _)| name_matches(pattern, tool_name))
        .map(|(_, limit)| *limit)
        .min();

    if let Some(limit) = cap {
        let used = counts.used(tool_name);
        // `used >= limit`, not `used + 1 > limit`: the counts saturate at
        // `u32::MAX`, and the second form would overflow there rather than
        // keep refusing.
        if used >= limit {
            return Err((
                DenialReason::CapExceeded {
                    tool_name: tool_name.to_string(),
                    limit,
                    used,
                },
                Decider::PerTurnCap,
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(pairs: &[(&str, u32)]) -> BTreeMap<String, u32> {
        pairs
            .iter()
            .map(|(name, limit)| ((*name).to_string(), *limit))
            .collect()
    }

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_string()).collect()
    }

    #[test]
    fn exact_patterns_do_not_match_by_prefix() {
        assert!(name_matches("bash", "bash"));
        assert!(!name_matches("bash", "bashful"));
        assert!(!name_matches("bash", "rebash"));
    }

    #[test]
    fn a_trailing_star_matches_by_prefix() {
        assert!(name_matches("mcp__fs__*", "mcp__fs__read_file"));
        assert!(name_matches("mcp__fs__*", "mcp__fs__"));
        assert!(!name_matches("mcp__fs__*", "mcp__network__get"));
    }

    #[test]
    fn no_rules_admits_everything() {
        let verdict = evaluate_rules(None, &[], &BTreeMap::new(), "bash", &TurnCounts::new());
        assert!(verdict.is_ok());
    }

    #[test]
    fn an_allowlist_refuses_a_tool_it_does_not_name() {
        let allow = names(&["read_file"]);
        let (reason, decider) = evaluate_rules(
            Some(&allow),
            &[],
            &BTreeMap::new(),
            "mcp__fs__delete",
            &TurnCounts::new(),
        )
        .expect_err("a tool outside the allowlist must be refused");

        assert_eq!(decider, Decider::Allowlist);
        assert_eq!(
            reason,
            DenialReason::NotAllowed {
                tool_name: "mcp__fs__delete".to_string()
            }
        );
    }

    #[test]
    fn the_denylist_wins_over_the_allowlist() {
        let allow = names(&["bash"]);
        let deny = names(&["bash"]);
        let (_, decider) = evaluate_rules(
            Some(&allow),
            &deny,
            &BTreeMap::new(),
            "bash",
            &TurnCounts::new(),
        )
        .expect_err("a tool on both lists must be refused");

        assert_eq!(decider, Decider::Denylist);
    }

    #[test]
    fn a_cap_admits_exactly_its_limit_then_refuses() {
        let limits = caps(&[("bash", 2)]);
        let mut counts = TurnCounts::new();

        for call in 0..2 {
            evaluate_rules(None, &[], &limits, "bash", &counts)
                .unwrap_or_else(|_| panic!("call {call} must be admitted"));
            counts.record("bash");
        }

        let (reason, decider) = evaluate_rules(None, &[], &limits, "bash", &counts)
            .expect_err("the third call must be refused");

        assert_eq!(decider, Decider::PerTurnCap);
        assert_eq!(
            reason,
            DenialReason::CapExceeded {
                tool_name: "bash".to_string(),
                limit: 2,
                used: 2
            }
        );
    }

    #[test]
    fn a_cap_only_counts_its_own_tool() {
        let limits = caps(&[("bash", 1)]);
        let mut counts = TurnCounts::new();
        counts.record("read_file");
        counts.record("read_file");

        assert!(evaluate_rules(None, &[], &limits, "bash", &counts).is_ok());
    }

    #[test]
    fn the_tightest_matching_cap_wins() {
        let limits = caps(&[("mcp__fs__*", 5), ("mcp__fs__delete", 1)]);
        let mut counts = TurnCounts::new();
        counts.record("mcp__fs__delete");

        let (_, decider) = evaluate_rules(None, &[], &limits, "mcp__fs__delete", &counts)
            .expect_err("the specific cap of 1 must bind, not the broad cap of 5");

        assert_eq!(decider, Decider::PerTurnCap);
    }

    #[test]
    fn counts_saturate_rather_than_wrapping() {
        let mut counts = TurnCounts::new();
        for _ in 0..3 {
            counts.record("bash");
        }
        assert_eq!(counts.used("bash"), 3);
        assert_eq!(counts.used("never_called"), 0);
    }

    #[test]
    fn denial_reasons_render_something_a_model_can_act_on() {
        let reason = DenialReason::CapExceeded {
            tool_name: "bash".to_string(),
            limit: 2,
            used: 2,
        };
        let rendered = reason.to_string();
        assert!(rendered.contains("bash"), "names the tool: {rendered}");
        assert!(rendered.contains('2'), "names the limit: {rendered}");
    }

    #[test]
    fn an_audit_refusal_says_the_trail_is_degraded_and_for_how_long() {
        let with_error = DenialReason::AuditDegraded {
            failures: 3,
            last_error: Some("No space left on device".to_string()),
        }
        .to_string();
        assert!(
            with_error.contains("audit trail is degraded"),
            "{with_error}"
        );
        assert!(with_error.contains("3 append failure"), "{with_error}");
        assert!(
            with_error.contains("No space left on device"),
            "{with_error}"
        );
        assert!(with_error.contains("restarted"), "{with_error}");

        let without_error = DenialReason::AuditDegraded {
            failures: 1,
            last_error: None,
        }
        .to_string();
        assert!(!without_error.contains("last error"), "{without_error}");

        let unreachable = DenialReason::AuditUnreachable {
            error: "timed out".to_string(),
        }
        .to_string();
        assert!(
            unreachable.contains("could not be consulted"),
            "{unreachable}"
        );
        assert!(unreachable.contains("timed out"), "{unreachable}");
    }

    #[test]
    fn the_audit_guard_is_a_decider_in_its_own_right() {
        assert_eq!(Decider::AuditGuard.to_string(), "audit_guard");
        assert_eq!(
            serde_json::to_string(&Decider::AuditGuard).expect("serializes"),
            "\"audit_guard\""
        );
        assert_eq!(
            serde_json::from_str::<Decider>("\"audit_guard\"").expect("parses"),
            Decider::AuditGuard
        );
    }
}
