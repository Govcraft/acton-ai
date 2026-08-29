//! The tool-approval policy gate.
//!
//! One decision point sits between "the model asked for a tool" and "the tool
//! ran". It covers built-ins, `#[tool]` functions and MCP tools uniformly,
//! because by the time the prompt loop dispatches a call they are all the same
//! thing: an entry in one list of tool specs.
//!
//! A policy has two halves. The rules — an allowlist, a denylist and per-turn
//! invocation caps — are pure and configurable from TOML. The hook is an async
//! callback that can ask a human, and exists only in code because a callback
//! cannot be written in a config file.
//!
//! # Denial is an outcome, not an error
//!
//! When the gate refuses a call, the tool does not run, the reason is fed back
//! to the model as that call's tool result, and the turn continues — the same
//! shape the loop already uses to hand back a schema-validation failure. A
//! refused call is never an error and never aborts the turn.
//!
//! # Nothing configured changes nothing
//!
//! With no policy set, the loop performs no extra work at all: no allocation,
//! no await, no branch beyond one `Option` check. That is the default.
//!
//! # Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! let ai = ActonAI::builder()
//!     .anthropic_from_env()
//!     .tool_policy(
//!         ToolPolicy::new()
//!             .deny(["bash"])
//!             .cap_per_turn("read_file", 5),
//!     )
//!     .on_tool_approval(|invocation| async move {
//!         if invocation.tool_name.starts_with("mcp__") {
//!             ApprovalDecision::deny("MCP tools need a human first")
//!         } else {
//!             ApprovalDecision::Approve
//!         }
//!     })
//!     .launch()
//!     .await?;
//! ```

mod hook;
mod rules;

pub use hook::{ApprovalDecision, ApprovalFuture, ApprovalHookFn, ToolInvocation};
pub use rules::{name_matches, Decider, DenialReason, TurnCounts};

pub(crate) use rules::evaluate_rules;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// The rules and hook applied to every tool call in a turn.
///
/// Build one with [`ToolPolicy::new`] and the `deny` / `allow` /
/// `cap_per_turn` / `on_approval` methods, or load one from a `[tool_policy]`
/// section (see [`ToolPolicyFileConfig`](crate::config::ToolPolicyFileConfig)).
#[derive(Clone, Default)]
pub struct ToolPolicy {
    /// `None` admits every tool; `Some` admits only the patterns listed.
    allow: Option<Vec<String>>,
    /// Patterns refused outright. Checked before the allowlist.
    deny: Vec<String>,
    /// Per-turn invocation caps, keyed by the same patterns.
    caps: BTreeMap<String, u32>,
    /// The async approval hook, when one was installed.
    hook: Option<Arc<dyn ApprovalHookFn>>,
}

impl fmt::Debug for ToolPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolPolicy")
            .field("allow", &self.allow)
            .field("deny", &self.deny)
            .field("caps", &self.caps)
            .field("hook", &self.hook.is_some())
            .finish()
    }
}

impl ToolPolicy {
    /// Creates a policy that admits every tool and caps nothing.
    ///
    /// On its own this changes no behavior; add rules or a hook to make it
    /// bite.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Admits only tools matching these patterns.
    ///
    /// Calling this twice extends the list rather than replacing it, so a
    /// policy assembled in pieces does not silently lose an entry.
    #[must_use]
    pub fn allow<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let entries = patterns.into_iter().map(Into::into);
        self.allow.get_or_insert_with(Vec::new).extend(entries);
        self
    }

    /// Refuses tools matching these patterns, whatever the allowlist says.
    #[must_use]
    pub fn deny<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.deny.extend(patterns.into_iter().map(Into::into));
        self
    }

    /// Caps how many times a matching tool may be called in a single turn.
    ///
    /// The cap resets at the start of every turn, not every round: a model
    /// that spreads five `bash` calls across five rounds still hits a cap of
    /// three.
    #[must_use]
    pub fn cap_per_turn(mut self, pattern: impl Into<String>, limit: u32) -> Self {
        self.caps.insert(pattern.into(), limit);
        self
    }

    /// Installs the async approval hook.
    ///
    /// The hook runs only for calls the rules already admitted, so it is never
    /// asked about something a denylist has settled.
    #[must_use]
    pub fn on_approval<H>(mut self, hook: H) -> Self
    where
        H: ApprovalHookFn + 'static,
    {
        self.hook = Some(Arc::new(hook));
        self
    }

    /// Replaces the hook with one already behind an `Arc`.
    ///
    /// Used when resolving a builder's hook onto a policy that came from a
    /// config file, which is the one place the hook and the rules arrive
    /// separately.
    pub(crate) fn set_hook(&mut self, hook: Arc<dyn ApprovalHookFn>) {
        self.hook = Some(hook);
    }

    /// The allowlist, or `None` when every tool is admitted.
    #[must_use]
    pub fn allowlist(&self) -> Option<&[String]> {
        self.allow.as_deref()
    }

    /// The denylist. Empty when nothing is refused outright.
    #[must_use]
    pub fn denylist(&self) -> &[String] {
        &self.deny
    }

    /// The per-turn caps, keyed by pattern.
    #[must_use]
    pub fn caps(&self) -> &BTreeMap<String, u32> {
        &self.caps
    }

    /// Whether an approval hook is installed.
    #[must_use]
    pub fn has_hook(&self) -> bool {
        self.hook.is_some()
    }

    /// Classifies one tool call the way the gate itself would, without
    /// running anything.
    ///
    /// This exists for embedders that render policy state *before* a call is
    /// made — "this tool will run unprompted", "this one will ask", "this one
    /// is blocked" — without reimplementing the rules. It is not a parallel
    /// implementation of the gate: [`decide`](Self::decide), the function the
    /// prompt loop enforces with, is built on this exact classification, so
    /// the two cannot drift apart.
    ///
    /// `counts` is how many calls the turn has already made (per-turn caps
    /// depend on it). For a "would this ask at the start of a turn" UI, pass
    /// [`TurnCounts::new()`].
    ///
    /// The classification is pure and answers from the rules alone; it never
    /// consults the approval hook. A call the rules admit is
    /// [`NeedsApproval`](ToolClassification::NeedsApproval) when a hook is
    /// installed — the hook *will* be asked, and what it will say is not
    /// knowable without asking it — and
    /// [`AutoAllow`](ToolClassification::AutoAllow) otherwise.
    ///
    /// ```
    /// use acton_ai::policy::{ToolClassification, ToolPolicy, TurnCounts};
    ///
    /// let policy = ToolPolicy::new().deny(["bash"]);
    /// let counts = TurnCounts::new();
    ///
    /// assert!(matches!(
    ///     policy.classify("bash", &counts),
    ///     ToolClassification::Deny { .. }
    /// ));
    /// assert!(matches!(
    ///     policy.classify("read_file", &counts),
    ///     ToolClassification::AutoAllow
    /// ));
    /// ```
    #[must_use]
    pub fn classify(&self, tool_name: &str, counts: &TurnCounts) -> ToolClassification {
        if let Err((reason, decided_by)) = evaluate_rules(
            self.allow.as_deref(),
            &self.deny,
            &self.caps,
            tool_name,
            counts,
        ) {
            return ToolClassification::Deny { reason, decided_by };
        }
        if self.hook.is_some() {
            ToolClassification::NeedsApproval
        } else {
            ToolClassification::AutoAllow
        }
    }

    /// Decides one tool call.
    ///
    /// Built on [`classify`](Self::classify) — the same function embedders
    /// query for "will this ask" UI — so the gate and the introspection
    /// surface can never disagree. Classification first, because it is pure
    /// and cheap; the hook runs only for a call the rules admitted. `counts`
    /// is *not* updated here: the caller records the invocation, so that a
    /// call refused by the gate does not consume the budget it was refused
    /// for.
    pub(crate) async fn decide(
        &self,
        invocation: ToolInvocation,
        counts: &TurnCounts,
    ) -> GateOutcome {
        match self.classify(&invocation.tool_name, counts) {
            ToolClassification::Deny { reason, decided_by } => {
                GateOutcome::Deny { reason, decided_by }
            }
            ToolClassification::AutoAllow => GateOutcome::Allow {
                arguments: invocation.arguments,
                decided_by: Decider::Rules,
            },
            ToolClassification::NeedsApproval => {
                let Some(hook) = self.hook.as_ref() else {
                    // Unreachable: `classify` only says NeedsApproval when a
                    // hook is installed, and `self` is borrowed for the whole
                    // match. If it ever became reachable, the rules have
                    // already admitted the call and there is nobody to ask,
                    // so the rules' admission stands.
                    return GateOutcome::Allow {
                        arguments: invocation.arguments,
                        decided_by: Decider::Rules,
                    };
                };

                let proposed = invocation.arguments.clone();
                // `SyncFuture` because `ApprovalFuture` is `dyn Future +
                // Send` without `Sync`, and holding it bare across this
                // await would make every turn future that consults a hook
                // `!Sync` — unusable inside acton-reactive's `Reply::pending`.
                match sync_wrapper::SyncFuture::new(hook.call(invocation)).await {
                    ApprovalDecision::Approve => GateOutcome::Allow {
                        arguments: proposed,
                        decided_by: Decider::Callback,
                    },
                    ApprovalDecision::ApproveWith { arguments } => GateOutcome::Allow {
                        arguments,
                        decided_by: Decider::Callback,
                    },
                    ApprovalDecision::Deny { reason } => GateOutcome::Deny {
                        reason: DenialReason::Callback { reason },
                        decided_by: Decider::Callback,
                    },
                }
            }
        }
    }
}

/// What the gate would do with one tool call — queryable without making it.
///
/// Returned by [`ToolPolicy::classify`]. This is the taxonomy an embedder
/// needs to render approval state in a UI (an ACP agent daemon marking which
/// tools will prompt the user, for instance) without maintaining its own copy
/// of the allowlist/denylist/cap rules.
///
/// Marked `#[non_exhaustive]` so a new classification can be added without
/// breaking downstream `match`es.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolClassification {
    /// The rules admit the call and no approval hook is installed: it will
    /// run without asking anyone.
    AutoAllow,
    /// The rules admit the call, and the installed approval hook will be
    /// consulted before it runs. What the hook will say cannot be known
    /// without asking it — this is the "this will ask" state.
    NeedsApproval,
    /// The rules refuse the call; it will not run and the hook will not be
    /// asked.
    Deny {
        /// Why it would be refused.
        reason: DenialReason,
        /// Which rule would refuse it.
        decided_by: Decider,
    },
}

/// What the gate decided about one tool call.
#[derive(Debug, Clone)]
pub(crate) enum GateOutcome {
    /// Run it, with these — possibly rewritten — arguments.
    Allow {
        /// The arguments to actually execute with.
        arguments: serde_json::Value,
        /// Which rule or party admitted it.
        decided_by: Decider,
    },
    /// Do not run it. The reason goes back to the model.
    Deny {
        /// Why it was refused.
        reason: DenialReason,
        /// Which rule or party refused it.
        decided_by: Decider,
    },
}

/// Renders a denial as the tool result the model reads.
///
/// Mirrors the shape the loop already uses for structured-output repair: a
/// plain sentence saying what happened and what to do instead, delivered as
/// the result of the call the model made, so the conversation stays
/// well-formed and the turn continues.
#[must_use]
pub(crate) fn denial_feedback(reason: &DenialReason) -> String {
    match reason {
        // The guard's verdict does not change for the rest of the session,
        // so the model is told that plainly rather than left to probe.
        DenialReason::AuditDegraded { .. } | DenialReason::AuditUnreachable { .. } => format!(
            "This tool call was not executed: {reason}. Do not retry it, and do not \
             attempt other tools that modify anything: they will be refused for the \
             rest of this session. Read-only tools remain available. Tell the user \
             what you could not do."
        ),
        _ => format!(
            "This tool call was not executed: {reason}. Do not retry it. \
             Continue using the tools that remain available to you, or tell the \
             user what you could not do."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CorrelationId, TurnId};
    use serde_json::json;

    fn invocation(tool_name: &str) -> ToolInvocation {
        ToolInvocation {
            tool_name: tool_name.to_string(),
            arguments: json!({"value": 1}),
            tool_call_id: "toolu_01".to_string(),
            correlation_id: CorrelationId::new(),
            turn_id: TurnId::new(),
        }
    }

    #[tokio::test]
    async fn an_empty_policy_admits_everything() {
        let policy = ToolPolicy::new();
        let outcome = policy.decide(invocation("bash"), &TurnCounts::new()).await;

        assert!(matches!(outcome, GateOutcome::Allow { .. }));
    }

    #[tokio::test]
    async fn the_hook_is_not_consulted_for_a_call_the_rules_refused() {
        let policy = ToolPolicy::new()
            .deny(["bash"])
            .on_approval(|_: ToolInvocation| async move {
                panic!("the hook must not run for a denylisted tool")
            });

        let outcome = policy.decide(invocation("bash"), &TurnCounts::new()).await;

        assert!(matches!(
            outcome,
            GateOutcome::Deny {
                decided_by: Decider::Denylist,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn a_hook_rewrite_is_what_gets_executed() {
        let policy = ToolPolicy::new().on_approval(|_: ToolInvocation| async move {
            ApprovalDecision::approve_with(json!({"value": 99}))
        });

        let outcome = policy.decide(invocation("echo"), &TurnCounts::new()).await;

        match outcome {
            GateOutcome::Allow {
                arguments,
                decided_by,
            } => {
                assert_eq!(arguments, json!({"value": 99}));
                assert_eq!(decided_by, Decider::Callback);
            }
            GateOutcome::Deny { .. } => panic!("the hook approved this call"),
        }
    }

    #[tokio::test]
    async fn a_hook_denial_carries_its_reason() {
        let policy = ToolPolicy::new()
            .on_approval(|_: ToolInvocation| async move { ApprovalDecision::deny("not today") });

        let outcome = policy.decide(invocation("echo"), &TurnCounts::new()).await;

        match outcome {
            GateOutcome::Deny { reason, decided_by } => {
                assert_eq!(decided_by, Decider::Callback);
                assert_eq!(
                    reason,
                    DenialReason::Callback {
                        reason: "not today".to_string()
                    }
                );
            }
            GateOutcome::Allow { .. } => panic!("the hook refused this call"),
        }
    }

    #[test]
    fn allow_extends_rather_than_replacing() {
        let policy = ToolPolicy::new().allow(["a"]).allow(["b"]);
        assert_eq!(
            policy.allowlist(),
            Some(["a".to_string(), "b".to_string()].as_slice())
        );
    }

    #[test]
    fn debug_reports_whether_a_hook_is_installed_without_naming_it() {
        let policy = ToolPolicy::new()
            .on_approval(|_: ToolInvocation| async move { ApprovalDecision::Approve });
        let rendered = format!("{policy:?}");
        assert!(rendered.contains("hook: true"), "{rendered}");
    }

    /// The gate outcome `decide` produced, reduced to the same taxonomy
    /// `classify` speaks, so the two can be compared.
    async fn gate_view(policy: &ToolPolicy, tool: &str, counts: &TurnCounts) -> ToolClassification {
        match policy.decide(invocation(tool), counts).await {
            GateOutcome::Allow {
                decided_by: Decider::Rules,
                ..
            } => ToolClassification::AutoAllow,
            GateOutcome::Allow {
                decided_by: Decider::Callback,
                ..
            } => ToolClassification::NeedsApproval,
            GateOutcome::Allow { .. } => panic!("the gate admitted with an unexpected decider"),
            GateOutcome::Deny { reason, decided_by } => {
                ToolClassification::Deny { reason, decided_by }
            }
        }
    }

    #[tokio::test]
    async fn classification_agrees_with_the_gate_for_the_default_case() {
        let policy = ToolPolicy::new();
        let counts = TurnCounts::new();

        assert_eq!(
            policy.classify("anything", &counts),
            ToolClassification::AutoAllow
        );
        assert_eq!(
            gate_view(&policy, "anything", &counts).await,
            policy.classify("anything", &counts),
        );
    }

    #[tokio::test]
    async fn classification_agrees_with_the_gate_for_the_allowlist() {
        let policy = ToolPolicy::new().allow(["read_file"]);
        let counts = TurnCounts::new();

        for tool in ["read_file", "bash"] {
            assert_eq!(
                policy.classify(tool, &counts),
                gate_view(&policy, tool, &counts).await,
                "classify and the gate must agree about '{tool}'",
            );
        }
        assert!(matches!(
            policy.classify("bash", &counts),
            ToolClassification::Deny {
                decided_by: Decider::Allowlist,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn classification_agrees_with_the_gate_for_the_denylist() {
        let policy = ToolPolicy::new().deny(["bash"]);
        let counts = TurnCounts::new();

        for tool in ["bash", "read_file"] {
            assert_eq!(
                policy.classify(tool, &counts),
                gate_view(&policy, tool, &counts).await,
                "classify and the gate must agree about '{tool}'",
            );
        }
        assert!(matches!(
            policy.classify("bash", &counts),
            ToolClassification::Deny {
                decided_by: Decider::Denylist,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn classification_agrees_with_the_gate_for_a_per_turn_cap() {
        let policy = ToolPolicy::new().cap_per_turn("bash", 1);
        let mut counts = TurnCounts::new();

        // Under the cap: both admit.
        assert_eq!(
            policy.classify("bash", &counts),
            gate_view(&policy, "bash", &counts).await,
        );
        assert_eq!(
            policy.classify("bash", &counts),
            ToolClassification::AutoAllow
        );

        // At the cap: both refuse, for the same reason.
        counts.record("bash");
        assert_eq!(
            policy.classify("bash", &counts),
            gate_view(&policy, "bash", &counts).await,
        );
        assert!(matches!(
            policy.classify("bash", &counts),
            ToolClassification::Deny {
                decided_by: Decider::PerTurnCap,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn a_hook_reads_as_needs_approval_before_the_hook_is_ever_run() {
        let policy = ToolPolicy::new()
            .on_approval(|_: ToolInvocation| async move { ApprovalDecision::Approve });
        let counts = TurnCounts::new();

        // classify never runs the hook, so "needs approval" is knowable
        // without side effects; the gate, which does run it, lands on the
        // same call-will-be-asked classification.
        assert_eq!(
            policy.classify("bash", &counts),
            ToolClassification::NeedsApproval
        );
        assert_eq!(
            gate_view(&policy, "bash", &counts).await,
            ToolClassification::NeedsApproval
        );
    }

    #[tokio::test]
    async fn a_denylisted_tool_is_deny_even_with_a_hook_installed() {
        // Precedence surfaces in the classification exactly as it does in
        // the gate: the hook is never the answer for a call the rules refuse.
        let policy = ToolPolicy::new()
            .deny(["bash"])
            .on_approval(|_: ToolInvocation| async move { ApprovalDecision::Approve });
        let counts = TurnCounts::new();

        assert_eq!(
            policy.classify("bash", &counts),
            gate_view(&policy, "bash", &counts).await,
        );
        assert!(matches!(
            policy.classify("bash", &counts),
            ToolClassification::Deny {
                decided_by: Decider::Denylist,
                ..
            }
        ));
    }

    #[test]
    fn denial_feedback_tells_the_model_not_to_retry() {
        let feedback = denial_feedback(&DenialReason::Denied {
            tool_name: "bash".to_string(),
        });
        assert!(feedback.contains("bash"), "{feedback}");
        assert!(feedback.contains("Do not retry"), "{feedback}");
    }
}
