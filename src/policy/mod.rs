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

    /// Decides one tool call.
    ///
    /// Rules first — they are pure and cheap — then the hook, which is only
    /// consulted for a call the rules admitted. `counts` is *not* updated
    /// here: the caller records the invocation, so that a call refused by the
    /// gate does not consume the budget it was refused for.
    pub(crate) async fn decide(
        &self,
        invocation: ToolInvocation,
        counts: &TurnCounts,
    ) -> GateOutcome {
        if let Err((reason, decided_by)) = evaluate_rules(
            self.allow.as_deref(),
            &self.deny,
            &self.caps,
            &invocation.tool_name,
            counts,
        ) {
            return GateOutcome::Deny { reason, decided_by };
        }

        let Some(hook) = self.hook.as_ref() else {
            return GateOutcome::Allow {
                arguments: invocation.arguments,
                decided_by: Decider::Rules,
            };
        };

        let proposed = invocation.arguments.clone();
        match hook.call(invocation).await {
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
    format!(
        "This tool call was not executed: {reason}. Do not retry it. \
         Continue using the tools that remain available to you, or tell the \
         user what you could not do."
    )
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

    #[test]
    fn denial_feedback_tells_the_model_not_to_retry() {
        let feedback = denial_feedback(&DenialReason::Denied {
            tool_name: "bash".to_string(),
        });
        assert!(feedback.contains("bash"), "{feedback}");
        assert!(feedback.contains("Do not retry"), "{feedback}");
    }
}
