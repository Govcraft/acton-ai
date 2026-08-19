//! The callback half of the policy gate.
//!
//! Rules cannot express "ask a human", so a policy may also carry an async
//! hook. It sees the tool name and the arguments the model proposed, and it
//! may approve, approve a rewritten set of arguments, or refuse with a reason.

use crate::types::{CorrelationId, TurnId};
use std::future::Future;
use std::pin::Pin;

/// What the approval hook is shown before a tool runs.
///
/// The IDs are here so a human-in-the-loop prompt can be correlated with the
/// turn that raised it, and with the audit entry that records the answer.
#[derive(Debug, Clone)]
pub struct ToolInvocation {
    /// The tool the model asked for, in the name the model sees — which for
    /// an MCP tool is the prefixed `mcp__{server}__{tool}` form.
    pub tool_name: String,
    /// The arguments the model proposed, before any rewriting.
    pub arguments: serde_json::Value,
    /// The correlation ID of the round that requested the call.
    pub correlation_id: CorrelationId,
    /// The turn the call belongs to.
    pub turn_id: TurnId,
}

/// What an approval hook decided.
///
/// Marked `#[non_exhaustive]` so further decisions (a deferral, say) can be
/// added without breaking downstream `match`es.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ApprovalDecision {
    /// Run the tool with the arguments the model proposed.
    Approve,
    /// Run the tool, but with these arguments instead.
    ///
    /// The rewritten arguments are what gets executed *and* what gets
    /// recorded, so the audit trail describes what actually ran.
    ApproveWith {
        /// The arguments to run instead.
        arguments: serde_json::Value,
    },
    /// Do not run the tool. The reason is fed back to the model.
    Deny {
        /// Why the call was refused, in words the model can act on.
        reason: String,
    },
}

impl ApprovalDecision {
    /// Refuses a call with the given reason.
    #[must_use]
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// Approves a call, substituting these arguments.
    #[must_use]
    pub fn approve_with(arguments: serde_json::Value) -> Self {
        Self::ApproveWith { arguments }
    }
}

/// The future an approval hook returns.
pub type ApprovalFuture = Pin<Box<dyn Future<Output = ApprovalDecision> + Send>>;

/// An async approval hook.
///
/// Implemented automatically for any `Fn(ToolInvocation) -> Future<Output =
/// ApprovalDecision>`, so callers write an async closure and never name this
/// trait.
///
/// The hook is awaited on the prompt loop's own task, between the model asking
/// for a tool and the tool running, so a hook that blocks holds up the turn.
/// That is exactly what a human-in-the-loop prompt wants; anything else should
/// return promptly.
pub trait ApprovalHookFn: Send + Sync {
    /// Decides whether this invocation may proceed.
    fn call(&self, invocation: ToolInvocation) -> ApprovalFuture;
}

impl<F, Fut> ApprovalHookFn for F
where
    F: Fn(ToolInvocation) -> Fut + Send + Sync,
    Fut: Future<Output = ApprovalDecision> + Send + 'static,
{
    fn call(&self, invocation: ToolInvocation) -> ApprovalFuture {
        Box::pin(self(invocation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn invocation() -> ToolInvocation {
        ToolInvocation {
            tool_name: "bash".to_string(),
            arguments: json!({"command": "ls"}),
            correlation_id: CorrelationId::new(),
            turn_id: TurnId::new(),
        }
    }

    #[tokio::test]
    async fn an_async_closure_is_a_hook() {
        let hook = |invocation: ToolInvocation| async move {
            if invocation.tool_name == "bash" {
                ApprovalDecision::deny("no shells today")
            } else {
                ApprovalDecision::Approve
            }
        };

        let decision = ApprovalHookFn::call(&hook, invocation()).await;

        assert_eq!(
            decision,
            ApprovalDecision::Deny {
                reason: "no shells today".to_string()
            }
        );
    }

    #[tokio::test]
    async fn a_hook_can_rewrite_the_arguments() {
        let hook = |_: ToolInvocation| async move {
            ApprovalDecision::approve_with(json!({"command": "ls -la"}))
        };

        let decision = ApprovalHookFn::call(&hook, invocation()).await;

        assert_eq!(
            decision,
            ApprovalDecision::ApproveWith {
                arguments: json!({"command": "ls -la"})
            }
        );
    }
}
