//! [`BuiltinExecutor`] — one built-in tool, callable, with its sandbox
//! routing already decided.
//!
//! This is the execution half of what `.use_builtins()` registers on a
//! prompt: the tool's in-process executor paired with the sandbox routing
//! its configuration asked for. Exposing it lets a downstream embedder wrap
//! built-in execution — with its own approval flow, logging, or protocol
//! adaptation — while keeping the exact sandboxing the framework would have
//! applied, instead of choosing between "call the raw executor and lose the
//! sandbox" and "re-implement the sandbox wiring".

use std::sync::Arc;

use serde_json::Value;

use crate::tools::definition::BoxedToolExecutor;
use crate::tools::sandbox::SandboxedExecution;
use crate::tools::tool_trait::ToolFuture;

/// A built-in tool's execution path, sandbox decision included.
///
/// Obtain one from [`ActonAI::builtin_executor`](crate::facade::ActonAI::builtin_executor),
/// which resolves the tool and applies the runtime's configuration: a tool
/// marked `sandboxed` on a runtime with a sandbox configured routes through
/// that sandbox, anything else runs in-process. Constructing one directly via
/// [`new`](Self::new) makes those decisions yours.
///
/// # This is the raw path
///
/// The tool-approval policy gate and the audit trail live in the prompt
/// loop, not in here. A wrapper registered on a prompt (via
/// [`PromptBuilder::tool`](crate::prompt::PromptBuilder::tool) or
/// [`add_tool`](crate::prompt::PromptBuilder::add_tool)) that delegates to
/// [`call`](Self::call) still gets both, because the loop wraps every
/// registered tool. Calling [`call`](Self::call) *outside* a prompt loop
/// runs the tool with no policy check and no audit entry — by design, since
/// an embedder driving its own loop owns that layer.
///
/// Cloning is cheap: the executor and the sandbox handle are shared.
#[derive(Clone)]
pub struct BuiltinExecutor {
    /// The tool's registered name, as the sandbox runner resolves it.
    tool_name: String,
    /// The in-process executor used when no sandbox routing applies.
    executor: Arc<BoxedToolExecutor>,
    /// The sandbox this tool's calls route through, when configured.
    sandbox: Option<SandboxedExecution>,
}

impl std::fmt::Debug for BuiltinExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuiltinExecutor")
            .field("tool_name", &self.tool_name)
            .field("sandboxed", &self.sandbox.is_some())
            .finish_non_exhaustive()
    }
}

impl BuiltinExecutor {
    /// Pairs a tool's executor with an explicit sandbox decision.
    ///
    /// For the runtime's own decision — sandbox iff the tool is configured
    /// `sandboxed` and a sandbox exists — use
    /// [`ActonAI::builtin_executor`](crate::facade::ActonAI::builtin_executor)
    /// instead. This constructor exists for embedders that manage their own
    /// [`SandboxedExecution`] (a test pinning the worker binary, a host with
    /// per-tenant sandbox configs) and for pairing a sandbox with executors
    /// obtained from [`BuiltinTools`](super::BuiltinTools) directly.
    ///
    /// When `sandbox` is `Some`, the named tool must be one the sandbox
    /// worker dispatches — see
    /// [`runner::supports`](crate::tools::sandbox::process::runner::supports).
    #[must_use]
    pub fn new(
        tool_name: impl Into<String>,
        executor: Arc<BoxedToolExecutor>,
        sandbox: Option<SandboxedExecution>,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            executor,
            sandbox,
        }
    }

    /// The tool's registered name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Whether calls route through a sandbox.
    #[must_use]
    pub fn is_sandboxed(&self) -> bool {
        self.sandbox.is_some()
    }

    /// Runs the tool against `args`, sandboxed when so configured.
    ///
    /// The returned future is the same shape a
    /// [`Tool::call`](crate::tools::Tool::call) returns, so a wrapper tool
    /// can hand it straight back:
    ///
    /// ```rust,ignore
    /// fn call(&self, args: serde_json::Value) -> ToolFuture {
    ///     // ... approval, logging, rewriting ...
    ///     self.inner.call(args)
    /// }
    /// ```
    ///
    /// The future resolves to a [`ToolError`](crate::tools::ToolError) when
    /// the sandbox cannot be created, the tool fails or times out, or — on
    /// the sandboxed path — the name is not one the sandbox worker knows.
    #[must_use = "the tool does not run until the returned future is awaited"]
    pub fn call(&self, args: Value) -> ToolFuture {
        match &self.sandbox {
            Some(sandbox) => {
                let sandbox = sandbox.clone();
                let tool_name = self.tool_name.clone();
                Box::pin(async move { sandbox.execute(&tool_name, args).await })
            }
            None => {
                let executor = Arc::clone(&self.executor);
                Box::pin(async move { executor.execute(args).await })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::builtins::BuiltinTools;
    use crate::tools::sandbox::StubSandboxFactory;
    use serde_json::json;

    fn calculate_executor() -> Arc<BoxedToolExecutor> {
        BuiltinTools::all()
            .get_executor("calculate")
            .expect("calculate is a built-in")
    }

    #[tokio::test]
    async fn an_unsandboxed_executor_runs_in_process() {
        let executor = BuiltinExecutor::new("calculate", calculate_executor(), None);

        assert!(!executor.is_sandboxed());
        let result = executor
            .call(json!({"expression": "2 + 2"}))
            .await
            .expect("in-process calculate succeeds");
        assert_eq!(result["result"], json!(4.0));
    }

    #[tokio::test]
    async fn a_sandboxed_executor_routes_through_the_sandbox() {
        let sandbox = SandboxedExecution::new(Arc::new(StubSandboxFactory::new()));
        let executor = BuiltinExecutor::new("bash", calculate_executor(), Some(sandbox));

        assert!(executor.is_sandboxed());
        let result = executor
            .call(json!({"command": "echo hi"}))
            .await
            .expect("the stub sandbox answers");
        // The stub echoes the dispatched tool name back, proving the call
        // went to the sandbox and not to the in-process executor.
        assert_eq!(result["code_preview"], json!("bash"));
        assert_eq!(result["status"], json!("stub"));
    }

    #[test]
    fn debug_names_the_tool_without_leaking_the_executor() {
        let executor = BuiltinExecutor::new("calculate", calculate_executor(), None);
        let rendered = format!("{executor:?}");
        assert!(rendered.contains("calculate"), "{rendered}");
        assert!(rendered.contains("sandboxed: false"), "{rendered}");
    }
}
