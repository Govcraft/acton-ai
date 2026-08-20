//! Downstream-embedder tests for the public sandbox execution surface.
//!
//! These are written the way an embedding crate (an ACP agent daemon, say)
//! would use the API: obtain [`SandboxedExecution`] / [`BuiltinExecutor`]
//! handles, wrap them in its own tool types, and run real work through the
//! re-exec'd `acton-ai` binary — no crate-private plumbing anywhere.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use acton_ai::prelude::*;
use acton_ai::tools::sandbox::process::runner;
use acton_ai::tools::sandbox::{HardeningMode, ProcessSandboxConfig, SandboxedExecution};
use acton_ai::tools::BuiltinExecutor;
use serde_json::json;

/// A provider that is configured but never contacted: the tests below assert
/// on wiring, not on turns, so nothing ever dials this address.
fn unused_provider() -> ProviderConfig {
    ProviderConfig::openai_compatible("http://127.0.0.1:9".to_string(), "unused-model")
}

/// Path to the crate's main binary, which honours the sandbox re-exec
/// contract. The test binary itself does not, so every test that actually
/// executes pins the worker explicitly.
fn acton_ai_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_acton-ai"))
}

fn sandbox_config() -> ProcessSandboxConfig {
    ProcessSandboxConfig::new()
        .with_timeout(Duration::from_secs(15))
        // Hardening would confine the child's landlock view and interact
        // badly with the test tempdirs; the hardened path has its own
        // coverage.
        .with_hardening(HardeningMode::Off)
}

/// What a downstream tool executor looks like: it owns its own concerns
/// (here: counting invocations, standing in for an approval flow) and
/// delegates the actual work to the framework's sandbox path.
struct CountingShell {
    sandbox: SandboxedExecution,
    invocations: Arc<AtomicUsize>,
}

impl Tool for CountingShell {
    fn name(&self) -> &'static str {
        "counting_shell"
    }

    fn description(&self) -> &'static str {
        "Runs a shell command under the process sandbox, counting calls."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        })
    }

    fn call(&self, args: serde_json::Value) -> ToolFuture {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        let sandbox = self.sandbox.clone();
        Box::pin(async move { sandbox.execute("bash", args).await })
    }
}

// =============================================================================
// 1. A custom downstream tool runs its work under the process sandbox
// =============================================================================

#[tokio::test]
async fn a_downstream_tool_wrapper_executes_under_the_process_sandbox() {
    let sandbox = SandboxedExecution::process_with_exe(acton_ai_binary(), sandbox_config())
        .expect("the crate's binary must canonicalize");
    let invocations = Arc::new(AtomicUsize::new(0));
    let tool = CountingShell {
        sandbox,
        invocations: Arc::clone(&invocations),
    };

    let result = tool
        .call(json!({"command": "echo embedder"}))
        .await
        .expect("the sandboxed command must succeed");

    let stdout = result
        .get("stdout")
        .and_then(|v| v.as_str())
        .expect("bash results carry stdout");
    assert_eq!(stdout.trim_end(), "embedder");
    assert_eq!(
        invocations.load(Ordering::SeqCst),
        1,
        "the wrapper's own concern ran around the sandboxed work"
    );
}

// =============================================================================
// 2. The facade hands out the same sandbox decision builtins get
// =============================================================================

#[tokio::test]
async fn the_facade_reports_the_builtin_sandbox_routing() {
    let ai = ActonAI::builder()
        .app_name("embedder-sandbox-wiring")
        .provider(unused_provider())
        .with_builtin_tools(&["bash", "read_file"])
        .with_process_sandbox_config(sandbox_config())
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let sandbox = ai
        .sandboxed_execution()
        .expect("a configured sandbox must be handed out");
    assert!(
        sandbox.is_available(),
        "the factory re-execs the current binary, which exists"
    );

    let bash = ai
        .builtin_executor("bash")
        .expect("bash was configured as a builtin");
    assert!(bash.is_sandboxed(), "bash is configured `sandboxed`");
    assert_eq!(bash.tool_name(), "bash");

    let read_file = ai
        .builtin_executor("read_file")
        .expect("read_file was configured as a builtin");
    assert!(
        !read_file.is_sandboxed(),
        "read_file is not a sandboxed tool, even with a sandbox configured"
    );

    assert!(
        ai.builtin_executor("no_such_tool").is_none(),
        "an unconfigured name yields no executor"
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn without_a_sandbox_the_handle_is_absent_and_builtins_run_in_process() {
    let ai = ActonAI::builder()
        .app_name("embedder-no-sandbox")
        .provider(unused_provider())
        .with_builtin_tools(&["bash", "read_file"])
        .launch()
        .await
        .expect("launching the runtime must succeed");

    assert!(
        ai.sandboxed_execution().is_none(),
        "no sandbox was configured, so no handle exists"
    );

    let bash = ai
        .builtin_executor("bash")
        .expect("bash was configured as a builtin");
    assert!(
        !bash.is_sandboxed(),
        "with no sandbox configured, even `sandboxed` tools run in-process"
    );

    // The in-process path works end to end: read a real file through the
    // exact executor the prompt loop would register.
    let file = tempfile::NamedTempFile::new().expect("a temp file");
    std::fs::write(file.path(), "embedder in-process read\n").expect("writes");
    let read_file = ai
        .builtin_executor("read_file")
        .expect("read_file was configured as a builtin");
    let result = read_file
        .call(json!({"path": file.path().to_str().unwrap()}))
        .await
        .expect("the in-process read must succeed");
    assert!(
        result["content"]
            .as_str()
            .expect("read_file results carry content")
            .contains("embedder in-process read"),
        "got: {result}"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 3. A BuiltinExecutor paired with a sandbox executes in the child
// =============================================================================

#[tokio::test]
async fn a_builtin_executor_paired_with_a_sandbox_executes_in_the_child() {
    let sandbox = SandboxedExecution::process_with_exe(acton_ai_binary(), sandbox_config())
        .expect("the crate's binary must canonicalize");
    let executor = acton_ai::tools::builtins::BuiltinTools::all()
        .get_executor("bash")
        .expect("bash is a builtin");

    let bash = BuiltinExecutor::new("bash", executor, Some(sandbox));
    assert!(bash.is_sandboxed());

    let result = bash
        .call(json!({"command": "echo from-the-child"}))
        .await
        .expect("the sandboxed builtin must succeed");
    assert_eq!(
        result
            .get("stdout")
            .and_then(|v| v.as_str())
            .expect("bash results carry stdout")
            .trim_end(),
        "from-the-child"
    );
}

// =============================================================================
// 4. The runner advertises exactly what it dispatches
// =============================================================================

#[test]
fn the_runner_advertises_the_dispatchable_tool_set() {
    for name in ["bash", "write_file", "edit_file"] {
        assert!(runner::supports(name), "{name} must be dispatchable");
    }
    for name in ["read_file", "grep", "glob", "calculate", "mcp__x__y"] {
        assert!(!runner::supports(name), "{name} must not be dispatchable");
    }
}

#[tokio::test]
async fn a_tool_the_runner_does_not_support_fails_inside_the_child() {
    let sandbox = SandboxedExecution::process_with_exe(acton_ai_binary(), sandbox_config())
        .expect("the crate's binary must canonicalize");

    let error = sandbox
        .execute("read_file", json!({"path": "/etc/hostname"}))
        .await
        .expect_err("read_file cannot cross the process boundary");
    assert!(
        error.to_string().contains("read_file"),
        "the error should name the unknown tool, got: {error}"
    );
}
