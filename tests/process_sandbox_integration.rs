//! End-to-end tests for the process sandbox.
//!
//! These tests spawn the `acton-ai` binary as a sandbox child via
//! `ProcessSandboxFactory::with_exe`. They cover:
//!
//! - the happy path (bash echo returns the expected stdout),
//! - deadline enforcement (a `sleep` command is killed by the parent), and
//! - the re-exec contract (the child recognizes
//!   `ACTON_AI_SANDBOX_RUNNER=1` and routes into the runner instead of the
//!   regular CLI).

use std::path::PathBuf;
use std::time::Duration;

use acton_ai::tools::sandbox::{ProcessSandboxConfig, ProcessSandboxFactory, SandboxFactory};
use serde_json::json;

/// Path to the crate's main binary. Cargo substitutes `-` with `_` in the
/// CARGO_BIN_EXE_* env var name.
fn acton_ai_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_acton-ai"))
}

fn config_with_timeout(ms: u64) -> ProcessSandboxConfig {
    ProcessSandboxConfig::new()
        .with_timeout(Duration::from_millis(ms))
        // Hardening would confine the child's landlock view and interact
        // badly with the test tempdirs. Integration coverage for hardened
        // mode lives in a follow-up task.
        .with_hardening(acton_ai::tools::sandbox::HardeningMode::Off)
}

#[tokio::test]
async fn process_sandbox_bash_echo_returns_stdout() {
    let factory = ProcessSandboxFactory::with_exe(acton_ai_binary(), config_with_timeout(15_000))
        .expect("factory must build against the crate's own binary");
    let sandbox = factory.create().await.expect("sandbox create");
    let result = sandbox
        .execute("bash", json!({"command": "echo hi"}))
        .await
        .expect("bash echo must succeed");
    let stdout = result
        .get("stdout")
        .and_then(|v| v.as_str())
        .expect("response must include stdout");
    assert_eq!(stdout.trim_end(), "hi");
}

#[tokio::test]
async fn process_sandbox_deadline_kills_runaway_child() {
    let factory = ProcessSandboxFactory::with_exe(acton_ai_binary(), config_with_timeout(500))
        .expect("factory must build");
    let sandbox = factory.create().await.expect("sandbox create");
    let err = sandbox
        .execute("bash", json!({"command": "sleep 30"}))
        .await
        .expect_err("sleep 30 with a 500ms budget must time out");
    let msg = err.to_string();
    assert!(
        msg.contains("timeout") || msg.contains("exceeded"),
        "error should surface a timeout, got: {msg}"
    );
}

/// Proves the re-exec contract: without the runner env var, running the
/// binary against the sandbox protocol would fall through to the CLI's clap
/// parser and fail. The factory always sets ACTON_AI_SANDBOX_RUNNER=1, so
/// the happy path already implicitly verifies the routing. This test goes a
/// step further and confirms the env-var constant that main.rs reads is
/// exactly the one the runner exports.
#[test]
fn runner_env_var_matches_main_guard_contract() {
    assert_eq!(
        acton_ai::tools::sandbox::process::runner::env_vars::RUNNER,
        "ACTON_AI_SANDBOX_RUNNER",
        "main.rs is hard-coded against this name; keep them in lockstep"
    );
}
