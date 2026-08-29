//! End-to-end coverage for the sandbox's declared path lists.
//!
//! The hardened child's landlock ruleset grants the system directories,
//! `$TMPDIR` and the session root, and nothing else. A user-installed
//! toolchain — `uv` at `~/.local/bin`, a `cargo` shim, a pnpm store — is
//! therefore found by the shell on `PATH` and then refused by the kernel,
//! which surfaces as a bare `Permission denied` with no mention of landlock
//! anywhere in it.
//!
//! These tests pin both halves of that behaviour against the real kernel:
//! the refusal when nothing is declared, and the success once the operator
//! declares the directory. Each case probes the refusal first and skips
//! itself when the refusal does not happen, because a kernel without
//! landlock runs the child unconfined under `BestEffort` and there is then
//! no boundary to widen.

#![cfg(all(target_os = "linux", feature = "sandbox-hardening"))]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::time::Duration;

use acton_ai::tools::sandbox::{
    HardeningMode, ProcessSandbox, ProcessSandboxConfig, ProcessSandboxFactory, Sandbox,
    SandboxFactory,
};
use serde_json::{json, Value};

fn acton_ai_binary() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_acton-ai"))
}

/// A hardened sandbox, configured by `customize`.
async fn hardened_sandbox(
    customize: impl FnOnce(ProcessSandboxConfig) -> ProcessSandboxConfig,
) -> Box<dyn Sandbox> {
    let config = customize(
        ProcessSandboxConfig::new()
            .with_timeout(Duration::from_millis(15_000))
            .with_hardening(HardeningMode::BestEffort),
    );
    ProcessSandboxFactory::with_exe(acton_ai_binary(), config)
        .expect("factory must build against the crate's own binary")
        .create()
        .await
        .expect("sandbox create")
}

/// Runs one shell command in the sandbox and returns the tool's response.
async fn run(sandbox: &dyn Sandbox, command: &str) -> Value {
    sandbox
        .execute("bash", json!({ "command": command }))
        .await
        .expect("the sandbox child must answer, whatever the command's fate")
}

fn succeeded(response: &Value) -> bool {
    response
        .get("success")
        .and_then(Value::as_bool)
        .expect("the bash tool always reports success")
}

fn stdout(response: &Value) -> String {
    response
        .get("stdout")
        .and_then(Value::as_str)
        .expect("the bash tool always reports stdout")
        .trim_end()
        .to_string()
}

/// Writes an executable script that announces itself, outside every
/// directory the built-in ruleset grants.
fn script_in(dir: &Path) -> std::path::PathBuf {
    let script = dir.join("announce");
    fs::write(&script, "#!/bin/sh\necho declared-and-running\n").expect("write script");
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).expect("chmod script");
    script
}

#[tokio::test]
async fn a_binary_outside_the_ruleset_is_refused_until_it_is_declared() {
    let dir = tempfile::tempdir().expect("tempdir");
    let script = script_in(dir.path());
    let command = script.display().to_string();

    let undeclared = hardened_sandbox(|cfg| cfg).await;
    let refused = run(undeclared.as_ref(), &command).await;
    if succeeded(&refused) {
        eprintln!("skipping: this kernel does not enforce landlock, so nothing is confined");
        return;
    }
    assert_eq!(
        refused.get("exit_code").and_then(Value::as_i64),
        Some(126),
        "a shell reports a binary it may not execute as 126: {refused}"
    );

    let declared = hardened_sandbox(|cfg| cfg.with_read_exec_paths([dir.path()])).await;
    let allowed = run(declared.as_ref(), &command).await;
    assert!(
        succeeded(&allowed),
        "declaring the directory must make the same command run: {allowed}"
    );
    assert_eq!(stdout(&allowed), "declared-and-running");
}

#[tokio::test]
async fn a_directory_outside_the_ruleset_is_unwritable_until_it_is_declared() {
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("cache-entry");
    let command = format!("touch {}", target.display());

    let undeclared = hardened_sandbox(|cfg| cfg).await;
    let refused = run(undeclared.as_ref(), &command).await;
    if succeeded(&refused) {
        eprintln!("skipping: this kernel does not enforce landlock, so nothing is confined");
        return;
    }

    let declared = hardened_sandbox(|cfg| cfg.with_read_write_paths([dir.path()])).await;
    let allowed = run(declared.as_ref(), &command).await;
    assert!(
        succeeded(&allowed),
        "declaring the directory must make the write land: {allowed}"
    );
    assert!(target.exists(), "the file the child touched must be there");
}

/// A path list the operator got wrong should fail as a configuration error
/// at startup, not as an unexplained denial three tool calls later.
#[test]
fn a_relative_declared_path_is_refused_before_any_child_is_spawned() {
    let config = ProcessSandboxConfig::new().with_read_exec_paths(["relative/bin"]);
    let err = ProcessSandboxFactory::with_exe(acton_ai_binary(), config)
        .expect_err("the factory validates the config it is handed");
    assert!(
        err.to_string().contains("absolute"),
        "the operator must be told what is wrong with it, got: {err}"
    );
}

/// A declared directory that does not exist narrows the ruleset rather than
/// widening it, so it is a warning and not a startup failure — including
/// under `Enforce`, which must not abort a deployment over a cache directory
/// its tools have not created yet.
#[tokio::test]
async fn a_missing_declared_path_does_not_abort_the_child() {
    let config = ProcessSandboxConfig::new()
        .with_timeout(Duration::from_millis(15_000))
        .with_hardening(HardeningMode::Enforce)
        .with_read_exec_paths(["/nonexistent/toolchain/bin"]);
    let sandbox = ProcessSandbox::new(acton_ai_binary(), config);

    let response = run(&sandbox, "echo still-here").await;
    assert!(succeeded(&response), "the child must still run: {response}");
    assert_eq!(stdout(&response), "still-here");
}
