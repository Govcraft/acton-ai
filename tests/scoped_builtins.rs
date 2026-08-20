//! One boundary, observed by every filesystem-capable builtin.
//!
//! A host that serves several workspaces from one runtime cannot let its tools
//! fall back on the process working directory and the system temp directory,
//! because those belong to the host rather than to the caller. These tests
//! pin the confined behaviour of each builtin, in-process and through the
//! sandbox, against a tree with the shapes that actually catch mistakes: a
//! nested directory, a sibling, a name that shares a prefix, a symlink out,
//! and `/tmp`.

use acton_ai::prelude::*;
use acton_ai::tools::builtins;
use acton_ai::tools::sandbox::{HardeningMode, ProcessSandboxConfig, SandboxedExecution};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// A workspace and the places just outside it.
struct Tree {
    _dir: tempfile::TempDir,
    root: PathBuf,
    nested: PathBuf,
    sibling: PathBuf,
}

fn tree() -> Tree {
    let dir = tempfile::tempdir().expect("a temp dir");
    let base = dir.path().canonicalize().expect("the temp dir resolves");
    let root = base.join("root");
    let nested = root.join("nested");
    let sibling = base.join("sibling");
    std::fs::create_dir_all(&nested).expect("creates the tree");
    std::fs::create_dir_all(&sibling).expect("creates the sibling");
    std::fs::write(root.join("inside.txt"), "inside\n").expect("writes");
    std::fs::write(nested.join("deeper.txt"), "deeper\n").expect("writes");
    std::fs::write(sibling.join("outside.txt"), "outside\n").expect("writes");
    Tree {
        _dir: dir,
        root,
        nested,
        sibling,
    }
}

/// Runs a builtin confined to `root`, as a host would build it.
async fn confined(
    name: &str,
    root: &Path,
    args: Value,
) -> Result<Value, acton_ai::tools::ToolError> {
    let executor = builtins::scoped_executor(name, root)
        .unwrap_or_else(|| panic!("{name} is a filesystem-capable builtin"));
    executor.execute(args).await
}

#[tokio::test]
async fn every_reading_tool_sees_inside_its_root() {
    let tree = tree();

    let read = confined(
        "read_file",
        &tree.root,
        json!({"path": tree.root.join("inside.txt").to_str().unwrap()}),
    )
    .await
    .expect("a file in the root is readable");
    assert!(read["content"]
        .as_str()
        .expect("content")
        .contains("inside"));

    let listing = confined(
        "list_directory",
        &tree.root,
        json!({"path": tree.root.to_str().unwrap()}),
    )
    .await
    .expect("the root itself is listable");
    assert!(listing.to_string().contains("inside.txt"));

    let globbed = confined("glob", &tree.root, json!({"pattern": "**/*.txt"}))
        .await
        .expect("an unqualified glob starts at the root");
    let matches = globbed["matches"].to_string();
    assert!(matches.contains("inside.txt"), "got: {matches}");
    assert!(
        !matches.contains("outside.txt"),
        "the glob reached out of the root: {matches}"
    );

    let grepped = confined("grep", &tree.root, json!({"pattern": "deeper"}))
        .await
        .expect("an unqualified grep starts at the root");
    assert!(grepped.to_string().contains("deeper.txt"));
}

#[tokio::test]
async fn a_sibling_directory_is_outside_every_tool() {
    let tree = tree();
    let outside = tree.sibling.join("outside.txt");
    let outside = outside.to_str().unwrap();

    for (name, args) in [
        ("read_file", json!({"path": outside})),
        (
            "write_file",
            json!({"path": outside, "content": "overwritten"}),
        ),
        (
            "edit_file",
            json!({"path": outside, "old_string": "outside", "new_string": "changed"}),
        ),
        (
            "list_directory",
            json!({"path": tree.sibling.to_str().unwrap()}),
        ),
        (
            "glob",
            json!({"pattern": "*.txt", "path": tree.sibling.to_str().unwrap()}),
        ),
        (
            "grep",
            json!({"pattern": "outside", "path": tree.sibling.to_str().unwrap()}),
        ),
    ] {
        let error = confined(name, &tree.root, args)
            .await
            .expect_err("a sibling of the root is not inside it");
        assert!(
            error.to_string().contains("outside allowed directories"),
            "{name} refused for the wrong reason: {error}"
        );
    }

    assert_eq!(
        std::fs::read_to_string(tree.sibling.join("outside.txt")).expect("still there"),
        "outside\n",
        "a refused write must not have happened anyway"
    );
}

#[tokio::test]
async fn the_system_temp_directory_is_not_implied_by_a_root() {
    // The default validator allows `std::env::temp_dir()`. A confined one
    // must not: on a shared host that is somebody else's directory.
    let tree = tree();
    let scratch = std::env::temp_dir().join("acton-ai-scoped-builtins-probe.txt");

    let error = confined(
        "write_file",
        &tree.root,
        json!({"path": scratch.to_str().unwrap(), "content": "should not land"}),
    )
    .await
    .expect_err("/tmp is outside a confined root");

    assert!(error.to_string().contains("outside allowed directories"));
    assert!(!scratch.exists(), "the write must not have happened");
}

#[cfg(unix)]
#[tokio::test]
async fn a_symlink_out_of_the_root_does_not_widen_it() {
    let tree = tree();
    let link = tree.root.join("way-out.txt");
    std::os::unix::fs::symlink(tree.sibling.join("outside.txt"), &link).expect("creates the link");

    let error = confined(
        "read_file",
        &tree.root,
        json!({"path": link.to_str().unwrap()}),
    )
    .await
    .expect_err("a link inside the root is not a file inside the root");

    assert!(
        error.to_string().contains("outside allowed directories"),
        "got: {error}"
    );
}

#[tokio::test]
async fn a_name_that_merely_starts_with_the_root_is_outside_it() {
    let tree = tree();
    let decoy = tree.root.with_file_name("rootless");
    std::fs::create_dir_all(&decoy).expect("creates the decoy");
    std::fs::write(decoy.join("f.txt"), "decoy\n").expect("writes");

    let error = confined(
        "read_file",
        &tree.root,
        json!({"path": decoy.join("f.txt").to_str().unwrap()}),
    )
    .await
    .expect_err("sharing a name prefix is not being inside");

    assert!(error.to_string().contains("outside allowed directories"));
}

#[tokio::test]
async fn writes_and_edits_land_inside_the_root() {
    let tree = tree();
    let created = tree.nested.join("created.txt");

    confined(
        "write_file",
        &tree.root,
        json!({"path": created.to_str().unwrap(), "content": "hello\n"}),
    )
    .await
    .expect("a write inside the root succeeds");
    assert_eq!(
        std::fs::read_to_string(&created).expect("the file exists"),
        "hello\n"
    );

    confined(
        "edit_file",
        &tree.root,
        json!({"path": created.to_str().unwrap(), "old_string": "hello", "new_string": "goodbye"}),
    )
    .await
    .expect("an edit inside the root succeeds");
    assert_eq!(
        std::fs::read_to_string(&created).expect("the file exists"),
        "goodbye\n"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn a_confined_shell_starts_in_its_root_and_cannot_leave_it() {
    let tree = tree();

    let result = confined("bash", &tree.root, json!({"command": "pwd"}))
        .await
        .expect("a confined shell runs");
    assert_eq!(
        Path::new(result["stdout"].as_str().expect("stdout").trim_end()),
        tree.root,
        "an unqualified command runs in the root, not the host's directory"
    );

    let error = confined(
        "bash",
        &tree.root,
        json!({"command": "pwd", "cwd": tree.sibling.to_str().unwrap()}),
    )
    .await
    .expect_err("a working directory outside the root is refused");
    assert!(error.to_string().contains("outside allowed directories"));
}

#[tokio::test]
async fn two_sessions_hold_two_boundaries_at_once() {
    // Concurrency is the case a shared, runtime-level validator gets wrong:
    // each session's executor has to carry its own root, not consult one.
    let first = tree();
    let second = tree();

    let (a, b) = tokio::join!(
        confined(
            "read_file",
            &first.root,
            json!({"path": second.root.join("inside.txt").to_str().unwrap()}),
        ),
        confined(
            "read_file",
            &second.root,
            json!({"path": second.root.join("inside.txt").to_str().unwrap()}),
        ),
    );

    assert!(
        a.is_err(),
        "one session's root must not admit another session's files"
    );
    assert!(b.is_ok(), "and its own session still reads its own files");
}

// =============================================================================
// The same boundary, across the sandbox's process edge
// =============================================================================

#[cfg(unix)]
#[tokio::test]
async fn a_sandboxed_tool_is_confined_to_the_root_it_was_given() {
    let tree = tree();
    let sandbox = SandboxedExecution::process_with_exe(
        PathBuf::from(env!("CARGO_BIN_EXE_acton-ai")),
        ProcessSandboxConfig::new()
            .with_timeout(Duration::from_secs(15))
            .with_hardening(HardeningMode::Off),
    )
    .expect("the crate's binary must canonicalize");

    // Inside: the write reaches the real workspace, which the unscoped
    // sandbox could not do at all — it ran in a throwaway directory.
    let target = tree.nested.join("from-the-child.txt");
    sandbox
        .execute_in(
            Some(&tree.root),
            "write_file",
            json!({"path": target.to_str().unwrap(), "content": "child wrote this\n"}),
        )
        .await
        .expect("a confined write inside the root succeeds");
    assert_eq!(
        std::fs::read_to_string(&target).expect("the child wrote a real file"),
        "child wrote this\n"
    );

    // Outside: refused in the child, by the same validator the in-process
    // path uses.
    let escape = tree.sibling.join("outside.txt");
    let error = sandbox
        .execute_in(
            Some(&tree.root),
            "write_file",
            json!({"path": escape.to_str().unwrap(), "content": "escaped"}),
        )
        .await
        .expect_err("the child must hold the same boundary as the parent");
    assert!(
        error.to_string().contains("outside allowed directories"),
        "got: {error}"
    );
    assert_eq!(
        std::fs::read_to_string(&escape).expect("still there"),
        "outside\n"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn a_prompt_confined_to_a_root_registers_confined_builtins() {
    let tree = tree();
    let ai = ActonAI::builder()
        .app_name("scoped-builtins")
        .provider(ProviderConfig::openai_compatible(
            "http://127.0.0.1:9".to_string(),
            "unused-model",
        ))
        .with_builtins()
        .manual_builtins()
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let executor = ai
        .builtin_executor_in("read_file", &tree.root)
        .expect("read_file is a builtin");
    assert_eq!(executor.root(), Some(tree.root.as_path()));

    executor
        .call(json!({"path": tree.root.join("inside.txt").to_str().unwrap()}))
        .await
        .expect("its own root is readable");

    let error = executor
        .call(json!({"path": tree.sibling.join("outside.txt").to_str().unwrap()}))
        .await
        .expect_err("and nothing else is");
    assert!(error.to_string().contains("outside allowed directories"));

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// Hardened: a confined child still has to be able to run something
// =============================================================================

/// Whether this kernel actually enforces landlock. Best-effort hardening logs
/// and continues where it does not, so the escape assertions below only mean
/// something on a kernel that has it.
#[cfg(all(unix, feature = "sandbox-hardening"))]
fn landlock_is_enforced() -> bool {
    std::fs::read_to_string("/sys/kernel/security/lsm")
        .map(|lsms| lsms.split(',').any(|lsm| lsm.trim() == "landlock"))
        .unwrap_or(false)
}

#[cfg(all(unix, feature = "sandbox-hardening"))]
#[tokio::test]
async fn a_hardened_child_can_still_spawn_a_shell_inside_its_root() {
    let tree = tree();
    std::fs::write(tree.root.join("greeting.txt"), "hardened hello\n").expect("writes");
    let sandbox = SandboxedExecution::process_with_exe(
        PathBuf::from(env!("CARGO_BIN_EXE_acton-ai")),
        ProcessSandboxConfig::new()
            .with_timeout(Duration::from_secs(15))
            .with_hardening(HardeningMode::BestEffort),
    )
    .expect("the crate's binary must canonicalize");

    // Spawning a process opens `/dev/null` for the null stdin. A landlock
    // ruleset that omits it makes every `bash` call fail with EACCES before
    // the command is even parsed, which is a sandbox that cannot run
    // anything rather than a sandbox that confines what it runs.
    let result = sandbox
        .execute_in(
            Some(&tree.root),
            "bash",
            json!({"command": "cat greeting.txt"}),
        )
        .await
        .expect("a hardened child must still be able to run a command");
    assert_eq!(
        result["stdout"].as_str().expect("bash reports stdout"),
        "hardened hello\n"
    );
}

#[cfg(all(unix, feature = "sandbox-hardening"))]
#[tokio::test]
async fn a_hardened_child_cannot_write_outside_its_root() {
    if !landlock_is_enforced() {
        return; // nothing to assert: hardening degraded to a warning
    }
    let tree = tree();
    let sandbox = SandboxedExecution::process_with_exe(
        PathBuf::from(env!("CARGO_BIN_EXE_acton-ai")),
        ProcessSandboxConfig::new()
            .with_timeout(Duration::from_secs(15))
            .with_hardening(HardeningMode::BestEffort),
    )
    .expect("the crate's binary must canonicalize");

    // The shell runs arbitrary commands, so the boundary here is the kernel's
    // rather than the path validator's: the sibling is unreachable even
    // though no tool argument named it.
    let escape = tree.sibling.join("outside.txt");
    let result = sandbox
        .execute_in(
            Some(&tree.root),
            "bash",
            json!({"command": format!("cat {0} ; echo escaped > {0}", escape.display())}),
        )
        .await
        .expect("the command runs; it is the kernel that refuses the access");
    assert!(
        result["stderr"]
            .as_str()
            .expect("bash reports stderr")
            .contains("Permission denied"),
        "got: {result}"
    );
    assert_eq!(
        std::fs::read_to_string(&escape).expect("still there"),
        "outside\n",
        "the sibling must be untouched"
    );
}
