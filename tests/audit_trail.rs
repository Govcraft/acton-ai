//! End-to-end tests for the tamper-evident audit trail.
//!
//! These drive the real stack — facade, prompt loop, audit actor, filesystem —
//! and then read the file back with the same public API an auditor would use,
//! so what they assert is what is actually on disk.
//!
//! The last section goes one step further and shells out to the real
//! `acton-ai` binary, because the library and the CLI are two independent
//! readers of the same chain and nothing else forces them to agree.
//!
//! # Determinism
//!
//! Nothing sleeps. `ActonAI::audit_head()` is an `ask` on the audit actor, and
//! mailboxes are FIFO: its reply cannot arrive until every invocation recorded
//! before it has been sealed and written.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, tool_named, MockServer, Round};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

/// A tool the scripted rounds can call.
fn tool_definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        idempotent: false,
        name: name.to_string(),
        description: "Echoes its argument back.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }),
    }
}

/// Reads the trail back the way `acton-ai audit verify` does.
fn read_trail(path: &Path) -> Vec<AuditEntry> {
    let contents = std::fs::read_to_string(path).expect("the trail must exist");
    acton_ai::audit::parse_entries(&contents).expect("the trail must parse")
}

/// Writes entries back as JSONL — the file a tamperer would leave behind.
fn write_trail(path: &Path, entries: &[AuditEntry]) {
    let mut contents = String::new();
    for entry in entries {
        contents.push_str(&entry.to_jsonl().expect("an entry must serialize"));
        contents.push('\n');
    }
    std::fs::write(path, contents).expect("the trail must be writable");
}

/// Exit code the CLI uses for "the chain does not verify".
///
/// Duplicated from `acton_ai::cli::error::exit_code` on purpose: this number
/// is the interface a cron job or compliance check keys on, so a test should
/// fail when it changes rather than quietly follow it.
const EXIT_AUDIT_CHAIN_BROKEN: i32 = 3;

/// What the `acton-ai` binary said about a trail.
struct CliVerdict {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

impl CliVerdict {
    /// The JSON report the CLI prints when the chain holds.
    fn report(&self) -> Value {
        serde_json::from_str(&self.stdout).unwrap_or_else(|error| {
            panic!(
                "`audit verify --json` must print one JSON object, got {error}\n\
                 stdout: {}\nstderr: {}",
                self.stdout, self.stderr
            )
        })
    }
}

/// Runs the real `acton-ai` binary the way an auditor would.
///
/// `--file` is always passed, so the command never consults a config file or
/// the developer's own trail: the test is hermetic regardless of what is in
/// `~/.config/acton-ai/`.
fn run_audit_verify(path: &Path) -> CliVerdict {
    let output = Command::new(env!("CARGO_BIN_EXE_acton-ai"))
        .args(["--json", "audit", "verify", "--file"])
        .arg(path)
        .output()
        .expect("the acton-ai binary must be runnable");

    CliVerdict {
        exit_code: output.status.code().expect("the CLI must exit, not signal"),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Launches a runtime auditing to `path`, drives `tool_calls` invocations
/// through it, and shuts it down cleanly.
///
/// Returns the live [`ChainHead`] the runtime reported. Taking it before
/// shutdown is the barrier: `audit_head()` is an `ask` on the audit actor and
/// mailboxes are FIFO, so its reply cannot arrive until every entry recorded
/// before it is sealed and on disk.
async fn record_trail(app_name: &str, path: &Path, tool_calls: usize) -> ChainHead {
    let mut rounds: Vec<Round> = (0..tool_calls)
        .map(|i| {
            Round::tool_call(
                &format!("call_{i}"),
                "echo",
                json!({"value": i.to_string()}),
            )
        })
        .collect();
    rounds.push(Round::text("done"));

    let server = MockServer::start(rounds).await;

    let ai = ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .audit_to(path)
        .launch()
        .await
        .expect("launching the runtime must succeed");

    if tool_calls > 0 {
        ai.prompt("echo")
            .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
            .collect()
            .await
            .expect("the turn must complete");
        assert_eq!(
            server.request_count(),
            tool_calls + 1,
            "each tool round plus the final answer"
        );
    }

    let head = ai
        .audit_head()
        .await
        .expect("an audited runtime must report a head");

    ai.shutdown().await.expect("clean shutdown");

    head
}

// =============================================================================
// 1. A successful invocation is recorded and the chain verifies
// =============================================================================

#[tokio::test]
async fn every_executed_tool_call_lands_in_the_trail() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})),
        Round::text("done"),
    ])
    .await;

    let ai = ActonAI::builder()
        .app_name("audit-success")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .audit(AuditConfig::new(&path).with_user("acct:alice"))
        .launch()
        .await
        .expect("launching the runtime must succeed");

    ai.prompt("echo")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    // The barrier: this cannot answer until the entry is written.
    let head = ai.audit_head().await.expect("the trail must report a head");
    assert_eq!(head.entries, 1);
    assert_eq!(head.sequence, 1);

    assert_eq!(server.request_count(), 2, "the tool round and the answer");

    let entries = read_trail(&path);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].tool_name, "echo");
    assert_eq!(entries[0].user.as_deref(), Some("acct:alice"));
    assert_eq!(
        entries[0].response_size_bytes,
        Some(u64::try_from(serde_json::to_vec(&json!({"value": "hi"})).unwrap().len()).unwrap()),
        "the size covers the complete serialized response"
    );

    // The provider's own ID for the call, not one the loop invented. This is
    // the join key between the trail and the lifecycle events an observer
    // saw live, so a value that only the audit path knows would make the two
    // records impossible to line up.
    assert_eq!(entries[0].tool_call_id, "call_1");

    // The name in the trail must be the name the model was offered. An
    // auditor matching entries against a tool inventory has only the name to
    // go on, and a schema with a `$ref` in it would not be checkable against
    // the recorded arguments either.
    let first = server.requests().remove(0);
    let offered = tool_named(&first, "echo").expect("the tool must be offered to the model");
    assert_eq!(offered["function"]["name"], json!(entries[0].tool_name));
    assert!(
        !contains_ref(&offered["function"]["parameters"]),
        "tool schemas must be self-contained on the wire: {offered}"
    );
    assert_eq!(entries[0].prev_hash, GENESIS_HASH);
    assert!(entries[0].decision.approved);
    assert!(matches!(entries[0].outcome, AuditOutcome::Success { .. }));
    assert_eq!(
        verify_chain(&entries)
            .expect("a fresh trail must verify")
            .hash,
        head.hash,
        "the head on disk and the head in the actor must agree"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 2. A refused invocation is recorded too
// =============================================================================

#[tokio::test]
async fn a_denied_call_is_recorded_as_denied_with_the_rule_that_refused_it() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let server = MockServer::start(vec![
        Round::tool_call("call_1", "bash", json!({"command": "rm -rf /"})),
        Round::text("understood"),
    ])
    .await;

    let ai = ActonAI::builder()
        .app_name("audit-denied")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .tool_policy(ToolPolicy::new().deny(["bash"]))
        .audit_to(&path)
        .launch()
        .await
        .expect("launching the runtime must succeed");

    ai.prompt("clean up")
        .with_tool(tool_definition("bash"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    ai.audit_head().await.expect("the trail must report a head");

    let entries = read_trail(&path);
    assert_eq!(
        entries.len(),
        1,
        "a call that never ran is still a call that was made"
    );
    assert!(!entries[0].decision.approved);
    assert_eq!(entries[0].decision.decided_by, Decider::Denylist);
    assert!(matches!(entries[0].outcome, AuditOutcome::Denied { .. }));
    assert_eq!(entries[0].response_size_bytes, None);

    // A refused call is identified exactly like one that ran. Anything else
    // would leave the one kind of entry an investigator most wants to trace
    // as the one kind that cannot be traced.
    assert_eq!(entries[0].tool_call_id, "call_1");

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 3. Secrets are redacted before they are written
// =============================================================================

#[tokio::test]
async fn secret_bearing_arguments_never_reach_the_file() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let server = MockServer::start(vec![
        Round::tool_call(
            "call_1",
            "echo",
            json!({"api_key": "sk-live-do-not-log-me", "value": "safe"}),
        ),
        Round::text("done"),
    ])
    .await;

    let ai = ActonAI::builder()
        .app_name("audit-redaction")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .audit_to(&path)
        .launch()
        .await
        .expect("launching the runtime must succeed");

    ai.prompt("echo")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    ai.audit_head().await.expect("the trail must report a head");

    let raw = std::fs::read_to_string(&path).expect("the trail must exist");
    assert!(
        !raw.contains("sk-live-do-not-log-me"),
        "a secret must never appear anywhere in the file: {raw}"
    );

    let entries = read_trail(&path);
    assert_eq!(entries[0].arguments["api_key"], json!("[redacted]"));
    assert_eq!(
        entries[0].arguments["value"],
        json!("safe"),
        "redaction must be surgical, not wholesale"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 3b. The trail records the call that ran, not the one that was proposed
// =============================================================================

#[tokio::test]
async fn a_hook_rewrite_is_what_the_trail_records() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "proposed"})),
        Round::text("done"),
    ])
    .await;

    let ai = ActonAI::builder()
        .app_name("audit-rewrite")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .on_tool_approval(|_invocation| async move {
            ApprovalDecision::approve_with(json!({"value": "rewritten"}))
        })
        .audit_to(&path)
        .launch()
        .await
        .expect("launching the runtime must succeed");

    ai.prompt("echo")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    ai.audit_head().await.expect("the trail must report a head");

    let entries = read_trail(&path);
    assert_eq!(
        entries[0].arguments,
        json!({"value": "rewritten"}),
        "the trail must describe what ran, not what was asked for"
    );
    assert_eq!(entries[0].decision.decided_by, Decider::Callback);

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 4. Tampering is evident
// =============================================================================

#[tokio::test]
async fn editing_an_entry_in_the_middle_breaks_the_chain_at_that_entry() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "one"})),
        Round::tool_call("call_2", "echo", json!({"value": "two"})),
        Round::tool_call("call_3", "echo", json!({"value": "three"})),
        Round::text("done"),
    ])
    .await;

    let ai = ActonAI::builder()
        .app_name("audit-tamper")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .audit_to(&path)
        .launch()
        .await
        .expect("launching the runtime must succeed");

    ai.prompt("echo three times")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    ai.audit_head().await.expect("the trail must report a head");
    ai.shutdown().await.expect("clean shutdown");

    let entries = read_trail(&path);
    assert_eq!(entries.len(), 3, "the test needs a middle entry to edit");
    assert!(
        verify_chain(&entries).is_ok(),
        "the trail must start intact"
    );

    // Rewrite the middle entry the way somebody covering their tracks would:
    // change what the tool was, leave everything else alone.
    let mut tampered = entries.clone();
    tampered[1].tool_name = "something_harmless".to_string();

    let broken = verify_chain(&tampered).expect_err("an edited entry must be caught");
    assert_eq!(
        broken.sequence, 2,
        "the break must be reported at the entry that was edited"
    );
    assert!(matches!(broken.kind, ChainBreakKind::HashMismatch { .. }));

    // And re-sealing that entry does not help: its successor still points at
    // the hash it used to have.
    tampered[1].hash = tampered[1].recompute_hash();

    let broken = verify_chain(&tampered).expect_err("re-sealing must not repair the chain");
    assert_eq!(
        broken.sequence, 3,
        "the break moves to the successor whose back-pointer is now wrong"
    );
    assert!(matches!(
        broken.kind,
        ChainBreakKind::PrevHashMismatch { .. }
    ));
}

// =============================================================================
// 5. A restarted process resumes the chain rather than starting a second one
// =============================================================================

#[tokio::test]
async fn a_second_run_appends_to_the_chain_the_first_one_left() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    for app in ["audit-resume-1", "audit-resume-2"] {
        let server = MockServer::start(vec![
            Round::tool_call("call_1", "echo", json!({"value": "hi"})),
            Round::text("done"),
        ])
        .await;

        let ai = ActonAI::builder()
            .app_name(app)
            .provider(ProviderConfig::openai_compatible(
                server.base_url().to_string(),
                "mock-model",
            ))
            .audit_to(&path)
            .launch()
            .await
            .expect("launching the runtime must succeed");

        ai.prompt("echo")
            .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
            .collect()
            .await
            .expect("the turn must complete");

        ai.audit_head().await.expect("the trail must report a head");
        ai.shutdown().await.expect("clean shutdown");
    }

    let entries = read_trail(&path);
    assert_eq!(entries.len(), 2, "the second run must append, not truncate");
    assert_eq!(entries[1].sequence, 2);
    assert_eq!(
        entries[1].prev_hash, entries[0].hash,
        "the resumed chain must link back to the first run's last entry"
    );
    verify_chain(&entries).expect("a resumed chain must still verify");
}

// =============================================================================
// 6. Off by default
// =============================================================================

#[tokio::test]
async fn without_configuration_nothing_is_recorded_and_asking_says_so() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})),
        Round::text("done"),
    ])
    .await;
    let ai = mock_llm::runtime_pointed_at(&server, "audit-absent").await;

    assert!(!ai.is_audited());

    ai.prompt("echo")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    // Deliberately an error rather than an empty chain: "nothing happened" and
    // "nothing was being recorded" are opposite findings.
    let error = ai
        .audit_head()
        .await
        .expect_err("an unconfigured trail must not pretend to be an empty one");
    assert!(error.to_string().contains("no audit trail is configured"));

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 7. The TOML half
// =============================================================================

#[tokio::test]
async fn a_toml_audit_section_arms_the_trail() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("nested").join("audit.jsonl");

    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})),
        Round::text("done"),
    ])
    .await;

    let toml = format!(
        "{}\n[audit]\npath = '{}'\nuser = 'acct:bob'\n",
        mock_llm::provider_toml("mock", &server, "mock-model"),
        path.display()
    );
    let config = acton_ai::config::from_str(&toml).expect("the config must parse");
    let ai = ActonAI::builder()
        .app_name("audit-toml")
        .apply_config(config)
        .expect("the config must apply")
        .launch()
        .await
        .expect("launching the runtime must succeed");

    assert!(ai.is_audited());

    ai.prompt("echo")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    ai.audit_head().await.expect("the trail must report a head");

    // The parent directory did not exist: resolution had to create it, or the
    // entry would have gone nowhere and nobody would have noticed.
    let entries = read_trail(&path);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].user.as_deref(), Some("acct:bob"));

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 8. The library and the CLI agree about the same trail
// =============================================================================
//
// There are two readers of the hash chain and they share almost no code.
// `ActonAI::audit_head()` asks the live audit actor what it has written and
// gets an in-memory answer. `acton-ai audit verify` is a *separate process*
// that opens the JSONL file, re-parses every entry, and re-walks the chain
// from genesis. That split is the point — an auditor who does not trust the
// running process can check its work — but it also means nothing forces the
// two answers to match. A change to how an entry is sealed, serialized, or
// counted could make the live head and the on-disk head diverge while every
// test above still passed. These pin them together.

#[tokio::test]
async fn the_cli_reports_the_same_head_the_runtime_does() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let head = record_trail("audit-parity", &path, 3).await;
    assert_eq!(head.entries, 3, "the test needs a chain to compare");

    let verdict = run_audit_verify(&path);

    assert_eq!(
        verdict.exit_code, 0,
        "an intact trail must exit 0\nstderr: {}",
        verdict.stderr
    );

    let report = verdict.report();

    assert_eq!(report["verified"], json!(true));
    assert_eq!(
        report["entries"],
        json!(head.entries),
        "the CLI counted a different number of entries than the runtime wrote"
    );
    assert_eq!(
        report["head_sequence"],
        json!(head.sequence),
        "the CLI and the runtime disagree about the last sequence number"
    );
    // The load-bearing one: the head hash is derived from every entry's full
    // contents, so agreement here means the bytes on disk hash to exactly what
    // the actor believed it wrote.
    assert_eq!(
        report["head_hash"],
        json!(head.hash),
        "the CLI re-hashed the trail to something other than the runtime's head"
    );
    // The identity the runtime sealed under is the one the file walk finds
    // in the entries, and it is the one the sidecar beside the trail holds.
    let trail_id = head
        .trail_id
        .as_ref()
        .expect("a spawned trail has an identity");
    assert_eq!(
        report["trail_id"],
        json!(trail_id.to_string()),
        "the CLI and the runtime disagree about the trail's identity"
    );
    let sidecar = std::fs::read_to_string(dir.path().join("audit.jsonl.trail"))
        .expect("the sidecar is written at launch");
    assert_eq!(sidecar.trim(), trail_id.to_string());

    // The report names the file it actually checked, which is what makes it
    // usable as evidence rather than just a green tick.
    assert_eq!(report["path"], json!(path.to_string_lossy().into_owned()));
}

#[tokio::test]
async fn a_tampered_trail_fails_the_cli_where_the_library_says_it_should() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let head = record_trail("audit-parity-tamper", &path, 3).await;
    assert_eq!(head.entries, 3, "the test needs a middle entry to edit");

    // The trail must start clean, or the tamper proves nothing.
    let clean = run_audit_verify(&path);
    assert_eq!(clean.exit_code, 0, "stderr: {}", clean.stderr);

    // Rewrite the middle entry the way somebody covering their tracks would:
    // change what the tool was, leave everything else alone.
    let mut entries = read_trail(&path);
    entries[1].tool_name = "something_harmless".to_string();
    write_trail(&path, &entries);

    // What the library says about it, in this process.
    let expected = verify_chain(&entries).expect_err("an edited entry must be caught");
    assert_eq!(expected.sequence, 2);
    assert!(matches!(expected.kind, ChainBreakKind::HashMismatch { .. }));

    // What the CLI says about it, in its own process, reading the file.
    let verdict = run_audit_verify(&path);

    assert_eq!(
        verdict.exit_code, EXIT_AUDIT_CHAIN_BROKEN,
        "a tampered trail must exit {EXIT_AUDIT_CHAIN_BROKEN} so a cron job notices\n\
         stdout: {}\nstderr: {}",
        verdict.stdout, verdict.stderr
    );
    assert!(
        verdict.stderr.contains(&expected.to_string()),
        "the CLI must name the same break the library found.\n\
         library: {expected}\nCLI stderr: {}",
        verdict.stderr
    );
    assert!(
        verdict.stdout.trim().is_empty(),
        "a broken chain must not also print a success report on stdout, got: {}",
        verdict.stdout
    );
}

#[tokio::test]
async fn an_armed_but_unused_trail_verifies_as_the_genesis_head() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    // Audit configured, no tool ever called. "Audited and nothing happened"
    // and "not audited" are opposite findings — that is why `audit_head()`
    // errors in the second case — so the CLI must not blur them either.
    let head = record_trail("audit-parity-empty", &path, 0).await;
    assert_eq!(head.entries, 0);
    assert_eq!(head.sequence, 0);
    assert_eq!(head.hash, GENESIS_HASH);

    let verdict = run_audit_verify(&path);

    assert_eq!(
        verdict.exit_code, 0,
        "an empty trail is intact, not broken\nstderr: {}",
        verdict.stderr
    );

    let report = verdict.report();
    assert_eq!(report["verified"], json!(true));
    assert_eq!(report["entries"], json!(0));
    assert_eq!(report["head_sequence"], json!(0));
    assert_eq!(
        report["head_hash"],
        json!(GENESIS_HASH),
        "an empty chain's head is the genesis hash, in both readers"
    );
}

#[test]
fn an_unreadable_file_is_not_reported_as_tampering() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");
    std::fs::write(&path, "this is not JSON\n").expect("writes");

    let verdict = run_audit_verify(&path);

    // "I cannot read this" and "this has been altered" are different findings
    // and only one of them is an incident. Exit code 3 is reserved for the
    // incident, so a garbled file must not raise one.
    assert_ne!(
        verdict.exit_code, EXIT_AUDIT_CHAIN_BROKEN,
        "an unparseable file must not be reported as a broken chain\nstderr: {}",
        verdict.stderr
    );
    assert_ne!(verdict.exit_code, 0, "it is still an error");
}

// =============================================================================
// 6. Durability and writer health
// =============================================================================

/// A tool declared idempotent, which the strict guard leaves alone.
fn idempotent_tool_definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        idempotent: true,
        ..tool_definition(name)
    }
}

/// Makes the trail unappendable from under a running writer.
///
/// The writer holds the trail's lock on an open descriptor and appends by
/// *path*, so replacing the file with a directory of the same name is exactly
/// the failure a disk yanked out mid-session produces: the next open fails,
/// the lock is untouched. This is the same simulation
/// `ensure_trail_exists_reports_an_unwritable_destination` uses, and it needs
/// no privileges — unlike `chattr +i` or a full tmpfs.
fn make_unappendable(path: &Path) {
    if path.is_file() {
        std::fs::remove_file(path).expect("the trail can be removed");
        std::fs::create_dir(path).expect("a directory can take its place");
    }
}

/// Launches a runtime auditing to `path` with the given durability.
async fn runtime_with_durability(
    app_name: &str,
    server: &MockServer,
    path: &Path,
    durability: AuditDurability,
) -> ActonAI {
    ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .audit(AuditConfig::new(path).with_durability(durability))
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

/// Three rounds: a mutating call that breaks the trail as it runs, a second
/// mutating call, then a read-only call.
fn rounds_around_a_failure() -> Vec<Round> {
    vec![
        Round::tool_call("call_1", "write_thing", json!({"value": "first"})),
        Round::tool_call("call_2", "write_thing", json!({"value": "second"})),
        Round::tool_call("call_3", "read_thing", json!({"value": "third"})),
        Round::text("done"),
    ]
}

/// Drives [`rounds_around_a_failure`] through `ai`, breaking the trail from
/// inside the first tool call.
async fn drive_around_a_failure(ai: &ActonAI, path: &Path) -> CollectedResponse {
    let trail = path.to_path_buf();
    ai.prompt("write then read")
        .with_tool(tool_definition("write_thing"), move |args| {
            let trail = trail.clone();
            async move {
                make_unappendable(&trail);
                Ok(args)
            }
        })
        .with_tool(idempotent_tool_definition("read_thing"), |args| async move {
            Ok(args)
        })
        .collect()
        .await
        .expect("the turn must complete")
}

#[tokio::test]
async fn strict_mode_refuses_mutating_calls_after_an_append_failure() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");
    let server = MockServer::start(rounds_around_a_failure()).await;
    let ai = runtime_with_durability("audit-strict", &server, &path, AuditDurability::Strict).await;

    let response = drive_around_a_failure(&ai, &path).await;

    assert_eq!(server.request_count(), 4, "every round was served");
    let calls = &response.tool_calls;
    assert_eq!(calls.len(), 3);
    assert!(calls[0].result.is_ok(), "the first call ran: {:?}", calls[0]);

    // The second mutating call is refused: its record could not be made.
    let refusal = calls[1]
        .result
        .as_ref()
        .expect_err("a mutating call after an append failure must be refused");
    assert!(refusal.contains("audit trail is degraded"), "{refusal}");
    assert!(refusal.contains("denied by policy"), "{refusal}");

    // The read-only call still ran: refusing a read protects nothing.
    assert!(calls[2].result.is_ok(), "an idempotent call runs: {:?}", calls[2]);

    let health = ai.audit_health().await.expect("health is reported");
    assert!(health.is_degraded());
    assert_eq!(health.state, AuditHealthState::Degraded);
    assert_eq!(health.durability, AuditDurability::Strict);
    assert_eq!(
        health.first_failed_sequence,
        Some(1),
        "the first call's own entry is the one that never landed"
    );
    assert_eq!(health.appended, 0);
    assert!(health.failures >= 1, "{health:?}");
    assert!(health.last_error.is_some());
    assert!(health.degraded_since.is_some());

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn best_effort_mode_continues_after_an_append_failure() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");
    let server = MockServer::start(rounds_around_a_failure()).await;
    let ai = runtime_with_durability(
        "audit-best-effort",
        &server,
        &path,
        AuditDurability::BestEffort,
    )
    .await;

    let response = drive_around_a_failure(&ai, &path).await;

    assert_eq!(server.request_count(), 4);
    assert!(
        response.tool_calls.iter().all(|call| call.result.is_ok()),
        "best effort never refuses a call over the trail: {:?}",
        response.tool_calls
    );

    // The failure is still visible — it is a state, not just a log line.
    let health = ai.audit_health().await.expect("health is reported");
    assert!(health.is_degraded());
    assert_eq!(health.durability, AuditDurability::BestEffort);
    assert_eq!(health.first_failed_sequence, Some(1));
    assert_eq!(health.failures, 3, "every call's entry failed to land");

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn strict_mode_acknowledges_each_entry_before_the_next_call() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "one"})),
        Round::tool_call("call_2", "echo", json!({"value": "two"})),
        Round::text("done"),
    ])
    .await;
    let ai = runtime_with_durability("audit-strict-ack", &server, &path, AuditDurability::Strict)
        .await;

    ai.prompt("echo twice")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    // No `audit_head()` barrier on purpose: under strict durability the loop
    // does not move past a call until its entry is acknowledged, so by the
    // time `collect()` returns the file already holds every entry.
    let entries = read_trail(&path);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].tool_call_id, "call_1");
    assert_eq!(entries[1].tool_call_id, "call_2");
    verify_chain(&entries).expect("the chain verifies");

    let health = ai.audit_health().await.expect("health is reported");
    assert_eq!(health.state, AuditHealthState::Healthy);
    assert_eq!(health.appended, 2);
    assert_eq!(health.failures, 0);
    assert_eq!(health.head.sequence, 2);

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn audit_health_reports_healthy_before_any_write_and_disabled_without_a_trail() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");
    let server = MockServer::start(vec![Round::text("hello")]).await;

    let armed = runtime_with_durability("audit-armed", &server, &path, AuditDurability::Strict).await;
    let health = armed.audit_health().await.expect("health is reported");
    assert_eq!(health.state, AuditHealthState::Healthy);
    assert_eq!(health.appended, 0);
    assert_eq!(health.failures, 0);
    assert_eq!(health.head.hash, GENESIS_HASH);
    assert_eq!(armed.audit_durability(), Some(AuditDurability::Strict));
    armed.shutdown().await.expect("clean shutdown");

    let unaudited = ActonAI::builder()
        .app_name("audit-none")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .launch()
        .await
        .expect("launching the runtime must succeed");
    let health = unaudited.audit_health().await.expect("disabled is an answer");
    assert_eq!(health.state, AuditHealthState::Disabled);
    assert!(!health.is_degraded());
    assert_eq!(unaudited.audit_durability(), None);
    unaudited.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn a_toml_durability_key_makes_the_trail_strict() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");
    let server = MockServer::start(vec![Round::text("hello")]).await;

    let toml = format!(
        "{}\n[audit]\npath = '{}'\ndurability = 'strict'\n",
        mock_llm::provider_toml("mock", &server, "mock-model"),
        path.display()
    );
    let config = acton_ai::config::from_str(&toml).expect("the config must parse");
    let ai = ActonAI::builder()
        .app_name("audit-toml-strict")
        .apply_config(config)
        .expect("the config must apply")
        .launch()
        .await
        .expect("launching the runtime must succeed");

    assert_eq!(ai.audit_durability(), Some(AuditDurability::Strict));
    ai.shutdown().await.expect("clean shutdown");
}
