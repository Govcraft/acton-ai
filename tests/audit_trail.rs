//! End-to-end tests for the tamper-evident audit trail.
//!
//! These drive the real stack — facade, prompt loop, audit actor, filesystem —
//! and then read the file back with the same public API an auditor would use,
//! so what they assert is what is actually on disk.
//!
//! # Determinism
//!
//! Nothing sleeps. `ActonAI::audit_head()` is an `ask` on the audit actor, and
//! mailboxes are FIFO: its reply cannot arrive until every invocation recorded
//! before it has been sealed and written.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, tool_named, MockServer, Round};
use serde_json::json;
use std::path::Path;

/// A tool the scripted rounds can call.
fn tool_definition(name: &str) -> ToolDefinition {
    ToolDefinition {
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
        .audit_to(&path)
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
        "{}\n[audit]\npath = \"{}\"\n",
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
    assert_eq!(read_trail(&path).len(), 1);

    ai.shutdown().await.expect("clean shutdown");
}
