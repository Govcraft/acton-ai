//! Crash recovery through the whole stack: facade, prompt loop, store, audit.
//!
//! Every test crafts the record a dead process would have left behind —
//! including the per-call pending ledger — saves it through the real
//! `MemoryStore`, and then drives a resume against a scripted provider. What
//! is asserted is what actually went out on the wire, what actually ran, and
//! what actually landed on disk.
//!
//! # Determinism
//!
//! Nothing sleeps. Store writes are `ask`ed, the audit head is the audit
//! barrier, and the mock server serves scripted rounds in order.

mod mock_llm;

use acton_ai::checkpoint::{
    CheckpointConfig, CheckpointRecord, CheckpointStatus, PendingCallState, PendingRound,
    PendingToolCall, ResumePolicy, TurnFingerprint, CHECKPOINT_FORMAT_VERSION,
};
use acton_ai::memory::{
    CheckpointList, CheckpointLoaded, CheckpointSaved, InitMemoryStore, ListCheckpoints,
    ListConversations, LoadCheckpoint, MemoryStore, PersistenceConfig, SaveCheckpoint,
};
use acton_ai::prelude::*;
use acton_ai::stream::ExecutedToolCall;
use acton_ai::types::CheckpointId;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use serde_json::json;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A missing reply is a bug, not a slow database.
const REPLY_DEADLINE: Duration = Duration::from_secs(5);

/// The tool the interrupted turns were working with.
fn note_tool() -> ToolDefinition {
    ToolDefinition {
        idempotent: false,
        name: "note".to_string(),
        description: "Records a note somewhere permanent.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }),
    }
}

/// A runtime pointed at `server`, checkpointing into `db_path` under `policy`.
async fn launch_with_checkpoints(
    server: &MockServer,
    app_name: &str,
    db_path: &str,
    policy: ResumePolicy,
) -> ActonAI {
    ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .checkpoint(CheckpointConfig::new(db_path).policy(policy))
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

/// The record a process that died mid-round would have left behind: one round
/// spent, and a pending ledger describing how far each tool call got.
fn interrupted_record(
    pending: PendingRound,
    already_executed: Vec<ExecutedToolCall>,
) -> CheckpointRecord {
    CheckpointRecord {
        id: CheckpointId::new(),
        conversation_id: None,
        // Never rechecked on the operator path: the facade vouches for the
        // record it just listed, and the original inputs were never stored.
        fingerprint: TurnFingerprint::from_hex("written by a previous process"),
        format_version: CHECKPOINT_FORMAT_VERSION,
        status: CheckpointStatus::InProgress,
        rounds_completed: 1,
        token_count: 5,
        usage: Usage::default(),
        messages: vec![Message::user("record two notes")],
        tool_calls: already_executed,
        final_text: None,
        stop_reason: None,
        structured_output: None,
        pending_round: Some(pending),
        resume_attempts: 0,
    }
}

fn pending_call(id: &str, text: &str, state: PendingCallState) -> PendingToolCall {
    PendingToolCall {
        call: acton_ai::messages::ToolCall {
            id: id.to_string(),
            name: "note".to_string(),
            arguments: json!({ "text": text }),
        },
        state,
    }
}

async fn save_record(store: &ActorHandle, record: CheckpointRecord) {
    let saved: CheckpointSaved = store
        .ask_with_timeout(SaveCheckpoint { record }, REPLY_DEADLINE)
        .await
        .expect("the store must answer a checkpoint write");
    saved.result.expect("the write must land");
}

async fn load_record(store: &ActorHandle, id: &CheckpointId) -> Option<CheckpointRecord> {
    let loaded: CheckpointLoaded = store
        .ask_with_timeout(LoadCheckpoint { id: id.clone() }, REPLY_DEADLINE)
        .await
        .expect("the store must answer a checkpoint lookup");
    loaded.result.expect("the lookup must succeed")
}

async fn list_by_status(store: &ActorHandle, status: CheckpointStatus) -> Vec<CheckpointRecord> {
    let listed: CheckpointList = store
        .ask_with_timeout(
            ListCheckpoints {
                status: Some(status),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer a checkpoint listing");
    listed.result.expect("the listing must succeed")
}

// =============================================================================
// 1. A resume runs the unfinished call and never the finished one
// =============================================================================

#[tokio::test]
async fn a_resume_skips_completed_calls_and_executes_the_pending_one() {
    let server = MockServer::start(vec![Round::text("both notes are recorded")]).await;
    let ai = launch_with_checkpoints(&server, "resume-skips", ":memory:", {
        ResumePolicy::ResumeOnRequest
    })
    .await;
    let store = ai.checkpoint_store().expect("checkpointing is configured");

    // The dead process finished call_1 — its result is in the ledger — and
    // never began call_2.
    let record = interrupted_record(
        PendingRound {
            assistant_text: "recording both notes".to_string(),
            calls: vec![
                pending_call(
                    "call_1",
                    "first",
                    PendingCallState::Completed {
                        result: "\"noted A\"".to_string(),
                    },
                ),
                pending_call("call_2", "second", PendingCallState::Pending),
            ],
        },
        vec![ExecutedToolCall::success(
            "call_1",
            "note",
            json!({ "text": "first" }),
            json!("noted A"),
        )],
    );
    save_record(&store, record.clone()).await;

    let executions = Arc::new(AtomicUsize::new(0));
    let seen = executions.clone();
    let response = ai
        .resume_turn(record.clone())
        .expect("the runtime is checkpointing")
        .with_tool(note_tool(), move |_args| {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(json!("noted B"))
            }
        })
        .collect()
        .await
        .expect("the resumed turn must complete");

    // Only the pending call ran. The completed one kept its stored result.
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(response.text, "both notes are recorded");

    // One dispatch: the settled round went straight into the next request.
    assert_eq!(server.request_count(), 1);
    let body = server.requests()[0].to_string();
    assert!(
        body.contains("noted A"),
        "the stored result must be fed back"
    );
    assert!(
        body.contains("noted B"),
        "the fresh result must be fed back"
    );

    // The resumed dispatch offers the same tool the turn was working with.
    let request = &server.requests()[0];
    let offered = tool_named(request, "note").expect("the note tool must be offered");
    assert!(!contains_ref(&offered["function"]["parameters"]));

    // The record is closed out: completed, with no pending round left.
    let finished = load_record(&store, &record.id).await.expect("still stored");
    assert_eq!(finished.status, CheckpointStatus::Completed);
    assert_eq!(finished.pending_round, None);
    assert_eq!(
        finished.final_text.as_deref(),
        Some("both notes are recorded")
    );
}

// =============================================================================
// 2. A started non-idempotent call is NOT re-run; the model is told
// =============================================================================

#[tokio::test]
async fn a_started_non_idempotent_call_is_not_re_run_and_the_model_hears_why() {
    let server = MockServer::start(vec![Round::text("understood, verifying first")]).await;
    let ai = launch_with_checkpoints(&server, "resume-uncertain", ":memory:", {
        ResumePolicy::ResumeOnRequest
    })
    .await;
    let store = ai.checkpoint_store().expect("checkpointing is configured");

    // The process died while call_1 was executing. `note` is not idempotent,
    // so whether the note landed is unknowable — and must stay that way.
    let record = interrupted_record(
        PendingRound {
            assistant_text: "recording the note".to_string(),
            calls: vec![pending_call("call_1", "only", PendingCallState::Started)],
        },
        Vec::new(),
    );
    save_record(&store, record.clone()).await;

    let executions = Arc::new(AtomicUsize::new(0));
    let seen = executions.clone();
    let response = ai
        .resume_turn(record)
        .expect("the runtime is checkpointing")
        .with_tool(note_tool(), move |_args| {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(json!("noted"))
            }
        })
        .collect()
        .await
        .expect("the resumed turn must complete");

    // Nothing ran: the uncertainty is surfaced, not resolved by re-running.
    assert_eq!(executions.load(Ordering::SeqCst), 0);

    // The model read the uncertainty as the call's tool result.
    let body = server.requests()[0].to_string();
    assert!(body.contains("NOT re-run"), "{body}");
    assert!(body.contains("interrupted"), "{body}");

    // The caller's record says the same thing the model read.
    assert!(response.tool_calls.iter().any(|call| {
        call.name == "note"
            && call
                .result
                .as_ref()
                .err()
                .is_some_and(|message| message.contains("NOT re-run"))
    }));
}

// =============================================================================
// 3. The default policy closes interrupted turns out as a recorded outcome
// =============================================================================

#[tokio::test]
async fn the_abandon_policy_records_the_outcome_without_running_anything() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let db_path = dir.path().join("checkpoints.db");
    let db_path = db_path.to_str().expect("a utf-8 path");

    // The previous process: writes one interrupted turn, then dies.
    let interrupted_id = seed_interrupted_turn(db_path).await;

    // The restarted process, under the default policy. `CheckpointConfig::new`
    // deliberately picks no policy, so this also pins what the default is.
    let server = MockServer::start(vec![]).await;
    let ai = ActonAI::builder()
        .app_name("abandon-outcome")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .checkpoint(CheckpointConfig::new(db_path))
        .launch()
        .await
        .expect("launching the runtime must succeed");

    // Nothing dispatched, nothing executed.
    assert_eq!(server.request_count(), 0);

    // The turn is no longer interrupted — it has an outcome.
    let store = ai.checkpoint_store().expect("checkpointing is configured");
    let abandoned = list_by_status(&store, CheckpointStatus::Abandoned).await;
    assert_eq!(abandoned.len(), 1);
    assert_eq!(abandoned[0].id, interrupted_id);
    // The evidence is intact: the progress the turn had made is still there.
    assert_eq!(abandoned[0].rounds_completed, 1);
    assert!(!abandoned[0].messages.is_empty());

    let waiting = ai.interrupted_turns().await.expect("the store must answer");
    assert!(
        waiting.is_empty(),
        "an abandoned turn is settled, not waiting"
    );

    // And abandonment is terminal: a resume of the record is refused.
    let error = ai
        .resume_turn(abandoned[0].clone())
        .expect("the runtime is checkpointing")
        .collect()
        .await
        .expect_err("an abandoned turn must not resume");
    assert!(error.to_string().contains("abandoned"), "{error}");

    // The refusal changed nothing: the record is still Abandoned, not
    // downgraded to Failed by the failing loop's mark on its way out. A
    // downgrade would put the turn back in interrupted_turns(), reopening a
    // record the operator's policy had closed.
    let still_abandoned = list_by_status(&store, CheckpointStatus::Abandoned).await;
    assert_eq!(still_abandoned.len(), 1);
    assert_eq!(still_abandoned[0].id, interrupted_id);
    let waiting = ai.interrupted_turns().await.expect("the store must answer");
    assert!(
        waiting.is_empty(),
        "a refused resume must not reopen an abandoned record"
    );
}

#[tokio::test]
async fn the_sweep_abandons_a_turn_that_is_out_of_attempts() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let db_path = dir.path().join("checkpoints.db");
    let db_path = db_path.to_str().expect("a utf-8 path");

    let server = MockServer::start(vec![]).await;
    let ai = launch_with_checkpoints(
        &server,
        "sweep-ceiling",
        db_path,
        ResumePolicy::ResumeOnRequest,
    )
    .await;
    let store = ai.checkpoint_store().expect("checkpointing is configured");

    // A turn that has already failed as many times as the default ceiling
    // grants.
    let mut record = interrupted_record(
        PendingRound {
            assistant_text: "noting".to_string(),
            calls: vec![pending_call("call_1", "first", PendingCallState::Pending)],
        },
        vec![],
    );
    record.status = CheckpointStatus::Failed;
    record.resume_attempts = CheckpointConfig::DEFAULT_MAX_RESUME_ATTEMPTS;
    let id = record.id.clone();
    save_record(&store, record).await;

    let resumed = ai.resume_interrupted().await.expect("the sweep must run");

    // Nothing resumed, nothing dispatched, nothing paid for.
    assert!(resumed.is_empty(), "an exhausted turn must not resume");
    assert_eq!(server.request_count(), 0);

    // The turn has a recorded outcome instead of another attempt.
    let abandoned = list_by_status(&store, CheckpointStatus::Abandoned).await;
    assert_eq!(abandoned.len(), 1);
    assert_eq!(abandoned[0].id, id);

    // And it will never be offered up again.
    let waiting = ai.interrupted_turns().await.expect("the store must answer");
    assert!(waiting.is_empty(), "an abandoned turn is settled");
}

/// Writes one in-progress checkpoint into `db_path` and shuts down, the way a
/// crashed process would have left it.
async fn seed_interrupted_turn(db_path: &str) -> CheckpointId {
    let mut runtime = ActonApp::launch_async().await;
    let store = MemoryStore::spawn(&mut runtime).await;
    store
        .send(InitMemoryStore {
            config: PersistenceConfig::new(db_path),
        })
        .await;
    // The ask is the barrier proving initialization completed.
    store
        .ask_with_timeout(
            ListConversations {
                agent_id: AgentId::new(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer once initialized");

    let record = interrupted_record(
        PendingRound {
            assistant_text: "recording the note".to_string(),
            calls: vec![pending_call("call_1", "only", PendingCallState::Started)],
        },
        Vec::new(),
    );
    let id = record.id.clone();
    save_record(&store, record).await;
    runtime.shutdown_all().await.expect("shutdown must succeed");
    id
}

// =============================================================================
// 4. No checkpoint config: nothing exists, nothing changes
// =============================================================================

#[tokio::test]
async fn without_checkpoint_config_nothing_is_recorded_and_nothing_changes() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "note", json!({ "text": "hi" })),
        Round::text("done"),
    ])
    .await;

    let ai = runtime_pointed_at(&server, "no-checkpoints").await;

    assert!(!ai.is_checkpointing());
    assert!(ai.checkpoint_store().is_none());

    let error = ai
        .interrupted_turns()
        .await
        .expect_err("no store means nothing to list");
    assert!(error.to_string().contains("checkpoint"), "{error}");

    // The tool loop runs exactly as it always did.
    let response = ai
        .prompt("note this")
        .with_tool(note_tool(), |_args| async move { Ok(json!("noted")) })
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(response.text, "done");
    assert_eq!(server.request_count(), 2, "the tool round and the answer");
}

// =============================================================================
// 5. The audit trail marks what ran under a resume — and only that
// =============================================================================

#[tokio::test]
async fn the_audit_trail_marks_resumed_executions_and_the_chain_verifies() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let audit_path = dir.path().join("audit.jsonl");

    let server = MockServer::start(vec![
        // Served to the resumed turn, after its settlement.
        Round::text("both notes are recorded"),
        // Served to the ordinary turn afterwards.
        Round::tool_call("call_9", "note", json!({ "text": "fresh" })),
        Round::text("done"),
    ])
    .await;

    let ai = ActonAI::builder()
        .app_name("audit-resumed")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .audit_to(&audit_path)
        .checkpoint(CheckpointConfig::new(":memory:").policy(ResumePolicy::ResumeOnRequest))
        .launch()
        .await
        .expect("launching the runtime must succeed");
    let store = ai.checkpoint_store().expect("checkpointing is configured");

    // A resumed turn whose settlement executes one call.
    let record = interrupted_record(
        PendingRound {
            assistant_text: "recording both notes".to_string(),
            calls: vec![pending_call("call_2", "second", PendingCallState::Pending)],
        },
        Vec::new(),
    );
    save_record(&store, record).await;

    ai.resume_turn(load_only_interrupted(&ai).await)
        .expect("the runtime is checkpointing")
        .with_tool(note_tool(), |_args| async move { Ok(json!("noted B")) })
        .collect()
        .await
        .expect("the resumed turn must complete");

    // An ordinary first-run turn through the same runtime.
    ai.prompt("note something fresh")
        .with_tool(note_tool(), |_args| async move { Ok(json!("noted C")) })
        .collect()
        .await
        .expect("the ordinary turn must complete");

    // The barrier: the head cannot answer until both entries are written.
    let head = ai.audit_head().await.expect("the trail must report a head");
    assert_eq!(head.entries, 4);

    let entries = read_trail(&audit_path);
    assert_eq!(entries.len(), 4);
    let invocations = entries
        .iter()
        .filter(|entry| entry.kind() == AuditEntryKind::Invocation)
        .collect::<Vec<_>>();
    assert!(
        invocations[0].resumed,
        "the settled call ran under a resume and must say so"
    );
    assert!(
        !invocations[1].resumed,
        "an ordinary call must not carry the marker"
    );

    // The marker is covered by the hashes and the chain still links.
    for entry in &entries {
        assert_eq!(entry.recompute_hash(), entry.hash);
    }
    assert_eq!(entries[2].prev_hash, entries[1].hash);

    // The first-run entry keeps the pre-marker byte shape, so trails written
    // before the field existed verify with the same code path.
    let first_run_line = std::fs::read_to_string(&audit_path)
        .expect("the trail must exist")
        .lines()
        .nth(2)
        .expect("the first-run invocation")
        .to_string();
    assert!(!first_run_line.contains("resumed"), "{first_run_line}");
}

/// The one interrupted turn the runtime is holding, via the public listing.
async fn load_only_interrupted(ai: &ActonAI) -> CheckpointRecord {
    let mut waiting = ai.interrupted_turns().await.expect("the store must answer");
    assert_eq!(waiting.len(), 1, "exactly one turn is waiting");
    waiting.remove(0)
}

/// Reads the trail back the way `acton-ai audit verify` does.
fn read_trail(path: &Path) -> Vec<AuditEntry> {
    let contents = std::fs::read_to_string(path).expect("the trail must exist");
    acton_ai::audit::parse_entries(&contents).expect("the trail must parse")
}
