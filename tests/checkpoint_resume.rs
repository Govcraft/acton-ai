//! Checkpoints survive the round trip through the real store, and a resume
//! planned over what comes back picks up where the turn stopped.
//!
//! These drive the actual `MemoryStore` actor against an in-memory libSQL
//! database. The planner's own decision table is unit-tested in
//! `src/checkpoint/plan.rs`; what is pinned here is that the table, the SQL,
//! the codec, and the actor agree with it — the seams a pure test cannot see.

use acton_ai::checkpoint::{
    advance, complete, plan_resume, CheckpointRecord, CheckpointStatus, FinalAnswer,
    PendingCallState, PendingRound, PendingToolCall, ResumePlan, RoundProgress, TurnFingerprint,
    TurnInputs,
};
use acton_ai::memory::{
    CheckpointList, CheckpointLoaded, CheckpointSaved, DeleteCheckpoint, InitMemoryStore,
    ListCheckpoints, ListConversations, LoadCheckpoint, MemoryStore, PersistenceConfig,
    SaveCheckpoint,
};
use acton_ai::messages::ToolCall;
use acton_ai::messages::{Message, StopReason, Usage};
use acton_ai::prelude::*;
use acton_ai::stream::ExecutedToolCall;
use acton_ai::types::{CheckpointId, ConversationId};
use std::time::Duration;

/// A missing reply is a bug, not a slow database, so nothing here waits
/// anywhere near the 30-second `ask` default.
const REPLY_DEADLINE: Duration = Duration::from_secs(5);

/// Spawns a store backed by a fresh in-memory database, ready to serve.
async fn spawn_ready_store(runtime: &mut ActorRuntime) -> ActorHandle {
    let store = MemoryStore::spawn(runtime).await;

    store
        .send(InitMemoryStore {
            config: PersistenceConfig::in_memory(),
        })
        .await;

    // A completed `ask` proves the init and the connection message behind it
    // have both been processed. No sleeping required.
    store
        .ask_with_timeout(
            ListConversations {
                agent_id: AgentId::new(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("store must answer once initialized");

    store
}

fn tool_names() -> Vec<String> {
    vec!["read_file".to_string()]
}

fn inputs(tools: &[String]) -> TurnInputs<'_> {
    TurnInputs {
        system_prompt: Some("be brief"),
        user_content: "summarize a.txt",
        tool_names: tools,
        provider: "claude",
        max_tool_rounds: 8,
        structured_schema: None,
    }
}

/// A record two rounds into a turn, with one tool already run.
fn mid_turn(id: CheckpointId, tools: &[String]) -> CheckpointRecord {
    advance(
        id,
        Some(ConversationId::new()),
        TurnFingerprint::of(&inputs(tools)),
        RoundProgress {
            rounds_completed: 2,
            messages: vec![
                Message::system("be brief"),
                Message::user("summarize a.txt"),
                Message::tool("call_1", "contents of a.txt"),
            ],
            tool_calls: vec![ExecutedToolCall::success(
                "call_1",
                "read_file",
                serde_json::json!({ "path": "a.txt" }),
                serde_json::json!("contents of a.txt"),
            )],
            token_count: 37,
            resume_attempts: 0,
            usage: Usage {
                input_tokens: 400,
                output_tokens: 60,
                ..Usage::default()
            },
            pending_round: None,
        },
    )
}

async fn save(store: &ActorHandle, record: CheckpointRecord) {
    let saved: CheckpointSaved = store
        .ask_with_timeout(SaveCheckpoint { record }, REPLY_DEADLINE)
        .await
        .expect("the store must answer a checkpoint write");
    saved.result.expect("the write must land");
}

async fn load(store: &ActorHandle, id: &CheckpointId) -> Option<CheckpointRecord> {
    let loaded: CheckpointLoaded = store
        .ask_with_timeout(LoadCheckpoint { id: id.clone() }, REPLY_DEADLINE)
        .await
        .expect("the store must answer a checkpoint lookup");
    loaded.result.expect("the lookup must succeed")
}

async fn list(store: &ActorHandle, status: Option<CheckpointStatus>) -> Vec<CheckpointRecord> {
    let listed: CheckpointList = store
        .ask_with_timeout(ListCheckpoints { status }, REPLY_DEADLINE)
        .await
        .expect("the store must answer a checkpoint listing");
    listed.result.expect("the listing must succeed")
}

#[tokio::test]
async fn a_saved_checkpoint_loads_back_identical() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let tools = tool_names();
    let record = mid_turn(CheckpointId::new(), &tools);
    save(&store, record.clone()).await;

    assert_eq!(load(&store, &record.id).await, Some(record));

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn a_completed_checkpoint_keeps_its_answer_through_the_database() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let tools = tool_names();
    let record = complete(
        mid_turn(CheckpointId::new(), &tools),
        FinalAnswer {
            text: "a.txt greets the reader".to_string(),
            stop_reason: StopReason::EndTurn,
            structured_output: Some(serde_json::json!({ "greeting": true })),
        },
    );
    save(&store, record.clone()).await;

    let loaded = load(&store, &record.id).await.expect("must be stored");
    assert_eq!(loaded.status, CheckpointStatus::Completed);
    assert_eq!(
        loaded.final_text.as_deref(),
        Some("a.txt greets the reader")
    );
    assert_eq!(loaded.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(
        loaded.structured_output,
        Some(serde_json::json!({ "greeting": true }))
    );

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn saving_the_same_id_twice_updates_rather_than_duplicates() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let tools = tool_names();
    let id = CheckpointId::new();
    save(&store, mid_turn(id.clone(), &tools)).await;

    let mut later = mid_turn(id.clone(), &tools);
    later.rounds_completed = 5;
    later.messages.push(Message::assistant("nearly there"));
    save(&store, later).await;

    let loaded = load(&store, &id).await.expect("must be stored");
    assert_eq!(loaded.rounds_completed, 5);
    assert_eq!(loaded.messages.len(), 4);
    assert_eq!(list(&store, None).await.len(), 1);

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn loading_an_unknown_checkpoint_returns_none() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    assert_eq!(load(&store, &CheckpointId::new()).await, None);

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn deleting_a_checkpoint_makes_it_load_as_none() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let tools = tool_names();
    let record = mid_turn(CheckpointId::new(), &tools);
    save(&store, record.clone()).await;

    let deleted: acton_ai::memory::OperationCompleted = store
        .ask_with_timeout(
            DeleteCheckpoint {
                id: record.id.clone(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer a checkpoint delete");
    deleted.result.expect("the delete must succeed");

    assert_eq!(load(&store, &record.id).await, None);

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn deleting_a_checkpoint_that_is_not_there_succeeds() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let deleted: acton_ai::memory::OperationCompleted = store
        .ask_with_timeout(
            DeleteCheckpoint {
                id: CheckpointId::new(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer a checkpoint delete");

    assert!(deleted.result.is_ok());

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn list_filters_by_status() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let tools = tool_names();
    let running = mid_turn(CheckpointId::new(), &tools);
    let finished = complete(
        mid_turn(CheckpointId::new(), &tools),
        FinalAnswer {
            text: "done".to_string(),
            stop_reason: StopReason::EndTurn,
            structured_output: None,
        },
    );
    save(&store, running.clone()).await;
    save(&store, finished.clone()).await;

    assert_eq!(list(&store, None).await.len(), 2);

    let in_progress = list(&store, Some(CheckpointStatus::InProgress)).await;
    assert_eq!(in_progress.len(), 1);
    assert_eq!(in_progress[0].id, running.id);

    let completed = list(&store, Some(CheckpointStatus::Completed)).await;
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].id, finished.id);

    assert!(list(&store, Some(CheckpointStatus::Failed))
        .await
        .is_empty());

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn a_persisted_checkpoint_plans_a_resume_from_where_it_stopped() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let tools = tool_names();
    let record = mid_turn(CheckpointId::new(), &tools);
    save(&store, record.clone()).await;

    let reloaded = load(&store, &record.id).await.expect("must be stored");
    let plan = plan_resume(Some(&reloaded), &inputs(&tools)).expect("must plan a resume");

    let ResumePlan::Resume {
        messages,
        rounds_completed,
        tool_calls,
        token_count,
        usage,
        pending_round,
        ..
    } = plan
    else {
        panic!("expected a resume, got {plan:?}");
    };
    assert_eq!(messages, record.messages);
    assert_eq!(pending_round, None);
    assert_eq!(rounds_completed, 2);
    assert_eq!(tool_calls, record.tool_calls);
    assert_eq!(token_count, 37);
    assert_eq!(usage.input_tokens, 400);

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn a_persisted_checkpoint_refuses_a_resume_of_a_different_prompt() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let tools = tool_names();
    let record = mid_turn(CheckpointId::new(), &tools);
    save(&store, record.clone()).await;

    let reloaded = load(&store, &record.id).await.expect("must be stored");
    let mut different = inputs(&tools);
    different.user_content = "summarize b.txt";

    let error = plan_resume(Some(&reloaded), &different).expect_err("must refuse");
    assert_eq!(error.checkpoint_id(), Some(&record.id));

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn a_checkpoint_request_before_initialization_reports_instead_of_hanging() {
    let mut runtime = ActonApp::launch_async().await;
    let store = MemoryStore::spawn(&mut runtime).await;

    let loaded: CheckpointLoaded = store
        .ask_with_timeout(
            LoadCheckpoint {
                id: CheckpointId::new(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("an uninitialized store must still answer");

    assert!(loaded.result.is_err());

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn a_pending_round_survives_the_database_round_trip() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let tools = tool_names();
    let mut record = mid_turn(CheckpointId::new(), &tools);
    record.pending_round = Some(PendingRound {
        assistant_text: "reading both files".to_string(),
        calls: vec![
            PendingToolCall {
                call: ToolCall {
                    id: "call_2".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({ "path": "b.txt" }),
                },
                state: PendingCallState::Completed {
                    result: "\"contents of b.txt\"".to_string(),
                },
            },
            PendingToolCall {
                call: ToolCall {
                    id: "call_3".to_string(),
                    name: "bash".to_string(),
                    arguments: serde_json::json!({ "command": "touch c.txt" }),
                },
                state: PendingCallState::Started,
            },
            PendingToolCall {
                call: ToolCall {
                    id: "call_4".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({ "path": "c.txt" }),
                },
                state: PendingCallState::Pending,
            },
        ],
    });
    save(&store, record.clone()).await;

    let reloaded = load(&store, &record.id).await.expect("must be stored");
    assert_eq!(reloaded, record);

    // And the plan carries it out of the database intact, because the
    // settlement rules turn on exactly these per-call states.
    let plan = plan_resume(Some(&reloaded), &inputs(&tools)).expect("must plan a resume");
    let ResumePlan::Resume { pending_round, .. } = plan else {
        panic!("expected a resume, got {plan:?}");
    };
    assert_eq!(pending_round, record.pending_round);

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn a_boundary_record_loads_back_with_no_pending_round() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let tools = tool_names();
    let record = mid_turn(CheckpointId::new(), &tools);
    save(&store, record.clone()).await;

    let reloaded = load(&store, &record.id).await.expect("must be stored");
    assert_eq!(reloaded.pending_round, None);

    runtime.shutdown_all().await.unwrap();
}
