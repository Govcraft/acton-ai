//! Session persistence through the `MemoryStore` actor and the prompt loop.
//!
//! An embedder that keeps its sessions in acton-ai's store — one named
//! session per client session, one checkpoint per turn — needs three things
//! from upstream: session messages on the store, a checkpoint that stays
//! resumable when the embedder cancels a turn, and a fingerprint that tells
//! the turns of one session apart. Each is exercised here against the real
//! actor and a scripted provider.
//!
//! # Determinism
//!
//! Store writes are `ask`ed, so every assertion reads state the store has
//! already acknowledged. The one wait is on the claim a dropped turn releases
//! from `Drop`, which is bounded by [`REPLY_DEADLINE`].

mod mock_llm;

use acton_ai::checkpoint::{
    CheckpointConfig, CheckpointRecord, CheckpointStatus, ResumePolicy, TurnFingerprint, TurnInputs,
};
use acton_ai::memory::{
    CheckpointClaims, CheckpointLoaded, CreateSession, DeleteSession, InitMemoryStore,
    ListCheckpointClaims, ListConversations, ListSessions, LoadCheckpoint, LoadConversation,
    MemoryStore, OperationCompleted, PersistenceConfig, ResolveSession, SaveMessage,
    SessionCreated, SessionInfo, SessionList, SessionResolved, TouchSession, UpdateSessionMetadata,
};
use acton_ai::prelude::*;
use acton_ai::types::{CheckpointId, ConversationId};
use mock_llm::{runtime_pointed_at, MockServer, Round};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

/// A missing reply is a bug, not a slow database.
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
    // have both been processed.
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

async fn create(store: &ActorHandle, name: &str, metadata: Option<&str>) -> ConversationId {
    let created: SessionCreated = store
        .ask_with_timeout(
            CreateSession {
                name: name.to_string(),
                agent_id: AgentId::new(),
                system_prompt: Some("Be brief.".to_string()),
                metadata: metadata.map(str::to_string),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer a create");
    created
        .result
        .expect("creating a fresh session must succeed")
}

async fn resolve(store: &ActorHandle, name: &str) -> Option<SessionInfo> {
    let resolved: SessionResolved = store
        .ask_with_timeout(
            ResolveSession {
                name: name.to_string(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer a resolve");
    resolved.result.expect("resolving must not fail")
}

async fn list(store: &ActorHandle) -> Vec<SessionInfo> {
    let listed: SessionList = store
        .ask_with_timeout(ListSessions, REPLY_DEADLINE)
        .await
        .expect("the store must answer a list");
    listed.result.expect("listing must not fail")
}

async fn operation<M>(store: &ActorHandle, message: M) -> OperationCompleted
where
    M: Request<Response = OperationCompleted> + ActonMessage + 'static,
{
    store
        .ask_with_timeout(message, REPLY_DEADLINE)
        .await
        .expect("the store must answer every operation")
}

async fn claims(store: &ActorHandle) -> Vec<CheckpointId> {
    let held: CheckpointClaims = store
        .ask_with_timeout(ListCheckpointClaims, REPLY_DEADLINE)
        .await
        .expect("the store must answer a claim listing");
    held.ids
}

async fn load_record(store: &ActorHandle, id: &CheckpointId) -> Option<CheckpointRecord> {
    let loaded: CheckpointLoaded = store
        .ask_with_timeout(LoadCheckpoint { id: id.clone() }, REPLY_DEADLINE)
        .await
        .expect("the store must answer a checkpoint load");
    loaded.result.expect("loading a checkpoint must not fail")
}

// =============================================================================
// Session messages
// =============================================================================

#[tokio::test]
async fn a_created_session_resolves_with_its_metadata_and_lists() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let conversation = create(&store, "thread_one", Some(r#"{"turns":0}"#)).await;

    let session = resolve(&store, "thread_one")
        .await
        .expect("the session was just created");
    assert_eq!(session.name, "thread_one");
    assert_eq!(session.conversation_id, conversation);
    assert_eq!(session.system_prompt.as_deref(), Some("Be brief."));
    assert_eq!(session.metadata.as_deref(), Some(r#"{"turns":0}"#));
    assert_eq!(session.message_count, 0);

    let bare = create(&store, "thread_two", None).await;
    assert_ne!(
        bare, conversation,
        "every session gets its own conversation"
    );

    let names: Vec<String> = list(&store).await.into_iter().map(|s| s.name).collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"thread_one".to_string()));
    assert!(names.contains(&"thread_two".to_string()));

    assert!(resolve(&store, "nobody").await.is_none());

    runtime.shutdown_all().await.expect("shutdown must succeed");
}

#[tokio::test]
async fn a_session_name_is_taken_once() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;
    create(&store, "taken", None).await;

    let again: SessionCreated = store
        .ask_with_timeout(
            CreateSession {
                name: "taken".to_string(),
                agent_id: AgentId::new(),
                system_prompt: None,
                metadata: None,
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer");
    assert!(
        again.result.is_err(),
        "a second session under the same name must be refused, not silently replace the first"
    );

    runtime.shutdown_all().await.expect("shutdown must succeed");
}

#[tokio::test]
async fn metadata_is_replaced_cleared_and_refused_for_an_unknown_session() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;
    create(&store, "meta", Some("v1")).await;

    let updated = operation(
        &store,
        UpdateSessionMetadata {
            name: "meta".to_string(),
            metadata: Some("v2".to_string()),
        },
    )
    .await;
    assert_eq!(updated.operation, "update_session_metadata");
    updated
        .result
        .expect("updating an existing session succeeds");
    assert_eq!(
        resolve(&store, "meta").await.unwrap().metadata.as_deref(),
        Some("v2")
    );

    operation(
        &store,
        UpdateSessionMetadata {
            name: "meta".to_string(),
            metadata: None,
        },
    )
    .await
    .result
    .expect("clearing succeeds");
    assert_eq!(resolve(&store, "meta").await.unwrap().metadata, None);

    let missing = operation(
        &store,
        UpdateSessionMetadata {
            name: "ghost".to_string(),
            metadata: Some("v1".to_string()),
        },
    )
    .await;
    let error = missing
        .result
        .expect_err("a session that does not exist cannot be updated");
    assert!(error.is_not_found(), "{error}");

    runtime.shutdown_all().await.expect("shutdown must succeed");
}

#[tokio::test]
async fn touching_and_deleting_a_session_through_the_store() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;
    let conversation = create(&store, "gone", None).await;

    // A message under the conversation, so the delete has something to
    // cascade over.
    store
        .ask_with_timeout(
            SaveMessage {
                conversation_id: conversation.clone(),
                message: Message::user("hello"),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer a save");
    assert_eq!(resolve(&store, "gone").await.unwrap().message_count, 1);

    operation(
        &store,
        TouchSession {
            name: "gone".to_string(),
        },
    )
    .await
    .result
    .expect("touching succeeds");

    let deleted = operation(
        &store,
        DeleteSession {
            name: "gone".to_string(),
        },
    )
    .await;
    assert_eq!(deleted.operation, "delete_session");
    deleted.result.expect("deleting succeeds");

    assert!(resolve(&store, "gone").await.is_none());
    let loaded = store
        .ask_with_timeout(
            LoadConversation {
                conversation_id: conversation,
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer a load");
    assert!(
        loaded.result.expect("loading succeeds").is_empty(),
        "the session's messages go with it"
    );

    // Deleting a name nobody holds is not an error.
    operation(
        &store,
        DeleteSession {
            name: "gone".to_string(),
        },
    )
    .await
    .result
    .expect("a repeated delete succeeds");

    runtime.shutdown_all().await.expect("shutdown must succeed");
}

// =============================================================================
// The facade
// =============================================================================

/// A runtime pointed at `server`, checkpointing in memory under `policy`.
async fn launch(server: &MockServer, app_name: &str, policy: ResumePolicy) -> ActonAI {
    ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .checkpoint(CheckpointConfig::new(":memory:").policy(policy))
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

#[tokio::test]
async fn the_facade_reports_the_resume_policy_it_launched_under() {
    let server = MockServer::start(Vec::new()).await;

    let on_request = launch(&server, "policy-on-request", ResumePolicy::ResumeOnRequest).await;
    assert_eq!(
        on_request.checkpoint_policy(),
        Some(ResumePolicy::ResumeOnRequest)
    );

    let auto = launch(&server, "policy-auto", ResumePolicy::ResumeAuto).await;
    assert_eq!(auto.checkpoint_policy(), Some(ResumePolicy::ResumeAuto));

    // No `[checkpoint]` section: no store, no policy.
    let none = runtime_pointed_at(&server, "policy-none").await;
    assert_eq!(none.checkpoint_policy(), None);
    assert!(!none.is_checkpointing());
}

// =============================================================================
// continue_with + checkpoint + conversation_id
// =============================================================================

#[tokio::test]
async fn a_history_driven_turn_checkpoints_under_its_conversation_and_its_own_question() {
    let server = MockServer::start(vec![
        Round::text("first answer"),
        Round::text("second answer"),
        Round::text("first answer again"),
    ])
    .await;
    let ai = launch(&server, "history-ckpt", ResumePolicy::ResumeOnRequest).await;
    let store = ai.checkpoint_store().expect("checkpointing is configured");
    let conversation = ConversationId::new();

    let first_history = vec![Message::user("What is Rust?")];
    let second_history = vec![
        Message::user("What is Rust?"),
        Message::assistant("first answer"),
        Message::user("How does ownership work?"),
    ];

    let first = CheckpointId::new();
    ai.continue_with(first_history.clone())
        .conversation_id(conversation.clone())
        .checkpoint(store.clone(), first.clone())
        .collect()
        .await
        .expect("the first turn completes");

    let second = CheckpointId::new();
    ai.continue_with(second_history)
        .conversation_id(conversation.clone())
        .checkpoint(store.clone(), second.clone())
        .collect()
        .await
        .expect("the second turn completes");

    let first_record = load_record(&store, &first).await.expect("recorded");
    let second_record = load_record(&store, &second).await.expect("recorded");

    // The sink and the conversation id compose: both records name the
    // conversation the embedder said they belong to.
    assert_eq!(first_record.conversation_id, Some(conversation.clone()));
    assert_eq!(second_record.conversation_id, Some(conversation.clone()));

    // Two turns of one session are two different turns.
    assert_ne!(
        first_record.fingerprint, second_record.fingerprint,
        "turns driven from a history must not all fingerprint identically"
    );

    // And the fingerprint is exactly the one taken over the history's last
    // user message, which is what lets a caller re-run a turn by id.
    let expected = TurnFingerprint::of(&TurnInputs {
        system_prompt: None,
        user_content: "What is Rust?",
        tool_names: &[],
        provider: ai.default_provider_name(),
        max_tool_rounds: ai.default_max_tool_rounds(),
        structured_schema: None,
    });
    assert_eq!(first_record.fingerprint, expected);

    // Running the same history under a fresh id starts a fresh turn — the
    // provider is asked again — and lands on the same fingerprint.
    let third = CheckpointId::new();
    ai.continue_with(first_history)
        .conversation_id(conversation)
        .checkpoint(store.clone(), third.clone())
        .collect()
        .await
        .expect("the third turn completes");
    assert_eq!(server.request_count(), 3);
    let third_record = load_record(&store, &third).await.expect("recorded");
    assert_eq!(third_record.fingerprint, first_record.fingerprint);

    // Nothing is left claimed once the turns are over.
    assert!(claims(&store).await.is_empty());
}

#[tokio::test]
async fn continue_with_attaches_no_sink_of_its_own() {
    let server = MockServer::start(vec![Round::text("answer")]).await;
    let ai = launch(&server, "history-no-sink", ResumePolicy::ResumeOnRequest).await;
    let store = ai.checkpoint_store().expect("checkpointing is configured");

    ai.continue_with(vec![Message::user("hello")])
        .collect()
        .await
        .expect("the turn completes");

    let listed: acton_ai::memory::CheckpointList = store
        .ask_with_timeout(
            acton_ai::memory::ListCheckpoints { status: None },
            REPLY_DEADLINE,
        )
        .await
        .expect("the store must answer a listing");
    assert!(
        listed.result.expect("listing succeeds").is_empty(),
        "a history-driven turn records nothing unless the caller attaches a sink"
    );
}

// =============================================================================
// A dropped turn releases its claim
// =============================================================================

fn blocking_tool() -> ToolDefinition {
    ToolDefinition {
        idempotent: false,
        name: "block".to_string(),
        description: "Never returns.".to_string(),
        input_schema: json!({"type": "object", "properties": {}}),
    }
}

#[tokio::test]
async fn dropping_a_turn_mid_tool_call_releases_its_checkpoint_claim() {
    let server = MockServer::start(vec![Round::tool_call("call_1", "block", json!({}))]).await;
    let ai = launch(&server, "drop-claim", ResumePolicy::ResumeOnRequest).await;
    let store = ai.checkpoint_store().expect("checkpointing is configured");

    let entered = Arc::new(Notify::new());
    let never = Arc::new(Notify::new());
    let id = CheckpointId::new();

    let turn = {
        let entered = entered.clone();
        let never = never.clone();
        ai.prompt("block forever")
            .checkpoint(store.clone(), id.clone())
            .with_tool(blocking_tool(), move |_args| {
                let entered = entered.clone();
                let never = never.clone();
                async move {
                    entered.notify_one();
                    never.notified().await;
                    Ok(json!("unreachable"))
                }
            })
            .collect()
    };

    // Run the turn until the tool is entered, then let the `select!` drop
    // the turn's future — exactly what an embedder's cancel does.
    tokio::select! {
        outcome = turn => panic!("the turn must not finish on its own: {outcome:?}"),
        () = entered.notified() => {}
    }

    // The claim is released from `Drop`, which can only spawn the release;
    // wait for it to land rather than asserting on a race.
    let deadline = tokio::time::Instant::now() + REPLY_DEADLINE;
    loop {
        let held = claims(&store).await;
        if held.is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the dropped turn's claim was never released: {held:?}"
        );
        tokio::task::yield_now().await;
    }

    // The record the dropped turn left is still in progress, and — now that
    // nothing claims it — it is an interrupted turn an embedder can abandon
    // or resume by name.
    let record = load_record(&store, &id)
        .await
        .expect("the pending round was recorded");
    assert_eq!(record.status, CheckpointStatus::InProgress);
    assert!(
        record.pending_round.is_some(),
        "the round the tool call belongs to must be on the record"
    );
    let interrupted = ai.interrupted_turns().await.expect("listing succeeds");
    assert!(
        interrupted.iter().any(|record| record.id == id),
        "a released claim is what lets the dropped turn count as interrupted"
    );
}
