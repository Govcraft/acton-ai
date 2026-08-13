//! Every request must get a definite answer.
//!
//! These tests pin the contract that used to be broken: the `MemoryStore` and
//! `ToolRegistry` failure paths logged and returned without replying, so a
//! caller waiting on an answer waited until the 30-second `ask` timeout and
//! then learned only that it had timed out.
//!
//! They also cover the concurrency the `act_on` conversion buys: read requests
//! now overlap instead of serialising behind one another.

use acton_ai::memory::{
    CreateConversation, GetContextWindow, GetLatestConversation, InitMemoryStore,
    ListConversations, LoadConversation, LoadMemories, MemoryStore, PersistenceConfig, SaveMessage,
    StoreMemory,
};
use acton_ai::prelude::*;
use acton_ai::tools::{
    ListTools, RegisterTool, ToolConfig, ToolExecutionFuture, ToolExecutorTrait, ToolRegistry,
};
use std::time::Duration;

/// `ask` fails after 30 seconds by default. A missing reply is a bug, not a
/// slow database, so these tests refuse to wait anywhere near that long.
const REPLY_DEADLINE: Duration = Duration::from_secs(5);

/// Spawns a store that has never been initialized.
async fn spawn_uninitialized_store(runtime: &mut ActorRuntime) -> ActorHandle {
    MemoryStore::spawn(runtime).await
}

/// Spawns a store backed by a fresh in-memory database, ready to serve requests.
async fn spawn_ready_store(runtime: &mut ActorRuntime) -> ActorHandle {
    let store = MemoryStore::spawn(runtime).await;

    store
        .send(InitMemoryStore {
            config: PersistenceConfig::in_memory(),
        })
        .await;

    // `InitMemoryStore` is handled by a mutable handler whose future is awaited
    // inline, and the connection it sends itself is enqueued behind it. A
    // completed `ask` proves both have been processed — no sleeping required.
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

#[tokio::test]
async fn uninitialized_store_reports_instead_of_going_silent() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_uninitialized_store(&mut runtime).await;

    let loaded = store
        .ask_with_timeout(
            LoadMemories {
                agent_id: AgentId::new(),
                limit: None,
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("an uninitialized store must still reply");

    let error = loaded
        .result
        .expect_err("there is no database to load memories from");
    assert!(
        error.is_not_initialized(),
        "expected a not-initialized error, got: {error}"
    );

    runtime.shutdown_all().await.expect("shutdown failed");
}

#[tokio::test]
async fn uninitialized_store_answers_writes_too() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_uninitialized_store(&mut runtime).await;

    let created = store
        .ask_with_timeout(
            CreateConversation {
                agent_id: AgentId::new(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("an uninitialized store must still reply to writes");

    assert!(created
        .result
        .expect_err("no database means no conversation")
        .is_not_initialized());

    runtime.shutdown_all().await.expect("shutdown failed");
}

#[tokio::test]
async fn uninitialized_store_answers_context_window_requests() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_uninitialized_store(&mut runtime).await;

    let response = store
        .ask_with_timeout(
            GetContextWindow {
                agent_id: AgentId::new(),
                system_prompt: "You are helpful.".to_string(),
                conversation: vec![Message::user("hello")],
                query_embedding: None,
                max_tokens: 1000,
                memory_limit: 5,
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("an uninitialized store must still reply");

    assert!(response
        .result
        .expect_err("no database means no context window")
        .is_not_initialized());

    runtime.shutdown_all().await.expect("shutdown failed");
}

#[tokio::test]
async fn conversation_round_trips_through_the_store() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;
    let agent_id = AgentId::new();

    let conversation_id = store
        .ask_with_timeout(
            CreateConversation {
                agent_id: agent_id.clone(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("store must reply")
        .result
        .expect("creating a conversation must succeed");

    for message in [Message::user("first"), Message::assistant("second")] {
        store
            .ask_with_timeout(
                SaveMessage {
                    conversation_id: conversation_id.clone(),
                    message,
                },
                REPLY_DEADLINE,
            )
            .await
            .expect("store must reply")
            .result
            .expect("saving a message must succeed");
    }

    let loaded = store
        .ask_with_timeout(
            LoadConversation {
                conversation_id: conversation_id.clone(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("store must reply")
        .result
        .expect("loading a conversation must succeed");

    // Both rows carry the same whole-second `created_at`, so this only holds
    // because the query breaks the tie on the monotonic rowid.
    let contents: Vec<&str> = loaded.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(contents, vec!["first", "second"]);

    let latest = store
        .ask_with_timeout(GetLatestConversation { agent_id }, REPLY_DEADLINE)
        .await
        .expect("store must reply")
        .result
        .expect("looking up the latest conversation must succeed");
    assert_eq!(latest, Some(conversation_id));

    runtime.shutdown_all().await.expect("shutdown failed");
}

#[tokio::test]
async fn absent_records_are_reported_as_empty_not_as_failures() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;

    let latest = store
        .ask_with_timeout(
            GetLatestConversation {
                agent_id: AgentId::new(),
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("store must reply")
        .result
        .expect("an agent with no conversations is not an error");

    assert_eq!(latest, None);

    runtime.shutdown_all().await.expect("shutdown failed");
}

#[tokio::test]
async fn concurrent_reads_all_get_answers() {
    let mut runtime = ActonApp::launch_async().await;
    let store = spawn_ready_store(&mut runtime).await;
    let agent_id = AgentId::new();

    store
        .ask_with_timeout(
            StoreMemory {
                agent_id: agent_id.clone(),
                content: "the sky is blue".to_string(),
                embedding: None,
            },
            REPLY_DEADLINE,
        )
        .await
        .expect("store must reply")
        .result
        .expect("storing a memory must succeed");

    // Issued together rather than one at a time: read handlers run
    // concurrently, and every one of them must still produce a reply.
    let reads = (0..16).map(|_| {
        let store = store.clone();
        let agent_id = agent_id.clone();
        async move {
            store
                .ask_with_timeout(
                    LoadMemories {
                        agent_id,
                        limit: None,
                    },
                    REPLY_DEADLINE,
                )
                .await
        }
    });

    let results = futures::future::join_all(reads).await;

    assert_eq!(results.len(), 16);
    for loaded in results {
        let memories = loaded
            .expect("every concurrent read must be answered")
            .result
            .expect("loading memories must succeed");
        assert_eq!(memories.len(), 1);
    }

    runtime.shutdown_all().await.expect("shutdown failed");
}

#[tokio::test]
async fn registry_answers_list_requests() {
    let mut runtime = ActonApp::launch_async().await;
    let registry = ToolRegistry::spawn(&mut runtime).await;

    let listed = registry
        .ask_with_timeout(ListTools, REPLY_DEADLINE)
        .await
        .expect("registry must reply to a list request");

    assert!(listed.tools.is_empty());

    runtime.shutdown_all().await.expect("shutdown failed");
}

/// Number of executions the rendezvous tool waits for before any can finish.
const RENDEZVOUS_PARTIES: usize = 4;

/// A tool that cannot complete alone.
///
/// Every execution waits on a shared barrier, so all `RENDEZVOUS_PARTIES` calls
/// finish together or none of them do. That makes it a decision procedure: if
/// the registry runs executions one at a time, the first call blocks forever
/// and the test times out.
#[derive(Debug)]
struct RendezvousTool {
    barrier: std::sync::Arc<tokio::sync::Barrier>,
}

impl RendezvousTool {
    fn definition() -> ToolDefinition {
        ToolDefinition {
            name: "rendezvous".to_string(),
            description: "Completes only when every concurrent call has arrived".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        }
    }
}

impl ToolExecutorTrait for RendezvousTool {
    fn execute(&self, _args: serde_json::Value) -> ToolExecutionFuture {
        let barrier = self.barrier.clone();
        Box::pin(async move {
            barrier.wait().await;
            Ok(serde_json::json!({ "arrived": true }))
        })
    }
}

#[tokio::test]
async fn registry_runs_executions_concurrently() {
    let mut runtime = ActonApp::launch_async().await;
    let registry = ToolRegistry::spawn(&mut runtime).await;

    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(RENDEZVOUS_PARTIES));
    registry
        .send(RegisterTool {
            config: ToolConfig::new(RendezvousTool::definition()),
            executor: std::sync::Arc::new(Box::new(RendezvousTool {
                barrier: barrier.clone(),
            }) as Box<dyn ToolExecutorTrait>),
        })
        .await;

    // Collect the tool responses the registry broadcasts on completion.
    let collector = ResponseCollector::subscribe(&mut runtime).await;

    for index in 0..RENDEZVOUS_PARTIES {
        registry
            .send(ExecuteTool {
                correlation_id: CorrelationId::new(),
                requesting_agent: AgentId::new(),
                tool_call: ToolCall {
                    id: format!("call_{index}"),
                    name: "rendezvous".to_string(),
                    arguments: serde_json::json!({}),
                },
            })
            .await;
    }

    // Under the old `try_mutate_on` handler each execution was awaited inline
    // on the message loop, so call 0 would wait for a partner that could never
    // be dispatched and this would time out instead of returning.
    let completed = collector.wait_for(RENDEZVOUS_PARTIES, REPLY_DEADLINE).await;

    assert_eq!(
        completed, RENDEZVOUS_PARTIES,
        "all executions must be in flight at once for the barrier to release"
    );

    runtime.shutdown_all().await.expect("shutdown failed");
}

/// Actor that counts the `ToolResponse` broadcasts the registry emits.
#[acton_actor]
struct ResponseCounter {
    seen: usize,
}

/// Test-side view of a subscribed [`ResponseCounter`].
struct ResponseCollector {
    handle: ActorHandle,
}

/// Asks the counter how many responses it has observed.
#[acton_message]
struct CountResponses;

/// The counter's answer.
#[acton_message]
struct ResponseCount {
    seen: usize,
}

impl Request for CountResponses {
    type Response = ResponseCount;
}

impl ResponseCollector {
    async fn subscribe(runtime: &mut ActorRuntime) -> Self {
        let mut builder = runtime.new_actor::<ResponseCounter>();

        builder.mutate_on::<ToolResponse>(|actor, _envelope| {
            actor.model.seen += 1;
            Reply::ready()
        });

        builder.act_on::<CountResponses>(|actor, envelope| {
            let reply = envelope.reply_envelope();
            let seen = actor.model.seen;
            Reply::pending(async move {
                reply.send(ResponseCount { seen }).await;
            })
        });

        // Subscriptions must be registered on the builder: acton-reactive
        // ignores them once the actor has started.
        builder.handle().subscribe::<ToolResponse>().await;

        Self {
            handle: builder.start().await,
        }
    }

    /// Polls until `target` responses have been seen or the deadline passes.
    ///
    /// Polling is the honest option here: the count is what is under test, so
    /// there is no earlier event to synchronise on. The deadline bounds a
    /// failure rather than defining success.
    async fn wait_for(&self, target: usize, deadline: Duration) -> usize {
        let started = std::time::Instant::now();

        loop {
            let seen = self
                .handle
                .ask_with_timeout(CountResponses, deadline)
                .await
                .expect("collector must answer")
                .seen;

            if seen >= target || started.elapsed() >= deadline {
                return seen;
            }

            tokio::task::yield_now().await;
        }
    }
}
