//! End-to-end tests for runtime-wide and per-conversation custom tools.
//!
//! These drive the real stack — facade, prompt loop, provider actor, HTTP
//! client — against the scripted server in [`mock_llm`], so what they assert
//! is what actually travelled the wire: whether the tool was offered, whether
//! it ran, and what the model was told about it afterwards.
//!
//! # Determinism
//!
//! Nothing sleeps. Tool execution is synchronous with respect to the loop —
//! a round cannot be sent until the previous round's tool results exist — so
//! a completed prompt is itself the barrier. The audit test additionally uses
//! `ActonAI::audit_head()` as an ask-barrier on the audit actor.

mod mock_llm;

use acton_ai::prelude::*;
use acton_ai::tools::ToolExecutionFuture;
use mock_llm::{contains_ref, tool_named, MockServer, Round};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

/// Every `tool`-role message body in one request, in order.
fn tool_results(request: &Value) -> Vec<String> {
    request["messages"]
        .as_array()
        .expect("every request carries messages")
        .iter()
        .filter(|message| message["role"] == "tool")
        .map(|message| message["content"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// Reads the audit trail back the way `acton-ai audit verify` does.
fn read_trail(path: &Path) -> Vec<AuditEntry> {
    let contents = std::fs::read_to_string(path).expect("the trail must exist");
    acton_ai::audit::parse_entries(&contents).expect("the trail must parse")
}

// =============================================================================
// 1. A builder-registered tool is offered and runs in a plain prompt
// =============================================================================

#[tokio::test]
async fn a_builder_registered_tool_is_offered_and_runs_in_a_prompt() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})),
        Round::text("done"),
    ])
    .await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    let ai = ActonAI::builder()
        .app_name("custom-global")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_tool(tool_definition("echo"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .launch()
        .await
        .expect("launching the runtime must succeed");

    // No per-prompt registration at all: the tool rides on `.prompt()` alone.
    let response = ai
        .prompt("echo hi")
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(ran.load(Ordering::SeqCst), 1, "the tool must have executed");
    assert_eq!(response.text, "done");
    assert_eq!(server.request_count(), 2, "the tool round and the answer");

    let first = server.requests().remove(0);
    tool_named(&first, "echo").expect("the runtime-wide tool must be offered to the model");

    let second = server.requests().remove(1);
    let results = tool_results(&second);
    assert_eq!(results.len(), 1, "the call needs exactly one result");
    assert!(
        results[0].contains("hi"),
        "the executor's output must be what the model reads: {}",
        results[0]
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 2. The same registration reaches Conversation::send with no extra wiring
// =============================================================================

#[tokio::test]
async fn a_builder_registered_tool_reaches_conversation_send() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "from-conv"})),
        Round::text("answered"),
    ])
    .await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    let ai = ActonAI::builder()
        .app_name("custom-global-conv")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_tool(tool_definition("echo"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let conversation = ai.conversation().build().await;
    let response = conversation
        .send("use the tool")
        .await
        .expect("the turn must complete");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        1,
        "the runtime-wide tool must execute inside Conversation::send"
    );
    assert_eq!(response.text, "answered");

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 3. A per-conversation tool is available on every send
// =============================================================================

#[tokio::test]
async fn a_conversation_tool_is_available_via_send() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "lookup", json!({"value": "order-42"})),
        Round::text("found it"),
    ])
    .await;

    let ai = mock_llm::runtime_pointed_at(&server, "custom-conv").await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    let conversation = ai
        .conversation()
        .with_tool(tool_definition("lookup"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .expect("a fresh name must register")
        .build()
        .await;

    let response = conversation
        .send("where is order 42?")
        .await
        .expect("the turn must complete");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        1,
        "the conversation tool must execute inside Conversation::send"
    );
    assert_eq!(response.text, "found it");
    assert_eq!(server.request_count(), 2, "the tool round and the answer");

    let first = server.requests().remove(0);
    tool_named(&first, "lookup").expect("the conversation tool must be offered to the model");

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 4. Custom tools sit behind the same policy gate and audit trail as builtins
// =============================================================================

#[tokio::test]
async fn a_denied_custom_tool_records_a_policy_outcome() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("audit.jsonl");

    let server = MockServer::start(vec![
        Round::tool_call("call_1", "wipe_disk", json!({"value": "all of it"})),
        Round::text("refused, moving on"),
    ])
    .await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    let ai = ActonAI::builder()
        .app_name("custom-denied")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_tool(tool_definition("wipe_disk"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .tool_policy(ToolPolicy::new().deny(["wipe_disk"]))
        .audit_to(&path)
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let response = ai
        .prompt("wipe the disk")
        .collect()
        .await
        .expect("a denial must not fail the turn");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        0,
        "a denied custom tool must never execute"
    );
    assert_eq!(response.text, "refused, moving on");

    // The ask-barrier: the head cannot answer until the entry is written.
    ai.audit_head().await.expect("the trail must report a head");

    let entries = read_trail(&path);
    assert_eq!(
        entries.len(),
        1,
        "the refused call must still be one audit entry"
    );
    assert_eq!(entries[0].tool_name, "wipe_disk");
    assert!(!entries[0].decision.approved);
    assert_eq!(entries[0].decision.decided_by, Decider::Denylist);
    assert!(
        matches!(entries[0].outcome, AuditOutcome::Denied { .. }),
        "a custom tool must record the same policy outcome a builtin would"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 5. Name collisions are configuration-time errors, never silent shadows
// =============================================================================

#[tokio::test]
async fn a_custom_tool_colliding_with_a_builtin_fails_launch() {
    let server = MockServer::start(vec![]).await;

    let result = ActonAI::builder()
        .app_name("custom-collision-builtin")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_builtins()
        .with_tool(tool_definition("bash"), |args| async move { Ok(args) })
        .launch()
        .await;

    let error = result.expect_err("a custom tool named after a builtin must fail the launch");
    let message = error.to_string();
    assert!(message.contains("bash"), "{message}");
    assert!(message.contains("built-in"), "{message}");
}

#[tokio::test]
async fn two_custom_tools_with_one_name_fail_launch() {
    let server = MockServer::start(vec![]).await;

    let result = ActonAI::builder()
        .app_name("custom-collision-duplicate")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .launch()
        .await;

    let error = result.expect_err("two custom tools with one name must fail the launch");
    let message = error.to_string();
    assert!(message.contains("echo"), "{message}");
    assert!(message.contains("another custom tool"), "{message}");
}

#[tokio::test]
async fn a_conversation_tool_colliding_with_a_builtin_is_refused_at_registration() {
    let server = MockServer::start(vec![]).await;

    let ai = ActonAI::builder()
        .app_name("conv-collision-builtin")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_builtins()
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let error = ai
        .conversation()
        .with_tool(tool_definition("bash"), |args| async move { Ok(args) })
        .expect_err("a conversation tool named after an injected builtin must be refused");
    let message = error.to_string();
    assert!(message.contains("bash"), "{message}");

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn a_conversation_tool_colliding_with_a_runtime_tool_is_refused_at_registration() {
    let server = MockServer::start(vec![]).await;

    let ai = ActonAI::builder()
        .app_name("conv-collision-global")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let error = ai
        .conversation()
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .expect_err("a conversation tool named after a runtime-wide tool must be refused");
    let message = error.to_string();
    assert!(message.contains("echo"), "{message}");

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn the_exit_tool_name_is_reserved_on_conversations() {
    let server = MockServer::start(vec![]).await;
    let ai = mock_llm::runtime_pointed_at(&server, "conv-collision-exit").await;

    let error = ai
        .conversation()
        .with_tool(tool_definition("exit_conversation"), |args| async move {
            Ok(args)
        })
        .expect_err("exit_conversation is reserved whether or not the exit tool is enabled yet");
    assert!(error.to_string().contains("exit_conversation"));

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 6. The executor-object and #[tool] registration shapes work end to end
// =============================================================================

/// An executor-object tool, the shape a reusable tool library exports.
#[derive(Debug)]
struct CountingExecutor {
    ran: Arc<AtomicUsize>,
}

impl acton_ai::tools::ToolExecutorTrait for CountingExecutor {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        let ran = self.ran.clone();
        Box::pin(async move {
            ran.fetch_add(1, Ordering::SeqCst);
            Ok(args)
        })
    }
}

#[tokio::test]
async fn a_tool_executor_object_registered_on_the_builder_runs() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "count", json!({"value": "one"})),
        Round::text("counted"),
    ])
    .await;

    let ran = Arc::new(AtomicUsize::new(0));

    let ai = ActonAI::builder()
        .app_name("custom-executor-object")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_tool_executor(
            tool_definition("count"),
            CountingExecutor { ran: ran.clone() },
        )
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let response = ai
        .prompt("count something")
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(ran.load(Ordering::SeqCst), 1);
    assert_eq!(response.text, "counted");

    ai.shutdown().await.expect("clean shutdown");
}

/// Adds two numbers.
#[tool]
async fn add(a: i64, b: i64) -> Result<Value, acton_ai::tools::ToolError> {
    Ok(json!({ "sum": a + b }))
}

#[tokio::test]
async fn a_tool_macro_value_registered_on_the_builder_runs() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "add", json!({"a": 40, "b": 2})),
        Round::text("the sum is 42"),
    ])
    .await;

    let ai = ActonAI::builder()
        .app_name("custom-tool-macro")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .add_tool(Add)
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let response = ai
        .prompt("what is 40 + 2?")
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(response.text, "the sum is 42");
    assert_eq!(server.request_count(), 2);

    // The macro generates the schema; registration through the builder must
    // not change its shape. Same self-containment bar as every other suite:
    // providers disagree on `$ref` support, so none may appear on the wire.
    let first = server.requests().remove(0);
    let offered = tool_named(&first, "add").expect("the macro tool must be offered to the model");
    assert!(
        !contains_ref(&offered["function"]["parameters"]),
        "{offered:#}"
    );

    let second = server.requests().remove(1);
    let results = tool_results(&second);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].contains("42"),
        "the macro tool's sum must be what the model reads: {}",
        results[0]
    );

    ai.shutdown().await.expect("clean shutdown");
}
