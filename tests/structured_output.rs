//! Integration tests for typed structured output (`PromptBuilder::extract`).
//!
//! These drive the real stack — HTTP client, provider actor, stream
//! collector, prompt loop — against a scripted OpenAI-compatible server, so
//! what is asserted is what actually goes out on the wire and what actually
//! comes back.
//!
//! # Wire shape
//!
//! The SSE bodies here replicate exactly what `OpenAIClient` parses (see
//! `src/llm/openai.rs`): `data: {…}` lines carrying a chunk with a non-empty
//! `id`, a `choices` array whose entries have `index`, `delta`, and an
//! optional `finish_reason`, terminated by `data: [DONE]`. Tool calls arrive
//! as `delta.tool_calls` entries with `index`/`id`/`function.name`/
//! `function.arguments`, where `arguments` is a **JSON-encoded string**, and
//! the client only emits the accumulated calls once it sees a
//! `finish_reason` — so every scripted round ends with a finish chunk.
//!
//! # Determinism
//!
//! Nothing sleeps. The server hands out scripted rounds in order and the
//! prompt loop blocks on the collector's completion signal, so each test's
//! request count is exact.

use acton_ai::prelude::*;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// =============================================================================
// Target types
// =============================================================================

#[derive(Debug, Deserialize, JsonSchema, PartialEq, Eq)]
struct LineItem {
    description: String,
    cents: u64,
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq, Eq)]
struct Invoice {
    vendor: String,
    total_cents: u64,
    line_items: Vec<LineItem>,
}

fn valid_invoice_arguments() -> Value {
    json!({
        "vendor": "Acme Supplies",
        "total_cents": 4_250_u64,
        "line_items": [
            {"description": "widget", "cents": 4_000_u64},
            {"description": "shipping", "cents": 250_u64},
        ],
    })
}

fn expected_invoice() -> Invoice {
    Invoice {
        vendor: "Acme Supplies".to_string(),
        total_cents: 4_250,
        line_items: vec![
            LineItem {
                description: "widget".to_string(),
                cents: 4_000,
            },
            LineItem {
                description: "shipping".to_string(),
                cents: 250,
            },
        ],
    }
}

/// Arguments that are valid JSON but the wrong shape for [`Invoice`]:
/// `total_cents` is a string where a `u64` is required.
fn invalid_invoice_arguments() -> Value {
    json!({
        "vendor": "Acme Supplies",
        "total_cents": "forty-two fifty",
        "line_items": [],
    })
}

// =============================================================================
// Scripted SSE responses
// =============================================================================

/// One tool call the mock server should emit.
#[derive(Clone)]
struct ScriptedToolCall {
    id: String,
    name: String,
    arguments: Value,
}

impl ScriptedToolCall {
    fn new(id: &str, name: &str, arguments: Value) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
        }
    }
}

/// One complete scripted response: some text, some tool calls, then a finish.
#[derive(Clone, Default)]
struct Round {
    text: Option<String>,
    tool_calls: Vec<ScriptedToolCall>,
}

impl Round {
    /// A plain prose answer that ends the turn.
    fn text(text: &str) -> Self {
        Self {
            text: Some(text.to_string()),
            tool_calls: Vec::new(),
        }
    }

    /// A round whose only content is one tool call.
    fn tool_call(id: &str, name: &str, arguments: Value) -> Self {
        Self {
            text: None,
            tool_calls: vec![ScriptedToolCall::new(id, name, arguments)],
        }
    }

    fn with_tool_call(mut self, id: &str, name: &str, arguments: Value) -> Self {
        self.tool_calls.push(ScriptedToolCall::new(id, name, arguments));
        self
    }

    /// Renders the round as an SSE body in the exact shape the OpenAI client
    /// parses.
    fn to_sse(&self) -> String {
        let mut body = String::new();
        let mut push = |chunk: Value| {
            body.push_str("data: ");
            body.push_str(&serde_json::to_string(&chunk).expect("chunk must serialize"));
            body.push_str("\n\n");
        };

        if let Some(ref text) = self.text {
            push(json!({
                "id": "chatcmpl-mock",
                "choices": [{"index": 0, "delta": {"content": text}}],
            }));
        }

        for (index, call) in self.tool_calls.iter().enumerate() {
            push(json!({
                "id": "chatcmpl-mock",
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": index,
                        "id": call.id,
                        "function": {
                            "name": call.name,
                            // The client accumulates `arguments` as a string
                            // and parses it once the round finishes.
                            "arguments": call.arguments.to_string(),
                        },
                    }]},
                }],
            }));
        }

        let finish_reason = if self.tool_calls.is_empty() {
            "stop"
        } else {
            "tool_calls"
        };
        push(json!({
            "id": "chatcmpl-mock",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }));

        body.push_str("data: [DONE]\n\n");
        body
    }
}

// =============================================================================
// Mock server
// =============================================================================

#[derive(Clone)]
struct MockState {
    /// Rounds to serve, in order.
    script: Arc<Vec<Round>>,
    /// How many requests have been served so far.
    served: Arc<AtomicUsize>,
    /// Every request body the server received, in order.
    received: Arc<Mutex<Vec<Value>>>,
}

/// A running scripted OpenAI-compatible server.
struct MockServer {
    base_url: String,
    received: Arc<Mutex<Vec<Value>>>,
}

impl MockServer {
    /// Binds an ephemeral port and starts serving `script`, one round per
    /// request.
    async fn start(script: Vec<Round>) -> Self {
        let received = Arc::new(Mutex::new(Vec::new()));
        let state = MockState {
            script: Arc::new(script),
            served: Arc::new(AtomicUsize::new(0)),
            received: Arc::clone(&received),
        };

        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binding an ephemeral port must succeed");
        let addr = listener
            .local_addr()
            .expect("a bound listener must have an address");

        tokio::spawn(async move {
            // Ends when the test's runtime is dropped.
            let _ = axum::serve(listener, app).await;
        });

        Self {
            base_url: format!("http://{addr}/v1"),
            received,
        }
    }

    /// Every request body received so far, in order.
    fn requests(&self) -> Vec<Value> {
        self.received
            .lock()
            .expect("request log must not be poisoned")
            .clone()
    }

    fn request_count(&self) -> usize {
        self.requests().len()
    }
}

async fn chat_completions(State(state): State<MockState>, Json(body): Json<Value>) -> Response {
    state
        .received
        .lock()
        .expect("request log must not be poisoned")
        .push(body);

    let index = state.served.fetch_add(1, Ordering::SeqCst);
    let Some(round) = state.script.get(index) else {
        // The prompt loop asked for more rounds than the test scripted —
        // surface that as a server error rather than hanging.
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("mock server has no scripted round #{index}"),
        )
            .into_response();
    };

    (
        [(header::CONTENT_TYPE, "text/event-stream")],
        round.to_sse(),
    )
        .into_response()
}

// =============================================================================
// Helpers
// =============================================================================

async fn runtime_pointed_at(server: &MockServer) -> ActonAI {
    ActonAI::builder()
        .app_name("structured-output-test")
        .provider(ProviderConfig::openai_compatible(
            server.base_url.clone(),
            "mock-model",
        ))
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

/// The `tool_choice` value carried by a recorded request body, if any.
fn tool_choice_of(request: &Value) -> Option<&Value> {
    request.get("tool_choice")
}

/// The synthetic tool's entry in a recorded request body.
fn structured_tool_in(request: &Value) -> Option<&Value> {
    request
        .get("tools")?
        .as_array()?
        .iter()
        .find(|tool| tool["function"]["name"] == STRUCTURED_OUTPUT_TOOL)
}

/// Recursively reports whether any `$ref` appears in a JSON value.
fn contains_ref(value: &Value) -> bool {
    match value {
        Value::Object(map) => map.contains_key("$ref") || map.values().any(contains_ref),
        Value::Array(items) => items.iter().any(contains_ref),
        _ => false,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[tokio::test]
async fn extract_returns_a_typed_value_and_forces_the_synthetic_tool() {
    let server = MockServer::start(vec![Round::tool_call(
        "call_1",
        STRUCTURED_OUTPUT_TOOL,
        valid_invoice_arguments(),
    )])
    .await;
    let runtime = runtime_pointed_at(&server).await;

    let invoice: Invoice = runtime
        .prompt("Extract the invoice from this email.")
        .extract::<Invoice>()
        .await
        .expect("extraction must succeed");

    assert_eq!(invoice, expected_invoice());

    let requests = server.requests();
    assert_eq!(requests.len(), 1, "one round should have sufficed");

    // With no other tools to work through, the choice is forced immediately.
    assert_eq!(
        tool_choice_of(&requests[0]),
        Some(&json!({
            "type": "function",
            "function": {"name": STRUCTURED_OUTPUT_TOOL}
        })),
        "request should force the structured_output call: {:#}",
        requests[0]
    );

    // And the synthetic tool carries the inlined schema of the target type.
    let tool = structured_tool_in(&requests[0]).expect("synthetic tool must be offered");
    let schema = &tool["function"]["parameters"];
    assert_eq!(schema["type"], "object", "schema: {schema:#}");
    assert!(schema["properties"].get("vendor").is_some());
    assert!(schema["properties"].get("total_cents").is_some());
    assert!(
        !contains_ref(schema),
        "subschemas must be inlined, found a $ref: {schema:#}"
    );
    assert_eq!(
        schema["properties"]["line_items"]["items"]["properties"]["cents"]["type"],
        "integer",
        "the nested item schema must be inlined in full: {schema:#}"
    );
}

#[tokio::test]
async fn invalid_arguments_trigger_a_repair_round_that_feeds_back_the_error() {
    let server = MockServer::start(vec![
        Round::tool_call(
            "call_bad",
            STRUCTURED_OUTPUT_TOOL,
            invalid_invoice_arguments(),
        ),
        Round::tool_call("call_good", STRUCTURED_OUTPUT_TOOL, valid_invoice_arguments()),
    ])
    .await;
    let runtime = runtime_pointed_at(&server).await;

    let invoice: Invoice = runtime
        .prompt("Extract the invoice.")
        .extract::<Invoice>()
        .await
        .expect("the repaired answer must be accepted");

    assert_eq!(invoice, expected_invoice());

    let requests = server.requests();
    assert_eq!(
        requests.len(),
        2,
        "exactly one repair round should have run"
    );

    // The second request must carry the rejected call plus the validation
    // feedback that tells the model what to fix.
    let messages = requests[1]["messages"]
        .as_array()
        .expect("messages must be an array");

    let echoed = messages
        .iter()
        .find(|m| m["role"] == "assistant" && m["tool_calls"].is_array())
        .expect("the rejected tool call must be echoed back to the model");
    assert_eq!(echoed["tool_calls"][0]["id"], "call_bad");

    let feedback = messages
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("a tool result carrying the validation error must be present");
    assert_eq!(feedback["tool_call_id"], "call_bad");
    let text = feedback["content"]
        .as_str()
        .expect("tool content must be a string");
    assert!(
        text.contains("Validation failed"),
        "feedback must name the failure: {text}"
    );
    assert!(
        text.contains("total_cents"),
        "feedback must name the offending field so the model can fix it: {text}"
    );

    // The repair round keeps the choice forced.
    assert_eq!(
        tool_choice_of(&requests[1]),
        Some(&json!({
            "type": "function",
            "function": {"name": STRUCTURED_OUTPUT_TOOL}
        })),
        "repair round must keep forcing the call: {:#}",
        requests[1]
    );
}

#[tokio::test]
async fn repeated_invalid_arguments_exhaust_the_repair_budget() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", STRUCTURED_OUTPUT_TOOL, invalid_invoice_arguments()),
        Round::tool_call("call_2", STRUCTURED_OUTPUT_TOOL, invalid_invoice_arguments()),
        Round::tool_call("call_3", STRUCTURED_OUTPUT_TOOL, invalid_invoice_arguments()),
    ])
    .await;
    let runtime = runtime_pointed_at(&server).await;

    let error = runtime
        .prompt("Extract the invoice.")
        .extract::<Invoice>()
        .await
        .expect_err("three invalid answers must fail the extraction");

    assert!(
        matches!(error.kind, ActonAIErrorKind::Extraction { .. }),
        "unexpected error kind: {error:?}"
    );

    let message = error.to_string();
    assert!(
        message.contains("validation exhausted"),
        "the error must name the failure class: {message}"
    );
    assert!(
        message.contains("total_cents") && message.contains("expected u64"),
        "the error must carry the serde error: {message}"
    );
    assert!(
        message.contains("forty-two fifty"),
        "the error must dump what the model actually produced: {message}"
    );

    assert_eq!(
        server.request_count(),
        3,
        "the budget is one attempt plus two repairs"
    );
}

#[tokio::test]
async fn a_real_tool_runs_first_and_a_stalled_round_is_nudged_into_answering() {
    let tool_ran = Arc::new(AtomicUsize::new(0));
    let observer = Arc::clone(&tool_ran);

    let server = MockServer::start(vec![
        // Round 1: the model uses the caller's own tool.
        Round::tool_call("call_lookup", "lookup_vendor", json!({"id": "v-7"})),
        // Round 2: prose, no answer recorded — this is the stall.
        Round::text("The vendor is Acme Supplies."),
        // Round 3: under a forced choice, it records the answer.
        Round::tool_call("call_answer", STRUCTURED_OUTPUT_TOOL, valid_invoice_arguments()),
    ])
    .await;
    let runtime = runtime_pointed_at(&server).await;

    let invoice: Invoice = runtime
        .prompt("Extract the invoice, looking the vendor up first.")
        .tool(
            "lookup_vendor",
            "Looks a vendor up by id",
            json!({
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            }),
            move |_args| {
                let observer = Arc::clone(&observer);
                async move {
                    observer.fetch_add(1, Ordering::SeqCst);
                    Ok(json!({"vendor": "Acme Supplies"}))
                }
            },
        )
        .extract::<Invoice>()
        .await
        .expect("extraction must succeed after the nudge");

    assert_eq!(invoice, expected_invoice());
    assert_eq!(
        tool_ran.load(Ordering::SeqCst),
        1,
        "the caller's real tool must actually have executed"
    );

    let requests = server.requests();
    assert_eq!(requests.len(), 3, "three rounds were scripted");

    // While real tools are in play the model chooses freely...
    assert_eq!(
        tool_choice_of(&requests[0]),
        Some(&json!("auto")),
        "round 1 must leave the choice to the model: {:#}",
        requests[0]
    );
    assert_eq!(
        tool_choice_of(&requests[1]),
        Some(&json!("auto")),
        "round 2 must still leave the choice to the model: {:#}",
        requests[1]
    );

    // ...until it stalls, at which point it is asked directly.
    assert_eq!(
        tool_choice_of(&requests[2]),
        Some(&json!({
            "type": "function",
            "function": {"name": STRUCTURED_OUTPUT_TOOL}
        })),
        "the round after the stall must force the answer: {:#}",
        requests[2]
    );
    let nudge = requests[2]["messages"]
        .as_array()
        .expect("messages must be an array")
        .iter()
        .rfind(|m| m["role"] == "user")
        .expect("a user message must be present");
    assert!(
        nudge["content"]
            .as_str()
            .is_some_and(|c| c.contains(STRUCTURED_OUTPUT_TOOL)),
        "the stalled round must be followed by an explicit nudge: {nudge:#}"
    );
}

#[tokio::test]
async fn a_sibling_tool_call_in_the_capturing_round_is_not_executed() {
    let tool_ran = Arc::new(AtomicUsize::new(0));
    let observer = Arc::clone(&tool_ran);

    // One round that both records the answer and asks for the real tool.
    // Capturing terminates the loop, so the sibling never runs.
    let server = MockServer::start(vec![Round::tool_call(
        "call_answer",
        STRUCTURED_OUTPUT_TOOL,
        valid_invoice_arguments(),
    )
    .with_tool_call("call_sibling", "lookup_vendor", json!({"id": "v-7"}))])
    .await;
    let runtime = runtime_pointed_at(&server).await;

    let invoice: Invoice = runtime
        .prompt("Extract the invoice.")
        .tool(
            "lookup_vendor",
            "Looks a vendor up by id",
            json!({
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            }),
            move |_args| {
                let observer = Arc::clone(&observer);
                async move {
                    observer.fetch_add(1, Ordering::SeqCst);
                    Ok(json!({"vendor": "Acme Supplies"}))
                }
            },
        )
        .extract::<Invoice>()
        .await
        .expect("the recorded answer must be accepted");

    assert_eq!(invoice, expected_invoice());
    assert_eq!(
        tool_ran.load(Ordering::SeqCst),
        0,
        "a sibling of the capturing call must not be executed"
    );
    assert_eq!(server.request_count(), 1, "capture ends the loop");
}

#[tokio::test]
async fn plain_collect_sends_no_tool_choice_at_all() {
    let server = MockServer::start(vec![Round::text("Paris.")]).await;
    let runtime = runtime_pointed_at(&server).await;

    let response = runtime
        .prompt("What is the capital of France?")
        .collect()
        .await
        .expect("a plain collect must succeed");

    assert_eq!(response.text, "Paris.");

    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    assert!(
        tool_choice_of(&requests[0]).is_none(),
        "collect() must not introduce a tool_choice key: {:#}",
        requests[0]
    );
    assert!(
        requests[0].get("tools").is_none(),
        "collect() without tools must not send a tools array: {:#}",
        requests[0]
    );
}

#[tokio::test]
async fn a_user_tool_may_not_claim_the_reserved_name() {
    let server = MockServer::start(Vec::new()).await;
    let runtime = runtime_pointed_at(&server).await;

    let error = runtime
        .prompt("Extract the invoice.")
        .tool(
            STRUCTURED_OUTPUT_TOOL,
            "A tool that collides with extraction",
            json!({"type": "object"}),
            |_args| async move { Ok(json!({})) },
        )
        .extract::<Invoice>()
        .await
        .expect_err("the collision must be rejected rather than shadowed");

    assert!(error.is_configuration(), "unexpected kind: {error:?}");
    assert!(
        error.to_string().contains(STRUCTURED_OUTPUT_TOOL),
        "the error must name the conflict: {error}"
    );
    assert_eq!(
        server.request_count(),
        0,
        "the collision must be caught before any request is sent"
    );
}
