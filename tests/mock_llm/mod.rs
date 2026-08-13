//! A scripted OpenAI-compatible server for driving the real stack in tests.
//!
//! Shared by `tests/structured_output.rs` and `tests/tool_macro.rs`. Both need
//! the same thing — a server that hands out pre-written rounds so a test can
//! assert on what actually went out on the wire and what actually came back —
//! and a second copy of it would be a second thing to keep in sync with
//! `src/llm/openai.rs`.
//!
//! Lives under `tests/mock_llm/` rather than as `tests/mock_llm.rs` because
//! Cargo compiles every `tests/*.rs` file as its own test binary, but leaves
//! subdirectories alone for exactly this purpose.
//!
//! # Wire shape
//!
//! The SSE bodies replicate what `OpenAIClient` parses: `data: {…}` lines
//! carrying a chunk with a non-empty `id` and a `choices` array whose entries
//! have `index`, `delta`, and an optional `finish_reason`, terminated by
//! `data: [DONE]`. Tool calls arrive as `delta.tool_calls` entries with
//! `index`/`id`/`function.name`/`function.arguments`, where `arguments` is a
//! **JSON-encoded string**, and the client only emits the accumulated calls
//! once it sees a `finish_reason` — so every scripted round ends with a
//! finish chunk.
//!
//! # Determinism
//!
//! Nothing sleeps. The server hands out scripted rounds in order and the
//! prompt loop blocks on the collector's completion signal, so each test's
//! request count is exact. Asking for more rounds than a test scripted is a
//! 500, not a hang.

use acton_ai::prelude::*;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// =============================================================================
// Scripted responses
// =============================================================================

/// One tool call the mock server should emit.
#[derive(Clone)]
pub struct ScriptedToolCall {
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

/// Token counts a round should report in its final usage chunk.
#[derive(Clone, Copy)]
pub struct ScriptedUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
}

/// One complete scripted response: some text, some tool calls, then a finish.
#[derive(Clone, Default)]
pub struct Round {
    text: Option<String>,
    tool_calls: Vec<ScriptedToolCall>,
    usage: Option<ScriptedUsage>,
}

impl Round {
    /// A plain prose answer that ends the turn.
    pub fn text(text: &str) -> Self {
        Self {
            text: Some(text.to_string()),
            tool_calls: Vec::new(),
            usage: None,
        }
    }

    /// A round whose only content is one tool call.
    pub fn tool_call(id: &str, name: &str, arguments: Value) -> Self {
        Self {
            text: None,
            tool_calls: vec![ScriptedToolCall::new(id, name, arguments)],
            usage: None,
        }
    }

    /// Makes this round report token usage in a final `stream_options` chunk.
    ///
    /// Without this the round sends no usage at all, standing in for an
    /// OpenAI-compatible server that ignores `stream_options`.
    #[must_use]
    pub fn with_usage(mut self, prompt_tokens: u64, completion_tokens: u64) -> Self {
        self.usage = Some(ScriptedUsage {
            prompt_tokens,
            completion_tokens,
            cached_tokens: 0,
        });
        self
    }

    /// Same as [`Self::with_usage`], but also reports a cached-prompt split.
    ///
    /// `prompt_tokens` is inclusive of `cached_tokens`, matching the real
    /// wire semantics — the client is responsible for splitting them.
    #[must_use]
    pub fn with_cached_usage(
        mut self,
        prompt_tokens: u64,
        completion_tokens: u64,
        cached_tokens: u64,
    ) -> Self {
        self.usage = Some(ScriptedUsage {
            prompt_tokens,
            completion_tokens,
            cached_tokens,
        });
        self
    }

    /// Adds another tool call to the same round, as a model emitting
    /// parallel calls would.
    #[must_use]
    pub fn with_tool_call(mut self, id: &str, name: &str, arguments: Value) -> Self {
        self.tool_calls
            .push(ScriptedToolCall::new(id, name, arguments));
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

        // The `stream_options` usage chunk: usage populated, `choices` EMPTY,
        // and sent *after* the finish chunk. A client that closes the round on
        // the finish chunk never sees it.
        if let Some(usage) = self.usage {
            push(json!({
                "id": "chatcmpl-mock",
                "choices": [],
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
                },
            }));
        }

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
pub struct MockServer {
    base_url: String,
    received: Arc<Mutex<Vec<Value>>>,
}

impl MockServer {
    /// Binds an ephemeral port and starts serving `script`, one round per
    /// request.
    pub async fn start(script: Vec<Round>) -> Self {
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
    pub fn requests(&self) -> Vec<Value> {
        self.received
            .lock()
            .expect("request log must not be poisoned")
            .clone()
    }

    /// How many requests the server has received.
    pub fn request_count(&self) -> usize {
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

/// Launches a runtime whose only provider is `server`.
pub async fn runtime_pointed_at(server: &MockServer, app_name: &str) -> ActonAI {
    ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url.clone(),
            "mock-model",
        ))
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

/// Finds a named tool's entry in a recorded request body.
pub fn tool_named<'a>(request: &'a Value, name: &str) -> Option<&'a Value> {
    request
        .get("tools")?
        .as_array()?
        .iter()
        .find(|tool| tool["function"]["name"] == name)
}

// =============================================================================
// Harness self-test
// =============================================================================

/// Pins the SSE shape this harness renders.
///
/// Every suite that uses the mock is really asserting against
/// `src/llm/openai.rs`'s parser, so the bytes here have to keep matching the
/// wire format that parser expects. This test is the thing that notices when
/// they drift — and, because Cargo compiles this module into each test binary
/// separately, it is also what keeps every builder below exercised no matter
/// which subset a given suite happens to call.
#[test]
fn scripted_rounds_render_the_wire_shape_the_client_parses() {
    let body = Round::text("hello")
        .with_tool_call("call_1", "echo", json!({"value": "hi"}))
        .with_cached_usage(100, 5, 40)
        .to_sse();

    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("every chunk must be valid JSON"))
        .collect();

    // Tool-call arguments ride as a JSON-encoded *string*, not an object.
    let tool_call = &chunks[1]["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(tool_call["function"]["arguments"], json!(r#"{"value":"hi"}"#));

    // A round with tool calls must finish as `tool_calls`, or the client
    // never flushes the accumulated call.
    let finish = &chunks[2];
    assert_eq!(finish["choices"][0]["finish_reason"], json!("tool_calls"));

    // The usage chunk comes *after* the finish chunk and carries an EMPTY
    // `choices` array — the case the parser has to tolerate.
    let usage = chunks.last().expect("a usage chunk must be present");
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["prompt_tokens"], json!(100));
    assert_eq!(usage["usage"]["completion_tokens"], json!(5));
    assert_eq!(usage["usage"]["prompt_tokens_details"]["cached_tokens"], json!(40));

    // `Round::with_usage` is the same path with no cache split.
    let plain = Round::text("hi").with_usage(7, 8).to_sse();
    assert!(plain.contains(r#""prompt_tokens":7"#), "{plain}");
    assert!(plain.ends_with("data: [DONE]\n\n"), "the stream must terminate");
}

/// Recursively reports whether any `$ref` appears in a JSON value.
///
/// Provider support for `$ref`/`$defs` inside a tool's `input_schema` is
/// inconsistent, so both suites assert their schemas are self-contained.
pub fn contains_ref(value: &Value) -> bool {
    match value {
        Value::Object(map) => map.contains_key("$ref") || map.values().any(contains_ref),
        Value::Array(items) => items.iter().any(contains_ref),
        _ => false,
    }
}
