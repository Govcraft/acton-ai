//! Integration tests for typed structured output (`PromptBuilder::extract`).
//!
//! These drive the real stack — HTTP client, provider actor, stream
//! collector, prompt loop — against a scripted OpenAI-compatible server, so
//! what is asserted is what actually goes out on the wire and what actually
//! comes back.
//!
//! The server itself lives in [`mock_llm`], shared with
//! `tests/tool_macro.rs`; see that module for the wire shape it reproduces
//! and why nothing here sleeps.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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
// Helpers
// =============================================================================

/// The `tool_choice` value carried by a recorded request body, if any.
fn tool_choice_of(request: &Value) -> Option<&Value> {
    request.get("tool_choice")
}

/// The synthetic tool's entry in a recorded request body.
fn structured_tool_in(request: &Value) -> Option<&Value> {
    tool_named(request, STRUCTURED_OUTPUT_TOOL)
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
    let runtime = runtime_pointed_at(&server, "structured-output-test").await;

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
        schema["properties"]["line_items"]["items"]["properties"]["cents"]["type"], "integer",
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
        Round::tool_call(
            "call_good",
            STRUCTURED_OUTPUT_TOOL,
            valid_invoice_arguments(),
        ),
    ])
    .await;
    let runtime = runtime_pointed_at(&server, "structured-output-test").await;

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
        Round::tool_call(
            "call_1",
            STRUCTURED_OUTPUT_TOOL,
            invalid_invoice_arguments(),
        ),
        Round::tool_call(
            "call_2",
            STRUCTURED_OUTPUT_TOOL,
            invalid_invoice_arguments(),
        ),
        Round::tool_call(
            "call_3",
            STRUCTURED_OUTPUT_TOOL,
            invalid_invoice_arguments(),
        ),
    ])
    .await;
    let runtime = runtime_pointed_at(&server, "structured-output-test").await;

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
        Round::tool_call(
            "call_answer",
            STRUCTURED_OUTPUT_TOOL,
            valid_invoice_arguments(),
        ),
    ])
    .await;
    let runtime = runtime_pointed_at(&server, "structured-output-test").await;

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
    let runtime = runtime_pointed_at(&server, "structured-output-test").await;

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
    let runtime = runtime_pointed_at(&server, "structured-output-test").await;

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
    let runtime = runtime_pointed_at(&server, "structured-output-test").await;

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
