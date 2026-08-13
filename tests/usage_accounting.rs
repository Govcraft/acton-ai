//! End-to-end tests for token-usage plumbing.
//!
//! These drive the real stack — facade, provider actor, OpenAI client, broker
//! — against the scripted server in [`mock_llm`], so what they assert is what
//! actually travels the wire and the broker, not a mocked-out approximation.
//!
//! # Determinism
//!
//! Nothing sleeps. Every wait is a barrier: the prompt loop blocks on the
//! collector's completion signal, and the broadcast-driven assertions wait on
//! `broker.ask(FlushBroadcasts)`, whose reply cannot arrive until every
//! earlier broadcast is sitting in each subscriber's inbox.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use serde_json::json;

/// A tool the scripted rounds can call, so a turn can span several provider
/// requests and exercise cross-round usage summation.
fn echo_tool() -> ToolDefinition {
    ToolDefinition {
        name: "echo".to_string(),
        description: "Echoes its argument back.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }),
    }
}

// =============================================================================
// 1. Streaming usage reaches CollectedResponse
// =============================================================================

#[tokio::test]
async fn streaming_usage_reaches_the_collected_response() {
    let server = MockServer::start(vec![Round::text("done").with_usage(120, 45)]).await;
    let ai = runtime_pointed_at(&server, "usage-single-round").await;

    let response = ai
        .prompt("hello")
        .collect()
        .await
        .expect("the scripted round must complete");

    assert_eq!(response.usage.input_tokens, 120);
    assert_eq!(response.usage.output_tokens, 45);

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn usage_is_summed_across_every_round_of_the_tool_loop() {
    // Two rounds with deliberately distinct counts: a loop that reported only
    // the last round would say 200/20, and one that reported only the first
    // would say 100/10. The correct answer is the sum, and only the sum.
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})).with_usage(100, 10),
        Round::text("all done").with_usage(200, 20),
    ])
    .await;
    let ai = runtime_pointed_at(&server, "usage-multi-round").await;

    let response = ai
        .prompt("use the tool")
        .with_tool(echo_tool(), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("both scripted rounds must complete");

    assert_eq!(server.request_count(), 2, "the test must drive two rounds");
    assert_eq!(response.usage.input_tokens, 300);
    assert_eq!(response.usage.output_tokens, 30);

    // The tool the loop ran must have reached the wire — otherwise the second
    // round happened for some other reason and the sum above proves nothing.
    let first = server.requests().remove(0);
    let echo = tool_named(&first, "echo").expect("the echo tool must be offered to the model");
    assert!(
        !contains_ref(&echo["function"]["parameters"]),
        "tool schemas must be self-contained on the wire: {echo}"
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn cached_prompt_tokens_are_split_out_of_the_input_count() {
    // OpenAI reports `prompt_tokens` inclusive of `cached_tokens`; the client
    // subtracts so `input_tokens` means "uncached input" on every provider.
    let server = MockServer::start(vec![Round::text("done").with_cached_usage(100, 5, 40)]).await;
    let ai = runtime_pointed_at(&server, "usage-cached").await;

    let response = ai.prompt("hello").collect().await.expect("round completes");

    assert_eq!(response.usage.input_tokens, 60);
    assert_eq!(response.usage.cache_read_tokens, 40);
    assert_eq!(response.usage.output_tokens, 5);

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 2. The request actually asks for usage
// =============================================================================

#[tokio::test]
async fn streaming_requests_ask_the_server_to_include_usage() {
    // Without this key on the wire, a real OpenAI server never sends the
    // final usage chunk and every figure above would silently be zero.
    let server = MockServer::start(vec![Round::text("done").with_usage(1, 1)]).await;
    let ai = runtime_pointed_at(&server, "usage-stream-options").await;

    ai.prompt("hello").collect().await.expect("round completes");

    let request = server.requests().pop().expect("one request was recorded");
    assert_eq!(
        request["stream_options"]["include_usage"],
        json!(true),
        "streaming requests must set stream_options.include_usage: {request}"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 5. Missing usage degrades to zero, never to an error
// =============================================================================

#[tokio::test]
async fn a_server_that_reports_no_usage_degrades_to_zero() {
    // `Round::text` without `.with_usage` emits no usage chunk at all —
    // exactly what an OpenAI-compatible server that ignores `stream_options`
    // does. The prompt must still succeed.
    let server = MockServer::start(vec![Round::text("done")]).await;
    let ai = runtime_pointed_at(&server, "usage-absent").await;

    let response = ai
        .prompt("hello")
        .collect()
        .await
        .expect("absent usage must not fail the request");

    assert_eq!(response.text, "done");
    assert!(
        response.usage.is_empty(),
        "unreported usage must read as zeros, got {:?}",
        response.usage
    );

    ai.shutdown().await.expect("clean shutdown");
}
