//! End-to-end tests for transient-stream retry in the prompt loop.
//!
//! A hosted provider's stream can die after the 200 headers — a dropped
//! connection, an in-band `data: {"error": …}` event — and that failure is
//! transient: the same request, dispatched again, usually succeeds. These
//! tests drive the real stack (facade, prompt loop, provider actor, OpenAI
//! client) against the scripted server in [`mock_llm`] and assert on the
//! wire: how many requests actually went out, and what the caller got back.
//!
//! # Determinism
//!
//! The only waits are the prompt loop's own retry backoffs (fixed, bounded)
//! and the collector barrier every prompt already blocks on. The mock serves
//! its script in order, so request counts are exact.

mod mock_llm;

use mock_llm::{runtime_pointed_at, MockServer, Round};

#[tokio::test]
async fn a_round_killed_mid_stream_is_retried_and_the_retry_answers() {
    let server = MockServer::start(vec![
        Round::mid_stream_error("partial ", "connection reset by peer"),
        Round::text("recovered on the second attempt"),
    ])
    .await;
    let runtime = runtime_pointed_at(&server, "stream-retry-test").await;

    let response = runtime
        .prompt("Say something.")
        .collect()
        .await
        .expect("a transient mid-stream failure must not fail the turn");

    // The retry's answer stands alone: the failed attempt's partial text
    // must not leak into the final response.
    assert_eq!(response.text, "recovered on the second attempt");
    assert_eq!(
        server.request_count(),
        2,
        "the round should have been dispatched exactly twice"
    );
}

#[tokio::test]
async fn a_persistently_broken_stream_fails_after_bounded_retries() {
    // One more scripted failure than the loop will ever ask for, so a
    // runaway retry loop shows up as a wrong request count rather than a
    // "no scripted round" server error.
    let server = MockServer::start(vec![
        Round::mid_stream_error("a", "connection reset by peer"),
        Round::mid_stream_error("b", "connection reset by peer"),
        Round::mid_stream_error("c", "connection reset by peer"),
        Round::mid_stream_error("d", "connection reset by peer"),
    ])
    .await;
    let runtime = runtime_pointed_at(&server, "stream-retry-exhaustion-test").await;

    let result = runtime.prompt("Say something.").collect().await;

    assert!(
        result.is_err(),
        "a failure that survives every retry must surface as an error"
    );
    assert_eq!(
        server.request_count(),
        3,
        "the loop should stop at the original dispatch plus two retries"
    );
}

#[tokio::test]
async fn a_hallucinated_tool_rejection_feeds_the_correction_into_the_retry() {
    let server = MockServer::start(vec![
        Round::mid_stream_error(
            "",
            "Tool call validation failed: tool call validation failed: \
             attempted to call tool 'repo_browser.print_tree' which was not in request.tools",
        ),
        Round::text("used a real tool this time"),
    ])
    .await;
    let runtime = runtime_pointed_at(&server, "stream-retry-correction-test").await;

    let response = runtime
        .prompt("Explore the repository.")
        .collect()
        .await
        .expect("the corrected retry must answer");
    assert_eq!(response.text, "used a real tool this time");

    // The retry is not a blind re-dispatch: its context must carry the
    // feedback an unknown-tool result would have carried, naming the tool
    // the model invented.
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    let retry_messages = requests[1]["messages"]
        .as_array()
        .expect("the retry must carry messages");
    let last = retry_messages
        .last()
        .expect("the retry must not have an empty context");
    assert_eq!(last["role"], "user");
    assert!(
        last["content"]
            .as_str()
            .expect("the correction is text")
            .contains("'repo_browser.print_tree'"),
        "the correction must name the hallucinated tool: {last}"
    );
}
