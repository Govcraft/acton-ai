//! Compile-time contract: a turn can live inside `Reply::pending`.
//!
//! acton-reactive's handler futures are `Pin<Box<dyn Future + Send + Sync>>`
//! (its `FutureBox`), so an embedder that drives a turn from inside one of
//! its own actor handlers needs `PromptBuilder` — and the future `collect()`
//! / `extract()` returns — to be `Send + Sync`. Before this contract existed,
//! every such embedder had to `tokio::task::spawn` the turn and shuttle the
//! result back, giving up structured cancellation and supervision for a
//! trait bound.
//!
//! `assert_send_sync` makes the contract a compile error rather than a
//! runtime discovery: if a future field or a callback slot regresses to
//! `!Sync`, this file stops building and names the culprit.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use serde_json::json;

fn assert_send_sync<T: Send + Sync>(value: T) -> T {
    value
}

#[tokio::test]
async fn a_bare_prompt_and_its_collect_future_are_send_and_sync() {
    let server = MockServer::start(vec![Round::text("ok")]).await;
    let ai = runtime_pointed_at(&server, "sync-probe-bare").await;

    let builder = assert_send_sync(ai.prompt("hello"));
    let response = assert_send_sync(builder.collect())
        .await
        .expect("the prompt");
    assert_eq!(response.text, "ok");
    assert_eq!(server.request_count(), 1);

    ai.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn a_fully_loaded_prompt_keeps_the_contract() {
    // Callbacks, a tool, and a tool-result callback are exactly the slots
    // that used to be `!Sync` (bare boxed `FnMut`s and `dyn Future + Send`
    // tool futures), so this is the regression the test exists to catch.
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "hi"})),
        Round::text("echoed"),
    ])
    .await;
    let ai = runtime_pointed_at(&server, "sync-probe-loaded").await;

    let builder = assert_send_sync(
        ai.prompt("use the echo tool")
            .system("Echo things.")
            .on_start(|| {})
            .on_token(|_t| {})
            .on_end(|_reason| {})
            .with_tool_callback(
                ToolDefinition {
                    name: "echo".to_string(),
                    description: "Echoes its argument.".to_string(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    }),
                },
                |args| async move { Ok(args) },
                |_result| {},
            ),
    );

    let response = assert_send_sync(builder.collect())
        .await
        .expect("the tool round");
    assert_eq!(response.text, "echoed");
    assert_eq!(response.tool_calls.len(), 1);

    // The Sync plumbing must not have changed what actually went out on the
    // wire: the tool was advertised, with an inlined (no `$ref`) schema.
    let request = server
        .requests()
        .into_iter()
        .next()
        .expect("the first round went out");
    let echo = tool_named(&request, "echo").expect("the echo tool was advertised");
    assert!(!contains_ref(&echo["function"]["parameters"]), "{echo}");

    ai.shutdown().await.expect("shutdown");
}
