//! End-to-end tests for the `get_context_remaining` built-in tool.
//!
//! These drive the real stack — facade, prompt loop, provider actor, HTTP
//! client — against the scripted server in [`mock_llm`], so what they assert
//! is what actually travelled the wire: the tool offered in the first round,
//! and the budget report fed back to the model in the second.
//!
//! # Determinism
//!
//! Nothing sleeps. The loop cannot send a round until the previous round's
//! tool results exist, so a completed prompt is itself the barrier. The
//! expected token estimate is computed with the very same [`ContextWindow`]
//! the runtime is launched with, so the assertions hold whatever the
//! estimator's arithmetic is.

mod mock_llm;

use acton_ai::memory::{ContextWindow, ContextWindowConfig};
use acton_ai::messages::Message;
use acton_ai::prelude::*;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use serde_json::{json, Value};

const PROMPT: &str = "How much of my context window is spent?";

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

/// A server scripted to call the tool once and then finish the turn.
async fn scripted_server() -> MockServer {
    MockServer::start(vec![
        Round::tool_call("call_ctx", "get_context_remaining", json!({})),
        Round::text("plenty of room"),
    ])
    .await
}

#[tokio::test]
async fn the_tool_reports_the_configured_window_over_the_wire() {
    let server = scripted_server().await;

    // The same window the runtime gets, kept for computing the expectation.
    let window = ContextWindow::new(ContextWindowConfig::with_max_tokens(4096));
    let ai = ActonAI::builder()
        .app_name("ctx-remaining-e2e")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .context_window(window.clone())
        .with_builtins()
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let response = ai
        .prompt(PROMPT)
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(response.text, "plenty of room");
    assert_eq!(server.request_count(), 2);

    // with_builtins registration, observed on the wire: the first round
    // offered the tool to the model.
    let first = server.requests().remove(0);
    let offered = tool_named(&first, "get_context_remaining")
        .expect("with_builtins must advertise get_context_remaining");
    assert!(
        !contains_ref(&offered["function"]["parameters"]),
        "tool schemas must be self-contained on the wire: {offered}"
    );

    // At call time the loop's live state was exactly the user message, so
    // the report must match the runtime's own estimator over that history.
    let expected_used = window.estimate_total_tokens(&[Message::user(PROMPT)]);
    let second = server.requests().remove(1);
    let results = tool_results(&second);
    assert_eq!(results.len(), 1, "the call needs exactly one result");
    let report: Value = serde_json::from_str(&results[0]).expect("the tool result must be JSON");
    assert_eq!(report["total_tokens"], 4096);
    assert_eq!(report["used_tokens"], expected_used);
    assert_eq!(report["remaining_tokens"], 4096 - expected_used);
    let expected_percent = (expected_used as f64 / 4096.0 * 10_000.0).round() / 100.0;
    assert!(
        (report["percent_used"].as_f64().expect("percent_used") - expected_percent).abs()
            < f64::EPSILON,
        "percent_used must be used/total to two decimals: {report}"
    );

    // The injected measurement is loop-internal: the assistant message echoed
    // back to the model must carry the call exactly as the model made it.
    let assistant_args = second["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == "assistant")
        .and_then(|message| message["tool_calls"][0]["function"]["arguments"].as_str())
        .expect("the assistant echo carries the tool call")
        .to_string();
    assert!(
        !assistant_args.contains("_context_state"),
        "the injected state must not leak into the conversation: {assistant_args}"
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn absent_config_falls_back_to_the_default_window() {
    let server = scripted_server().await;

    // No [context] section, no per-provider context_window_tokens, no
    // override: the runtime resolves ContextWindowConfig::default().
    let ai = ActonAI::builder()
        .app_name("ctx-remaining-fallback")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_builtins()
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let response = ai
        .prompt(PROMPT)
        .collect()
        .await
        .expect("the turn must complete");
    assert_eq!(response.text, "plenty of room");

    let default_total = ContextWindowConfig::default().max_tokens;
    let second = server.requests().remove(1);
    let results = tool_results(&second);
    assert_eq!(results.len(), 1);
    let report: Value = serde_json::from_str(&results[0]).expect("the tool result must be JSON");
    assert_eq!(
        report["total_tokens"], default_total,
        "no config anywhere must fall back to the built-in default: {report}"
    );
    let used = report["used_tokens"].as_u64().expect("used_tokens") as usize;
    assert!(used > 0, "a non-empty history estimates above zero");
    assert_eq!(report["remaining_tokens"], default_total - used);

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn without_builtins_the_tool_is_not_offered() {
    let server = MockServer::start(vec![Round::text("nothing to see")]).await;
    let ai = runtime_pointed_at(&server, "ctx-remaining-absent").await;

    ai.prompt(PROMPT)
        .collect()
        .await
        .expect("the turn must complete");

    let first = server.requests().remove(0);
    assert!(
        tool_named(&first, "get_context_remaining").is_none(),
        "the tool is opt-in via with_builtins, not ambient"
    );

    ai.shutdown().await.expect("clean shutdown");
}
