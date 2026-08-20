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
use mock_llm::{contains_ref, provider_toml, runtime_pointed_at, tool_named, MockServer, Round};
use serde_json::json;

/// Waits until every broadcast issued so far has been handed to every
/// subscriber's inbox.
///
/// This is the barrier that makes the accountant assertions deterministic
/// without a sleep. A completed prompt only proves the *collector* saw the
/// round's `LLMStreamEnd`; the `UsageReport` that follows it may still be in
/// flight. `FlushBroadcasts` cannot answer until it is not, and because
/// mailboxes are FIFO the `GetUsage` issued afterwards is necessarily
/// processed behind it.
async fn flush_broadcasts(ai: &ActonAI) {
    ai.runtime()
        .broker()
        .ask(FlushBroadcasts)
        .await
        .expect("the broker must answer a flush");
}

/// Launches a runtime from real TOML, so the config path is under test too.
async fn runtime_from_toml(toml: &str, app_name: &str) -> ActonAI {
    let config = acton_ai::config::from_str(toml).expect("the config must parse");
    ActonAI::builder()
        .app_name(app_name)
        .apply_config(config)
        .expect("the config must apply")
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

/// A tool the scripted rounds can call, so a turn can span several provider
/// requests and exercise cross-round usage summation.
fn echo_tool() -> ToolDefinition {
    ToolDefinition {
        idempotent: false,
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

// =============================================================================
// 3. The accountant tallies across providers
// =============================================================================

#[tokio::test]
async fn the_accountant_tallies_per_provider_per_model_and_in_total() {
    let claude = MockServer::start(vec![Round::text("a").with_usage(100, 10)]).await;
    let local = MockServer::start(vec![Round::text("b").with_usage(7, 3)]).await;

    // `default_provider` is a root key, so it must precede every [table]
    // header — a bare key after one belongs to that table.
    let toml = format!(
        "default_provider = \"claude\"\n{}{}",
        provider_toml("claude", &claude, "sonnet-mock"),
        provider_toml("local", &local, "qwen-mock"),
    );
    let ai = runtime_from_toml(&toml, "usage-two-providers").await;

    ai.prompt("one").provider("claude").collect().await.unwrap();
    ai.prompt("two").provider("local").collect().await.unwrap();

    flush_broadcasts(&ai).await;
    let usage = ai.usage().await.expect("tracking is on by default");

    // Grand totals.
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.totals.input_tokens, 107);
    assert_eq!(usage.totals.output_tokens, 13);

    // Per provider — keyed by the CONFIGURED name, not the vendor. Both of
    // these are "openai" clients, so a vendor-keyed tally would collapse them
    // into one entry of 107 tokens.
    let claude_tally = usage.provider("claude").expect("claude must be tallied");
    assert_eq!(claude_tally.usage.input_tokens, 100);
    assert_eq!(claude_tally.requests, 1);

    let local_tally = usage.provider("local").expect("local must be tallied");
    assert_eq!(local_tally.usage.input_tokens, 7);
    assert_eq!(local_tally.requests, 1);

    // Per model within each provider.
    assert_eq!(
        claude_tally
            .model("sonnet-mock")
            .unwrap()
            .usage
            .output_tokens,
        10
    );
    assert_eq!(
        local_tally.model("qwen-mock").unwrap().usage.output_tokens,
        3
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn usage_accumulates_across_successive_prompts() {
    let server = MockServer::start(vec![
        Round::text("one").with_usage(10, 1),
        Round::text("two").with_usage(20, 2),
    ])
    .await;
    let ai = runtime_pointed_at(&server, "usage-accumulates").await;

    ai.prompt("first").collect().await.unwrap();
    ai.prompt("second").collect().await.unwrap();

    flush_broadcasts(&ai).await;
    let usage = ai.usage().await.unwrap();

    assert_eq!(usage.requests, 2);
    assert_eq!(usage.totals.input_tokens, 30);
    assert_eq!(usage.totals.output_tokens, 3);

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 4. Pricing
// =============================================================================

#[tokio::test]
async fn configured_pricing_produces_exact_integer_costs() {
    // 1_500_000 input @ $3/MTok = 4_500_000 micro-USD.
    //   500_000 output @ $15/MTok = 7_500_000 micro-USD.
    // Total 12_000_000 micro-USD = $12.00 — computed by hand, asserted exactly.
    let server = MockServer::start(vec![Round::text("done").with_usage(1_500_000, 500_000)]).await;

    let toml = format!(
        "{}\n[providers.claude.pricing]\ninput_per_mtok = 3.0\noutput_per_mtok = 15.0\n",
        provider_toml("claude", &server, "sonnet-mock"),
    );
    let ai = runtime_from_toml(&toml, "usage-pricing").await;

    ai.prompt("hello").collect().await.unwrap();
    flush_broadcasts(&ai).await;
    let usage = ai.usage().await.unwrap();

    let cost = usage.provider("claude").unwrap().cost.expect("priced");
    assert_eq!(cost.input_microusd, 4_500_000);
    assert_eq!(cost.output_microusd, 7_500_000);
    assert_eq!(cost.total_microusd, 12_000_000);
    assert!((cost.total_usd() - 12.0).abs() < f64::EPSILON);

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn pricing_floors_sub_microdollar_remainders() {
    // 7 input tokens @ $0.5/MTok = 3.5 micro-USD, which floors to 3.
    let server = MockServer::start(vec![Round::text("done").with_usage(7, 0)]).await;

    let toml = format!(
        "{}\n[providers.claude.pricing]\ninput_per_mtok = 0.5\noutput_per_mtok = 0.5\n",
        provider_toml("claude", &server, "sonnet-mock"),
    );
    let ai = runtime_from_toml(&toml, "usage-pricing-floor").await;

    ai.prompt("hello").collect().await.unwrap();
    flush_broadcasts(&ai).await;
    let usage = ai.usage().await.unwrap();

    assert_eq!(
        usage
            .provider("claude")
            .unwrap()
            .cost
            .unwrap()
            .total_microusd,
        3,
        "the remainder floors; it never rounds up"
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn an_unpriced_provider_still_counts_tokens_but_reports_no_cost() {
    let priced = MockServer::start(vec![Round::text("a").with_usage(1_000_000, 0)]).await;
    let unpriced = MockServer::start(vec![Round::text("b").with_usage(4_242, 99)]).await;

    let toml = format!(
        "default_provider = \"claude\"\n{}\n[providers.claude.pricing]\ninput_per_mtok = 3.0\noutput_per_mtok = 15.0\n\n{}",
        provider_toml("claude", &priced, "sonnet-mock"),
        provider_toml("local", &unpriced, "qwen-mock"),
    );
    let ai = runtime_from_toml(&toml, "usage-partial-pricing").await;

    ai.prompt("one").provider("claude").collect().await.unwrap();
    ai.prompt("two").provider("local").collect().await.unwrap();

    flush_broadcasts(&ai).await;
    let usage = ai.usage().await.unwrap();

    // The unpriced provider's tokens are counted in full...
    let local = usage.provider("local").unwrap();
    assert_eq!(local.usage.input_tokens, 4_242);
    assert_eq!(local.usage.output_tokens, 99);
    assert_eq!(local.requests, 1);

    // ...but its cost is unknown, never a fabricated $0.00.
    assert!(
        local.cost.is_none(),
        "an unpriced provider must not price to zero"
    );

    // The priced one is unaffected.
    assert_eq!(
        usage
            .provider("claude")
            .unwrap()
            .cost
            .unwrap()
            .total_microusd,
        3_000_000
    );

    // And no grand total is offered, because one that silently omitted the
    // unpriced provider would understate the bill.
    assert!(usage.cost.is_none());
    assert!(usage.total_usd().is_none());

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 5. The toggle
// =============================================================================

#[tokio::test]
async fn disabling_tracking_makes_usage_an_error_not_an_empty_snapshot() {
    let server = MockServer::start(vec![Round::text("done").with_usage(100, 10)]).await;
    let config = acton_ai::config::from_str(&provider_toml("claude", &server, "sonnet-mock"))
        .expect("config parses");
    let ai = ActonAI::builder()
        .app_name("usage-disabled")
        .apply_config(config)
        .expect("config applies")
        .usage_tracking(false)
        .launch()
        .await
        .expect("launching must succeed");

    assert!(!ai.is_usage_tracking());

    // Prompting still works — the toggle governs only whether anything
    // listens to the reports providers broadcast regardless.
    let response = ai
        .prompt("hello")
        .collect()
        .await
        .expect("prompts still work");
    assert_eq!(response.text, "done");
    assert_eq!(response.usage.input_tokens, 100);

    let error = ai
        .usage()
        .await
        .expect_err("usage must not answer when disabled");
    assert!(
        error.is_configuration(),
        "expected a configuration error, got {error}"
    );
    let message = error.to_string();
    assert!(
        message.contains("usage_tracking"),
        "the error must name the knob that turns it back on: {message}"
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn tracking_is_on_by_default() {
    let server = MockServer::start(vec![Round::text("done").with_usage(1, 1)]).await;
    let ai = runtime_pointed_at(&server, "usage-default-on").await;

    assert!(ai.is_usage_tracking());
    assert!(
        ai.usage().await.is_ok(),
        "a runtime that configured nothing must still track usage"
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn the_toml_toggle_turns_tracking_off() {
    let server = MockServer::start(vec![Round::text("done").with_usage(1, 1)]).await;
    let toml = format!(
        "{}\n[defaults]\nusage_tracking = false\n",
        provider_toml("claude", &server, "sonnet-mock"),
    );
    let ai = runtime_from_toml(&toml, "usage-toml-off").await;

    assert!(!ai.is_usage_tracking());
    assert!(ai.usage().await.is_err());

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn the_builder_toggle_wins_over_the_toml_one() {
    let server = MockServer::start(vec![Round::text("done").with_usage(1, 1)]).await;
    let toml = format!(
        "{}\n[defaults]\nusage_tracking = false\n",
        provider_toml("claude", &server, "sonnet-mock"),
    );
    let config = acton_ai::config::from_str(&toml).expect("config parses");

    let ai = ActonAI::builder()
        .app_name("usage-builder-wins")
        .usage_tracking(true)
        .apply_config(config)
        .expect("config applies")
        .launch()
        .await
        .expect("launching must succeed");

    assert!(
        ai.is_usage_tracking(),
        "an explicit builder call must outrank the config file"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 6. Shutdown
// =============================================================================

#[tokio::test]
async fn the_accountant_does_not_hang_shutdown() {
    let server = MockServer::start(vec![Round::text("done").with_usage(5, 5)]).await;
    let ai = runtime_pointed_at(&server, "usage-shutdown").await;

    ai.prompt("hello").collect().await.unwrap();
    flush_broadcasts(&ai).await;
    ai.usage().await.unwrap();

    // A supervised or IO-holding actor could stall here; a plain one must not.
    ai.shutdown()
        .await
        .expect("the accountant must shut down with everything else");
}

// =============================================================================
// 7. Conversation turns
// =============================================================================

#[tokio::test]
async fn each_conversation_turn_reports_its_own_usage() {
    // A Conversation turn runs the same prompt loop as `prompt().collect()`,
    // so the cross-round summation proved above applies here too. What is
    // specific to conversations is that the figure is *per turn*: the second
    // send must not carry the first's usage forward, even though the two
    // share one collector session and one history.
    let server = MockServer::start(vec![
        Round::text("first answer").with_usage(100, 10),
        Round::text("second answer").with_usage(5, 1),
    ])
    .await;
    let ai = runtime_pointed_at(&server, "usage-conversation").await;

    let conversation = ai.conversation().build().await;

    let first = conversation.send("hello").await.expect("first turn");
    assert_eq!(first.usage.input_tokens, 100);
    assert_eq!(first.usage.output_tokens, 10);

    let second = conversation.send("thanks").await.expect("second turn");
    assert_eq!(
        second.usage.input_tokens, 5,
        "each turn reports its own usage, not a running total"
    );
    assert_eq!(second.usage.output_tokens, 1);

    // The accountant is what holds the running total across turns.
    flush_broadcasts(&ai).await;
    let usage = ai.usage().await.unwrap();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.totals.input_tokens, 105);

    ai.shutdown().await.expect("clean shutdown");
}
