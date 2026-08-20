//! End-to-end tests for budget enforcement.
//!
//! These drive the real stack — facade, prompt loop, provider actor, OpenAI
//! client, accountant, broker — against the scripted server in [`mock_llm`],
//! so what they assert is what a refusal actually does, not a mocked-out
//! approximation of one.
//!
//! # Determinism
//!
//! Nothing sleeps. Every wait is a barrier: a prompt blocks on the collector,
//! `broker.ask(FlushBroadcasts)` proves every broadcast so far is sitting in
//! each subscriber's inbox, and an `ask` to a subscriber afterwards proves it
//! processed that broadcast first, because mailboxes are FIFO.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, provider_toml, runtime_pointed_at, tool_named, MockServer, Round};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Rates that make the arithmetic legible: 1 MTok of input is exactly $3.00.
const PRICING: &str = "[providers.claude.pricing]\ninput_per_mtok = 3.0\noutput_per_mtok = 15.0\n";

/// Waits until every broadcast issued so far has reached every subscriber's
/// inbox. See `tests/usage_accounting.rs` for why this is the barrier.
async fn flush_broadcasts(ai: &ActonAI) {
    ai.runtime()
        .broker()
        .ask(FlushBroadcasts)
        .await
        .expect("the broker must answer a flush");
}

/// Launches a runtime from real TOML, so the `[budget]` path is under test
/// alongside the enforcement it configures.
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

/// A priced single-provider config pointing at `server`, plus a `[budget]`.
fn toml_with_budget(server: &MockServer, budget: &str) -> String {
    format!(
        "{}{PRICING}{budget}",
        provider_toml("claude", server, "sonnet-mock")
    )
}

// =============================================================================
// 1. Refusal
// =============================================================================

#[tokio::test]
async fn a_prompt_past_the_cap_is_refused_and_names_the_cap() {
    // The first round spends $3.00 against a $2.00 cap: it completes, because
    // enforcement is pre-flight and the cap was clear when it was checked.
    // The second prompt is the one that must be refused.
    let server = MockServer::start(vec![
        Round::text("first").with_usage(1_000_000, 0),
        Round::text("second").with_usage(1_000_000, 0),
    ])
    .await;
    let ai = runtime_from_toml(
        &toml_with_budget(&server, "[budget]\ntotal_usd = 2.00\n"),
        "budget-refusal",
    )
    .await;

    ai.prompt("first")
        .collect()
        .await
        .expect("the first prompt is under the cap when it is checked");
    flush_broadcasts(&ai).await;

    let err = ai
        .prompt("second")
        .collect()
        .await
        .expect_err("the second prompt must be refused");

    assert!(err.is_budget_exceeded(), "err = {err}");
    assert!(
        matches!(
            err.kind,
            ActonAIErrorKind::BudgetExceeded {
                scope: BudgetScope::Total,
                limit_microusd: 2_000_000,
                spent_microusd: 3_000_000,
            }
        ),
        "err = {err:?}"
    );

    let message = err.to_string();
    assert!(
        message.contains("$2.0000"),
        "the cap must appear: {message}"
    );
    assert!(
        message.contains("[budget]"),
        "the message must name the knob to turn: {message}"
    );

    assert_eq!(
        server.request_count(),
        1,
        "a refusal must not reach the wire"
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn a_provider_cap_refuses_only_that_provider() {
    let server = MockServer::start(vec![
        Round::text("first").with_usage(1_000_000, 0),
        Round::text("second").with_usage(1_000_000, 0),
    ])
    .await;

    // Two provider entries pointing at the same scripted server, so the only
    // thing separating them is the cap.
    // The root-level key leads: anything after a table header belongs to
    // that table.
    let toml = format!(
        "default_provider = \"claude\"\n{}{PRICING}{}\
         [providers.local.pricing]\ninput_per_mtok = 3.0\noutput_per_mtok = 15.0\n\
         [budget.providers]\nclaude = 2.00\n",
        provider_toml("claude", &server, "sonnet-mock"),
        provider_toml("local", &server, "qwen-mock"),
    );
    let ai = runtime_from_toml(&toml, "budget-per-provider").await;

    ai.prompt("first").collect().await.expect("under the cap");
    flush_broadcasts(&ai).await;

    let err = ai
        .prompt("second")
        .provider("claude")
        .collect()
        .await
        .expect_err("claude is over its cap");
    assert!(
        matches!(
            err.kind,
            ActonAIErrorKind::BudgetExceeded {
                scope: BudgetScope::Provider(ref name),
                ..
            } if name == "claude"
        ),
        "err = {err:?}"
    );

    // The uncapped provider is untouched by claude's ceiling.
    ai.prompt("second")
        .provider("local")
        .collect()
        .await
        .expect("local has no cap of its own and there is no total cap");

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn a_cap_refuses_the_tool_round_as_well_as_the_first_request() {
    // Round one calls a tool and spends past the cap; the loop must refuse
    // the follow-up request instead of completing the turn.
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", serde_json::json!({"value": "hi"}))
            .with_usage(1_000_000, 0),
        Round::text("never reached").with_usage(1_000_000, 0),
    ])
    .await;
    let ai = runtime_from_toml(
        &toml_with_budget(&server, "[budget]\ntotal_usd = 2.00\n"),
        "budget-tool-round",
    )
    .await;

    let err = ai
        .prompt("use the tool")
        .with_tool(
            ToolDefinition {
                idempotent: false,
                name: "echo".to_string(),
                description: "Echoes its argument back.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                }),
            },
            |args| async move { Ok(args) },
        )
        .collect()
        .await
        .expect_err("the second round must be refused");

    assert!(err.is_budget_exceeded(), "err = {err}");
    assert_eq!(
        server.request_count(),
        1,
        "only the first round may reach the wire"
    );

    // The refused round is the *second* one, so the first must have gone out
    // carrying the tool — otherwise the refusal proves nothing about tool
    // rounds being checked.
    let first = server.requests().remove(0);
    let echo = tool_named(&first, "echo").expect("the echo tool must be offered to the model");
    assert!(
        !contains_ref(&echo["function"]["parameters"]),
        "tool schemas must be self-contained on the wire: {echo}"
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn without_a_budget_nothing_is_refused() {
    let server = MockServer::start(vec![
        Round::text("first").with_usage(1_000_000_000, 0),
        Round::text("second").with_usage(1_000_000_000, 0),
    ])
    .await;
    let ai = runtime_pointed_at(&server, "budget-absent").await;

    ai.prompt("first").collect().await.expect("no cap");
    flush_broadcasts(&ai).await;
    ai.prompt("second")
        .collect()
        .await
        .expect("still no cap, however much was spent");

    assert!(!ai.is_budget_enforced());
    assert!(ai.usage().await.expect("usage").budget.is_none());

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 2. Status
// =============================================================================

#[tokio::test]
async fn the_usage_snapshot_reports_where_spending_stands() {
    let server = MockServer::start(vec![Round::text("done").with_usage(1_000_000, 0)]).await;
    let ai = runtime_from_toml(
        &toml_with_budget(
            &server,
            "[budget]\ntotal_usd = 4.00\n\n[budget.providers]\nclaude = 3.50\n",
        ),
        "budget-status",
    )
    .await;

    ai.prompt("hello").collect().await.expect("under the cap");
    flush_broadcasts(&ai).await;

    let usage = ai.usage().await.expect("usage");
    let budget = usage
        .budget
        .expect("a configured budget appears in the snapshot");

    let total = budget.total.expect("a total cap was configured");
    assert_eq!(total.spent_microusd, 3_000_000);
    assert_eq!(total.remaining_microusd(), 1_000_000);
    assert_eq!(total.percent_used(), 75);
    assert!((budget.remaining_usd().unwrap() - 1.0).abs() < f64::EPSILON);
    assert!(!budget.is_exceeded());

    let claude = budget.provider("claude").expect("a provider cap");
    assert_eq!(claude.remaining_microusd(), 500_000);

    // The figure the cap is compared against is the figure `usage()` reports.
    assert_eq!(
        usage.cost.expect("everything is priced").total_microusd,
        total.spent_microusd
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 3. Events
// =============================================================================

/// Collects every [`BudgetEvent`] on the broker and answers for them.
///
/// A test-owned subscriber, which is both the point (it proves the events are
/// broadcast for anyone, not just the facade's callback) and the barrier: the
/// `ask` below cannot be answered before the events ahead of it in the same
/// FIFO inbox have been handled.
#[acton_actor]
struct EventSpy {
    seen: Vec<String>,
}

#[acton_message]
struct GetSeenEvents;

impl Request for GetSeenEvents {
    type Response = SeenEvents;
}

#[acton_message]
struct SeenEvents {
    events: Vec<String>,
}

async fn spawn_event_spy(runtime: &mut ActorRuntime) -> ActorHandle {
    let mut builder = runtime.new_actor_with_name::<EventSpy>("event_spy".to_string());

    builder.mutate_on::<BudgetEvent>(|actor, envelope| {
        actor.model.seen.push(format!("{:?}", envelope.message()));
        Reply::ready()
    });

    builder.act_on::<GetSeenEvents>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let events = actor.model.seen.clone();
        Reply::pending(async move {
            reply.send(SeenEvents { events }).await;
        })
    });

    // On the builder, before start: a subscription registered afterwards is
    // silently ignored.
    builder.handle().subscribe::<BudgetEvent>().await;

    builder.start().await
}

#[tokio::test]
async fn a_warning_fires_once_however_many_reports_follow_it() {
    // $3.00 a round against a $10.00 cap warning at 80% ($8.00):
    // 30%, then 60%, then 90% — one crossing — then 120%, which is the cap
    // itself and must not repeat the warning.
    let server = MockServer::start(vec![
        Round::text("one").with_usage(1_000_000, 0),
        Round::text("two").with_usage(1_000_000, 0),
        Round::text("three").with_usage(1_000_000, 0),
        Round::text("four").with_usage(1_000_000, 0),
    ])
    .await;
    let mut ai = runtime_from_toml(
        &toml_with_budget(&server, "[budget]\ntotal_usd = 10.00\n"),
        "budget-warning",
    )
    .await;

    let spy = spawn_event_spy(ai.runtime_mut()).await;

    for prompt in ["one", "two", "three"] {
        ai.prompt(prompt).collect().await.expect("under the cap");
        flush_broadcasts(&ai).await;
        // Asking the accountant proves it has folded the report ahead of this
        // ask, and therefore broadcast any crossing the report caused.
        ai.usage().await.expect("usage");
    }
    // And that broadcast has reached the spy's inbox.
    flush_broadcasts(&ai).await;

    let seen = spy.ask(GetSeenEvents).await.expect("the spy must answer");
    let warnings = seen
        .events
        .iter()
        .filter(|event| event.contains("ThresholdCrossed"))
        .count();

    assert_eq!(
        warnings, 1,
        "the 80% warning must fire once, not once per report: {:?}",
        seen.events
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn crossing_the_cap_broadcasts_exceeded_once() {
    let server = MockServer::start(vec![
        Round::text("one").with_usage(1_000_000, 0),
        Round::text("two").with_usage(1_000_000, 0),
    ])
    .await;
    // Warnings off, so the only events are the cap itself.
    let mut ai = runtime_from_toml(
        &toml_with_budget(&server, "[budget]\ntotal_usd = 4.00\nwarn_at_percent = 0\n"),
        "budget-exceeded-event",
    )
    .await;

    let spy = spawn_event_spy(ai.runtime_mut()).await;

    ai.prompt("one").collect().await.expect("under the cap");
    flush_broadcasts(&ai).await;
    ai.usage().await.expect("usage");

    // $6.00 of a $4.00 cap: over, and the request that took it there is the
    // one already in flight, which is allowed to complete.
    ai.prompt("two")
        .collect()
        .await
        .expect("checked before the cap was hit");
    flush_broadcasts(&ai).await;
    ai.usage().await.expect("usage");
    flush_broadcasts(&ai).await;

    let seen = spy.ask(GetSeenEvents).await.expect("the spy must answer");
    assert_eq!(
        seen.events.len(),
        1,
        "exactly one event, and it must be the cap: {:?}",
        seen.events
    );
    assert!(seen.events[0].contains("Exceeded"), "{:?}", seen.events);

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn the_on_budget_event_callback_fires() {
    let server = MockServer::start(vec![Round::text("one").with_usage(1_000_000, 0)]).await;

    // A channel, not a lock plus a poll: `recv().await` is a barrier that
    // completes exactly when the callback has run.
    let (tx, mut rx) = mpsc::unbounded_channel();
    let tx = Arc::new(tx);

    let toml = format!(
        "{}{PRICING}[budget]\ntotal_usd = 2.00\n",
        provider_toml("claude", &server, "sonnet-mock")
    );
    let config = acton_ai::config::from_str(&toml).expect("the config must parse");
    let ai = ActonAI::builder()
        .app_name("budget-callback")
        .apply_config(config)
        .expect("apply_config")
        .on_budget_event(move |event| {
            let _ = tx.send(event.to_string());
        })
        .launch()
        .await
        .expect("launch");

    ai.prompt("one")
        .collect()
        .await
        .expect("the first round completes");

    // $3.00 of a $2.00 cap crosses both the 80% warning and the cap itself.
    let first = rx.recv().await.expect("the callback must fire");
    let second = rx.recv().await.expect("the cap event must follow");

    assert!(first.contains("total budget at"), "first = {first}");
    assert!(second.contains("exceeded"), "second = {second}");

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 4. Fail-closed configuration
// =============================================================================

#[tokio::test]
async fn a_budget_over_an_unpriced_provider_fails_the_launch() {
    let server = MockServer::start(vec![Round::text("unused")]).await;
    let toml = format!(
        "{}[budget]\ntotal_usd = 5.00\n",
        provider_toml("claude", &server, "sonnet-mock")
    );
    let config = acton_ai::config::from_str(&toml).expect("the config must parse");

    let err = ActonAI::builder()
        .app_name("budget-unpriced")
        .apply_config(config)
        .expect("apply_config")
        .launch()
        .await
        .expect_err("an unpriced provider under a budget must fail closed");

    let message = err.to_string();
    assert!(message.contains("claude"), "err = {message}");
    assert!(message.contains("input_per_mtok"), "err = {message}");
}

#[tokio::test]
async fn allow_unpriced_launches_and_counts_that_usage_as_zero() {
    let server = MockServer::start(vec![
        Round::text("one").with_usage(1_000_000_000, 0),
        Round::text("two").with_usage(1_000_000_000, 0),
    ])
    .await;
    let toml = format!(
        "{}[budget]\ntotal_usd = 5.00\nallow_unpriced = true\n",
        provider_toml("claude", &server, "sonnet-mock")
    );
    let ai = runtime_from_toml(&toml, "budget-allow-unpriced").await;

    ai.prompt("one").collect().await.expect("launch succeeded");
    flush_broadcasts(&ai).await;
    ai.prompt("two")
        .collect()
        .await
        .expect("unpriced usage counts as $0, so the cap is never approached");

    let budget = ai.usage().await.expect("usage").budget.expect("a budget");
    assert_eq!(budget.total.expect("a total cap").spent_microusd, 0);
    assert!(budget.allow_unpriced);

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn unpriced_usage_reaching_the_accountant_refuses_the_next_request() {
    // `claude` is priced and capped; `rogue` is not configured at all, so its
    // report is spending the cap cannot see. Reported straight to the broker,
    // which is exactly the low-level path the launch check cannot cover.
    let server = MockServer::start(vec![Round::text("one").with_usage(1_000, 0)]).await;
    let ai = runtime_from_toml(
        &toml_with_budget(&server, "[budget]\ntotal_usd = 5.00\n"),
        "budget-runtime-unpriced",
    )
    .await;

    ai.runtime()
        .broker()
        .broadcast(UsageReport {
            provider: "rogue".to_string(),
            model: "mystery".to_string(),
            correlation_id: CorrelationId::new(),
            agent_id: AgentId::new(),
            usage: Usage {
                input_tokens: 500,
                ..Usage::default()
            },
        })
        .await;
    flush_broadcasts(&ai).await;

    let err = ai
        .prompt("one")
        .collect()
        .await
        .expect_err("a cap that cannot see part of the spend must refuse");

    let message = err.to_string();
    assert!(err.is_configuration(), "err = {message}");
    assert!(message.contains("rogue"), "err = {message}");
    assert!(message.contains("allow_unpriced"), "err = {message}");

    ai.shutdown().await.expect("clean shutdown");
}
