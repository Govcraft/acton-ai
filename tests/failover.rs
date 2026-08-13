//! End-to-end tests for failover chains, circuit breaking, and model
//! degradation.
//!
//! These drive the real stack — facade, prompt loop, provider actors, OpenAI
//! client, broker — against the scripted server in [`mock_llm`]. Two mock
//! servers stand in for two vendors, so "did the request actually go
//! somewhere else" is answered by the wire rather than by a mock's
//! expectations.
//!
//! # Determinism
//!
//! Nothing sleeps. Every wait is a barrier: a prompt blocks on the collector,
//! `broker.ask(FlushBroadcasts)` proves every broadcast so far is sitting in
//! each subscriber's inbox, and an `ask` to a subscriber afterwards proves it
//! processed that broadcast first, because mailboxes are FIFO.
//!
//! The half-open tests configure a **one-nanosecond** cooldown rather than
//! sleeping out a real one. Any deadline that short has already passed by the
//! time the next instruction runs, so "the cooldown elapsed" is a fact about
//! the state machine rather than a race with the clock. Expressing it needs
//! the builder API: TOML's `cooldown_secs` cannot say anything below a
//! second, and zero is rejected at launch.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{
    contains_ref, models_requested, provider_toml, runtime_pointed_at, tool_named, MockServer,
    Round,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// =============================================================================
// Harness
// =============================================================================

/// Collects every [`FailoverEvent`] the runtime broadcasts.
///
/// A shared `Vec` rather than a channel: these assertions are about the whole
/// sequence of events a turn produced, and a channel would make reading them
/// back another thing to synchronise.
#[derive(Clone, Default)]
struct EventLog {
    events: Arc<Mutex<Vec<FailoverEvent>>>,
}

impl EventLog {
    fn recorder(&self) -> impl Fn(FailoverEvent) + Send + Sync + 'static {
        let events = Arc::clone(&self.events);
        move |event| {
            events
                .lock()
                .expect("the event log must not be poisoned")
                .push(event);
        }
    }

    fn all(&self) -> Vec<FailoverEvent> {
        self.events
            .lock()
            .expect("the event log must not be poisoned")
            .clone()
    }

    /// The `kind()` label of every recorded event, in order.
    fn kinds(&self) -> Vec<&'static str> {
        self.all().iter().map(FailoverEvent::kind).collect()
    }
}

/// Waits until every broadcast issued so far has reached every subscriber's
/// inbox. See `tests/usage_accounting.rs` for why this is the barrier.
async fn flush_broadcasts(ai: &ActonAI) {
    ai.runtime()
        .broker()
        .ask(FlushBroadcasts)
        .await
        .expect("the broker must answer a flush");
}

/// A two-provider config: `primary` chains to `backup`.
///
/// Built from real TOML so the `failover` and `[circuit_breaker]` keys are
/// under test alongside the routing they configure.
fn chained_toml(primary: &MockServer, backup: &MockServer, breaker: &str) -> String {
    format!(
        "default_provider = \"primary\"\n\
         {}failover = [\"backup\"]\n{breaker}\n{}",
        provider_toml("primary", primary, "primary-model"),
        provider_toml("backup", backup, "backup-model"),
    )
}

/// Launches a runtime from TOML with a failover-event recorder attached.
async fn runtime_from_toml(toml: &str, app_name: &str, log: &EventLog) -> ActonAI {
    let config = acton_ai::config::from_str(toml).expect("the config must parse");
    ActonAI::builder()
        .app_name(app_name)
        .apply_config(config)
        .expect("the config must apply")
        .on_failover_event(log.recorder())
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

// =============================================================================
// 1. Chain routing
// =============================================================================

#[tokio::test]
async fn a_failed_primary_hands_the_round_to_the_next_provider_in_the_chain() {
    let primary = MockServer::start(vec![Round::server_error()]).await;
    let backup = MockServer::start(vec![Round::text("served by the backup")]).await;
    let log = EventLog::default();
    let ai = runtime_from_toml(&chained_toml(&primary, &backup, ""), "failover-chain", &log).await;

    let response = ai
        .prompt("who is up?")
        .collect()
        .await
        .expect("the chain must produce an answer even though the primary is down");

    assert_eq!(response.text, "served by the backup");
    // Both were tried, in order, exactly once.
    assert_eq!(primary.request_count(), 1);
    assert_eq!(backup.request_count(), 1);

    flush_broadcasts(&ai).await;
    assert!(
        log.all().contains(&FailoverEvent::FailedOver {
            from: "primary".to_string(),
            to: "backup".to_string(),
        }),
        "the hop must be observable, not just its outcome: {:?}",
        log.all(),
    );
}

#[tokio::test]
async fn the_provider_that_served_is_the_provider_that_is_billed() {
    let primary = MockServer::start(vec![Round::server_error()]).await;
    let backup = MockServer::start(vec![Round::text("served").with_usage(1_000, 500)]).await;
    let log = EventLog::default();
    let ai = runtime_from_toml(&chained_toml(&primary, &backup, ""), "failover-usage", &log).await;

    ai.prompt("who pays?")
        .collect()
        .await
        .expect("the backup must serve");
    flush_broadcasts(&ai).await;

    let usage = ai.usage().await.expect("usage must be readable");
    let backup_usage = usage
        .providers
        .get("backup")
        .expect("the serving provider must appear in the snapshot");
    assert_eq!(backup_usage.usage.input_tokens, 1_000);
    assert_eq!(backup_usage.usage.output_tokens, 500);
    // The primary never completed a request, so it must not be billed for
    // one — the tokens belong to whoever actually produced them.
    assert!(
        usage
            .providers
            .get("primary")
            .is_none_or(|p| p.usage.input_tokens == 0),
        "the failed primary must not be billed: {usage:?}",
    );
}

#[tokio::test]
async fn a_chain_with_nowhere_left_to_go_names_every_candidate_it_tried() {
    let primary = MockServer::start(vec![Round::server_error()]).await;
    let backup = MockServer::start(vec![Round::server_error()]).await;
    let log = EventLog::default();
    let ai = runtime_from_toml(
        &chained_toml(&primary, &backup, ""),
        "failover-exhausted",
        &log,
    )
    .await;

    let error = ai
        .prompt("anyone?")
        .collect()
        .await
        .expect_err("with every provider down there is no answer to give");

    assert!(
        error.is_all_providers_failed(),
        "an exhausted chain is its own failure mode, not a bare prompt failure: {error}",
    );
    let attempts = error
        .provider_attempts()
        .expect("an exhausted chain must carry its attempts");
    assert_eq!(
        attempts
            .iter()
            .map(|a| a.provider.as_str())
            .collect::<Vec<_>>(),
        vec!["primary", "backup"],
        "attempts are recorded in the order they were tried",
    );
    // This message is what an operator reads at 3am; both names have to be in
    // it without going and looking anything else up.
    let message = error.to_string();
    assert!(message.contains("`primary`"), "{message}");
    assert!(message.contains("`backup`"), "{message}");
}

// =============================================================================
// 2. Circuit breaking
// =============================================================================

#[tokio::test]
async fn the_circuit_opens_at_the_threshold_and_the_primary_stops_seeing_traffic() {
    // Threshold 2 with a cooldown long enough that nothing can reopen it
    // during the test.
    let primary = MockServer::start(vec![Round::server_error(), Round::server_error()]).await;
    let backup = MockServer::start(vec![
        Round::text("one"),
        Round::text("two"),
        Round::text("three"),
    ])
    .await;
    let log = EventLog::default();
    let ai = runtime_from_toml(
        &chained_toml(
            &primary,
            &backup,
            "[providers.primary.circuit_breaker]\nfailure_threshold = 2\ncooldown_secs = 300\n",
        ),
        "failover-breaker",
        &log,
    )
    .await;

    for prompt in ["one", "two", "three"] {
        ai.prompt(prompt)
            .collect()
            .await
            .expect("the backup serves every round");
        flush_broadcasts(&ai).await;
    }

    // Two failures opened the circuit; the third prompt never reached the
    // primary at all. The wire is the truth here.
    assert_eq!(
        primary.request_count(),
        2,
        "an open circuit must stop sending, not keep failing",
    );
    assert_eq!(backup.request_count(), 3);

    assert!(
        log.all().contains(&FailoverEvent::CircuitOpened {
            provider: "primary".to_string(),
            consecutive_failures: 2,
            cooldown_secs: 300,
        }),
        "opening is the event operators alert on: {:?}",
        log.all(),
    );
}

#[tokio::test]
async fn a_cooled_down_circuit_lets_one_real_request_through_and_closes_on_success() {
    // The probe is real traffic: the primary's second scripted round is the
    // one that proves it recovered.
    let primary = MockServer::start(vec![Round::server_error(), Round::text("recovered")]).await;
    let backup = MockServer::start(vec![Round::text("covering")]).await;
    let log = EventLog::default();

    let ai = ActonAI::builder()
        .app_name("failover-half-open")
        .provider_named(
            "primary",
            ProviderConfig::openai_compatible(primary.base_url(), "primary-model")
                // One failure opens it, and the cooldown is over before the
                // next instruction runs — see the module docs.
                .with_circuit_breaker(CircuitBreakerConfig::new(1, Duration::from_nanos(1)))
                .with_failover(["backup"]),
        )
        .provider_named(
            "backup",
            ProviderConfig::openai_compatible(backup.base_url(), "backup-model"),
        )
        .default_provider("primary")
        .on_failover_event(log.recorder())
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let first = ai
        .prompt("first")
        .collect()
        .await
        .expect("the backup covers the failure");
    assert_eq!(first.text, "covering");
    flush_broadcasts(&ai).await;
    assert!(log.kinds().contains(&"circuit_opened"), "{:?}", log.kinds());

    let second = ai
        .prompt("second")
        .collect()
        .await
        .expect("the half-open probe must be allowed through");
    assert_eq!(
        second.text, "recovered",
        "a half-open circuit sends the next real request to the primary",
    );
    assert_eq!(primary.request_count(), 2);
    assert_eq!(
        backup.request_count(),
        1,
        "once the primary is healthy again the backup is left alone",
    );

    flush_broadcasts(&ai).await;
    assert!(
        log.all().contains(&FailoverEvent::CircuitClosed {
            provider: "primary".to_string(),
        }),
        "recovery is as worth broadcasting as failure: {:?}",
        log.all(),
    );
}

#[tokio::test]
async fn a_circuit_still_inside_its_cooldown_stays_open() {
    let primary = MockServer::start(vec![Round::server_error(), Round::text("never asked")]).await;
    let backup = MockServer::start(vec![Round::text("first"), Round::text("second")]).await;
    let log = EventLog::default();
    let ai = runtime_from_toml(
        &chained_toml(
            &primary,
            &backup,
            "[providers.primary.circuit_breaker]\nfailure_threshold = 1\ncooldown_secs = 3600\n",
        ),
        "failover-still-open",
        &log,
    )
    .await;

    for prompt in ["first", "second"] {
        ai.prompt(prompt)
            .collect()
            .await
            .expect("the backup serves both");
    }

    assert_eq!(
        primary.request_count(),
        1,
        "an hour-long cooldown must not admit a probe on the very next round",
    );
    assert_eq!(backup.request_count(), 2);
}

// =============================================================================
// 3. Model degradation
// =============================================================================

#[tokio::test]
async fn a_rate_limited_provider_degrades_to_its_fallback_model_on_the_wire() {
    // One 429 with a retry-after, then a normal round. The second request is
    // the degraded one: same provider, cheaper model.
    let server = MockServer::start(vec![
        Round::rate_limited(60),
        Round::text("answered by the cheaper model").with_usage(10, 20),
    ])
    .await;
    let log = EventLog::default();
    let toml = format!(
        "{}fallback_model = \"cheap-model\"\n",
        provider_toml("solo", &server, "expensive-model"),
    );
    let ai = runtime_from_toml(&toml, "failover-degrade", &log).await;

    let response = ai
        .prompt("degrade me")
        .collect()
        .await
        .expect("a rate limit must degrade the model, not fail the turn");
    assert_eq!(response.text, "answered by the cheaper model");

    // The wire is the only honest witness to which model served.
    assert_eq!(
        models_requested(&server),
        vec!["expensive-model".to_string(), "cheap-model".to_string()],
        "the retry must go out under the fallback model",
    );

    flush_broadcasts(&ai).await;
    let usage = ai.usage().await.expect("usage must be readable");
    let solo = usage
        .provider("solo")
        .expect("the provider must have recorded usage");
    assert!(
        solo.model("cheap-model").is_some(),
        "the bill must name the model that served: {:?}",
        solo.models.keys().collect::<Vec<_>>(),
    );
    assert!(
        solo.model("expensive-model").is_none(),
        "the throttled model never served, so it must not be billed",
    );

    let degraded = log
        .all()
        .into_iter()
        .find_map(|event| match event {
            FailoverEvent::ModelDegraded {
                provider,
                from_model,
                to_model,
                retry_after_secs,
            } => Some((provider, from_model, to_model, retry_after_secs)),
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!(
                "nobody should discover a silent model swap from a bill: {:?}",
                log.all()
            )
        });
    assert_eq!(
        (
            degraded.0.as_str(),
            degraded.1.as_str(),
            degraded.2.as_str()
        ),
        ("solo", "expensive-model", "cheap-model"),
    );
    // The window reported is what is *left* of the provider's retry-after, so
    // it lands just under the 60 seconds the mock asked for rather than on it.
    assert!(
        (55..=60).contains(&degraded.3),
        "the event must carry the remaining window: {}",
        degraded.3,
    );
}

#[tokio::test]
async fn a_failed_over_round_carries_the_same_tools_to_the_next_provider() {
    // Mid-loop failover is the case that could silently drop context: the
    // backup has to receive the same tool definitions and the same
    // conversation, or the model it serves is answering a different question.
    let primary = MockServer::start(vec![
        Round::tool_call("call_1", "lookup", serde_json::json!({"key": "answer"})),
        Round::server_error(),
    ])
    .await;
    let backup = MockServer::start(vec![Round::text("the tool said 42")]).await;
    let log = EventLog::default();
    let ai = runtime_from_toml(&chained_toml(&primary, &backup, ""), "failover-tools", &log).await;

    let response = ai
        .prompt("look it up")
        .tool(
            "lookup",
            "Looks a key up",
            serde_json::json!({
                "type": "object",
                "properties": { "key": { "type": "string" } },
                "required": ["key"]
            }),
            |_args| async move { Ok(serde_json::json!({ "value": 42 })) },
        )
        .collect()
        .await
        .expect("the backup must finish the tool loop the primary started");
    assert_eq!(response.text, "the tool said 42");
    assert_eq!(response.tool_calls.len(), 1);

    let handed_over = &backup.requests()[0];
    let tool = tool_named(handed_over, "lookup")
        .expect("the backup must be offered the same tool the primary was");
    assert!(
        !contains_ref(tool),
        "schemas must stay self-contained across a hop: {tool}",
    );
    // The executed tool's result has to travel too, or the backup is being
    // asked to answer without it.
    let messages = handed_over["messages"]
        .as_array()
        .expect("a request must carry messages");
    assert!(
        messages
            .iter()
            .any(|message| message["role"] == "tool"
                && message["content"].to_string().contains("42")),
        "the tool result must be handed over: {messages:?}",
    );
}

// =============================================================================
// 4. Budget composition
// =============================================================================

#[tokio::test]
async fn a_candidate_the_budget_denies_is_skipped_and_recorded() {
    // The backup's own cap is already spent, so the chain has a healthy
    // provider it still may not use. That is a skip, not a crash — and the
    // final error has to say which.
    let primary = MockServer::start(vec![
        Round::text("first").with_usage(1_000_000, 0),
        Round::server_error(),
    ])
    .await;
    let backup = MockServer::start(vec![Round::text("must never be reached")]).await;
    let log = EventLog::default();
    let toml = format!(
        "default_provider = \"primary\"\n\
         {}failover = [\"backup\"]\n\
         [providers.primary.pricing]\ninput_per_mtok = 3.0\noutput_per_mtok = 15.0\n\
         {}[providers.backup.pricing]\ninput_per_mtok = 3.0\noutput_per_mtok = 15.0\n\
         [budget]\ntotal_usd = 2.00\n",
        provider_toml("primary", &primary, "primary-model"),
        provider_toml("backup", &backup, "backup-model"),
    );
    let ai = runtime_from_toml(&toml, "failover-budget", &log).await;

    ai.prompt("spend it")
        .collect()
        .await
        .expect("the first round is under the cap when it is checked");
    flush_broadcasts(&ai).await;

    let error = ai
        .prompt("try again")
        .collect()
        .await
        .expect_err("the cap is blown, so no candidate may be dispatched to");

    assert!(error.is_all_providers_failed(), "{error}");
    let attempts = error
        .provider_attempts()
        .expect("an exhausted chain must carry its attempts");
    assert_eq!(attempts.len(), 2);
    assert!(
        attempts.iter().all(|a| a.reason.contains("budget")),
        "each skip must say the cap refused it: {attempts:?}",
    );
    assert_eq!(
        backup.request_count(),
        0,
        "a budget-denied candidate is never dispatched to",
    );
}

// =============================================================================
// 5. Zero-change guarantee
// =============================================================================

#[tokio::test]
async fn a_provider_with_no_chain_reports_its_own_failure_as_before() {
    let server = MockServer::start(vec![Round::server_error()]).await;
    let ai = runtime_pointed_at(&server, "failover-unchained").await;

    let error = ai
        .prompt("no chain here")
        .collect()
        .await
        .expect_err("the provider is down");

    assert!(
        !error.is_all_providers_failed(),
        "with no chain there is no chain to report as exhausted: {error}",
    );
    assert!(
        error.to_string().contains("prompt execution failed"),
        "the pre-failover error is unchanged: {error}",
    );
}
