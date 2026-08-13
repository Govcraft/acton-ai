//! End-to-end tests for OpenTelemetry export.
//!
//! These drive the real stack — facade, prompt loop, provider actor, OpenAI
//! client, telemetry actor, broker — against the scripted server in
//! [`mock_llm`], with in-memory exporters standing in for OTLP. What they
//! assert is therefore what a collector would actually receive, not a mocked
//! approximation of it.
//!
//! # Determinism
//!
//! Nothing sleeps. Every wait is a barrier: a prompt blocks on the collector,
//! `broker.ask(FlushBroadcasts)` proves every broadcast so far is sitting in
//! each subscriber's inbox, and `guard.flush()` drives the batch span
//! processor and the periodic metrics reader on demand rather than waiting
//! out an export interval.
//!
//! # One runtime per process
//!
//! OpenTelemetry's providers are process-wide globals and last-writer-wins.
//! Each test here installs its own, which is only sound because
//! `cargo nextest run` — the runner this repo uses — runs every test in its
//! own process.
//!
//! # Why a hand-written span exporter
//!
//! `opentelemetry_sdk`'s `InMemorySpanExporter` **clears its buffer on
//! shutdown**, and its `keep_records_on_shutdown` builder option is
//! `#[cfg(test)]` inside that crate, so it is unavailable here. The
//! flush-on-shutdown test asserts on spans that exist *after* `shutdown()`,
//! so it needs an exporter that retains them. [`RecordingSpanExporter`] is
//! that, and it doubles as proof that the injectable seam accepts any
//! exporter rather than only the SDK's own.

mod mock_llm;

use acton_ai::prelude::*;
use acton_ai::telemetry::{install_with_exporters, Telemetry, TelemetryGuard};
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use std::sync::{Arc, Mutex};

// =============================================================================
// Harness
// =============================================================================

/// A span exporter that keeps everything it is given, including across
/// shutdown. See the module docs for why the SDK's own will not do.
#[derive(Debug, Clone, Default)]
struct RecordingSpanExporter {
    spans: Arc<Mutex<Vec<SpanData>>>,
}

impl RecordingSpanExporter {
    fn spans(&self) -> Vec<SpanData> {
        self.spans
            .lock()
            .expect("the span buffer must lock")
            .clone()
    }
}

impl SpanExporter for RecordingSpanExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.spans
            .lock()
            .expect("the span buffer must lock")
            .extend(batch);
        Ok(())
    }
}

/// Everything a test needs to read back what was exported.
struct Harness {
    guard: TelemetryGuard,
    spans: RecordingSpanExporter,
    metrics: InMemoryMetricExporter,
}

impl Harness {
    /// Installs in-memory providers through the same seam `launch()` uses.
    fn install() -> Self {
        let spans = RecordingSpanExporter::default();
        let metrics = InMemoryMetricExporter::default();
        let config = Telemetry::otlp("http://localhost:4318")
            .service_name("telemetry-test")
            .to_config()
            .expect("the test endpoint must be valid");

        let guard = install_with_exporters(&config, spans.clone(), metrics.clone());
        Self {
            guard,
            spans,
            metrics,
        }
    }

    /// Forces both providers to export, so assertions never wait on a timer.
    fn flush(&self) {
        self.guard.flush();
    }

    fn spans_named(&self, name: &str) -> Vec<SpanData> {
        self.spans
            .spans()
            .into_iter()
            .filter(|span| span.name == name)
            .collect()
    }

    /// Sums a `u64` counter's data points, optionally filtered to one
    /// attribute pair.
    fn counter_total(&self, metric: &str, attribute: Option<(&str, &str)>) -> u64 {
        let mut total = 0;
        for resource in self
            .metrics
            .get_finished_metrics()
            .expect("metrics must be readable")
        {
            for scope in resource.scope_metrics() {
                for m in scope.metrics().filter(|m| m.name() == metric) {
                    let AggregatedMetrics::U64(MetricData::Sum(sum)) = m.data() else {
                        continue;
                    };
                    for point in sum.data_points() {
                        let matches = attribute.is_none_or(|(key, value)| {
                            point
                                .attributes()
                                .any(|kv| kv.key.as_str() == key && kv.value.as_str() == value)
                        });
                        if matches {
                            total += point.value();
                        }
                    }
                }
            }
        }
        total
    }

    /// How many samples a duration histogram recorded.
    fn histogram_count(&self, metric: &str) -> u64 {
        let mut count = 0;
        for resource in self
            .metrics
            .get_finished_metrics()
            .expect("metrics must be readable")
        {
            for scope in resource.scope_metrics() {
                for m in scope.metrics().filter(|m| m.name() == metric) {
                    let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = m.data() else {
                        continue;
                    };
                    for point in histogram.data_points() {
                        count += point.count();
                    }
                }
            }
        }
        count
    }
}

/// Reads a span attribute, rendered as a string.
///
/// Owned rather than borrowed because `Value::as_str` returns a `Cow`:
/// string attributes borrow, but numeric and boolean ones are formatted on
/// demand. Returning the owned value keeps this correct for every attribute
/// type rather than quietly yielding nothing for the numeric ones.
fn attribute(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.as_str().into_owned())
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

/// Launches a runtime that emits into the already-installed providers.
///
/// Built from real TOML because that is also how the provider's model name
/// reaches the round spans, which one of these tests asserts on.
async fn runtime_with_telemetry(server: &MockServer, app_name: &str) -> ActonAI {
    let config = acton_ai::config::from_str(&mock_llm::provider_toml("mock", server, "mock-model"))
        .expect("the provider config must parse");

    ActonAI::builder()
        .app_name(app_name)
        .apply_config(config)
        .expect("the config must apply")
        .telemetry_from_globals()
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

/// A prompt with one calculator tool, scripted to be called once.
fn calculator_prompt(ai: &ActonAI) -> acton_ai::prompt::PromptBuilder {
    ai.prompt("What is 17 * 42?").tool(
        "calculator",
        "Evaluates an arithmetic expression",
        serde_json::json!({
            "type": "object",
            "properties": { "expression": { "type": "string" } },
            "required": ["expression"]
        }),
        |args| async move {
            // A deliberately identifiable argument and result, so the test
            // asserting they never reach a span has something to look for.
            let expression = args["expression"].as_str().unwrap_or_default().to_string();
            Ok(serde_json::json!({
                "echoed": expression,
                "result": "SECRET-TOOL-PAYLOAD-714"
            }))
        },
    )
}

// =============================================================================
// 1. Trace shape
// =============================================================================

#[tokio::test]
async fn a_turn_with_a_tool_round_yields_a_correctly_parented_span_tree() {
    let harness = Harness::install();
    let server = MockServer::start(vec![
        Round::tool_call(
            "call-1",
            "calculator",
            serde_json::json!({ "expression": "17 * 42" }),
        )
        .with_usage(100, 10),
        Round::text("714").with_usage(50, 5),
    ])
    .await;
    let ai = runtime_with_telemetry(&server, "telemetry-shape").await;

    calculator_prompt(&ai)
        .collect()
        .await
        .expect("the scripted rounds must complete");

    harness.flush();

    let turns = harness.spans_named("acton_ai.turn");
    let rounds = harness.spans_named("acton_ai.llm_round");
    let tools = harness.spans_named("acton_ai.tool");

    assert_eq!(turns.len(), 1, "one collect() is one turn");
    assert_eq!(rounds.len(), 2, "a tool round means a second dispatch");
    assert_eq!(tools.len(), 1, "the scripted round calls one tool");

    // Round spans are only worth anything if they correspond to requests that
    // actually went out. Counting spans alone would pass just as happily if
    // the loop emitted them without dispatching.
    assert_eq!(
        server.request_count(),
        rounds.len(),
        "every llm_round span must correspond to a real request on the wire"
    );

    // Parenting is the claim worth checking: a flat set of spans with the
    // right names would look identical in a count-only assertion.
    let turn = &turns[0];
    let turn_span_id = turn.span_context.span_id();
    assert!(
        !turn.parent_span_id.to_string().chars().any(|c| c != '0'),
        "the turn is a root span, not a child of ambient context"
    );
    for child in rounds.iter().chain(tools.iter()) {
        assert_eq!(
            child.parent_span_id, turn_span_id,
            "`{}` must be parented to the turn",
            child.name
        );
        assert_eq!(
            child.span_context.trace_id(),
            turn.span_context.trace_id(),
            "`{}` must share the turn's trace",
            child.name
        );
    }
}

#[tokio::test]
async fn round_spans_carry_the_provider_model_and_the_rounds_usage() {
    let harness = Harness::install();
    let server = MockServer::start(vec![Round::text("hello").with_usage(120, 34)]).await;
    let ai = runtime_with_telemetry(&server, "telemetry-attrs").await;

    ai.prompt("hi").collect().await.expect("the round runs");
    harness.flush();

    let rounds = harness.spans_named("acton_ai.llm_round");
    let round = rounds.first().expect("one round was dispatched");

    assert_eq!(
        attribute(round, "acton_ai.model").as_deref(),
        Some("mock-model")
    );
    assert!(
        attribute(round, "acton_ai.provider").is_some(),
        "the round names the provider it was billed to"
    );
    assert!(
        attribute(round, "acton_ai.correlation_id").is_some(),
        "IDs belong on spans, and this is where they go"
    );

    // Usage is recorded per round, not only summed on the turn.
    assert_eq!(
        attribute(round, "acton_ai.tokens.input").as_deref(),
        Some("120"),
        "the round records its own input tokens"
    );

    let turns = harness.spans_named("acton_ai.turn");
    let turn = turns.first().expect("the turn closed");
    assert_eq!(attribute(turn, "acton_ai.outcome").as_deref(), Some("ok"));
}

#[tokio::test]
async fn tool_arguments_and_results_never_reach_exported_span_data() {
    // The deliberate choice this test defends: payloads are user data and
    // unbounded, so a span carries the tool's name and outcome and nothing
    // else. A regression here is a data leak, not a cosmetic slip.
    let harness = Harness::install();
    let server = MockServer::start(vec![
        Round::tool_call(
            "call-1",
            "calculator",
            serde_json::json!({ "expression": "17 * 42" }),
        )
        .with_usage(10, 1),
        Round::text("714").with_usage(10, 1),
    ])
    .await;
    let ai = runtime_with_telemetry(&server, "telemetry-privacy").await;

    calculator_prompt(&ai).collect().await.expect("it runs");
    harness.flush();

    // First establish that the payloads genuinely existed in this run and
    // reached the model. Without this, the absence asserted below would also
    // be satisfied by the tool never having run at all — the test would pass
    // for entirely the wrong reason.
    let requests = server.requests();
    assert_eq!(
        requests.len(),
        2,
        "the tool round produced a second request"
    );

    let offered = tool_named(&requests[0], "calculator")
        .expect("the calculator tool must be offered to the model");
    let parameters = &offered["function"]["parameters"];
    // Checked before the `$ref` assertion below, which a missing schema would
    // otherwise satisfy vacuously.
    assert!(
        parameters.is_object(),
        "the tool's schema must actually be on the wire: {offered}"
    );
    assert!(
        !contains_ref(parameters),
        "tool schemas must be self-contained on the wire: {offered}"
    );

    let wire = format!("{requests:?}");
    assert!(
        wire.contains("17 * 42"),
        "the arguments must reach the model, or their absence from spans proves nothing"
    );
    assert!(
        wire.contains("SECRET-TOOL-PAYLOAD-714"),
        "the result must be fed back to the model, or its absence from spans proves nothing"
    );

    // Present on the wire, absent from telemetry — that is the whole claim.
    let rendered = format!("{:?}", harness.spans.spans());
    assert!(
        !rendered.contains("SECRET-TOOL-PAYLOAD-714"),
        "a tool result must never appear in span data"
    );
    assert!(
        !rendered.contains("17 * 42"),
        "tool arguments must never appear in span data"
    );

    // The name is recorded, so the absence above is discipline rather than
    // the tool span simply being missing.
    let tools = harness.spans_named("acton_ai.tool");
    assert_eq!(
        attribute(&tools[0], "acton_ai.tool").as_deref(),
        Some("calculator")
    );
}

// =============================================================================
// 2. Metrics
// =============================================================================

#[tokio::test]
async fn the_token_counter_sums_what_the_provider_reported() {
    let harness = Harness::install();
    let server = MockServer::start(vec![
        Round::tool_call(
            "call-1",
            "calculator",
            serde_json::json!({ "expression": "17 * 42" }),
        )
        .with_usage(100, 10),
        Round::text("714").with_usage(50, 5),
    ])
    .await;
    let ai = runtime_with_telemetry(&server, "telemetry-tokens").await;

    calculator_prompt(&ai).collect().await.expect("it runs");

    // Two barriers, no sleeps: the broadcasts reach the telemetry actor's
    // inbox, then an ask proves it processed them (mailboxes are FIFO).
    flush_broadcasts(&ai).await;
    ai.usage().await.expect("the accountant answers");
    harness.flush();

    assert_eq!(
        harness.counter_total("acton_ai.tokens", Some(("kind", "input"))),
        150,
        "100 + 50 input tokens across the two rounds"
    );
    assert_eq!(
        harness.counter_total("acton_ai.tokens", Some(("kind", "output"))),
        15,
        "10 + 5 output tokens across the two rounds"
    );
    // Tied to the wire rather than asserted as a bare literal: the counter is
    // only correct if it agrees with how many requests the server actually saw.
    assert_eq!(
        harness.counter_total("acton_ai.llm.requests", None),
        server.request_count() as u64,
        "the request counter must agree with what reached the wire"
    );
}

#[tokio::test]
async fn duration_histograms_record_one_sample_per_round_and_per_tool() {
    let harness = Harness::install();
    let server = MockServer::start(vec![
        Round::tool_call(
            "call-1",
            "calculator",
            serde_json::json!({ "expression": "17 * 42" }),
        )
        .with_usage(10, 1),
        Round::text("714").with_usage(10, 1),
    ])
    .await;
    let ai = runtime_with_telemetry(&server, "telemetry-durations").await;

    calculator_prompt(&ai).collect().await.expect("it runs");
    harness.flush();

    assert_eq!(
        harness.histogram_count("acton_ai.llm.request.duration"),
        2,
        "one latency sample per dispatch"
    );
    assert_eq!(
        harness.histogram_count("acton_ai.tool.duration"),
        1,
        "one duration sample per tool execution"
    );
    // Only one: the first scripted round emits a tool call and no text, so it
    // never has a first token. Recording one anyway would report a latency
    // that nothing actually waited for.
    assert_eq!(
        harness.histogram_count("acton_ai.llm.time_to_first_token"),
        1,
        "only the round that streams text has a first token"
    );
}

// =============================================================================
// 3. Composition with budgets (#7)
// =============================================================================

#[tokio::test]
async fn a_budget_denial_is_recorded_on_the_turn_span_as_budget_exceeded() {
    // The label operators filter on once a cap is in force. It composes with
    // the pre-flight budget check rather than duplicating it.
    let harness = Harness::install();
    let server = MockServer::start(vec![
        Round::text("first").with_usage(1_000_000, 0),
        Round::text("second").with_usage(1_000_000, 0),
    ])
    .await;

    let toml = format!(
        "{}[providers.claude.pricing]\ninput_per_mtok = 3.0\noutput_per_mtok = 15.0\n\
         [budget]\ntotal_usd = 2.00\n",
        mock_llm::provider_toml("claude", &server, "sonnet-mock")
    );
    let config = acton_ai::config::from_str(&toml).expect("the config must parse");
    let ai = ActonAI::builder()
        .app_name("telemetry-budget")
        .apply_config(config)
        .expect("the config must apply")
        .telemetry_from_globals()
        .launch()
        .await
        .expect("launching must succeed");

    // The first prompt spends $3.00 against a $2.00 cap; it completes because
    // enforcement is pre-flight and the cap was clear when it was checked.
    ai.prompt("first").collect().await.expect("under the cap");
    flush_broadcasts(&ai).await;

    let refusal = ai
        .prompt("second")
        .collect()
        .await
        .expect_err("the cap must refuse the second prompt");
    assert!(matches!(
        refusal.kind,
        acton_ai::error::ActonAIErrorKind::BudgetExceeded { .. }
    ));

    harness.flush();

    let turns = harness.spans_named("acton_ai.turn");
    assert_eq!(turns.len(), 2, "a refused turn is still a turn");
    let outcomes: Vec<Option<String>> = turns
        .iter()
        .map(|turn| attribute(turn, "acton_ai.outcome"))
        .collect();
    assert!(
        outcomes
            .iter()
            .any(|o| o.as_deref() == Some("budget_exceeded")),
        "the refused turn records the budget denial; got {outcomes:?}"
    );

    // The refusal is pre-flight, so the refused turn dispatched nothing and
    // must say so. Counting the loop iteration it never completed would
    // overstate every denied turn by one round.
    let refused = turns
        .iter()
        .find(|turn| attribute(turn, "acton_ai.outcome").as_deref() == Some("budget_exceeded"))
        .expect("the refused turn was found above");
    assert_eq!(
        attribute(refused, "acton_ai.rounds").as_deref(),
        Some("0"),
        "a turn refused before dispatch made no rounds"
    );

    // The wire agrees: the refusal is pre-flight, so the second prompt cost
    // nothing. This is what makes the zero above meaningful rather than a
    // number the span happened to carry.
    assert_eq!(
        server.request_count(),
        1,
        "only the first prompt may reach the wire; the refused one must not"
    );

    // The budget event the accountant broadcast was counted too.
    assert!(
        harness.counter_total("acton_ai.budget.events", None) >= 1,
        "crossing a cap is a counted event"
    );
}

// =============================================================================
// 4. Flush on shutdown
// =============================================================================

#[tokio::test]
async fn spans_recorded_just_before_shutdown_survive_it() {
    // Without a flush ordered after the actors stop, the last batch is still
    // sitting in the batch processor when the process ends — and the end of a
    // run is exactly the part an operator went looking for.
    //
    // The runtime must *own* the providers for this to be its job, so this
    // test hands the guard over rather than using `telemetry_from_globals`,
    // where the application keeps the lifecycle and this flush is deliberately
    // not performed.
    let spans = RecordingSpanExporter::default();
    let metrics = InMemoryMetricExporter::default();
    let config = Telemetry::otlp("http://localhost:4318")
        .to_config()
        .expect("the test endpoint must be valid");
    let guard = install_with_exporters(&config, spans.clone(), metrics);

    let server = MockServer::start(vec![Round::text("last words").with_usage(7, 3)]).await;
    let provider =
        acton_ai::config::from_str(&mock_llm::provider_toml("mock", &server, "mock-model"))
            .expect("the provider config must parse");
    let ai = ActonAI::builder()
        .app_name("telemetry-shutdown")
        .apply_config(provider)
        .expect("the config must apply")
        .telemetry_guard(guard)
        .launch()
        .await
        .expect("launching the runtime must succeed");

    ai.prompt("hi").collect().await.expect("the round runs");

    // Deliberately no flush here: `shutdown()` is what has to do it.
    ai.shutdown().await.expect("shutdown must succeed");

    let exported = spans.spans();
    let named = |name: &str| exported.iter().filter(|span| span.name == name).count();
    assert_eq!(
        named("acton_ai.turn"),
        1,
        "the turn span must be exported by shutdown, not lost in the batch"
    );
    assert_eq!(named("acton_ai.llm_round"), 1);
}

// =============================================================================
// 4b. Failover composition
// =============================================================================

#[tokio::test]
async fn a_failed_over_turn_yields_a_round_span_per_provider_and_counts_the_hop() {
    // Two dispatches inside one turn, to two different providers. If the
    // spans collapsed onto the configured provider, an operator reading a
    // trace would never learn the request moved.
    let harness = Harness::install();
    let primary = MockServer::start(vec![Round::server_error()]).await;
    let backup = MockServer::start(vec![Round::text("served").with_usage(11, 22)]).await;
    let toml = format!(
        "default_provider = \"primary\"\n\
         {}failover = [\"backup\"]\n{}",
        mock_llm::provider_toml("primary", &primary, "primary-model"),
        mock_llm::provider_toml("backup", &backup, "backup-model"),
    );
    let config = acton_ai::config::from_str(&toml).expect("the config must parse");
    let ai = ActonAI::builder()
        .app_name("telemetry-failover")
        .apply_config(config)
        .expect("the config must apply")
        .telemetry_from_globals()
        .launch()
        .await
        .expect("launching the runtime must succeed");

    ai.prompt("who is up?")
        .collect()
        .await
        .expect("the backup serves the round");
    flush_broadcasts(&ai).await;
    harness.flush();

    let mut dispatched: Vec<String> = harness
        .spans_named("acton_ai.llm_round")
        .iter()
        .filter_map(|span| attribute(span, "acton_ai.provider"))
        .collect();
    dispatched.sort();
    assert_eq!(
        dispatched,
        vec!["backup".to_string(), "primary".to_string()],
        "each dispatch gets its own round span, labelled with where it went",
    );

    // Each round span carries the model that served it, not the entry
    // provider's configured one.
    let models: Vec<String> = harness
        .spans_named("acton_ai.llm_round")
        .iter()
        .filter_map(|span| attribute(span, "acton_ai.model"))
        .collect();
    assert!(
        models.contains(&"backup-model".to_string()),
        "the serving model must appear: {models:?}",
    );

    assert_eq!(
        harness.counter_total("acton_ai.failover.events", Some(("kind", "failed_over"))),
        1,
        "the hop itself is a counted event",
    );
    assert_eq!(
        harness.counter_total("acton_ai.failover.events", Some(("provider", "primary"))),
        1,
        "the counter names the provider that gave up the round",
    );
}

// =============================================================================
// 5. Configuration
// =============================================================================

#[tokio::test]
async fn a_telemetry_section_with_a_bad_endpoint_fails_the_launch() {
    let server = MockServer::start(vec![]).await;
    let toml = format!(
        "{}[telemetry]\notlp_endpoint = \"grpc://localhost:4317\"\n",
        mock_llm::provider_toml("claude", &server, "sonnet-mock")
    );
    let config = acton_ai::config::from_str(&toml).expect("the section must parse");

    let error = ActonAI::builder()
        .app_name("telemetry-bad-endpoint")
        .apply_config(config)
        .expect("the config must apply")
        .launch()
        .await
        .expect_err("a non-HTTP endpoint must not launch");

    assert!(
        error.to_string().contains("telemetry.otlp_endpoint"),
        "the error names the offending key: {error}"
    );
}

#[tokio::test]
async fn a_runtime_without_telemetry_configured_reports_none() {
    let server = MockServer::start(vec![Round::text("hi")]).await;
    let ai = runtime_pointed_at(&server, "telemetry-absent").await;

    assert!(
        format!("{ai:?}").contains("telemetry: false"),
        "an unconfigured runtime must not claim telemetry"
    );
}
