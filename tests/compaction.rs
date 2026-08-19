//! End-to-end tests for auto-compaction inside the prompt loop.
//!
//! Compaction is the one feature that rewrites what the model sees, so the
//! only assertions worth making are against **what actually went out on the
//! wire**. These drive the real stack — facade, prompt loop, provider actor,
//! OpenAI client, broker — against the scripted server in [`mock_llm`], and
//! read both the summarization request and the compacted history back out of
//! the request bodies the server received.
//!
//! # Determinism
//!
//! Nothing sleeps. A prompt blocks on the collector, so by the time
//! `collect()` returns every round it drove has already been served — the
//! summarization round included, since it is served by the same scripted
//! server in the same order; `broker.ask(FlushBroadcasts)` proves every
//! broadcast is sitting in each subscriber's inbox, and an `ask` to the
//! subscriber afterwards proves it processed that broadcast first, because
//! mailboxes are FIFO.
//!
//! # The arithmetic
//!
//! The char-ratio estimator prices a message at `chars * 0.25 + 4`. Every
//! tool result here is [`BIG_RESULT_CHARS`] = 8000 chars ≈ 2000 tokens, so
//! against [`WINDOW_TOKENS`] = 4000 (trigger at 80% = 3200) the history is
//! under the trigger with one tool exchange (~2050) and over it with two
//! (~4100). Compaction therefore fires exactly once, at the top of the third
//! iteration, where the elided span holds a whole 8000-char exchange and the
//! summary that replaces it cannot fail to be smaller.

mod mock_llm;

use acton_ai::memory::{
    CompactionConfig, ContextWindow, ContextWindowConfig, KeepRecentTurns, TruncationStrategy,
    COMPACTION_NOTICE, COMPACTION_PROMPT,
};
use acton_ai::prelude::*;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use serde_json::{json, Value};

/// A tool result big enough that two of them outgrow the test window.
const BIG_RESULT_CHARS: usize = 8_000;

/// The window every compacting test here runs against. See the module docs
/// for why this number and [`BIG_RESULT_CHARS`] are what they are.
const WINDOW_TOKENS: usize = 4_000;

/// What the scripted provider "writes" when asked to summarize.
const SUMMARY: &str = "SUMMARY: the user asked for a dump; two large blobs were \
                       retrieved and their contents were filler.";

/// The window every test here runs against: small, and with nothing held back
/// for the response, so the arithmetic in the assertions is the arithmetic the
/// loop does.
fn tight_window(max_tokens: usize) -> ContextWindow {
    ContextWindow::new(ContextWindowConfig {
        max_tokens,
        truncation_strategy: TruncationStrategy::KeepRecent,
        reserved_for_response: 0,
        tokens_per_char: 0.25,
    })
}

/// The policy under test: default threshold, keep only the last exchange.
fn keep_one() -> CompactionConfig {
    CompactionConfig::default().with_keep_recent_turns(KeepRecentTurns::new(1).unwrap())
}

/// The script the compacting tests share: two tool rounds, then the
/// summarization the third iteration will ask for, then the answer.
fn compacting_script() -> Vec<Round> {
    vec![
        Round::tool_call("call-0", "dump", json!({"which": 0})),
        Round::tool_call("call-1", "dump", json!({"which": 1})),
        Round::text(SUMMARY),
        Round::text("done"),
    ]
}

/// A script of `rounds` tool calls followed by a final prose answer, for the
/// tests where nothing compacts and no summarization round is served.
fn tool_loop_script(rounds: usize) -> Vec<Round> {
    let mut script: Vec<Round> = (0..rounds)
        .map(|i| Round::tool_call(&format!("call-{i}"), "dump", json!({"which": i})))
        .collect();
    script.push(Round::text("done"));
    script
}

/// The `messages` array of one request the server received.
fn messages_of(request: &Value) -> &Vec<Value> {
    request["messages"]
        .as_array()
        .expect("every request must carry a messages array")
}

/// Whether any message in this request carries the compaction notice.
fn carries_notice(request: &Value) -> bool {
    messages_of(request).iter().any(|message| {
        message["content"]
            .as_str()
            .is_some_and(|text| text.contains(COMPACTION_NOTICE))
    })
}

/// Whether this request is the summarization request itself.
fn is_summarization(request: &Value) -> bool {
    messages_of(request).iter().any(|message| {
        message["content"]
            .as_str()
            .is_some_and(|text| text.contains(COMPACTION_PROMPT))
    })
}

/// Runs a prompt whose only tool returns [`BIG_RESULT_CHARS`] of filler.
async fn run_dump_loop(ai: &ActonAI) -> Result<CollectedResponse, ActonAIError> {
    ai.prompt("dump everything")
        .system("You are terse.")
        .tool(
            "dump",
            "Returns a large blob of text",
            json!({
                "type": "object",
                "properties": {"which": {"type": "integer"}},
                "required": ["which"],
            }),
            |args: Value| async move {
                let which = args["which"].as_i64().unwrap_or_default();
                Ok(json!({ "blob": format!("{which}:{}", "z".repeat(BIG_RESULT_CHARS)) }))
            },
        )
        .max_tool_rounds(12)
        .collect()
        .await
}

/// A compacting runtime pointed at `server`.
async fn compacting_runtime(server: &MockServer, app_name: &str) -> ActonAI {
    ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .context_window(tight_window(WINDOW_TOKENS))
        .compaction(keep_one())
        .launch()
        .await
        .expect("launch")
}

// =============================================================================
// The loop
// =============================================================================

#[tokio::test]
async fn a_tool_loop_that_outgrows_its_window_is_compacted_on_the_wire() {
    let server = MockServer::start(compacting_script()).await;
    let ai = compacting_runtime(&server, "compaction-wire").await;

    let response = run_dump_loop(&ai).await.expect("the loop must finish");

    assert_eq!(
        server.request_count(),
        4,
        "two tool rounds, one summarization, one answer",
    );
    let requests = server.requests();

    // The third request on the wire is the summarization: it carries the
    // fixed prompt, a transcript of what is being elided, and no tools — the
    // model is being asked to write, not to work.
    assert!(
        !is_summarization(&requests[0]) && !is_summarization(&requests[1]),
        "nothing to compact yet",
    );
    assert!(
        is_summarization(&requests[2]),
        "the third request must be the summarization",
    );
    assert!(
        requests[2]
            .get("tools")
            .is_none_or(|tools| tools.as_array().is_none_or(Vec::is_empty)),
        "a summarization request offers no tools",
    );
    assert!(
        !carries_notice(&requests[2]),
        "the summarization sees the raw transcript, not a notice about itself",
    );

    // The fourth request is the turn continuing on the compacted history:
    // the notice and the provider-written summary in place of the elided
    // span, the last exchange verbatim.
    assert!(
        carries_notice(&requests[3]),
        "the compacted history is marked"
    );
    assert!(
        messages_of(&requests[3])
            .iter()
            .any(|m| m["content"].as_str().is_some_and(|t| t.contains(SUMMARY))),
        "the summary the provider wrote is what the model sees",
    );
    assert!(
        messages_of(&requests[3])
            .iter()
            .any(|m| m["role"] == "tool" && m["tool_call_id"] == "call-1"),
        "the kept exchange survives verbatim",
    );
    assert!(
        messages_of(&requests[3]).len() < messages_of(&requests[1]).len() + 2,
        "the point of compaction: the history stopped growing",
    );

    // Compaction rewrites the history and nothing else. The tool list is
    // rebuilt from the same definitions every round, so a compacted request
    // still has to offer the model exactly the tool it has been calling —
    // with its schema inlined, since the providers reject `$ref`.
    for request in [&requests[0], &requests[1], &requests[3]] {
        let dump = tool_named(request, "dump").expect("every working round must offer the tool");
        assert!(
            !contains_ref(&dump["function"]["parameters"]),
            "a tool schema must go out inlined",
        );
    }

    // And the caller was told: one record, carrying the summary and the
    // measured effect, which is what persistence stores.
    assert!(response.was_compacted());
    assert_eq!(response.compactions.len(), 1);
    let record = &response.compactions[0];
    assert_eq!(record.summary, SUMMARY);
    assert!(record.outcome.tokens_after < record.outcome.tokens_before);
    assert!(record.outcome.messages_elided > 0);
    assert!(
        record.as_message().content.starts_with(COMPACTION_NOTICE),
        "the persistable message is the marked one",
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn a_compacted_request_still_answers_every_tool_call_it_carries() {
    // The invariant a provider actually enforces: a `tool_use` block with no
    // matching `tool_result` poisons the conversation for every later turn.
    // Compaction splits the history, so it is exactly the operation that
    // could break this.
    let server = MockServer::start(compacting_script()).await;
    let ai = compacting_runtime(&server, "compaction-pairing").await;

    run_dump_loop(&ai).await.expect("the loop must finish");

    for (index, request) in server.requests().iter().enumerate() {
        let messages = messages_of(request);

        let issued: Vec<&str> = messages
            .iter()
            .filter_map(|m| m.get("tool_calls")?.as_array())
            .flatten()
            .filter_map(|call| call["id"].as_str())
            .collect();
        let answered: Vec<&str> = messages
            .iter()
            .filter(|m| m["role"] == "tool")
            .filter_map(|m| m["tool_call_id"].as_str())
            .collect();

        for id in &answered {
            assert!(
                issued.contains(id),
                "request #{index} answers a call it never issued: {id}",
            );
        }
        for id in &issued {
            assert!(
                answered.contains(id),
                "request #{index} issues an unanswered call: {id}",
            );
        }
    }

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn a_runtime_without_compaction_sends_the_whole_history() {
    // The control. Same tool, no policy: the history grows round on round,
    // no summarization request ever goes out, and the response carries no
    // records — the disabled path is the behavior this crate always had.
    let server = MockServer::start(tool_loop_script(4)).await;
    // A stock runtime: the default window, and no policy asking for anything
    // to be done about it.
    let ai = runtime_pointed_at(&server, "compaction-off").await;

    let response = run_dump_loop(&ai).await.expect("the loop must finish");

    let requests = server.requests();
    assert!(
        !requests.iter().any(carries_notice),
        "nothing should be compacted without a policy",
    );
    assert!(
        !requests.iter().any(is_summarization),
        "no summarization request should ever go out without a policy",
    );
    assert!(!response.was_compacted());
    assert!(response.compactions.is_empty());

    let first = messages_of(&requests[0]).len();
    let last = messages_of(requests.last().expect("at least one request")).len();
    assert!(
        last > first + 3,
        "the unbounded history should have grown: {first} then {last}",
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn a_history_that_fits_is_never_compacted() {
    // A policy in force is not a licence to rewrite a history that was never
    // in danger — or to spend the caller's money summarizing it. The window
    // here is far larger than the traffic, and the script has no
    // summarization round for the loop to consume: asking for one anyway
    // would derail the scripted rounds and fail the count below.
    let server = MockServer::start(tool_loop_script(2)).await;
    let ai = ActonAI::builder()
        .app_name("compaction-idle")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .context_window(tight_window(1_000_000))
        .compaction(CompactionConfig::default())
        .launch()
        .await
        .expect("launch");

    let response = run_dump_loop(&ai).await.expect("the loop must finish");

    assert_eq!(server.request_count(), 3, "two tool rounds and the answer");
    assert!(
        !server.requests().iter().any(carries_notice),
        "a history inside its budget must go out untouched",
    );
    assert!(response.compactions.is_empty());

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn compaction_without_a_context_window_is_inert() {
    // `without_context_window` is an explicit choice to ship everything. A
    // policy alone has no budget to measure against, so it must do nothing
    // rather than guess at one.
    let server = MockServer::start(tool_loop_script(3)).await;
    let ai = ActonAI::builder()
        .app_name("compaction-no-window")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .without_context_window()
        .compaction(CompactionConfig::default())
        .launch()
        .await
        .expect("launch");

    let response = run_dump_loop(&ai).await.expect("the loop must finish");

    assert!(
        !server.requests().iter().any(is_summarization),
        "with no window there is no budget, so nothing to compact against",
    );
    assert!(response.compactions.is_empty());

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn a_failed_summarization_never_costs_the_turn() {
    // The provider that writes the summaries is the provider that is already
    // under pressure, so the summarization can fail. When it does, the turn
    // must proceed with its full history and take its chances — losing the
    // turn to an optimization would be strictly worse than never optimizing —
    // and the gate must not pay for another attempt every round.
    let script = vec![
        Round::tool_call("call-0", "dump", json!({"which": 0})),
        Round::tool_call("call-1", "dump", json!({"which": 1})),
        // The third iteration's summarization request lands here.
        Round::server_error(),
        Round::text("done"),
    ];
    let server = MockServer::start(script).await;
    let ai = compacting_runtime(&server, "compaction-failure").await;

    let response = run_dump_loop(&ai).await.expect("the turn must survive");

    assert_eq!(response.text, "done");
    assert!(!response.was_compacted());
    assert!(
        !server.requests().iter().any(carries_notice),
        "a failed summarization must leave the history untouched",
    );
    // Exactly one summarization attempt: the failure latched the gate, so
    // the fourth request is the turn itself, not another paid attempt.
    assert_eq!(
        server
            .requests()
            .iter()
            .filter(|request| is_summarization(request))
            .count(),
        1,
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// The lifecycle event
// =============================================================================

/// Records every [`TurnLifecycle`] broadcast so a test can prove the event
/// reached subscribers rather than merely that the loop meant to send it.
#[acton_actor]
struct LifecycleSpy {
    compactions: Vec<(u64, u64, u64)>,
}

#[acton_message]
struct GetCompactions;

impl Request for GetCompactions {
    type Response = SeenCompactions;
}

#[acton_message]
struct SeenCompactions {
    events: Vec<(u64, u64, u64)>,
}

async fn spawn_lifecycle_spy(runtime: &mut ActorRuntime) -> ActorHandle {
    let mut builder = runtime.new_actor_with_name::<LifecycleSpy>("lifecycle_spy".to_string());

    builder.mutate_on::<TurnLifecycle>(|actor, envelope| {
        if let TurnLifecycle::ContextCompacted {
            tokens_before,
            tokens_after,
            messages_elided,
            ..
        } = envelope.message()
        {
            actor
                .model
                .compactions
                .push((*tokens_before, *tokens_after, *messages_elided));
        }
        Reply::ready()
    });

    builder.act_on::<GetCompactions>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let events = actor.model.compactions.clone();
        Reply::pending(async move {
            reply.send(SeenCompactions { events }).await;
        })
    });

    // On the builder, before start: a subscription registered afterwards is
    // silently ignored.
    builder.handle().subscribe::<TurnLifecycle>().await;

    builder.start().await
}

#[tokio::test]
async fn every_compaction_announces_itself_on_the_lifecycle_channel() {
    let server = MockServer::start(compacting_script()).await;
    let mut ai = compacting_runtime(&server, "compaction-lifecycle").await;

    let spy = spawn_lifecycle_spy(ai.runtime_mut()).await;

    run_dump_loop(&ai).await.expect("the loop must finish");

    ai.runtime()
        .broker()
        .ask(FlushBroadcasts)
        .await
        .expect("the broker must answer a flush");

    let seen = spy.ask(GetCompactions).await.expect("the spy must answer");

    assert!(
        !seen.events.is_empty(),
        "a compaction that nothing can observe is a compaction nobody can debug",
    );
    for (before, after, elided) in &seen.events {
        assert!(after < before, "reported {after} tokens after {before}");
        assert!(*elided > 0, "an event must name what it removed");
    }

    ai.shutdown().await.expect("clean shutdown");
}
