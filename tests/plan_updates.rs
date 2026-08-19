//! End-to-end tests for the `update_plan` tool and the `PlanUpdated`
//! broadcast it produces.
//!
//! These drive the real stack — facade, prompt loop, provider actor, OpenAI
//! client, built-in tool, broker — against the scripted server in
//! [`mock_llm`]. The subscriber is a test-owned actor rather than a callback,
//! which is the point twice over: it proves the plan is published for *anyone*
//! on the broker, not just for machinery the crate happens to own, and its
//! FIFO inbox is the barrier that makes the assertions deterministic.
//!
//! Nothing here sleeps. Every wait is either `collect()` returning or an
//! `ask` — `FlushBroadcasts` to prove the broker handed the message to every
//! subscriber, then a question to the spy, which cannot be answered before the
//! messages already in its inbox have been handled.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use serde_json::json;
use std::time::Duration;

/// An unanswered `ask` fails after 30 seconds by default. A missing reply here
/// would be a bug, not slow work, so these tests refuse to wait that long.
const REPLY_DEADLINE: Duration = Duration::from_secs(5);

// =============================================================================
// The subscriber
// =============================================================================

/// Collects every [`PlanUpdated`] on the broker and answers for them.
#[acton_actor]
struct PlanSpy {
    /// Every plan seen, in the order the broker delivered it.
    seen: Vec<PlanUpdated>,
}

/// Asks the spy what it has seen. Also the test's barrier.
#[acton_message]
struct GetSeenPlans;

impl Request for GetSeenPlans {
    type Response = SeenPlans;
}

/// The spy's answer.
#[acton_message]
struct SeenPlans {
    /// Every plan update the spy received, in order.
    updates: Vec<PlanUpdated>,
}

/// Spawns a spy subscribed to [`PlanUpdated`].
async fn spawn_plan_spy(runtime: &mut ActorRuntime) -> ActorHandle {
    let mut builder = runtime.new_actor_with_name::<PlanSpy>("plan_spy".to_string());

    builder.mutate_on::<PlanUpdated>(|actor, envelope| {
        actor.model.seen.push(envelope.message().clone());
        Reply::ready()
    });

    builder.act_on::<GetSeenPlans>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let updates = actor.model.seen.clone();
        Reply::pending(async move {
            reply.send(SeenPlans { updates }).await;
        })
    });

    // On the builder, before `start()`: a subscription registered afterwards
    // is silently ignored.
    builder.handle().subscribe::<PlanUpdated>().await;

    builder.start().await
}

// =============================================================================
// Harness
// =============================================================================

/// A runtime pointed at `server` with only the `update_plan` built-in enabled.
async fn runtime_with_update_plan(server: &MockServer, app_name: &str) -> ActonAI {
    ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .with_builtin_tools(&["update_plan"])
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

/// Everything the spy has seen, once the broker has finished delivering.
///
/// `FlushBroadcasts` cannot be answered until every earlier broadcast is
/// sitting in every subscriber's inbox, and the spy's own `ask` queues behind
/// whatever landed there — two barriers, no sleeping.
async fn seen_plans(ai: &ActonAI, spy: &ActorHandle) -> Vec<PlanUpdated> {
    ai.runtime()
        .broker()
        .ask(FlushBroadcasts)
        .await
        .expect("the broker must answer a flush");

    spy.ask_with_timeout(GetSeenPlans, REPLY_DEADLINE)
        .await
        .expect("the spy must answer")
        .updates
}

/// The steps of one update, as `(title, status)` pairs.
fn steps_of(update: &PlanUpdated) -> Vec<(&str, PlanStepStatus)> {
    update
        .plan
        .steps()
        .iter()
        .map(|step| (step.title().as_str(), step.status()))
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

/// The whole point: a plan the model records reaches an ordinary subscriber.
#[tokio::test]
async fn a_recorded_plan_reaches_an_ordinary_subscriber() {
    let server = MockServer::start(vec![
        Round::tool_call(
            "call_1",
            "update_plan",
            json!({
                "steps": [
                    {"title": "read the failing test", "status": "completed"},
                    {"title": "fix the parser", "status": "in_progress"},
                    {"title": "run the suite", "status": "pending"},
                ],
                "note": "parser first",
            }),
        ),
        Round::text("Plan recorded; working on the parser."),
    ])
    .await;
    let mut ai = runtime_with_update_plan(&server, "plan-updates-basic").await;
    let spy = spawn_plan_spy(ai.runtime_mut()).await;

    let response = ai
        .prompt("Fix the parser bug.")
        .collect()
        .await
        .expect("the prompt must complete");

    assert_eq!(response.text, "Plan recorded; working on the parser.");
    assert_eq!(
        server.request_count(),
        2,
        "one round to record the plan, one to answer"
    );

    let updates = seen_plans(&ai, &spy).await;
    assert_eq!(updates.len(), 1, "one call, one broadcast");

    let update = &updates[0];
    assert_eq!(
        steps_of(update),
        vec![
            ("read the failing test", PlanStepStatus::Completed),
            ("fix the parser", PlanStepStatus::InProgress),
            ("run the suite", PlanStepStatus::Pending),
        ]
    );
    assert_eq!(
        update.plan.note().map(PlanNote::as_str),
        Some("parser first")
    );
    assert_eq!(update.plan.completed_count(), 1);
    assert!(!update.plan.is_complete());

    // The message has to say which call it came from, or a subscriber cannot
    // line it up with the tool events around it.
    assert_eq!(update.tool_call_id, "call_1");
    assert!(
        !update.turn_id.to_string().is_empty(),
        "a plan belongs to a turn"
    );

    // The turn's owned state surfaces on the response too, for callers who
    // want the plan rather than the stream of updates.
    assert_eq!(
        response.plan.as_ref(),
        Some(&update.plan),
        "the response must carry the plan the turn ended with"
    );

    ai.shutdown().await.expect("clean shutdown");
}

/// The tool is offered to the model with the schema the parser enforces.
#[tokio::test]
async fn the_tool_is_offered_with_the_statuses_it_accepts() {
    let server = MockServer::start(vec![Round::text("Nothing to plan.")]).await;
    let ai = runtime_with_update_plan(&server, "plan-updates-schema").await;

    ai.prompt("Say hello.")
        .collect()
        .await
        .expect("the prompt must complete");

    let requests = server.requests();
    let offered =
        tool_named(&requests[0], "update_plan").expect("the built-in must be offered to the model");

    let schema = &offered["function"]["parameters"];
    assert_eq!(schema["required"], json!(["steps"]), "{schema:#}");
    assert_eq!(
        schema["properties"]["steps"]["items"]["properties"]["status"]["enum"],
        json!(["pending", "in_progress", "completed"]),
        "{schema:#}"
    );
    assert!(
        !contains_ref(schema),
        "providers differ on $ref support, so the schema must be self-contained: {schema:#}"
    );

    ai.shutdown().await.expect("clean shutdown");
}

/// A plan that breaks a rule is refused: nothing is published, and the model
/// is told what to fix rather than being left to guess.
#[tokio::test]
async fn a_rejected_plan_broadcasts_nothing_and_tells_the_model_why() {
    let server = MockServer::start(vec![
        Round::tool_call(
            "call_1",
            "update_plan",
            json!({
                "steps": [
                    {"title": "write the fix", "status": "in_progress"},
                    {"title": "write the test", "status": "in_progress"},
                ],
            }),
        ),
        Round::text("Sorry, corrected."),
    ])
    .await;
    let mut ai = runtime_with_update_plan(&server, "plan-updates-rejected").await;
    let spy = spawn_plan_spy(ai.runtime_mut()).await;

    let response = ai
        .prompt("Fix and test.")
        .collect()
        .await
        .expect("a refused plan is a normal tool result, not a failed turn");

    assert!(
        seen_plans(&ai, &spy).await.is_empty(),
        "a plan that never validated must never be published"
    );
    assert!(
        response.plan.is_none(),
        "a plan that never validated must never become the turn's state"
    );

    // The refusal has to reach the model, or it cannot correct itself.
    let requests = server.requests();
    let follow_up = serde_json::to_string(&requests[1]).expect("the request must serialize");
    assert!(
        follow_up.contains("in_progress"),
        "the validation message must be fed back as the tool result: {follow_up}"
    );

    ai.shutdown().await.expect("clean shutdown");
}

/// Two updates in one turn arrive in the order the model made them, both
/// carrying the same turn.
#[tokio::test]
async fn successive_updates_arrive_in_order() {
    let server = MockServer::start(vec![
        Round::tool_call(
            "call_1",
            "update_plan",
            json!({
                "steps": [
                    {"title": "reproduce", "status": "in_progress"},
                    {"title": "patch", "status": "pending"},
                ],
            }),
        ),
        Round::tool_call(
            "call_2",
            "update_plan",
            json!({
                "steps": [
                    {"title": "reproduce", "status": "completed"},
                    {"title": "patch", "status": "completed"},
                ],
                "note": "done",
            }),
        ),
        Round::text("Both steps are done."),
    ])
    .await;
    let mut ai = runtime_with_update_plan(&server, "plan-updates-progression").await;
    let spy = spawn_plan_spy(ai.runtime_mut()).await;

    let response = ai
        .prompt("Reproduce and patch.")
        .collect()
        .await
        .expect("the prompt must complete");

    let updates = seen_plans(&ai, &spy).await;
    assert_eq!(updates.len(), 2, "two calls, two broadcasts");

    assert_eq!(
        steps_of(&updates[0]),
        vec![
            ("reproduce", PlanStepStatus::InProgress),
            ("patch", PlanStepStatus::Pending),
        ],
        "the first update is the plan as it was, not as it ended up"
    );
    assert!(!updates[0].plan.is_complete());

    assert_eq!(
        steps_of(&updates[1]),
        vec![
            ("reproduce", PlanStepStatus::Completed),
            ("patch", PlanStepStatus::Completed),
        ]
    );
    assert!(updates[1].plan.is_complete());

    assert_eq!(updates[0].tool_call_id, "call_1");
    assert_eq!(updates[1].tool_call_id, "call_2");
    assert_eq!(
        updates[0].turn_id, updates[1].turn_id,
        "both updates belong to the same turn"
    );
    assert_ne!(
        updates[0].correlation_id, updates[1].correlation_id,
        "but to different rounds of it"
    );

    // The turn-owned state carried across the rounds: the second update
    // replaced the first, and the turn ended holding the replacement.
    assert_eq!(
        response.plan.as_ref(),
        Some(&updates[1].plan),
        "the response must carry the last plan recorded, not the first"
    );

    ai.shutdown().await.expect("clean shutdown");
}

/// A refused update does not disturb the plan the turn already holds: the
/// state carries across rounds, through the failed round, to the response.
#[tokio::test]
async fn a_refused_update_leaves_the_recorded_plan_standing() {
    let server = MockServer::start(vec![
        Round::tool_call(
            "call_1",
            "update_plan",
            json!({
                "steps": [
                    {"title": "reproduce", "status": "in_progress"},
                    {"title": "patch", "status": "pending"},
                ],
            }),
        ),
        // A single-step plan: the tool's own anti-pattern, refused with a
        // corrective sentence rather than recorded.
        Round::tool_call(
            "call_2",
            "update_plan",
            json!({
                "steps": [{"title": "do everything", "status": "in_progress"}],
            }),
        ),
        Round::text("Continuing with the original plan."),
    ])
    .await;
    let mut ai = runtime_with_update_plan(&server, "plan-updates-refusal-keeps-state").await;
    let spy = spawn_plan_spy(ai.runtime_mut()).await;

    let response = ai
        .prompt("Reproduce and patch.")
        .collect()
        .await
        .expect("the prompt must complete");

    let updates = seen_plans(&ai, &spy).await;
    assert_eq!(updates.len(), 1, "only the valid update is published");

    assert_eq!(
        response.plan.as_ref(),
        Some(&updates[0].plan),
        "the refused update must not displace the plan already recorded"
    );

    // And the refusal itself was corrective: the model was told to split the
    // work or skip planning, in the tool result of the third request.
    let requests = server.requests();
    let follow_up = serde_json::to_string(&requests[2]).expect("the request must serialize");
    assert!(
        follow_up.contains("at least two"),
        "the single-step refusal must reach the model: {follow_up}"
    );
    assert!(follow_up.contains("skip planning"), "{follow_up}");

    ai.shutdown().await.expect("clean shutdown");
}

/// Another tool running in the same turn publishes nothing, even though the
/// round loop inspects every result.
#[tokio::test]
async fn a_different_tool_publishes_no_plan() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "impostor", json!({"plan": {"steps": []}})),
        Round::text("Done."),
    ])
    .await;
    // No built-ins here at all: the round loop still inspects every result,
    // and must still find no plan in this one.
    let mut ai = runtime_pointed_at(&server, "plan-updates-other-tool").await;
    let spy = spawn_plan_spy(ai.runtime_mut()).await;

    // Returns something a careless reader could mistake for a recorded plan.
    let recorded_plan = json!({
        "plan": {
            "steps": [
                {"title": "not really a plan", "status": "pending"},
                {"title": "still not one", "status": "pending"},
            ],
            "note": null,
        }
    });
    ai.prompt("Call the impostor.")
        .with_tool(
            ToolDefinition {
                name: "impostor".to_string(),
                description: "Returns something plan-shaped.".to_string(),
                input_schema: json!({"type": "object", "properties": {}}),
            },
            move |_args| {
                let value = recorded_plan.clone();
                async move { Ok(value) }
            },
        )
        .collect()
        .await
        .expect("the prompt must complete");

    assert!(
        seen_plans(&ai, &spy).await.is_empty(),
        "only update_plan publishes a plan; the tool's name is what decides"
    );

    ai.shutdown().await.expect("clean shutdown");
}
