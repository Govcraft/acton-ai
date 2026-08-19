//! End-to-end tests for the tool-approval gate.
//!
//! These drive the real stack — facade, prompt loop, provider actor, HTTP
//! client — against the scripted server in [`mock_llm`], so what they assert
//! is what actually travelled the wire: whether the tool ran, and what the
//! model was told about it in the following round.
//!
//! # Determinism
//!
//! Nothing sleeps. The gate is synchronous with respect to the loop — a round
//! cannot be sent until the previous round's tool results exist — so a
//! completed prompt is itself the barrier.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, tool_named, MockServer, Round};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A tool the scripted rounds can call.
fn tool_definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: "Echoes its argument back.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }),
    }
}

/// Every `tool`-role message body in one request, in order.
///
/// This is the only honest way to check what the model was told: the loop's
/// own bookkeeping could agree with itself and still send nothing.
fn tool_results(request: &Value) -> Vec<String> {
    request["messages"]
        .as_array()
        .expect("every request carries messages")
        .iter()
        .filter(|message| message["role"] == "tool")
        .map(|message| message["content"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// Launches a runtime pointed at `server` with `policy` in force.
async fn runtime_with_policy(server: &MockServer, app_name: &str, policy: ToolPolicy) -> ActonAI {
    ActonAI::builder()
        .app_name(app_name)
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .tool_policy(policy)
        .launch()
        .await
        .expect("launching the runtime must succeed")
}

// =============================================================================
// 1. A denial is fed back to the model and the turn continues
// =============================================================================

#[tokio::test]
async fn a_denied_call_never_runs_and_the_turn_carries_on() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "bash", json!({"command": "rm -rf /"})),
        Round::text("understood, I will not do that"),
    ])
    .await;
    let ai = runtime_with_policy(&server, "policy-deny", ToolPolicy::new().deny(["bash"])).await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    let response = ai
        .prompt("clean up")
        .with_tool(tool_definition("bash"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .collect()
        .await
        .expect("a denial must not fail the turn");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        0,
        "a denied tool must never be executed"
    );
    assert_eq!(
        response.text, "understood, I will not do that",
        "the turn must continue to its scripted end"
    );
    assert_eq!(
        server.request_count(),
        2,
        "the loop must send a second round carrying the denial"
    );

    // The gate is enforcement, not negotiation: the model is never told a
    // tool is restricted, so the denied tool must still be advertised exactly
    // as it was before any policy existed.
    let first = server.requests().remove(0);
    let offered = tool_named(&first, "bash").expect("a denied tool is still offered to the model");
    assert!(
        !contains_ref(&offered["function"]["parameters"]),
        "tool schemas must be self-contained on the wire: {offered}"
    );

    let second = server.requests().remove(1);
    let results = tool_results(&second);
    assert_eq!(results.len(), 1, "the denied call still needs a result");
    assert!(
        results[0].contains("denied by policy"),
        "the model must be told why: {}",
        results[0]
    );
    assert!(
        results[0].contains("Do not retry"),
        "the model must be told not to retry: {}",
        results[0]
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 2. Per-turn caps
// =============================================================================

#[tokio::test]
async fn a_per_turn_cap_admits_exactly_its_limit_and_then_refuses() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "one"})),
        Round::tool_call("call_2", "echo", json!({"value": "two"})),
        Round::text("stopped"),
    ])
    .await;
    let ai = runtime_with_policy(
        &server,
        "policy-cap",
        ToolPolicy::new().cap_per_turn("echo", 1),
    )
    .await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    let response = ai
        .prompt("echo twice")
        .with_tool(tool_definition("echo"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .collect()
        .await
        .expect("the capped turn must still complete");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        1,
        "a cap of 1 must admit exactly one call"
    );
    assert_eq!(response.text, "stopped");

    let third = server.requests().remove(2);
    let results = tool_results(&third);
    let refusal = results.last().expect("the refused call needs a result");
    assert!(
        refusal.contains("at most 1 time(s) per turn"),
        "the refusal must name the cap: {refusal}"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 3. Allowlists, including MCP-shaped names
// =============================================================================

#[tokio::test]
async fn an_allowlist_refuses_an_mcp_tool_from_a_server_it_does_not_name() {
    let server = MockServer::start(vec![
        Round::tool_call(
            "call_1",
            "mcp__fs__read_file",
            json!({"path": "/etc/shadow"}),
        ),
        Round::text("blocked"),
    ])
    .await;
    // A whole-server pattern, which is the shape an operator actually writes.
    let ai = runtime_with_policy(
        &server,
        "policy-allow-mcp",
        ToolPolicy::new().allow(["mcp__docs__*"]),
    )
    .await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    ai.prompt("read it")
        .with_tool(tool_definition("mcp__fs__read_file"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        0,
        "a tool outside the allowlist must never run"
    );

    let second = server.requests().remove(1);
    let results = tool_results(&second);
    assert!(
        results[0].contains("not on this agent's allowlist"),
        "the refusal must say what happened: {}",
        results[0]
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn an_allowlist_admits_the_server_it_does_name() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "mcp__docs__search", json!({"q": "rust"})),
        Round::text("found it"),
    ])
    .await;
    let ai = runtime_with_policy(
        &server,
        "policy-allow-mcp-ok",
        ToolPolicy::new().allow(["mcp__docs__*"]),
    )
    .await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    ai.prompt("search")
        .with_tool(tool_definition("mcp__docs__search"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        1,
        "a prefix pattern must admit its own server's tools"
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 4. The approval hook
// =============================================================================

#[tokio::test]
async fn the_hook_can_rewrite_the_arguments_that_actually_run() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "original"})),
        Round::text("done"),
    ])
    .await;

    let ai = ActonAI::builder()
        .app_name("policy-hook-rewrite")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .on_tool_approval(|_invocation| async move {
            ApprovalDecision::approve_with(json!({"value": "rewritten"}))
        })
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let seen: Arc<std::sync::Mutex<Vec<Value>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorder = seen.clone();

    ai.prompt("echo")
        .with_tool(tool_definition("echo"), move |args| {
            let recorder = recorder.clone();
            async move {
                recorder.lock().expect("not poisoned").push(args.clone());
                Ok(args)
            }
        })
        .collect()
        .await
        .expect("the turn must complete");

    let observed = seen.lock().expect("not poisoned").clone();
    assert_eq!(
        observed,
        vec![json!({"value": "rewritten"})],
        "the executor must receive the hook's arguments, not the model's"
    );

    ai.shutdown().await.expect("clean shutdown");
}

#[tokio::test]
async fn a_hook_denial_reaches_the_model_with_its_own_reason() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "echo", json!({"value": "x"})),
        Round::text("fine"),
    ])
    .await;

    let ai = ActonAI::builder()
        .app_name("policy-hook-deny")
        .provider(ProviderConfig::openai_compatible(
            server.base_url().to_string(),
            "mock-model",
        ))
        .on_tool_approval(|_invocation| async move {
            ApprovalDecision::deny("a change ticket is required")
        })
        .launch()
        .await
        .expect("launching the runtime must succeed");

    ai.prompt("echo")
        .with_tool(tool_definition("echo"), |args| async move { Ok(args) })
        .collect()
        .await
        .expect("the turn must complete");

    let second = server.requests().remove(1);
    let results = tool_results(&second);
    assert!(
        results[0].contains("a change ticket is required"),
        "the hook's own words must reach the model: {}",
        results[0]
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 5. No policy configured changes nothing
// =============================================================================

#[tokio::test]
async fn without_a_policy_every_tool_still_runs() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "bash", json!({"command": "ls"})),
        Round::text("listed"),
    ])
    .await;
    let ai = mock_llm::runtime_pointed_at(&server, "policy-absent").await;

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    let response = ai
        .prompt("list files")
        .with_tool(tool_definition("bash"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        1,
        "an unconfigured runtime must gate nothing"
    );
    assert_eq!(response.text, "listed");

    let second = server.requests().remove(1);
    let results = tool_results(&second);
    assert!(
        !results[0].contains("denied by policy"),
        "no denial may appear when no policy exists: {}",
        results[0]
    );

    ai.shutdown().await.expect("clean shutdown");
}

// =============================================================================
// 6. The TOML half
// =============================================================================

#[tokio::test]
async fn a_toml_policy_is_in_force_at_launch() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "bash", json!({"command": "ls"})),
        Round::text("nope"),
    ])
    .await;

    let toml = format!(
        "{}\n[tool_policy]\ndeny = [\"bash\"]\n",
        mock_llm::provider_toml("mock", &server, "mock-model")
    );
    let config = acton_ai::config::from_str(&toml).expect("the config must parse");
    let ai = ActonAI::builder()
        .app_name("policy-toml")
        .apply_config(config)
        .expect("the config must apply")
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();

    ai.prompt("list files")
        .with_tool(tool_definition("bash"), move |args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(args)
            }
        })
        .collect()
        .await
        .expect("the turn must complete");

    assert_eq!(
        ran.load(Ordering::SeqCst),
        0,
        "a `[tool_policy]` section must bite without any builder call"
    );

    ai.shutdown().await.expect("clean shutdown");
}
