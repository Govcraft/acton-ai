//! Tests for the `#[tool]` attribute macro.
//!
//! These live in `tests/` on purpose. The macro expands to `::acton_ai::…`
//! paths, which only resolve *downstream* of the crate — inside `src/` the
//! crate is `crate`, not `acton_ai`, so a unit test could not exercise the
//! real expansion. An integration test is the first place the generated code
//! compiles the way a user's would.
//!
//! Three layers here:
//!
//! 1. assertions on the generated `Tool` impl — name, description, schema,
//!    and `call` behavior, including the error paths a model actually hits;
//! 2. `trybuild` cases pinning the compile errors for misuse (see
//!    [`ui_cases`] and `tests/ui/`);
//! 3. one end-to-end test that drives the real stack against the scripted
//!    server in [`mock_llm`], proving the generated name, description, and
//!    schema reach the wire and that the real function body runs.

mod mock_llm;

use acton_ai::prelude::*;
use mock_llm::{contains_ref, runtime_pointed_at, tool_named, MockServer, Round};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// =============================================================================
// Tools under test
// =============================================================================

/// Computes the result of a math expression.
///
/// Pass the expression as a string; `precision` controls how many decimal
/// places come back.
#[tool]
async fn calculator(expr: String, precision: Option<u8>) -> Result<serde_json::Value, ToolError> {
    let places = usize::from(precision.unwrap_or(2));
    let value = match expr.as_str() {
        "42 * 17" => 714.0,
        "1 / 3" => 1.0 / 3.0,
        other => {
            return Err(ToolError::execution_failed(
                "calculator",
                format!("unsupported expression: {other}"),
            ))
        }
    };
    Ok(json!({ "result": format!("{value:.places$}") }))
}

/// Returns a fixed timestamp.
#[tool]
async fn now() -> Result<String, ToolError> {
    Ok("2026-08-12T00:00:00Z".to_string())
}

/// A nested parameter type, to prove struct parameters are inlined.
#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
struct RetryPolicy {
    attempts: u8,
    backoff_ms: u32,
}

/// Fetches a URL with a retry policy.
///
/// The policy is a struct parameter, so its schema has to be inlined rather
/// than referenced.
#[tool]
async fn fetch_with_policy(url: String, policy: RetryPolicy) -> Result<RetryPolicy, ToolError> {
    if url.is_empty() {
        return Err(ToolError::validation_failed("fetch_with_policy", "empty url"));
    }
    Ok(policy)
}

/// A tool whose parameters are named after the expansion's own locals.
///
/// If the macro reused a user parameter name as a generated binding, `args`
/// would shadow the arguments value between one extraction and the next and
/// the second parameter would fail to resolve. This compiling *is* the test;
/// the assertion below checks it also behaves.
#[tool]
async fn shadow_trap(args: String, result: String) -> Result<String, ToolError> {
    Ok(format!("{args}/{result}"))
}

/// A public tool, to prove the generated struct inherits visibility.
///
/// A `pub` function inside an integration test is unremarkable, but the
/// struct being `pub` too is what lets a user define tools in a library.
#[tool]
pub async fn public_tool() -> Result<u8, ToolError> {
    Ok(1)
}

// =============================================================================
// name() and description()
// =============================================================================

#[test]
fn name_is_the_function_name_verbatim() {
    assert_eq!(Calculator.name(), "calculator");
    assert_eq!(FetchWithPolicy.name(), "fetch_with_policy");
    assert_eq!(
        Now.name(),
        "now",
        "snake_case is what the LLM APIs expect, so the name is not transformed"
    );
}

#[test]
fn the_generated_struct_is_named_in_pascal_case() {
    // Naming the types at all is the assertion: `fetch_with_policy` produced
    // `FetchWithPolicy`, not `Fetch_with_policy` or `Fetchwithpolicy`.
    let _: FetchWithPolicy = FetchWithPolicy;
    let _: ShadowTrap = ShadowTrap;
    let _: PublicTool = PublicTool;
}

#[test]
fn description_is_the_doc_comment() {
    assert_eq!(Now.description(), "Returns a fixed timestamp.");
}

#[test]
fn a_multi_line_doc_comment_round_trips_with_its_paragraphs() {
    assert_eq!(
        Calculator.description(),
        "Computes the result of a math expression.\n\n\
         Pass the expression as a string; `precision` controls how many decimal\n\
         places come back.",
        "the model should see the same paragraphs the author wrote"
    );
}

#[test]
fn description_is_never_empty() {
    // The macro rejects a missing doc comment at compile time (see the
    // trybuild case); this guards the other direction — that a present doc
    // comment is not silently dropped somewhere in extraction.
    for description in [
        Calculator.description(),
        Now.description(),
        FetchWithPolicy.description(),
        ShadowTrap.description(),
        PublicTool.description(),
    ] {
        assert!(!description.trim().is_empty());
    }
}

// =============================================================================
// input_schema()
// =============================================================================

#[test]
fn schema_has_one_property_per_parameter_named_after_it() {
    let schema = Calculator.input_schema();

    assert_eq!(schema["type"], "object");
    let properties = schema["properties"]
        .as_object()
        .expect("properties must be an object");
    assert_eq!(properties.len(), 2, "{schema:#}");
    assert!(properties.contains_key("expr"), "{schema:#}");
    assert!(properties.contains_key("precision"), "{schema:#}");
}

#[test]
fn required_lists_every_parameter_that_is_not_an_option() {
    let schema = Calculator.input_schema();

    assert_eq!(
        schema["required"],
        json!(["expr"]),
        "`precision: Option<u8>` must not be required: {schema:#}"
    );
}

#[test]
fn a_tool_with_no_optional_parameters_requires_all_of_them() {
    let schema = FetchWithPolicy.input_schema();

    assert_eq!(schema["required"], json!(["url", "policy"]), "{schema:#}");
}

#[test]
fn a_zero_parameter_tool_gets_an_empty_object_schema() {
    let schema = Now.input_schema();

    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"], json!({}), "{schema:#}");
    assert_eq!(schema["required"], json!([]), "{schema:#}");
}

#[test]
fn a_struct_parameter_is_inlined_with_no_ref() {
    let schema = FetchWithPolicy.input_schema();

    assert!(
        !contains_ref(&schema),
        "provider support for $ref is inconsistent, so the schema must be \
         self-contained: {schema:#}"
    );
    assert!(schema.get("$defs").is_none(), "{schema:#}");

    let policy = &schema["properties"]["policy"];
    assert_eq!(policy["type"], "object", "{schema:#}");
    assert!(policy["properties"].get("attempts").is_some(), "{schema:#}");
    assert!(
        policy["properties"].get("backoff_ms").is_some(),
        "{schema:#}"
    );
}

#[test]
fn required_follows_the_signature_order() {
    // Alphabetically this would be `["policy", "url"]`, so the assertion in
    // `a_tool_with_no_optional_parameters_requires_all_of_them` is really an
    // ordering assertion too. Stated here so the intent survives an edit.
    let schema = FetchWithPolicy.input_schema();

    assert_eq!(
        schema["required"][0], "url",
        "`url` is declared first: {schema:#}"
    );
}

#[test]
fn parameter_schemas_describe_the_declared_type() {
    let schema = Calculator.input_schema();

    assert_eq!(schema["properties"]["expr"]["type"], "string", "{schema:#}");
}

// =============================================================================
// call()
// =============================================================================

#[tokio::test]
async fn call_runs_the_function_and_serializes_the_result() {
    let result = Calculator
        .call(json!({"expr": "42 * 17", "precision": 1}))
        .await
        .expect("a well-formed call must succeed");

    assert_eq!(result, json!({"result": "714.0"}));
}

#[tokio::test]
async fn call_omitting_an_optional_parameter_uses_the_function_default() {
    let result = Calculator
        .call(json!({"expr": "42 * 17"}))
        .await
        .expect("an absent optional parameter must be None, not an error");

    assert_eq!(
        result,
        json!({"result": "714.00"}),
        "the body's `unwrap_or(2)` should have applied"
    );
}

#[tokio::test]
async fn call_accepts_an_explicit_null_for_an_optional_parameter() {
    let result = Calculator
        .call(json!({"expr": "42 * 17", "precision": null}))
        .await
        .expect("an explicit null must deserialize to None");

    assert_eq!(result, json!({"result": "714.00"}));
}

#[tokio::test]
async fn call_with_no_arguments_works_for_a_zero_parameter_tool() {
    let result = Now
        .call(json!({}))
        .await
        .expect("a zero-parameter tool needs nothing from the model");

    assert_eq!(result, json!("2026-08-12T00:00:00Z"));
}

#[tokio::test]
async fn a_missing_required_argument_is_reported_by_name() {
    let error = Calculator
        .call(json!({"precision": 2}))
        .await
        .expect_err("a missing required argument must not reach the function");

    let message = error.to_string();
    assert!(
        message.contains("expr"),
        "the model can only fix what the error names: {message}"
    );
    assert!(message.contains("missing"), "{message}");
}

#[tokio::test]
async fn a_type_mismatch_is_reported_by_name() {
    let error = Calculator
        .call(json!({"expr": "42 * 17", "precision": "lots"}))
        .await
        .expect_err("a string in a u8 parameter must not reach the function");

    let message = error.to_string();
    assert!(
        message.contains("precision"),
        "the error must name the offending parameter, not just the type: {message}"
    );
    assert!(message.contains("invalid type"), "{message}");
}

#[tokio::test]
async fn a_mismatch_inside_a_struct_parameter_carries_the_full_path() {
    let error = FetchWithPolicy
        .call(json!({
            "url": "https://example.com",
            "policy": {"attempts": "many", "backoff_ms": 100},
        }))
        .await
        .expect_err("a bad nested field must be rejected");

    let message = error.to_string();
    assert!(
        message.contains("policy") && message.contains("attempts"),
        "serde_path_to_error should name the path, not just the leaf: {message}"
    );
}

#[tokio::test]
async fn non_object_arguments_are_rejected_rather_than_panicking() {
    let error = Calculator
        .call(json!("42 * 17"))
        .await
        .expect_err("a bare string is not an arguments object");

    assert!(error.to_string().contains("expr"), "{error}");
}

#[tokio::test]
async fn an_error_from_the_function_body_propagates_unchanged() {
    let error = Calculator
        .call(json!({"expr": "quantum"}))
        .await
        .expect_err("the body rejects unsupported expressions");

    assert!(
        error.to_string().contains("unsupported expression"),
        "the macro must not swallow or rewrap the body's error: {error}"
    );
}

#[tokio::test]
async fn a_parameter_named_like_the_expansions_locals_still_resolves() {
    let result = ShadowTrap
        .call(json!({"args": "a", "result": "b"}))
        .await
        .expect("parameters named `args`/`result` must not shadow the expansion");

    assert_eq!(result, json!("a/b"));
}

#[tokio::test]
async fn a_non_value_return_type_is_serialized() {
    let result = FetchWithPolicy
        .call(json!({
            "url": "https://example.com",
            "policy": {"attempts": 3, "backoff_ms": 250},
        }))
        .await
        .expect("a Serialize return type must be converted to a Value");

    assert_eq!(result, json!({"attempts": 3, "backoff_ms": 250}));
}

// =============================================================================
// The original function survives
// =============================================================================

#[tokio::test]
async fn the_annotated_function_is_still_callable_directly() {
    let direct = calculator("42 * 17".to_string(), Some(1))
        .await
        .expect("the original function must be untouched");

    assert_eq!(direct, json!({"result": "714.0"}));
}

#[tokio::test]
async fn the_original_function_keeps_its_typed_signature() {
    // Not `serde_json::Value` — the function returns `RetryPolicy`, and it
    // only stays unit-testable if the macro left that alone.
    let policy: RetryPolicy = fetch_with_policy(
        "https://example.com".to_string(),
        RetryPolicy {
            attempts: 3,
            backoff_ms: 250,
        },
    )
    .await
    .expect("the original function must be untouched");

    assert_eq!(policy.attempts, 3);
}

#[tokio::test]
async fn a_zero_parameter_function_is_still_callable_directly() {
    assert_eq!(now().await.expect("must succeed"), "2026-08-12T00:00:00Z");
}

// =============================================================================
// The trait object path
// =============================================================================

#[tokio::test]
async fn a_generated_tool_works_behind_a_trait_object() {
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(Calculator), Box::new(Now)];

    assert_eq!(tools[0].name(), "calculator");
    assert_eq!(
        tools[1]
            .call(json!({}))
            .await
            .expect("dispatch through dyn Tool must work"),
        json!("2026-08-12T00:00:00Z")
    );
}

// =============================================================================
// Compile-fail cases
// =============================================================================

/// Locks the diagnostics for every documented misuse.
///
/// One `#[test]` rather than one per file because `trybuild` batches a whole
/// directory into a single `cargo build`; splitting it would multiply the
/// slowest part of this suite by the number of cases.
#[test]
fn ui_cases() {
    let cases = trybuild::TestCases::new();
    // The happy path, compiled through the same harness, so a change that
    // breaks ordinary expansion fails here too and not only in the tests
    // above.
    cases.pass("tests/ui/pass_calculator.rs");

    cases.compile_fail("tests/ui/missing_doc_comment.rs");
    cases.compile_fail("tests/ui/not_async.rs");
    cases.compile_fail("tests/ui/wrong_return_type.rs");
    cases.compile_fail("tests/ui/self_receiver.rs");
    cases.compile_fail("tests/ui/generic_function.rs");
    cases.compile_fail("tests/ui/destructuring_parameter.rs");
    cases.compile_fail("tests/ui/attribute_arguments.rs");
}

// =============================================================================
// End to end
// =============================================================================

/// The generated tool, driven by a scripted model through the real stack.
///
/// Asserts both directions: that the generated name, description, and schema
/// reach the wire in the shape the provider expects, and that the model's
/// call actually runs the annotated function body and the result flows back
/// into a completed response.
#[tokio::test]
async fn a_generated_tool_round_trips_through_the_prompt_loop() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "calculator", json!({"expr": "42 * 17"})),
        Round::text("42 * 17 is 714.00."),
    ])
    .await;
    let runtime = runtime_pointed_at(&server, "tool-macro-test").await;

    let response = runtime
        .prompt("What is 42 * 17?")
        .add_tool(Calculator)
        .collect()
        .await
        .expect("the prompt must complete");

    assert_eq!(response.text, "42 * 17 is 714.00.");
    assert_eq!(
        server.request_count(),
        2,
        "one round to call the tool, one to answer"
    );

    // -- what went out on the wire --
    let requests = server.requests();
    let offered = tool_named(&requests[0], "calculator")
        .expect("the generated tool must be offered to the model");

    assert_eq!(
        offered["function"]["description"],
        json!(Calculator.description()),
        "the doc comment must reach the provider"
    );
    let schema = &offered["function"]["parameters"];
    assert_eq!(schema["type"], "object", "{schema:#}");
    assert_eq!(schema["required"], json!(["expr"]), "{schema:#}");
    assert!(
        schema["properties"].get("precision").is_some(),
        "optional parameters still appear as properties: {schema:#}"
    );
    assert!(!contains_ref(schema), "{schema:#}");

    // -- what came back --
    let follow_up = serde_json::to_string(&requests[1]).expect("request must serialize");
    assert!(
        follow_up.contains("714.00"),
        "the real function body must have run and its result fed back: {follow_up}"
    );
}

/// Two calls to the same generated tool in one round both execute.
#[tokio::test]
async fn parallel_calls_to_a_generated_tool_both_run() {
    let server = MockServer::start(vec![
        Round::tool_call("call_1", "calculator", json!({"expr": "42 * 17"}))
            .with_tool_call("call_2", "calculator", json!({"expr": "1 / 3", "precision": 3})),
        Round::text("Done."),
    ])
    .await;
    let runtime = runtime_pointed_at(&server, "tool-macro-parallel-test").await;

    let response = runtime
        .prompt("Compute both.")
        .add_tool(Calculator)
        .collect()
        .await
        .expect("the prompt must complete");

    assert_eq!(response.text, "Done.");
    let follow_up =
        serde_json::to_string(&server.requests()[1]).expect("request must serialize");
    assert!(follow_up.contains("714.00"), "{follow_up}");
    assert!(follow_up.contains("0.333"), "{follow_up}");
}

/// A tool error from the generated glue is reported to the model rather than
/// aborting the prompt.
#[tokio::test]
async fn a_bad_argument_is_fed_back_to_the_model_not_raised() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);

    let server = MockServer::start(vec![
        Round::tool_call("call_1", "calculator", json!({"precision": 2})),
        Round::text("Sorry, I need an expression."),
    ])
    .await;
    let runtime = runtime_pointed_at(&server, "tool-macro-error-test").await;

    let response = runtime
        .prompt("Compute something.")
        .add_tool(Calculator)
        .on_token(move |_| {
            observed.fetch_add(1, Ordering::SeqCst);
        })
        .collect()
        .await
        .expect("a tool error must not fail the prompt");

    assert_eq!(response.text, "Sorry, I need an expression.");
    assert!(calls.load(Ordering::SeqCst) > 0, "the stream must have run");

    let follow_up =
        serde_json::to_string(&server.requests()[1]).expect("request must serialize");
    assert!(
        follow_up.contains("expr"),
        "the model needs to be told which parameter was missing: {follow_up}"
    );
}
