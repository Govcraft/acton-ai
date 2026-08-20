//! Context-window budget built-in tool.
//!
//! Reports how much of the conversation's context window is spent: the total
//! window (resolved at launch from per-provider `context_window_tokens`,
//! falling back to `[context] max_tokens`, then the built-in default), an
//! estimate of the tokens the current history occupies, and what remains.
//!
//! The conversation history has exactly one owner — the prompt loop that is
//! serving the turn — so this tool never holds a handle to it. Instead the
//! loop measures the live message state at call time with the runtime's own
//! [`ContextWindow`] estimator and injects the two resulting numbers into the
//! call's arguments under [`CONTEXT_STATE_ARG`]. What is left for the tool is
//! pure arithmetic, which is what makes it testable without a runtime.

use crate::memory::ContextWindow;
use crate::messages::{Message, ToolDefinition};
use crate::tools::actor::{ExecuteToolDirect, ToolActor, ToolActorResponse};
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// The tool's registered name.
pub const GET_CONTEXT_REMAINING_TOOL: &str = "get_context_remaining";

/// The argument key the prompt loop injects the measured budget under.
///
/// Underscore-prefixed so it reads as internal: the model never supplies it,
/// and the input schema does not advertise it.
pub(crate) const CONTEXT_STATE_ARG: &str = "_context_state";

/// The two numbers the prompt loop measures at call time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextBudget {
    /// The full context window, in tokens.
    pub(crate) total_tokens: usize,
    /// Estimated tokens the live message state occupies.
    pub(crate) used_tokens: usize,
}

/// Measures the live message state against the runtime's context window.
///
/// Pure: reads the window's configured budget and runs its estimator over the
/// messages. This is the single estimation path — the same estimator that
/// decides truncation decides what this tool reports.
pub(crate) fn measure_context_budget(
    window: &ContextWindow,
    messages: &[Message],
) -> ContextBudget {
    ContextBudget {
        total_tokens: window.config().max_tokens,
        used_tokens: window.estimate_total_tokens(messages),
    }
}

/// Returns the call's arguments with the measured budget injected.
///
/// `None` for the window (a runtime built with `without_context_window`)
/// injects nothing, and the tool answers with an error explaining why. A
/// non-object `arguments` value (some providers send `null` for a no-argument
/// call) is replaced with a fresh object rather than failing.
pub(crate) fn inject_context_state(
    arguments: &Value,
    window: Option<&ContextWindow>,
    messages: &[Message],
) -> Value {
    let Some(window) = window else {
        return arguments.clone();
    };

    let budget = measure_context_budget(window, messages);
    let mut args = match arguments {
        Value::Object(map) => Value::Object(map.clone()),
        _ => json!({}),
    };
    if let Value::Object(ref mut map) = args {
        map.insert(
            CONTEXT_STATE_ARG.to_string(),
            serde_json::to_value(&budget).unwrap_or(Value::Null),
        );
    }
    args
}

/// The report the model receives.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct ContextReport {
    /// The full context window, in tokens.
    total_tokens: usize,
    /// Estimated tokens the current history occupies.
    used_tokens: usize,
    /// Tokens still available (never negative).
    remaining_tokens: usize,
    /// Used as a percentage of total, rounded to two decimals.
    percent_used: f64,
}

/// Builds the report from a measured budget. Pure arithmetic.
fn context_report(budget: &ContextBudget) -> ContextReport {
    let percent_used = if budget.total_tokens > 0 {
        let ratio = budget.used_tokens as f64 / budget.total_tokens as f64;
        (ratio * 10_000.0).round() / 100.0
    } else {
        0.0
    };
    ContextReport {
        total_tokens: budget.total_tokens,
        used_tokens: budget.used_tokens,
        remaining_tokens: budget.total_tokens.saturating_sub(budget.used_tokens),
        percent_used,
    }
}

/// Arguments for the `get_context_remaining` tool.
#[derive(Debug, Deserialize)]
struct GetContextRemainingArgs {
    /// The budget the prompt loop measured at call time.
    #[serde(rename = "_context_state", default)]
    context_state: Option<ContextBudget>,
}

/// Context-remaining tool executor.
///
/// Pure consumer of the budget the prompt loop injects; see the module
/// documentation for why the measurement does not happen here.
#[derive(Debug, Default, Clone)]
pub struct GetContextRemainingTool;

/// Context-remaining tool actor state.
///
/// Exists for parity with the other built-in tool actors. Driven directly —
/// outside a managed prompt turn — there is no live conversation to measure,
/// so it reports the same "state unavailable" error the executor does.
#[acton_actor]
pub struct GetContextRemainingToolActor;

impl GetContextRemainingTool {
    /// Creates a new context-remaining tool.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        ToolConfig::new(ToolDefinition {
            name: GET_CONTEXT_REMAINING_TOOL.to_string(),
            description: "Report the conversation's context-window budget: total tokens, an \
                          estimate of tokens used by the current history, tokens remaining, and \
                          percent used. Takes no arguments."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        })
    }
}

impl ToolExecutorTrait for GetContextRemainingTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let args: GetContextRemainingArgs = serde_json::from_value(args).map_err(|e| {
                ToolError::validation_failed(
                    GET_CONTEXT_REMAINING_TOOL,
                    format!("invalid arguments: {e}"),
                )
            })?;

            let budget = args.context_state.ok_or_else(|| {
                ToolError::execution_failed(
                    GET_CONTEXT_REMAINING_TOOL,
                    "live conversation state is unavailable: this tool only works inside a \
                     managed prompt or conversation turn on a runtime with a context window \
                     configured (not one built with without_context_window)",
                )
            })?;

            serde_json::to_value(context_report(&budget)).map_err(|e| {
                ToolError::execution_failed(
                    GET_CONTEXT_REMAINING_TOOL,
                    format!("failed to serialize report: {e}"),
                )
            })
        })
    }

    fn validate_args(&self, _args: &Value) -> Result<(), ToolError> {
        // No caller-supplied arguments; the injected state is validated in
        // execute, where its absence has a precise error.
        Ok(())
    }
}

impl ToolActor for GetContextRemainingToolActor {
    fn name() -> &'static str {
        GET_CONTEXT_REMAINING_TOOL
    }

    fn definition() -> ToolDefinition {
        GetContextRemainingTool::config().definition
    }

    async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder =
            runtime.new_actor_with_name::<Self>("get_context_remaining_tool".to_string());

        builder.act_on::<ExecuteToolDirect>(|actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let tool = GetContextRemainingTool::new();
                let result = tool.execute(args).await;

                let response = match result {
                    Ok(value) => {
                        let result_str = serde_json::to_string(&value)
                            .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e));
                        ToolActorResponse::success(correlation_id, tool_call_id, result_str)
                    }
                    Err(e) => ToolActorResponse::error(correlation_id, tool_call_id, e.to_string()),
                };

                broker.broadcast(response).await;
            })
        });

        builder.start().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{CharRatioEstimator, ContextWindowConfig, TokenEstimator};
    use std::sync::Arc;

    fn window_with(max_tokens: usize) -> ContextWindow {
        ContextWindow::new(ContextWindowConfig::with_max_tokens(max_tokens))
            .with_estimator(Arc::new(CharRatioEstimator::new(0.25)))
    }

    fn known_history() -> Vec<Message> {
        vec![
            Message::system("You are a helpful assistant."),
            Message::user("What is the capital of France?"),
            Message::assistant("The capital of France is Paris."),
        ]
    }

    #[test]
    fn measure_matches_the_shared_estimator() {
        let window = window_with(1000);
        let history = known_history();

        let budget = measure_context_budget(&window, &history);

        // The same estimator the truncation path uses, not a second one.
        let expected: usize = history
            .iter()
            .map(|m| CharRatioEstimator::new(0.25).estimate_message(m))
            .sum();
        assert_eq!(budget.used_tokens, expected);
        assert_eq!(budget.used_tokens, window.estimate_total_tokens(&history));
        assert_eq!(budget.total_tokens, 1000);
    }

    #[test]
    fn report_arithmetic_is_exact() {
        let report = context_report(&ContextBudget {
            total_tokens: 8000,
            used_tokens: 2000,
        });

        assert_eq!(report.total_tokens, 8000);
        assert_eq!(report.used_tokens, 2000);
        assert_eq!(report.remaining_tokens, 6000);
        assert!((report.percent_used - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_percent_rounds_to_two_decimals() {
        let report = context_report(&ContextBudget {
            total_tokens: 3,
            used_tokens: 1,
        });
        assert!((report.percent_used - 33.33).abs() < f64::EPSILON);
    }

    #[test]
    fn report_remaining_saturates_when_over_budget() {
        let report = context_report(&ContextBudget {
            total_tokens: 100,
            used_tokens: 150,
        });
        assert_eq!(report.remaining_tokens, 0);
        assert!((report.percent_used - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_zero_total_is_zero_percent() {
        let report = context_report(&ContextBudget {
            total_tokens: 0,
            used_tokens: 0,
        });
        assert_eq!(report.remaining_tokens, 0);
        assert!(report.percent_used.abs() < f64::EPSILON);
    }

    #[test]
    fn absent_config_falls_back_to_default_window() {
        // A runtime with no [context] section and no per-provider
        // context_window_tokens resolves ContextWindowConfig::default(),
        // whose budget this tool then reports.
        let window = ContextWindow::new(ContextWindowConfig::default());
        let budget = measure_context_budget(&window, &known_history());
        assert_eq!(
            budget.total_tokens,
            ContextWindowConfig::default().max_tokens
        );
    }

    #[test]
    fn inject_adds_measured_state_to_arguments() {
        let window = window_with(500);
        let history = known_history();

        let args = inject_context_state(&json!({}), Some(&window), &history);

        let state = &args[CONTEXT_STATE_ARG];
        assert_eq!(state["total_tokens"], 500);
        assert_eq!(state["used_tokens"], window.estimate_total_tokens(&history));
    }

    #[test]
    fn inject_replaces_non_object_arguments() {
        let window = window_with(500);
        let args = inject_context_state(&Value::Null, Some(&window), &known_history());
        assert!(args[CONTEXT_STATE_ARG].is_object());
    }

    #[test]
    fn inject_without_a_window_leaves_arguments_untouched() {
        let args = inject_context_state(&json!({}), None, &known_history());
        assert_eq!(args, json!({}));
    }

    #[tokio::test]
    async fn execute_reports_the_injected_budget() {
        let window = window_with(1000);
        let history = known_history();
        let used = window.estimate_total_tokens(&history);
        let args = inject_context_state(&json!({}), Some(&window), &history);

        let tool = GetContextRemainingTool::new();
        let result = tool.execute(args).await.unwrap();

        assert_eq!(result["total_tokens"], 1000);
        assert_eq!(result["used_tokens"], used);
        assert_eq!(result["remaining_tokens"], 1000 - used);
        let expected_percent = (used as f64 / 1000.0 * 10_000.0).round() / 100.0;
        assert!((result["percent_used"].as_f64().unwrap() - expected_percent).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn execute_without_injected_state_is_a_clear_error() {
        let tool = GetContextRemainingTool::new();
        let err = tool.execute(json!({})).await.unwrap_err();
        assert!(err.to_string().contains("live conversation state"));
    }

    #[test]
    fn config_has_correct_schema() {
        let config = GetContextRemainingTool::config();
        assert_eq!(config.definition.name, GET_CONTEXT_REMAINING_TOOL);
        assert!(config.definition.description.contains("context-window"));
        assert!(!config.sandboxed, "pure computation must not be sandboxed");

        let schema = &config.definition.input_schema;
        assert_eq!(schema["type"], "object");
        assert!(
            schema["properties"].as_object().unwrap().is_empty(),
            "the injected state key must not be advertised to the model"
        );
    }
}
