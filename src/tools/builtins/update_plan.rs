//! The `update_plan` built-in tool: the model writes down what it is doing.
//!
//! Unusual among the built-ins in that it *does* nothing. There is no file to
//! read, no process to spawn, no socket to open — the whole tool is
//! [`Plan::parse`](crate::tools::plan::Plan::parse) plus the echo of what it
//! recorded. The observable effect happens one level up: the prompt round loop
//! recognizes a successful `update_plan` result and broadcasts
//! [`PlanUpdated`](crate::messages::PlanUpdated), which is how a UI, a log,
//! or an operator's status line learns what the model thinks it is doing.
//!
//! Because the executor is pure, a refused plan costs nothing and is worth
//! being strict about: the refusal comes back to the model as an ordinary tool
//! result naming the problem, and models correct these on the next round.

use crate::messages::ToolDefinition;
use crate::tools::actor::{ExecuteToolDirect, ToolActor, ToolActorResponse};
use crate::tools::plan::{Plan, PlanStepStatus, RawPlanStep, UPDATE_PLAN_TOOL};
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use acton_reactive::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

/// What the model sends when it calls `update_plan`.
#[derive(Debug, Deserialize)]
struct UpdatePlanArgs {
    /// The steps, in order.
    steps: Vec<RawPlanStep>,
    /// An optional note about the plan as a whole.
    #[serde(default)]
    note: Option<String>,
}

/// The `update_plan` tool executor.
///
/// Validates a model-supplied plan and echoes back the normalized version.
/// Holds no state and touches nothing outside its arguments.
#[derive(Debug, Default, Clone)]
pub struct UpdatePlanTool;

/// `update_plan` tool actor state.
///
/// Wraps [`UpdatePlanTool`] for per-agent tool spawning.
#[acton_actor]
pub struct UpdatePlanToolActor;

impl UpdatePlanTool {
    /// Creates a new `update_plan` tool.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        ToolConfig::new(ToolDefinition {
            // Validate-and-echo over its own arguments: sending the same
            // plan twice records the same plan, so checkpoint resume may
            // safely re-run a call a crash left uncertain.
            idempotent: true,
            name: UPDATE_PLAN_TOOL.to_string(),
            description: "Maintain a structured plan for the current task and show your \
                 progress through it. Use it only for genuinely multi-step work: skip \
                 planning entirely for trivial tasks, and never send a single-step plan \
                 — it will be rejected. Send the WHOLE plan on every call, not a diff. \
                 Keep at most one step 'in_progress' at a time, and call update_plan \
                 again immediately after completing a step, marking it 'completed' and \
                 the next step 'in_progress'. Steps are short, distinct phrases."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "The complete ordered list of steps, resent in full on every call. At least two steps: a plan too small to split is a plan not worth sending.",
                        "minItems": crate::tools::plan::MIN_PLAN_STEPS,
                        "maxItems": crate::tools::plan::MAX_PLAN_STEPS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "What this step does, in a few words"
                                },
                                "status": {
                                    "type": "string",
                                    "description": "Where this step stands; at most one step may be 'in_progress'",
                                    "enum": PlanStepStatus::all()
                                        .iter()
                                        .map(|status| status.as_str())
                                        .collect::<Vec<_>>()
                                }
                            },
                            "required": ["title", "status"]
                        }
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional one-line note about the plan or why it changed"
                    }
                },
                "required": ["steps"]
            }),
        })
    }

    /// Validates model-supplied arguments into a plan.
    ///
    /// The single place arguments become a [`Plan`], so `execute` and
    /// `validate_args` cannot disagree about what is acceptable.
    ///
    /// # Errors
    ///
    /// A validation [`ToolError`] whose message is the sentence handed back to
    /// the model: either serde's account of unusable JSON, or the
    /// [`PlanError`](crate::tools::plan::PlanError) naming the broken rule.
    fn plan_from_args(args: &Value) -> Result<Plan, ToolError> {
        let args: UpdatePlanArgs = serde_json::from_value(args.clone()).map_err(|e| {
            ToolError::validation_failed(UPDATE_PLAN_TOOL, format!("invalid arguments: {e}"))
        })?;

        Plan::parse(&args.steps, args.note.as_deref())
            .map_err(|e| ToolError::validation_failed(UPDATE_PLAN_TOOL, e.to_string()))
    }

    /// Renders a recorded plan as the tool's result value.
    ///
    /// The normalized plan is echoed back under `plan` for two readers: the
    /// model, which sees exactly what was recorded rather than what it sent,
    /// and the round loop, which reads it back out with
    /// [`plan_from_tool_result`](crate::tools::plan::plan_from_tool_result) to
    /// broadcast it.
    fn result_value(plan: &Plan) -> Result<Value, ToolError> {
        let encoded = serde_json::to_value(plan).map_err(|e| {
            ToolError::execution_failed(UPDATE_PLAN_TOOL, format!("could not encode the plan: {e}"))
        })?;

        Ok(json!({
            "status": "recorded",
            "plan": encoded,
            "completed": plan.completed_count(),
            "total": plan.step_count(),
        }))
    }
}

impl ToolExecutorTrait for UpdatePlanTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let plan = Self::plan_from_args(&args)?;
            Self::result_value(&plan)
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        Self::plan_from_args(args).map(|_| ())
    }
}

impl ToolActor for UpdatePlanToolActor {
    fn name() -> &'static str {
        UPDATE_PLAN_TOOL
    }

    fn definition() -> ToolDefinition {
        UpdatePlanTool::config().definition
    }

    async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<Self>("update_plan_tool".to_string());

        builder.act_on::<ExecuteToolDirect>(|actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let result = UpdatePlanTool::new().execute(args).await;

                let response = match result {
                    Ok(value) => match serde_json::to_string(&value) {
                        Ok(encoded) => {
                            ToolActorResponse::success(correlation_id, tool_call_id, encoded)
                        }
                        Err(e) => ToolActorResponse::error(
                            correlation_id,
                            tool_call_id,
                            format!("could not encode the result: {e}"),
                        ),
                    },
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
    use crate::tools::plan::plan_from_tool_result;

    fn valid_args() -> Value {
        json!({
            "steps": [
                {"title": "read the failing test", "status": "completed"},
                {"title": "fix the parser", "status": "in_progress"},
                {"title": "run the suite", "status": "pending"},
            ],
            "note": "parser first, then the suite",
        })
    }

    #[tokio::test]
    async fn a_valid_plan_is_recorded_and_echoed_back() {
        let result = UpdatePlanTool::new()
            .execute(valid_args())
            .await
            .expect("a well-formed plan is recorded");

        assert_eq!(result["status"], "recorded");
        assert_eq!(result["completed"], 1);
        assert_eq!(result["total"], 3);

        // The echo is what the model reads back, and what the round loop
        // broadcasts — so it has to be a plan again, not just plan-shaped.
        let plan = plan_from_tool_result(UPDATE_PLAN_TOOL, &result)
            .expect("the result must carry a recoverable plan");
        assert_eq!(plan.step_count(), 3);
        assert_eq!(
            plan.in_progress().map(|step| step.title().as_str()),
            Some("fix the parser")
        );
    }

    #[tokio::test]
    async fn the_echoed_plan_is_normalized_rather_than_repeated() {
        let result = UpdatePlanTool::new()
            .execute(json!({
                "steps": [
                    {"title": "  tidy the imports  ", "status": "pending"},
                    {"title": "run the lints", "status": "pending"},
                ],
                "note": "   ",
            }))
            .await
            .expect("padding is not a failure");

        assert_eq!(result["plan"]["steps"][0]["title"], "tidy the imports");
        assert!(
            result["plan"]["note"].is_null(),
            "a blank note is recorded as absent: {result}"
        );
    }

    #[tokio::test]
    async fn two_steps_in_progress_are_refused_with_a_usable_sentence() {
        let error = UpdatePlanTool::new()
            .execute(json!({
                "steps": [
                    {"title": "a", "status": "in_progress"},
                    {"title": "b", "status": "in_progress"},
                ],
            }))
            .await
            .expect_err("only one step may be in flight");

        let message = error.to_string();
        assert!(
            message.contains("plan steps 1 and 2 are both in_progress"),
            "the model must be told which steps clash: {message}"
        );
    }

    #[tokio::test]
    async fn an_empty_plan_is_refused() {
        let error = UpdatePlanTool::new()
            .execute(json!({"steps": []}))
            .await
            .expect_err("a plan needs steps");

        assert!(error.to_string().contains("at least two"), "{error}");
    }

    #[tokio::test]
    async fn a_single_step_plan_is_refused_with_a_corrective_message() {
        let error = UpdatePlanTool::new()
            .execute(json!({
                "steps": [{"title": "do the whole task", "status": "in_progress"}],
            }))
            .await
            .expect_err("a single-step plan is the tool's own anti-pattern");

        // The message must tell the model what to do instead — split the
        // work, or skip planning — not merely that it was wrong.
        let message = error.to_string();
        assert!(message.contains("at least two"), "{message}");
        assert!(message.contains("skip planning"), "{message}");
    }

    #[tokio::test]
    async fn malformed_arguments_are_refused_rather_than_panicking() {
        let tool = UpdatePlanTool::new();

        for args in [
            json!({}),
            json!({"steps": "not an array"}),
            json!({"steps": [{"status": "pending"}]}),
            json!({"steps": [{"title": "a", "status": "done"}]}),
        ] {
            let error = tool
                .execute(args.clone())
                .await
                .expect_err("model-supplied arguments are untrusted");
            assert!(
                error.to_string().contains("invalid arguments"),
                "{args} -> {error}"
            );
        }
    }

    #[tokio::test]
    async fn validate_args_agrees_with_execute() {
        let tool = UpdatePlanTool::new();

        assert!(tool.validate_args(&valid_args()).is_ok());
        assert!(tool.execute(valid_args()).await.is_ok());

        let bad = json!({"steps": [{"title": "  ", "status": "pending"}]});
        assert!(tool.validate_args(&bad).is_err());
        assert!(tool.execute(bad).await.is_err());
    }

    #[test]
    fn the_tool_needs_no_sandbox() {
        // It reads its arguments and returns a value. There is nothing here
        // for a subprocess to contain.
        assert!(!UpdatePlanTool::new().requires_sandbox());
    }

    #[test]
    fn the_offered_schema_names_the_three_statuses() {
        let config = UpdatePlanTool::config();
        assert_eq!(config.definition.name, UPDATE_PLAN_TOOL);
        assert_eq!(UpdatePlanToolActor::definition(), config.definition);
        assert_eq!(UpdatePlanToolActor::name(), UPDATE_PLAN_TOOL);

        let schema = &config.definition.input_schema;
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"], json!(["steps"]));

        let statuses = &schema["properties"]["steps"]["items"]["properties"]["status"]["enum"];
        assert_eq!(
            statuses,
            &json!(["pending", "in_progress", "completed"]),
            "the schema must offer exactly the statuses the parser accepts"
        );
    }
}
