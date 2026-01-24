//! Activate skill built-in tool.
//!
//! Activates a skill and returns its full instructions.
//! Only available when the `agent-skills` feature is enabled.

use crate::messages::ToolDefinition;
use crate::skills::SkillRegistry;
use crate::tools::actor::{ExecuteToolDirect, ToolActor, ToolActorResponse};
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, RwLock};

/// Activate skill tool executor.
///
/// Activates a skill by name and returns its full instructions.
#[derive(Debug)]
pub struct ActivateSkillTool {
    /// Reference to the skill registry
    registry: Arc<RwLock<SkillRegistry>>,
}

/// Activate skill tool actor state.
///
/// This actor wraps the skill registry for per-agent skill activation.
#[acton_actor]
pub struct ActivateSkillToolActor {
    /// Reference to the skill registry
    registry: Arc<RwLock<SkillRegistry>>,
}

/// Arguments for the activate_skill tool.
#[derive(Debug, Deserialize)]
struct ActivateSkillArgs {
    /// Name of the skill to activate
    name: String,
}

/// Result of activating a skill.
#[derive(Debug, Serialize)]
struct ActivateSkillResult {
    /// Skill name
    name: String,
    /// Skill description
    description: String,
    /// Full instructions content
    instructions: String,
    /// Path to the skill file
    path: String,
    /// Tags associated with the skill
    tags: Vec<String>,
}

impl ActivateSkillTool {
    /// Creates a new activate skill tool with the given registry.
    #[must_use]
    pub fn new(registry: Arc<RwLock<SkillRegistry>>) -> Self {
        Self { registry }
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        ToolConfig::new(ToolDefinition {
            name: "activate_skill".to_string(),
            description: "Activate a skill and receive its full instructions. Call list_skills first to see available skills.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to activate"
                    }
                },
                "required": ["name"]
            }),
        })
    }
}

impl ToolExecutorTrait for ActivateSkillTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        let registry = Arc::clone(&self.registry);

        Box::pin(async move {
            let args: ActivateSkillArgs = serde_json::from_value(args).map_err(|e| {
                ToolError::validation_failed("activate_skill", format!("invalid arguments: {e}"))
            })?;

            if args.name.is_empty() {
                return Err(ToolError::validation_failed(
                    "activate_skill",
                    "skill name cannot be empty",
                ));
            }

            let registry_guard = registry.read().map_err(|e| {
                ToolError::execution_failed(
                    "activate_skill",
                    format!("failed to acquire registry lock: {e}"),
                )
            })?;

            let skill = registry_guard.get(&args.name).ok_or_else(|| {
                ToolError::execution_failed(
                    "activate_skill",
                    format!("skill '{}' not found", args.name),
                )
            })?;

            let result = ActivateSkillResult {
                name: skill.name().to_string(),
                description: skill.description().to_string(),
                instructions: skill.instructions().to_string(),
                path: skill.info.path.display().to_string(),
                tags: skill.info.tags.clone(),
            };

            Ok(json!(result))
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: ActivateSkillArgs = serde_json::from_value(args.clone()).map_err(|e| {
            ToolError::validation_failed("activate_skill", format!("invalid arguments: {e}"))
        })?;

        if args.name.is_empty() {
            return Err(ToolError::validation_failed(
                "activate_skill",
                "skill name cannot be empty",
            ));
        }

        Ok(())
    }
}

impl ToolActor for ActivateSkillToolActor {
    fn name() -> &'static str {
        "activate_skill"
    }

    fn definition() -> ToolDefinition {
        ActivateSkillTool::config().definition
    }

    async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        // This is a placeholder - the actual spawn requires a registry reference
        // Use spawn_with_registry instead
        let mut builder = runtime.new_actor_with_name::<Self>("activate_skill_tool".to_string());

        builder.act_on::<ExecuteToolDirect>(|actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let registry = Arc::clone(&actor.model.registry);
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let tool = ActivateSkillTool::new(registry);
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

impl ActivateSkillToolActor {
    /// Spawns the tool actor with a registry reference.
    ///
    /// This is the preferred way to spawn the activate_skill tool actor.
    pub async fn spawn_with_registry(
        runtime: &mut ActorRuntime,
        registry: Arc<RwLock<SkillRegistry>>,
    ) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<Self>("activate_skill_tool".to_string());

        // Capture the registry in the handler closure
        builder.act_on::<ExecuteToolDirect>(move |actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let registry = Arc::clone(&registry);
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let tool = ActivateSkillTool::new(registry);
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
    use crate::skills::{LoadedSkill, SkillInfo};

    fn create_test_registry() -> Arc<RwLock<SkillRegistry>> {
        let mut registry = SkillRegistry::new();

        registry.add(LoadedSkill {
            info: SkillInfo {
                name: "code-review".to_string(),
                description: "Review code for quality".to_string(),
                path: std::path::PathBuf::from("/skills/code-review.md"),
                tags: vec!["code".to_string(), "review".to_string()],
            },
            content: "# Code Review\n\nReview the code carefully.".to_string(),
            triggers: vec![],
            enabled_by_default: false,
        });

        Arc::new(RwLock::new(registry))
    }

    #[tokio::test]
    async fn activate_existing_skill() {
        let registry = create_test_registry();
        let tool = ActivateSkillTool::new(registry);

        let result = tool
            .execute(json!({
                "name": "code-review"
            }))
            .await
            .unwrap();

        assert_eq!(result["name"], "code-review");
        assert_eq!(result["description"], "Review code for quality");
        assert!(result["instructions"]
            .as_str()
            .unwrap()
            .contains("Code Review"));
        assert_eq!(result["path"], "/skills/code-review.md");
    }

    #[tokio::test]
    async fn activate_nonexistent_skill() {
        let registry = create_test_registry();
        let tool = ActivateSkillTool::new(registry);

        let result = tool
            .execute(json!({
                "name": "nonexistent"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn activate_empty_name_fails() {
        let registry = create_test_registry();
        let tool = ActivateSkillTool::new(registry);

        let result = tool
            .execute(json!({
                "name": ""
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[tokio::test]
    async fn activate_missing_name_fails() {
        let registry = create_test_registry();
        let tool = ActivateSkillTool::new(registry);

        let result = tool.execute(json!({})).await;

        assert!(result.is_err());
    }

    #[test]
    fn config_has_correct_schema() {
        let config = ActivateSkillTool::config();
        assert_eq!(config.definition.name, "activate_skill");
        assert!(config.definition.description.contains("Activate a skill"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["name"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("name")));
    }

    #[test]
    fn validate_args_empty_name() {
        let registry = create_test_registry();
        let tool = ActivateSkillTool::new(registry);

        let result = tool.validate_args(&json!({"name": ""}));
        assert!(result.is_err());
    }

    #[test]
    fn validate_args_valid() {
        let registry = create_test_registry();
        let tool = ActivateSkillTool::new(registry);

        let result = tool.validate_args(&json!({"name": "code-review"}));
        assert!(result.is_ok());
    }
}
