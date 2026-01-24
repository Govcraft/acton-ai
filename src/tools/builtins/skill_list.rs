//! List skills built-in tool.
//!
//! Lists available agent skills with metadata (progressive disclosure).
//! Only available when the `agent-skills` feature is enabled.

use crate::messages::ToolDefinition;
use crate::skills::{SkillInfo, SkillRegistry};
use crate::tools::actor::{ExecuteToolDirect, ToolActor, ToolActorResponse};
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use acton_reactive::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, RwLock};

/// List skills tool executor.
///
/// Lists available skills with optional filtering.
#[derive(Debug)]
pub struct ListSkillsTool {
    /// Reference to the skill registry
    registry: Arc<RwLock<SkillRegistry>>,
}

/// List skills tool actor state.
///
/// This actor wraps the skill registry for per-agent skill listing.
#[acton_actor]
pub struct ListSkillsToolActor {
    /// Reference to the skill registry
    registry: Arc<RwLock<SkillRegistry>>,
}

/// Arguments for the list_skills tool.
#[derive(Debug, Deserialize)]
struct ListSkillsArgs {
    /// Optional filter pattern for skill names
    #[serde(default)]
    filter: Option<String>,
}

/// Result of listing skills.
#[derive(Debug, Serialize)]
struct ListSkillsResult {
    /// List of skills matching the filter
    skills: Vec<SkillSummary>,
    /// Total count of matching skills
    count: usize,
}

/// Summary of a single skill.
#[derive(Debug, Serialize)]
struct SkillSummary {
    /// Skill name
    name: String,
    /// Skill description
    description: String,
    /// Tags associated with the skill
    tags: Vec<String>,
}

impl From<&SkillInfo> for SkillSummary {
    fn from(info: &SkillInfo) -> Self {
        Self {
            name: info.name.clone(),
            description: info.description.clone(),
            tags: info.tags.clone(),
        }
    }
}

impl ListSkillsTool {
    /// Creates a new list skills tool with the given registry.
    #[must_use]
    pub fn new(registry: Arc<RwLock<SkillRegistry>>) -> Self {
        Self { registry }
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        ToolConfig::new(ToolDefinition {
            name: "list_skills".to_string(),
            description: "List available agent skills with their descriptions. Use this to discover what skills are available before activating one.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional filter pattern for skill names (case-insensitive substring match)"
                    }
                }
            }),
        })
    }
}

impl ToolExecutorTrait for ListSkillsTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        let registry = Arc::clone(&self.registry);

        Box::pin(async move {
            let args: ListSkillsArgs = serde_json::from_value(args).map_err(|e| {
                ToolError::validation_failed("list_skills", format!("invalid arguments: {e}"))
            })?;

            let registry_guard = registry.read().map_err(|e| {
                ToolError::execution_failed(
                    "list_skills",
                    format!("failed to acquire registry lock: {e}"),
                )
            })?;

            let skills: Vec<SkillSummary> = registry_guard
                .list()
                .iter()
                .filter(|info| {
                    args.filter.as_ref().is_none_or(|pattern| {
                        let pattern_lower = pattern.to_lowercase();
                        info.name.to_lowercase().contains(&pattern_lower)
                            || info.description.to_lowercase().contains(&pattern_lower)
                    })
                })
                .map(SkillSummary::from)
                .collect();

            let count = skills.len();

            Ok(json!(ListSkillsResult { skills, count }))
        })
    }

    fn validate_args(&self, _args: &Value) -> Result<(), ToolError> {
        // No required arguments, validation always passes
        Ok(())
    }
}

impl ToolActor for ListSkillsToolActor {
    fn name() -> &'static str {
        "list_skills"
    }

    fn definition() -> ToolDefinition {
        ListSkillsTool::config().definition
    }

    async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        // This is a placeholder - the actual spawn requires a registry reference
        // Use spawn_with_registry instead
        let mut builder = runtime.new_actor_with_name::<Self>("list_skills_tool".to_string());

        builder.act_on::<ExecuteToolDirect>(|actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let registry = Arc::clone(&actor.model.registry);
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let tool = ListSkillsTool::new(registry);
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

impl ListSkillsToolActor {
    /// Spawns the tool actor with a registry reference.
    ///
    /// This is the preferred way to spawn the list_skills tool actor.
    pub async fn spawn_with_registry(
        runtime: &mut ActorRuntime,
        registry: Arc<RwLock<SkillRegistry>>,
    ) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<Self>("list_skills_tool".to_string());

        // Capture the registry in the handler closure
        builder.act_on::<ExecuteToolDirect>(move |actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let registry = Arc::clone(&registry);
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let tool = ListSkillsTool::new(registry);
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
    use crate::skills::LoadedSkill;

    fn create_test_registry() -> Arc<RwLock<SkillRegistry>> {
        let mut registry = SkillRegistry::new();

        registry.add(LoadedSkill {
            info: SkillInfo {
                name: "code-review".to_string(),
                description: "Review code for quality".to_string(),
                path: std::path::PathBuf::from("/skills/code-review.md"),
                tags: vec!["code".to_string(), "review".to_string()],
            },
            content: "# Code Review\n\nInstructions...".to_string(),
            triggers: vec![],
            enabled_by_default: false,
        });

        registry.add(LoadedSkill {
            info: SkillInfo {
                name: "documentation".to_string(),
                description: "Generate documentation".to_string(),
                path: std::path::PathBuf::from("/skills/documentation.md"),
                tags: vec!["docs".to_string()],
            },
            content: "# Documentation\n\nInstructions...".to_string(),
            triggers: vec![],
            enabled_by_default: false,
        });

        Arc::new(RwLock::new(registry))
    }

    #[tokio::test]
    async fn list_all_skills() {
        let registry = create_test_registry();
        let tool = ListSkillsTool::new(registry);

        let result = tool.execute(json!({})).await.unwrap();

        assert_eq!(result["count"], 2);
        let skills = result["skills"].as_array().unwrap();
        assert_eq!(skills.len(), 2);
    }

    #[tokio::test]
    async fn list_skills_with_filter() {
        let registry = create_test_registry();
        let tool = ListSkillsTool::new(registry);

        let result = tool
            .execute(json!({
                "filter": "code"
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 1);
        let skills = result["skills"].as_array().unwrap();
        assert_eq!(skills[0]["name"], "code-review");
    }

    #[tokio::test]
    async fn list_skills_filter_no_match() {
        let registry = create_test_registry();
        let tool = ListSkillsTool::new(registry);

        let result = tool
            .execute(json!({
                "filter": "nonexistent"
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 0);
        let skills = result["skills"].as_array().unwrap();
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn list_skills_filter_case_insensitive() {
        let registry = create_test_registry();
        let tool = ListSkillsTool::new(registry);

        let result = tool
            .execute(json!({
                "filter": "CODE"
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 1);
    }

    #[test]
    fn config_has_correct_schema() {
        let config = ListSkillsTool::config();
        assert_eq!(config.definition.name, "list_skills");
        assert!(config.definition.description.contains("List available"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["filter"].is_object());
    }
}
