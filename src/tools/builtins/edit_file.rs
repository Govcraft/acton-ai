//! Edit file built-in tool.
//!
//! Makes targeted string replacements in files.

use crate::messages::ToolDefinition;
use crate::tools::actor::{ExecuteToolDirect, ToolActor, ToolActorResponse};
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use acton_reactive::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

/// Edit file tool executor.
///
/// Makes targeted string replacements in files.
#[derive(Debug, Default, Clone)]
pub struct EditFileTool;

/// Edit file tool actor state.
///
/// This actor wraps the `EditFileTool` executor for per-agent tool spawning.
#[acton_actor]
pub struct EditFileToolActor;

/// Arguments for the edit_file tool.
#[derive(Debug, Deserialize)]
struct EditFileArgs {
    /// Absolute path to the file to edit
    path: String,
    /// Exact string to find and replace
    old_string: String,
    /// Replacement string
    new_string: String,
    /// Replace all occurrences (default: false)
    #[serde(default)]
    replace_all: bool,
}

impl EditFileTool {
    /// Creates a new edit file tool.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        use crate::messages::ToolDefinition;

        ToolConfig::new(ToolDefinition {
            name: "edit_file".to_string(),
            description: "Make targeted string replacements in a file. The old_string must be found exactly once unless replace_all is true.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false, which requires exactly one match)"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        })
    }

    /// Generates a simple diff-style output.
    fn generate_diff(old_content: &str, new_content: &str, path: &str) -> String {
        let mut diff = String::new();
        diff.push_str(&format!("--- {path}\n"));
        diff.push_str(&format!("+++ {path}\n"));

        let old_lines: Vec<&str> = old_content.lines().collect();
        let new_lines: Vec<&str> = new_content.lines().collect();

        // Simple line-by-line diff
        let mut old_idx = 0;
        let mut new_idx = 0;

        while old_idx < old_lines.len() || new_idx < new_lines.len() {
            let old_line = old_lines.get(old_idx);
            let new_line = new_lines.get(new_idx);

            match (old_line, new_line) {
                (Some(o), Some(n)) if o == n => {
                    diff.push_str(&format!(" {o}\n"));
                    old_idx += 1;
                    new_idx += 1;
                }
                (Some(o), Some(n)) => {
                    diff.push_str(&format!("-{o}\n"));
                    diff.push_str(&format!("+{n}\n"));
                    old_idx += 1;
                    new_idx += 1;
                }
                (Some(o), None) => {
                    diff.push_str(&format!("-{o}\n"));
                    old_idx += 1;
                }
                (None, Some(n)) => {
                    diff.push_str(&format!("+{n}\n"));
                    new_idx += 1;
                }
                (None, None) => break,
            }
        }

        diff
    }
}

impl ToolExecutorTrait for EditFileTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let args: EditFileArgs = serde_json::from_value(args).map_err(|e| {
                ToolError::validation_failed("edit_file", format!("invalid arguments: {e}"))
            })?;

            let path = Path::new(&args.path);

            // Validate path
            if !path.is_absolute() {
                return Err(ToolError::validation_failed(
                    "edit_file",
                    "path must be absolute",
                ));
            }

            if !path.exists() {
                return Err(ToolError::execution_failed(
                    "edit_file",
                    format!("file does not exist: {}", args.path),
                ));
            }

            if !path.is_file() {
                return Err(ToolError::execution_failed(
                    "edit_file",
                    format!("path is not a file: {}", args.path),
                ));
            }

            // Validate that old_string != new_string
            if args.old_string == args.new_string {
                return Err(ToolError::validation_failed(
                    "edit_file",
                    "old_string and new_string must be different",
                ));
            }

            // Read the file
            let content = tokio::fs::read_to_string(path).await.map_err(|e| {
                ToolError::execution_failed("edit_file", format!("failed to read file: {e}"))
            })?;

            // Count occurrences
            let match_count = content.matches(&args.old_string).count();

            if match_count == 0 {
                return Err(ToolError::execution_failed(
                    "edit_file",
                    "old_string not found in file; verify the exact content to replace",
                ));
            }

            if !args.replace_all && match_count > 1 {
                return Err(ToolError::execution_failed(
                    "edit_file",
                    format!(
                        "old_string found {match_count} times; use replace_all: true to replace all, or provide more context to make it unique"
                    ),
                ));
            }

            // Perform replacement
            let new_content = if args.replace_all {
                content.replace(&args.old_string, &args.new_string)
            } else {
                content.replacen(&args.old_string, &args.new_string, 1)
            };

            // Generate diff
            let diff = Self::generate_diff(&content, &new_content, &args.path);

            // Write the file
            tokio::fs::write(path, &new_content).await.map_err(|e| {
                ToolError::execution_failed("edit_file", format!("failed to write file: {e}"))
            })?;

            Ok(json!({
                "success": true,
                "path": args.path,
                "replacements": match_count,
                "diff": diff
            }))
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: EditFileArgs = serde_json::from_value(args.clone()).map_err(|e| {
            ToolError::validation_failed("edit_file", format!("invalid arguments: {e}"))
        })?;

        if args.path.is_empty() {
            return Err(ToolError::validation_failed(
                "edit_file",
                "path cannot be empty",
            ));
        }

        if args.old_string.is_empty() {
            return Err(ToolError::validation_failed(
                "edit_file",
                "old_string cannot be empty",
            ));
        }

        Ok(())
    }
}

impl ToolActor for EditFileToolActor {
    fn name() -> &'static str {
        "edit_file"
    }

    fn definition() -> ToolDefinition {
        EditFileTool::config().definition
    }

    async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<Self>("edit_file_tool".to_string());

        builder.act_on::<ExecuteToolDirect>(|actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let tool = EditFileTool::new();
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
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn edit_file_single_replacement() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello world").unwrap();

        let tool = EditFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "world",
                "new_string": "rust"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["replacements"], 1);

        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello rust");
    }

    #[tokio::test]
    async fn edit_file_replace_all() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "foo foo foo").unwrap();

        let tool = EditFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": true
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["replacements"], 3);

        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "bar bar bar");
    }

    #[tokio::test]
    async fn edit_file_multiline_replacement() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "line1\nold content\nline3").unwrap();

        let tool = EditFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "old content",
                "new_string": "new content\nwith multiple lines"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());

        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "line1\nnew content\nwith multiple lines\nline3");
    }

    #[tokio::test]
    async fn edit_file_ambiguous_without_replace_all() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "foo foo foo").unwrap();

        let tool = EditFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "foo",
                "new_string": "bar"
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("3 times"));
        assert!(err.contains("replace_all"));
    }

    #[tokio::test]
    async fn edit_file_not_found_in_content() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello world").unwrap();

        let tool = EditFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "xyz",
                "new_string": "abc"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn edit_file_same_string_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello world").unwrap();

        let tool = EditFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "hello"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("different"));
    }

    #[tokio::test]
    async fn edit_file_not_exists() {
        let tool = EditFileTool::new();
        let result = tool
            .execute(json!({
                "path": "/nonexistent/file.txt",
                "old_string": "old",
                "new_string": "new"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn edit_file_relative_path_rejected() {
        let tool = EditFileTool::new();
        let result = tool
            .execute(json!({
                "path": "relative/path.txt",
                "old_string": "old",
                "new_string": "new"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("absolute"));
    }

    #[tokio::test]
    async fn edit_file_generates_diff() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "line1\nold\nline3").unwrap();

        let tool = EditFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "old",
                "new_string": "new"
            }))
            .await
            .unwrap();

        let diff = result["diff"].as_str().unwrap();
        assert!(diff.contains("-old"));
        assert!(diff.contains("+new"));
    }

    #[test]
    fn config_has_correct_schema() {
        let config = EditFileTool::config();
        assert_eq!(config.definition.name, "edit_file");
        assert!(config.definition.description.contains("replacement"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["old_string"].is_object());
        assert!(schema["properties"]["new_string"].is_object());
        assert!(schema["properties"]["replace_all"].is_object());
    }
}
