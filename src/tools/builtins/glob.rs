//! Glob pattern matching built-in tool.
//!
//! Finds files matching a glob pattern.

use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use glob::glob as glob_match;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

/// Glob pattern matching tool executor.
///
/// Finds files matching a glob pattern like `**/*.rs`.
#[derive(Debug, Default, Clone)]
pub struct GlobTool;

/// Arguments for the glob tool.
#[derive(Debug, Deserialize)]
struct GlobArgs {
    /// Glob pattern to match (e.g., "**/*.rs")
    pattern: String,
    /// Base directory (default: current working directory)
    #[serde(default)]
    path: Option<String>,
}

/// Maximum number of results to return.
const MAX_RESULTS: usize = 1000;

impl GlobTool {
    /// Creates a new glob tool.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        use crate::messages::ToolDefinition;

        ToolConfig::new(ToolDefinition {
            name: "glob".to_string(),
            description: "Find files matching a glob pattern (e.g., **/*.rs). Returns up to 1000 matching paths.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match (e.g., '**/*.rs', 'src/**/*.ts')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search in (default: current working directory)"
                    }
                },
                "required": ["pattern"]
            }),
        })
    }
}

impl ToolExecutorTrait for GlobTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let args: GlobArgs = serde_json::from_value(args)
                .map_err(|e| ToolError::validation_failed("glob", format!("invalid arguments: {e}")))?;

            // Determine base path
            let base_path = match &args.path {
                Some(p) => {
                    let path = Path::new(p);
                    if !path.is_absolute() {
                        return Err(ToolError::validation_failed(
                            "glob",
                            "path must be absolute",
                        ));
                    }
                    if !path.exists() {
                        return Err(ToolError::execution_failed(
                            "glob",
                            format!("directory does not exist: {p}"),
                        ));
                    }
                    path.to_path_buf()
                }
                None => std::env::current_dir().map_err(|e| {
                    ToolError::execution_failed("glob", format!("failed to get current directory: {e}"))
                })?,
            };

            // Construct the full pattern
            let full_pattern = base_path.join(&args.pattern);
            let pattern_str = full_pattern.to_string_lossy();

            // Execute glob
            let paths = glob_match(&pattern_str)
                .map_err(|e| ToolError::validation_failed("glob", format!("invalid glob pattern: {e}")))?;

            let mut matches: Vec<String> = Vec::new();
            let mut truncated = false;

            for entry in paths {
                if matches.len() >= MAX_RESULTS {
                    truncated = true;
                    break;
                }

                match entry {
                    Ok(path) => {
                        matches.push(path.to_string_lossy().to_string());
                    }
                    Err(e) => {
                        // Log but continue on permission errors, etc.
                        tracing::debug!("glob entry error: {e}");
                    }
                }
            }

            // Sort results for deterministic output
            matches.sort();

            Ok(json!({
                "matches": matches,
                "count": matches.len(),
                "truncated": truncated,
                "pattern": args.pattern,
                "base_path": base_path.to_string_lossy()
            }))
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: GlobArgs = serde_json::from_value(args.clone())
            .map_err(|e| ToolError::validation_failed("glob", format!("invalid arguments: {e}")))?;

        if args.pattern.is_empty() {
            return Err(ToolError::validation_failed("glob", "pattern cannot be empty"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn glob_finds_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("file1.txt"), "content").unwrap();
        fs::write(dir.path().join("file2.txt"), "content").unwrap();
        fs::write(dir.path().join("file.rs"), "content").unwrap();

        let tool = GlobTool::new();
        let result = tool
            .execute(json!({
                "pattern": "*.txt",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 2);

        let matches = result["matches"].as_array().unwrap();
        let match_strs: Vec<&str> = matches.iter().map(|m| m.as_str().unwrap()).collect();

        assert!(match_strs.iter().any(|m| m.ends_with("file1.txt")));
        assert!(match_strs.iter().any(|m| m.ends_with("file2.txt")));
        assert!(!match_strs.iter().any(|m| m.ends_with("file.rs")));
    }

    #[tokio::test]
    async fn glob_recursive() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();

        fs::write(dir.path().join("top.txt"), "content").unwrap();
        fs::write(subdir.join("nested.txt"), "content").unwrap();

        let tool = GlobTool::new();
        let result = tool
            .execute(json!({
                "pattern": "**/*.txt",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 2);

        let matches = result["matches"].as_array().unwrap();
        let match_strs: Vec<&str> = matches.iter().map(|m| m.as_str().unwrap()).collect();

        assert!(match_strs.iter().any(|m| m.ends_with("top.txt")));
        assert!(match_strs.iter().any(|m| m.ends_with("nested.txt")));
    }

    #[tokio::test]
    async fn glob_no_matches() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("file.txt"), "content").unwrap();

        let tool = GlobTool::new();
        let result = tool
            .execute(json!({
                "pattern": "*.rs",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 0);
        assert!(result["matches"].as_array().unwrap().is_empty());
        assert!(!result["truncated"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn glob_invalid_pattern() {
        let dir = TempDir::new().unwrap();

        let tool = GlobTool::new();
        let result = tool
            .execute(json!({
                "pattern": "[invalid",
                "path": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid glob pattern"));
    }

    #[tokio::test]
    async fn glob_relative_path_rejected() {
        let tool = GlobTool::new();
        let result = tool
            .execute(json!({
                "pattern": "*.txt",
                "path": "relative/path"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("absolute"));
    }

    #[tokio::test]
    async fn glob_directory_not_found() {
        let tool = GlobTool::new();
        let result = tool
            .execute(json!({
                "pattern": "*.txt",
                "path": "/nonexistent/directory"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn config_has_correct_schema() {
        let config = GlobTool::config();
        assert_eq!(config.definition.name, "glob");
        assert!(config.definition.description.contains("glob pattern"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["pattern"].is_object());
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("pattern")));
    }
}
