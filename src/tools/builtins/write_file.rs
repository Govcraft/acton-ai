//! Write file built-in tool.
//!
//! Writes content to a file, creating parent directories if needed.

use acton_reactive::prelude::tokio;
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

/// Write file tool executor.
///
/// Writes content to a file, creating parent directories as needed.
#[derive(Debug, Default, Clone)]
pub struct WriteFileTool;

/// Arguments for the write_file tool.
#[derive(Debug, Deserialize)]
struct WriteFileArgs {
    /// Absolute path to the file to write
    path: String,
    /// Content to write to the file
    content: String,
}

impl WriteFileTool {
    /// Creates a new write file tool.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        use crate::messages::ToolDefinition;

        ToolConfig::new(ToolDefinition {
            name: "write_file".to_string(),
            description: "Write content to a file, creating parent directories if needed. Overwrites existing files.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        })
    }
}

impl ToolExecutorTrait for WriteFileTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let args: WriteFileArgs = serde_json::from_value(args)
                .map_err(|e| ToolError::validation_failed("write_file", format!("invalid arguments: {e}")))?;

            let path = Path::new(&args.path);

            // Validate path
            if !path.is_absolute() {
                return Err(ToolError::validation_failed(
                    "write_file",
                    "path must be absolute",
                ));
            }

            // Create parent directories if they don't exist
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|e| ToolError::execution_failed("write_file", format!("failed to create parent directories: {e}")))?;
                }
            }

            // Write the file
            let bytes_written = args.content.len();
            tokio::fs::write(path, &args.content)
                .await
                .map_err(|e| ToolError::execution_failed("write_file", format!("failed to write file: {e}")))?;

            Ok(json!({
                "success": true,
                "path": args.path,
                "bytes_written": bytes_written
            }))
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: WriteFileArgs = serde_json::from_value(args.clone())
            .map_err(|e| ToolError::validation_failed("write_file", format!("invalid arguments: {e}")))?;

        if args.path.is_empty() {
            return Err(ToolError::validation_failed("write_file", "path cannot be empty"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn write_file_basic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");

        let tool = WriteFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "content": "hello world"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["bytes_written"], 11);

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello world");
    }

    #[tokio::test]
    async fn write_file_creates_directories() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join("deep").join("test.txt");

        let tool = WriteFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "content": "nested content"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "nested content");
    }

    #[tokio::test]
    async fn write_file_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");

        std::fs::write(&path, "original content").unwrap();

        let tool = WriteFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "content": "new content"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "new content");
    }

    #[tokio::test]
    async fn write_file_relative_path_rejected() {
        let tool = WriteFileTool::new();
        let result = tool
            .execute(json!({
                "path": "relative/path.txt",
                "content": "test"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("absolute"));
    }

    #[tokio::test]
    async fn write_file_empty_content_allowed() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.txt");

        let tool = WriteFileTool::new();
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "content": ""
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["bytes_written"], 0);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.is_empty());
    }

    #[test]
    fn config_has_correct_schema() {
        let config = WriteFileTool::config();
        assert_eq!(config.definition.name, "write_file");
        assert!(config.definition.description.contains("Write content"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["content"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("path")));
        assert!(schema["required"].as_array().unwrap().contains(&json!("content")));
    }
}
