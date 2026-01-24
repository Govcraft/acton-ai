//! List directory built-in tool.
//!
//! Lists directory contents with metadata.

use acton_reactive::prelude::tokio;
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

/// List directory tool executor.
///
/// Lists directory contents with file type, size, and modification time.
#[derive(Debug, Default, Clone)]
pub struct ListDirectoryTool;

/// Arguments for the list_directory tool.
#[derive(Debug, Deserialize)]
struct ListDirectoryArgs {
    /// Directory path to list
    path: String,
}

/// Information about a directory entry.
#[derive(Debug, Serialize)]
struct DirEntry {
    /// Entry name
    name: String,
    /// Entry type: "file", "dir", or "symlink"
    entry_type: String,
    /// Size in bytes (for files)
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
    /// Last modified timestamp (ISO 8601)
    #[serde(skip_serializing_if = "Option::is_none")]
    modified: Option<String>,
}

impl ListDirectoryTool {
    /// Creates a new list directory tool.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        use crate::messages::ToolDefinition;

        ToolConfig::new(ToolDefinition {
            name: "list_directory".to_string(),
            description: "List directory contents with metadata (type, size, modified time).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list"
                    }
                },
                "required": ["path"]
            }),
        })
    }

    /// Formats a system time as ISO 8601.
    fn format_time(time: std::time::SystemTime) -> Option<String> {
        let datetime = chrono::DateTime::<chrono::Utc>::from(time);
        Some(datetime.format("%Y-%m-%dT%H:%M:%SZ").to_string())
    }
}

impl ToolExecutorTrait for ListDirectoryTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let args: ListDirectoryArgs = serde_json::from_value(args)
                .map_err(|e| ToolError::validation_failed("list_directory", format!("invalid arguments: {e}")))?;

            let path = Path::new(&args.path);

            // Validate path
            if !path.is_absolute() {
                return Err(ToolError::validation_failed(
                    "list_directory",
                    "path must be absolute",
                ));
            }

            if !path.exists() {
                return Err(ToolError::execution_failed(
                    "list_directory",
                    format!("directory does not exist: {}", args.path),
                ));
            }

            if !path.is_dir() {
                return Err(ToolError::execution_failed(
                    "list_directory",
                    format!("path is not a directory: {}", args.path),
                ));
            }

            // Read directory entries
            let mut read_dir = tokio::fs::read_dir(path)
                .await
                .map_err(|e| ToolError::execution_failed("list_directory", format!("failed to read directory: {e}")))?;

            let mut entries = Vec::new();

            while let Some(entry) = read_dir.next_entry().await.map_err(|e| {
                ToolError::execution_failed("list_directory", format!("failed to read entry: {e}"))
            })? {
                let name = entry.file_name().to_string_lossy().to_string();

                let metadata = entry.metadata().await.ok();
                let file_type = entry.file_type().await.ok();

                let entry_type = match file_type {
                    Some(ft) if ft.is_dir() => "dir",
                    Some(ft) if ft.is_symlink() => "symlink",
                    _ => "file",
                };

                let size = metadata.as_ref().and_then(|m| {
                    if m.is_file() {
                        Some(m.len())
                    } else {
                        None
                    }
                });

                let modified = metadata
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .and_then(Self::format_time);

                entries.push(DirEntry {
                    name,
                    entry_type: entry_type.to_string(),
                    size,
                    modified,
                });
            }

            // Sort by name
            entries.sort_by(|a, b| a.name.cmp(&b.name));

            Ok(json!({
                "path": args.path,
                "entries": entries,
                "count": entries.len()
            }))
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: ListDirectoryArgs = serde_json::from_value(args.clone())
            .map_err(|e| ToolError::validation_failed("list_directory", format!("invalid arguments: {e}")))?;

        if args.path.is_empty() {
            return Err(ToolError::validation_failed("list_directory", "path cannot be empty"));
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
    async fn list_directory_basic() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("file1.txt"), "content1").unwrap();
        fs::write(dir.path().join("file2.txt"), "content2").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = ListDirectoryTool::new();
        let result = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 3);

        let entries = result["entries"].as_array().unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e["name"].as_str().unwrap()).collect();

        assert!(names.contains(&"file1.txt"));
        assert!(names.contains(&"file2.txt"));
        assert!(names.contains(&"subdir"));
    }

    #[tokio::test]
    async fn list_directory_shows_types() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("file.txt"), "content").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = ListDirectoryTool::new();
        let result = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        let entries = result["entries"].as_array().unwrap();

        let file_entry = entries.iter().find(|e| e["name"] == "file.txt").unwrap();
        let dir_entry = entries.iter().find(|e| e["name"] == "subdir").unwrap();

        assert_eq!(file_entry["entry_type"], "file");
        assert_eq!(dir_entry["entry_type"], "dir");
    }

    #[tokio::test]
    async fn list_directory_shows_size() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("file.txt"), "hello").unwrap();

        let tool = ListDirectoryTool::new();
        let result = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        let entries = result["entries"].as_array().unwrap();
        let file_entry = entries.iter().find(|e| e["name"] == "file.txt").unwrap();

        assert_eq!(file_entry["size"], 5);
    }

    #[tokio::test]
    async fn list_directory_empty() {
        let dir = TempDir::new().unwrap();

        let tool = ListDirectoryTool::new();
        let result = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 0);
        assert!(result["entries"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_directory_not_found() {
        let tool = ListDirectoryTool::new();
        let result = tool
            .execute(json!({
                "path": "/nonexistent/directory"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn list_directory_not_a_directory() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("file.txt");
        fs::write(&file_path, "content").unwrap();

        let tool = ListDirectoryTool::new();
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap()
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a directory"));
    }

    #[tokio::test]
    async fn list_directory_relative_path_rejected() {
        let tool = ListDirectoryTool::new();
        let result = tool
            .execute(json!({
                "path": "relative/path"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("absolute"));
    }

    #[test]
    fn config_has_correct_schema() {
        let config = ListDirectoryTool::config();
        assert_eq!(config.definition.name, "list_directory");
        assert!(config.definition.description.contains("List directory"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("path")));
    }
}
