//! Read file built-in tool.
//!
//! Reads file contents and returns them with line numbers.

use acton_reactive::prelude::tokio;
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

/// Read file tool executor.
///
/// Reads file contents with optional offset and limit,
/// returning content with line numbers.
#[derive(Debug, Default, Clone)]
pub struct ReadFileTool;

/// Arguments for the read_file tool.
#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    /// Absolute path to the file to read
    path: String,
    /// Line number to start from (1-indexed)
    #[serde(default)]
    offset: Option<usize>,
    /// Maximum number of lines to read
    #[serde(default)]
    limit: Option<usize>,
}

/// Result of reading a file.
#[derive(Debug, Serialize)]
struct ReadFileResult {
    /// File content with line numbers
    content: String,
    /// Total number of lines in the file
    total_lines: usize,
    /// Starting line number (1-indexed)
    start_line: usize,
    /// Ending line number (1-indexed, inclusive)
    end_line: usize,
    /// Whether the file was truncated
    truncated: bool,
}

impl ReadFileTool {
    /// Creates a new read file tool.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        use crate::messages::ToolDefinition;

        ToolConfig::new(ToolDefinition {
            name: "read_file".to_string(),
            description: "Read file contents with optional line offset and limit. Returns content with line numbers.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start from (1-indexed, default: 1)",
                        "minimum": 1
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 2000)",
                        "minimum": 1
                    }
                },
                "required": ["path"]
            }),
        })
    }

    /// Reads and formats file content with line numbers.
    async fn read_with_line_numbers(
        path: &Path,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<ReadFileResult, ToolError> {
        // Default values
        let start_line = offset.unwrap_or(1).max(1);
        let max_lines = limit.unwrap_or(2000);
        const MAX_LINE_LENGTH: usize = 2000;

        // Read file content
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::execution_failed("read_file", format!("failed to read file: {e}")))?;

        let all_lines: Vec<&str> = content.lines().collect();
        let total_lines = all_lines.len();

        // Calculate range (convert to 0-indexed)
        let start_idx = (start_line - 1).min(total_lines);
        let end_idx = (start_idx + max_lines).min(total_lines);

        // Format lines with numbers
        let mut formatted = String::new();
        let line_num_width = end_idx.to_string().len().max(4);

        for (idx, line) in all_lines.iter().enumerate().skip(start_idx).take(end_idx - start_idx) {
            let line_num = idx + 1;
            let truncated_line = if line.len() > MAX_LINE_LENGTH {
                format!("{}...", &line[..MAX_LINE_LENGTH])
            } else {
                line.to_string()
            };
            formatted.push_str(&format!("{:>width$}\t{}\n", line_num, truncated_line, width = line_num_width));
        }

        let truncated = end_idx < total_lines;
        let actual_end = if end_idx > start_idx { end_idx } else { start_line };

        Ok(ReadFileResult {
            content: formatted,
            total_lines,
            start_line,
            end_line: actual_end,
            truncated,
        })
    }
}

impl ToolExecutorTrait for ReadFileTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let args: ReadFileArgs = serde_json::from_value(args)
                .map_err(|e| ToolError::validation_failed("read_file", format!("invalid arguments: {e}")))?;

            let path = Path::new(&args.path);

            // Validate path
            if !path.is_absolute() {
                return Err(ToolError::validation_failed(
                    "read_file",
                    "path must be absolute",
                ));
            }

            if !path.exists() {
                return Err(ToolError::execution_failed(
                    "read_file",
                    format!("file does not exist: {}", args.path),
                ));
            }

            if !path.is_file() {
                return Err(ToolError::execution_failed(
                    "read_file",
                    format!("path is not a file: {}", args.path),
                ));
            }

            // Check if file is likely binary
            if let Ok(content) = tokio::fs::read(path).await {
                let sample_size = content.len().min(8192);
                let null_count = content[..sample_size].iter().filter(|&&b| b == 0).count();
                if null_count > sample_size / 10 {
                    return Err(ToolError::execution_failed(
                        "read_file",
                        "file appears to be binary; use a different tool for binary files",
                    ));
                }
            }

            let result = Self::read_with_line_numbers(path, args.offset, args.limit).await?;

            Ok(json!({
                "content": result.content,
                "total_lines": result.total_lines,
                "start_line": result.start_line,
                "end_line": result.end_line,
                "truncated": result.truncated
            }))
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: ReadFileArgs = serde_json::from_value(args.clone())
            .map_err(|e| ToolError::validation_failed("read_file", format!("invalid arguments: {e}")))?;

        if args.path.is_empty() {
            return Err(ToolError::validation_failed("read_file", "path cannot be empty"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn read_file_basic() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "line 1").unwrap();
        writeln!(file, "line 2").unwrap();
        writeln!(file, "line 3").unwrap();

        let tool = ReadFileTool::new();
        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(result["content"].as_str().unwrap().contains("line 1"));
        assert!(result["content"].as_str().unwrap().contains("line 2"));
        assert!(result["content"].as_str().unwrap().contains("line 3"));
        assert_eq!(result["total_lines"], 3);
        assert!(!result["truncated"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn read_file_with_offset() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(file, "line {i}").unwrap();
        }

        let tool = ReadFileTool::new();
        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap(),
                "offset": 5
            }))
            .await
            .unwrap();

        let content = result["content"].as_str().unwrap();
        // Note: We check for "line 1\n" to avoid matching "line 10" which contains "line 1"
        assert!(!content.contains("line 1\n"));
        assert!(!content.contains("line 4\n"));
        assert!(content.contains("line 5"));
        assert!(content.contains("line 10"));
        assert_eq!(result["start_line"], 5);
    }

    #[tokio::test]
    async fn read_file_with_limit() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(file, "line {i}").unwrap();
        }

        let tool = ReadFileTool::new();
        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap(),
                "limit": 3
            }))
            .await
            .unwrap();

        let content = result["content"].as_str().unwrap();
        assert!(content.contains("line 1"));
        assert!(content.contains("line 3"));
        assert!(!content.contains("line 4"));
        assert!(result["truncated"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn read_file_not_found() {
        let tool = ReadFileTool::new();
        let result = tool
            .execute(json!({
                "path": "/nonexistent/file.txt"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn read_file_relative_path_rejected() {
        let tool = ReadFileTool::new();
        let result = tool
            .execute(json!({
                "path": "relative/path.txt"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("absolute"));
    }

    #[test]
    fn config_has_correct_schema() {
        let config = ReadFileTool::config();
        assert_eq!(config.definition.name, "read_file");
        assert!(config.definition.description.contains("Read file"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["offset"].is_object());
        assert!(schema["properties"]["limit"].is_object());
    }
}
