//! Grep content search built-in tool.
//!
//! Searches file contents with regex support.

use crate::messages::ToolDefinition;
use crate::tools::actor::{ExecuteToolDirect, ToolActor, ToolActorResponse};
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use acton_reactive::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use walkdir::WalkDir;

/// Grep content search tool executor.
///
/// Searches file contents using regex patterns.
#[derive(Debug, Default, Clone)]
pub struct GrepTool;

/// Grep tool actor state.
///
/// This actor wraps the `GrepTool` executor for per-agent tool spawning.
#[acton_actor]
pub struct GrepToolActor;

/// Arguments for the grep tool.
#[derive(Debug, Deserialize)]
struct GrepArgs {
    /// Regex pattern to search for
    pattern: String,
    /// File or directory to search in (default: current directory)
    #[serde(default)]
    path: Option<String>,
    /// File glob pattern to filter files (e.g., "*.rs")
    #[serde(default)]
    glob: Option<String>,
    /// Number of context lines before and after each match
    #[serde(default)]
    context_lines: Option<usize>,
    /// Case insensitive search
    #[serde(default)]
    ignore_case: Option<bool>,
}

/// A single match result.
#[derive(Debug, Serialize)]
struct GrepMatch {
    /// File path
    file: String,
    /// Line number (1-indexed)
    line: usize,
    /// Matched line content
    content: String,
    /// Context lines before the match
    #[serde(skip_serializing_if = "Vec::is_empty")]
    before: Vec<String>,
    /// Context lines after the match
    #[serde(skip_serializing_if = "Vec::is_empty")]
    after: Vec<String>,
}

/// Maximum number of matches to return.
const MAX_MATCHES: usize = 500;

/// Maximum file size to search (10MB).
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

impl GrepTool {
    /// Creates a new grep tool.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        use crate::messages::ToolDefinition;

        ToolConfig::new(ToolDefinition {
            name: "grep".to_string(),
            description: "Search file contents using regex patterns. Returns matching lines with optional context.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default: current directory)"
                    },
                    "glob": {
                        "type": "string",
                        "description": "File pattern to filter files (e.g., '*.rs', '*.{ts,tsx}')"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines before and after each match",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case insensitive search (default: false)"
                    }
                },
                "required": ["pattern"]
            }),
        })
    }

    /// Checks if a filename matches a glob-like pattern.
    fn matches_glob(filename: &str, pattern: &str) -> bool {
        // Simple glob matching for common patterns
        if let Some(ext) = pattern.strip_prefix("*.") {
            filename.ends_with(&format!(".{ext}"))
        } else if pattern.contains('*') {
            // Convert simple glob to regex
            let regex_pattern = pattern
                .replace('.', "\\.")
                .replace('*', ".*")
                .replace('?', ".");
            Regex::new(&format!("^{regex_pattern}$"))
                .map(|re| re.is_match(filename))
                .unwrap_or(false)
        } else {
            filename == pattern
        }
    }

    /// Searches a single file for matches.
    fn search_file(
        path: &Path,
        regex: &Regex,
        context_lines: usize,
    ) -> Result<Vec<GrepMatch>, ToolError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            ToolError::execution_failed("grep", format!("failed to read {}: {e}", path.display()))
        })?;

        let lines: Vec<&str> = content.lines().collect();
        let mut matches = Vec::new();

        for (idx, line) in lines.iter().enumerate() {
            if regex.is_match(line) {
                let before: Vec<String> = if context_lines > 0 && idx > 0 {
                    let start = idx.saturating_sub(context_lines);
                    lines[start..idx].iter().map(|s| (*s).to_string()).collect()
                } else {
                    Vec::new()
                };

                let after: Vec<String> = if context_lines > 0 {
                    let end = (idx + 1 + context_lines).min(lines.len());
                    lines[idx + 1..end]
                        .iter()
                        .map(|s| (*s).to_string())
                        .collect()
                } else {
                    Vec::new()
                };

                matches.push(GrepMatch {
                    file: path.to_string_lossy().to_string(),
                    line: idx + 1,
                    content: (*line).to_string(),
                    before,
                    after,
                });
            }
        }

        Ok(matches)
    }
}

impl ToolExecutorTrait for GrepTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let args: GrepArgs = serde_json::from_value(args).map_err(|e| {
                ToolError::validation_failed("grep", format!("invalid arguments: {e}"))
            })?;

            // Build regex
            let pattern = if args.ignore_case.unwrap_or(false) {
                format!("(?i){}", args.pattern)
            } else {
                args.pattern.clone()
            };

            let regex = Regex::new(&pattern).map_err(|e| {
                ToolError::validation_failed("grep", format!("invalid regex pattern: {e}"))
            })?;

            // Determine search path
            let search_path = match &args.path {
                Some(p) => {
                    let path = Path::new(p);
                    if !path.is_absolute() {
                        return Err(ToolError::validation_failed(
                            "grep",
                            "path must be absolute",
                        ));
                    }
                    if !path.exists() {
                        return Err(ToolError::execution_failed(
                            "grep",
                            format!("path does not exist: {p}"),
                        ));
                    }
                    path.to_path_buf()
                }
                None => std::env::current_dir().map_err(|e| {
                    ToolError::execution_failed(
                        "grep",
                        format!("failed to get current directory: {e}"),
                    )
                })?,
            };

            let context_lines = args.context_lines.unwrap_or(0).min(10);
            let glob_pattern = args.glob.as_deref();

            let mut all_matches = Vec::new();
            let mut files_searched = 0;
            let mut truncated = false;

            // Search single file or directory
            if search_path.is_file() {
                if let Ok(matches) = Self::search_file(&search_path, &regex, context_lines) {
                    all_matches.extend(matches);
                    files_searched = 1;
                }
            } else {
                // Walk directory
                for entry in WalkDir::new(&search_path)
                    .follow_links(false)
                    .into_iter()
                    .filter_map(Result::ok)
                {
                    if all_matches.len() >= MAX_MATCHES {
                        truncated = true;
                        break;
                    }

                    let path = entry.path();

                    // Skip directories
                    if !path.is_file() {
                        continue;
                    }

                    // Skip hidden files and directories
                    if entry
                        .file_name()
                        .to_str()
                        .map(|s| s.starts_with('.'))
                        .unwrap_or(false)
                    {
                        continue;
                    }

                    // Apply glob filter
                    if let Some(glob) = glob_pattern {
                        let filename = entry.file_name().to_string_lossy();
                        if !Self::matches_glob(&filename, glob) {
                            continue;
                        }
                    }

                    // Skip large files
                    if let Ok(metadata) = entry.metadata() {
                        if metadata.len() > MAX_FILE_SIZE {
                            continue;
                        }
                    }

                    // Skip binary files (quick heuristic)
                    if let Ok(content) = std::fs::read(path) {
                        let sample_size = content.len().min(8192);
                        let null_count = content[..sample_size].iter().filter(|&&b| b == 0).count();
                        if null_count > sample_size / 10 {
                            continue;
                        }
                    }

                    if let Ok(matches) = Self::search_file(path, &regex, context_lines) {
                        let remaining = MAX_MATCHES - all_matches.len();
                        if matches.len() > remaining {
                            all_matches.extend(matches.into_iter().take(remaining));
                            truncated = true;
                            break;
                        }
                        all_matches.extend(matches);
                    }
                    files_searched += 1;
                }
            }

            Ok(json!({
                "matches": all_matches,
                "count": all_matches.len(),
                "files_searched": files_searched,
                "truncated": truncated,
                "pattern": args.pattern
            }))
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: GrepArgs = serde_json::from_value(args.clone())
            .map_err(|e| ToolError::validation_failed("grep", format!("invalid arguments: {e}")))?;

        if args.pattern.is_empty() {
            return Err(ToolError::validation_failed(
                "grep",
                "pattern cannot be empty",
            ));
        }

        // Validate regex
        let pattern = if args.ignore_case.unwrap_or(false) {
            format!("(?i){}", args.pattern)
        } else {
            args.pattern
        };

        Regex::new(&pattern).map_err(|e| {
            ToolError::validation_failed("grep", format!("invalid regex pattern: {e}"))
        })?;

        Ok(())
    }
}

impl ToolActor for GrepToolActor {
    fn name() -> &'static str {
        "grep"
    }

    fn definition() -> ToolDefinition {
        GrepTool::config().definition
    }

    async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<Self>("grep_tool".to_string());

        builder.act_on::<ExecuteToolDirect>(|actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let tool = GrepTool::new();
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
    async fn grep_finds_matches() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again",
        )
        .unwrap();

        let tool = GrepTool::new();
        let result = tool
            .execute(json!({
                "pattern": "hello",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 2);

        let matches = result["matches"].as_array().unwrap();
        assert!(matches[0]["content"].as_str().unwrap().contains("hello"));
        assert!(matches[1]["content"].as_str().unwrap().contains("hello"));
    }

    #[tokio::test]
    async fn grep_with_context() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("test.txt"),
            "line 1\nline 2\nmatch here\nline 4\nline 5",
        )
        .unwrap();

        let tool = GrepTool::new();
        let result = tool
            .execute(json!({
                "pattern": "match",
                "path": dir.path().to_str().unwrap(),
                "context_lines": 2
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 1);

        let matches = result["matches"].as_array().unwrap();
        let m = &matches[0];

        assert_eq!(m["line"], 3);
        assert_eq!(m["before"].as_array().unwrap().len(), 2);
        assert_eq!(m["after"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn grep_case_insensitive() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("test.txt"),
            "Hello World\nHELLO WORLD\nhello world",
        )
        .unwrap();

        let tool = GrepTool::new();
        let result = tool
            .execute(json!({
                "pattern": "hello",
                "path": dir.path().to_str().unwrap(),
                "ignore_case": true
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 3);
    }

    #[tokio::test]
    async fn grep_with_glob_filter() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("test.txt"), "fn main() {}").unwrap();

        let tool = GrepTool::new();
        let result = tool
            .execute(json!({
                "pattern": "fn main",
                "path": dir.path().to_str().unwrap(),
                "glob": "*.rs"
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 1);

        let matches = result["matches"].as_array().unwrap();
        assert!(matches[0]["file"].as_str().unwrap().ends_with(".rs"));
    }

    #[tokio::test]
    async fn grep_regex_pattern() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.txt"), "foo123\nbar456\nbaz789").unwrap();

        let tool = GrepTool::new();
        let result = tool
            .execute(json!({
                "pattern": "\\d{3}$",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 3);
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.txt"), "hello world").unwrap();

        let tool = GrepTool::new();
        let result = tool
            .execute(json!({
                "pattern": "xyz",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert_eq!(result["count"], 0);
        assert!(!result["truncated"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn grep_invalid_regex() {
        let tool = GrepTool::new();
        let result = tool
            .execute(json!({
                "pattern": "[invalid"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid regex"));
    }

    #[tokio::test]
    async fn grep_relative_path_rejected() {
        let tool = GrepTool::new();
        let result = tool
            .execute(json!({
                "pattern": "test",
                "path": "relative/path"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("absolute"));
    }

    #[tokio::test]
    async fn grep_path_not_found() {
        let tool = GrepTool::new();
        let result = tool
            .execute(json!({
                "pattern": "test",
                "path": "/nonexistent/path"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn matches_glob_extension() {
        assert!(GrepTool::matches_glob("file.rs", "*.rs"));
        assert!(!GrepTool::matches_glob("file.txt", "*.rs"));
    }

    #[test]
    fn matches_glob_exact() {
        assert!(GrepTool::matches_glob("Cargo.toml", "Cargo.toml"));
        assert!(!GrepTool::matches_glob("other.toml", "Cargo.toml"));
    }

    #[test]
    fn config_has_correct_schema() {
        let config = GrepTool::config();
        assert_eq!(config.definition.name, "grep");
        assert!(config.definition.description.contains("regex"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["pattern"].is_object());
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["glob"].is_object());
        assert!(schema["properties"]["context_lines"].is_object());
        assert!(schema["properties"]["ignore_case"].is_object());
    }
}
