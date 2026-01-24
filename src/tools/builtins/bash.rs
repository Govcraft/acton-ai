//! Bash command execution built-in tool.
//!
//! Executes shell commands with timeout and output capture.

use acton_reactive::prelude::tokio;
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

/// Bash command execution tool executor.
///
/// Executes shell commands and captures output.
#[derive(Debug, Clone)]
pub struct BashTool {
    /// Default timeout in seconds
    default_timeout: u64,
    /// Maximum allowed timeout in seconds
    max_timeout: u64,
}

impl Default for BashTool {
    fn default() -> Self {
        Self {
            default_timeout: 120,
            max_timeout: 600,
        }
    }
}

/// Arguments for the bash tool.
#[derive(Debug, Deserialize)]
struct BashArgs {
    /// Command to execute
    command: String,
    /// Timeout in seconds (default: 120, max: 600)
    #[serde(default)]
    timeout: Option<u64>,
    /// Working directory (default: current directory)
    #[serde(default)]
    cwd: Option<String>,
}

/// Maximum output size to capture (1MB).
const MAX_OUTPUT_SIZE: usize = 1024 * 1024;

impl BashTool {
    /// Creates a new bash tool with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new bash tool with custom timeout settings.
    #[must_use]
    pub fn with_timeouts(default_timeout: u64, max_timeout: u64) -> Self {
        Self {
            default_timeout,
            max_timeout,
        }
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        use crate::messages::ToolDefinition;

        ToolConfig::new(ToolDefinition {
            name: "bash".to_string(),
            description: "Execute a shell command and capture its output. Use for system operations, git commands, build tools, etc.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 120, max: 600)",
                        "minimum": 1,
                        "maximum": 600
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command (default: current directory)"
                    }
                },
                "required": ["command"]
            }),
        })
        .with_sandbox(true) // Mark as requiring sandbox by default
    }

    /// Truncates output if it exceeds the maximum size.
    fn truncate_output(output: &str) -> (String, bool) {
        if output.len() > MAX_OUTPUT_SIZE {
            let truncated = &output[..MAX_OUTPUT_SIZE];
            // Try to truncate at a line boundary
            let last_newline = truncated.rfind('\n').unwrap_or(MAX_OUTPUT_SIZE);
            (
                format!(
                    "{}\n\n... (output truncated, {} bytes total)",
                    &output[..last_newline],
                    output.len()
                ),
                true,
            )
        } else {
            (output.to_string(), false)
        }
    }
}

impl ToolExecutorTrait for BashTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        let default_timeout = self.default_timeout;
        let max_timeout = self.max_timeout;

        Box::pin(async move {
            let args: BashArgs = serde_json::from_value(args)
                .map_err(|e| ToolError::validation_failed("bash", format!("invalid arguments: {e}")))?;

            // Validate empty command early
            if args.command.is_empty() {
                return Err(ToolError::validation_failed("bash", "command cannot be empty"));
            }

            // Validate and set timeout
            let timeout_secs = args.timeout.unwrap_or(default_timeout).min(max_timeout);

            // Validate working directory if provided
            if let Some(ref cwd) = args.cwd {
                let path = Path::new(cwd);
                if !path.is_absolute() {
                    return Err(ToolError::validation_failed(
                        "bash",
                        "cwd must be an absolute path",
                    ));
                }
                if !path.exists() {
                    return Err(ToolError::execution_failed(
                        "bash",
                        format!("working directory does not exist: {cwd}"),
                    ));
                }
                if !path.is_dir() {
                    return Err(ToolError::execution_failed(
                        "bash",
                        format!("cwd is not a directory: {cwd}"),
                    ));
                }
            }

            // Build the command
            let mut cmd = Command::new("bash");
            cmd.arg("-c")
                .arg(&args.command)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .stdin(Stdio::null());

            // Set working directory if provided
            if let Some(ref cwd) = args.cwd {
                cmd.current_dir(cwd);
            }

            // Spawn the process
            let mut child = cmd.spawn().map_err(|e| {
                ToolError::execution_failed("bash", format!("failed to spawn process: {e}"))
            })?;

            // Wait for completion with timeout
            let timeout_duration = Duration::from_secs(timeout_secs);
            let result = timeout(timeout_duration, async {
                // Read stdout and stderr
                let mut stdout_buf = Vec::new();
                let mut stderr_buf = Vec::new();

                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_end(&mut stdout_buf).await;
                }

                if let Some(mut stderr) = child.stderr.take() {
                    let _ = stderr.read_to_end(&mut stderr_buf).await;
                }

                let status = child.wait().await?;

                Ok::<_, std::io::Error>((status, stdout_buf, stderr_buf))
            })
            .await;

            match result {
                Ok(Ok((status, stdout_buf, stderr_buf))) => {
                    let stdout = String::from_utf8_lossy(&stdout_buf);
                    let stderr = String::from_utf8_lossy(&stderr_buf);

                    let (stdout_str, stdout_truncated) = Self::truncate_output(&stdout);
                    let (stderr_str, stderr_truncated) = Self::truncate_output(&stderr);

                    let exit_code = status.code().unwrap_or(-1);

                    Ok(json!({
                        "exit_code": exit_code,
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                        "success": status.success(),
                        "truncated": stdout_truncated || stderr_truncated
                    }))
                }
                Ok(Err(e)) => Err(ToolError::execution_failed(
                    "bash",
                    format!("process error: {e}"),
                )),
                Err(_) => {
                    // Timeout - try to kill the process
                    let _ = child.kill().await;
                    Err(ToolError::timeout(
                        "bash",
                        Duration::from_secs(timeout_secs),
                    ))
                }
            }
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: BashArgs = serde_json::from_value(args.clone())
            .map_err(|e| ToolError::validation_failed("bash", format!("invalid arguments: {e}")))?;

        if args.command.is_empty() {
            return Err(ToolError::validation_failed("bash", "command cannot be empty"));
        }

        // Basic safety check for obviously dangerous commands
        let dangerous_patterns = ["rm -rf /", ":(){ :|:& };:", "mkfs.", "dd if=/dev/zero of=/dev/"];
        for pattern in &dangerous_patterns {
            if args.command.contains(pattern) {
                return Err(ToolError::validation_failed(
                    "bash",
                    format!("command contains dangerous pattern: {pattern}"),
                ));
            }
        }

        Ok(())
    }

    fn requires_sandbox(&self) -> bool {
        true
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.default_timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn bash_simple_command() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({
                "command": "echo 'hello world'"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["exit_code"], 0);
        assert!(result["stdout"].as_str().unwrap().contains("hello world"));
    }

    #[tokio::test]
    async fn bash_with_stderr() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({
                "command": "echo 'error' >&2"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert!(result["stderr"].as_str().unwrap().contains("error"));
    }

    #[tokio::test]
    async fn bash_exit_code() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({
                "command": "exit 42"
            }))
            .await
            .unwrap();

        assert!(!result["success"].as_bool().unwrap());
        assert_eq!(result["exit_code"], 42);
    }

    #[tokio::test]
    async fn bash_with_cwd() {
        let dir = TempDir::new().unwrap();

        let tool = BashTool::new();
        let result = tool
            .execute(json!({
                "command": "pwd",
                "cwd": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        // The output should contain the temp directory path
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.contains(dir.path().file_name().unwrap().to_str().unwrap()));
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tool = BashTool::with_timeouts(1, 5); // 1 second timeout
        let result = tool
            .execute(json!({
                "command": "sleep 10",
                "timeout": 1
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn bash_invalid_cwd() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({
                "command": "echo test",
                "cwd": "/nonexistent/directory"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn bash_relative_cwd_rejected() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({
                "command": "echo test",
                "cwd": "relative/path"
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("absolute"));
    }

    #[tokio::test]
    async fn bash_empty_command_rejected() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({
                "command": ""
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn bash_dangerous_command_rejected() {
        let tool = BashTool::new();
        let result = tool.validate_args(&json!({
            "command": "rm -rf /"
        }));

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dangerous"));
    }

    #[test]
    fn bash_requires_sandbox() {
        let tool = BashTool::new();
        assert!(tool.requires_sandbox());
    }

    #[test]
    fn truncate_output_small() {
        let (output, truncated) = BashTool::truncate_output("small output");
        assert_eq!(output, "small output");
        assert!(!truncated);
    }

    #[test]
    fn truncate_output_large() {
        let large_output = "x".repeat(MAX_OUTPUT_SIZE + 1000);
        let (output, truncated) = BashTool::truncate_output(&large_output);
        assert!(output.len() <= MAX_OUTPUT_SIZE + 100); // Some overhead for truncation message
        assert!(truncated);
        assert!(output.contains("truncated"));
    }

    #[test]
    fn config_has_correct_schema() {
        let config = BashTool::config();
        assert_eq!(config.definition.name, "bash");
        assert!(config.definition.description.contains("shell command"));
        assert!(config.sandboxed); // Should require sandbox

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["command"].is_object());
        assert!(schema["properties"]["timeout"].is_object());
        assert!(schema["properties"]["cwd"].is_object());
    }
}
