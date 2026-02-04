//! Rust code execution built-in tool.
//!
//! Provides secure execution of agent-generated Rust code with compiler
//! verification before sandbox execution.

use crate::messages::ToolDefinition;
use crate::tools::actor::{ExecuteToolDirect, ToolActor, ToolActorResponse};
use crate::tools::compiler::{CompilationError, CompilationErrorKind, RustCompiler};
use crate::tools::sandbox::{GuestBinarySource, HyperlightSandbox, Sandbox, SandboxConfig};
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use acton_reactive::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Tool for executing agent-generated Rust code with compiler verification.
///
/// This tool provides a secure way for agents to execute Rust code:
///
/// 1. Code is wrapped in a no_std template with `#![forbid(unsafe_code)]`
/// 2. Clippy runs with `-D warnings` to catch issues
/// 3. Code is compiled to `x86_64-unknown-none` target
/// 4. Binary executes in a Hyperlight sandbox with hardware isolation
///
/// # Security Model
///
/// The tool provides defense-in-depth:
/// - **Compile-time**: Clippy catches common bugs and security issues
/// - **Build-time**: `forbid(unsafe_code)` prevents unsafe Rust
/// - **Runtime**: Hyperlight micro-VM provides hardware isolation
///
/// # Example Tool Call
///
/// ```json
/// {
///   "code": "input.to_uppercase()",
///   "input": "hello world"
/// }
/// ```
///
/// Returns:
/// ```json
/// {
///   "output": "HELLO WORLD",
///   "success": true
/// }
/// ```
#[derive(Debug)]
pub struct RustCodeTool {
    compiler: Arc<RustCompiler>,
}

/// rust_code tool actor state.
///
/// This actor wraps the `RustCodeTool` executor for per-agent tool spawning.
/// The compiler is initialized lazily on first execution.
#[acton_actor]
pub struct RustCodeToolActor;

/// Global compiler instance (initialized lazily).
static COMPILER: OnceLock<Result<Arc<RustCompiler>, String>> = OnceLock::new();

/// Gets or initializes the global compiler instance.
fn get_compiler() -> Result<Arc<RustCompiler>, String> {
    COMPILER
        .get_or_init(|| RustCompiler::new().map(Arc::new).map_err(|e| e.to_string()))
        .clone()
}

/// Arguments for the rust_code tool.
#[derive(Debug, Deserialize)]
struct RustCodeArgs {
    /// Rust code to execute (function body).
    code: String,

    /// Input string passed to the function (optional).
    #[serde(default)]
    input: Option<String>,

    /// Timeout in seconds (optional, default 30).
    #[serde(default)]
    timeout_secs: Option<u64>,
}

impl RustCodeTool {
    /// Creates a new rust_code tool.
    ///
    /// # Errors
    ///
    /// Returns an error if the `RustCompiler` cannot be created (missing toolchain).
    pub fn new() -> Result<Self, CompilationError> {
        Ok(Self {
            compiler: Arc::new(RustCompiler::new()?),
        })
    }

    /// Creates a new rust_code tool with a shared compiler.
    #[must_use]
    pub fn with_compiler(compiler: Arc<RustCompiler>) -> Self {
        Self { compiler }
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        ToolConfig::new(Self::definition())
            .with_sandbox(true)
            .with_timeout(Duration::from_secs(60))
    }

    /// Returns the tool definition.
    #[must_use]
    pub fn definition() -> ToolDefinition {
        ToolDefinition {
            name: "rust_code".to_string(),
            description: "Execute Rust code in a secure sandbox. Code is compiler-verified \
                         (clippy with -D warnings) before execution. Write a function body \
                         that takes `input: String` and returns `String`. The code runs in a \
                         no_std environment with #![forbid(unsafe_code)]."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Rust function body. Receives `input: String`, must return `String`. \
                                       Example: `input.to_uppercase()` or `format!(\"Result: {}\", input)`"
                    },
                    "input": {
                        "type": "string",
                        "description": "Input string passed to the function (default: empty string)"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30, max: 300)"
                    }
                },
                "required": ["code"]
            }),
        }
    }

    /// Executes Rust code with the given arguments.
    async fn execute_inner(
        compiler: Arc<RustCompiler>,
        args: RustCodeArgs,
    ) -> Result<Value, ToolError> {
        let input = args.input.unwrap_or_default();
        let timeout_secs = args.timeout_secs.unwrap_or(30).min(300);

        // 1. Compile the code (with clippy verification)
        let compiled = compiler
            .compile(&args.code)
            .map_err(|e| Self::compilation_error_to_tool_error(&e))?;

        // 2. Create sandbox with compiled binary
        let config = SandboxConfig::new()
            .with_guest_binary(GuestBinarySource::FromBytes(compiled.into_bytes()))
            .with_timeout(Duration::from_secs(timeout_secs));

        let sandbox =
            HyperlightSandbox::new(config).map_err(|e| ToolError::sandbox_error(e.to_string()))?;

        // 3. Execute in sandbox (synchronous, via spawn_blocking)
        let result = tokio::task::spawn_blocking(move || {
            sandbox.execute_sync("run_code", json!({ "input": input }))
        })
        .await
        .map_err(|e| ToolError::sandbox_error(format!("spawn_blocking failed: {e}")))?;

        // 4. Return result
        result.map(|output| {
            json!({
                "output": output,
                "success": true
            })
        })
    }

    /// Converts a `CompilationError` to a `ToolError` with helpful messages.
    fn compilation_error_to_tool_error(error: &CompilationError) -> ToolError {
        match error.kind() {
            CompilationErrorKind::ClippyFailed {
                errors,
                error_count,
            } => ToolError::validation_failed(
                "rust_code",
                format!(
                    "Code has {} clippy error(s). Fix the issues and retry:\n\n{}",
                    error_count, errors
                ),
            ),
            CompilationErrorKind::CompilationFailed { errors, .. } => ToolError::validation_failed(
                "rust_code",
                format!(
                    "Code failed to compile. Check for syntax errors:\n\n{}",
                    errors
                ),
            ),
            CompilationErrorKind::TemplateFailed { reason } => {
                ToolError::validation_failed("rust_code", reason.clone())
            }
            CompilationErrorKind::ToolchainError {
                missing,
                install_hint,
            } => ToolError::execution_failed(
                "rust_code",
                format!(
                    "Required tooling '{}' not available. Install with: {}",
                    missing, install_hint
                ),
            ),
            _ => ToolError::execution_failed("rust_code", error.to_string()),
        }
    }
}

impl ToolExecutorTrait for RustCodeTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        let compiler = Arc::clone(&self.compiler);

        Box::pin(async move {
            let args: RustCodeArgs = serde_json::from_value(args).map_err(|e| {
                ToolError::validation_failed("rust_code", format!("invalid arguments: {e}"))
            })?;

            Self::execute_inner(compiler, args).await
        })
    }

    fn requires_sandbox(&self) -> bool {
        true
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(60)
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: RustCodeArgs = serde_json::from_value(args.clone()).map_err(|e| {
            ToolError::validation_failed("rust_code", format!("invalid arguments: {e}"))
        })?;

        if args.code.trim().is_empty() {
            return Err(ToolError::validation_failed(
                "rust_code",
                "code cannot be empty",
            ));
        }

        // Basic sanity check on code length
        if args.code.len() > 100_000 {
            return Err(ToolError::validation_failed(
                "rust_code",
                "code too long (max 100KB)",
            ));
        }

        Ok(())
    }
}

impl ToolActor for RustCodeToolActor {
    fn name() -> &'static str {
        "rust_code"
    }

    fn definition() -> ToolDefinition {
        RustCodeTool::definition()
    }

    async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<Self>("rust_code_tool".to_string());

        builder.act_on::<ExecuteToolDirect>(|actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let response = match get_compiler() {
                    Ok(compiler) => {
                        let tool = RustCodeTool::with_compiler(compiler);
                        match tool.execute(args).await {
                            Ok(value) => {
                                let result_str = serde_json::to_string(&value)
                                    .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e));
                                ToolActorResponse::success(correlation_id, tool_call_id, result_str)
                            }
                            Err(e) => ToolActorResponse::error(
                                correlation_id,
                                tool_call_id,
                                e.to_string(),
                            ),
                        }
                    }
                    Err(e) => ToolActorResponse::error(
                        correlation_id,
                        tool_call_id,
                        format!("RustCompiler not initialized: {e}"),
                    ),
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

    // --- RustCodeArgs tests ---

    #[test]
    fn rust_code_args_deserialize_minimal() {
        let json = json!({ "code": "input.to_uppercase()" });
        let args: RustCodeArgs = serde_json::from_value(json).unwrap();

        assert_eq!(args.code, "input.to_uppercase()");
        assert!(args.input.is_none());
        assert!(args.timeout_secs.is_none());
    }

    #[test]
    fn rust_code_args_deserialize_full() {
        let json = json!({
            "code": "input.to_uppercase()",
            "input": "hello",
            "timeout_secs": 60
        });
        let args: RustCodeArgs = serde_json::from_value(json).unwrap();

        assert_eq!(args.code, "input.to_uppercase()");
        assert_eq!(args.input, Some("hello".to_string()));
        assert_eq!(args.timeout_secs, Some(60));
    }

    // --- ToolDefinition tests ---

    #[test]
    fn rust_code_definition_name() {
        let def = RustCodeTool::definition();
        assert_eq!(def.name, "rust_code");
    }

    #[test]
    fn rust_code_definition_description() {
        let def = RustCodeTool::definition();
        assert!(def.description.contains("secure sandbox"));
        assert!(def.description.contains("clippy"));
        assert!(def.description.contains("forbid(unsafe_code)"));
    }

    #[test]
    fn rust_code_definition_schema() {
        let def = RustCodeTool::definition();
        let schema = &def.input_schema;

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["code"].is_object());
        assert!(schema["properties"]["input"].is_object());
        assert!(schema["properties"]["timeout_secs"].is_object());
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("code")));
    }

    // --- ToolConfig tests ---

    #[test]
    fn rust_code_config_sandboxed() {
        let config = RustCodeTool::config();
        assert!(config.sandboxed);
    }

    #[test]
    fn rust_code_config_timeout() {
        let config = RustCodeTool::config();
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    // --- ToolActor tests ---

    #[test]
    fn rust_code_actor_name() {
        assert_eq!(RustCodeToolActor::name(), "rust_code");
    }

    #[test]
    fn rust_code_actor_definition() {
        let def = RustCodeToolActor::definition();
        assert_eq!(def.name, "rust_code");
    }

    // --- Validation tests (using ToolExecutorTrait) ---
    // Note: These tests don't require the compiler to be initialized

    #[test]
    #[ignore = "requires rust toolchain"]
    fn validate_args_empty_code_fails() {
        // This test requires the compiler which needs the toolchain
        if let Ok(tool) = RustCodeTool::new() {
            let result = tool.validate_args(&json!({ "code": "" }));
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("empty"));
        }
    }

    #[test]
    #[ignore = "requires rust toolchain"]
    fn validate_args_whitespace_code_fails() {
        if let Ok(tool) = RustCodeTool::new() {
            let result = tool.validate_args(&json!({ "code": "   \n\t   " }));
            assert!(result.is_err());
        }
    }

    #[test]
    #[ignore = "requires rust toolchain"]
    fn validate_args_code_too_long_fails() {
        if let Ok(tool) = RustCodeTool::new() {
            let long_code = "a".repeat(100_001);
            let result = tool.validate_args(&json!({ "code": long_code }));
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("too long"));
        }
    }

    #[test]
    #[ignore = "requires rust toolchain"]
    fn validate_args_valid_code_passes() {
        if let Ok(tool) = RustCodeTool::new() {
            let result = tool.validate_args(&json!({ "code": "input.to_uppercase()" }));
            assert!(result.is_ok());
        }
    }

    // --- Error conversion tests ---

    #[test]
    fn compilation_error_to_tool_error_clippy() {
        use crate::tools::compiler::CompilationError;

        let error = CompilationError::clippy_failed("unused variable", 1);
        let tool_error = RustCodeTool::compilation_error_to_tool_error(&error);

        let msg = tool_error.to_string();
        assert!(msg.contains("clippy"));
        assert!(msg.contains("1"));
    }

    #[test]
    fn compilation_error_to_tool_error_compilation() {
        use crate::tools::compiler::CompilationError;

        let error = CompilationError::compilation_failed("syntax error", Some(1));
        let tool_error = RustCodeTool::compilation_error_to_tool_error(&error);

        let msg = tool_error.to_string();
        assert!(msg.contains("compile"));
    }

    #[test]
    fn compilation_error_to_tool_error_template() {
        use crate::tools::compiler::CompilationError;

        let error = CompilationError::template_failed("empty code");
        let tool_error = RustCodeTool::compilation_error_to_tool_error(&error);

        let msg = tool_error.to_string();
        assert!(msg.contains("empty code"));
    }

    #[test]
    fn compilation_error_to_tool_error_toolchain() {
        use crate::tools::compiler::CompilationError;

        let error = CompilationError::toolchain_error("cargo", "install rustup");
        let tool_error = RustCodeTool::compilation_error_to_tool_error(&error);

        let msg = tool_error.to_string();
        assert!(msg.contains("cargo"));
        assert!(msg.contains("install rustup"));
    }
}
