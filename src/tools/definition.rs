//! Tool definition and executor traits.
//!
//! Defines the core traits for tool execution and the ToolConfig structure
//! that wraps tool definitions with execution configuration.

use crate::messages::ToolDefinition;
use crate::tools::error::ToolError;
use serde_json::Value;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

// Re-export ToolDefinition for convenience
pub use crate::messages::ToolDefinition as ToolDef;

/// Configuration for a registered tool.
#[derive(Debug, Clone)]
pub struct ToolConfig {
    /// The tool definition
    pub definition: ToolDefinition,
    /// Whether this tool requires sandbox execution
    pub sandboxed: bool,
    /// Execution timeout
    pub timeout: Duration,
}

impl ToolConfig {
    /// Creates a new tool configuration.
    #[must_use]
    pub fn new(definition: ToolDefinition) -> Self {
        Self {
            definition,
            sandboxed: false,
            timeout: Duration::from_secs(30),
        }
    }

    /// Sets whether the tool requires sandbox execution.
    #[must_use]
    pub fn with_sandbox(mut self, sandboxed: bool) -> Self {
        self.sandboxed = sandboxed;
        self
    }

    /// Sets the execution timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            definition: ToolDefinition {
                name: "unnamed".to_string(),
                description: "An unnamed tool".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
            sandboxed: false,
            timeout: Duration::from_secs(30),
        }
    }
}

/// The result type for tool execution futures.
pub type ToolExecutionFuture =
    Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + Sync + 'static>>;

/// Trait for executing tools.
///
/// Implement this trait to create custom tool executors.
/// Tools can be executed inline or in a sandbox depending on configuration.
///
/// # Example
///
/// ```rust
/// use acton_ai::tools::{ToolExecutorTrait, ToolError, ToolExecutionFuture};
/// use serde_json::Value;
///
/// #[derive(Debug)]
/// struct EchoTool;
///
/// impl ToolExecutorTrait for EchoTool {
///     fn execute(&self, args: Value) -> ToolExecutionFuture {
///         Box::pin(async move {
///             Ok(args)
///         })
///     }
/// }
/// ```
pub trait ToolExecutorTrait: Send + Sync + Debug {
    /// Executes the tool with the given arguments.
    ///
    /// # Arguments
    ///
    /// * `args` - JSON value containing the tool arguments
    ///
    /// # Returns
    ///
    /// A future that resolves to the result of the tool execution as a JSON value,
    /// or an error.
    fn execute(&self, args: Value) -> ToolExecutionFuture;

    /// Returns whether this tool requires sandbox execution.
    ///
    /// Sandboxed tools run in isolated environments (e.g., Hyperlight micro-VMs)
    /// for security when executing untrusted code.
    fn requires_sandbox(&self) -> bool {
        false
    }

    /// Returns the tool's timeout duration.
    fn timeout(&self) -> Duration {
        Duration::from_secs(30)
    }

    /// Validates the input arguments before execution.
    ///
    /// Override this to provide custom validation logic.
    /// The default implementation accepts any arguments.
    fn validate_args(&self, _args: &Value) -> Result<(), ToolError> {
        Ok(())
    }
}

/// A boxed tool executor for dynamic dispatch.
pub type BoxedToolExecutor = Box<dyn ToolExecutorTrait>;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_definition(name: &str, description: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn tool_config_default_timeout() {
        let config = ToolConfig::new(make_test_definition("t", "d"));
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn tool_config_with_sandbox() {
        let config = ToolConfig::new(make_test_definition("t", "d")).with_sandbox(true);
        assert!(config.sandboxed);
    }

    #[test]
    fn tool_config_with_timeout() {
        let config =
            ToolConfig::new(make_test_definition("t", "d")).with_timeout(Duration::from_secs(60));
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn tool_config_default() {
        let config = ToolConfig::default();
        assert_eq!(config.definition.name, "unnamed");
        assert!(!config.sandboxed);
        assert_eq!(config.timeout, Duration::from_secs(30));
    }
}
