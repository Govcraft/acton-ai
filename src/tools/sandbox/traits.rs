//! Sandbox trait definitions.
//!
//! Defines the interface for sandboxed code execution.

use crate::tools::error::ToolError;
use serde_json::Value;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;

/// The result type for sandbox execution futures.
pub type SandboxExecutionFuture =
    Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + Sync + 'static>>;

/// The result type for sandbox factory futures.
pub type SandboxFactoryFuture =
    Pin<Box<dyn Future<Output = Result<Box<dyn Sandbox>, ToolError>> + Send + Sync + 'static>>;

/// Trait for sandboxed code execution.
///
/// Sandboxes provide isolated environments for executing untrusted code.
/// The primary implementation uses Hyperlight micro-VMs for hardware isolation.
///
/// # Thread Safety
///
/// Sandboxes must be `Send + Sync` to support use across async contexts.
/// Implementations should ensure thread-safe internal state management.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::{Sandbox, SandboxFactory, HyperlightSandboxFactory};
///
/// let factory = HyperlightSandboxFactory::new()?;
/// let sandbox = factory.create().await?;
///
/// let result = sandbox.execute("echo hello", serde_json::json!({})).await?;
/// sandbox.destroy();
/// ```
pub trait Sandbox: Send + Sync + Debug {
    /// Executes code in the sandbox.
    ///
    /// # Arguments
    ///
    /// * `code` - The code or command to execute
    /// * `args` - Arguments to pass to the code (as JSON)
    ///
    /// # Returns
    ///
    /// The result of execution as a JSON value, or an error.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::SandboxError` if execution fails within the sandbox.
    fn execute(&self, code: &str, args: Value) -> SandboxExecutionFuture;

    /// Destroys the sandbox, releasing all resources.
    ///
    /// After calling this, the sandbox cannot be used again.
    /// Subsequent calls to `execute` will return an error.
    fn destroy(&mut self);

    /// Returns whether the sandbox is still usable.
    ///
    /// Returns `false` after `destroy()` has been called.
    fn is_alive(&self) -> bool;

    /// Executes code synchronously (for use in blocking contexts).
    ///
    /// This method is intended for use with `tokio::task::spawn_blocking`
    /// when the sandbox implementation requires synchronous execution
    /// (e.g., Hyperlight's `MultiUseSandbox::call`).
    ///
    /// # Arguments
    ///
    /// * `code` - The code or command to execute
    /// * `args` - Arguments to pass to the code (as JSON)
    ///
    /// # Returns
    ///
    /// The result of execution as a JSON value, or an error.
    ///
    /// # Default Implementation
    ///
    /// Returns an error indicating synchronous execution is not supported.
    /// Override this method for sandboxes that support synchronous execution.
    fn execute_sync(&self, code: &str, args: Value) -> Result<Value, ToolError> {
        let _ = (code, args); // Suppress unused parameter warnings
        Err(ToolError::sandbox_error(
            "synchronous execution not supported by this sandbox",
        ))
    }
}

/// Factory for creating sandbox instances.
///
/// This allows different sandbox implementations to be plugged in
/// without changing the tool execution code.
///
/// # Availability
///
/// Use `is_available()` to check if the factory can create sandboxes
/// on the current system (e.g., hypervisor presence).
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::{SandboxFactory, HyperlightSandboxFactory};
///
/// let factory = HyperlightSandboxFactory::new_with_fallback();
/// if factory.is_available() {
///     let sandbox = factory.create().await?;
///     // Use sandbox...
/// }
/// ```
pub trait SandboxFactory: Send + Sync + Debug {
    /// Creates a new sandbox instance.
    ///
    /// # Returns
    ///
    /// A boxed sandbox instance, or an error if creation fails.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::SandboxError` if the sandbox cannot be created.
    fn create(&self) -> SandboxFactoryFuture;

    /// Returns whether this factory can create sandboxes.
    ///
    /// For Hyperlight, this checks if a hypervisor is available.
    /// For stub implementations, this returns `true`.
    fn is_available(&self) -> bool {
        true
    }
}
