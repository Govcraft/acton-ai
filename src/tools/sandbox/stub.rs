//! Stub sandbox implementation.
//!
//! Provides a placeholder sandbox for development and testing.
//! This implementation does NOT actually sandbox code.

use super::traits::{Sandbox, SandboxExecutionFuture, SandboxFactory, SandboxFactoryFuture};
use crate::tools::error::ToolError;
use serde_json::Value;

/// A stub sandbox implementation for development and testing.
///
/// This implementation does not actually sandbox code - it's a placeholder
/// until Hyperlight integration is complete. It should NOT be used for
/// executing untrusted code in production.
///
/// # Warning
///
/// Commands executed through this sandbox run directly on the host system
/// without any isolation. Use `HyperlightSandbox` for production workloads.
#[derive(Debug, Default)]
pub struct StubSandbox {
    /// Whether the sandbox has been destroyed
    destroyed: bool,
}

impl StubSandbox {
    /// Creates a new stub sandbox.
    #[must_use]
    pub fn new() -> Self {
        Self { destroyed: false }
    }
}

impl Sandbox for StubSandbox {
    fn execute(&self, code: &str, args: Value) -> SandboxExecutionFuture {
        if self.destroyed {
            return Box::pin(
                async move { Err(ToolError::sandbox_error("sandbox has been destroyed")) },
            );
        }

        // Stub implementation just returns a placeholder response
        // In production, this would execute code in a Hyperlight micro-VM
        let code_preview = if code.len() > 50 {
            code[..50].to_string()
        } else {
            code.to_string()
        };

        let code_len = code.len();

        Box::pin(async move {
            tracing::warn!(
                code_len = code_len,
                "StubSandbox: NOT actually sandboxing code execution"
            );

            Ok(serde_json::json!({
                "status": "stub",
                "message": "StubSandbox does not execute code",
                "code_preview": code_preview,
                "args": args
            }))
        })
    }

    fn destroy(&mut self) {
        self.destroyed = true;
        tracing::debug!("StubSandbox destroyed");
    }

    fn is_alive(&self) -> bool {
        !self.destroyed
    }

    fn execute_sync(&self, code: &str, args: Value) -> Result<Value, ToolError> {
        if self.destroyed {
            return Err(ToolError::sandbox_error("sandbox has been destroyed"));
        }

        // Stub implementation just returns a placeholder response
        let code_preview = if code.len() > 50 {
            code[..50].to_string()
        } else {
            code.to_string()
        };

        tracing::warn!(
            code_len = code.len(),
            "StubSandbox: NOT actually sandboxing code execution (sync)"
        );

        Ok(serde_json::json!({
            "status": "stub",
            "message": "StubSandbox does not execute code",
            "code_preview": code_preview,
            "args": args
        }))
    }
}

/// A stub sandbox factory that creates StubSandbox instances.
///
/// This factory always reports as available and creates stub sandboxes
/// that do not provide actual isolation.
#[derive(Debug, Default, Clone)]
pub struct StubSandboxFactory;

impl StubSandboxFactory {
    /// Creates a new stub sandbox factory.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl SandboxFactory for StubSandboxFactory {
    fn create(&self) -> SandboxFactoryFuture {
        Box::pin(async move { Ok(Box::new(StubSandbox::new()) as Box<dyn Sandbox>) })
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_sandbox_execute_returns_stub_response() {
        let sandbox = StubSandbox::new();
        let result = sandbox
            .execute("some code", serde_json::json!({"arg": 1}))
            .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value.get("status").unwrap(), "stub");
    }

    #[tokio::test]
    async fn stub_sandbox_destroy_prevents_execution() {
        let mut sandbox = StubSandbox::new();
        sandbox.destroy();
        assert!(!sandbox.is_alive());
        let result = sandbox.execute("code", serde_json::json!({})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("destroyed"));
    }

    #[tokio::test]
    async fn stub_sandbox_is_alive_initially() {
        let sandbox = StubSandbox::new();
        assert!(sandbox.is_alive());
    }

    #[tokio::test]
    async fn stub_sandbox_truncates_long_code() {
        let sandbox = StubSandbox::new();
        let long_code = "x".repeat(100);
        let result = sandbox.execute(&long_code, serde_json::json!({})).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        let preview = value.get("code_preview").unwrap().as_str().unwrap();
        assert_eq!(preview.len(), 50);
    }

    #[tokio::test]
    async fn stub_sandbox_factory_creates_sandbox() {
        let factory = StubSandboxFactory::new();
        let sandbox = factory.create().await;
        assert!(sandbox.is_ok());
        let sandbox = sandbox.unwrap();
        assert!(sandbox.is_alive());
    }

    #[test]
    fn stub_sandbox_factory_is_clone() {
        let factory1 = StubSandboxFactory::new();
        let factory2 = factory1.clone();
        // Just verify it compiles and works
        assert!(format!("{:?}", factory2).contains("StubSandboxFactory"));
    }

    #[test]
    fn stub_sandbox_factory_is_available() {
        let factory = StubSandboxFactory::new();
        assert!(factory.is_available());
    }

    #[test]
    fn stub_sandbox_execute_sync_returns_stub_response() {
        let sandbox = StubSandbox::new();
        let result = sandbox.execute_sync("some code", serde_json::json!({"arg": 1}));
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value.get("status").unwrap(), "stub");
    }

    #[test]
    fn stub_sandbox_execute_sync_after_destroy_fails() {
        let mut sandbox = StubSandbox::new();
        sandbox.destroy();
        let result = sandbox.execute_sync("code", serde_json::json!({}));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("destroyed"));
    }

    #[test]
    fn stub_sandbox_execute_sync_truncates_long_code() {
        let sandbox = StubSandbox::new();
        let long_code = "x".repeat(100);
        let result = sandbox.execute_sync(&long_code, serde_json::json!({}));
        assert!(result.is_ok());
        let value = result.unwrap();
        let preview = value.get("code_preview").unwrap().as_str().unwrap();
        assert_eq!(preview.len(), 50);
    }
}
