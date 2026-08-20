//! [`SandboxedExecution`] — the handle that turns a configured
//! [`SandboxFactory`] into something you can call.
//!
//! A sandbox is one-shot by design: every invocation gets a fresh child so a
//! tool cannot leave state behind for the next one. That lifecycle —
//! `create` → `execute` → `destroy` — is three lines, and three lines is
//! exactly the amount of code that gets copied slightly wrong. This type owns
//! the only copy of it, and it is the same copy the prompt loop runs, so an
//! embedder driving its own loop gets the sandbox the framework actually
//! ships rather than a re-derivation of it.
//!
//! # Example
//!
//! ```rust,ignore
//! use acton_ai::tools::sandbox::{ProcessSandboxConfig, SandboxedExecution};
//! use std::time::Duration;
//!
//! let sandbox = SandboxedExecution::process(
//!     ProcessSandboxConfig::new().with_timeout(Duration::from_secs(30)),
//! )?;
//!
//! let out = sandbox.execute("bash", serde_json::json!({"command": "echo hi"})).await?;
//! assert_eq!(out["stdout"].as_str().unwrap().trim_end(), "hi");
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use super::process::{ProcessSandboxConfig, ProcessSandboxFactory};
use super::traits::SandboxFactory;
use crate::tools::error::ToolError;

/// A callable handle over a configured sandbox factory.
///
/// Cloning is cheap — the handle is an `Arc` around the factory, and the
/// factory itself holds only a path and a config. Clone it freely into the
/// closures and actors that need to run tools.
///
/// Each [`execute`](Self::execute) call builds a sandbox, runs one
/// invocation, and destroys it. Nothing is reused between calls.
#[derive(Debug, Clone)]
pub struct SandboxedExecution {
    factory: Arc<dyn SandboxFactory>,
}

impl SandboxedExecution {
    /// Wraps an already-built factory.
    ///
    /// Use this to plug in a custom [`SandboxFactory`]; the two `process_*`
    /// constructors cover the shipped
    /// [`ProcessSandbox`](super::ProcessSandbox).
    #[must_use]
    pub fn new(factory: Arc<dyn SandboxFactory>) -> Self {
        Self { factory }
    }

    /// Builds a handle over a [`ProcessSandbox`](super::ProcessSandbox) that
    /// re-execs the currently running binary.
    ///
    /// This is the production constructor, and it requires that the running
    /// binary honours the sandbox re-exec contract — that is, that its `main`
    /// detours into [`runner::main`](super::process::runner::main) when
    /// `ACTON_AI_SANDBOX_RUNNER=1` is set, the way `acton-ai`'s does. A binary
    /// that does not will spawn a child that ignores the protocol.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] when the current executable cannot be resolved
    /// or canonicalized, or when `config` fails
    /// [`ProcessSandboxConfig::validate`].
    pub fn process(config: ProcessSandboxConfig) -> Result<Self, ToolError> {
        Ok(Self::new(Arc::new(ProcessSandboxFactory::new(config)?)))
    }

    /// Builds a handle over a [`ProcessSandbox`](super::ProcessSandbox) that
    /// re-execs an explicit binary.
    ///
    /// The counterpart to [`process`](Self::process) for tests and for hosts
    /// that ship the sandbox worker as a separate executable:
    /// `env!("CARGO_BIN_EXE_acton-ai")` names the crate's own binary without
    /// consulting `current_exe`.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] when `exe` cannot be canonicalized — a missing
    /// binary is reported here rather than at the first call — or when
    /// `config` fails validation.
    pub fn process_with_exe(exe: PathBuf, config: ProcessSandboxConfig) -> Result<Self, ToolError> {
        Ok(Self::new(Arc::new(ProcessSandboxFactory::with_exe(
            exe, config,
        )?)))
    }

    /// The factory this handle calls.
    ///
    /// Exposed for inspection — downcasting to
    /// [`ProcessSandboxFactory`] to read its `exe()` and `config()`, say.
    #[must_use]
    pub fn factory(&self) -> &Arc<dyn SandboxFactory> {
        &self.factory
    }

    /// Whether the factory believes it can produce a sandbox here.
    ///
    /// For the process sandbox this checks that the target executable still
    /// exists. It is a pre-flight hint, not a guarantee: a `true` here can
    /// still be followed by a failing [`execute`](Self::execute).
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.factory.is_available()
    }

    /// Runs one tool invocation in a fresh sandbox.
    ///
    /// The sandbox is created, given exactly this call, and destroyed —
    /// including when the call fails, so a failed invocation cannot leak a
    /// child into the next one.
    ///
    /// `tool_name` must be one the sandbox worker knows how to dispatch;
    /// for the shipped runner that is `bash`, `write_file`, or `edit_file`.
    /// An unknown name comes back as an error from inside the child rather
    /// than being caught here, because the parent deliberately holds no
    /// registry of what the child can run.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] when the sandbox cannot be created, when the
    /// child breaches its wall-clock deadline or resource limits, or when the
    /// tool itself fails.
    pub async fn execute(&self, tool_name: &str, args: Value) -> Result<Value, ToolError> {
        let mut sandbox = self.factory.create().await?;
        let result = sandbox.execute(tool_name, args).await;
        sandbox.destroy();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::sandbox::HardeningMode;
    use serde_json::json;
    use std::time::Duration;

    fn test_config() -> ProcessSandboxConfig {
        ProcessSandboxConfig::new()
            .with_timeout(Duration::from_secs(15))
            .with_hardening(HardeningMode::Off)
    }

    #[test]
    fn a_missing_binary_is_rejected_when_the_handle_is_built() {
        // Reporting this at construction rather than at the first call is the
        // whole point: a host that mis-configures the worker path finds out
        // at startup, not the first time a model asks for a shell.
        let error = SandboxedExecution::process_with_exe(
            PathBuf::from("/nonexistent/acton-ai-sandboxed-execution-test"),
            test_config(),
        )
        .expect_err("a missing binary must be refused up front");

        assert!(
            error.to_string().contains("canonicalize"),
            "the error should name the path problem, got: {error}"
        );
    }

    #[test]
    fn an_invalid_config_is_rejected_when_the_handle_is_built() {
        let mut config = test_config();
        config.env_allowlist.clear();
        let exe = std::env::current_exe().expect("the test binary has a path");

        let error = SandboxedExecution::process_with_exe(exe, config)
            .expect_err("an invalid config must be refused");

        assert!(error.to_string().contains("env_allowlist"), "{error}");
    }

    #[test]
    fn a_handle_over_an_existing_binary_reports_itself_available() {
        let exe = std::env::current_exe().expect("the test binary has a path");
        let handle = SandboxedExecution::process_with_exe(exe, test_config())
            .expect("the test binary canonicalizes");

        assert!(handle.is_available());
    }

    #[tokio::test]
    async fn a_factory_that_cannot_create_surfaces_its_error_from_execute() {
        // The lifecycle must not swallow a creation failure and report it as
        // a tool failure — an operator debugging this needs to know the
        // sandbox never came up, not that bash misbehaved.
        let handle = SandboxedExecution::new(Arc::new(RefusingFactory));

        let error = handle
            .execute("bash", json!({"command": "true"}))
            .await
            .expect_err("a factory that refuses must fail the call");

        assert!(error.to_string().contains("refuses to create"), "{error}");
    }

    /// A factory that always refuses, so the lifecycle's error path can be
    /// exercised without spawning anything.
    #[derive(Debug)]
    struct RefusingFactory;

    impl SandboxFactory for RefusingFactory {
        fn create(&self) -> crate::tools::sandbox::SandboxFactoryFuture {
            Box::pin(async { Err(ToolError::sandbox_error("this factory refuses to create")) })
        }

        fn is_available(&self) -> bool {
            false
        }
    }
}
