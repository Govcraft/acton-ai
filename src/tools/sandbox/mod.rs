//! Sandbox trait and implementations.
//!
//! Defines the interface for sandboxed code execution. The production
//! implementation is [`ProcessSandbox`], a portable subprocess-based sandbox
//! with best-effort OS hardening (landlock + seccomp on Linux 5.13+).
//!
//! ## Overview
//!
//! Sandboxes provide isolated environments for executing untrusted code.
//! The [`Sandbox`] trait defines the execution interface, while
//! [`SandboxFactory`] handles sandbox lifecycle management.
//!
//! ## Implementations
//!
//! - [`ProcessSandbox`]: Production implementation. Re-execs the current
//!   binary as a child process, applies rlimits + landlock/seccomp, and
//!   enforces a wall-clock timeout in the parent.
//! - `StubSandbox` (test-only): placeholder that does NOT sandbox code.
//!
//! ## Usage
//!
//! [`SandboxedExecution`] is the handle most callers want — it owns the
//! one-shot `create` → `execute` → `destroy` lifecycle so nobody has to
//! reimplement it:
//!
//! ```rust,ignore
//! use acton_ai::tools::sandbox::{ProcessSandboxConfig, SandboxedExecution};
//! use std::time::Duration;
//!
//! let sandbox = SandboxedExecution::process(
//!     ProcessSandboxConfig::new().with_timeout(Duration::from_secs(30)),
//! )?;
//! let result = sandbox.execute("bash", serde_json::json!({"command": "echo hi"})).await?;
//! ```
//!
//! The trait layer underneath stays available for custom implementations:
//!
//! ```rust,ignore
//! use acton_ai::tools::sandbox::{ProcessSandboxConfig, ProcessSandboxFactory, SandboxFactory};
//!
//! let factory = ProcessSandboxFactory::new(ProcessSandboxConfig::new())?;
//! let sandbox = factory.create().await?;
//! let result = sandbox.execute("bash", serde_json::json!({"command": "echo hi"})).await?;
//! ```

// Stub implementation is only available in tests (security concern in production)
#[cfg(test)]
mod stub;
mod traits;

pub mod execution;
pub mod process;

// Re-export traits
pub use traits::{Sandbox, SandboxExecutionFuture, SandboxFactory, SandboxFactoryFuture};

// Re-export the callable handle over a configured factory.
pub use execution::SandboxedExecution;

// Stub implementation is only available in tests
#[cfg(test)]
pub use stub::{StubSandbox, StubSandboxFactory};

// Re-export process sandbox implementation.
pub use process::{HardeningMode, ProcessSandbox, ProcessSandboxConfig, ProcessSandboxFactory};
