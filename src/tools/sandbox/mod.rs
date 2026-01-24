//! Sandbox trait and implementations.
//!
//! Defines the interface for sandboxed code execution and provides
//! implementations for both development (stub) and production (Hyperlight).
//!
//! ## Overview
//!
//! Sandboxes provide isolated environments for executing untrusted code.
//! The `Sandbox` trait defines the execution interface, while `SandboxFactory`
//! handles sandbox lifecycle management.
//!
//! ## Implementations
//!
//! - **StubSandbox**: Development placeholder that does NOT sandbox code
//! - **HyperlightSandbox**: Production implementation using Hyperlight micro-VMs
//!   (requires the `hyperlight` feature)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use acton_ai::tools::sandbox::{SandboxFactory, StubSandboxFactory};
//!
//! // Create a factory (use HyperlightSandboxFactory in production)
//! let factory = StubSandboxFactory::new();
//!
//! // Create a sandbox instance
//! let sandbox = factory.create().await?;
//!
//! // Execute code in the sandbox
//! let result = sandbox.execute("echo hello", serde_json::json!({})).await?;
//! ```
//!
//! ## Feature Flags
//!
//! - `hyperlight`: Enables the Hyperlight micro-VM sandbox implementation

mod stub;
mod traits;

#[cfg(feature = "hyperlight")]
pub mod hyperlight;

// Re-export traits
pub use traits::{Sandbox, SandboxExecutionFuture, SandboxFactory, SandboxFactoryFuture};

// Re-export stub implementation
pub use stub::{StubSandbox, StubSandboxFactory};

// Re-export Hyperlight implementation when feature is enabled
#[cfg(feature = "hyperlight")]
pub use hyperlight::{
    AcquireSandbox, AutoSandboxFactory, GetPoolMetrics, GuestBinarySource, HyperlightSandbox,
    HyperlightSandboxFactory, InitPool, InternalReleaseSandbox, PoolMetrics, PoolMetricsResponse,
    PooledSandbox, ReleaseSandbox, SandboxConfig, SandboxErrorKind, SandboxPool, WarmPool,
};
