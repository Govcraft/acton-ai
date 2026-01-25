//! Sandbox trait and implementations.
//!
//! Defines the interface for sandboxed code execution and provides
//! implementations for production (Hyperlight).
//!
//! ## Overview
//!
//! Sandboxes provide isolated environments for executing untrusted code.
//! The `Sandbox` trait defines the execution interface, while `SandboxFactory`
//! handles sandbox lifecycle management.
//!
//! ## Implementations
//!
//! - **HyperlightSandbox**: Production implementation using Hyperlight micro-VMs
//!   (requires the `hyperlight` feature)
//! - **StubSandbox**: Test-only placeholder that does NOT sandbox code
//!
//! ## Usage
//!
//! ```rust,ignore
//! use acton_ai::tools::sandbox::hyperlight::{SandboxProvider, SandboxConfig, WarmPool};
//! use acton_reactive::prelude::*;
//!
//! // Create provider - fails explicitly if platform unsupported
//! let provider = SandboxProvider::new(SandboxConfig::default())?;
//!
//! // Spawn the pool actor
//! let pool_handle = provider.spawn(&mut runtime).await;
//!
//! // Warm up and use pool
//! pool_handle.send(WarmPool { count: 4 }).await;
//! ```
//!
//! ## Feature Flags
//!
//! - `hyperlight`: Enables the Hyperlight micro-VM sandbox implementation

// Stub implementation is only available in tests (security concern in production)
#[cfg(test)]
mod stub;
mod traits;

#[cfg(feature = "hyperlight")]
pub mod hyperlight;

// Re-export traits
pub use traits::{Sandbox, SandboxExecutionFuture, SandboxFactory, SandboxFactoryFuture};

// Stub implementation is only available in tests
#[cfg(test)]
pub use stub::{StubSandbox, StubSandboxFactory};

// Re-export Hyperlight implementation when feature is enabled
#[cfg(feature = "hyperlight")]
pub use hyperlight::{
    AcquireSandbox, GetPoolMetrics, GuestBinarySource, HyperlightSandbox,
    HyperlightSandboxFactory, InitPool, InternalReleaseSandbox, PoolMetrics, PoolMetricsResponse,
    PooledSandbox, ReleaseSandbox, SandboxConfig, SandboxErrorKind, SandboxPool, SandboxProvider,
    WarmPool,
};
