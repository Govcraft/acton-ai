//! Hyperlight micro-VM sandbox implementation.
//!
//! This module provides hardware-isolated sandbox execution using
//! Microsoft's Hyperlight technology. Hyperlight creates lightweight
//! micro-VMs with 1-2ms cold start time for secure code execution.
//!
//! ## Features
//!
//! - **Hardware Isolation**: Uses KVM (Linux) or Hyper-V (Windows)
//! - **Fast Cold Start**: 1-2ms to create new micro-VM
//! - **Pre-warmed Pool**: Optional pool of ready sandboxes
//! - **Configurable Limits**: Memory and timeout configuration
//!
//! ## Requirements
//!
//! - Linux with KVM or Windows with Hyper-V
//! - The `hyperlight` feature enabled in Cargo.toml
//!
//! ## Usage
//!
//! ```rust,ignore
//! use acton_ai::tools::sandbox::{
//!     HyperlightSandbox, HyperlightSandboxFactory, SandboxConfig
//! };
//!
//! // Create a factory
//! let factory = HyperlightSandboxFactory::new()?;
//!
//! // Create a sandbox
//! let sandbox = factory.create().await?;
//!
//! // Execute code
//! let result = sandbox.execute("echo hello", serde_json::json!({})).await?;
//! ```
//!
//! ## Pool Usage
//!
//! For lower latency, use the sandbox pool:
//!
//! ```rust,ignore
//! use acton_ai::tools::sandbox::{SandboxPool, SandboxConfig, WarmPool};
//!
//! // Create pool actor
//! let config = SandboxConfig::new().with_pool_size(Some(4));
//! let pool = SandboxPool::spawn(&mut runtime, config).await;
//!
//! // Warm up with pre-created sandboxes
//! pool.send(WarmPool { count: 4 }).await;
//!
//! // Acquire sandbox (returns pooled sandbox with auto-release)
//! let (tx, rx) = tokio::sync::oneshot::channel();
//! pool.send(AcquireSandbox { reply: tx }).await;
//! let sandbox = rx.await??;
//!
//! // Use sandbox...
//! ```

mod config;
mod error;
mod factory;
mod pool;
mod sandbox;

// Re-export all public types
pub use config::{
    GuestBinarySource, SandboxConfig, DEFAULT_MEMORY_LIMIT, DEFAULT_POOL_SIZE, DEFAULT_TIMEOUT,
};
pub use error::SandboxErrorKind;
pub use factory::{AutoSandboxFactory, HyperlightSandboxFactory};
pub use pool::{
    AcquireSandbox, GetPoolMetrics, InitPool, InternalReleaseSandbox, PoolMetrics,
    PoolMetricsResponse, PooledSandbox, ReleaseSandbox, SandboxPool, WarmPool,
};
pub use sandbox::HyperlightSandbox;
