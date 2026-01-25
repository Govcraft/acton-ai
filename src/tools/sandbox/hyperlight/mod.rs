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
mod guest;
mod pool;
mod provider;
mod sandbox;

// Re-export all public types
pub use config::{
    GuestBinarySource, PoolConfig, SandboxConfig, DEFAULT_MAX_EXECUTIONS_BEFORE_RECYCLE,
    DEFAULT_MAX_PER_TYPE, DEFAULT_MEMORY_LIMIT, DEFAULT_POOL_SIZE, DEFAULT_TIMEOUT,
    DEFAULT_WARMUP_COUNT,
};
pub use error::SandboxErrorKind;
pub use factory::HyperlightSandboxFactory;
pub use guest::{GuestBinaries, GuestType, GUEST_BINARIES};
pub use pool::{
    AcquireSandbox, GetPoolMetrics, GuestPoolMetrics, InitPool, InternalReleaseSandbox,
    PoolMetrics, PoolMetricsResponse, PooledSandbox, ReleaseSandbox, SandboxPool, WarmPool,
};
pub use provider::SandboxProvider;
pub use sandbox::HyperlightSandbox;

// AutoSandboxFactory is only available in tests (it silently falls back to stub)
#[cfg(test)]
pub use factory::AutoSandboxFactory;
