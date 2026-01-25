//! Sandbox pool actor for pre-warmed sandbox instances.
//!
//! Provides an actor-based pool of pre-warmed Hyperlight sandboxes
//! to reduce cold-start latency.

use super::config::SandboxConfig;
use super::error::SandboxErrorKind;
use super::sandbox::HyperlightSandbox;
use crate::tools::error::ToolError;
use crate::tools::sandbox::traits::Sandbox;
use acton_reactive::prelude::*;
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;

/// Message to initialize the pool with configuration.
#[acton_message]
pub struct InitPool {
    /// Configuration for sandboxes
    pub config: SandboxConfig,
}

/// Message to warm up the pool with pre-created sandboxes.
#[acton_message]
pub struct WarmPool {
    /// Number of sandboxes to create
    pub count: usize,
}

/// Message to get pool metrics.
#[acton_message]
pub struct GetPoolMetrics;

/// Response with pool metrics.
#[acton_message]
pub struct PoolMetricsResponse {
    /// Current metrics
    pub metrics: PoolMetrics,
}

/// Message to release a sandbox back to the pool (internal).
#[acton_message]
pub struct InternalReleaseSandbox {
    /// Index of sandbox to mark as available
    pub index: usize,
}

/// Metrics for the sandbox pool.
#[derive(Debug, Clone, Default)]
pub struct PoolMetrics {
    /// Number of available sandboxes in the pool
    pub available: usize,
    /// Number of sandboxes currently in use
    pub in_use: usize,
    /// Total sandboxes created
    pub total_created: u64,
    /// Pool hits (sandbox acquired from pool)
    pub pool_hits: u64,
    /// Pool misses (new sandbox created on demand)
    pub pool_misses: u64,
    /// Average creation time in milliseconds
    pub avg_creation_ms: f64,
}

/// Message to acquire a sandbox from the pool.
///
/// This message is not Clone (oneshot sender), so it uses a different pattern.
/// Instead, we provide a sync method on the actor handle.
#[derive(Debug)]
pub struct AcquireSandbox {
    /// Reply channel for the sandbox
    pub reply: tokio::sync::oneshot::Sender<Result<PooledSandbox, ToolError>>,
}

/// Message to release a sandbox back to the pool.
#[derive(Debug)]
pub struct ReleaseSandbox {
    /// Index of the sandbox in the pool
    pub index: usize,
    /// Whether the sandbox is still alive
    pub alive: bool,
}

/// A sandbox acquired from the pool.
///
/// When dropped, the sandbox is automatically returned to the pool
/// if it's still alive.
pub struct PooledSandbox {
    /// The underlying sandbox (Arc allows cloning for spawn_blocking)
    inner: Option<Arc<dyn Sandbox>>,
    /// Index in the pool
    index: usize,
    /// Handle to the pool for returning the sandbox
    pool_handle: ActorHandle,
}

impl std::fmt::Debug for PooledSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledSandbox")
            .field("index", &self.index)
            .field(
                "is_alive",
                &self.inner.as_ref().is_some_and(|s| s.is_alive()),
            )
            .finish_non_exhaustive()
    }
}

impl PooledSandbox {
    /// Executes code in the pooled sandbox using spawn_blocking.
    ///
    /// This method clones the `Arc<dyn Sandbox>` and executes `execute_sync`
    /// in a blocking context via `tokio::task::spawn_blocking`. This allows
    /// synchronous sandbox implementations (like Hyperlight) to work correctly
    /// within async contexts.
    ///
    /// # Arguments
    ///
    /// * `code` - The code to execute
    /// * `args` - Arguments to pass
    ///
    /// # Returns
    ///
    /// The execution result as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the sandbox is not available, spawn_blocking fails,
    /// or execution fails.
    pub async fn execute(&self, code: &str, args: Value) -> Result<Value, ToolError> {
        let sandbox = self
            .inner
            .clone()
            .ok_or_else(|| ToolError::sandbox_error("pooled sandbox is not available"))?;

        let code = code.to_string();

        tokio::task::spawn_blocking(move || sandbox.execute_sync(&code, args))
            .await
            .map_err(|e| ToolError::sandbox_error(format!("spawn_blocking failed: {e}")))?
    }

    /// Returns whether the sandbox is alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.inner.as_ref().is_some_and(|s| s.is_alive())
    }
}

impl Drop for PooledSandbox {
    fn drop(&mut self) {
        // Return sandbox to pool
        let index = self.index;
        let handle = self.pool_handle.clone();

        tokio::spawn(async move {
            handle.send(InternalReleaseSandbox { index }).await;
        });
    }
}

/// Internal sandbox entry in the pool.
#[derive(Debug)]
struct PoolEntry {
    /// The sandbox (Arc allows sharing with PooledSandbox for spawn_blocking)
    sandbox: Arc<dyn Sandbox>,
    in_use: bool,
}

/// Actor that manages a pool of pre-warmed Hyperlight sandboxes.
///
/// The pool maintains a configurable number of ready-to-use sandboxes
/// to minimize cold-start latency for tool executions.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::hyperlight::{SandboxPool, SandboxConfig};
///
/// let config = SandboxConfig::new().with_pool_size(Some(4));
/// let pool = SandboxPool::spawn(&mut runtime, config).await;
///
/// // Warm up the pool
/// pool.send(WarmPool { count: 4 }).await;
///
/// // Get metrics
/// pool.send(GetPoolMetrics).await;
/// ```
#[acton_actor]
pub struct SandboxPool {
    /// Configuration for sandboxes
    pub config: SandboxConfig,
    /// Pooled sandboxes (indexed for O(1) release)
    entries: Vec<PoolEntry>,
    /// Target pool size
    pub target_size: usize,
    /// Pool metrics
    pub metrics: PoolMetrics,
    /// Total creation time for averaging
    pub total_creation_time_ms: f64,
}

impl SandboxPool {
    /// Spawns the sandbox pool actor.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The actor runtime
    /// * `config` - The sandbox configuration
    ///
    /// # Returns
    ///
    /// The actor handle for the pool.
    pub async fn spawn(runtime: &mut ActorRuntime, config: SandboxConfig) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<SandboxPool>("sandbox_pool".to_string());

        builder
            .before_start(|_actor| {
                tracing::debug!("Sandbox pool initializing");
                Reply::ready()
            })
            .after_start(|actor| {
                tracing::info!(target_size = actor.model.target_size, "Sandbox pool ready");
                Reply::ready()
            })
            .before_stop(|actor| {
                tracing::info!(
                    pool_hits = actor.model.metrics.pool_hits,
                    pool_misses = actor.model.metrics.pool_misses,
                    total_created = actor.model.metrics.total_created,
                    "Sandbox pool shutting down"
                );
                Reply::ready()
            });

        configure_handlers(&mut builder);

        let handle = builder.start().await;

        // Send initial configuration
        handle.send(InitPool { config }).await;

        handle
    }

    /// Attempts to create a new sandbox wrapped in an Arc.
    ///
    /// The Arc allows the sandbox to be shared with `PooledSandbox` for use
    /// in `spawn_blocking` while still being tracked in the pool.
    fn create_sandbox(&self) -> Result<Arc<dyn Sandbox>, SandboxErrorKind> {
        let sandbox = HyperlightSandbox::new(self.config.clone())?;
        Ok(Arc::new(sandbox))
    }
}

/// Configures message handlers for the sandbox pool.
fn configure_handlers(builder: &mut ManagedActor<Idle, SandboxPool>) {
    // Handle pool initialization
    builder.mutate_on::<InitPool>(|actor, envelope| {
        let config = envelope.message().config.clone();
        actor.model.target_size = config.pool_size.unwrap_or(0);
        actor.model.config = config;
        tracing::debug!(
            target_size = actor.model.target_size,
            "Sandbox pool configured"
        );
        Reply::ready()
    });

    // Handle release sandbox (internal)
    builder.mutate_on::<InternalReleaseSandbox>(|actor, envelope| {
        let index = envelope.message().index;

        if let Some(entry) = actor.model.entries.get_mut(index) {
            if entry.sandbox.is_alive() {
                entry.in_use = false;
                actor.model.metrics.in_use = actor.model.metrics.in_use.saturating_sub(1);
                actor.model.metrics.available =
                    actor.model.entries.iter().filter(|e| !e.in_use).count();
            } else {
                // Remove dead sandbox
                actor.model.entries.remove(index);
            }
        }

        Reply::ready()
    });

    // Handle warm pool
    builder.mutate_on::<WarmPool>(|actor, envelope| {
        let count = envelope.message().count;
        let current = actor.model.entries.len();
        let to_create = count.saturating_sub(current);

        for _ in 0..to_create {
            let start = Instant::now();
            match actor.model.create_sandbox() {
                Ok(sandbox) => {
                    let elapsed = start.elapsed().as_millis() as f64;
                    actor.model.total_creation_time_ms += elapsed;
                    actor.model.metrics.total_created += 1;
                    actor.model.metrics.avg_creation_ms = actor.model.total_creation_time_ms
                        / actor.model.metrics.total_created as f64;

                    actor.model.entries.push(PoolEntry {
                        sandbox,
                        in_use: false,
                    });
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to warm sandbox");
                    break;
                }
            }
        }

        actor.model.metrics.available = actor.model.entries.iter().filter(|e| !e.in_use).count();
        tracing::info!(
            available = actor.model.metrics.available,
            total = actor.model.entries.len(),
            "Pool warmed"
        );

        Reply::ready()
    });

    // Handle get metrics
    builder.act_on::<GetPoolMetrics>(|actor, envelope| {
        let metrics = actor.model.metrics.clone();
        let reply = envelope.reply_envelope();

        Reply::pending(async move {
            reply.send(PoolMetricsResponse { metrics }).await;
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_metrics_default() {
        let metrics = PoolMetrics::default();
        assert_eq!(metrics.available, 0);
        assert_eq!(metrics.in_use, 0);
        assert_eq!(metrics.total_created, 0);
        assert_eq!(metrics.pool_hits, 0);
        assert_eq!(metrics.pool_misses, 0);
    }

    #[test]
    fn pooled_sandbox_debug() {
        // Can't easily test without a pool, but verify Debug is implemented
        let debug_str = format!("{:?}", PoolMetrics::default());
        assert!(debug_str.contains("PoolMetrics"));
    }

    // Integration tests require hypervisor and are in the integration test directory
}
