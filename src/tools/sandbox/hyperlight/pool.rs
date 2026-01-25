//! Sandbox pool actor for pre-warmed sandbox instances.
//!
//! Provides an actor-based pool of pre-warmed Hyperlight sandboxes
//! to reduce cold-start latency. Supports multiple guest types with
//! separate pools and metrics for each.

use super::config::{PoolConfig, SandboxConfig};
use super::error::SandboxErrorKind;
use super::guest::GuestType;
use super::sandbox::HyperlightSandbox;
use crate::tools::error::ToolError;
use crate::tools::sandbox::traits::Sandbox;
use acton_reactive::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Message to initialize the pool with configuration.
#[acton_message]
pub struct InitPool {
    /// Configuration for sandboxes
    pub sandbox_config: SandboxConfig,
    /// Configuration for the pool
    pub pool_config: PoolConfig,
}

/// Message to warm up the pool with pre-created sandboxes.
#[acton_message]
pub struct WarmPool {
    /// Number of sandboxes to create per guest type
    pub count: usize,
    /// Specific guest type to warm, or None for all types
    pub guest_type: Option<GuestType>,
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
    /// The guest type of the sandbox
    pub guest_type: GuestType,
    /// Index of sandbox to mark as available
    pub index: usize,
    /// Number of executions performed
    pub execution_count: usize,
}

/// Metrics for a single guest type's pool.
#[derive(Debug, Clone, Default)]
pub struct GuestPoolMetrics {
    /// Number of available sandboxes for this guest type
    pub available: usize,
    /// Number of sandboxes currently in use
    pub in_use: usize,
    /// Total sandboxes created for this guest type
    pub total_created: u64,
    /// Pool hits (sandbox acquired from pool)
    pub pool_hits: u64,
    /// Pool misses (new sandbox created on demand)
    pub pool_misses: u64,
    /// Average creation time in milliseconds
    pub avg_creation_ms: f64,
    /// Total creation time for averaging (internal)
    total_creation_time_ms: f64,
}

impl GuestPoolMetrics {
    /// Records a new sandbox creation with the given duration.
    pub fn record_creation(&mut self, duration_ms: f64) {
        self.total_created += 1;
        self.total_creation_time_ms += duration_ms;
        self.avg_creation_ms = self.total_creation_time_ms / self.total_created as f64;
    }
}

/// Aggregated metrics for the entire sandbox pool.
#[derive(Debug, Clone, Default)]
pub struct PoolMetrics {
    /// Metrics per guest type
    pub by_guest_type: HashMap<GuestType, GuestPoolMetrics>,
    /// Total available across all guest types
    pub total_available: usize,
    /// Total in use across all guest types
    pub total_in_use: usize,
}

impl PoolMetrics {
    /// Updates aggregate totals from per-guest-type metrics.
    pub fn update_totals(&mut self) {
        self.total_available = self.by_guest_type.values().map(|m| m.available).sum();
        self.total_in_use = self.by_guest_type.values().map(|m| m.in_use).sum();
    }
}

/// Message to acquire a sandbox from the pool.
///
/// This message is not Clone (oneshot sender), so it uses a different pattern.
/// Instead, we provide a sync method on the actor handle.
#[derive(Debug)]
pub struct AcquireSandbox {
    /// The guest type to acquire
    pub guest_type: GuestType,
    /// Reply channel for the sandbox
    pub reply: tokio::sync::oneshot::Sender<Result<PooledSandbox, ToolError>>,
}

/// Message to release a sandbox back to the pool.
#[derive(Debug)]
pub struct ReleaseSandbox {
    /// The guest type of the sandbox
    pub guest_type: GuestType,
    /// Index of the sandbox in the pool
    pub index: usize,
    /// Whether the sandbox is still alive
    pub alive: bool,
}

/// A sandbox acquired from the pool.
///
/// When dropped, the sandbox is automatically returned to the pool
/// if it's still alive and hasn't exceeded the execution limit.
pub struct PooledSandbox {
    /// The underlying sandbox (Arc allows cloning for spawn_blocking)
    inner: Option<Arc<dyn Sandbox>>,
    /// Index in the guest-type pool
    index: usize,
    /// The guest type of this sandbox
    guest_type: GuestType,
    /// Number of executions performed
    execution_count: usize,
    /// Handle to the pool for returning the sandbox
    pool_handle: ActorHandle,
}

impl std::fmt::Debug for PooledSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledSandbox")
            .field("index", &self.index)
            .field("guest_type", &self.guest_type)
            .field("execution_count", &self.execution_count)
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
    pub async fn execute(&mut self, code: &str, args: Value) -> Result<Value, ToolError> {
        let sandbox = self
            .inner
            .clone()
            .ok_or_else(|| ToolError::sandbox_error("pooled sandbox is not available"))?;

        let code = code.to_string();

        let result = tokio::task::spawn_blocking(move || sandbox.execute_sync(&code, args))
            .await
            .map_err(|e| ToolError::sandbox_error(format!("spawn_blocking failed: {e}")))?;

        self.execution_count += 1;
        result
    }

    /// Returns whether the sandbox is alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.inner.as_ref().is_some_and(|s| s.is_alive())
    }

    /// Returns the guest type of this sandbox.
    #[must_use]
    pub fn guest_type(&self) -> GuestType {
        self.guest_type
    }

    /// Returns the number of executions performed.
    #[must_use]
    pub fn execution_count(&self) -> usize {
        self.execution_count
    }
}

impl Drop for PooledSandbox {
    fn drop(&mut self) {
        // Return sandbox to pool
        let guest_type = self.guest_type;
        let index = self.index;
        let execution_count = self.execution_count;
        let handle = self.pool_handle.clone();

        tokio::spawn(async move {
            handle
                .send(InternalReleaseSandbox {
                    guest_type,
                    index,
                    execution_count,
                })
                .await;
        });
    }
}

/// Internal pool for a single guest type.
#[derive(Debug)]
struct GuestPool {
    /// Pooled sandboxes (indexed for O(1) release)
    entries: Vec<PoolEntry>,
    /// Metrics for this guest type
    metrics: GuestPoolMetrics,
}

impl GuestPool {
    /// Creates a new empty guest pool.
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            metrics: GuestPoolMetrics::default(),
        }
    }

    /// Returns the number of available sandboxes.
    fn available_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.in_use).count()
    }
}

/// Internal sandbox entry in the pool.
#[derive(Debug)]
struct PoolEntry {
    /// The sandbox (Arc allows sharing with PooledSandbox for spawn_blocking)
    sandbox: Arc<dyn Sandbox>,
    /// Whether currently in use
    in_use: bool,
    /// Number of executions performed
    execution_count: usize,
}

/// Actor that manages pools of pre-warmed Hyperlight sandboxes by guest type.
///
/// Maintains separate pools for each guest type (Shell, Http) to minimize
/// cold-start latency for tool executions.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::hyperlight::{SandboxPool, SandboxConfig, PoolConfig};
///
/// let sandbox_config = SandboxConfig::default();
/// let pool_config = PoolConfig::default().with_warmup_count(8);
/// let pool = SandboxPool::spawn(&mut runtime, sandbox_config, pool_config).await;
///
/// // Warm up the pool for all guest types
/// pool.send(WarmPool { count: 4, guest_type: None }).await;
///
/// // Get metrics
/// pool.send(GetPoolMetrics).await;
/// ```
#[acton_actor]
pub struct SandboxPool {
    /// Configuration for individual sandboxes
    pub sandbox_config: SandboxConfig,
    /// Configuration for the pool itself
    pub pool_config: PoolConfig,
    /// Per-guest-type pools
    pools: HashMap<GuestType, GuestPool>,
}

impl SandboxPool {
    /// Spawns the sandbox pool actor with configurations.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The actor runtime
    /// * `sandbox_config` - Configuration for individual sandboxes
    /// * `pool_config` - Configuration for the pool
    ///
    /// # Returns
    ///
    /// The actor handle for the pool.
    pub async fn spawn(
        runtime: &mut ActorRuntime,
        sandbox_config: SandboxConfig,
        pool_config: PoolConfig,
    ) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<SandboxPool>("sandbox_pool".to_string());

        builder
            .before_start(|_actor| {
                tracing::debug!("Sandbox pool initializing");
                Reply::ready()
            })
            .after_start(|actor| {
                let pool_count = actor.model.pools.len();
                tracing::info!(
                    max_per_type = actor.model.pool_config.max_per_type,
                    guest_types = pool_count,
                    "Sandbox pool ready"
                );
                Reply::ready()
            })
            .before_stop(|actor| {
                let total_hits: u64 = actor
                    .model
                    .pools
                    .values()
                    .map(|p| p.metrics.pool_hits)
                    .sum();
                let total_misses: u64 = actor
                    .model
                    .pools
                    .values()
                    .map(|p| p.metrics.pool_misses)
                    .sum();
                let total_created: u64 = actor
                    .model
                    .pools
                    .values()
                    .map(|p| p.metrics.total_created)
                    .sum();
                tracing::info!(
                    pool_hits = total_hits,
                    pool_misses = total_misses,
                    total_created = total_created,
                    "Sandbox pool shutting down"
                );
                Reply::ready()
            });

        configure_handlers(&mut builder);

        let handle = builder.start().await;

        // Send initial configuration
        handle
            .send(InitPool {
                sandbox_config,
                pool_config,
            })
            .await;

        handle
    }

    /// Creates a sandbox for the specified guest type.
    fn create_sandbox(&self, guest_type: GuestType) -> Result<Arc<dyn Sandbox>, SandboxErrorKind> {
        let sandbox = HyperlightSandbox::new_with_guest_type(
            self.sandbox_config.clone(),
            guest_type,
        )?;
        Ok(Arc::new(sandbox))
    }

    /// Initializes sub-pools for all known guest types.
    fn init_pools() -> HashMap<GuestType, GuestPool> {
        GuestType::all().map(|gt| (gt, GuestPool::new())).collect()
    }

    /// Acquires an available sandbox or creates a new one.
    ///
    /// This method finds an available pre-warmed sandbox or creates a new one
    /// if the pool hasn't reached capacity. The returned `PooledSandbox` will
    /// automatically return itself to the pool when dropped.
    ///
    /// # Arguments
    ///
    /// * `guest_type` - The type of guest sandbox to acquire
    /// * `pool_handle` - Handle to this pool actor for sandbox return on drop
    ///
    /// # Returns
    ///
    /// A pooled sandbox ready for execution, or an error if:
    /// - The guest type is not available in the pool
    /// - The pool is exhausted (at capacity with all sandboxes in use)
    /// - Sandbox creation fails
    ///
    /// # Note
    ///
    /// This method is called internally by the pool's message handlers but is
    /// also exposed for direct access when needed.
    pub fn acquire_from_pool(
        &mut self,
        guest_type: GuestType,
        pool_handle: ActorHandle,
    ) -> Result<PooledSandbox, SandboxErrorKind> {
        let max_per_type = self.pool_config.max_per_type;

        // First check if the pool exists
        if !self.pools.contains_key(&guest_type) {
            return Err(SandboxErrorKind::InvalidGuestType { guest_type });
        }

        // Try to find an available sandbox
        let pool = self.pools.get_mut(&guest_type).unwrap();
        let found_index = pool
            .entries
            .iter()
            .enumerate()
            .find(|(_, entry)| !entry.in_use && entry.sandbox.is_alive())
            .map(|(index, _)| index);

        if let Some(index) = found_index {
            let entry = &mut pool.entries[index];
            entry.in_use = true;
            let sandbox = entry.sandbox.clone();
            let execution_count = entry.execution_count;

            pool.metrics.in_use += 1;
            pool.metrics.pool_hits += 1;
            pool.metrics.available = pool.entries.iter().filter(|e| !e.in_use).count();

            return Ok(PooledSandbox {
                inner: Some(sandbox),
                index,
                guest_type,
                execution_count,
                pool_handle,
            });
        }

        // No available sandbox, create new if under limit
        let pool_len = pool.entries.len();
        if pool_len < max_per_type {
            let start = Instant::now();
            let sandbox = self.create_sandbox(guest_type)?;
            let elapsed = start.elapsed().as_millis() as f64;

            let pool = self.pools.get_mut(&guest_type).unwrap();
            pool.metrics.record_creation(elapsed);
            pool.metrics.pool_misses += 1;

            let index = pool.entries.len();
            pool.entries.push(PoolEntry {
                sandbox: sandbox.clone(),
                in_use: true,
                execution_count: 0,
            });
            pool.metrics.in_use += 1;

            return Ok(PooledSandbox {
                inner: Some(sandbox),
                index,
                guest_type,
                execution_count: 0,
                pool_handle,
            });
        }

        // Pool exhausted
        Err(SandboxErrorKind::PoolExhausted {
            pool_size: pool_len,
        })
    }
}

/// Configures message handlers for the sandbox pool.
fn configure_handlers(builder: &mut ManagedActor<Idle, SandboxPool>) {
    // Handle pool initialization
    builder.mutate_on::<InitPool>(|actor, envelope| {
        let msg = envelope.message();
        actor.model.sandbox_config = msg.sandbox_config.clone();
        actor.model.pool_config = msg.pool_config.clone();
        actor.model.pools = SandboxPool::init_pools();
        tracing::debug!(
            max_per_type = actor.model.pool_config.max_per_type,
            warmup_count = actor.model.pool_config.warmup_count,
            "Sandbox pool configured"
        );
        Reply::ready()
    });

    // Handle release sandbox (internal)
    builder.mutate_on::<InternalReleaseSandbox>(|actor, envelope| {
        let msg = envelope.message();
        let guest_type = msg.guest_type;
        let index = msg.index;
        let execution_count = msg.execution_count;
        let max_executions = actor.model.pool_config.max_executions_before_recycle;

        if let Some(pool) = actor.model.pools.get_mut(&guest_type) {
            if let Some(entry) = pool.entries.get_mut(index) {
                entry.execution_count = execution_count;

                // Check if sandbox should be recycled
                if execution_count >= max_executions || !entry.sandbox.is_alive() {
                    // Remove the exhausted/dead sandbox
                    pool.entries.remove(index);
                    tracing::debug!(
                        guest_type = %guest_type,
                        execution_count = execution_count,
                        "Sandbox recycled"
                    );
                } else {
                    // Return to pool
                    entry.in_use = false;
                    pool.metrics.in_use = pool.metrics.in_use.saturating_sub(1);
                    pool.metrics.available = pool.available_count();
                }
            }
        }

        Reply::ready()
    });

    // Handle warm pool
    builder.mutate_on::<WarmPool>(|actor, envelope| {
        let msg = envelope.message();
        let count = msg.count;
        let requested_guest_type = msg.guest_type;

        let guest_types: Vec<GuestType> = match requested_guest_type {
            Some(gt) => vec![gt],
            None => GuestType::all().collect(),
        };

        for guest_type in guest_types {
            // Determine how many sandboxes to create
            let to_create = {
                let pool = match actor.model.pools.get(&guest_type) {
                    Some(p) => p,
                    None => continue,
                };
                count.saturating_sub(pool.entries.len())
            };

            // Create sandboxes one at a time, re-acquiring mutable borrow each time
            for _ in 0..to_create {
                let start = Instant::now();
                let sandbox_result = actor.model.create_sandbox(guest_type);

                let pool = match actor.model.pools.get_mut(&guest_type) {
                    Some(p) => p,
                    None => break,
                };

                match sandbox_result {
                    Ok(sandbox) => {
                        let elapsed = start.elapsed().as_millis() as f64;
                        pool.metrics.record_creation(elapsed);

                        pool.entries.push(PoolEntry {
                            sandbox,
                            in_use: false,
                            execution_count: 0,
                        });
                    }
                    Err(e) => {
                        tracing::warn!(
                            guest_type = %guest_type,
                            error = %e,
                            "Failed to warm sandbox"
                        );
                        break;
                    }
                }
            }

            // Update available count and log
            if let Some(pool) = actor.model.pools.get_mut(&guest_type) {
                pool.metrics.available = pool.available_count();
                tracing::info!(
                    guest_type = %guest_type,
                    available = pool.metrics.available,
                    total = pool.entries.len(),
                    "Pool warmed"
                );
            }
        }

        Reply::ready()
    });

    // Handle get metrics
    builder.act_on::<GetPoolMetrics>(|actor, envelope| {
        let mut metrics = PoolMetrics::default();

        for (guest_type, pool) in &actor.model.pools {
            metrics
                .by_guest_type
                .insert(*guest_type, pool.metrics.clone());
        }
        metrics.update_totals();

        let reply = envelope.reply_envelope();

        Reply::pending(async move {
            reply.send(PoolMetricsResponse { metrics }).await;
        })
    });

    // Note: AcquireSandbox cannot be registered as a handler because it contains
    // a oneshot::Sender which is not Clone. Instead, use the AcquireSandboxRequest
    // pattern with a separate response message, or access the pool through
    // a service layer that manages acquisition.
}

#[cfg(test)]
mod tests {
    use super::*;

    // GuestPoolMetrics tests
    #[test]
    fn guest_pool_metrics_default() {
        let metrics = GuestPoolMetrics::default();
        assert_eq!(metrics.available, 0);
        assert_eq!(metrics.in_use, 0);
        assert_eq!(metrics.total_created, 0);
        assert_eq!(metrics.pool_hits, 0);
        assert_eq!(metrics.pool_misses, 0);
        assert!((metrics.avg_creation_ms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn guest_pool_metrics_record_creation() {
        let mut metrics = GuestPoolMetrics::default();
        metrics.record_creation(10.0);
        assert_eq!(metrics.total_created, 1);
        assert!((metrics.avg_creation_ms - 10.0).abs() < f64::EPSILON);

        metrics.record_creation(20.0);
        assert_eq!(metrics.total_created, 2);
        assert!((metrics.avg_creation_ms - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn guest_pool_metrics_is_clone() {
        let mut metrics = GuestPoolMetrics::default();
        metrics.record_creation(10.0);
        let cloned = metrics.clone();
        assert_eq!(metrics.total_created, cloned.total_created);
        assert!((metrics.avg_creation_ms - cloned.avg_creation_ms).abs() < f64::EPSILON);
    }

    #[test]
    fn guest_pool_metrics_is_debug() {
        let metrics = GuestPoolMetrics::default();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("GuestPoolMetrics"));
        assert!(debug.contains("available"));
    }

    // PoolMetrics tests
    #[test]
    fn pool_metrics_default() {
        let metrics = PoolMetrics::default();
        assert_eq!(metrics.total_available, 0);
        assert_eq!(metrics.total_in_use, 0);
        assert!(metrics.by_guest_type.is_empty());
    }

    #[test]
    fn pool_metrics_update_totals() {
        let mut metrics = PoolMetrics::default();

        let mut shell_metrics = GuestPoolMetrics::default();
        shell_metrics.available = 3;
        shell_metrics.in_use = 1;

        let mut http_metrics = GuestPoolMetrics::default();
        http_metrics.available = 2;
        http_metrics.in_use = 2;

        metrics
            .by_guest_type
            .insert(GuestType::Shell, shell_metrics);
        metrics.by_guest_type.insert(GuestType::Http, http_metrics);
        metrics.update_totals();

        assert_eq!(metrics.total_available, 5);
        assert_eq!(metrics.total_in_use, 3);
    }

    #[test]
    fn pool_metrics_is_clone() {
        let mut metrics = PoolMetrics::default();
        let mut shell_metrics = GuestPoolMetrics::default();
        shell_metrics.available = 5;
        metrics
            .by_guest_type
            .insert(GuestType::Shell, shell_metrics);
        metrics.update_totals();

        let cloned = metrics.clone();
        assert_eq!(metrics.total_available, cloned.total_available);
    }

    #[test]
    fn pool_metrics_is_debug() {
        let metrics = PoolMetrics::default();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("PoolMetrics"));
        assert!(debug.contains("total_available"));
    }

    // GuestPool tests
    #[test]
    fn guest_pool_new() {
        let pool = GuestPool::new();
        assert!(pool.entries.is_empty());
        assert_eq!(pool.available_count(), 0);
    }

    // PooledSandbox tests
    #[test]
    fn pooled_sandbox_guest_type() {
        // Verify the getter method signature exists (can't construct without pool)
        let _: fn(&PooledSandbox) -> GuestType = PooledSandbox::guest_type;
    }

    #[test]
    fn pooled_sandbox_execution_count() {
        // Verify the getter method signature exists (can't construct without pool)
        let _: fn(&PooledSandbox) -> usize = PooledSandbox::execution_count;
    }

    // WarmPool tests
    #[test]
    fn warm_pool_with_specific_guest_type() {
        let warm = WarmPool {
            count: 4,
            guest_type: Some(GuestType::Shell),
        };
        assert_eq!(warm.count, 4);
        assert_eq!(warm.guest_type, Some(GuestType::Shell));
    }

    #[test]
    fn warm_pool_for_all_guest_types() {
        let warm = WarmPool {
            count: 4,
            guest_type: None,
        };
        assert_eq!(warm.count, 4);
        assert!(warm.guest_type.is_none());
    }

    // InternalReleaseSandbox tests
    #[test]
    fn internal_release_sandbox_fields() {
        let release = InternalReleaseSandbox {
            guest_type: GuestType::Http,
            index: 5,
            execution_count: 10,
        };
        assert_eq!(release.guest_type, GuestType::Http);
        assert_eq!(release.index, 5);
        assert_eq!(release.execution_count, 10);
    }

    // Integration tests require hypervisor - marked as ignored
    #[test]
    #[ignore = "requires hypervisor"]
    fn pool_creates_sandboxes_for_all_guest_types() {
        // Would test that pools are initialized for Shell and Http
    }

    #[test]
    #[ignore = "requires hypervisor"]
    fn pool_acquires_correct_guest_type() {
        // Would test acquiring Shell vs Http sandboxes
    }

    #[test]
    #[ignore = "requires hypervisor"]
    fn pool_respects_max_per_type_limit() {
        // Would test that pool exhaustion happens per type
    }

    #[test]
    #[ignore = "requires hypervisor"]
    fn pool_recycles_sandbox_after_max_executions() {
        // Would test execution count tracking and recycling
    }
}
