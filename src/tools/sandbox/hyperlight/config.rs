//! Sandbox configuration types.
//!
//! Provides configuration options for Hyperlight sandbox instances.

use super::error::SandboxErrorKind;
use std::path::PathBuf;
use std::time::Duration;

/// Default memory limit for sandbox instances (64 MB).
pub const DEFAULT_MEMORY_LIMIT: usize = 64 * 1024 * 1024;

/// Default execution timeout (30 seconds).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default sandbox pool size.
pub const DEFAULT_POOL_SIZE: usize = 4;

/// Default warmup count per guest type.
pub const DEFAULT_WARMUP_COUNT: usize = 4;

/// Default maximum sandboxes per guest type.
pub const DEFAULT_MAX_PER_TYPE: usize = 32;

/// Default maximum executions before recycling a sandbox.
pub const DEFAULT_MAX_EXECUTIONS_BEFORE_RECYCLE: usize = 1000;

/// Source for the guest binary.
///
/// Hyperlight requires a specially compiled guest binary that runs
/// inside the micro-VM.
#[derive(Debug, Clone)]
pub enum GuestBinarySource {
    /// Use the embedded shell executor guest binary.
    ///
    /// This is the recommended option for most use cases, as it provides
    /// a pre-built guest capable of executing shell commands.
    Embedded,

    /// Load a custom guest binary from a file path.
    ///
    /// Use this for custom guest implementations or testing.
    FromPath(PathBuf),

    /// Use guest binary data directly from memory.
    ///
    /// Use this when the guest binary is already loaded or embedded
    /// in the host application.
    FromBytes(Vec<u8>),
}

impl Default for GuestBinarySource {
    fn default() -> Self {
        Self::Embedded
    }
}

/// Configuration for Hyperlight sandbox instances.
///
/// This struct provides a builder pattern for configuring sandbox
/// memory limits, timeouts, and pooling behavior.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::SandboxConfig;
/// use std::time::Duration;
///
/// let config = SandboxConfig::new()
///     .with_memory_limit(128 * 1024 * 1024)  // 128 MB
///     .with_timeout(Duration::from_secs(60))
///     .with_pool_size(8);
/// ```
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum memory available to the guest (in bytes).
    ///
    /// Default: 64 MB
    pub memory_limit: usize,

    /// Maximum execution time before timeout.
    ///
    /// Default: 30 seconds
    pub timeout: Duration,

    /// Source of the guest binary to execute.
    ///
    /// Default: Embedded shell executor
    pub guest_binary: GuestBinarySource,

    /// Number of pre-warmed sandbox instances in the pool.
    ///
    /// Set to `None` to disable pooling (each request creates a new sandbox).
    /// Default: 4 instances
    pub pool_size: Option<usize>,

    /// Whether to enable debug output from the guest.
    ///
    /// When enabled, guest print statements are forwarded to the host's
    /// tracing output.
    /// Default: false
    pub debug_output: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            memory_limit: DEFAULT_MEMORY_LIMIT,
            timeout: DEFAULT_TIMEOUT,
            guest_binary: GuestBinarySource::default(),
            pool_size: Some(DEFAULT_POOL_SIZE),
            debug_output: false,
        }
    }
}

impl SandboxConfig {
    /// Creates a new sandbox configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the memory limit for sandbox instances.
    ///
    /// # Arguments
    ///
    /// * `limit` - Memory limit in bytes (must be at least 1 MB)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = SandboxConfig::new()
    ///     .with_memory_limit(128 * 1024 * 1024); // 128 MB
    /// ```
    #[must_use]
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Sets the execution timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum execution duration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = SandboxConfig::new()
    ///     .with_timeout(Duration::from_secs(60));
    /// ```
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the guest binary source.
    ///
    /// # Arguments
    ///
    /// * `source` - The guest binary source
    #[must_use]
    pub fn with_guest_binary(mut self, source: GuestBinarySource) -> Self {
        self.guest_binary = source;
        self
    }

    /// Sets the sandbox pool size.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of pre-warmed sandboxes, or `None` to disable pooling
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use a pool of 8 sandboxes
    /// let config = SandboxConfig::new().with_pool_size(Some(8));
    ///
    /// // Disable pooling (create new sandbox for each request)
    /// let config = SandboxConfig::new().with_pool_size(None);
    /// ```
    #[must_use]
    pub fn with_pool_size(mut self, size: Option<usize>) -> Self {
        self.pool_size = size;
        self
    }

    /// Enables debug output from the guest.
    #[must_use]
    pub fn with_debug_output(mut self, enabled: bool) -> Self {
        self.debug_output = enabled;
        self
    }

    /// Disables the sandbox pool (each request creates a new sandbox).
    #[must_use]
    pub fn without_pool(self) -> Self {
        self.with_pool_size(None)
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns `SandboxErrorKind::InvalidConfiguration` if any values are invalid.
    pub fn validate(&self) -> Result<(), SandboxErrorKind> {
        // Memory limit must be at least 1 MB
        const MIN_MEMORY: usize = 1024 * 1024;
        if self.memory_limit < MIN_MEMORY {
            return Err(SandboxErrorKind::InvalidConfiguration {
                field: "memory_limit".to_string(),
                reason: format!("must be at least 1 MB, got {} bytes", self.memory_limit),
            });
        }

        // Timeout must be positive
        if self.timeout.is_zero() {
            return Err(SandboxErrorKind::InvalidConfiguration {
                field: "timeout".to_string(),
                reason: "must be greater than zero".to_string(),
            });
        }

        // Pool size, if Some, must be positive
        if let Some(0) = self.pool_size {
            return Err(SandboxErrorKind::InvalidConfiguration {
                field: "pool_size".to_string(),
                reason: "must be greater than zero or None".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_values() {
        let config = SandboxConfig::default();
        assert_eq!(config.memory_limit, DEFAULT_MEMORY_LIMIT);
        assert_eq!(config.timeout, DEFAULT_TIMEOUT);
        assert_eq!(config.pool_size, Some(DEFAULT_POOL_SIZE));
        assert!(!config.debug_output);
    }

    #[test]
    fn config_builder_memory_limit() {
        let config = SandboxConfig::new().with_memory_limit(128 * 1024 * 1024);
        assert_eq!(config.memory_limit, 128 * 1024 * 1024);
    }

    #[test]
    fn config_builder_timeout() {
        let config = SandboxConfig::new().with_timeout(Duration::from_secs(60));
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn config_builder_pool_size() {
        let config = SandboxConfig::new().with_pool_size(Some(8));
        assert_eq!(config.pool_size, Some(8));
    }

    #[test]
    fn config_without_pool() {
        let config = SandboxConfig::new().without_pool();
        assert_eq!(config.pool_size, None);
    }

    #[test]
    fn config_debug_output() {
        let config = SandboxConfig::new().with_debug_output(true);
        assert!(config.debug_output);
    }

    #[test]
    fn config_validate_valid() {
        let config = SandboxConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validate_memory_too_small() {
        let config = SandboxConfig::new().with_memory_limit(1000); // Less than 1 MB
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SandboxErrorKind::InvalidConfiguration { .. }));
        assert!(err.to_string().contains("memory_limit"));
    }

    #[test]
    fn config_validate_zero_timeout() {
        let config = SandboxConfig::new().with_timeout(Duration::ZERO);
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SandboxErrorKind::InvalidConfiguration { .. }));
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn config_validate_zero_pool_size() {
        let config = SandboxConfig::new().with_pool_size(Some(0));
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SandboxErrorKind::InvalidConfiguration { .. }));
        assert!(err.to_string().contains("pool_size"));
    }

    #[test]
    fn config_validate_none_pool_size() {
        let config = SandboxConfig::new().with_pool_size(None);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn guest_binary_source_default_is_embedded() {
        let source = GuestBinarySource::default();
        assert!(matches!(source, GuestBinarySource::Embedded));
    }

    #[test]
    fn config_is_clone() {
        let config1 = SandboxConfig::new().with_memory_limit(100 * 1024 * 1024);
        let config2 = config1.clone();
        assert_eq!(config1.memory_limit, config2.memory_limit);
    }
}

/// Configuration for the sandbox pool.
///
/// Controls pool behavior including warmup counts, maximum pool sizes,
/// and recycling thresholds per guest type.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::hyperlight::PoolConfig;
///
/// let config = PoolConfig::default()
///     .with_warmup_count(8)
///     .with_max_per_type(64);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolConfig {
    /// Number of sandboxes to pre-warm per guest type.
    ///
    /// Default: 4
    pub warmup_count: usize,

    /// Maximum sandboxes per guest type.
    ///
    /// Default: 32
    pub max_per_type: usize,

    /// Maximum executions before recycling a sandbox.
    ///
    /// After this many executions, the sandbox is discarded and replaced
    /// with a fresh instance to prevent resource leaks.
    ///
    /// Default: 1000
    pub max_executions_before_recycle: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            warmup_count: DEFAULT_WARMUP_COUNT,
            max_per_type: DEFAULT_MAX_PER_TYPE,
            max_executions_before_recycle: DEFAULT_MAX_EXECUTIONS_BEFORE_RECYCLE,
        }
    }
}

impl PoolConfig {
    /// Creates a new pool configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the warmup count per guest type.
    #[must_use]
    pub fn with_warmup_count(mut self, count: usize) -> Self {
        self.warmup_count = count;
        self
    }

    /// Sets the maximum sandboxes per guest type.
    #[must_use]
    pub fn with_max_per_type(mut self, max: usize) -> Self {
        self.max_per_type = max;
        self
    }

    /// Sets the maximum executions before recycling.
    #[must_use]
    pub fn with_max_executions_before_recycle(mut self, max: usize) -> Self {
        self.max_executions_before_recycle = max;
        self
    }

    /// Validates the pool configuration.
    ///
    /// # Errors
    ///
    /// Returns `SandboxErrorKind::InvalidConfiguration` if values are invalid.
    pub fn validate(&self) -> Result<(), SandboxErrorKind> {
        if self.max_per_type == 0 {
            return Err(SandboxErrorKind::InvalidConfiguration {
                field: "max_per_type".to_string(),
                reason: "must be greater than zero".to_string(),
            });
        }

        if self.max_executions_before_recycle == 0 {
            return Err(SandboxErrorKind::InvalidConfiguration {
                field: "max_executions_before_recycle".to_string(),
                reason: "must be greater than zero".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod pool_config_tests {
    use super::*;

    #[test]
    fn pool_config_default_values() {
        let config = PoolConfig::default();
        assert_eq!(config.warmup_count, DEFAULT_WARMUP_COUNT);
        assert_eq!(config.max_per_type, DEFAULT_MAX_PER_TYPE);
        assert_eq!(
            config.max_executions_before_recycle,
            DEFAULT_MAX_EXECUTIONS_BEFORE_RECYCLE
        );
    }

    #[test]
    fn pool_config_builder_warmup_count() {
        let config = PoolConfig::new().with_warmup_count(8);
        assert_eq!(config.warmup_count, 8);
    }

    #[test]
    fn pool_config_builder_max_per_type() {
        let config = PoolConfig::new().with_max_per_type(64);
        assert_eq!(config.max_per_type, 64);
    }

    #[test]
    fn pool_config_builder_max_executions() {
        let config = PoolConfig::new().with_max_executions_before_recycle(500);
        assert_eq!(config.max_executions_before_recycle, 500);
    }

    #[test]
    fn pool_config_validate_valid() {
        let config = PoolConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pool_config_validate_zero_max_per_type() {
        let config = PoolConfig::new().with_max_per_type(0);
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SandboxErrorKind::InvalidConfiguration { .. }));
        assert!(err.to_string().contains("max_per_type"));
    }

    #[test]
    fn pool_config_validate_zero_max_executions() {
        let config = PoolConfig::new().with_max_executions_before_recycle(0);
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SandboxErrorKind::InvalidConfiguration { .. }));
        assert!(err.to_string().contains("max_executions"));
    }

    #[test]
    fn pool_config_is_clone() {
        let config1 = PoolConfig::new().with_warmup_count(10);
        let config2 = config1.clone();
        assert_eq!(config1, config2);
    }

    #[test]
    fn pool_config_is_debug() {
        let config = PoolConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("PoolConfig"));
        assert!(debug.contains("warmup_count"));
    }

    #[test]
    fn pool_config_equality() {
        let config1 = PoolConfig::new().with_warmup_count(10);
        let config2 = PoolConfig::new().with_warmup_count(10);
        let config3 = PoolConfig::new().with_warmup_count(20);
        assert_eq!(config1, config2);
        assert_ne!(config1, config3);
    }
}
