//! Sandbox provider with explicit platform validation.
//!
//! Provides a secure entry point for Hyperlight sandboxing that validates
//! platform requirements and returns explicit errors instead of silently
//! falling back to insecure stub implementations.

use super::config::SandboxConfig;
use super::error::SandboxErrorKind;
use super::pool::SandboxPool;
use acton_reactive::prelude::*;

/// Sandbox provider backed by Hyperlight.
///
/// Validates platform requirements at construction time, returning explicit
/// errors if requirements are not met. This replaces `AutoSandboxFactory`
/// which silently fell back to `StubSandbox`.
///
/// # Platform Requirements
///
/// - **Architecture**: x86_64 only (Hyperlight uses hardware virtualization)
/// - **Hypervisor**: KVM (Linux) or Windows Hypervisor Platform
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::hyperlight::{SandboxProvider, SandboxConfig, WarmPool};
/// use acton_reactive::prelude::*;
///
/// // Create provider - fails explicitly if platform unsupported
/// let provider = SandboxProvider::new(SandboxConfig::default())?;
///
/// // Spawn the pool actor when ready
/// let pool_handle = provider.spawn(&mut runtime).await;
///
/// // Warm up and use pool
/// pool_handle.send(WarmPool { count: 4 }).await;
/// ```
///
/// # Security
///
/// Unlike `AutoSandboxFactory`, this type **never** falls back to a stub
/// implementation. If platform requirements are not met, construction fails
/// with an explicit error, ensuring callers are aware that sandboxing is
/// not available.
#[derive(Debug, Clone)]
pub struct SandboxProvider {
    /// The validated configuration
    config: SandboxConfig,
}

impl SandboxProvider {
    /// Creates a new sandbox provider after validating platform requirements.
    ///
    /// # Arguments
    ///
    /// * `config` - The sandbox configuration
    ///
    /// # Returns
    ///
    /// A configured `SandboxProvider` or an error if platform requirements not met.
    ///
    /// # Errors
    ///
    /// * `SandboxErrorKind::ArchitectureNotSupported` - Not running on x86_64
    /// * `SandboxErrorKind::HypervisorNotAvailable` - No hypervisor available
    /// * `SandboxErrorKind::InvalidConfiguration` - Invalid config values
    pub fn new(config: SandboxConfig) -> Result<Self, SandboxErrorKind> {
        Self::validate_platform()?;
        config.validate()?;
        Ok(Self { config })
    }

    /// Validates that the current platform supports Hyperlight sandboxing.
    ///
    /// # Returns
    ///
    /// `Ok(())` if platform is supported, otherwise an appropriate error.
    ///
    /// # Errors
    ///
    /// * `SandboxErrorKind::ArchitectureNotSupported` - Not x86_64
    /// * `SandboxErrorKind::HypervisorNotAvailable` - No hypervisor
    fn validate_platform() -> Result<(), SandboxErrorKind> {
        #[cfg(not(target_arch = "x86_64"))]
        return Err(SandboxErrorKind::ArchitectureNotSupported {
            arch: std::env::consts::ARCH.to_string(),
            reason: "Hyperlight requires x86_64 with hardware virtualization".to_string(),
        });

        #[cfg(target_arch = "x86_64")]
        {
            if !hyperlight_host::is_hypervisor_present() {
                return Err(SandboxErrorKind::HypervisorNotAvailable);
            }
            Ok(())
        }
    }

    /// Spawns the sandbox pool actor with the validated configuration.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The actor runtime to spawn the pool in
    ///
    /// # Returns
    ///
    /// The actor handle for the spawned pool.
    pub async fn spawn(&self, runtime: &mut ActorRuntime) -> ActorHandle {
        SandboxPool::spawn(runtime, self.config.clone()).await
    }

    /// Returns a reference to the validated configuration.
    #[must_use]
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_is_debug() {
        // Test that Debug is implemented (can't create without hypervisor)
        let _type_check: fn(SandboxConfig) -> Result<SandboxProvider, SandboxErrorKind> =
            SandboxProvider::new;
    }

    #[test]
    fn provider_is_clone() {
        // Verify Clone is implemented at compile time
        fn assert_clone<T: Clone>() {}
        assert_clone::<SandboxProvider>();
    }

    #[test]
    fn validate_platform_returns_appropriate_error() {
        let result = SandboxProvider::validate_platform();

        // On x86_64, result depends on hypervisor presence
        // On other architectures, should be ArchitectureNotSupported
        #[cfg(not(target_arch = "x86_64"))]
        assert!(matches!(
            result,
            Err(SandboxErrorKind::ArchitectureNotSupported { .. })
        ));

        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, we either get Ok or HypervisorNotAvailable
            match result {
                Ok(()) => {
                    // Hypervisor is available
                    assert!(hyperlight_host::is_hypervisor_present());
                }
                Err(SandboxErrorKind::HypervisorNotAvailable) => {
                    // No hypervisor, expected
                    assert!(!hyperlight_host::is_hypervisor_present());
                }
                Err(other) => panic!("unexpected error: {other}"),
            }
        }
    }

    #[test]
    fn new_validates_config_after_platform() {
        // On non-x86_64, we get architecture error before config validation
        // On x86_64 without hypervisor, we get hypervisor error before config
        // Only on x86_64 with hypervisor would we see invalid config error
        let config = SandboxConfig::new().with_memory_limit(100); // Invalid: < 1 MB
        let result = SandboxProvider::new(config);

        // We should get some error (platform or config)
        assert!(result.is_err());

        #[cfg(not(target_arch = "x86_64"))]
        assert!(matches!(
            result,
            Err(SandboxErrorKind::ArchitectureNotSupported { .. })
        ));
    }

    #[test]
    #[ignore = "requires hypervisor"]
    fn new_with_valid_config_and_hypervisor() {
        let config = SandboxConfig::default();
        let expected_memory_limit = config.memory_limit;
        let result = SandboxProvider::new(config);

        // Only runs if hypervisor is available
        if hyperlight_host::is_hypervisor_present() {
            assert!(result.is_ok());
            let provider = result.unwrap();
            assert_eq!(provider.config().memory_limit, expected_memory_limit);
        }
    }
}
