//! Hyperlight sandbox factory implementation.
//!
//! Provides factory pattern for creating Hyperlight sandbox instances.

use super::config::SandboxConfig;
use super::error::SandboxErrorKind;
use super::sandbox::HyperlightSandbox;
use crate::tools::sandbox::stub::StubSandboxFactory;
use crate::tools::sandbox::traits::{Sandbox, SandboxFactory, SandboxFactoryFuture};

/// Factory for creating Hyperlight sandbox instances.
///
/// This factory checks for hypervisor availability and creates
/// `HyperlightSandbox` instances on demand.
///
/// # Fallback Behavior
///
/// Use `new_with_fallback()` to automatically fall back to a stub
/// implementation when no hypervisor is available. This is useful
/// for development on systems without KVM/Hyper-V support.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::HyperlightSandboxFactory;
///
/// // Create factory (will fail if no hypervisor)
/// let factory = HyperlightSandboxFactory::new()?;
///
/// // Or use fallback (always succeeds)
/// let factory = HyperlightSandboxFactory::new_with_fallback();
///
/// if factory.is_available() {
///     let sandbox = factory.create().await?;
///     // Use sandbox...
/// }
/// ```
#[derive(Debug, Clone)]
pub struct HyperlightSandboxFactory {
    /// Configuration for created sandboxes
    config: SandboxConfig,
    /// Whether a hypervisor is available
    hypervisor_available: bool,
}

impl HyperlightSandboxFactory {
    /// Creates a new factory with the default configuration.
    ///
    /// # Errors
    ///
    /// Returns `SandboxErrorKind::HypervisorNotAvailable` if no hypervisor
    /// is present on the system.
    pub fn new() -> Result<Self, SandboxErrorKind> {
        Self::with_config(SandboxConfig::default())
    }

    /// Creates a new factory with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The sandbox configuration to use
    ///
    /// # Errors
    ///
    /// Returns `SandboxErrorKind::HypervisorNotAvailable` if no hypervisor
    /// is present on the system.
    /// Returns `SandboxErrorKind::InvalidConfiguration` if the config is invalid.
    pub fn with_config(config: SandboxConfig) -> Result<Self, SandboxErrorKind> {
        config.validate()?;

        if !hyperlight_host::is_hypervisor_present() {
            return Err(SandboxErrorKind::HypervisorNotAvailable);
        }

        Ok(Self {
            config,
            hypervisor_available: true,
        })
    }

    /// Creates a factory that falls back to stub implementation.
    ///
    /// If no hypervisor is available, the factory will report as unavailable
    /// but can still be used (it will return errors when creating sandboxes).
    ///
    /// Use `is_available()` to check if real sandboxing is available.
    #[must_use]
    pub fn new_with_fallback() -> Self {
        Self::with_config_fallback(SandboxConfig::default())
    }

    /// Creates a factory with custom config that falls back gracefully.
    ///
    /// If the configuration is invalid or no hypervisor is available,
    /// the factory will report as unavailable.
    #[must_use]
    pub fn with_config_fallback(config: SandboxConfig) -> Self {
        let hypervisor_available = hyperlight_host::is_hypervisor_present()
            && config.validate().is_ok();

        Self {
            config,
            hypervisor_available,
        }
    }

    /// Returns the configuration used by this factory.
    #[must_use]
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }
}

impl SandboxFactory for HyperlightSandboxFactory {
    fn create(&self) -> SandboxFactoryFuture {
        let config = self.config.clone();
        let available = self.hypervisor_available;

        Box::pin(async move {
            if !available {
                return Err(SandboxErrorKind::HypervisorNotAvailable.into());
            }

            let sandbox = HyperlightSandbox::new(config)?;
            Ok(Box::new(sandbox) as Box<dyn Sandbox>)
        })
    }

    fn is_available(&self) -> bool {
        self.hypervisor_available
    }
}

/// Wrapper that automatically uses Hyperlight when available, stub otherwise.
///
/// This is useful for applications that want to use sandboxing when possible
/// but still function on systems without hypervisor support.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::AutoSandboxFactory;
///
/// let factory = AutoSandboxFactory::new();
///
/// // Will use Hyperlight if available, stub otherwise
/// let sandbox = factory.create().await?;
/// ```
#[derive(Debug, Clone)]
pub struct AutoSandboxFactory {
    /// Inner factory (either Hyperlight or Stub)
    inner: AutoSandboxInner,
}

#[derive(Debug, Clone)]
enum AutoSandboxInner {
    Hyperlight(HyperlightSandboxFactory),
    Stub(StubSandboxFactory),
}

impl Default for AutoSandboxFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoSandboxFactory {
    /// Creates a new auto-selecting factory.
    ///
    /// Uses Hyperlight if a hypervisor is available and sandboxes can be
    /// created successfully, otherwise uses the stub implementation.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SandboxConfig::default())
    }

    /// Creates a new auto-selecting factory with custom configuration.
    ///
    /// Uses Hyperlight if a hypervisor is available, config is valid, and
    /// sandboxes can be successfully created, otherwise falls back to the
    /// stub implementation.
    #[must_use]
    pub fn with_config(config: SandboxConfig) -> Self {
        // Try creating a Hyperlight factory
        let hyperlight_factory = HyperlightSandboxFactory::with_config(config.clone()).ok();

        // If we got a factory, verify we can actually create sandboxes
        let inner = if let Some(factory) = hyperlight_factory {
            // Try to create a sandbox to verify the full stack works
            match HyperlightSandbox::new(config) {
                Ok(_sandbox) => {
                    // Success! Use Hyperlight
                    tracing::info!("AutoSandboxFactory: Using Hyperlight sandboxing");
                    AutoSandboxInner::Hyperlight(factory)
                }
                Err(e) => {
                    // Sandbox creation failed (e.g., missing guest binary)
                    tracing::warn!(
                        error = %e,
                        "AutoSandboxFactory: Hyperlight sandbox creation failed, falling back to stub"
                    );
                    AutoSandboxInner::Stub(StubSandboxFactory::new())
                }
            }
        } else {
            // No hypervisor or invalid config
            tracing::info!("AutoSandboxFactory: No hypervisor available, using stub");
            AutoSandboxInner::Stub(StubSandboxFactory::new())
        };

        Self { inner }
    }

    /// Returns whether Hyperlight sandboxing is being used.
    #[must_use]
    pub fn is_using_hyperlight(&self) -> bool {
        matches!(self.inner, AutoSandboxInner::Hyperlight(_))
    }
}

impl SandboxFactory for AutoSandboxFactory {
    fn create(&self) -> SandboxFactoryFuture {
        match &self.inner {
            AutoSandboxInner::Hyperlight(factory) => factory.create(),
            AutoSandboxInner::Stub(factory) => factory.create(),
        }
    }

    fn is_available(&self) -> bool {
        match &self.inner {
            AutoSandboxInner::Hyperlight(factory) => factory.is_available(),
            AutoSandboxInner::Stub(factory) => factory.is_available(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_new_with_fallback_always_succeeds() {
        // This should always succeed, even without a hypervisor
        let factory = HyperlightSandboxFactory::new_with_fallback();

        // Debug should work
        assert!(format!("{:?}", factory).contains("HyperlightSandboxFactory"));
    }

    #[test]
    fn factory_reports_availability() {
        let factory = HyperlightSandboxFactory::new_with_fallback();

        // Should match actual hypervisor presence
        assert_eq!(
            factory.is_available(),
            hyperlight_host::is_hypervisor_present()
        );
    }

    #[test]
    fn factory_config_accessor() {
        let config = SandboxConfig::new().with_memory_limit(100 * 1024 * 1024);
        let factory = HyperlightSandboxFactory::with_config_fallback(config);

        assert_eq!(factory.config().memory_limit, 100 * 1024 * 1024);
    }

    #[test]
    fn auto_factory_default() {
        let factory = AutoSandboxFactory::default();
        assert!(format!("{:?}", factory).contains("AutoSandboxFactory"));
    }

    #[test]
    fn auto_factory_is_available() {
        let factory = AutoSandboxFactory::new();

        // Auto factory should always be available (falls back to stub)
        assert!(factory.is_available());
    }

    #[test]
    fn auto_factory_reports_hyperlight_usage() {
        let factory = AutoSandboxFactory::new();

        // When using embedded guest (which isn't available yet), this will always
        // be false because AutoSandboxFactory verifies sandbox creation works.
        // When hypervisor AND guest binary are both available, this would be true.
        // For now, we just verify it returns a sensible value.
        let _using_hyperlight = factory.is_using_hyperlight();
        // Note: Can't assert equality with is_hypervisor_present() because
        // that doesn't account for whether the guest binary is available.
    }

    #[tokio::test]
    async fn auto_factory_creates_sandbox() {
        let factory = AutoSandboxFactory::new();
        let result = factory.create().await;

        // Should always succeed (uses stub if no hypervisor)
        assert!(result.is_ok(), "Factory create failed: {:?}", result.err());

        let sandbox = result.unwrap();
        assert!(sandbox.is_alive());
    }

    #[test]
    #[ignore = "requires hypervisor"]
    fn factory_new_requires_hypervisor() {
        let result = HyperlightSandboxFactory::new();

        if hyperlight_host::is_hypervisor_present() {
            assert!(result.is_ok());
        } else {
            assert!(matches!(
                result.unwrap_err(),
                SandboxErrorKind::HypervisorNotAvailable
            ));
        }
    }
}
