//! Portable process-level sandbox.
//!
//! Executes tool invocations in a re-exec'd child process guarded by
//! rlimits, timeouts, and (on Linux) landlock + seccomp restrictions.
//!
//! The module is split into four layers:
//!
//! - [`protocol`] — synchronous, length-prefixed JSON framing used on the
//!   stdin/stdout pipes between parent and child.
//! - [`config`] — [`ProcessSandboxConfig`] and [`HardeningMode`], the tunables
//!   that govern resource limits and OS hardening.
//! - [`hardening`] — Linux landlock + seccomp installation (feature-gated via
//!   `sandbox-hardening`). A no-op stub is provided on non-Linux platforms.
//! - [`runner`] — the child-side entry point invoked when the binary is
//!   re-execed with `ACTON_AI_SANDBOX_RUNNER=1`. Not called directly by
//!   library consumers.
//! - [`sandbox`] + [`factory`] — the parent-side [`crate::tools::sandbox::Sandbox`]
//!   and [`crate::tools::sandbox::SandboxFactory`] implementations.

pub mod config;
pub mod factory;
pub mod protocol;
pub mod runner;
pub mod sandbox;

#[cfg(target_os = "linux")]
pub mod hardening;

/// Non-Linux stub so callers can invoke `hardening::apply(&cfg)` unconditionally.
#[cfg(not(target_os = "linux"))]
pub mod hardening {
    use super::config::ProcessSandboxConfig;
    use crate::tools::ToolError;

    /// No-op on non-Linux platforms.
    ///
    /// # Errors
    ///
    /// Never fails; returns [`Ok`] for API parity with the Linux implementation.
    pub fn apply(_cfg: &ProcessSandboxConfig) -> Result<(), ToolError> {
        Ok(())
    }
}

pub use config::{HardeningMode, ProcessSandboxConfig};
pub use factory::ProcessSandboxFactory;
pub use protocol::{Request, Response};
pub use sandbox::ProcessSandbox;
