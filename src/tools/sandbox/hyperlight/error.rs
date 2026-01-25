//! Hyperlight-specific error types.
//!
//! These errors provide detailed information about sandbox failures
//! while mapping to the standard `ToolError::SandboxError` for API compatibility.

use super::guest::GuestType;
use crate::tools::error::ToolError;
use std::fmt;
use std::time::Duration;

/// Specific error types for Hyperlight sandbox operations.
///
/// These provide detailed context for sandbox failures while maintaining
/// compatibility with the broader `ToolError` system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxErrorKind {
    /// No hypervisor available on the system.
    ///
    /// Hyperlight requires either KVM (Linux) or Windows Hypervisor Platform.
    HypervisorNotAvailable,

    /// Failed to create the sandbox.
    ///
    /// This can occur due to resource exhaustion, permission issues,
    /// or hypervisor errors during VM initialization.
    CreationFailed {
        /// Detailed reason for the failure
        reason: String,
    },

    /// Guest execution timed out.
    ///
    /// The sandboxed code exceeded the configured timeout duration.
    ExecutionTimeout {
        /// The timeout that was exceeded
        duration: Duration,
    },

    /// Guest exceeded memory limit.
    ///
    /// The sandboxed code attempted to use more memory than allocated.
    MemoryLimitExceeded {
        /// Configured memory limit in bytes
        limit: usize,
    },

    /// All pooled sandboxes are in use.
    ///
    /// The sandbox pool has reached capacity and cannot provide
    /// additional instances.
    PoolExhausted {
        /// Current pool size
        pool_size: usize,
    },

    /// Failed to call a guest function.
    ///
    /// The host attempted to invoke a function in the guest that
    /// either doesn't exist or failed during execution.
    GuestCallFailed {
        /// Name of the function that failed
        function: String,
        /// Detailed reason for the failure
        reason: String,
    },

    /// Sandbox has already been destroyed.
    ///
    /// Attempted to use a sandbox after `destroy()` was called.
    AlreadyDestroyed,

    /// Invalid configuration provided.
    ///
    /// The sandbox configuration contains invalid values.
    InvalidConfiguration {
        /// The configuration field that is invalid
        field: String,
        /// Why it's invalid
        reason: String,
    },

    /// Architecture not supported by Hyperlight.
    ///
    /// Hyperlight requires x86_64 architecture with hardware virtualization support.
    ArchitectureNotSupported {
        /// The current architecture (e.g., "aarch64", "arm")
        arch: String,
        /// Reason for incompatibility
        reason: String,
    },

    /// Invalid guest type requested from pool.
    ///
    /// The requested guest type is not supported or not configured in the pool.
    InvalidGuestType {
        /// The guest type that was requested
        guest_type: GuestType,
    },
}

impl SandboxErrorKind {
    /// Converts this error into a `ToolError`.
    ///
    /// All Hyperlight-specific errors map to `ToolError::SandboxError`
    /// for API compatibility.
    #[must_use]
    pub fn into_tool_error(self) -> ToolError {
        ToolError::sandbox_error(self.to_string())
    }
}

impl fmt::Display for SandboxErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HypervisorNotAvailable => {
                write!(
                    f,
                    "no hypervisor available; Hyperlight requires KVM (Linux) or \
                     Windows Hypervisor Platform"
                )
            }
            Self::CreationFailed { reason } => {
                write!(f, "failed to create sandbox: {reason}")
            }
            Self::ExecutionTimeout { duration } => {
                write!(
                    f,
                    "sandbox execution timed out after {} seconds",
                    duration.as_secs()
                )
            }
            Self::MemoryLimitExceeded { limit } => {
                write!(
                    f,
                    "sandbox exceeded memory limit of {} MB",
                    limit / (1024 * 1024)
                )
            }
            Self::PoolExhausted { pool_size } => {
                write!(
                    f,
                    "sandbox pool exhausted; all {} instances are in use",
                    pool_size
                )
            }
            Self::GuestCallFailed { function, reason } => {
                write!(f, "guest function '{}' failed: {}", function, reason)
            }
            Self::AlreadyDestroyed => {
                write!(f, "sandbox has already been destroyed")
            }
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    f,
                    "invalid sandbox configuration for '{}': {}",
                    field, reason
                )
            }
            Self::ArchitectureNotSupported { arch, reason } => {
                write!(
                    f,
                    "architecture '{}' not supported: {}; Hyperlight requires x86_64",
                    arch, reason
                )
            }
            Self::InvalidGuestType { guest_type } => {
                write!(
                    f,
                    "invalid guest type '{}' requested; guest type not available in pool",
                    guest_type
                )
            }
        }
    }
}

impl std::error::Error for SandboxErrorKind {}

impl From<SandboxErrorKind> for ToolError {
    fn from(kind: SandboxErrorKind) -> Self {
        kind.into_tool_error()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypervisor_not_available_display() {
        let err = SandboxErrorKind::HypervisorNotAvailable;
        let msg = err.to_string();
        assert!(msg.contains("hypervisor"));
        assert!(msg.contains("KVM"));
    }

    #[test]
    fn creation_failed_display() {
        let err = SandboxErrorKind::CreationFailed {
            reason: "permission denied".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("create sandbox"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn execution_timeout_display() {
        let err = SandboxErrorKind::ExecutionTimeout {
            duration: Duration::from_secs(30),
        };
        let msg = err.to_string();
        assert!(msg.contains("timed out"));
        assert!(msg.contains("30 seconds"));
    }

    #[test]
    fn memory_limit_exceeded_display() {
        let err = SandboxErrorKind::MemoryLimitExceeded {
            limit: 64 * 1024 * 1024,
        };
        let msg = err.to_string();
        assert!(msg.contains("memory limit"));
        assert!(msg.contains("64 MB"));
    }

    #[test]
    fn pool_exhausted_display() {
        let err = SandboxErrorKind::PoolExhausted { pool_size: 4 };
        let msg = err.to_string();
        assert!(msg.contains("pool exhausted"));
        assert!(msg.contains("4 instances"));
    }

    #[test]
    fn guest_call_failed_display() {
        let err = SandboxErrorKind::GuestCallFailed {
            function: "execute_shell".to_string(),
            reason: "invalid command".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("execute_shell"));
        assert!(msg.contains("invalid command"));
    }

    #[test]
    fn already_destroyed_display() {
        let err = SandboxErrorKind::AlreadyDestroyed;
        let msg = err.to_string();
        assert!(msg.contains("destroyed"));
    }

    #[test]
    fn invalid_configuration_display() {
        let err = SandboxErrorKind::InvalidConfiguration {
            field: "memory_limit".to_string(),
            reason: "must be positive".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("memory_limit"));
        assert!(msg.contains("must be positive"));
    }

    #[test]
    fn converts_to_tool_error() {
        let err = SandboxErrorKind::HypervisorNotAvailable;
        let tool_err: ToolError = err.into();
        assert!(tool_err.to_string().contains("sandbox error"));
    }

    #[test]
    fn architecture_not_supported_display() {
        let err = SandboxErrorKind::ArchitectureNotSupported {
            arch: "aarch64".to_string(),
            reason: "Hyperlight requires x86_64 with hardware virtualization".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("aarch64"));
        assert!(msg.contains("x86_64"));
        assert!(msg.contains("not supported"));
    }

    #[test]
    fn architecture_not_supported_converts_to_tool_error() {
        let err = SandboxErrorKind::ArchitectureNotSupported {
            arch: "arm".to_string(),
            reason: "test".to_string(),
        };
        let tool_err: ToolError = err.into();
        assert!(tool_err.to_string().contains("sandbox error"));
    }

    #[test]
    fn invalid_guest_type_display() {
        let err = SandboxErrorKind::InvalidGuestType {
            guest_type: GuestType::Shell,
        };
        let msg = err.to_string();
        assert!(msg.contains("shell"));
        assert!(msg.contains("invalid"));
    }

    #[test]
    fn invalid_guest_type_converts_to_tool_error() {
        let err = SandboxErrorKind::InvalidGuestType {
            guest_type: GuestType::Http,
        };
        let tool_err: ToolError = err.into();
        assert!(tool_err.to_string().contains("sandbox error"));
    }
}
