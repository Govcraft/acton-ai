//! Kernel actor module.
//!
//! The Kernel is the central coordinator and supervisor of the entire
//! Acton-AI system. It manages agent lifecycles and routes inter-agent
//! communication.

mod actor;
mod config;
mod discovery;
mod logging;

pub use actor::{InitKernel, Kernel, KernelMetrics};
pub use config::KernelConfig;
pub use discovery::CapabilityRegistry;
pub use logging::{
    init_and_store_logging, init_journald_logging, journald_layer, mark_subscriber_installed,
    LogLevel, LoggingConfig, LoggingError, LoggingErrorKind,
};
