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
    get_log_dir, init_and_store_logging, init_file_logging, LoggingConfig, LoggingError,
    LoggingErrorKind, LoggingGuard, LogLevel,
};
