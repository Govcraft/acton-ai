//! Kernel actor module.
//!
//! The Kernel is the central coordinator and supervisor of the entire
//! Acton-AI system. It manages agent lifecycles and routes inter-agent
//! communication.

mod actor;
mod config;

pub use actor::{InitKernel, Kernel, KernelMetrics};
pub use config::KernelConfig;
