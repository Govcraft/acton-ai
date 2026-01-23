//! Core type definitions for the Acton-AI framework.
//!
//! This module contains all domain-specific types including:
//! - Identity types (AgentId, CorrelationId, ToolName)
//! - Configuration types
//! - State enumerations

mod agent_id;
mod correlation_id;
mod tool_name;

pub use agent_id::{AgentId, InvalidAgentId};
pub use correlation_id::{CorrelationId, InvalidCorrelationId};
pub use tool_name::{InvalidToolName, ToolName};
