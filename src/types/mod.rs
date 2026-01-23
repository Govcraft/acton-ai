//! Core type definitions for the Acton-AI framework.
//!
//! This module contains all domain-specific types including:
//! - Identity types (AgentId, ConversationId, CorrelationId, MemoryId, MessageId, ToolName)
//! - Configuration types
//! - State enumerations

mod agent_id;
mod conversation_id;
mod correlation_id;
mod memory_id;
mod message_id;
mod tool_name;

pub use agent_id::{AgentId, InvalidAgentId};
pub use conversation_id::{ConversationId, InvalidConversationId};
pub use correlation_id::{CorrelationId, InvalidCorrelationId};
pub use memory_id::{InvalidMemoryId, MemoryId};
pub use message_id::{InvalidMessageId, MessageId};
pub use tool_name::{InvalidToolName, ToolName};
