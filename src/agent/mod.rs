//! Agent actor module.
//!
//! This module contains the Agent actor implementation, which represents
//! an individual AI agent with its own state, tools, and reasoning loop.

mod actor;
mod config;
mod delegation;
mod state;

pub use actor::{Agent, InitAgent, PendingLLMRequest};
pub use config::AgentConfig;
pub use delegation::{DelegatedTask, DelegatedTaskState, DelegationTracker, IncomingTaskInfo};
pub use state::AgentState;
