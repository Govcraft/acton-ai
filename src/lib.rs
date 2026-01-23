//! # Acton-AI: Agentic AI Framework
//!
//! An agentic AI framework where each agent is an actor, leveraging acton-reactive's
//! supervision, pub/sub, and fault tolerance to create resilient, concurrent AI systems.
//!
//! ## Architecture
//!
//! - **Kernel**: Central supervisor managing all agents
//! - **Agent**: Individual AI agents with reasoning loops
//! - **LLM Provider**: Manages streaming LLM API calls with rate limiting
//! - **Tool Registry**: Registers and executes tools via supervised child actors
//! - **Memory Store**: Persistence via Turso/libSQL
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut app = ActonApp::launch_async().await;
//!     let kernel = Kernel::spawn(&mut app).await;
//!
//!     let agent_id = kernel.spawn_agent(AgentConfig::default()).await;
//!     kernel.send_prompt(agent_id, "Hello, agent!").await;
//!
//!     app.shutdown_all().await.unwrap();
//! }
//! ```

pub mod agent;
pub mod error;
pub mod kernel;
pub mod messages;
pub mod types;

// Future modules (Phase 2+)
// pub mod llm;
// pub mod memory;
// pub mod tools;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::agent::{Agent, AgentConfig, AgentState, InitAgent};
    pub use crate::error::{AgentError, KernelError};
    pub use crate::kernel::{InitKernel, Kernel, KernelConfig, KernelMetrics};
    pub use crate::messages::*;
    pub use crate::types::{AgentId, CorrelationId};

    // Re-export acton-reactive prelude
    pub use acton_reactive::prelude::*;
}
