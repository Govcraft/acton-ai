//! Tool system for the Acton-AI framework.
//!
//! This module provides the infrastructure for tool registration and execution:
//!
//! - **Tool Registry**: Central actor that manages tool registration and dispatch
//! - **Tool Executor**: Supervised child actors for executing individual tools
//! - **Tool Actors**: Per-agent tool actors for isolated tool execution
//! - **Sandbox**: Interface for sandboxed code execution (Hyperlight integration)
//!
//! ## Architecture
//!
//! ### Per-Agent Tool Actors (Recommended)
//!
//! ```text
//! +-------------------------------------------------------------+
//! |                      Agent Actor                             |
//! |                                                              |
//! |  tool_handles: HashMap<String, ActorHandle>                 |
//! |                                                              |
//! +-------------------------------------------------------------+
//!                            |
//!                            | supervises
//!                            v
//! +-------------------------------------------------------------+
//! |                   Tool Actor (per tool)                      |
//! |                                                              |
//! |  ExecuteToolDirect --> execute(args) --> ToolActorResponse  |
//! |                                                              |
//! +-------------------------------------------------------------+
//! ```
//!
//! ### Global Tool Registry (Legacy)
//!
//! ```text
//! +-------------------------------------------------------------+
//! |                   Tool Registry Actor                        |
//! |                                                              |
//! |  RegisterTool --> tools: HashMap<String, RegisteredTool>    |
//! |  UnregisterTool                                             |
//! |  ExecuteTool --> Spawns ToolExecutor (Temporary)            |
//! |  ListTools --> Returns Vec<ToolDefinition>                  |
//! |                                                              |
//! +-------------------------------------------------------------+
//! ```
//!
//! ## Usage
//!
//! ### Per-Agent Tools (Recommended)
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! // Configure agent with specific tools
//! let config = AgentConfig::new("You are helpful.")
//!     .with_tools(&["read_file", "write_file", "glob"]);
//!
//! // Tools are spawned automatically when the agent initializes
//! ```
//!
//! ### Global Registry (Legacy)
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//! use acton_ai::tools::{ToolRegistry, ToolDefinition, ToolConfig, RegisterTool};
//!
//! // Spawn the registry
//! let registry = ToolRegistry::spawn(&mut runtime).await;
//!
//! // Register a tool
//! registry.send(RegisterTool {
//!     config: ToolConfig::new(ToolDefinition::new(
//!         "calculator",
//!         "Performs arithmetic",
//!         serde_json::json!({...}),
//!     )),
//!     executor: Arc::new(Box::new(CalculatorExecutor)),
//! }).await;
//! ```

pub mod actor;
pub mod builtins;
pub mod definition;
pub mod error;
pub mod executor;
pub mod registry;
pub mod sandbox;

// Re-exports
pub use actor::{ExecuteToolDirect, ToolActor, ToolActorResponse, ToolExecutor as ToolExecutorAsync};
pub use crate::messages::ToolDefinition;
pub use definition::{BoxedToolExecutor, ToolConfig, ToolExecutionFuture, ToolExecutorTrait};
pub use error::{ToolError, ToolErrorKind};
pub use executor::{Execute, InitExecutor, ToolExecutor};
pub use registry::{
    ConfigureSandbox, InitToolRegistry, ListTools, RegisterTool, RegisteredTool, RegistryMetrics,
    ToolListResponse, ToolRegistry, UnregisterTool,
};
pub use sandbox::{
    Sandbox, SandboxExecutionFuture, SandboxFactory, SandboxFactoryFuture, StubSandbox,
    StubSandboxFactory,
};
