//! Tool system for the Acton-AI framework.
//!
//! This module provides the infrastructure for tool registration and execution:
//!
//! - **Tool Registry**: Central actor that manages tool registration and dispatch
//! - **Tool Executor**: Supervised child actors for executing individual tools
//! - **Sandbox**: Interface for sandboxed code execution (Hyperlight integration)
//!
//! ## Architecture
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
//!                            |
//!                            | supervises (RestartPolicy::Temporary)
//!                            v
//! +-------------------------------------------------------------+
//! |                   Tool Executor Actor                        |
//! |                                                              |
//! |  Execute --> executor.execute(args) --> ToolResponse        |
//! |              or sandbox.execute(code, args)                  |
//! |                                                              |
//! |  before_stop: cleanup sandbox                                |
//! +-------------------------------------------------------------+
//! ```
//!
//! ## Usage
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
//!
//! // Execute a tool (usually triggered by Agent)
//! registry.send(ExecuteTool {
//!     correlation_id: CorrelationId::new(),
//!     tool_call: ToolCall { ... },
//!     requesting_agent: agent_id,
//! }).await;
//! ```

pub mod builtins;
pub mod definition;
pub mod error;
pub mod executor;
pub mod registry;
pub mod sandbox;

// Re-exports
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
