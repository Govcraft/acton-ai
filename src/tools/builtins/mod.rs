//! Built-in tools for the Acton-AI framework.
//!
//! This module will contain pre-built tools that can be registered with the
//! Tool Registry. Currently empty; tools will be added as needed.
//!
//! ## Planned Tools
//!
//! - **Echo**: Simple tool that echoes input (for testing)
//! - **Calculator**: Basic arithmetic operations
//! - **HttpFetch**: Make HTTP requests (sandboxed)
//!
//! ## Creating Custom Tools
//!
//! Implement the `ToolExecutorTrait` to create custom tools:
//!
//! ```rust
//! use acton_ai::tools::{ToolExecutorTrait, ToolError, ToolExecutionFuture};
//! use serde_json::Value;
//!
//! #[derive(Debug)]
//! struct MyTool;
//!
//! impl ToolExecutorTrait for MyTool {
//!     fn execute(&self, args: Value) -> ToolExecutionFuture {
//!         Box::pin(async move {
//!             // Implementation
//!             Ok(serde_json::json!({"result": "success"}))
//!         })
//!     }
//! }
//! ```
