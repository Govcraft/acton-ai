//! Built-in tools for the Acton-AI framework.
//!
//! This module provides pre-built tools that can be registered with the
//! Tool Registry, enabling agents to interact with the filesystem, execute
//! commands, search content, and perform calculations.
//!
//! ## Available Tools
//!
//! ### Filesystem Tools
//! - **read_file**: Read file contents with line numbers
//! - **write_file**: Write content to files
//! - **edit_file**: Make targeted string replacements
//! - **list_directory**: List directory contents with metadata
//! - **glob**: Find files matching glob patterns
//! - **grep**: Search file contents with regex
//!
//! ### Execution Tools
//! - **bash**: Execute shell commands (sandboxed by default)
//! - **calculate**: Evaluate mathematical expressions
//! - **rust_code**: Execute compiler-verified Rust code (requires hyperlight feature)
//!
//! ### Web Tools
//! - **web_fetch**: Fetch content from URLs
//!
//! ## Usage
//!
//! ### Using the High-Level API
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! let runtime = ActonAI::builder()
//!     .app_name("my-app")
//!     .ollama("qwen2.5:7b")
//!     .with_builtins()  // Enable all built-in tools
//!     .launch()
//!     .await?;
//! ```
//!
//! ### Selective Tool Registration
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! let runtime = ActonAI::builder()
//!     .app_name("my-app")
//!     .ollama("qwen2.5:7b")
//!     .with_builtin_tools(&["read_file", "write_file", "glob"])
//!     .launch()
//!     .await?;
//! ```
//!
//! ### Direct Tool Access
//!
//! ```rust,ignore
//! use acton_ai::tools::builtins::{ReadFileTool, BuiltinTools};
//!
//! // Get tool configuration for manual registration
//! let config = ReadFileTool::config();
//!
//! // Or list all available tools
//! let tools = BuiltinTools::available();
//! ```

mod bash;
mod calculate;
mod edit_file;
mod glob;
mod grep;
mod list_directory;
mod read_file;
mod rust_code;
mod web_fetch;
mod write_file;

#[cfg(feature = "agent-skills")]
mod skill_activate;
#[cfg(feature = "agent-skills")]
mod skill_list;

// Re-export tool implementations
pub use bash::{BashTool, BashToolActor};
pub use calculate::{CalculateTool, CalculateToolActor};
pub use edit_file::{EditFileTool, EditFileToolActor};
pub use glob::{GlobTool, GlobToolActor};
pub use grep::{GrepTool, GrepToolActor};
pub use list_directory::{ListDirectoryTool, ListDirectoryToolActor};
pub use read_file::{ReadFileTool, ReadFileToolActor};
pub use web_fetch::{WebFetchTool, WebFetchToolActor};
pub use write_file::{WriteFileTool, WriteFileToolActor};

pub use rust_code::{RustCodeTool, RustCodeToolActor};

#[cfg(feature = "agent-skills")]
pub use skill_activate::{ActivateSkillTool, ActivateSkillToolActor};
#[cfg(feature = "agent-skills")]
pub use skill_list::{ListSkillsTool, ListSkillsToolActor};

use crate::messages::ToolDefinition;
use crate::tools::actor::ToolActor;
use crate::tools::{BoxedToolExecutor, ToolConfig, ToolError, ToolErrorKind};
use acton_reactive::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry of built-in tools.
///
/// Provides methods to access tool configurations and executors
/// for registration with the tool system.
#[derive(Debug, Default)]
pub struct BuiltinTools {
    /// Tool configurations by name
    configs: HashMap<String, ToolConfig>,
    /// Tool executors by name
    executors: HashMap<String, Arc<BoxedToolExecutor>>,
}

impl BuiltinTools {
    /// Creates a new registry with all built-in tools.
    #[must_use]
    pub fn all() -> Self {
        let mut registry = Self::default();

        // Register all tools
        registry.register(
            "read_file",
            ReadFileTool::config(),
            Box::new(ReadFileTool::new()),
        );
        registry.register(
            "write_file",
            WriteFileTool::config(),
            Box::new(WriteFileTool::new()),
        );
        registry.register(
            "edit_file",
            EditFileTool::config(),
            Box::new(EditFileTool::new()),
        );
        registry.register(
            "list_directory",
            ListDirectoryTool::config(),
            Box::new(ListDirectoryTool::new()),
        );
        registry.register("glob", GlobTool::config(), Box::new(GlobTool::new()));
        registry.register("grep", GrepTool::config(), Box::new(GrepTool::new()));
        registry.register("bash", BashTool::config(), Box::new(BashTool::new()));
        registry.register(
            "calculate",
            CalculateTool::config(),
            Box::new(CalculateTool::new()),
        );
        registry.register(
            "web_fetch",
            WebFetchTool::config(),
            Box::new(WebFetchTool::new()),
        );

        registry
    }

    /// Creates a new registry with only the specified tools.
    ///
    /// # Arguments
    ///
    /// * `tools` - Names of tools to include
    ///
    /// # Returns
    ///
    /// A registry containing only the specified tools, or an error if
    /// an unknown tool name is provided.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tools = BuiltinTools::select(&["read_file", "write_file"])?;
    /// ```
    pub fn select(tools: &[&str]) -> Result<Self, ToolError> {
        let all = Self::all();
        let mut registry = Self::default();

        for name in tools {
            let config = all.configs.get(*name).ok_or_else(|| {
                ToolError::new(ToolErrorKind::NotFound {
                    tool_name: (*name).to_string(),
                })
            })?;

            let executor = all.executors.get(*name).ok_or_else(|| {
                ToolError::new(ToolErrorKind::Internal {
                    message: format!("executor not found for tool: {name}"),
                })
            })?;

            registry.configs.insert((*name).to_string(), config.clone());
            registry
                .executors
                .insert((*name).to_string(), Arc::clone(executor));
        }

        Ok(registry)
    }

    /// Lists all available built-in tool names.
    ///
    /// Note: Not all tools may be registered in `all()`. Some tools like `rust_code`
    /// require external dependencies (Rust toolchain) and are only available via
    /// `spawn_tool_actor()`.
    #[must_use]
    pub fn available() -> Vec<&'static str> {
        vec![
            "read_file",
            "write_file",
            "edit_file",
            "list_directory",
            "glob",
            "grep",
            "bash",
            "calculate",
            "web_fetch",
            "rust_code",
        ]
    }

    /// Returns the configuration for a specific tool.
    #[must_use]
    pub fn get_config(&self, name: &str) -> Option<&ToolConfig> {
        self.configs.get(name)
    }

    /// Returns the executor for a specific tool.
    #[must_use]
    pub fn get_executor(&self, name: &str) -> Option<Arc<BoxedToolExecutor>> {
        self.executors.get(name).cloned()
    }

    /// Returns an iterator over all tool configurations.
    pub fn configs(&self) -> impl Iterator<Item = (&String, &ToolConfig)> {
        self.configs.iter()
    }

    /// Returns an iterator over all tool executors.
    pub fn executors(&self) -> impl Iterator<Item = (&String, &Arc<BoxedToolExecutor>)> {
        self.executors.iter()
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Returns true if no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }

    /// Internal method to register a tool.
    fn register(&mut self, name: &str, config: ToolConfig, executor: BoxedToolExecutor) {
        self.configs.insert(name.to_string(), config);
        self.executors.insert(name.to_string(), Arc::new(executor));
    }
}

/// Spawns a builtin tool actor by name.
///
/// Returns the actor handle and tool definition, or an error if the tool name
/// is unknown.
///
/// # Arguments
///
/// * `runtime` - The actor runtime to spawn the tool in
/// * `tool_name` - Name of the builtin tool to spawn
///
/// # Returns
///
/// A tuple of (ActorHandle, ToolDefinition) on success, or ToolError if the
/// tool name is not recognized.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::builtins::spawn_tool_actor;
///
/// let (handle, definition) = spawn_tool_actor(&mut runtime, "read_file").await?;
/// ```
pub async fn spawn_tool_actor(
    runtime: &mut ActorRuntime,
    tool_name: &str,
) -> Result<(ActorHandle, ToolDefinition), ToolError> {
    match tool_name {
        "read_file" => {
            let handle = ReadFileToolActor::spawn(runtime).await;
            let definition = ReadFileToolActor::definition();
            Ok((handle, definition))
        }
        "write_file" => {
            let handle = WriteFileToolActor::spawn(runtime).await;
            let definition = WriteFileToolActor::definition();
            Ok((handle, definition))
        }
        "edit_file" => {
            let handle = EditFileToolActor::spawn(runtime).await;
            let definition = EditFileToolActor::definition();
            Ok((handle, definition))
        }
        "list_directory" => {
            let handle = ListDirectoryToolActor::spawn(runtime).await;
            let definition = ListDirectoryToolActor::definition();
            Ok((handle, definition))
        }
        "glob" => {
            let handle = GlobToolActor::spawn(runtime).await;
            let definition = GlobToolActor::definition();
            Ok((handle, definition))
        }
        "grep" => {
            let handle = GrepToolActor::spawn(runtime).await;
            let definition = GrepToolActor::definition();
            Ok((handle, definition))
        }
        "bash" => {
            let handle = BashToolActor::spawn(runtime).await;
            let definition = BashToolActor::definition();
            Ok((handle, definition))
        }
        "calculate" => {
            let handle = CalculateToolActor::spawn(runtime).await;
            let definition = CalculateToolActor::definition();
            Ok((handle, definition))
        }
        "web_fetch" => {
            let handle = WebFetchToolActor::spawn(runtime).await;
            let definition = WebFetchToolActor::definition();
            Ok((handle, definition))
        }
        "rust_code" => {
            let handle = RustCodeToolActor::spawn(runtime).await;
            let definition = RustCodeToolActor::definition();
            Ok((handle, definition))
        }
        _ => Err(ToolError::not_found(tool_name)),
    }
}

/// Returns the tool definition for a builtin tool by name.
///
/// # Arguments
///
/// * `tool_name` - Name of the builtin tool
///
/// # Returns
///
/// The tool definition, or an error if the tool name is not recognized.
pub fn get_tool_definition(tool_name: &str) -> Result<ToolDefinition, ToolError> {
    match tool_name {
        "read_file" => Ok(ReadFileToolActor::definition()),
        "write_file" => Ok(WriteFileToolActor::definition()),
        "edit_file" => Ok(EditFileToolActor::definition()),
        "list_directory" => Ok(ListDirectoryToolActor::definition()),
        "glob" => Ok(GlobToolActor::definition()),
        "grep" => Ok(GrepToolActor::definition()),
        "bash" => Ok(BashToolActor::definition()),
        "calculate" => Ok(CalculateToolActor::definition()),
        "web_fetch" => Ok(WebFetchToolActor::definition()),
        "rust_code" => Ok(RustCodeToolActor::definition()),
        _ => Err(ToolError::not_found(tool_name)),
    }
}

/// Spawns all skill-related tool actors.
///
/// This spawns the `list_skills` and `activate_skill` tool actors, which are
/// only available when the `agent-skills` feature is enabled.
///
/// # Arguments
///
/// * `runtime` - The actor runtime to spawn the tools in
/// * `registry` - The skill registry to use for the skill tools
///
/// # Returns
///
/// A vector of (name, handle, definition) tuples for the spawned skill tools.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::builtins::spawn_skill_tool_actors;
/// use acton_ai::skills::SkillRegistry;
/// use std::sync::Arc;
///
/// let registry = Arc::new(SkillRegistry::new());
/// let skill_tools = spawn_skill_tool_actors(&mut runtime, registry).await;
/// ```
#[cfg(feature = "agent-skills")]
pub async fn spawn_skill_tool_actors(
    runtime: &mut ActorRuntime,
    registry: std::sync::Arc<crate::skills::SkillRegistry>,
) -> Vec<(String, ActorHandle, ToolDefinition)> {
    let mut tools = Vec::new();

    // Spawn list_skills tool
    let list_handle = ListSkillsToolActor::spawn_with_registry(runtime, registry.clone()).await;
    let list_def = ListSkillsTool::config().definition;
    tools.push(("list_skills".to_string(), list_handle, list_def));

    // Spawn activate_skill tool
    let activate_handle = ActivateSkillToolActor::spawn_with_registry(runtime, registry).await;
    let activate_def = ActivateSkillTool::config().definition;
    tools.push(("activate_skill".to_string(), activate_handle, activate_def));

    tools
}

/// Returns the names of all skill-related tools.
///
/// Only available when the `agent-skills` feature is enabled.
#[cfg(feature = "agent-skills")]
#[must_use]
pub fn skill_tool_names() -> Vec<&'static str> {
    vec!["list_skills", "activate_skill"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_tools_all_creates_all_tools() {
        let tools = BuiltinTools::all();
        assert_eq!(tools.len(), 9);

        // rust_code is not in all() because it requires toolchain
        for name in BuiltinTools::available()
            .iter()
            .filter(|n| **n != "rust_code")
        {
            assert!(
                tools.get_config(name).is_some(),
                "missing config for {name}"
            );
            assert!(
                tools.get_executor(name).is_some(),
                "missing executor for {name}"
            );
        }
    }

    #[test]
    fn builtin_tools_select_specific() {
        let tools = BuiltinTools::select(&["read_file", "write_file"]).unwrap();
        assert_eq!(tools.len(), 2);

        assert!(tools.get_config("read_file").is_some());
        assert!(tools.get_config("write_file").is_some());
        assert!(tools.get_config("bash").is_none());
    }

    #[test]
    fn builtin_tools_select_unknown_fails() {
        let result = BuiltinTools::select(&["read_file", "unknown_tool"]);
        assert!(result.is_err());
    }

    #[test]
    fn builtin_tools_available_returns_all_names() {
        let names = BuiltinTools::available();
        assert_eq!(names.len(), 10);

        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"edit_file"));
        assert!(names.contains(&"list_directory"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"calculate"));
        assert!(names.contains(&"web_fetch"));
        assert!(names.contains(&"rust_code"));
    }

    #[test]
    fn builtin_tools_configs_iterator() {
        let tools = BuiltinTools::all();
        let configs: Vec<_> = tools.configs().collect();
        assert_eq!(configs.len(), 9);
    }

    #[test]
    fn builtin_tools_executors_iterator() {
        let tools = BuiltinTools::all();
        let executors: Vec<_> = tools.executors().collect();
        assert_eq!(executors.len(), 9);
    }

    #[test]
    fn builtin_tools_empty() {
        let tools = BuiltinTools::default();
        assert!(tools.is_empty());
        assert_eq!(tools.len(), 0);
    }

    #[test]
    fn tool_configs_have_unique_names() {
        let tools = BuiltinTools::all();
        let mut names: Vec<_> = tools
            .configs()
            .map(|(_, c)| c.definition.name.clone())
            .collect();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), original_len, "duplicate tool names found");
    }

    #[test]
    fn tool_configs_have_descriptions() {
        let tools = BuiltinTools::all();
        for (name, config) in tools.configs() {
            assert!(
                !config.definition.description.is_empty(),
                "tool {name} has empty description"
            );
        }
    }

    #[test]
    fn tool_configs_have_valid_schemas() {
        let tools = BuiltinTools::all();
        for (name, config) in tools.configs() {
            let schema = &config.definition.input_schema;
            assert!(schema.is_object(), "tool {name} schema is not an object");
            assert!(
                schema.get("type").is_some(),
                "tool {name} schema missing type"
            );
        }
    }
}
