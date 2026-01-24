//! Agent configuration.
//!
//! Defines configuration options for creating and customizing agents.

use crate::types::AgentId;
use serde::{Deserialize, Serialize};
#[cfg(feature = "agent-skills")]
use std::path::PathBuf;

/// Configuration for creating a new agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Optional pre-assigned ID for the agent
    pub id: Option<AgentId>,
    /// The system prompt that defines the agent's behavior
    pub system_prompt: String,
    /// Optional display name for the agent
    pub name: Option<String>,
    /// Maximum number of messages to keep in conversation history
    pub max_conversation_length: usize,
    /// Whether to enable streaming responses
    pub enable_streaming: bool,
    /// Names of builtin tools to enable for this agent.
    ///
    /// If empty, no builtin tools are enabled.
    /// Use `with_all_builtins()` to enable all available builtin tools.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Paths to skill files or directories to load for this agent.
    ///
    /// Only available when the `agent-skills` feature is enabled.
    #[cfg(feature = "agent-skills")]
    #[serde(default)]
    pub skill_paths: Vec<PathBuf>,
}

impl AgentConfig {
    /// Creates a new agent configuration with the given system prompt.
    ///
    /// # Arguments
    ///
    /// * `system_prompt` - The system prompt that defines the agent's behavior
    ///
    /// # Examples
    ///
    /// ```
    /// use acton_ai::agent::AgentConfig;
    ///
    /// let config = AgentConfig::new("You are a helpful assistant.");
    /// assert!(!config.system_prompt.is_empty());
    /// ```
    #[must_use]
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            id: None,
            system_prompt: system_prompt.into(),
            name: None,
            max_conversation_length: 100,
            enable_streaming: true,
            tools: Vec::new(),
            #[cfg(feature = "agent-skills")]
            skill_paths: Vec::new(),
        }
    }

    /// Sets a pre-assigned ID for the agent.
    #[must_use]
    pub fn with_id(mut self, id: AgentId) -> Self {
        self.id = Some(id);
        self
    }

    /// Sets a display name for the agent.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the maximum conversation length.
    #[must_use]
    pub fn with_max_conversation_length(mut self, length: usize) -> Self {
        self.max_conversation_length = length;
        self
    }

    /// Enables or disables streaming responses.
    #[must_use]
    pub fn with_streaming(mut self, enable: bool) -> Self {
        self.enable_streaming = enable;
        self
    }

    /// Sets the list of builtin tools to enable for this agent.
    ///
    /// # Arguments
    ///
    /// * `tools` - Names of builtin tools to enable (e.g., "read_file", "bash")
    ///
    /// # Example
    ///
    /// ```rust
    /// use acton_ai::agent::AgentConfig;
    ///
    /// let config = AgentConfig::new("You are helpful.")
    ///     .with_tools(&["read_file", "write_file", "glob"]);
    /// ```
    #[must_use]
    pub fn with_tools(mut self, tools: &[&str]) -> Self {
        self.tools = tools.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Enables all available builtin tools for this agent.
    ///
    /// This is a convenience method equivalent to calling `with_tools` with
    /// all available tool names.
    ///
    /// # Example
    ///
    /// ```rust
    /// use acton_ai::agent::AgentConfig;
    ///
    /// let config = AgentConfig::new("You are helpful.")
    ///     .with_all_builtins();
    /// ```
    #[must_use]
    pub fn with_all_builtins(mut self) -> Self {
        use crate::tools::builtins::BuiltinTools;
        self.tools = BuiltinTools::available()
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        self
    }

    /// Adds a single tool to the list of enabled tools.
    ///
    /// # Arguments
    ///
    /// * `tool` - Name of the builtin tool to add
    ///
    /// # Example
    ///
    /// ```rust
    /// use acton_ai::agent::AgentConfig;
    ///
    /// let config = AgentConfig::new("You are helpful.")
    ///     .with_tool("read_file")
    ///     .with_tool("write_file");
    /// ```
    #[must_use]
    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Sets the skill paths to load for this agent.
    ///
    /// Only available when the `agent-skills` feature is enabled.
    ///
    /// # Arguments
    ///
    /// * `paths` - Paths to skill files or directories
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::agent::AgentConfig;
    /// use std::path::PathBuf;
    ///
    /// let config = AgentConfig::new("You are helpful.")
    ///     .with_skill_paths(&[PathBuf::from("./skills")]);
    /// ```
    #[cfg(feature = "agent-skills")]
    #[must_use]
    pub fn with_skill_paths(mut self, paths: &[PathBuf]) -> Self {
        self.skill_paths = paths.to_vec();
        self
    }

    /// Adds a single skill path to the list of paths to load.
    ///
    /// Only available when the `agent-skills` feature is enabled.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to a skill file or directory
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::agent::AgentConfig;
    /// use std::path::PathBuf;
    ///
    /// let config = AgentConfig::new("You are helpful.")
    ///     .with_skill_path(PathBuf::from("./skills/coding.md"))
    ///     .with_skill_path(PathBuf::from("./skills/debugging"));
    /// ```
    #[cfg(feature = "agent-skills")]
    #[must_use]
    pub fn with_skill_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.skill_paths.push(path.into());
        self
    }

    /// Returns the agent ID, generating a new one if not set.
    #[must_use]
    pub fn agent_id(&self) -> AgentId {
        self.id.clone().unwrap_or_default()
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self::new("You are a helpful AI assistant.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_config_with_system_prompt() {
        let config = AgentConfig::new("Custom prompt");
        assert_eq!(config.system_prompt, "Custom prompt");
        assert!(config.id.is_none());
        assert!(config.name.is_none());
    }

    #[test]
    fn default_has_reasonable_values() {
        let config = AgentConfig::default();
        assert!(!config.system_prompt.is_empty());
        assert_eq!(config.max_conversation_length, 100);
        assert!(config.enable_streaming);
    }

    #[test]
    fn builder_pattern() {
        let id = AgentId::new();
        let config = AgentConfig::new("Test")
            .with_id(id.clone())
            .with_name("TestAgent")
            .with_max_conversation_length(50)
            .with_streaming(false);

        assert_eq!(config.id, Some(id));
        assert_eq!(config.name, Some("TestAgent".to_string()));
        assert_eq!(config.max_conversation_length, 50);
        assert!(!config.enable_streaming);
    }

    #[test]
    fn agent_id_generates_new_when_none() {
        let config = AgentConfig::new("Test");
        let id1 = config.agent_id();
        let id2 = config.agent_id();

        // Each call generates a new ID when not pre-set
        assert_ne!(id1, id2);
    }

    #[test]
    fn agent_id_returns_preset_when_set() {
        let preset_id = AgentId::new();
        let config = AgentConfig::new("Test").with_id(preset_id.clone());

        let id1 = config.agent_id();
        let id2 = config.agent_id();

        assert_eq!(id1, preset_id);
        assert_eq!(id2, preset_id);
    }

    #[test]
    fn serialization_roundtrip() {
        let config = AgentConfig::new("Test agent")
            .with_name("TestBot")
            .with_max_conversation_length(50);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }

    #[test]
    fn with_tools_sets_tool_list() {
        let config = AgentConfig::new("Test").with_tools(&["read_file", "write_file", "glob"]);

        assert_eq!(config.tools.len(), 3);
        assert!(config.tools.contains(&"read_file".to_string()));
        assert!(config.tools.contains(&"write_file".to_string()));
        assert!(config.tools.contains(&"glob".to_string()));
    }

    #[test]
    fn with_tool_adds_single_tool() {
        let config = AgentConfig::new("Test")
            .with_tool("read_file")
            .with_tool("write_file");

        assert_eq!(config.tools.len(), 2);
        assert!(config.tools.contains(&"read_file".to_string()));
        assert!(config.tools.contains(&"write_file".to_string()));
    }

    #[test]
    fn with_all_builtins_adds_all_tools() {
        let config = AgentConfig::new("Test").with_all_builtins();

        // Should have all 9 builtin tools
        assert_eq!(config.tools.len(), 9);
        assert!(config.tools.contains(&"read_file".to_string()));
        assert!(config.tools.contains(&"bash".to_string()));
        assert!(config.tools.contains(&"calculate".to_string()));
    }

    #[test]
    fn default_has_no_tools() {
        let config = AgentConfig::default();
        assert!(config.tools.is_empty());
    }

    #[test]
    fn serialization_roundtrip_with_tools() {
        let config = AgentConfig::new("Test agent").with_tools(&["read_file", "bash"]);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
        assert_eq!(deserialized.tools, vec!["read_file", "bash"]);
    }

    #[cfg(feature = "agent-skills")]
    mod skills_tests {
        use super::*;

        #[test]
        fn with_skill_paths_sets_paths() {
            let config = AgentConfig::new("Test")
                .with_skill_paths(&[PathBuf::from("./skills"), PathBuf::from("./more-skills")]);

            assert_eq!(config.skill_paths.len(), 2);
            assert!(config.skill_paths.contains(&PathBuf::from("./skills")));
            assert!(config.skill_paths.contains(&PathBuf::from("./more-skills")));
        }

        #[test]
        fn with_skill_path_adds_single_path() {
            let config = AgentConfig::new("Test")
                .with_skill_path("./skills/coding.md")
                .with_skill_path("./skills/debugging");

            assert_eq!(config.skill_paths.len(), 2);
            assert!(config
                .skill_paths
                .contains(&PathBuf::from("./skills/coding.md")));
            assert!(config
                .skill_paths
                .contains(&PathBuf::from("./skills/debugging")));
        }

        #[test]
        fn default_has_no_skill_paths() {
            let config = AgentConfig::default();
            assert!(config.skill_paths.is_empty());
        }

        #[test]
        fn serialization_roundtrip_with_skill_paths() {
            let config =
                AgentConfig::new("Test agent").with_skill_paths(&[PathBuf::from("./skills")]);

            let json = serde_json::to_string(&config).unwrap();
            let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();

            assert_eq!(config, deserialized);
            assert_eq!(deserialized.skill_paths, vec![PathBuf::from("./skills")]);
        }
    }
}
