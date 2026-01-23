//! Agent configuration.
//!
//! Defines configuration options for creating and customizing agents.

use crate::types::AgentId;
use serde::{Deserialize, Serialize};

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
}
