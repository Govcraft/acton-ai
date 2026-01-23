//! Kernel configuration.
//!
//! Defines configuration options for the Kernel actor.

use serde::{Deserialize, Serialize};

/// Configuration for the Kernel actor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Maximum number of agents that can be spawned
    pub max_agents: usize,
    /// Whether to enable agent metrics collection
    pub enable_metrics: bool,
    /// Default system prompt for agents without one specified
    pub default_system_prompt: Option<String>,
}

impl KernelConfig {
    /// Creates a new kernel configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of agents.
    #[must_use]
    pub fn with_max_agents(mut self, max: usize) -> Self {
        self.max_agents = max;
        self
    }

    /// Enables or disables metrics collection.
    #[must_use]
    pub fn with_metrics(mut self, enable: bool) -> Self {
        self.enable_metrics = enable;
        self
    }

    /// Sets the default system prompt for agents.
    #[must_use]
    pub fn with_default_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.default_system_prompt = Some(prompt.into());
        self
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            max_agents: 100,
            enable_metrics: true,
            default_system_prompt: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_reasonable_values() {
        let config = KernelConfig::default();
        assert_eq!(config.max_agents, 100);
        assert!(config.enable_metrics);
        assert!(config.default_system_prompt.is_none());
    }

    #[test]
    fn builder_pattern() {
        let config = KernelConfig::new()
            .with_max_agents(50)
            .with_metrics(false)
            .with_default_system_prompt("Default prompt");

        assert_eq!(config.max_agents, 50);
        assert!(!config.enable_metrics);
        assert_eq!(
            config.default_system_prompt,
            Some("Default prompt".to_string())
        );
    }

    #[test]
    fn serialization_roundtrip() {
        let config = KernelConfig::new()
            .with_max_agents(25)
            .with_default_system_prompt("Test");

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: KernelConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }
}
