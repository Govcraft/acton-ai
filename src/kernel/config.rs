//! Kernel configuration.
//!
//! Defines configuration options for the Kernel actor.

use crate::kernel::logging::LoggingConfig;
use serde::{Deserialize, Serialize};

/// Configuration for the Kernel actor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Maximum number of agents that can be spawned.
    pub max_agents: usize,
    /// Whether to enable agent metrics collection.
    pub enable_metrics: bool,
    /// Default system prompt for agents without one specified.
    pub default_system_prompt: Option<String>,
    /// File logging configuration. None disables file logging.
    pub logging: Option<LoggingConfig>,
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

    /// Sets the logging configuration.
    #[must_use]
    pub fn with_logging(mut self, config: LoggingConfig) -> Self {
        self.logging = Some(config);
        self
    }

    /// Disables file logging.
    #[must_use]
    pub fn without_logging(mut self) -> Self {
        self.logging = None;
        self
    }

    /// Sets the application name for log files.
    ///
    /// If logging is not yet configured, creates a default logging config
    /// with the specified app name.
    #[must_use]
    pub fn with_app_name(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        if let Some(ref mut logging) = self.logging {
            logging.app_name = name;
        } else {
            self.logging = Some(LoggingConfig::default().with_app_name(name));
        }
        self
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            max_agents: 100,
            enable_metrics: true,
            default_system_prompt: None,
            logging: Some(LoggingConfig::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::logging::LogLevel;

    #[test]
    fn default_has_reasonable_values() {
        let config = KernelConfig::default();
        assert_eq!(config.max_agents, 100);
        assert!(config.enable_metrics);
        assert!(config.default_system_prompt.is_none());
        assert!(config.logging.is_some());
    }

    #[test]
    fn default_includes_logging() {
        let config = KernelConfig::default();
        let logging = config.logging.unwrap();
        assert!(logging.enabled);
        assert_eq!(logging.app_name, "acton-ai");
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
    fn with_logging_sets_config() {
        let logging_config = LoggingConfig::default()
            .with_app_name("custom-app")
            .with_level(LogLevel::Debug);

        let config = KernelConfig::default().with_logging(logging_config.clone());

        assert_eq!(config.logging, Some(logging_config));
    }

    #[test]
    fn without_logging_disables_logging() {
        let config = KernelConfig::default().without_logging();
        assert!(config.logging.is_none());
    }

    #[test]
    fn with_app_name_updates_existing_logging() {
        let config = KernelConfig::default().with_app_name("my-agent");

        let logging = config.logging.unwrap();
        assert_eq!(logging.app_name, "my-agent");
    }

    #[test]
    fn with_app_name_creates_logging_if_none() {
        let config = KernelConfig::default()
            .without_logging()
            .with_app_name("new-app");

        let logging = config.logging.unwrap();
        assert_eq!(logging.app_name, "new-app");
    }

    #[test]
    fn serialization_roundtrip() {
        let config = KernelConfig::new()
            .with_max_agents(25)
            .with_default_system_prompt("Test")
            .with_app_name("test-app");

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: KernelConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }

    #[test]
    fn serialization_roundtrip_without_logging() {
        let config = KernelConfig::new()
            .with_max_agents(25)
            .without_logging();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: KernelConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }
}
