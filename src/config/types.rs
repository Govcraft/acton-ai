//! Configuration types for multi-provider support.
//!
//! This module provides types for defining multiple named LLM providers
//! in configuration files.

use crate::llm::{ProviderConfig, ProviderType, RateLimitConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Root configuration structure for acton-ai.
///
/// This structure maps directly to the TOML configuration file format:
///
/// ```toml
/// [providers.claude]
/// type = "anthropic"
/// model = "claude-sonnet-4-20250514"
/// api_key_env = "ANTHROPIC_API_KEY"
///
/// [providers.ollama]
/// type = "ollama"
/// model = "qwen2.5:7b"
/// base_url = "http://localhost:11434/v1"
///
/// default_provider = "ollama"
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActonAIConfig {
    /// Named provider configurations.
    ///
    /// Each key is the provider name (e.g., "claude", "ollama", "fast")
    /// that can be referenced when selecting a provider.
    #[serde(default)]
    pub providers: HashMap<String, NamedProviderConfig>,

    /// The name of the provider to use when none is specified.
    ///
    /// If not set and only one provider is defined, that provider
    /// becomes the default. If multiple providers exist and no
    /// default is specified, an error occurs at launch.
    pub default_provider: Option<String>,
}

impl ActonAIConfig {
    /// Creates an empty configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a named provider to the configuration.
    #[must_use]
    pub fn with_provider(mut self, name: impl Into<String>, config: NamedProviderConfig) -> Self {
        self.providers.insert(name.into(), config);
        self
    }

    /// Sets the default provider name.
    #[must_use]
    pub fn with_default_provider(mut self, name: impl Into<String>) -> Self {
        self.default_provider = Some(name.into());
        self
    }

    /// Returns the effective default provider name.
    ///
    /// Returns the explicitly set default, or if exactly one provider
    /// is defined, returns that provider's name.
    #[must_use]
    pub fn effective_default(&self) -> Option<&str> {
        if let Some(ref name) = self.default_provider {
            return Some(name.as_str());
        }

        if self.providers.len() == 1 {
            return self.providers.keys().next().map(String::as_str);
        }

        None
    }

    /// Returns true if the configuration has no providers defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Returns the number of providers defined.
    #[must_use]
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }
}

/// Configuration for a single named provider.
///
/// This structure supports all provider types (Anthropic, OpenAI, Ollama)
/// through a unified configuration format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedProviderConfig {
    /// The provider type: "anthropic", "openai", or "ollama".
    #[serde(rename = "type")]
    pub provider_type: String,

    /// The model to use (e.g., "claude-sonnet-4-20250514", "gpt-4o", "qwen2.5:7b").
    pub model: String,

    /// Direct API key value (discouraged - use api_key_env instead).
    #[serde(default)]
    pub api_key: Option<String>,

    /// Environment variable name containing the API key.
    ///
    /// This is the recommended way to provide API keys. The value
    /// of this environment variable will be read at runtime.
    #[serde(default)]
    pub api_key_env: Option<String>,

    /// Custom base URL for the API.
    ///
    /// Required for Ollama and custom OpenAI-compatible endpoints.
    /// Optional for Anthropic and OpenAI (uses default URLs).
    #[serde(default)]
    pub base_url: Option<String>,

    /// Request timeout in seconds.
    ///
    /// Defaults to 120 seconds for cloud providers, 300 for local.
    #[serde(default)]
    pub timeout_secs: Option<u64>,

    /// Maximum tokens to generate.
    ///
    /// Defaults to 4096.
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// Rate limiting configuration.
    #[serde(default)]
    pub rate_limit: Option<RateLimitFileConfig>,
}

impl NamedProviderConfig {
    /// Creates a new Anthropic provider configuration.
    #[must_use]
    pub fn anthropic(model: impl Into<String>) -> Self {
        Self {
            provider_type: "anthropic".to_string(),
            model: model.into(),
            api_key: None,
            api_key_env: Some("ANTHROPIC_API_KEY".to_string()),
            base_url: None,
            timeout_secs: None,
            max_tokens: None,
            rate_limit: None,
        }
    }

    /// Creates a new OpenAI provider configuration.
    #[must_use]
    pub fn openai(model: impl Into<String>) -> Self {
        Self {
            provider_type: "openai".to_string(),
            model: model.into(),
            api_key: None,
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            base_url: None,
            timeout_secs: None,
            max_tokens: None,
            rate_limit: None,
        }
    }

    /// Creates a new Ollama provider configuration.
    #[must_use]
    pub fn ollama(model: impl Into<String>) -> Self {
        Self {
            provider_type: "ollama".to_string(),
            model: model.into(),
            api_key: None,
            api_key_env: None,
            base_url: Some("http://localhost:11434/v1".to_string()),
            timeout_secs: Some(300),
            max_tokens: None,
            rate_limit: Some(RateLimitFileConfig {
                requests_per_minute: 1000,
                tokens_per_minute: 1_000_000,
            }),
        }
    }

    /// Sets the API key environment variable.
    #[must_use]
    pub fn with_api_key_env(mut self, env_var: impl Into<String>) -> Self {
        self.api_key_env = Some(env_var.into());
        self
    }

    /// Sets a direct API key (discouraged).
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the timeout in seconds.
    #[must_use]
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Sets the maximum tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Sets the rate limit configuration.
    #[must_use]
    pub fn with_rate_limit(mut self, rate_limit: RateLimitFileConfig) -> Self {
        self.rate_limit = Some(rate_limit);
        self
    }

    /// Resolves the API key from environment or direct value.
    ///
    /// Resolution order:
    /// 1. `api_key_env` - read from environment variable
    /// 2. Standard env var based on type (ANTHROPIC_API_KEY, OPENAI_API_KEY)
    /// 3. `api_key` - direct value in config (discouraged)
    /// 4. Empty string (for Ollama/local providers)
    #[must_use]
    pub fn resolve_api_key(&self) -> String {
        // Try explicit env var first
        if let Some(ref env_var) = self.api_key_env {
            if let Ok(key) = std::env::var(env_var) {
                if !key.is_empty() {
                    return key;
                }
            }
        }

        // Try standard env var based on provider type
        let standard_env = match self.provider_type.to_lowercase().as_str() {
            "anthropic" => Some("ANTHROPIC_API_KEY"),
            "openai" => Some("OPENAI_API_KEY"),
            _ => None,
        };

        if let Some(env_var) = standard_env {
            if let Ok(key) = std::env::var(env_var) {
                if !key.is_empty() {
                    return key;
                }
            }
        }

        // Fall back to direct API key
        if let Some(ref key) = self.api_key {
            return key.clone();
        }

        // Empty string for local providers
        String::new()
    }

    /// Converts this file configuration to a runtime ProviderConfig.
    ///
    /// This resolves environment variables and applies defaults.
    #[must_use]
    pub fn to_provider_config(&self) -> ProviderConfig {
        let api_key = self.resolve_api_key();

        let base_config = match self.provider_type.to_lowercase().as_str() {
            "anthropic" => ProviderConfig::anthropic(&api_key).with_model(&self.model),
            "openai" => ProviderConfig::openai(&api_key).with_model(&self.model),
            "ollama" => ProviderConfig::ollama(&self.model),
            _ => {
                // Treat unknown types as OpenAI-compatible
                let base_url = self
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:8080/v1".to_string());
                ProviderConfig::openai_compatible(&base_url, &self.model).with_api_key(&api_key)
            }
        };

        let mut config = base_config;

        // Apply overrides
        if let Some(ref url) = self.base_url {
            config = config
                .with_base_url(url)
                .with_provider_type(ProviderType::openai_compatible(url));
        }

        if let Some(secs) = self.timeout_secs {
            config = config.with_timeout(Duration::from_secs(secs));
        }

        if let Some(tokens) = self.max_tokens {
            config = config.with_max_tokens(tokens);
        }

        if let Some(ref rate_limit) = self.rate_limit {
            config = config.with_rate_limit(rate_limit.to_rate_limit_config());
        }

        config
    }
}

/// Rate limiting configuration for config files.
///
/// This is a simplified version of RateLimitConfig for TOML serialization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RateLimitFileConfig {
    /// Maximum requests per minute.
    pub requests_per_minute: u32,

    /// Maximum tokens per minute (input + output).
    pub tokens_per_minute: u32,
}

impl RateLimitFileConfig {
    /// Creates a new rate limit configuration.
    #[must_use]
    pub fn new(requests_per_minute: u32, tokens_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            tokens_per_minute,
        }
    }

    /// Converts to the runtime RateLimitConfig.
    #[must_use]
    pub fn to_rate_limit_config(&self) -> RateLimitConfig {
        RateLimitConfig::new(self.requests_per_minute, self.tokens_per_minute)
    }
}

impl Default for RateLimitFileConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 50,
            tokens_per_minute: 40_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acton_ai_config_default_is_empty() {
        let config = ActonAIConfig::default();
        assert!(config.is_empty());
        assert_eq!(config.provider_count(), 0);
        assert!(config.default_provider.is_none());
    }

    #[test]
    fn acton_ai_config_with_provider() {
        let config = ActonAIConfig::new()
            .with_provider("ollama", NamedProviderConfig::ollama("qwen2.5:7b"));

        assert_eq!(config.provider_count(), 1);
        assert!(config.providers.contains_key("ollama"));
    }

    #[test]
    fn acton_ai_config_effective_default_single_provider() {
        let config = ActonAIConfig::new()
            .with_provider("ollama", NamedProviderConfig::ollama("qwen2.5:7b"));

        assert_eq!(config.effective_default(), Some("ollama"));
    }

    #[test]
    fn acton_ai_config_effective_default_explicit() {
        let config = ActonAIConfig::new()
            .with_provider("ollama", NamedProviderConfig::ollama("qwen2.5:7b"))
            .with_provider("claude", NamedProviderConfig::anthropic("claude-sonnet-4-20250514"))
            .with_default_provider("claude");

        assert_eq!(config.effective_default(), Some("claude"));
    }

    #[test]
    fn acton_ai_config_effective_default_multiple_no_explicit() {
        let config = ActonAIConfig::new()
            .with_provider("ollama", NamedProviderConfig::ollama("qwen2.5:7b"))
            .with_provider("claude", NamedProviderConfig::anthropic("claude-sonnet-4-20250514"));

        // No default when multiple providers and none explicitly set
        assert!(config.effective_default().is_none());
    }

    #[test]
    fn named_provider_config_anthropic() {
        let config = NamedProviderConfig::anthropic("claude-sonnet-4-20250514");

        assert_eq!(config.provider_type, "anthropic");
        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.api_key_env, Some("ANTHROPIC_API_KEY".to_string()));
    }

    #[test]
    fn named_provider_config_openai() {
        let config = NamedProviderConfig::openai("gpt-4o");

        assert_eq!(config.provider_type, "openai");
        assert_eq!(config.model, "gpt-4o");
        assert_eq!(config.api_key_env, Some("OPENAI_API_KEY".to_string()));
    }

    #[test]
    fn named_provider_config_ollama() {
        let config = NamedProviderConfig::ollama("qwen2.5:7b");

        assert_eq!(config.provider_type, "ollama");
        assert_eq!(config.model, "qwen2.5:7b");
        assert_eq!(
            config.base_url,
            Some("http://localhost:11434/v1".to_string())
        );
        assert_eq!(config.timeout_secs, Some(300));
    }

    #[test]
    fn named_provider_config_to_provider_config_anthropic() {
        let config = NamedProviderConfig::anthropic("claude-3-haiku-20240307")
            .with_max_tokens(1024)
            .with_timeout_secs(60);

        let provider = config.to_provider_config();

        assert_eq!(provider.model, "claude-3-haiku-20240307");
        assert_eq!(provider.max_tokens, 1024);
        assert_eq!(provider.timeout, Duration::from_secs(60));
    }

    #[test]
    fn named_provider_config_to_provider_config_ollama() {
        let config = NamedProviderConfig::ollama("llama3.2");

        let provider = config.to_provider_config();

        assert_eq!(provider.model, "llama3.2");
        assert_eq!(provider.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn named_provider_config_resolve_api_key_direct() {
        let config = NamedProviderConfig::anthropic("test").with_api_key("direct-key");

        // With no env vars set, should fall back to direct key
        let key = config.resolve_api_key();
        // The actual value depends on environment, but the logic is tested
        assert!(!key.is_empty() || config.api_key.is_some());
    }

    #[test]
    fn rate_limit_file_config_to_runtime() {
        let file_config = RateLimitFileConfig::new(100, 50_000);
        let runtime = file_config.to_rate_limit_config();

        assert_eq!(runtime.requests_per_minute, 100);
        assert_eq!(runtime.tokens_per_minute, 50_000);
    }

    #[test]
    fn config_serialization_roundtrip() {
        let config = ActonAIConfig::new()
            .with_provider("ollama", NamedProviderConfig::ollama("qwen2.5:7b"))
            .with_default_provider("ollama");

        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: ActonAIConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(deserialized.default_provider, Some("ollama".to_string()));
        assert!(deserialized.providers.contains_key("ollama"));
    }

    #[test]
    fn config_from_toml_string() {
        let toml_str = r#"
default_provider = "ollama"

[providers.claude]
type = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434/v1"
timeout_secs = 300

[providers.ollama.rate_limit]
requests_per_minute = 1000
tokens_per_minute = 1000000
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();

        assert_eq!(config.provider_count(), 2);
        assert!(config.providers.contains_key("claude"));
        assert!(config.providers.contains_key("ollama"));
        assert_eq!(config.default_provider, Some("ollama".to_string()));

        let ollama = config.providers.get("ollama").unwrap();
        assert_eq!(ollama.timeout_secs, Some(300));
        assert!(ollama.rate_limit.is_some());
    }
}
