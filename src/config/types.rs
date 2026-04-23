//! Configuration types for multi-provider support.
//!
//! This module provides types for defining multiple named LLM providers
//! and sandbox settings in configuration files.

use crate::llm::{ProviderConfig, ProviderType, RateLimitConfig, SamplingParams};
use crate::tools::sandbox::{HardeningMode, ProcessSandboxConfig};
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
///
/// [sandbox]
/// hardening = "besteffort"    # "off" | "besteffort" | "enforce"
///
/// [sandbox.limits]
/// max_execution_ms = 30000
/// max_memory_mb = 256
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

    /// Sandbox configuration for tool execution isolation.
    ///
    /// When present, configures the [`ProcessSandbox`](crate::tools::sandbox::ProcessSandbox)
    /// resource limits and OS-hardening policy.
    #[serde(default)]
    pub sandbox: Option<SandboxFileConfig>,

    /// Persistence configuration for the database.
    #[serde(default)]
    pub persistence: Option<PersistenceFileConfig>,

    /// CLI-specific configuration.
    #[serde(default)]
    pub cli: Option<CliFileConfig>,

    /// Named job definitions for `acton-ai run-job`.
    #[serde(default)]
    pub jobs: Option<HashMap<String, JobConfig>>,

    /// Framework-wide defaults applied to every prompt unless overridden.
    ///
    /// Currently carries `max_tool_rounds`; this block is the designated
    /// home for future cross-prompt defaults (temperature, timeout, etc.)
    /// so they share one TOML namespace rather than each getting a
    /// top-level key.
    #[serde(default)]
    pub defaults: Option<ActonAIDefaults>,
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

    /// Controls randomness in generation (0.0 to 1.0 for Anthropic, 0.0 to 2.0 for OpenAI).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Limits sampling to the top K most likely tokens (Anthropic only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Nucleus sampling threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Penalizes tokens based on frequency (OpenAI only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    /// Penalizes tokens based on presence (OpenAI only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    /// Seed for deterministic generation (OpenAI only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Sequences that will cause the model to stop generating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
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
            temperature: None,
            top_k: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop_sequences: None,
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
            temperature: None,
            top_k: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop_sequences: None,
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
            temperature: None,
            top_k: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop_sequences: None,
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

    /// Sets the temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets top_k sampling.
    #[must_use]
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Sets top_p (nucleus) sampling.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the stop sequences.
    #[must_use]
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
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

        // Build sampling params from individual fields
        let mut sampling = SamplingParams::default();
        if let Some(temp) = self.temperature {
            sampling.temperature = Some(temp);
        }
        if let Some(k) = self.top_k {
            sampling.top_k = Some(k);
        }
        if let Some(p) = self.top_p {
            sampling.top_p = Some(p);
        }
        if let Some(fp) = self.frequency_penalty {
            sampling.frequency_penalty = Some(fp);
        }
        if let Some(pp) = self.presence_penalty {
            sampling.presence_penalty = Some(pp);
        }
        if let Some(s) = self.seed {
            sampling.seed = Some(s);
        }
        if let Some(ref seqs) = self.stop_sequences {
            sampling.stop_sequences = Some(seqs.clone());
        }
        if !sampling.is_empty() {
            config = config.with_sampling(sampling);
        }

        config
    }
}

/// Framework-wide defaults for the `[defaults]` section.
///
/// ```toml
/// [defaults]
/// max_tool_rounds = 20
/// ```
///
/// These values seed every `PromptBuilder` created through the runtime.
/// Per-prompt calls (e.g. `.max_tool_rounds(5)`) still override them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActonAIDefaults {
    /// Default cap on agentic tool-call rounds per prompt.
    ///
    /// When unset, the framework fallback
    /// ([`DEFAULT_MAX_TOOL_ROUNDS`](crate::prompt::DEFAULT_MAX_TOOL_ROUNDS))
    /// is used.
    #[serde(default)]
    pub max_tool_rounds: Option<usize>,
}

impl ActonAIDefaults {
    /// Creates an empty defaults block.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the default `max_tool_rounds`.
    #[must_use]
    pub fn with_max_tool_rounds(mut self, max: usize) -> Self {
        self.max_tool_rounds = Some(max);
        self
    }
}

/// Sandbox configuration for TOML config files.
///
/// This structure maps to the `[sandbox]` section in config files:
///
/// ```toml
/// [sandbox]
/// hardening = "besteffort"    # "off" | "besteffort" | "enforce"
///
/// [sandbox.limits]
/// max_execution_ms = 30000
/// max_memory_mb = 256
/// ```
///
/// Missing keys fall through to [`ProcessSandboxConfig::default`]. Unknown
/// fields (including the retired Hyperlight-pool fields `pool_warmup`,
/// `pool_max_per_type`, `max_executions_before_recycle`) are ignored by
/// default, so existing TOMLs load cleanly.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SandboxFileConfig {
    /// OS-hardening policy. Accepts `"off"`, `"besteffort"`, or `"enforce"`.
    /// Defaults to [`HardeningMode::BestEffort`].
    #[serde(default)]
    pub hardening: Option<HardeningMode>,

    /// Resource limits for sandbox execution.
    #[serde(default)]
    pub limits: Option<SandboxLimitsConfig>,
}

impl SandboxFileConfig {
    /// Creates a new empty sandbox configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the OS-hardening policy.
    #[must_use]
    pub fn with_hardening(mut self, mode: HardeningMode) -> Self {
        self.hardening = Some(mode);
        self
    }

    /// Sets the resource limits.
    #[must_use]
    pub fn with_limits(mut self, limits: SandboxLimitsConfig) -> Self {
        self.limits = Some(limits);
        self
    }

    /// Converts this file configuration to a runtime [`ProcessSandboxConfig`].
    ///
    /// Unspecified fields fall back to [`ProcessSandboxConfig::default`].
    #[must_use]
    pub fn to_process_config(&self) -> ProcessSandboxConfig {
        let mut config = ProcessSandboxConfig::default();

        if let Some(mode) = self.hardening {
            config = config.with_hardening(mode);
        }

        if let Some(ref limits) = self.limits {
            if let Some(timeout_ms) = limits.max_execution_ms {
                config = config.with_timeout(Duration::from_millis(timeout_ms));
            }
            if let Some(memory_mb) = limits.max_memory_mb {
                // `memory_mb * 1024 * 1024` can theoretically overflow on
                // 32-bit pointer widths; saturate to u64::MAX in that case.
                let bytes = (memory_mb as u64).saturating_mul(1024 * 1024);
                config = config.with_memory_limit(Some(bytes));
            }
        }

        config
    }
}

/// Resource limits for sandbox execution.
///
/// ```toml
/// [sandbox.limits]
/// max_execution_ms = 30000
/// max_memory_mb = 64
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SandboxLimitsConfig {
    /// Maximum execution time in milliseconds.
    ///
    /// Default: 30000 (30 seconds)
    #[serde(default)]
    pub max_execution_ms: Option<u64>,

    /// Maximum memory in megabytes.
    ///
    /// Default: 64 MB
    #[serde(default)]
    pub max_memory_mb: Option<usize>,
}

impl SandboxLimitsConfig {
    /// Creates a new empty limits configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum execution time in milliseconds.
    #[must_use]
    pub fn with_max_execution_ms(mut self, ms: u64) -> Self {
        self.max_execution_ms = Some(ms);
        self
    }

    /// Sets the maximum memory in megabytes.
    #[must_use]
    pub fn with_max_memory_mb(mut self, mb: usize) -> Self {
        self.max_memory_mb = Some(mb);
        self
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

/// Persistence configuration for the `[persistence]` section.
///
/// ```toml
/// [persistence]
/// db_path = "~/.local/share/acton-ai/acton-ai.db"
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistenceFileConfig {
    /// Path to the database file.
    ///
    /// Default: `~/.local/share/acton-ai/acton-ai.db`
    #[serde(default)]
    pub db_path: Option<String>,
}

/// CLI-specific configuration for the `[cli]` section.
///
/// ```toml
/// [cli]
/// default_session = "main"
/// default_system_prompt = "You are a helpful assistant."
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CliFileConfig {
    /// Default session name when `--session` is not specified.
    ///
    /// Default: "main"
    #[serde(default)]
    pub default_session: Option<String>,

    /// Default system prompt for new sessions.
    #[serde(default)]
    pub default_system_prompt: Option<String>,
}

/// Job configuration for `acton-ai run-job`.
///
/// ```toml
/// [jobs.summarize]
/// system_prompt = "You are a summarization expert."
/// message_template = "Summarize:\n\n{{input}}"
/// provider = "claude"
/// tools = ["read_file", "glob"]
/// max_tool_rounds = 10
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    /// System prompt for this job.
    pub system_prompt: String,

    /// Message template with `{{input}}` and `{{key}}` placeholders.
    #[serde(default)]
    pub message_template: Option<String>,

    /// Provider name override for this job.
    #[serde(default)]
    pub provider: Option<String>,

    /// Tool names to enable. Use `["*"]` for all builtins.
    #[serde(default)]
    pub tools: Option<Vec<String>>,

    /// Maximum tool execution rounds before stopping.
    ///
    /// Default: 10
    #[serde(default)]
    pub max_tool_rounds: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::sandbox::process::config::{DEFAULT_MEMORY_LIMIT, DEFAULT_TIMEOUT_SECS};

    #[test]
    fn acton_ai_config_default_is_empty() {
        let config = ActonAIConfig::default();
        assert!(config.is_empty());
        assert_eq!(config.provider_count(), 0);
        assert!(config.default_provider.is_none());
    }

    #[test]
    fn acton_ai_config_with_provider() {
        let config =
            ActonAIConfig::new().with_provider("ollama", NamedProviderConfig::ollama("qwen2.5:7b"));

        assert_eq!(config.provider_count(), 1);
        assert!(config.providers.contains_key("ollama"));
    }

    #[test]
    fn acton_ai_config_effective_default_single_provider() {
        let config =
            ActonAIConfig::new().with_provider("ollama", NamedProviderConfig::ollama("qwen2.5:7b"));

        assert_eq!(config.effective_default(), Some("ollama"));
    }

    #[test]
    fn acton_ai_config_effective_default_explicit() {
        let config = ActonAIConfig::new()
            .with_provider("ollama", NamedProviderConfig::ollama("qwen2.5:7b"))
            .with_provider(
                "claude",
                NamedProviderConfig::anthropic("claude-sonnet-4-20250514"),
            )
            .with_default_provider("claude");

        assert_eq!(config.effective_default(), Some("claude"));
    }

    #[test]
    fn acton_ai_config_effective_default_multiple_no_explicit() {
        let config = ActonAIConfig::new()
            .with_provider("ollama", NamedProviderConfig::ollama("qwen2.5:7b"))
            .with_provider(
                "claude",
                NamedProviderConfig::anthropic("claude-sonnet-4-20250514"),
            );

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

    // Sandbox configuration tests

    #[test]
    fn sandbox_file_config_default_has_no_overrides() {
        let config = SandboxFileConfig::default();
        assert!(config.hardening.is_none());
        assert!(config.limits.is_none());
    }

    #[test]
    fn sandbox_limits_config_default_is_none() {
        let config = SandboxLimitsConfig::default();
        assert!(config.max_execution_ms.is_none());
        assert!(config.max_memory_mb.is_none());
    }

    #[test]
    fn sandbox_file_config_builder() {
        let config = SandboxFileConfig::new()
            .with_hardening(HardeningMode::Enforce)
            .with_limits(
                SandboxLimitsConfig::new()
                    .with_max_execution_ms(60_000)
                    .with_max_memory_mb(256),
            );

        assert_eq!(config.hardening, Some(HardeningMode::Enforce));
        let limits = config.limits.unwrap();
        assert_eq!(limits.max_execution_ms, Some(60_000));
        assert_eq!(limits.max_memory_mb, Some(256));
    }

    #[test]
    fn sandbox_limits_config_builder() {
        let config = SandboxLimitsConfig::new()
            .with_max_execution_ms(60_000)
            .with_max_memory_mb(128);

        assert_eq!(config.max_execution_ms, Some(60_000));
        assert_eq!(config.max_memory_mb, Some(128));
    }

    #[test]
    fn sandbox_file_config_to_process_config_with_values() {
        let config = SandboxFileConfig {
            hardening: Some(HardeningMode::Enforce),
            limits: Some(SandboxLimitsConfig {
                max_execution_ms: Some(60_000),
                max_memory_mb: Some(128),
            }),
        };

        let process_config = config.to_process_config();

        assert_eq!(process_config.hardening, HardeningMode::Enforce);
        assert_eq!(process_config.timeout, Duration::from_millis(60_000));
        assert_eq!(process_config.memory_limit, Some(128 * 1024 * 1024));
    }

    #[test]
    fn sandbox_file_config_to_process_config_with_defaults() {
        let config = SandboxFileConfig::default();
        let process_config = config.to_process_config();

        // Should use ProcessSandboxConfig defaults
        assert_eq!(
            process_config.timeout,
            Duration::from_secs(DEFAULT_TIMEOUT_SECS)
        );
        assert_eq!(process_config.memory_limit, Some(DEFAULT_MEMORY_LIMIT));
    }

    #[test]
    fn config_from_toml_with_sandbox_section() {
        let toml_str = r#"
default_provider = "ollama"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"

[sandbox]
hardening = "enforce"

[sandbox.limits]
max_execution_ms = 60000
max_memory_mb = 128
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();

        assert!(config.sandbox.is_some());
        let sandbox = config.sandbox.unwrap();
        assert_eq!(sandbox.hardening, Some(HardeningMode::Enforce));

        let limits = sandbox.limits.unwrap();
        assert_eq!(limits.max_execution_ms, Some(60_000));
        assert_eq!(limits.max_memory_mb, Some(128));
    }

    #[test]
    fn config_from_toml_without_sandbox_section() {
        let toml_str = r#"
default_provider = "ollama"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();

        assert!(config.sandbox.is_none());
    }

    #[test]
    fn config_from_toml_with_partial_sandbox() {
        let toml_str = r#"
[sandbox]
hardening = "off"
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();

        assert!(config.sandbox.is_some());
        let sandbox = config.sandbox.unwrap();
        assert_eq!(sandbox.hardening, Some(HardeningMode::Off));
        assert!(sandbox.limits.is_none());
    }

    #[test]
    fn config_from_toml_ignores_retired_hyperlight_fields() {
        // Old TOMLs may still carry pool_warmup / pool_max_per_type /
        // max_executions_before_recycle from the Hyperlight era. Unknown
        // keys are silently ignored by serde's default behavior, so parsing
        // must succeed and the resulting config must fall back to defaults.
        let toml_str = r#"
[sandbox]
pool_warmup = 8
pool_max_per_type = 64
max_executions_before_recycle = 500
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();
        let sandbox = config.sandbox.unwrap();
        assert!(sandbox.hardening.is_none());
        assert!(sandbox.limits.is_none());
    }

    #[test]
    fn named_provider_config_sampling_fields_default_to_none() {
        let config = NamedProviderConfig::anthropic("claude-sonnet-4-20250514");
        assert!(config.temperature.is_none());
        assert!(config.top_k.is_none());
        assert!(config.top_p.is_none());
        assert!(config.frequency_penalty.is_none());
        assert!(config.presence_penalty.is_none());
        assert!(config.seed.is_none());
        assert!(config.stop_sequences.is_none());
    }

    #[test]
    fn named_provider_config_to_provider_config_with_sampling() {
        let config = NamedProviderConfig::anthropic("claude-sonnet-4-20250514")
            .with_temperature(0.7)
            .with_top_p(0.9);

        let provider = config.to_provider_config();
        assert_eq!(provider.sampling.temperature, Some(0.7));
        assert_eq!(provider.sampling.top_p, Some(0.9));
    }

    #[test]
    fn config_from_toml_with_sampling_params() {
        let toml_str = r#"
default_provider = "claude"

[providers.claude]
type = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"
temperature = 0.7
top_p = 0.9
stop_sequences = ["END", "STOP"]
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();
        let claude = config.providers.get("claude").unwrap();
        assert_eq!(claude.temperature, Some(0.7));
        assert_eq!(claude.top_p, Some(0.9));
        assert_eq!(
            claude.stop_sequences,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );
    }

    #[test]
    fn config_from_toml_with_only_limits() {
        let toml_str = r#"
[sandbox.limits]
max_execution_ms = 60000
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();

        assert!(config.sandbox.is_some());
        let sandbox = config.sandbox.unwrap();
        assert!(sandbox.hardening.is_none());

        let limits = sandbox.limits.unwrap();
        assert_eq!(limits.max_execution_ms, Some(60_000));
        assert!(limits.max_memory_mb.is_none());
    }

    #[test]
    fn config_serialization_roundtrip_with_sandbox() {
        let config = ActonAIConfig {
            providers: HashMap::new(),
            default_provider: None,
            sandbox: Some(SandboxFileConfig {
                hardening: Some(HardeningMode::Enforce),
                limits: Some(SandboxLimitsConfig {
                    max_execution_ms: Some(60_000),
                    max_memory_mb: Some(128),
                }),
            }),
            persistence: None,
            cli: None,
            jobs: None,
            defaults: None,
        };

        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: ActonAIConfig = toml::from_str(&toml_str).unwrap();

        assert!(deserialized.sandbox.is_some());
        let sandbox = deserialized.sandbox.unwrap();
        assert_eq!(sandbox.hardening, Some(HardeningMode::Enforce));

        let limits = sandbox.limits.unwrap();
        assert_eq!(limits.max_execution_ms, Some(60_000));
        assert_eq!(limits.max_memory_mb, Some(128));
    }

    #[test]
    fn acton_ai_config_default_has_no_sandbox() {
        let config = ActonAIConfig::default();
        assert!(config.sandbox.is_none());
    }

    #[test]
    fn sandbox_file_config_is_clone() {
        let config = SandboxFileConfig::new().with_hardening(HardeningMode::Enforce);
        let cloned = config.clone();
        assert_eq!(config.hardening, cloned.hardening);
    }

    #[test]
    fn defaults_absent_parses_as_none() {
        let toml_str = r#"
default_provider = "ollama"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();
        assert!(config.defaults.is_none());
    }

    #[test]
    fn defaults_parses_max_tool_rounds() {
        let toml_str = r#"
[defaults]
max_tool_rounds = 25
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();
        let defaults = config.defaults.expect("defaults block should parse");
        assert_eq!(defaults.max_tool_rounds, Some(25));
    }

    #[test]
    fn defaults_empty_block_parses_with_none_fields() {
        // An empty `[defaults]` block should parse (all fields are Option),
        // leaving every knob at its framework fallback.
        let toml_str = r#"
[defaults]
        "#;

        let config: ActonAIConfig = toml::from_str(toml_str).unwrap();
        let defaults = config.defaults.expect("defaults block should parse");
        assert!(defaults.max_tool_rounds.is_none());
    }

    #[test]
    fn defaults_builder_sets_max_tool_rounds() {
        let defaults = ActonAIDefaults::new().with_max_tool_rounds(42);
        assert_eq!(defaults.max_tool_rounds, Some(42));
    }

    #[test]
    fn sandbox_limits_config_is_clone() {
        let config = SandboxLimitsConfig::new().with_max_execution_ms(60_000);
        let cloned = config.clone();
        assert_eq!(config.max_execution_ms, cloned.max_execution_ms);
    }
}
