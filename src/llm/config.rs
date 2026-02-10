//! LLM provider configuration.
//!
//! Configuration types for the LLM Provider actor including API settings,
//! rate limiting, and retry behavior.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// The type of LLM provider to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderType {
    /// Anthropic Claude API
    Anthropic,
    /// OpenAI-compatible API (including Ollama, vLLM, LocalAI, etc.)
    OpenAI {
        /// Base URL for the API (e.g., "http://localhost:11434/v1" for Ollama)
        base_url: String,
    },
}

impl Default for ProviderType {
    fn default() -> Self {
        Self::Anthropic
    }
}

impl ProviderType {
    /// Creates an OpenAI-compatible provider with the given base URL.
    #[must_use]
    pub fn openai_compatible(base_url: impl Into<String>) -> Self {
        Self::OpenAI {
            base_url: base_url.into(),
        }
    }

    /// Creates an OpenAI-compatible provider configured for Ollama.
    #[must_use]
    pub fn ollama() -> Self {
        Self::OpenAI {
            base_url: "http://localhost:11434/v1".to_string(),
        }
    }

    /// Creates an OpenAI-compatible provider configured for OpenAI.
    #[must_use]
    pub fn openai() -> Self {
        Self::OpenAI {
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }
}

/// Sampling parameters for LLM text generation.
///
/// These parameters control the randomness and creativity of the model's output.
/// All fields are optional — only set values are sent to the API.
///
/// The merge order is: per-prompt overrides > per-provider defaults > API defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Controls randomness in generation.
    ///
    /// - Anthropic: 0.0 to 1.0 (default 1.0)
    /// - OpenAI: 0.0 to 2.0 (default 1.0)
    ///
    /// Lower values produce more deterministic output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Limits sampling to the top K most likely tokens.
    ///
    /// Supported by Anthropic. Not supported by OpenAI (ignored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Nucleus sampling: limits to tokens whose cumulative probability
    /// exceeds this threshold.
    ///
    /// Supported by both Anthropic and OpenAI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Penalizes tokens based on their frequency in the generated text so far.
    ///
    /// Supported by OpenAI (-2.0 to 2.0). Not supported by Anthropic (ignored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    /// Penalizes tokens based on whether they appear in the generated text so far.
    ///
    /// Supported by OpenAI (-2.0 to 2.0). Not supported by Anthropic (ignored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    /// Seed for deterministic generation.
    ///
    /// Supported by OpenAI. Not supported by Anthropic (ignored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Sequences that will cause the model to stop generating.
    ///
    /// - Anthropic: `stop_sequences` field
    /// - OpenAI: `stop` field
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

impl SamplingParams {
    /// Creates new empty sampling parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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

    /// Sets the frequency penalty.
    #[must_use]
    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Sets the presence penalty.
    #[must_use]
    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Sets the seed for deterministic generation.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the stop sequences.
    #[must_use]
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Returns true if no parameters are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
            && self.seed.is_none()
            && self.stop_sequences.is_none()
    }

    /// Merges two `SamplingParams`, with `overrides` taking precedence.
    ///
    /// Fields set in `overrides` replace fields in `self`.
    #[must_use]
    pub fn merge_with(&self, overrides: &SamplingParams) -> SamplingParams {
        SamplingParams {
            temperature: overrides.temperature.or(self.temperature),
            top_k: overrides.top_k.or(self.top_k),
            top_p: overrides.top_p.or(self.top_p),
            frequency_penalty: overrides.frequency_penalty.or(self.frequency_penalty),
            presence_penalty: overrides.presence_penalty.or(self.presence_penalty),
            seed: overrides.seed.or(self.seed),
            stop_sequences: overrides
                .stop_sequences
                .clone()
                .or_else(|| self.stop_sequences.clone()),
        }
    }
}

/// Configuration for the LLM Provider actor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// The type of LLM provider
    pub provider_type: ProviderType,
    /// The API key for authentication (may be empty for local providers)
    pub api_key: String,
    /// The model to use (e.g., "claude-sonnet-4-20250514")
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Base URL for the API
    pub base_url: String,
    /// API version header
    pub api_version: String,
    /// Request timeout
    pub timeout: Duration,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// Retry configuration
    pub retry: RetryConfig,
    /// Default sampling parameters for this provider
    #[serde(default, skip_serializing_if = "SamplingParams::is_empty")]
    pub sampling: SamplingParams,
}

impl ProviderConfig {
    /// Creates a new provider configuration with the given API key.
    ///
    /// Uses default values for all other settings optimized for Anthropic Claude.
    /// This is equivalent to calling `anthropic()` for backwards compatibility.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The Anthropic API key
    ///
    /// # Examples
    ///
    /// ```
    /// use acton_ai::llm::ProviderConfig;
    ///
    /// let config = ProviderConfig::new("sk-ant-api03-...");
    /// assert!(!config.api_key.is_empty());
    /// ```
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::anthropic(api_key)
    }

    /// Creates a new provider configuration for Anthropic.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The Anthropic API key
    ///
    /// # Examples
    ///
    /// ```
    /// use acton_ai::llm::ProviderConfig;
    ///
    /// let config = ProviderConfig::anthropic("sk-ant-api03-...");
    /// assert!(!config.api_key.is_empty());
    /// ```
    #[must_use]
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::Anthropic,
            api_key: api_key.into(),
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            base_url: "https://api.anthropic.com".to_string(),
            api_version: "2023-06-01".to_string(),
            timeout: Duration::from_secs(120),
            rate_limit: RateLimitConfig::default(),
            retry: RetryConfig::default(),
            sampling: SamplingParams::default(),
        }
    }

    /// Creates a new provider configuration for Ollama.
    ///
    /// Ollama runs locally and does not require an API key.
    ///
    /// # Arguments
    ///
    /// * `model` - The model name (e.g., "llama3.2", "mistral")
    ///
    /// # Examples
    ///
    /// ```
    /// use acton_ai::llm::ProviderConfig;
    ///
    /// let config = ProviderConfig::ollama("llama3.2");
    /// assert!(config.api_key.is_empty());
    /// ```
    #[must_use]
    pub fn ollama(model: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::ollama(),
            api_key: String::new(),
            model: model.into(),
            max_tokens: 4096,
            base_url: "http://localhost:11434/v1".to_string(),
            api_version: String::new(),
            timeout: Duration::from_secs(300), // Longer timeout for local inference
            rate_limit: RateLimitConfig::new(1000, 1_000_000), // High limits for local
            retry: RetryConfig::default(),
            sampling: SamplingParams::default(),
        }
    }

    /// Creates a new provider configuration for OpenAI.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The OpenAI API key
    ///
    /// # Examples
    ///
    /// ```
    /// use acton_ai::llm::ProviderConfig;
    ///
    /// let config = ProviderConfig::openai("sk-...");
    /// assert!(!config.api_key.is_empty());
    /// ```
    #[must_use]
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::openai(),
            api_key: api_key.into(),
            model: "gpt-4o".to_string(),
            max_tokens: 4096,
            base_url: "https://api.openai.com/v1".to_string(),
            api_version: String::new(),
            timeout: Duration::from_secs(120),
            rate_limit: RateLimitConfig::default(),
            retry: RetryConfig::default(),
            sampling: SamplingParams::default(),
        }
    }

    /// Creates a configuration for a custom OpenAI-compatible endpoint.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of the API
    /// * `model` - The model name
    ///
    /// # Examples
    ///
    /// ```
    /// use acton_ai::llm::ProviderConfig;
    ///
    /// let config = ProviderConfig::openai_compatible("http://localhost:8080/v1", "custom-model");
    /// assert!(config.api_key.is_empty());
    /// ```
    #[must_use]
    pub fn openai_compatible(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let base_url_string = base_url.into();
        Self {
            provider_type: ProviderType::openai_compatible(&base_url_string),
            api_key: String::new(),
            model: model.into(),
            max_tokens: 4096,
            base_url: base_url_string,
            api_version: String::new(),
            timeout: Duration::from_secs(300),
            rate_limit: RateLimitConfig::new(1000, 1_000_000),
            retry: RetryConfig::default(),
            sampling: SamplingParams::default(),
        }
    }

    /// Sets the provider type.
    #[must_use]
    pub fn with_provider_type(mut self, provider_type: ProviderType) -> Self {
        self.provider_type = provider_type;
        self
    }

    /// Sets the API key.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Sets the model to use.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the maximum tokens to generate.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets the base URL for the API.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the API version header.
    #[must_use]
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = api_version.into();
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the rate limit configuration.
    #[must_use]
    pub fn with_rate_limit(mut self, rate_limit: RateLimitConfig) -> Self {
        self.rate_limit = rate_limit;
        self
    }

    /// Sets the retry configuration.
    #[must_use]
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// Sets the default sampling parameters for this provider.
    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }

    /// Sets the default temperature for this provider.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.sampling.temperature = Some(temperature);
        self
    }

    /// Sets the default top_p for this provider.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.sampling.top_p = Some(top_p);
        self
    }

    /// Sets the default top_k for this provider.
    #[must_use]
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.sampling.top_k = Some(top_k);
        self
    }

    /// Sets the default stop sequences for this provider.
    #[must_use]
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.sampling.stop_sequences = Some(sequences);
        self
    }

    /// Returns the full API endpoint URL for messages.
    #[must_use]
    pub fn messages_endpoint(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }
}

/// Rate limiting configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per minute
    pub requests_per_minute: u32,
    /// Maximum tokens per minute (input + output)
    pub tokens_per_minute: u32,
    /// Whether to queue requests when rate limited
    pub queue_when_limited: bool,
    /// Maximum queue size (0 = unlimited)
    pub max_queue_size: usize,
}

impl RateLimitConfig {
    /// Creates a new rate limit configuration.
    #[must_use]
    pub fn new(requests_per_minute: u32, tokens_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            tokens_per_minute,
            queue_when_limited: true,
            max_queue_size: 100,
        }
    }

    /// Disables queuing when rate limited (requests will fail immediately).
    #[must_use]
    pub fn without_queueing(mut self) -> Self {
        self.queue_when_limited = false;
        self
    }

    /// Sets the maximum queue size.
    #[must_use]
    pub fn with_max_queue_size(mut self, max_size: usize) -> Self {
        self.max_queue_size = max_size;
        self
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        // Default limits for Anthropic API (Tier 1)
        Self {
            requests_per_minute: 50,
            tokens_per_minute: 40_000,
            queue_when_limited: true,
            max_queue_size: 100,
        }
    }
}

/// Retry configuration for failed requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier (exponential factor)
    pub backoff_multiplier: u32,
    /// Whether to add jitter to backoff
    pub jitter: bool,
}

impl RetryConfig {
    /// Creates a new retry configuration.
    #[must_use]
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(60),
            backoff_multiplier: 2,
            jitter: true,
        }
    }

    /// Disables retries.
    #[must_use]
    pub fn no_retries() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Sets the initial backoff duration.
    #[must_use]
    pub fn with_initial_backoff(mut self, duration: Duration) -> Self {
        self.initial_backoff = duration;
        self
    }

    /// Sets the maximum backoff duration.
    #[must_use]
    pub fn with_max_backoff(mut self, duration: Duration) -> Self {
        self.max_backoff = duration;
        self
    }

    /// Sets the backoff multiplier.
    #[must_use]
    pub fn with_backoff_multiplier(mut self, multiplier: u32) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Disables jitter.
    #[must_use]
    pub fn without_jitter(mut self) -> Self {
        self.jitter = false;
        self
    }

    /// Calculates the backoff duration for a given attempt.
    #[must_use]
    pub fn backoff_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let multiplier = self.backoff_multiplier.saturating_pow(attempt - 1);
        let backoff = self.initial_backoff.saturating_mul(multiplier);

        if backoff > self.max_backoff {
            self.max_backoff
        } else {
            backoff
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(60),
            backoff_multiplier: 2,
            jitter: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ProviderType tests
    #[test]
    fn provider_type_default_is_anthropic() {
        assert_eq!(ProviderType::default(), ProviderType::Anthropic);
    }

    #[test]
    fn provider_type_ollama_creates_correct_url() {
        let provider = ProviderType::ollama();
        assert!(matches!(
            provider,
            ProviderType::OpenAI { base_url } if base_url == "http://localhost:11434/v1"
        ));
    }

    #[test]
    fn provider_type_openai_creates_correct_url() {
        let provider = ProviderType::openai();
        assert!(matches!(
            provider,
            ProviderType::OpenAI { base_url } if base_url == "https://api.openai.com/v1"
        ));
    }

    #[test]
    fn provider_type_openai_compatible_uses_custom_url() {
        let provider = ProviderType::openai_compatible("http://custom:8000/v1");
        assert!(matches!(
            provider,
            ProviderType::OpenAI { base_url } if base_url == "http://custom:8000/v1"
        ));
    }

    // ProviderConfig tests
    #[test]
    fn provider_config_new() {
        let config = ProviderConfig::new("test-api-key");

        assert_eq!(config.provider_type, ProviderType::Anthropic);
        assert_eq!(config.api_key, "test-api-key");
        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn provider_config_anthropic_creates_anthropic_provider() {
        let config = ProviderConfig::anthropic("test-key");
        assert_eq!(config.provider_type, ProviderType::Anthropic);
        assert_eq!(config.api_key, "test-key");
        assert!(config.model.contains("claude"));
    }

    #[test]
    fn provider_config_ollama_has_empty_api_key() {
        let config = ProviderConfig::ollama("llama3.2");
        assert!(matches!(config.provider_type, ProviderType::OpenAI { .. }));
        assert!(config.api_key.is_empty());
        assert_eq!(config.model, "llama3.2");
        assert_eq!(config.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn provider_config_openai_creates_openai_provider() {
        let config = ProviderConfig::openai("test-key");
        assert!(matches!(config.provider_type, ProviderType::OpenAI { .. }));
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model, "gpt-4o");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn provider_config_openai_compatible_creates_custom_provider() {
        let config = ProviderConfig::openai_compatible("http://custom:8000/v1", "custom-model");
        assert!(matches!(config.provider_type, ProviderType::OpenAI { .. }));
        assert!(config.api_key.is_empty());
        assert_eq!(config.model, "custom-model");
        assert_eq!(config.base_url, "http://custom:8000/v1");
    }

    #[test]
    fn provider_config_new_is_anthropic_for_backwards_compat() {
        let config = ProviderConfig::new("test-key");
        assert_eq!(config.provider_type, ProviderType::Anthropic);
    }

    #[test]
    fn provider_config_builder_pattern() {
        let config = ProviderConfig::new("test-key")
            .with_model("claude-3-haiku-20240307")
            .with_max_tokens(1024)
            .with_timeout(Duration::from_secs(30));

        assert_eq!(config.model, "claude-3-haiku-20240307");
        assert_eq!(config.max_tokens, 1024);
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn provider_config_messages_endpoint() {
        let config = ProviderConfig::new("test-key");
        assert_eq!(
            config.messages_endpoint(),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn provider_config_custom_base_url() {
        let config = ProviderConfig::new("test-key").with_base_url("https://custom.api.com");

        assert_eq!(
            config.messages_endpoint(),
            "https://custom.api.com/v1/messages"
        );
    }

    #[test]
    fn provider_config_with_provider_type() {
        let config =
            ProviderConfig::anthropic("test-key").with_provider_type(ProviderType::ollama());

        assert!(matches!(config.provider_type, ProviderType::OpenAI { .. }));
    }

    #[test]
    fn provider_config_with_api_key() {
        let config = ProviderConfig::ollama("llama3.2").with_api_key("new-key");

        assert_eq!(config.api_key, "new-key");
    }

    #[test]
    fn rate_limit_config_default() {
        let config = RateLimitConfig::default();

        assert_eq!(config.requests_per_minute, 50);
        assert_eq!(config.tokens_per_minute, 40_000);
        assert!(config.queue_when_limited);
    }

    #[test]
    fn rate_limit_config_without_queueing() {
        let config = RateLimitConfig::default().without_queueing();

        assert!(!config.queue_when_limited);
    }

    #[test]
    fn retry_config_default() {
        let config = RetryConfig::default();

        assert_eq!(config.max_retries, 3);
        assert!(config.jitter);
    }

    #[test]
    fn retry_config_no_retries() {
        let config = RetryConfig::no_retries();

        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn retry_config_backoff_for_attempt() {
        let config = RetryConfig::new(5)
            .with_initial_backoff(Duration::from_secs(1))
            .with_backoff_multiplier(2)
            .without_jitter();

        assert_eq!(config.backoff_for_attempt(0), Duration::ZERO);
        assert_eq!(config.backoff_for_attempt(1), Duration::from_secs(1));
        assert_eq!(config.backoff_for_attempt(2), Duration::from_secs(2));
        assert_eq!(config.backoff_for_attempt(3), Duration::from_secs(4));
        assert_eq!(config.backoff_for_attempt(4), Duration::from_secs(8));
    }

    #[test]
    fn retry_config_backoff_respects_max() {
        let config = RetryConfig::new(10)
            .with_initial_backoff(Duration::from_secs(1))
            .with_max_backoff(Duration::from_secs(5))
            .with_backoff_multiplier(2);

        // 1, 2, 4, 8 -> should cap at 5
        assert_eq!(config.backoff_for_attempt(4), Duration::from_secs(5));
        assert_eq!(config.backoff_for_attempt(10), Duration::from_secs(5));
    }

    #[test]
    fn provider_config_serialization_roundtrip() {
        let config = ProviderConfig::new("test-key").with_model("claude-3-haiku-20240307");

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ProviderConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }

    // SamplingParams tests

    #[test]
    fn sampling_params_default_is_all_none() {
        let params = SamplingParams::default();
        assert!(params.temperature.is_none());
        assert!(params.top_k.is_none());
        assert!(params.top_p.is_none());
        assert!(params.frequency_penalty.is_none());
        assert!(params.presence_penalty.is_none());
        assert!(params.seed.is_none());
        assert!(params.stop_sequences.is_none());
    }

    #[test]
    fn sampling_params_is_empty_when_default() {
        assert!(SamplingParams::default().is_empty());
    }

    #[test]
    fn sampling_params_is_not_empty_with_temperature() {
        let params = SamplingParams::new().with_temperature(0.7);
        assert!(!params.is_empty());
    }

    #[test]
    fn sampling_params_builder_pattern() {
        let params = SamplingParams::new()
            .with_temperature(0.7)
            .with_top_k(40)
            .with_top_p(0.9)
            .with_frequency_penalty(0.5)
            .with_presence_penalty(0.3)
            .with_seed(42)
            .with_stop_sequences(vec!["END".to_string()]);

        assert_eq!(params.temperature, Some(0.7));
        assert_eq!(params.top_k, Some(40));
        assert_eq!(params.top_p, Some(0.9));
        assert_eq!(params.frequency_penalty, Some(0.5));
        assert_eq!(params.presence_penalty, Some(0.3));
        assert_eq!(params.seed, Some(42));
        assert_eq!(
            params.stop_sequences,
            Some(vec!["END".to_string()])
        );
    }

    #[test]
    fn sampling_params_merge_override_takes_precedence() {
        let base = SamplingParams::new().with_temperature(0.5);
        let overrides = SamplingParams::new().with_temperature(0.9);

        let merged = base.merge_with(&overrides);
        assert_eq!(merged.temperature, Some(0.9));
    }

    #[test]
    fn sampling_params_merge_preserves_base_when_override_is_none() {
        let base = SamplingParams::new().with_top_p(0.8).with_temperature(0.5);
        let overrides = SamplingParams::new().with_temperature(0.9);

        let merged = base.merge_with(&overrides);
        assert_eq!(merged.temperature, Some(0.9));
        assert_eq!(merged.top_p, Some(0.8));
    }

    #[test]
    fn sampling_params_merge_both_empty() {
        let base = SamplingParams::default();
        let overrides = SamplingParams::default();
        let merged = base.merge_with(&overrides);
        assert!(merged.is_empty());
    }

    #[test]
    fn sampling_params_serialization_roundtrip() {
        let params = SamplingParams::new()
            .with_temperature(0.7)
            .with_top_k(40);

        let json = serde_json::to_string(&params).unwrap();
        let deserialized: SamplingParams = serde_json::from_str(&json).unwrap();

        assert_eq!(params, deserialized);
    }

    #[test]
    fn sampling_params_serialization_skips_none() {
        let params = SamplingParams::new().with_temperature(0.7);
        let json = serde_json::to_string(&params).unwrap();

        assert!(json.contains("temperature"));
        assert!(!json.contains("top_k"));
        assert!(!json.contains("top_p"));
    }

    #[test]
    fn provider_config_with_sampling() {
        let config = ProviderConfig::new("test-key")
            .with_sampling(SamplingParams::new().with_temperature(0.7));

        assert_eq!(config.sampling.temperature, Some(0.7));
    }

    #[test]
    fn provider_config_with_temperature() {
        let config = ProviderConfig::new("test-key").with_temperature(0.5);
        assert_eq!(config.sampling.temperature, Some(0.5));
    }

    #[test]
    fn provider_config_serialization_roundtrip_with_sampling() {
        let config = ProviderConfig::new("test-key")
            .with_model("claude-3-haiku-20240307")
            .with_temperature(0.7);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ProviderConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }
}
