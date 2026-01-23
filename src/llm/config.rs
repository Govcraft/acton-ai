//! LLM provider configuration.
//!
//! Configuration types for the LLM Provider actor including API settings,
//! rate limiting, and retry behavior.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the LLM Provider actor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// The API key for authentication
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
}

impl ProviderConfig {
    /// Creates a new provider configuration with the given API key.
    ///
    /// Uses default values for all other settings optimized for Anthropic Claude.
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
        Self {
            api_key: api_key.into(),
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            base_url: "https://api.anthropic.com".to_string(),
            api_version: "2023-06-01".to_string(),
            timeout: Duration::from_secs(120),
            rate_limit: RateLimitConfig::default(),
            retry: RetryConfig::default(),
        }
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

    #[test]
    fn provider_config_new() {
        let config = ProviderConfig::new("test-api-key");

        assert_eq!(config.api_key, "test-api-key");
        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_tokens, 4096);
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
}
