//! LLM error types.
//!
//! Custom error types for LLM operations including network errors,
//! rate limiting, API errors, and streaming errors.

use std::fmt;
use std::time::Duration;

/// Errors that can occur in the LLM provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LLMError {
    /// The specific error that occurred
    pub kind: LLMErrorKind,
}

/// Specific LLM error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LLMErrorKind {
    /// Network error when communicating with the API
    Network {
        /// Description of the network error
        message: String,
    },
    /// Rate limit exceeded
    RateLimited {
        /// Time to wait before retrying
        retry_after: Duration,
    },
    /// API returned an error response
    ApiError {
        /// HTTP status code
        status_code: u16,
        /// Error message from the API
        message: String,
        /// Error type from the API (if available)
        error_type: Option<String>,
    },
    /// Authentication failed
    AuthenticationFailed {
        /// Reason for authentication failure
        reason: String,
    },
    /// Invalid request parameters
    InvalidRequest {
        /// Description of what was invalid
        reason: String,
    },
    /// Streaming error
    StreamError {
        /// Description of the streaming error
        message: String,
    },
    /// JSON parsing error
    ParseError {
        /// Description of the parse error
        message: String,
    },
    /// Provider is shutting down
    ShuttingDown,
    /// Configuration error
    InvalidConfig {
        /// The configuration field that was invalid
        field: String,
        /// Why it was invalid
        reason: String,
    },
    /// Model overloaded or unavailable
    ModelOverloaded {
        /// The model that was overloaded
        model: String,
    },
    /// Request timeout
    Timeout {
        /// The timeout duration that was exceeded
        duration: Duration,
    },
}

impl LLMError {
    /// Creates a new LLMError with the given kind.
    #[must_use]
    pub fn new(kind: LLMErrorKind) -> Self {
        Self { kind }
    }

    /// Creates a network error.
    #[must_use]
    pub fn network(message: impl Into<String>) -> Self {
        Self::new(LLMErrorKind::Network {
            message: message.into(),
        })
    }

    /// Creates a rate limited error.
    #[must_use]
    pub fn rate_limited(retry_after: Duration) -> Self {
        Self::new(LLMErrorKind::RateLimited { retry_after })
    }

    /// Creates an API error.
    #[must_use]
    pub fn api_error(
        status_code: u16,
        message: impl Into<String>,
        error_type: Option<String>,
    ) -> Self {
        Self::new(LLMErrorKind::ApiError {
            status_code,
            message: message.into(),
            error_type,
        })
    }

    /// Creates an authentication failed error.
    #[must_use]
    pub fn authentication_failed(reason: impl Into<String>) -> Self {
        Self::new(LLMErrorKind::AuthenticationFailed {
            reason: reason.into(),
        })
    }

    /// Creates an invalid request error.
    #[must_use]
    pub fn invalid_request(reason: impl Into<String>) -> Self {
        Self::new(LLMErrorKind::InvalidRequest {
            reason: reason.into(),
        })
    }

    /// Creates a stream error.
    #[must_use]
    pub fn stream_error(message: impl Into<String>) -> Self {
        Self::new(LLMErrorKind::StreamError {
            message: message.into(),
        })
    }

    /// Creates a parse error.
    #[must_use]
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::new(LLMErrorKind::ParseError {
            message: message.into(),
        })
    }

    /// Creates a shutting down error.
    #[must_use]
    pub fn shutting_down() -> Self {
        Self::new(LLMErrorKind::ShuttingDown)
    }

    /// Creates an invalid config error.
    #[must_use]
    pub fn invalid_config(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(LLMErrorKind::InvalidConfig {
            field: field.into(),
            reason: reason.into(),
        })
    }

    /// Creates a model overloaded error.
    #[must_use]
    pub fn model_overloaded(model: impl Into<String>) -> Self {
        Self::new(LLMErrorKind::ModelOverloaded {
            model: model.into(),
        })
    }

    /// Creates a timeout error.
    #[must_use]
    pub fn timeout(duration: Duration) -> Self {
        Self::new(LLMErrorKind::Timeout { duration })
    }

    /// Returns true if this error is retriable.
    #[must_use]
    pub fn is_retriable(&self) -> bool {
        matches!(
            self.kind,
            LLMErrorKind::Network { .. }
                | LLMErrorKind::RateLimited { .. }
                | LLMErrorKind::ModelOverloaded { .. }
                | LLMErrorKind::Timeout { .. }
                | LLMErrorKind::ApiError {
                    status_code: 500..=599,
                    ..
                }
        )
    }

    /// Returns the retry-after duration if this is a rate limit error.
    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        match &self.kind {
            LLMErrorKind::RateLimited { retry_after } => Some(*retry_after),
            _ => None,
        }
    }
}

impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            LLMErrorKind::Network { message } => {
                write!(
                    f,
                    "network error communicating with LLM API: {}; check network connectivity",
                    message
                )
            }
            LLMErrorKind::RateLimited { retry_after } => {
                write!(
                    f,
                    "rate limit exceeded; retry after {} seconds",
                    retry_after.as_secs()
                )
            }
            LLMErrorKind::ApiError {
                status_code,
                message,
                error_type,
            } => {
                if let Some(error_type) = error_type {
                    write!(
                        f,
                        "API error (HTTP {}): {} (type: {})",
                        status_code, message, error_type
                    )
                } else {
                    write!(f, "API error (HTTP {}): {}", status_code, message)
                }
            }
            LLMErrorKind::AuthenticationFailed { reason } => {
                write!(
                    f,
                    "authentication failed: {}; verify API key is valid",
                    reason
                )
            }
            LLMErrorKind::InvalidRequest { reason } => {
                write!(f, "invalid request: {}; check request parameters", reason)
            }
            LLMErrorKind::StreamError { message } => {
                write!(f, "streaming error: {}", message)
            }
            LLMErrorKind::ParseError { message } => {
                write!(f, "failed to parse API response: {}", message)
            }
            LLMErrorKind::ShuttingDown => {
                write!(
                    f,
                    "LLM provider is shutting down; cannot accept new requests"
                )
            }
            LLMErrorKind::InvalidConfig { field, reason } => {
                write!(f, "invalid configuration for '{}': {}", field, reason)
            }
            LLMErrorKind::ModelOverloaded { model } => {
                write!(
                    f,
                    "model '{}' is overloaded; retry after a short delay",
                    model
                )
            }
            LLMErrorKind::Timeout { duration } => {
                write!(f, "request timed out after {} seconds", duration.as_secs())
            }
        }
    }
}

impl std::error::Error for LLMError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_error_display() {
        let error = LLMError::network("connection refused");

        let message = error.to_string();
        assert!(message.contains("network error"));
        assert!(message.contains("connection refused"));
    }

    #[test]
    fn rate_limited_error_display() {
        let error = LLMError::rate_limited(Duration::from_secs(30));

        let message = error.to_string();
        assert!(message.contains("rate limit"));
        assert!(message.contains("30"));
    }

    #[test]
    fn api_error_with_type_display() {
        let error = LLMError::api_error(
            400,
            "invalid model parameter",
            Some("invalid_request_error".to_string()),
        );

        let message = error.to_string();
        assert!(message.contains("400"));
        assert!(message.contains("invalid model parameter"));
        assert!(message.contains("invalid_request_error"));
    }

    #[test]
    fn api_error_without_type_display() {
        let error = LLMError::api_error(500, "internal server error", None);

        let message = error.to_string();
        assert!(message.contains("500"));
        assert!(message.contains("internal server error"));
        assert!(!message.contains("type:"));
    }

    #[test]
    fn is_retriable_for_network_errors() {
        assert!(LLMError::network("timeout").is_retriable());
    }

    #[test]
    fn is_retriable_for_rate_limited() {
        assert!(LLMError::rate_limited(Duration::from_secs(10)).is_retriable());
    }

    #[test]
    fn is_retriable_for_server_errors() {
        assert!(LLMError::api_error(500, "internal error", None).is_retriable());
        assert!(LLMError::api_error(503, "service unavailable", None).is_retriable());
    }

    #[test]
    fn is_not_retriable_for_client_errors() {
        assert!(!LLMError::api_error(400, "bad request", None).is_retriable());
        assert!(!LLMError::api_error(401, "unauthorized", None).is_retriable());
    }

    #[test]
    fn is_not_retriable_for_auth_errors() {
        assert!(!LLMError::authentication_failed("invalid key").is_retriable());
    }

    #[test]
    fn retry_after_returns_duration_for_rate_limited() {
        let error = LLMError::rate_limited(Duration::from_secs(60));
        assert_eq!(error.retry_after(), Some(Duration::from_secs(60)));
    }

    #[test]
    fn retry_after_returns_none_for_other_errors() {
        let error = LLMError::network("connection refused");
        assert_eq!(error.retry_after(), None);
    }

    #[test]
    fn errors_are_clone() {
        let error1 = LLMError::shutting_down();
        let error2 = error1.clone();
        assert_eq!(error1, error2);
    }

    #[test]
    fn errors_are_eq() {
        let error1 = LLMError::shutting_down();
        let error2 = LLMError::shutting_down();
        assert_eq!(error1, error2);

        let error3 = LLMError::network("different");
        assert_ne!(error1, error3);
    }

    #[test]
    fn timeout_error_display() {
        let error = LLMError::timeout(Duration::from_secs(120));

        let message = error.to_string();
        assert!(message.contains("timed out"));
        assert!(message.contains("120"));
    }

    #[test]
    fn model_overloaded_is_retriable() {
        let error = LLMError::model_overloaded("claude-3-opus-20240229");
        assert!(error.is_retriable());
    }
}
