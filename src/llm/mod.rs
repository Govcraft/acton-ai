//! LLM provider module.
//!
//! This module contains the LLM Provider actor implementation, Anthropic API client,
//! and streaming message handling for token-by-token responses.

mod anthropic;
mod config;
mod error;
mod provider;
mod streaming;

pub use anthropic::AnthropicClient;
pub use config::{ProviderConfig, RateLimitConfig};
pub use error::{LLMError, LLMErrorKind};
pub use provider::{InitLLMProvider, LLMProvider};
pub use streaming::{ActiveStream, StreamAccumulator};
