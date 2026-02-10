//! LLM provider module.
//!
//! This module contains the LLM Provider actor implementation, API clients for
//! Anthropic and OpenAI-compatible endpoints, and streaming message handling
//! for token-by-token responses.

mod anthropic;
mod client;
mod config;
mod error;
mod openai;
mod provider;
mod streaming;

pub use anthropic::AnthropicClient;
pub use client::{LLMClient, LLMClientResponse, LLMEventStream, LLMStreamEvent};
pub use config::{ProviderConfig, ProviderType, RateLimitConfig, SamplingParams};
pub use error::{LLMError, LLMErrorKind};
pub use openai::OpenAIClient;
pub use provider::{InitLLMProvider, LLMProvider};
pub use streaming::{ActiveStream, StreamAccumulator};
