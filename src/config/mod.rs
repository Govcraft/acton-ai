//! Configuration management for acton-ai.
//!
//! This module provides types and functions for loading and managing
//! acton-ai configuration, including support for multiple named LLM providers.
//!
//! # Configuration File Format
//!
//! Configuration is stored in TOML format. The search order is:
//! 1. `./acton-ai.toml` (project-local)
//! 2. `~/.config/acton-ai/config.toml` (XDG config)
//!
//! # Example Configuration
//!
//! ```toml
//! # Define multiple providers by name
//! [providers.claude]
//! type = "anthropic"
//! model = "claude-sonnet-4-20250514"
//! api_key_env = "ANTHROPIC_API_KEY"
//!
//! [providers.ollama]
//! type = "ollama"
//! model = "qwen2.5:7b"
//! base_url = "http://localhost:11434/v1"
//! timeout_secs = 300
//!
//! [providers.ollama.rate_limit]
//! requests_per_minute = 1000
//! tokens_per_minute = 1000000
//!
//! [providers.fast]
//! type = "openai"
//! model = "gpt-4o-mini"
//! api_key_env = "OPENAI_API_KEY"
//!
//! # Which provider to use when none specified
//! default_provider = "ollama"
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use acton_ai::config;
//!
//! // Load from default search paths
//! let config = config::load()?;
//!
//! // Load from a specific path
//! let config = config::from_path(Path::new("/etc/acton-ai/config.toml"))?;
//!
//! // Parse from a string
//! let config = config::from_str(toml_content)?;
//! ```

mod file;
mod types;

// Re-export file loading functions
pub use file::{from_path, from_str, load, search_paths, xdg_config_dir};

// Re-export types
pub use types::{ActonAIConfig, NamedProviderConfig, RateLimitFileConfig};
