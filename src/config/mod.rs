//! Configuration management for acton-ai.
//!
//! This module provides types and functions for loading and managing
//! acton-ai configuration, including support for multiple named LLM providers
//! and sandbox settings.
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
//!
//! # Sandbox configuration (optional). Tool calls that write run in a
//! # re-exec'd child of this binary, hardened with landlock and seccomp
//! # where the kernel offers them.
//! [sandbox]
//! hardening = "best-effort"   # "off" | "best-effort" | "enforce"
//! # Replaces the default list; name every variable the child still needs.
//! env_allowlist = ["PATH", "LANG", "LC_ALL", "HOME", "TMPDIR", "UV_CACHE_DIR"]
//!
//! [sandbox.limits]
//! max_execution_ms = 30000
//! max_memory_mb = 64
//!
//! # Directories the hardened child may reach beyond the system paths,
//! # `$TMPDIR` and the session root. User-installed toolchains live here:
//! # without an entry, a binary the shell finds on PATH is refused by the
//! # kernel as a bare "Permission denied".
//! [sandbox.paths]
//! read_exec = ["~/.local/bin", "~/.local/share/uv"]
//! read_write = ["~/.cache/uv"]
//!
//! # External MCP servers. Use `command` (stdio) or `url` (streamable HTTP),
//! # never both. Their tools reach the LLM as `mcp__{server}__{tool}`.
//! [mcp_servers.filesystem]
//! command = "npx"
//! args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
//!
//! [mcp_servers.linear]
//! url = "https://mcp.linear.app/mcp"
//! auth_token_env = "LINEAR_MCP_TOKEN"
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
pub use types::{
    parse_truncation_strategy, ActonAIConfig, ActonAIDefaults, AuditFileConfig, BudgetFileConfig,
    CheckpointFileConfig, CircuitBreakerFileConfig, CliFileConfig, ContextFileConfig,
    IntrospectionFileConfig, JobConfig, McpServerConfig, NamedProviderConfig,
    PersistenceFileConfig, PricingFileConfig, RateLimitFileConfig, SandboxFileConfig,
    SandboxLimitsConfig, SandboxPathsConfig, SkillsFileConfig, TelemetryFileConfig,
    ToolPolicyFileConfig, DEFAULT_MCP_TOOL_TIMEOUT_SECS,
};
