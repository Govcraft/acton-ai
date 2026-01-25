//! High-level facade for ActonAI.
//!
//! This module provides a simplified API for common use cases, hiding the
//! complexity of actor setup, kernel spawning, and provider configuration.
//!
//! # Single Provider Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), ActonAIError> {
//!     let runtime = ActonAI::builder()
//!         .app_name("my-app")
//!         .ollama("qwen2.5:7b")
//!         .launch()
//!         .await?;
//!
//!     runtime
//!         .prompt("What is the capital of France?")
//!         .on_token(|t| print!("{t}"))
//!         .collect()
//!         .await?;
//!
//!     println!();
//!     Ok(())
//! }
//! ```
//!
//! # Multi-Provider Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), ActonAIError> {
//!     let runtime = ActonAI::builder()
//!         .app_name("my-app")
//!         .provider_named("claude", ProviderConfig::anthropic("sk-..."))
//!         .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
//!         .default_provider("local")
//!         .launch()
//!         .await?;
//!
//!     // Use default provider (local)
//!     runtime.prompt("Quick question").collect().await?;
//!
//!     // Use specific provider
//!     runtime.prompt("Complex reasoning").provider("claude").collect().await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Config File Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), ActonAIError> {
//!     let runtime = ActonAI::builder()
//!         .app_name("my-app")
//!         .from_config()?  // Load providers from config file
//!         .with_builtins()
//!         .launch()
//!         .await?;
//!
//!     runtime.prompt("Hello").collect().await?;
//!     Ok(())
//! }
//! ```

use crate::config::{self, ActonAIConfig};
use crate::conversation::ConversationBuilder;
use crate::error::{ActonAIError, ActonAIErrorKind};
use crate::kernel::{Kernel, KernelConfig};
use crate::llm::{LLMProvider, ProviderConfig};
use crate::messages::Message;
use crate::prompt::PromptBuilder;
use crate::tools::builtins::BuiltinTools;
use crate::tools::sandbox::{HyperlightSandboxFactory, SandboxConfig, SandboxFactory, SandboxPool};
use acton_reactive::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// The default provider name used when registering single providers.
pub const DEFAULT_PROVIDER_NAME: &str = "default";

/// High-level facade for interacting with ActonAI.
///
/// `ActonAI` encapsulates the runtime, kernel, and LLM providers, providing
/// a simplified API for common operations. It handles all the actor setup
/// and subscription management automatically.
///
/// # Single Provider Example
///
/// ```rust,ignore
/// let runtime = ActonAI::builder()
///     .app_name("my-app")
///     .ollama("llama3.2")
///     .launch()
///     .await?;
///
/// runtime
///     .prompt("Hello!")
///     .on_token(|t| print!("{t}"))
///     .collect()
///     .await?;
/// ```
///
/// # Multi-Provider Example
///
/// ```rust,ignore
/// let runtime = ActonAI::builder()
///     .app_name("my-app")
///     .provider_named("claude", ProviderConfig::anthropic("sk-..."))
///     .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
///     .default_provider("local")
///     .launch()
///     .await?;
///
/// // Use specific provider
/// runtime.prompt("Complex task").provider("claude").collect().await?;
/// ```
pub struct ActonAI {
    /// The underlying actor runtime
    runtime: ActorRuntime,
    /// Named LLM provider handles
    providers: HashMap<String, ActorHandle>,
    /// The name of the default provider
    default_provider: String,
    /// Built-in tools (if enabled)
    builtins: Option<BuiltinTools>,
    /// Whether to automatically enable builtins on each prompt
    auto_builtins: bool,
    /// Whether the runtime has been shut down
    is_shutdown: bool,
}

impl std::fmt::Debug for ActonAI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActonAI")
            .field("is_shutdown", &self.is_shutdown)
            .field("has_builtins", &self.builtins.is_some())
            .field("auto_builtins", &self.auto_builtins)
            .field("provider_count", &self.providers.len())
            .field("default_provider", &self.default_provider)
            .finish_non_exhaustive()
    }
}

impl ActonAI {
    /// Creates a new builder for configuring ActonAI.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("llama3.2")
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn builder() -> ActonAIBuilder {
        ActonAIBuilder::default()
    }

    /// Creates a prompt builder for sending a message to the LLM.
    ///
    /// If built-in tools were configured with [`with_builtins`](ActonAIBuilder::with_builtins)
    /// or [`with_builtin_tools`](ActonAIBuilder::with_builtin_tools), they are automatically
    /// enabled on the prompt. Use [`manual_builtins`](ActonAIBuilder::manual_builtins) during
    /// builder configuration to disable auto-enabling.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime
    ///     .prompt("What is 2 + 2?")
    ///     .system("Be concise.")
    ///     .on_token(|t| print!("{t}"))
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn prompt(&self, content: impl Into<String>) -> PromptBuilder<'_> {
        let mut builder = PromptBuilder::new(self, content.into());
        if self.auto_builtins && self.builtins.is_some() {
            builder = builder.use_builtins();
        }
        builder
    }

    /// Returns a reference to the underlying actor runtime.
    ///
    /// This provides an escape hatch for advanced use cases that need
    /// direct access to the actor system.
    #[must_use]
    pub fn runtime(&self) -> &ActorRuntime {
        &self.runtime
    }

    /// Returns a mutable reference to the underlying actor runtime.
    ///
    /// This provides an escape hatch for advanced use cases that need
    /// direct access to the actor system.
    pub fn runtime_mut(&mut self) -> &mut ActorRuntime {
        &mut self.runtime
    }

    /// Returns a clone of the default LLM provider handle.
    ///
    /// This can be used to send requests directly to the provider
    /// for advanced use cases.
    #[must_use]
    pub fn provider_handle(&self) -> ActorHandle {
        self.providers
            .get(&self.default_provider)
            .cloned()
            .expect("default provider must exist")
    }

    /// Returns a clone of a named LLM provider handle.
    ///
    /// Returns `None` if no provider with the given name exists.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(handle) = runtime.provider_handle_named("claude") {
    ///     // Send requests directly to the claude provider
    /// }
    /// ```
    #[must_use]
    pub fn provider_handle_named(&self, name: &str) -> Option<ActorHandle> {
        self.providers.get(name).cloned()
    }

    /// Returns the name of the default provider.
    #[must_use]
    pub fn default_provider_name(&self) -> &str {
        &self.default_provider
    }

    /// Returns an iterator over the names of all registered providers.
    pub fn provider_names(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    /// Returns the number of registered providers.
    #[must_use]
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Returns true if a provider with the given name exists.
    #[must_use]
    pub fn has_provider(&self, name: &str) -> bool {
        self.providers.contains_key(name)
    }

    /// Returns whether the runtime has been shut down.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown
    }

    /// Returns a reference to the built-in tools, if enabled.
    ///
    /// Returns `None` if built-in tools were not configured with
    /// [`with_builtins`](ActonAIBuilder::with_builtins) or
    /// [`with_builtin_tools`](ActonAIBuilder::with_builtin_tools).
    #[must_use]
    pub fn builtins(&self) -> Option<&BuiltinTools> {
        self.builtins.as_ref()
    }

    /// Returns whether built-in tools are enabled.
    #[must_use]
    pub fn has_builtins(&self) -> bool {
        self.builtins.is_some()
    }

    /// Returns whether builtins are automatically enabled on each prompt.
    ///
    /// When true, [`prompt()`](Self::prompt), [`continue_with()`](Self::continue_with),
    /// and [`conversation()`](Self::conversation) automatically add builtins without
    /// requiring [`use_builtins()`](crate::prompt::PromptBuilder::use_builtins).
    #[must_use]
    pub fn is_auto_builtins(&self) -> bool {
        self.auto_builtins
    }

    /// Continues a conversation from existing messages.
    ///
    /// This is a clearer alternative to `.prompt("").messages(...)` when you want
    /// to continue a conversation without adding a new user message. The provided
    /// messages become the conversation history.
    ///
    /// If [`with_builtins`](ActonAIBuilder::with_builtins) was configured, builtins
    /// are automatically enabled (unless [`manual_builtins`](ActonAIBuilder::manual_builtins)
    /// was used).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let history = vec![
    ///     Message::user("What is Rust?"),
    ///     Message::assistant("Rust is a systems programming language..."),
    ///     Message::user("How does ownership work?"),
    /// ];
    ///
    /// let response = runtime
    ///     .continue_with(history)
    ///     .system("Be concise.")
    ///     .on_token(|t| print!("{t}"))
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn continue_with(&self, messages: impl IntoIterator<Item = Message>) -> PromptBuilder<'_> {
        let mut builder = PromptBuilder::new(self, String::new());
        builder = builder.messages(messages);
        if self.auto_builtins && self.builtins.is_some() {
            builder = builder.use_builtins();
        }
        builder
    }

    /// Starts a managed conversation session.
    ///
    /// This returns a [`ConversationBuilder`] that can be used to configure
    /// and create a [`Conversation`](crate::conversation::Conversation) with
    /// automatic history management.
    ///
    /// Using `Conversation` eliminates the boilerplate of manually tracking
    /// conversation history - messages are automatically added to history
    /// after each exchange.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut conv = runtime.conversation()
    ///     .system("You are a helpful assistant.")
    ///     .build();
    ///
    /// // Each send() automatically manages history
    /// let response = conv.send("What is Rust?").await?;
    /// println!("Assistant: {}", response.text);
    ///
    /// // Conversation remembers context
    /// let response = conv.send("How does ownership work?").await?;
    /// println!("Assistant: {}", response.text);
    /// ```
    #[must_use]
    pub fn conversation(&self) -> ConversationBuilder<'_> {
        ConversationBuilder::new(self)
    }

    /// Shuts down the runtime gracefully.
    ///
    /// This stops all actors and releases resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the shutdown fails.
    pub async fn shutdown(mut self) -> Result<(), ActonAIError> {
        self.is_shutdown = true;
        self.runtime
            .shutdown_all()
            .await
            .map_err(|e| ActonAIError::launch_failed(e.to_string()))
    }
}

/// Configuration for built-in tools.
#[derive(Default, Clone)]
enum BuiltinToolsConfig {
    /// No built-in tools
    #[default]
    None,
    /// All built-in tools
    All,
    /// Specific tools by name
    Select(Vec<String>),
}

/// Configuration for sandbox execution.
#[derive(Default, Clone)]
enum SandboxMode {
    /// No sandbox (default)
    #[default]
    None,
    /// Use Hyperlight sandbox factory
    Hyperlight(SandboxConfig),
    /// Use sandbox pool with pre-warmed instances
    Pool {
        /// Pool size (number of pre-warmed sandboxes)
        pool_size: usize,
        /// Configuration for each sandbox
        config: SandboxConfig,
    },
}

/// Builder for configuring and launching ActonAI.
///
/// # Single Provider Example
///
/// ```rust,ignore
/// let runtime = ActonAI::builder()
///     .app_name("my-chat-app")
///     .ollama("qwen2.5:7b")
///     .launch()
///     .await?;
/// ```
///
/// # Multi-Provider Example
///
/// ```rust,ignore
/// let runtime = ActonAI::builder()
///     .app_name("my-app")
///     .provider_named("claude", ProviderConfig::anthropic("sk-..."))
///     .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
///     .default_provider("local")
///     .launch()
///     .await?;
/// ```
///
/// # Config File Example
///
/// ```rust,ignore
/// let runtime = ActonAI::builder()
///     .app_name("my-app")
///     .from_config()?  // Load from config file
///     .launch()
///     .await?;
/// ```
#[derive(Default)]
pub struct ActonAIBuilder {
    app_name: Option<String>,
    /// Named provider configurations
    providers: HashMap<String, ProviderConfig>,
    /// The name of the default provider
    default_provider_name: Option<String>,
    builtins: BuiltinToolsConfig,
    auto_builtins: bool,
    sandbox_mode: SandboxMode,
}

impl ActonAIBuilder {
    /// Sets the application name for logging and identification.
    ///
    /// This name is used in log files and for identifying the application.
    #[must_use]
    pub fn app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = Some(name.into());
        self
    }

    // =========================================================================
    // Multi-Provider API (new)
    // =========================================================================

    /// Registers a named provider configuration.
    ///
    /// This allows multiple providers to be configured, each with a unique name.
    /// Use [`default_provider`](Self::default_provider) to set which provider is
    /// used when none is specified.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .provider_named("claude", ProviderConfig::anthropic("sk-..."))
    ///     .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
    ///     .default_provider("local")
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn provider_named(mut self, name: impl Into<String>, config: ProviderConfig) -> Self {
        self.providers.insert(name.into(), config);
        self
    }

    /// Sets the name of the default provider.
    ///
    /// The default provider is used when no provider is specified on a prompt.
    /// If not set and only one provider exists, that provider becomes the default.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
    ///     .provider_named("cloud", ProviderConfig::anthropic("sk-..."))
    ///     .default_provider("local")
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn default_provider(mut self, name: impl Into<String>) -> Self {
        self.default_provider_name = Some(name.into());
        self
    }

    /// Loads provider configurations from a config file.
    ///
    /// This searches for configuration in the following order:
    /// 1. `./acton-ai.toml` (project-local)
    /// 2. `~/.config/acton-ai/config.toml` (XDG config)
    ///
    /// If no config file is found, this is a no-op (returns Ok).
    /// Providers loaded from config are merged with any already registered.
    ///
    /// # Errors
    ///
    /// Returns an error if a config file exists but cannot be parsed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .from_config()?
    ///     .launch()
    ///     .await?;
    /// ```
    pub fn from_config(self) -> Result<Self, ActonAIError> {
        let config = config::load()?;
        self.apply_config(config)
    }

    /// Loads provider configurations from a specific file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .from_config_file("/etc/acton-ai/config.toml")?
    ///     .launch()
    ///     .await?;
    /// ```
    pub fn from_config_file(self, path: impl AsRef<Path>) -> Result<Self, ActonAIError> {
        let config = config::from_path(path.as_ref())?;
        self.apply_config(config)
    }

    /// Attempts to load from config file, ignoring errors if no config exists.
    ///
    /// This is useful when config files are optional. Parse errors are still
    /// returned as Err.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .try_from_config()?  // OK if no config file
    ///     .ollama("qwen2.5:7b")  // Fallback provider
    ///     .launch()
    ///     .await?;
    /// ```
    pub fn try_from_config(self) -> Result<Self, ActonAIError> {
        self.from_config()
    }

    /// Applies an ActonAIConfig to this builder.
    fn apply_config(mut self, config: ActonAIConfig) -> Result<Self, ActonAIError> {
        // Convert and add each provider
        for (name, provider_config) in config.providers {
            let runtime_config = provider_config.to_provider_config();
            self.providers.insert(name, runtime_config);
        }

        // Set default provider if specified and we don't have one
        if self.default_provider_name.is_none() {
            self.default_provider_name = config.default_provider;
        }

        Ok(self)
    }

    // =========================================================================
    // Single-Provider API (backwards compatible)
    // =========================================================================

    /// Configures for Ollama with the specified model.
    ///
    /// This registers the provider as "default". For multi-provider setups,
    /// use [`provider_named`](Self::provider_named) instead.
    ///
    /// Ollama runs locally and doesn't require an API key.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("llama3.2")
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn ollama(self, model: impl Into<String>) -> Self {
        self.provider_named(DEFAULT_PROVIDER_NAME, ProviderConfig::ollama(model))
    }

    /// Configures for Ollama with a custom URL and model.
    ///
    /// Use this when Ollama is running on a non-default address.
    /// Registers as "default" provider.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama_at("http://192.168.1.100:11434/v1", "llama3.2")
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn ollama_at(self, base_url: impl Into<String>, model: impl Into<String>) -> Self {
        self.provider_named(
            DEFAULT_PROVIDER_NAME,
            ProviderConfig::openai_compatible(base_url, model),
        )
    }

    /// Configures for Anthropic Claude with the specified API key.
    ///
    /// Uses the default Claude model. Registers as "default" provider.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .anthropic("sk-ant-...")
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn anthropic(self, api_key: impl Into<String>) -> Self {
        self.provider_named(DEFAULT_PROVIDER_NAME, ProviderConfig::anthropic(api_key))
    }

    /// Configures for Anthropic Claude with a specific model.
    ///
    /// Registers as "default" provider.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .anthropic_model("sk-ant-...", "claude-3-haiku-20240307")
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn anthropic_model(self, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        self.provider_named(
            DEFAULT_PROVIDER_NAME,
            ProviderConfig::anthropic(api_key).with_model(model),
        )
    }

    /// Configures for OpenAI with the specified API key.
    ///
    /// Uses the default GPT model. Registers as "default" provider.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .openai("sk-...")
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn openai(self, api_key: impl Into<String>) -> Self {
        self.provider_named(DEFAULT_PROVIDER_NAME, ProviderConfig::openai(api_key))
    }

    /// Configures for OpenAI with a specific model.
    ///
    /// Registers as "default" provider.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .openai_model("sk-...", "gpt-4-turbo")
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn openai_model(self, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        self.provider_named(
            DEFAULT_PROVIDER_NAME,
            ProviderConfig::openai(api_key).with_model(model),
        )
    }

    /// Sets a custom provider configuration.
    ///
    /// Use this for advanced configuration or custom OpenAI-compatible providers.
    /// Registers as "default" provider.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ProviderConfig::openai_compatible("http://localhost:8080/v1", "my-model")
    ///     .with_timeout(Duration::from_secs(60));
    ///
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .provider(config)
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn provider(self, config: ProviderConfig) -> Self {
        self.provider_named(DEFAULT_PROVIDER_NAME, config)
    }

    /// Enables all built-in tools with automatic enabling on each prompt.
    ///
    /// Built-in tools include:
    /// - `read_file`: Read file contents with line numbers
    /// - `write_file`: Write content to files
    /// - `edit_file`: Make targeted string replacements
    /// - `list_directory`: List directory contents
    /// - `glob`: Find files matching glob patterns
    /// - `grep`: Search file contents with regex
    /// - `bash`: Execute shell commands
    /// - `calculate`: Evaluate mathematical expressions
    /// - `web_fetch`: Fetch content from URLs
    ///
    /// When using this method, builtins are automatically enabled on every prompt
    /// created via [`prompt()`](ActonAI::prompt), [`continue_with()`](ActonAI::continue_with),
    /// or [`conversation()`](ActonAI::conversation). You don't need to call
    /// [`use_builtins()`](crate::prompt::PromptBuilder::use_builtins) on each prompt.
    ///
    /// Use [`manual_builtins()`](Self::manual_builtins) after this to opt out of
    /// auto-enabling while still having builtins available.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_builtins()
    ///     .launch()
    ///     .await?;
    ///
    /// // Builtins are automatically available - no need for .use_builtins()
    /// runtime
    ///     .prompt("List files in the current directory")
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_builtins(mut self) -> Self {
        self.builtins = BuiltinToolsConfig::All;
        self.auto_builtins = true;
        self
    }

    /// Disables auto-enabling of builtins on each prompt.
    ///
    /// When called after [`with_builtins()`](Self::with_builtins) or
    /// [`with_builtin_tools()`](Self::with_builtin_tools), this opts out of
    /// automatically adding builtins to each prompt. You'll need to manually
    /// call [`use_builtins()`](crate::prompt::PromptBuilder::use_builtins) on
    /// prompts where you want builtins available.
    ///
    /// This is useful when you only want builtins on specific prompts rather
    /// than all prompts.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_builtins()
    ///     .manual_builtins()  // Opt out of auto-enable
    ///     .launch()
    ///     .await?;
    ///
    /// // Must explicitly enable builtins
    /// runtime
    ///     .prompt("List files")
    ///     .use_builtins()  // Now required
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn manual_builtins(mut self) -> Self {
        self.auto_builtins = false;
        self
    }

    /// Enables specific built-in tools by name with automatic enabling on each prompt.
    ///
    /// See [`with_builtins`](Self::with_builtins) for the list of available tools.
    ///
    /// Like `with_builtins()`, this automatically enables the selected tools on every
    /// prompt. Use [`manual_builtins()`](Self::manual_builtins) to opt out.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_builtin_tools(&["read_file", "write_file", "glob"])
    ///     .launch()
    ///     .await?;
    ///
    /// // Selected tools are automatically available
    /// runtime
    ///     .prompt("Read the README")
    ///     .collect()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_builtin_tools(mut self, tools: &[&str]) -> Self {
        self.builtins =
            BuiltinToolsConfig::Select(tools.iter().map(|s| (*s).to_string()).collect());
        self.auto_builtins = true;
        self
    }

    /// Enables Hyperlight sandbox for hardware-isolated tool execution.
    ///
    /// Uses Microsoft's Hyperlight micro-VM technology for strong isolation
    /// with 1-2ms cold start times. Requires a hypervisor (KVM on Linux,
    /// Hyper-V on Windows).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_hyperlight_sandbox()
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_hyperlight_sandbox(mut self) -> Self {
        self.sandbox_mode = SandboxMode::Hyperlight(SandboxConfig::default());
        self
    }

    /// Enables Hyperlight sandbox with custom configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::tools::sandbox::SandboxConfig;
    /// use std::time::Duration;
    ///
    /// let config = SandboxConfig::new()
    ///     .with_memory_limit(128 * 1024 * 1024)
    ///     .with_timeout(Duration::from_secs(60));
    ///
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_hyperlight_sandbox_config(config)
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_hyperlight_sandbox_config(mut self, config: SandboxConfig) -> Self {
        self.sandbox_mode = SandboxMode::Hyperlight(config);
        self
    }

    /// Enables a pool of pre-warmed Hyperlight sandboxes.
    ///
    /// Pool size determines how many sandboxes are kept ready for immediate use,
    /// reducing latency for tool executions. Recommended for high-throughput
    /// scenarios.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_sandbox_pool(4)  // Keep 4 sandboxes warm
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_sandbox_pool(mut self, pool_size: usize) -> Self {
        self.sandbox_mode = SandboxMode::Pool {
            pool_size,
            config: SandboxConfig::default(),
        };
        self
    }

    /// Enables a pool of pre-warmed Hyperlight sandboxes with custom configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::tools::sandbox::SandboxConfig;
    ///
    /// let config = SandboxConfig::new()
    ///     .with_memory_limit(64 * 1024 * 1024);
    ///
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_sandbox_pool_config(4, config)
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_sandbox_pool_config(mut self, pool_size: usize, config: SandboxConfig) -> Self {
        self.sandbox_mode = SandboxMode::Pool { pool_size, config };
        self
    }

    /// Launches the ActonAI runtime with the configured settings.
    ///
    /// This spawns the actor runtime, kernel, and LLM providers.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No provider is configured
    /// - Default provider is specified but doesn't exist
    /// - Multiple providers exist but no default is specified
    /// - The runtime fails to launch
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("llama3.2")
    ///     .launch()
    ///     .await?;
    /// ```
    pub async fn launch(self) -> Result<ActonAI, ActonAIError> {
        // Validate we have at least one provider
        if self.providers.is_empty() {
            return Err(ActonAIError::new(ActonAIErrorKind::Configuration {
                field: "provider".to_string(),
                reason: "no LLM provider configured; use ollama(), anthropic(), openai(), provider(), provider_named(), or from_config()".to_string(),
            }));
        }

        // Determine the default provider name
        let default_provider_name = self.resolve_default_provider_name()?;

        let app_name = self.app_name.unwrap_or_else(|| "acton-ai".to_string());

        // Launch the actor runtime
        let mut runtime = ActonApp::launch_async().await;

        // Spawn the kernel with the app name for logging
        let kernel_config = KernelConfig::default().with_app_name(&app_name);
        let _kernel = Kernel::spawn_with_config(&mut runtime, kernel_config).await;

        // Spawn all LLM providers
        let mut providers = HashMap::new();
        for (name, config) in self.providers {
            let handle = LLMProvider::spawn(&mut runtime, config).await;
            providers.insert(name, handle);
        }

        // Initialize sandbox if configured
        {
            use crate::tools::sandbox::hyperlight::WarmPool;

            match self.sandbox_mode {
                SandboxMode::None => {}
                SandboxMode::Hyperlight(config) => {
                    // Use fallback factory that gracefully handles missing hypervisor
                    let factory = HyperlightSandboxFactory::with_config_fallback(config);
                    if !factory.is_available() {
                        tracing::warn!(
                            "Hyperlight sandbox configured but hypervisor not available; \
                             sandboxed tools will fail"
                        );
                    }
                    // Store the factory for tool registry configuration
                    // Note: The tool registry will be configured separately
                    let _factory = Arc::new(factory);
                    tracing::info!("Hyperlight sandbox factory initialized");
                }
                SandboxMode::Pool { pool_size, config } => {
                    // Spawn the sandbox pool actor with default pool config
                    let pool_config = crate::tools::sandbox::hyperlight::PoolConfig::default();
                    let pool_handle =
                        SandboxPool::spawn(&mut runtime, config, pool_config).await;

                    // Warm up the pool for all guest types
                    pool_handle
                        .send(WarmPool {
                            count: pool_size,
                            guest_type: None,
                        })
                        .await;

                    tracing::info!(pool_size, "Hyperlight sandbox pool initialized and warmed");
                }
            }
        }

        // Load built-in tools if configured
        let builtins = match self.builtins {
            BuiltinToolsConfig::None => None,
            BuiltinToolsConfig::All => Some(BuiltinTools::all()),
            BuiltinToolsConfig::Select(ref tools) => {
                let tool_refs: Vec<&str> = tools.iter().map(String::as_str).collect();
                Some(BuiltinTools::select(&tool_refs).map_err(|e| {
                    ActonAIError::new(ActonAIErrorKind::Configuration {
                        field: "builtins".to_string(),
                        reason: e.to_string(),
                    })
                })?)
            }
        };

        Ok(ActonAI {
            runtime,
            providers,
            default_provider: default_provider_name,
            builtins,
            auto_builtins: self.auto_builtins,
            is_shutdown: false,
        })
    }

    /// Resolves the default provider name from configuration.
    fn resolve_default_provider_name(&self) -> Result<String, ActonAIError> {
        // If explicitly set, validate it exists
        if let Some(ref name) = self.default_provider_name {
            if self.providers.contains_key(name) {
                return Ok(name.clone());
            }
            return Err(ActonAIError::new(ActonAIErrorKind::Configuration {
                field: "default_provider".to_string(),
                reason: format!(
                    "default provider '{}' not found; available providers: {}",
                    name,
                    self.providers
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }));
        }

        // If only one provider, use it as default
        if self.providers.len() == 1 {
            return Ok(self.providers.keys().next().unwrap().clone());
        }

        // Check if "default" provider exists (from single-provider API)
        if self.providers.contains_key(DEFAULT_PROVIDER_NAME) {
            return Ok(DEFAULT_PROVIDER_NAME.to_string());
        }

        // Multiple providers but no default specified
        Err(ActonAIError::new(ActonAIErrorKind::Configuration {
            field: "default_provider".to_string(),
            reason: format!(
                "multiple providers configured but no default specified; use default_provider() to set one; available: {}",
                self.providers.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        }))
    }

    /// Returns whether auto-builtins is currently enabled.
    ///
    /// This is useful for testing or debugging.
    #[must_use]
    pub fn is_auto_builtins(&self) -> bool {
        self.auto_builtins
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_default_has_no_provider() {
        let builder = ActonAIBuilder::default();
        assert!(builder.providers.is_empty());
        assert!(builder.app_name.is_none());
    }

    #[test]
    fn builder_app_name_sets_name() {
        let builder = ActonAI::builder().app_name("test-app");
        assert_eq!(builder.app_name, Some("test-app".to_string()));
    }

    #[test]
    fn builder_ollama_sets_provider() {
        let builder = ActonAI::builder().ollama("llama3.2");
        assert!(!builder.providers.is_empty());

        let config = builder.providers.get(DEFAULT_PROVIDER_NAME).unwrap();
        assert_eq!(config.model, "llama3.2");
        assert!(config.api_key.is_empty());
    }

    #[test]
    fn builder_ollama_at_sets_custom_url() {
        let builder = ActonAI::builder().ollama_at("http://custom:11434/v1", "llama3.2");
        assert!(!builder.providers.is_empty());

        let config = builder.providers.get(DEFAULT_PROVIDER_NAME).unwrap();
        assert_eq!(config.model, "llama3.2");
        assert_eq!(config.base_url, "http://custom:11434/v1");
    }

    #[test]
    fn builder_anthropic_sets_provider() {
        let builder = ActonAI::builder().anthropic("sk-ant-test");
        assert!(!builder.providers.is_empty());

        let config = builder.providers.get(DEFAULT_PROVIDER_NAME).unwrap();
        assert_eq!(config.api_key, "sk-ant-test");
        assert!(config.model.contains("claude"));
    }

    #[test]
    fn builder_anthropic_model_sets_custom_model() {
        let builder = ActonAI::builder().anthropic_model("sk-ant-test", "claude-3-haiku");
        assert!(!builder.providers.is_empty());

        let config = builder.providers.get(DEFAULT_PROVIDER_NAME).unwrap();
        assert_eq!(config.api_key, "sk-ant-test");
        assert_eq!(config.model, "claude-3-haiku");
    }

    #[test]
    fn builder_openai_sets_provider() {
        let builder = ActonAI::builder().openai("sk-test");
        assert!(!builder.providers.is_empty());

        let config = builder.providers.get(DEFAULT_PROVIDER_NAME).unwrap();
        assert_eq!(config.api_key, "sk-test");
        assert!(config.model.contains("gpt"));
    }

    #[test]
    fn builder_openai_model_sets_custom_model() {
        let builder = ActonAI::builder().openai_model("sk-test", "gpt-4-turbo");
        assert!(!builder.providers.is_empty());

        let config = builder.providers.get(DEFAULT_PROVIDER_NAME).unwrap();
        assert_eq!(config.api_key, "sk-test");
        assert_eq!(config.model, "gpt-4-turbo");
    }

    #[test]
    fn builder_provider_sets_custom_config() {
        let custom_config =
            ProviderConfig::openai_compatible("http://custom:8080/v1", "custom-model");
        let builder = ActonAI::builder().provider(custom_config);
        assert!(!builder.providers.is_empty());

        let config = builder.providers.get(DEFAULT_PROVIDER_NAME).unwrap();
        assert_eq!(config.model, "custom-model");
        assert_eq!(config.base_url, "http://custom:8080/v1");
    }

    #[tokio::test]
    async fn launch_fails_without_provider() {
        let result = ActonAI::builder().app_name("test").launch().await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_configuration());
        assert!(err.to_string().contains("provider"));
    }

    #[test]
    fn with_builtins_enables_auto_builtins() {
        let builder = ActonAI::builder().with_builtins();
        assert!(builder.is_auto_builtins());
    }

    #[test]
    fn with_builtin_tools_enables_auto_builtins() {
        let builder = ActonAI::builder().with_builtin_tools(&["bash", "read_file"]);
        assert!(builder.is_auto_builtins());
    }

    #[test]
    fn manual_builtins_disables_auto_builtins() {
        let builder = ActonAI::builder().with_builtins().manual_builtins();
        assert!(!builder.is_auto_builtins());
    }

    #[test]
    fn default_builder_has_no_auto_builtins() {
        let builder = ActonAI::builder();
        assert!(!builder.is_auto_builtins());
    }

    // Multi-provider tests
    #[test]
    fn builder_provider_named_adds_named_provider() {
        let builder = ActonAI::builder()
            .provider_named("claude", ProviderConfig::anthropic("sk-test"))
            .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"));

        assert_eq!(builder.providers.len(), 2);
        assert!(builder.providers.contains_key("claude"));
        assert!(builder.providers.contains_key("local"));
    }

    #[test]
    fn builder_default_provider_sets_name() {
        let builder = ActonAI::builder()
            .provider_named("claude", ProviderConfig::anthropic("sk-test"))
            .default_provider("claude");

        assert_eq!(builder.default_provider_name, Some("claude".to_string()));
    }

    #[test]
    fn resolve_default_single_provider() {
        let builder = ActonAI::builder().provider_named("only-one", ProviderConfig::ollama("test"));

        let name = builder.resolve_default_provider_name().unwrap();
        assert_eq!(name, "only-one");
    }

    #[test]
    fn resolve_default_explicit() {
        let builder = ActonAI::builder()
            .provider_named("a", ProviderConfig::ollama("test-a"))
            .provider_named("b", ProviderConfig::ollama("test-b"))
            .default_provider("b");

        let name = builder.resolve_default_provider_name().unwrap();
        assert_eq!(name, "b");
    }

    #[test]
    fn resolve_default_uses_default_name() {
        let builder = ActonAI::builder()
            .ollama("test") // Registers as "default"
            .provider_named("other", ProviderConfig::anthropic("sk-test"));

        let name = builder.resolve_default_provider_name().unwrap();
        assert_eq!(name, DEFAULT_PROVIDER_NAME);
    }

    #[test]
    fn resolve_default_fails_multiple_no_explicit() {
        let builder = ActonAI::builder()
            .provider_named("a", ProviderConfig::ollama("test-a"))
            .provider_named("b", ProviderConfig::ollama("test-b"));

        let result = builder.resolve_default_provider_name();
        assert!(result.is_err());
    }

    #[test]
    fn resolve_default_fails_invalid_name() {
        let builder = ActonAI::builder()
            .provider_named("actual", ProviderConfig::ollama("test"))
            .default_provider("nonexistent");

        let result = builder.resolve_default_provider_name();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("nonexistent"));
    }
}
