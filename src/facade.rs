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

use crate::config::{self, ActonAIConfig, SandboxFileConfig};
use crate::conversation::ConversationBuilder;
use crate::error::{ActonAIError, ActonAIErrorKind};
use crate::kernel::{Kernel, KernelConfig};
use crate::llm::{LLMProvider, ProviderConfig};
use crate::messages::Message;
use crate::prompt::PromptBuilder;
use crate::tools::builtins::BuiltinTools;
use crate::tools::sandbox::{ProcessSandboxConfig, ProcessSandboxFactory, SandboxFactory};
use acton_reactive::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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
/// Internal state shared via `Arc`.
pub(crate) struct ActonAIInner {
    /// The underlying actor runtime
    pub(crate) runtime: ActorRuntime,
    /// Named LLM provider handles
    pub(crate) providers: HashMap<String, ActorHandle>,
    /// The name of the default provider
    pub(crate) default_provider: String,
    /// Built-in tools (if enabled)
    pub(crate) builtins: Option<BuiltinTools>,
    /// Whether to automatically enable builtins on each prompt
    pub(crate) auto_builtins: bool,
    /// Loaded skill registry shared across every prompt (if configured).
    ///
    /// When present, [`ActonAI::prompt`] and [`ActonAI::continue_with`]
    /// auto-register `list_skills` and `activate_skill` tools backed by this
    /// registry so skills are available without per-call wiring.
    pub(crate) skills: Option<Arc<crate::skills::SkillRegistry>>,
    /// Shared sandbox factory used to wrap sandboxed builtin tool calls.
    ///
    /// `None` when no sandbox is configured; in that case sandboxed tools
    /// still execute in-process (matching pre-ProcessSandbox behavior) but
    /// with no OS-level isolation.
    pub(crate) sandbox_factory: Option<Arc<dyn SandboxFactory>>,
    /// Default `max_tool_rounds` seeded into every new `PromptBuilder`.
    ///
    /// Resolved at launch from the cascade:
    /// `DEFAULT_MAX_TOOL_ROUNDS → [defaults] TOML → builder override`.
    pub(crate) default_max_tool_rounds: usize,
    /// Default context window applied to every [`Conversation`](crate::conversation::Conversation).
    ///
    /// Resolved at launch from the cascade:
    /// per-provider `context_window_tokens` → `[context] max_tokens` → 8192.
    /// `None` means unbounded history (explicit opt-out via
    /// [`ActonAIBuilder::without_context_window`]).
    pub(crate) context_window: Option<crate::memory::ContextWindow>,
    /// Whether the runtime has been shut down
    pub(crate) is_shutdown: AtomicBool,
}

pub struct ActonAI {
    pub(crate) inner: Arc<ActonAIInner>,
}

impl Clone for ActonAI {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for ActonAI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActonAI")
            .field(
                "is_shutdown",
                &self.inner.is_shutdown.load(Ordering::SeqCst),
            )
            .field("has_builtins", &self.inner.builtins.is_some())
            .field("auto_builtins", &self.inner.auto_builtins)
            .field("provider_count", &self.inner.providers.len())
            .field("default_provider", &self.inner.default_provider)
            .field(
                "default_max_tool_rounds",
                &self.inner.default_max_tool_rounds,
            )
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
    pub fn prompt(&self, content: impl Into<String>) -> PromptBuilder {
        let mut builder = PromptBuilder::new(self.clone(), content.into());
        if self.inner.auto_builtins && self.inner.builtins.is_some() {
            builder = builder.use_builtins();
        }
        builder = self.inject_skill_tools(builder);
        builder
    }

    /// Returns a reference to the underlying actor runtime.
    ///
    /// This provides an escape hatch for advanced use cases that need
    /// direct access to the actor system.
    #[must_use]
    pub fn runtime(&self) -> &ActorRuntime {
        &self.inner.runtime
    }

    /// Returns a mutable reference to the underlying actor runtime.
    ///
    /// This provides an escape hatch for advanced use cases that need
    /// direct access to the actor system.
    ///
    /// # Panics
    ///
    /// Panics if there are other clones of this `ActonAI` handle. This
    /// matches the previous `&mut self` semantics — you can only call it
    /// if you have exclusive access.
    pub fn runtime_mut(&mut self) -> &mut ActorRuntime {
        &mut Arc::get_mut(&mut self.inner)
            .expect("cannot get mutable runtime: ActonAI is shared")
            .runtime
    }

    /// Returns a clone of the default LLM provider handle.
    ///
    /// This can be used to send requests directly to the provider
    /// for advanced use cases.
    #[must_use]
    pub fn provider_handle(&self) -> ActorHandle {
        self.inner
            .providers
            .get(&self.inner.default_provider)
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
        self.inner.providers.get(name).cloned()
    }

    /// Returns the name of the default provider.
    #[must_use]
    pub fn default_provider_name(&self) -> &str {
        &self.inner.default_provider
    }

    /// Returns the default `max_tool_rounds` seeded into every new prompt.
    ///
    /// Resolved at launch from the cascade
    /// [`DEFAULT_MAX_TOOL_ROUNDS`](crate::prompt::DEFAULT_MAX_TOOL_ROUNDS)
    /// → `[defaults]` TOML → [`ActonAIBuilder::max_tool_rounds`]. Per-prompt
    /// [`PromptBuilder::max_tool_rounds`](crate::prompt::PromptBuilder::max_tool_rounds)
    /// calls still override this value.
    #[must_use]
    pub fn default_max_tool_rounds(&self) -> usize {
        self.inner.default_max_tool_rounds
    }

    /// Returns an iterator over the names of all registered providers.
    pub fn provider_names(&self) -> impl Iterator<Item = &str> {
        self.inner.providers.keys().map(String::as_str)
    }

    /// Returns the number of registered providers.
    #[must_use]
    pub fn provider_count(&self) -> usize {
        self.inner.providers.len()
    }

    /// Returns true if a provider with the given name exists.
    #[must_use]
    pub fn has_provider(&self, name: &str) -> bool {
        self.inner.providers.contains_key(name)
    }

    /// Returns whether the runtime has been shut down.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.inner.is_shutdown.load(Ordering::SeqCst)
    }

    /// Returns a reference to the built-in tools, if enabled.
    ///
    /// Returns `None` if built-in tools were not configured with
    /// [`with_builtins`](ActonAIBuilder::with_builtins) or
    /// [`with_builtin_tools`](ActonAIBuilder::with_builtin_tools).
    #[must_use]
    pub fn builtins(&self) -> Option<&BuiltinTools> {
        self.inner.builtins.as_ref()
    }

    /// Returns a reference to the loaded skill registry, if configured.
    ///
    /// Populated when the builder was given
    /// [`with_skill_paths`](ActonAIBuilder::with_skill_paths) (or picked up
    /// `[skills] paths = [...]` from a TOML config). Returns `None`
    /// otherwise.
    #[must_use]
    pub fn skills(&self) -> Option<&Arc<crate::skills::SkillRegistry>> {
        self.inner.skills.as_ref()
    }

    /// Returns the default context window applied to every
    /// [`Conversation`](crate::conversation::Conversation) built from this
    /// runtime. `None` indicates the runtime was launched with
    /// [`without_context_window`](ActonAIBuilder::without_context_window).
    #[must_use]
    pub fn context_window(&self) -> Option<&crate::memory::ContextWindow> {
        self.inner.context_window.as_ref()
    }

    /// Returns the sandbox factory shared by sandboxed builtin tool calls.
    ///
    /// Populated when the builder was configured with
    /// [`with_process_sandbox`](ActonAIBuilder::with_process_sandbox) or
    /// [`with_process_sandbox_config`](ActonAIBuilder::with_process_sandbox_config)
    /// (or an equivalent TOML `[sandbox]` section). Returns `None` otherwise.
    #[must_use]
    pub(crate) fn sandbox_factory(&self) -> Option<&Arc<dyn SandboxFactory>> {
        self.inner.sandbox_factory.as_ref()
    }

    /// Returns whether built-in tools are enabled.
    #[must_use]
    pub fn has_builtins(&self) -> bool {
        self.inner.builtins.is_some()
    }

    /// Returns whether builtins are automatically enabled on each prompt.
    ///
    /// When true, [`prompt()`](Self::prompt), [`continue_with()`](Self::continue_with),
    /// and [`conversation()`](Self::conversation) automatically add builtins without
    /// requiring [`use_builtins()`](crate::prompt::PromptBuilder::use_builtins).
    #[must_use]
    pub fn is_auto_builtins(&self) -> bool {
        self.inner.auto_builtins
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
    pub fn continue_with(&self, messages: impl IntoIterator<Item = Message>) -> PromptBuilder {
        let mut builder = PromptBuilder::new(self.clone(), String::new());
        builder = builder.messages(messages);
        if self.inner.auto_builtins && self.inner.builtins.is_some() {
            builder = builder.use_builtins();
        }
        builder = self.inject_skill_tools(builder);
        builder
    }

    /// Registers the `list_skills` / `activate_skill` tools on `builder` when
    /// a skill registry is configured. No-op otherwise.
    ///
    /// Called from both [`prompt`](Self::prompt) and
    /// [`continue_with`](Self::continue_with) so every `PromptBuilder` the
    /// facade hands out — including the ones [`Conversation`] rebuilds per
    /// turn — has the skill tools available without per-call wiring.
    #[inline]
    fn inject_skill_tools(&self, builder: PromptBuilder) -> PromptBuilder {
        if let Some(registry) = &self.inner.skills {
            use crate::tools::builtins::{ActivateSkillTool, ListSkillsTool};
            use crate::tools::ToolExecutorTrait;
            let list_tool = ListSkillsTool::new(Arc::clone(registry));
            let activate_tool = ActivateSkillTool::new(Arc::clone(registry));
            return builder
                .with_tool(ListSkillsTool::config().definition, move |args| {
                    list_tool.execute(args)
                })
                .with_tool(ActivateSkillTool::config().definition, move |args| {
                    activate_tool.execute(args)
                });
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
    /// let conv = runtime.conversation()
    ///     .system("You are a helpful assistant.")
    ///     .build()
    ///     .await;
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
    pub fn conversation(&self) -> ConversationBuilder {
        ConversationBuilder::new(self.clone())
    }

    /// Shuts down the runtime gracefully.
    ///
    /// This stops all actors and releases resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the shutdown fails.
    pub async fn shutdown(self) -> Result<(), ActonAIError> {
        self.inner.is_shutdown.store(true, Ordering::SeqCst);
        // Get the runtime clone for shutdown. The Arc may still be shared,
        // so we clone the ActorRuntime (which is itself cheap to clone).
        let mut runtime = self.inner.runtime.clone();
        runtime
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
    /// No sandbox (default). Sandboxed tools execute in-process.
    #[default]
    None,
    /// Use the portable [`ProcessSandbox`](crate::tools::sandbox::ProcessSandbox)
    /// with the supplied configuration.
    Process(ProcessSandboxConfig),
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
    /// Skill paths staged by [`ActonAIBuilder::with_skill_paths`] /
    /// [`with_skill_path`]. Loaded once in [`launch`](Self::launch) into a
    /// shared [`SkillRegistry`](crate::skills::SkillRegistry).
    skill_paths: Vec<PathBuf>,
    /// Framework-wide default for the agentic tool-call loop cap.
    ///
    /// `None` means "use whatever [`apply_config`](Self::apply_config) finds
    /// in the TOML, else fall back to
    /// [`DEFAULT_MAX_TOOL_ROUNDS`](crate::prompt::DEFAULT_MAX_TOOL_ROUNDS)".
    default_max_tool_rounds: Option<usize>,
    /// Context-window settings loaded from `[context]` in the TOML via
    /// [`apply_config`](Self::apply_config).
    context_config: Option<crate::config::ContextFileConfig>,
    /// Explicit [`ContextWindow`](crate::memory::ContextWindow) override set
    /// via [`context_window`](Self::context_window). Wins over config cascade
    /// at launch.
    context_window_override: Option<crate::memory::ContextWindow>,
    /// Set by [`without_context_window`](Self::without_context_window) to
    /// explicitly disable per-turn truncation for [`Conversation`].
    context_window_disabled: bool,
    /// Per-provider native context window, captured from `[providers.<name>].context_window_tokens`
    /// during [`apply_config`](Self::apply_config). At launch the entry for the
    /// resolved default provider wins over the global `[context] max_tokens`.
    context_window_per_provider: HashMap<String, usize>,
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

    /// Sets the framework-wide default cap on agentic tool-call rounds.
    ///
    /// Takes precedence over a `[defaults] max_tool_rounds` value loaded from
    /// a config file. Per-prompt
    /// [`PromptBuilder::max_tool_rounds`](crate::prompt::PromptBuilder::max_tool_rounds)
    /// calls override this on a per-request basis.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .max_tool_rounds(25)  // raise cap for this application
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn max_tool_rounds(mut self, max: usize) -> Self {
        self.default_max_tool_rounds = Some(max);
        self
    }

    /// Installs an explicit [`ContextWindow`](crate::memory::ContextWindow).
    ///
    /// Wins over the per-provider and config-file cascade at
    /// [`launch`](Self::launch). Useful when the caller has a pre-built
    /// window with a custom [`TokenEstimator`](crate::memory::TokenEstimator).
    #[must_use]
    pub fn context_window(mut self, window: crate::memory::ContextWindow) -> Self {
        self.context_window_override = Some(window);
        self.context_window_disabled = false;
        self
    }

    /// Disables automatic per-turn history truncation in
    /// [`Conversation`](crate::conversation::Conversation).
    ///
    /// Every turn will ship the full accumulated history — the pre-wiring
    /// behavior. Use for workloads where you've prepared the history yourself
    /// or are running against a provider with unbounded context.
    #[must_use]
    pub fn without_context_window(mut self) -> Self {
        self.context_window_disabled = true;
        self.context_window_override = None;
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

    /// Applies an [`ActonAIConfig`] to this builder.
    ///
    /// This is useful when you've already loaded the configuration and want
    /// to apply it to the builder without going through file loading again.
    pub fn apply_config(mut self, config: ActonAIConfig) -> Result<Self, ActonAIError> {
        // Convert and add each provider
        for (name, provider_config) in config.providers {
            if let Some(tokens) = provider_config.context_window_tokens {
                self.context_window_per_provider.insert(name.clone(), tokens);
            }
            let runtime_config = provider_config.to_provider_config();
            self.providers.insert(name, runtime_config);
        }

        // Set default provider if specified and we don't have one
        if self.default_provider_name.is_none() {
            self.default_provider_name = config.default_provider;
        }

        // Apply sandbox configuration if present and no programmatic sandbox was set
        if let Some(sandbox_config) = config.sandbox {
            self = self.apply_sandbox_file_config(&sandbox_config);
        }

        // Apply [defaults] block — only if the builder hasn't been given an
        // explicit override already. Builder > config, constant is the floor.
        if self.default_max_tool_rounds.is_none() {
            if let Some(defaults) = config.defaults {
                self.default_max_tool_rounds = defaults.max_tool_rounds;
            }
        }

        // Merge `[skills] paths` onto whatever was staged programmatically so
        // CLI flags and config union (matching how providers are merged
        // above). Builder-stage paths come first; config-stage paths append.
        if let Some(skills_cfg) = config.skills {
            self.skill_paths.extend(skills_cfg.paths);
        }

        // Stash `[context]` settings — resolved at launch after the default
        // provider is known so per-provider overrides can win.
        if let Some(context_cfg) = config.context {
            self.context_config = Some(context_cfg);
        }

        Ok(self)
    }

    /// Applies sandbox configuration from file to this builder.
    ///
    /// Only applies if no programmatic sandbox mode is already configured.
    fn apply_sandbox_file_config(mut self, config: &SandboxFileConfig) -> Self {
        // Only apply file config if no sandbox mode has been explicitly set
        if matches!(self.sandbox_mode, SandboxMode::None) {
            self.sandbox_mode = SandboxMode::Process(config.to_process_config());
        }
        self
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

    /// Stages skill paths to be loaded into a shared
    /// [`SkillRegistry`](crate::skills::SkillRegistry) at launch.
    ///
    /// Each path may be a directory (scanned recursively for `.md` files) or
    /// a single skill file. Calling this replaces any previously-staged
    /// paths; use [`with_skill_path`](Self::with_skill_path) to append one at
    /// a time.
    ///
    /// Once loaded, the registry is shared across every prompt produced by
    /// [`prompt()`](ActonAI::prompt), [`continue_with()`](ActonAI::continue_with),
    /// and [`conversation()`](ActonAI::conversation): each call auto-registers
    /// `list_skills` and `activate_skill` tools without per-call wiring.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::path::PathBuf;
    ///
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_skill_paths(&[PathBuf::from("./skills")])
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_skill_paths(mut self, paths: &[PathBuf]) -> Self {
        self.skill_paths = paths.to_vec();
        self
    }

    /// Appends a single skill path to the set staged by
    /// [`with_skill_paths`](Self::with_skill_paths).
    #[must_use]
    pub fn with_skill_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.skill_paths.push(path.into());
        self
    }

    /// Enables the portable [`ProcessSandbox`](crate::tools::sandbox::ProcessSandbox)
    /// for sandboxed tool execution.
    ///
    /// Each sandboxed tool call re-execs the current binary as a child
    /// process, applies rlimits, and on Linux kernels supporting landlock +
    /// seccomp (5.13+) installs best-effort filesystem and syscall filters
    /// before running the tool. The parent enforces a wall-clock timeout
    /// and kills the child's process group on overrun.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_process_sandbox()
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_process_sandbox(mut self) -> Self {
        self.sandbox_mode = SandboxMode::Process(ProcessSandboxConfig::default());
        self
    }

    /// Enables the [`ProcessSandbox`](crate::tools::sandbox::ProcessSandbox)
    /// with a custom configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::tools::sandbox::{HardeningMode, ProcessSandboxConfig};
    /// use std::time::Duration;
    ///
    /// let cfg = ProcessSandboxConfig::new()
    ///     .with_timeout(Duration::from_secs(60))
    ///     .with_hardening(HardeningMode::Enforce);
    ///
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_process_sandbox_config(cfg)
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_process_sandbox_config(mut self, config: ProcessSandboxConfig) -> Self {
        self.sandbox_mode = SandboxMode::Process(config);
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
    pub async fn launch(mut self) -> Result<ActonAI, ActonAIError> {
        // Validate we have at least one provider
        if self.providers.is_empty() {
            return Err(ActonAIError::new(ActonAIErrorKind::Configuration {
                field: "provider".to_string(),
                reason: "no LLM provider configured; use ollama(), anthropic(), openai(), provider(), provider_named(), or from_config()".to_string(),
            }));
        }

        // Determine the default provider name + capture its model for later
        // context-window resolution (the HashMap gets consumed in the spawn
        // loop below).
        let default_provider_name = self.resolve_default_provider_name()?;
        let default_provider_model = self
            .providers
            .get(&default_provider_name)
            .map(|p| p.model.clone())
            .unwrap_or_default();

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

        // Initialize sandbox if configured.
        //
        // Ownership note: the produced factory is retained on
        // `ActonAIInner::sandbox_factory` so `.use_builtins()` can wrap
        // sandboxed tool executors at prompt construction time. Prior to
        // this refactor the factory was constructed and dropped, which
        // silently bypassed sandboxing for all facade callers.
        let sandbox_factory: Option<Arc<dyn SandboxFactory>> = match self.sandbox_mode {
            SandboxMode::None => None,
            SandboxMode::Process(cfg) => {
                let factory = ProcessSandboxFactory::new(cfg).map_err(|err| {
                    ActonAIError::new(ActonAIErrorKind::Configuration {
                        field: "sandbox".to_string(),
                        reason: format!("failed to initialize process sandbox: {err}"),
                    })
                })?;
                tracing::info!(
                    exe = %factory.exe().display(),
                    "process sandbox factory initialized"
                );
                Some(Arc::new(factory) as Arc<dyn SandboxFactory>)
            }
        };

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

        let default_max_tool_rounds = self
            .default_max_tool_rounds
            .unwrap_or(crate::prompt::DEFAULT_MAX_TOOL_ROUNDS);

        // Load the skill registry if any skill paths were staged. The paths
        // were pre-validated at the CLI boundary in most cases, but a stray
        // missing path still surfaces cleanly here as a Configuration error.
        let skills = if self.skill_paths.is_empty() {
            None
        } else {
            let path_refs: Vec<&Path> = self.skill_paths.iter().map(PathBuf::as_path).collect();
            let registry = crate::skills::SkillRegistry::from_paths(&path_refs)
                .await
                .map_err(|e| {
                    ActonAIError::new(ActonAIErrorKind::Configuration {
                        field: "skills".to_string(),
                        reason: e.to_string(),
                    })
                })?;
            tracing::info!(count = registry.len(), "skill registry loaded");
            Some(Arc::new(registry))
        };

        let context_window = resolve_context_window(
            self.context_window_override.take(),
            self.context_window_disabled,
            self.context_config.as_ref(),
            self.context_window_per_provider.get(&default_provider_name).copied(),
            &default_provider_model,
        );

        Ok(ActonAI {
            inner: Arc::new(ActonAIInner {
                runtime,
                providers,
                default_provider: default_provider_name,
                builtins,
                auto_builtins: self.auto_builtins,
                skills,
                sandbox_factory,
                default_max_tool_rounds,
                context_window,
                is_shutdown: AtomicBool::new(false),
            }),
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

/// Resolves the runtime [`ContextWindow`] at [`ActonAIBuilder::launch`] time.
///
/// Precedence (highest to lowest):
/// 1. `builder.context_window_override` (explicit API call).
/// 2. `builder.context_window_disabled` (explicit opt-out) → `None`.
/// 3. Per-provider `context_window_tokens` for the default provider.
/// 4. `[context] max_tokens` from TOML.
/// 5. [`ContextWindowConfig::default`] (8192).
///
/// The estimator is selected from the default provider's model name via
/// [`TiktokenEstimator::for_model`], which falls back to `cl100k_base` for
/// unknown models.
fn resolve_context_window(
    override_window: Option<crate::memory::ContextWindow>,
    disabled: bool,
    context_config: Option<&crate::config::ContextFileConfig>,
    per_provider_tokens: Option<usize>,
    default_provider_model: &str,
) -> Option<crate::memory::ContextWindow> {
    use crate::memory::{
        ContextWindow, ContextWindowConfig, TiktokenEstimator, TokenEstimator,
        TruncationStrategy,
    };

    if let Some(window) = override_window {
        return Some(window);
    }
    if disabled {
        return None;
    }

    let default_cfg = ContextWindowConfig::default();

    let max_tokens = per_provider_tokens
        .or_else(|| context_config.and_then(|c| c.max_tokens))
        .unwrap_or(default_cfg.max_tokens);

    let reserved_for_response = context_config
        .and_then(|c| c.reserved_for_response)
        .unwrap_or(default_cfg.reserved_for_response);

    let strategy = context_config
        .and_then(|c| c.strategy.as_deref())
        .and_then(crate::config::parse_truncation_strategy)
        .unwrap_or(TruncationStrategy::KeepRecent);

    let config = ContextWindowConfig {
        max_tokens,
        truncation_strategy: strategy,
        reserved_for_response,
        tokens_per_char: default_cfg.tokens_per_char,
    };

    let estimator: Arc<dyn TokenEstimator> =
        Arc::new(TiktokenEstimator::for_model(default_provider_model));

    let window = ContextWindow::new(config).with_estimator(estimator);
    tracing::info!(
        max_tokens = window.config().max_tokens,
        reserved_for_response = window.config().reserved_for_response,
        strategy = ?window.config().truncation_strategy,
        estimator = window.estimator_name(),
        "context window resolved",
    );
    Some(window)
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

    // max_tool_rounds cascade tests:
    // constant (DEFAULT_MAX_TOOL_ROUNDS) → [defaults] TOML → builder override
    // → per-prompt PromptBuilder.max_tool_rounds() (not exercised here — that
    // path is a trivial setter covered in prompt.rs).

    #[tokio::test]
    async fn default_max_tool_rounds_falls_back_to_constant() {
        let runtime = ActonAI::builder()
            .ollama("test")
            .launch()
            .await
            .expect("launch");

        assert_eq!(
            runtime.default_max_tool_rounds(),
            crate::prompt::DEFAULT_MAX_TOOL_ROUNDS
        );

        let prompt = runtime.prompt("hi");
        assert_eq!(
            prompt.current_max_tool_rounds(),
            crate::prompt::DEFAULT_MAX_TOOL_ROUNDS
        );
    }

    #[tokio::test]
    async fn builder_max_tool_rounds_is_applied_to_runtime_and_prompts() {
        let runtime = ActonAI::builder()
            .ollama("test")
            .max_tool_rounds(42)
            .launch()
            .await
            .expect("launch");

        assert_eq!(runtime.default_max_tool_rounds(), 42);
        assert_eq!(runtime.prompt("hi").current_max_tool_rounds(), 42);
    }

    #[tokio::test]
    async fn toml_defaults_max_tool_rounds_is_applied() {
        let config = crate::config::ActonAIConfig::new()
            .with_provider(
                "ollama",
                crate::config::NamedProviderConfig::ollama("test"),
            )
            .with_default_provider("ollama");
        // Inject the [defaults] block manually.
        let config = crate::config::ActonAIConfig {
            defaults: Some(crate::config::ActonAIDefaults::new().with_max_tool_rounds(33)),
            ..config
        };

        let runtime = ActonAI::builder()
            .apply_config(config)
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        assert_eq!(runtime.default_max_tool_rounds(), 33);
    }

    #[tokio::test]
    async fn builder_max_tool_rounds_overrides_toml_defaults() {
        // Builder wins: user explicitly set 7 in code, config says 33.
        let config = crate::config::ActonAIConfig::new()
            .with_provider(
                "ollama",
                crate::config::NamedProviderConfig::ollama("test"),
            )
            .with_default_provider("ollama");
        let config = crate::config::ActonAIConfig {
            defaults: Some(crate::config::ActonAIDefaults::new().with_max_tool_rounds(33)),
            ..config
        };

        let runtime = ActonAI::builder()
            .max_tool_rounds(7)
            .apply_config(config)
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        assert_eq!(runtime.default_max_tool_rounds(), 7);
    }

    #[tokio::test]
    async fn per_prompt_max_tool_rounds_still_overrides_runtime_default() {
        let runtime = ActonAI::builder()
            .ollama("test")
            .max_tool_rounds(25)
            .launch()
            .await
            .expect("launch");

        let prompt = runtime.prompt("hi").max_tool_rounds(3);
        assert_eq!(prompt.current_max_tool_rounds(), 3);
        // Runtime default unchanged for subsequent prompts.
        assert_eq!(runtime.prompt("other").current_max_tool_rounds(), 25);
    }

    #[test]
    fn builder_with_skill_paths_stores_paths() {
        // We can't observe the private field directly, but launch() is the
        // observable boundary and is tested below. Here we just confirm the
        // builder methods chain and don't drop what they're given by calling
        // `with_skill_path` after `with_skill_paths`.
        let builder = ActonAI::builder()
            .ollama("test")
            .with_skill_paths(&[PathBuf::from("./a"), PathBuf::from("./b")])
            .with_skill_path("./c");
        assert_eq!(
            builder.skill_paths,
            vec![
                PathBuf::from("./a"),
                PathBuf::from("./b"),
                PathBuf::from("./c"),
            ]
        );
    }

    #[tokio::test]
    async fn launch_with_skills_exposes_registry() {
        // Stage a temp dir containing one minimal skill file, launch with it,
        // and assert the registry is exposed and holds the skill.
        let dir = tempfile::tempdir().expect("tempdir");
        let skill_path = dir.path().join("sample.md");
        std::fs::write(
            &skill_path,
            "---\nname: sample\ndescription: A sample skill\n---\n\n# Sample\n\nContent.\n",
        )
        .expect("write skill file");

        let runtime = ActonAI::builder()
            .ollama("test")
            .with_skill_path(dir.path().to_path_buf())
            .launch()
            .await
            .expect("launch");

        let registry = runtime.skills().expect("skill registry present");
        assert_eq!(registry.len(), 1);
        let names: Vec<String> = registry.list().iter().map(|s| s.name.clone()).collect();
        assert!(names.contains(&"sample".to_string()), "names = {names:?}");
    }

    #[tokio::test]
    async fn launch_without_skills_has_no_registry() {
        let runtime = ActonAI::builder()
            .ollama("test")
            .launch()
            .await
            .expect("launch");
        assert!(runtime.skills().is_none());
    }

    #[tokio::test]
    async fn apply_config_merges_skills_section() {
        let dir = tempfile::tempdir().expect("tempdir");
        let toml_path = dir.path().join("toml-skill.md");
        std::fs::write(
            &toml_path,
            "---\nname: from-toml\ndescription: From the [skills] section\n---\n\n# TOML\n",
        )
        .expect("write skill");

        let config = crate::config::ActonAIConfig::new()
            .with_provider(
                "ollama",
                crate::config::NamedProviderConfig::ollama("test"),
            )
            .with_default_provider("ollama");
        let config = crate::config::ActonAIConfig {
            skills: Some(crate::config::SkillsFileConfig {
                paths: vec![toml_path.clone()],
            }),
            ..config
        };

        let runtime = ActonAI::builder()
            .apply_config(config)
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        let registry = runtime.skills().expect("registry");
        assert_eq!(registry.len(), 1);
    }
}
