//! High-level facade for ActonAI.
//!
//! This module provides a simplified API for common use cases, hiding the
//! complexity of actor setup, logging, and provider configuration.
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

use crate::accounting::{Budget, BudgetConfig, BudgetEvent, GetUsage, PricingTable, UsageSnapshot};
use crate::config::{self, ActonAIConfig, SandboxFileConfig};
use crate::conversation::ConversationBuilder;
use crate::error::{ActonAIError, ActonAIErrorKind};
use crate::llm::{FailoverEvent, LLMProvider, ProviderConfig};
use crate::logging::{init_and_store_logging, LoggingConfig};
use crate::messages::{Message, ToolDefinition};
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
/// `ActonAI` encapsulates the runtime and LLM providers, providing
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
    /// Model served by each configured provider, keyed by provider name.
    ///
    /// Read by the prompt loop to label round spans and latency metrics.
    pub(crate) provider_models: HashMap<String, String>,
    /// Failover chain configured for each provider, keyed by provider name.
    ///
    /// Only providers with a non-empty `failover` list appear. Absence is
    /// what lets the prompt loop skip health asks entirely: a runtime that
    /// configured no chains performs no extra round trips per round.
    pub(crate) provider_failover: HashMap<String, Vec<String>>,
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
    /// Handle to the cost accountant, when usage tracking is enabled.
    ///
    /// `None` means tracking was switched off, which
    /// [`ActonAI::usage`] reports as a configuration error rather than as an
    /// empty snapshot — zeros would read as "spent nothing".
    pub(crate) accountant: Option<ActorHandle>,
    /// The caps in force, when a budget was configured.
    ///
    /// Kept beside the accountant handle so the prompt loop can skip the
    /// pre-flight ask entirely — not merely ignore its answer — when nothing
    /// is capped.
    pub(crate) budget: Option<BudgetConfig>,
    /// Tools contributed by configured MCP servers, and the supervised actors
    /// owning their connections.
    ///
    /// `None` when no `[mcp_servers.*]` entry and no
    /// [`with_mcp_server`](ActonAIBuilder::with_mcp_server) call was made.
    /// When present, these tools are injected into every prompt: configuring
    /// a server *is* the request to use it, so there is no separate opt-in.
    pub(crate) mcp: Option<crate::mcp::McpTools>,
    /// Custom tools registered once on the builder and injected into every
    /// prompt and conversation turn.
    ///
    /// Staged with [`ActonAIBuilder::with_tool`],
    /// [`with_tool_executor`](ActonAIBuilder::with_tool_executor), or
    /// [`add_tool`](ActonAIBuilder::add_tool); empty when none were. Names
    /// were checked against the built-ins, skill tools, MCP tools, and each
    /// other at launch, so injection can never silently shadow anything.
    pub(crate) custom_tools: Vec<crate::prompt::SharedToolSpec>,
    /// Installed telemetry, when it was configured.
    ///
    /// `None` means no `[telemetry]` section and no builder call, which is
    /// the state in which nothing here costs anything.
    pub(crate) telemetry: Option<TelemetryRuntime>,
    /// Whether new turns are admitted.
    ///
    /// Always present, and always `Running` at launch. Unlike the surfaces
    /// that drive it, admission itself costs nothing when unused: the prompt
    /// loop reads one relaxed atomic per turn.
    pub(crate) admission: crate::introspection::AdmissionGate,
    /// The control socket, when introspection was configured.
    ///
    /// `None` means nothing is listening, which is the default. Compiling the
    /// `ipc` feature in does not create a socket; only `[introspection]` or
    /// [`ActonAIBuilder::introspection`] does.
    pub(crate) introspection: Option<IntrospectionRuntime>,
    /// The rules applied to every tool call, when a policy was configured.
    ///
    /// `None` is what lets the prompt loop skip the gate entirely rather than
    /// consult an empty policy on every call — the same shape `budget` uses.
    pub(crate) tool_policy: Option<crate::policy::ToolPolicy>,
    /// Handle to the audit log, when a trail was configured.
    pub(crate) audit: Option<ActorHandle>,
    /// Redaction and path settings for the trail, kept beside the handle so
    /// the loop can redact without asking the actor anything.
    pub(crate) audit_config: Option<crate::audit::AuditConfig>,
    /// Whether the runtime has been shut down
    pub(crate) is_shutdown: AtomicBool,
}

/// The introspection socket a launched runtime owns.
///
/// Holding one means a listener is accepting connections, so the socket file
/// exists and must be taken down before the runtime is.
pub(crate) struct IntrospectionRuntime {
    /// Where the socket is bound. Kept so [`ActonAI::introspection_socket`]
    /// can tell a caller the address the process actually resolved, which is
    /// otherwise unknowable when the default PID-suffixed scheme was used.
    pub(crate) socket_path: std::path::PathBuf,
    /// Behind a `Mutex` because shutdown runs through a shared `Arc`, and
    /// stopping the listener needs ownership of the handle.
    #[cfg(feature = "ipc")]
    listener: std::sync::Mutex<Option<acton_reactive::ipc::IpcListenerHandle>>,
}

impl IntrospectionRuntime {
    /// Stops accepting connections and removes the socket file.
    ///
    /// Idempotent. Called first in [`ActonAI::shutdown`], before the actors:
    /// a listener still accepting after the actor behind it has stopped would
    /// answer `acton-ai status` with a connection that hangs or a routing
    /// error, which is a worse answer than a closed socket.
    pub(crate) fn shutdown(&self) {
        #[cfg(feature = "ipc")]
        {
            let taken = self.listener.lock().ok().and_then(|mut slot| slot.take());
            if let Some(listener) = taken {
                listener.stop();
            }
            // acton-reactive's accept loop unlinks the socket too, but only
            // after it notices the cancellation, from a spawned task — so
            // "has the file gone?" immediately after `shutdown()` would be a
            // race. Doing it here makes the answer deterministic. The two are
            // not redundant: `stop()` ends the accept loop, this ends the
            // filesystem entry, and each on its own leaves half a listener
            // behind.
            if let Err(error) = std::fs::remove_file(&self.socket_path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::debug!(
                        path = %self.socket_path.display(),
                        %error,
                        "introspection socket file could not be removed"
                    );
                }
            }
        }
    }
}

/// The telemetry a launched runtime owns.
///
/// Holding one means telemetry is on and the [`TelemetryActor`] was spawned.
/// The guard inside is `None` when the providers belong to the surrounding
/// application ([`ActonAIBuilder::telemetry_from_globals`]) — this crate must
/// then never shut them down, because it does not own them and other parts of
/// the application are still using them.
///
/// [`TelemetryActor`]: crate::telemetry::TelemetryActor
pub(crate) struct TelemetryRuntime {
    /// Behind a `Mutex` because shutdown runs through a shared `Arc`, and
    /// shutting the providers down needs ownership.
    #[cfg(feature = "otel")]
    guard: std::sync::Mutex<Option<crate::telemetry::TelemetryGuard>>,
}

impl TelemetryRuntime {
    /// Flushes and shuts down the providers this runtime installed.
    ///
    /// Idempotent, and a no-op when the providers came from the application.
    pub(crate) fn shutdown(&self) {
        #[cfg(feature = "otel")]
        {
            let taken = self.guard.lock().ok().and_then(|mut slot| slot.take());
            if let Some(mut guard) = taken {
                guard.shutdown();
            }
        }
    }
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
            .field("usage_tracking", &self.inner.accountant.is_some())
            .field("telemetry", &self.inner.telemetry.is_some())
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
        builder = self.inject_mcp_tools(builder);
        builder = self.inject_custom_tools(builder);
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

    /// The model a configured provider serves, if that provider exists.
    ///
    /// Used to label telemetry with the model actually dispatched to, rather
    /// than leaving it to be inferred from the provider name.
    #[must_use]
    pub fn provider_model(&self, name: &str) -> Option<&str> {
        self.inner.provider_models.get(name).map(String::as_str)
    }

    /// The failover chain configured for a provider, if it has one.
    ///
    /// The returned slice is the list of *fallbacks*, in order; the provider
    /// itself is not in it. `None` means no chain was configured, which is
    /// the state in which the prompt loop asks nothing extra per round.
    #[must_use]
    pub fn provider_failover(&self, name: &str) -> Option<&[String]> {
        self.inner.provider_failover.get(name).map(Vec::as_slice)
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

    /// Returns whether usage tracking is enabled for this runtime.
    #[must_use]
    pub fn is_usage_tracking(&self) -> bool {
        self.inner.accountant.is_some()
    }

    /// Returns whether a spending cap is being enforced.
    ///
    /// When true, every provider dispatch is preceded by a budget check and
    /// can fail with
    /// [`ActonAIErrorKind::BudgetExceeded`](crate::error::ActonAIErrorKind::BudgetExceeded).
    #[must_use]
    pub fn is_budget_enforced(&self) -> bool {
        self.inner.budget.is_some()
    }

    /// The accountant to pre-flight against, or `None` when nothing is
    /// capped.
    ///
    /// Returning `None` is what lets the prompt loop skip the ask round-trip
    /// altogether rather than paying for it and discarding the answer.
    pub(crate) fn budget_accountant(&self) -> Option<&ActorHandle> {
        self.inner
            .budget
            .as_ref()
            .and(self.inner.accountant.as_ref())
    }

    /// The tool-approval policy in force, or `None` when none was configured.
    ///
    /// `None` is what keeps an unconfigured runtime's tool path exactly as it
    /// was: the loop never builds an invocation, never awaits a hook, and
    /// never allocates.
    pub(crate) fn tool_policy(&self) -> Option<&crate::policy::ToolPolicy> {
        self.inner.tool_policy.as_ref()
    }

    /// The audit log and its settings, or `None` when no trail is configured.
    pub(crate) fn audit(&self) -> Option<(&ActorHandle, &crate::audit::AuditConfig)> {
        self.inner
            .audit
            .as_ref()
            .zip(self.inner.audit_config.as_ref())
    }

    /// Whether tool invocations are being recorded.
    #[must_use]
    pub fn is_audited(&self) -> bool {
        self.inner.audit.is_some()
    }

    /// Where the audit trail's hash chain currently ends.
    ///
    /// Doubles as a barrier: mailboxes are FIFO, so a reply proves every
    /// invocation recorded before this call has already been written. That is
    /// how an audited flow is awaited without sleeping.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when no audit trail is configured — an
    /// empty chain would read as "nothing happened" rather than "nothing was
    /// being recorded" — or a provider error if the audit actor cannot be
    /// reached.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let head = ai.audit_head().await?;
    /// println!("{} entries, head {}", head.entries, head.hash);
    /// ```
    pub async fn audit_head(&self) -> Result<crate::audit::ChainHead, ActonAIError> {
        let Some(audit) = self.inner.audit.as_ref() else {
            return Err(ActonAIError::configuration(
                "audit",
                "no audit trail is configured, so nothing was recorded; enable it with \
                 ActonAIBuilder::audit(..) or an `[audit]` section in your config file",
            ));
        };

        audit.ask(crate::audit::GetChainHead).await.map_err(|e| {
            ActonAIError::provider_error(format!("could not reach the audit log: {e}"))
        })
    }

    /// Returns a snapshot of the tokens — and, where pricing is configured,
    /// the cost — every provider in this runtime has accumulated.
    ///
    /// Taking a snapshot neither resets the totals nor pauses tallying.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when usage tracking was switched off
    /// with [`ActonAIBuilder::usage_tracking`] or `usage_tracking = false`.
    /// It is deliberately not an empty snapshot: a wall of zeros would be
    /// indistinguishable from a runtime that genuinely spent nothing.
    ///
    /// Returns a provider error if the accountant cannot be reached, which
    /// in practice means the runtime is shutting down.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let usage = ai.usage().await?;
    /// println!("{} requests, {} tokens", usage.requests, usage.totals.total_tokens());
    /// if let Some(usd) = usage.total_usd() {
    ///     println!("about ${usd:.4}");
    /// }
    /// ```
    pub async fn usage(&self) -> Result<UsageSnapshot, ActonAIError> {
        let Some(accountant) = self.inner.accountant.as_ref() else {
            return Err(ActonAIError::configuration(
                "usage_tracking",
                "usage tracking is disabled, so no usage was recorded; enable it with \
                 ActonAIBuilder::usage_tracking(true) or `usage_tracking = true` under \
                 [defaults] in your config file",
            ));
        };

        accountant.ask(GetUsage).await.map_err(|e| {
            ActonAIError::provider_error(format!("could not reach the cost accountant: {e}"))
        })
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

    /// Returns a handle that executes tools under this runtime's sandbox.
    ///
    /// `Some` when the builder was configured with
    /// [`with_process_sandbox`](ActonAIBuilder::with_process_sandbox) or
    /// [`with_process_sandbox_config`](ActonAIBuilder::with_process_sandbox_config)
    /// (or an equivalent TOML `[sandbox]` section); `None` otherwise. The
    /// handle shares the runtime's factory, so what it executes is isolated
    /// exactly as a sandboxed builtin would be.
    ///
    /// This exists for embedders that implement their own tool executors —
    /// an ACP agent daemon wrapping builtin execution with its own approval
    /// protocol, say — and need to run the underlying work on the same
    /// sandbox path `.use_builtins()` uses, rather than losing isolation by
    /// calling executors in-process. For wrapping one specific builtin,
    /// [`builtin_executor`](Self::builtin_executor) already pairs the tool
    /// with this decision.
    ///
    /// Note that a sandbox configured here re-execs the **current binary**;
    /// an embedder's `main` must call
    /// [`runner::run_if_sandbox_child`](crate::tools::sandbox::process::runner::run_if_sandbox_child)
    /// first thing, as documented there.
    #[must_use]
    pub fn sandboxed_execution(&self) -> Option<crate::tools::sandbox::SandboxedExecution> {
        self.inner
            .sandbox_factory
            .as_ref()
            .map(|factory| crate::tools::sandbox::SandboxedExecution::new(Arc::clone(factory)))
    }

    /// Returns the named builtin's execution path, sandbox routing included.
    ///
    /// `None` when builtins were not configured on this runtime or the name
    /// is not among them. The returned executor makes the same decision
    /// `.use_builtins()` makes for the prompt loop — and is in fact what the
    /// prompt loop registers — so a tool configured `sandboxed` on a runtime
    /// with a sandbox routes through it, and everything else runs in-process.
    ///
    /// This exists for embedders that wrap builtin execution (approval
    /// flows, protocol adapters, extra logging) while keeping the sandboxing
    /// the runtime would have applied. Note that the prompt loop's
    /// tool-approval gate and audit trail wrap *registered* tools: a wrapper
    /// registered on a prompt keeps both, a direct
    /// [`call`](crate::tools::builtins::BuiltinExecutor::call) outside the
    /// loop deliberately has neither.
    #[must_use]
    pub fn builtin_executor(&self, name: &str) -> Option<crate::tools::builtins::BuiltinExecutor> {
        let builtins = self.inner.builtins.as_ref()?;
        let config = builtins.get_config(name)?;
        let executor = builtins.get_executor(name)?;
        let sandbox = if config.sandboxed {
            self.sandboxed_execution()
        } else {
            None
        };
        Some(crate::tools::builtins::BuiltinExecutor::new(
            name, executor, sandbox,
        ))
    }

    /// Returns the MCP tools discovered at launch, if any server is configured.
    ///
    /// `None` means no `[mcp_servers.*]` entry and no
    /// [`with_mcp_server`](ActonAIBuilder::with_mcp_server) call.
    #[must_use]
    pub fn mcp(&self) -> Option<&crate::mcp::McpTools> {
        self.inner.mcp.as_ref()
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
        builder = self.inject_mcp_tools(builder);
        builder = self.inject_custom_tools(builder);
        builder
    }

    /// Registers every discovered MCP tool on `builder`. No-op when no MCP
    /// server is configured.
    ///
    /// There is no opt-out: configuring a server is the request to use it.
    /// Each executor holds a
    /// [`SupervisedChild`](acton_reactive::prelude::SupervisedChild), so a
    /// server that reconnects mid-conversation keeps working without the
    /// prompt being rebuilt.
    #[inline]
    fn inject_mcp_tools(&self, mut builder: PromptBuilder) -> PromptBuilder {
        let Some(mcp) = &self.inner.mcp else {
            return builder;
        };

        for (name, config) in mcp.configs() {
            let Some(executor) = mcp.get_executor(name) else {
                continue;
            };
            builder = builder.with_tool(config.definition.clone(), move |args| {
                let executor = Arc::clone(&executor);
                async move { executor.execute(args).await }
            });
        }

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

    /// Registers every runtime-wide custom tool on `builder`. No-op when none
    /// were staged.
    ///
    /// Called from both [`prompt`](Self::prompt) and
    /// [`continue_with`](Self::continue_with) — the same pair of sites the
    /// skill and MCP injections use — so every `PromptBuilder` the facade
    /// hands out, including the ones a
    /// [`Conversation`](crate::conversation::Conversation) rebuilds per turn,
    /// carries the tools registered once at build time.
    #[inline]
    fn inject_custom_tools(&self, builder: PromptBuilder) -> PromptBuilder {
        if self.inner.custom_tools.is_empty() {
            return builder;
        }
        builder.with_tool_specs(
            self.inner
                .custom_tools
                .iter()
                .map(crate::prompt::SharedToolSpec::to_tool_spec),
        )
    }

    /// Every tool name [`prompt`](Self::prompt) and
    /// [`continue_with`](Self::continue_with) inject automatically.
    ///
    /// This is the collision set a per-conversation tool is checked against:
    /// a name in this list is already taken on every turn the conversation
    /// will run, so registering it again could only shadow or duplicate.
    pub(crate) fn injected_tool_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if self.inner.auto_builtins {
            if let Some(builtins) = &self.inner.builtins {
                names.extend(
                    builtins
                        .configs()
                        .map(|(_, config)| config.definition.name.clone()),
                );
            }
        }
        if self.inner.skills.is_some() {
            use crate::tools::builtins::{ActivateSkillTool, ListSkillsTool};
            names.push(ListSkillsTool::config().definition.name);
            names.push(ActivateSkillTool::config().definition.name);
        }
        if let Some(mcp) = &self.inner.mcp {
            names.extend(
                mcp.configs()
                    .map(|(_, config)| config.definition.name.clone()),
            );
        }
        names.extend(
            self.inner
                .custom_tools
                .iter()
                .map(|spec| spec.name().to_string()),
        );
        names
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
    /// Whether this runtime is currently admitting new turns, and if not, why.
    ///
    /// Free to call: one relaxed atomic load.
    #[must_use]
    pub fn admission_state(&self) -> crate::introspection::AdmissionState {
        self.inner.admission.state()
    }

    /// Stops admitting new turns. Turns already running are unaffected.
    ///
    /// The in-process twin of `acton-ai pause`, and available whether or not
    /// the `ipc` feature is compiled in — an embedder with its own control
    /// plane needs this lever without needing a socket.
    ///
    /// A turn refused while paused fails with
    /// [`ActonAIError::is_turns_not_admitted`], which is a refusal rather than
    /// a failure: nothing was sent and nothing was spent.
    ///
    /// Returns the state now in force.
    pub fn pause(&self) -> crate::introspection::AdmissionState {
        self.inner.admission.pause()
    }

    /// Admits new turns again after a [`pause`](Self::pause).
    ///
    /// Also lifts a [`drain`](Self::drain), for the operator who started one
    /// and changed their mind before the process went down.
    ///
    /// Returns the state now in force.
    pub fn resume(&self) -> crate::introspection::AdmissionState {
        self.inner.admission.resume()
    }

    /// Stops admitting new turns, with the intent of shutting down.
    ///
    /// Identical to [`pause`](Self::pause) from a turn's point of view; the
    /// difference is what it tells an operator reading `acton-ai status`. A
    /// paused process is waiting for a human, a draining one is waiting for
    /// its own in-flight work.
    ///
    /// This returns immediately — it closes the door, it does not wait for the
    /// room to empty. `acton-ai drain --wait` polls the status surface for
    /// that; in-process, a caller that needs the same thing awaits its own
    /// outstanding calls, which it is holding anyway.
    ///
    /// Returns the state now in force.
    pub fn drain(&self) -> crate::introspection::AdmissionState {
        self.inner.admission.drain()
    }

    /// The path of this runtime's control socket, when one is listening.
    ///
    /// `None` when introspection was not configured. Worth asking for even
    /// when it was: the default scheme suffixes the PID, so this is the only
    /// way to learn the address from inside the process.
    #[must_use]
    pub fn introspection_socket(&self) -> Option<&std::path::Path> {
        self.inner
            .introspection
            .as_ref()
            .map(|introspection| introspection.socket_path.as_path())
    }

    pub async fn shutdown(self) -> Result<(), ActonAIError> {
        self.inner.is_shutdown.store(true, Ordering::SeqCst);

        // The socket goes first, before the actors it fronts. A listener that
        // outlived its actor would accept a connection and then fail to route
        // it, which reads to an operator as a broken process rather than a
        // stopped one.
        if let Some(introspection) = self.inner.introspection.as_ref() {
            introspection.shutdown();
        }

        // Get the runtime clone for shutdown. The Arc may still be shared,
        // so we clone the ActorRuntime (which is itself cheap to clone).
        let mut runtime = self.inner.runtime.clone();
        let stopped = runtime
            .shutdown_all()
            .await
            .map_err(|e| ActonAIError::launch_failed(e.to_string()));

        // Telemetry goes last, deliberately: the actors above broadcast on
        // their way down, and the telemetry actor records those broadcasts.
        // Flushing first would drop exactly the final batch — the spans and
        // counters covering the end of the run, which are the ones an
        // operator went looking for.
        //
        // Run even when stopping the actors failed: a failed shutdown is
        // precisely when the telemetry is worth having.
        if let Some(telemetry) = self.inner.telemetry.as_ref() {
            telemetry.shutdown();
        }

        stopped
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
    /// Custom tools staged by [`with_tool`](Self::with_tool),
    /// [`with_tool_executor`](Self::with_tool_executor), and
    /// [`add_tool`](Self::add_tool). Name-checked against the built-ins, the
    /// skill tools, the MCP tools, and each other in
    /// [`launch`](Self::launch), then injected into every prompt.
    custom_tools: Vec<crate::prompt::SharedToolSpec>,
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
    /// External MCP servers staged by [`with_mcp_server`](Self::with_mcp_server)
    /// and by `[mcp_servers.*]` in TOML. Each becomes one supervised
    /// connection actor at launch.
    mcp_servers: HashMap<String, crate::config::McpServerConfig>,
    /// Whether to spawn the cost accountant at launch.
    ///
    /// `None` means "not decided here", which lets `[defaults]
    /// usage_tracking` speak; the framework default when neither does is
    /// **on**. Builder beats TOML beats default, matching `max_tool_rounds`.
    usage_tracking: Option<bool>,
    /// Pricing per configured provider, collected from
    /// `[providers.<name>.pricing]` and converted to integer micro-USD at
    /// load. Empty when nothing is priced, which is a supported state.
    pricing: PricingTable,
    /// Caps set programmatically with [`budget`](Self::budget) or
    /// [`budget_usd`](Self::budget_usd).
    ///
    /// Present, this replaces `[budget]` from the config file **wholesale** —
    /// the same rule `usage_tracking` follows, and for the same reason: a
    /// half-merged budget is a cap nobody wrote.
    budget: Option<Budget>,
    /// The `[budget]` section from a config file, used only when the builder
    /// sets no budget of its own. Resolved at launch rather than in
    /// [`apply_config`](Self::apply_config), so `.budget(..)` wins whether it
    /// is called before or after `.from_config()`.
    budget_file: Option<crate::config::BudgetFileConfig>,
    /// Callback registered with [`on_budget_event`](Self::on_budget_event).
    budget_event_callback: Option<Arc<dyn Fn(BudgetEvent) + Send + Sync>>,
    /// Callback registered with [`on_failover_event`](Self::on_failover_event).
    failover_event_callback: Option<Arc<dyn Fn(FailoverEvent) + Send + Sync>>,
    /// Telemetry set programmatically with [`telemetry`](Self::telemetry),
    /// [`telemetry_otlp`](Self::telemetry_otlp), or
    /// [`telemetry_from_globals`](Self::telemetry_from_globals).
    ///
    /// Present, this replaces `[telemetry]` from the config file
    /// **wholesale** — the same rule `budget` and `usage_tracking` follow.
    #[cfg(feature = "otel")]
    telemetry: Option<TelemetrySetup>,
    /// The `[telemetry]` section from a config file, used only when the
    /// builder sets no telemetry of its own. Resolved at launch rather than
    /// in [`apply_config`](Self::apply_config), so a builder call wins
    /// whether it comes before or after `.from_config()`.
    ///
    /// Unconditional: a build without the `otel` feature still has to notice
    /// this section and refuse the launch rather than ignore it.
    telemetry_file: Option<crate::config::TelemetryFileConfig>,
    /// Introspection set programmatically by
    /// [`introspection`](Self::introspection) or
    /// [`introspection_at`](Self::introspection_at).
    ///
    /// Present, this replaces `[introspection]` from the config file
    /// **wholesale** — the same rule `budget` and `telemetry` follow.
    introspection: Option<crate::introspection::IntrospectionConfig>,
    /// The `[introspection]` section from a config file, used only when the
    /// builder sets none of its own.
    ///
    /// Unconditional: a build without the `ipc` feature still has to notice
    /// this section at launch and refuse, rather than ignore it.
    introspection_file: Option<crate::config::IntrospectionFileConfig>,
    /// A policy set programmatically with [`tool_policy`](Self::tool_policy).
    ///
    /// Present, this replaces `[tool_policy]` from the config file
    /// **wholesale** — the rule budgets already follow, and for the same
    /// reason: a half-merged policy is a rule nobody wrote.
    tool_policy: Option<crate::policy::ToolPolicy>,
    /// The `[tool_policy]` section, used only when the builder sets no policy
    /// of its own. Resolved at launch rather than in
    /// [`apply_config`](Self::apply_config), so `.tool_policy(..)` wins
    /// whether it is called before or after `.from_config()`.
    tool_policy_file: Option<crate::config::ToolPolicyFileConfig>,
    /// The approval hook, which has no TOML form and so is always the
    /// builder's. Attached to whichever policy resolves.
    tool_approval_hook: Option<Arc<dyn crate::policy::ApprovalHookFn>>,
    /// Audit settings set programmatically with [`audit`](Self::audit).
    audit: Option<crate::audit::AuditConfig>,
    /// The `[audit]` section, used only when the builder sets none.
    audit_file: Option<crate::config::AuditFileConfig>,
    /// Whether `SIGTERM` should start a drain.
    drain_on_sigterm: bool,
}

/// How a runtime's telemetry providers come to exist.
///
/// Two genuinely different situations rather than one with a flag: either
/// this crate installs the providers, or the surrounding application already
/// installed its own and this crate should emit into them.
#[cfg(feature = "otel")]
#[derive(Debug)]
enum TelemetrySetup {
    /// Install providers of our own, exporting over OTLP.
    Otlp(Box<crate::telemetry::Telemetry>),
    /// Take ownership of providers the caller already assembled through
    /// [`install_with_exporters`](crate::telemetry::install_with_exporters).
    Guard(Box<crate::telemetry::TelemetryGuard>),
    /// Emit into whatever providers are already installed as the process
    /// globals. Installs nothing and holds no guard.
    Globals,
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

    /// Enables or disables token-usage tracking. **Enabled by default.**
    ///
    /// When enabled the runtime spawns a cost accountant that tallies the
    /// usage every provider broadcasts, readable through
    /// [`ActonAI::usage`]. When disabled that actor is simply not spawned —
    /// providers still broadcast, since with no subscriber a broadcast costs
    /// nothing — and `usage()` returns a configuration error.
    ///
    /// Takes precedence over `usage_tracking` under `[defaults]` in a config
    /// file.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .ollama("qwen2.5:7b")
    ///     .usage_tracking(false)  // opt out
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn usage_tracking(mut self, enabled: bool) -> Self {
        self.usage_tracking = Some(enabled);
        self
    }

    /// Installs pricing for one configured provider.
    ///
    /// The programmatic twin of `[providers.<name>.pricing]`, and what makes
    /// a budget possible without a config file: caps compare priced spend, so
    /// a provider with no rates has no spend to compare.
    ///
    /// Repeatable, and last write wins — the same rule
    /// [`apply_config`](Self::apply_config) already follows for providers, so
    /// a call placed after `from_config()` overrides the file's rates and one
    /// placed before is overridden by them.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::prelude::ModelPricing;
    ///
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .pricing(DEFAULT_PROVIDER_NAME, ModelPricing::from_dollars_per_mtok(3.0, 15.0))
    ///     .budget_usd(5.00)
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn pricing(
        mut self,
        provider: impl Into<String>,
        pricing: crate::accounting::ModelPricing,
    ) -> Self {
        self.pricing.insert(provider, pricing);
        self
    }

    /// Caps process-wide spending at `dollars`, refusing requests past it.
    ///
    /// The one-liner form of [`budget`](Self::budget): a process-wide cap,
    /// warning at [`DEFAULT_WARN_AT_PERCENT`](crate::accounting::DEFAULT_WARN_AT_PERCENT),
    /// refusing at the cap. Reach for the full form when you need
    /// per-provider caps, a different warning threshold, or unpriced
    /// providers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .budget_usd(5.00)
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn budget_usd(self, dollars: f64) -> Self {
        self.budget(Budget::usd(dollars))
    }

    /// Installs spending caps enforced before every provider dispatch.
    ///
    /// Replaces any `[budget]` section from a config file wholesale, whether
    /// this is called before or after `from_config()`.
    ///
    /// Two things fail the launch rather than quietly weakening the cap: a
    /// budget alongside `usage_tracking(false)` (there would be nothing
    /// counting), and a budget alongside a configured provider that has no
    /// pricing (the cap could never be reached). [`Budget::allow_unpriced`]
    /// opts out of the second.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::prelude::Budget;
    ///
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .budget(
    ///         Budget::usd(5.00)
    ///             .provider("claude", 2.00)
    ///             .warn_at_percent(50),
    ///     )
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Runs `callback` for every [`BudgetEvent`] the accountant broadcasts.
    ///
    /// The callback runs inside a small subscriber actor's message loop, so
    /// it must be cheap and must not block — log, increment a counter, send
    /// on a channel. Anything slower belongs in an actor of your own
    /// subscribed to [`BudgetEvent`] directly.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .budget_usd(5.00)
    ///     .on_budget_event(|event| eprintln!("budget: {event}"))
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn on_budget_event(
        mut self,
        callback: impl Fn(BudgetEvent) + Send + Sync + 'static,
    ) -> Self {
        self.budget_event_callback = Some(Arc::new(callback));
        self
    }

    /// Runs `callback` for every [`FailoverEvent`] broadcast on the broker.
    ///
    /// This is how an operator finds out that a provider tripped open, that a
    /// round was served by a fallback, or that a rate-limited provider
    /// quietly switched to a cheaper model. Same discipline as
    /// [`on_budget_event`](Self::on_budget_event): the callback runs inside a
    /// small subscriber actor's message loop, so it must be cheap and must
    /// not block.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .on_failover_event(|event| eprintln!("failover: {event}"))
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn on_failover_event(
        mut self,
        callback: impl Fn(FailoverEvent) + Send + Sync + 'static,
    ) -> Self {
        self.failover_event_callback = Some(Arc::new(callback));
        self
    }

    // =========================================================================
    // Telemetry
    // =========================================================================

    /// Exports traces and metrics over OTLP to `endpoint`.
    ///
    /// The one-liner form of [`telemetry`](Self::telemetry). `endpoint` is
    /// the collector's base HTTP address; the exporter appends `/v1/traces`
    /// and `/v1/metrics`.
    ///
    /// Replaces any `[telemetry]` section from a config file wholesale.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .telemetry_otlp("http://localhost:4318")
    ///     .launch()
    ///     .await?;
    /// ```
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn telemetry_otlp(self, endpoint: impl Into<String>) -> Self {
        self.telemetry(crate::telemetry::Telemetry::otlp(endpoint))
    }

    /// Exports traces and metrics using the supplied [`Telemetry`] settings.
    ///
    /// Replaces any `[telemetry]` section from a config file wholesale,
    /// whether `.from_config()` was called before or after this — a
    /// half-merged telemetry config points at an endpoint nobody wrote.
    ///
    /// The endpoint and interval are validated at launch, so this stays
    /// infallible and every configuration error surfaces from one place.
    ///
    /// [`Telemetry`]: crate::telemetry::Telemetry
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::telemetry::Telemetry;
    ///
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .telemetry(
    ///         Telemetry::otlp("https://collector.example:4318")
    ///             .service_name("my-agent")
    ///             .metrics_interval_secs(15)
    ///             .header("authorization", "Bearer ..."),
    ///     )
    ///     .launch()
    ///     .await?;
    /// ```
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn telemetry(mut self, telemetry: crate::telemetry::Telemetry) -> Self {
        self.telemetry = Some(TelemetrySetup::Otlp(Box::new(telemetry)));
        self
    }

    /// Emits into the OpenTelemetry providers already installed in this
    /// process instead of installing any.
    ///
    /// For applications that have already configured OpenTelemetry for
    /// themselves — an HTTP server, a job runner — and want this crate's
    /// spans and metrics to land in the same pipeline rather than in a
    /// second, competing one. Nothing is installed and no exporter is built,
    /// so the endpoint, service name, and interval all come from whatever the
    /// application set up.
    ///
    /// If no providers are installed, OpenTelemetry's globals are no-ops and
    /// this is simply telemetry that goes nowhere.
    ///
    /// # Lifecycle
    ///
    /// The application owns those providers, so
    /// [`shutdown`](ActonAI::shutdown) deliberately neither flushes nor shuts
    /// them down — doing either to a provider the rest of the application is
    /// still using would be worse than the batch it saved. Flush them
    /// yourself before exiting, or use
    /// [`telemetry_guard`](Self::telemetry_guard) to hand this runtime the
    /// lifecycle instead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // The application installed its own providers earlier in main().
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .telemetry_from_globals()
    ///     .launch()
    ///     .await?;
    /// ```
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn telemetry_from_globals(mut self) -> Self {
        self.telemetry = Some(TelemetrySetup::Globals);
        self
    }

    /// Emits into providers assembled with
    /// [`install_with_exporters`](crate::telemetry::install_with_exporters),
    /// and takes over their lifecycle.
    ///
    /// This is the escape hatch from OTLP: build the providers around any
    /// exporters you like — a different protocol, a file, an in-memory
    /// recorder — and hand the resulting guard over. Unlike
    /// [`telemetry_from_globals`](Self::telemetry_from_globals), this runtime
    /// now owns them, so [`shutdown`](ActonAI::shutdown) flushes and shuts
    /// them down after the actors stop, exactly as it does for OTLP.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::telemetry::{install_with_exporters, Telemetry};
    ///
    /// let config = Telemetry::otlp("http://unused:4318").to_config()?;
    /// let guard = install_with_exporters(&config, my_span_exporter, my_metric_exporter);
    ///
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .telemetry_guard(guard)
    ///     .launch()
    ///     .await?;
    /// ```
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn telemetry_guard(mut self, guard: crate::telemetry::TelemetryGuard) -> Self {
        self.telemetry = Some(TelemetrySetup::Guard(Box::new(guard)));
        self
    }

    /// Listens for `acton-ai status`, `pause`, `resume`, and `drain` on a
    /// control socket.
    ///
    /// The socket goes at
    /// `$XDG_RUNTIME_DIR/acton-ai/<app-name>-<pid>.sock`, owner-only. The PID
    /// suffix keeps two processes for one user from colliding; use
    /// [`introspection_at`](Self::introspection_at) when something outside the
    /// process needs a predictable address, and
    /// [`ActonAI::introspection_socket`] to learn the resolved one from
    /// inside.
    ///
    /// Replaces any `[introspection]` section from a config file wholesale.
    ///
    /// ```rust,no_run
    /// # use acton_ai::prelude::*;
    /// # async fn run() -> Result<(), ActonAIError> {
    /// let ai = ActonAI::builder()
    ///     .app_name("my-agent")
    ///     .ollama("llama3.2")
    ///     .introspection()
    ///     .drain_on_sigterm()
    ///     .launch()
    ///     .await?;
    ///
    /// println!("status socket: {}", ai.introspection_socket().unwrap().display());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn introspection(mut self) -> Self {
        self.introspection = Some(crate::introspection::IntrospectionConfig::default());
        self
    }

    /// Installs the rules applied to every tool call before it runs.
    ///
    /// Covers built-ins, `#[tool]` functions and MCP tools uniformly, because
    /// the gate sits at the one point the prompt loop dispatches a call.
    /// Replaces any `[tool_policy]` section wholesale, whether this is called
    /// before or after `from_config()`.
    ///
    /// A refused call is not an error: the tool does not run, the reason goes
    /// back to the model as that call's result, and the turn continues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use acton_ai::prelude::*;
    ///
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .tool_policy(
    ///         ToolPolicy::new()
    ///             .allow(["read_file", "mcp__fs__*"])
    ///             .deny(["bash"])
    ///             .cap_per_turn("mcp__fs__*", 5),
    ///     )
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn tool_policy(mut self, policy: crate::policy::ToolPolicy) -> Self {
        self.tool_policy = Some(policy);
        self
    }

    /// Runs `hook` before every tool call the rules already admitted.
    ///
    /// This is the human-in-the-loop seam: the hook may approve, approve a
    /// rewritten set of arguments, or refuse with a reason. It is awaited on
    /// the prompt loop's own task, so a hook that waits for a person holds the
    /// turn open — which is the point.
    ///
    /// A hook cannot be written in TOML, so this applies to whichever policy
    /// resolves, whether that came from the builder or a config file. Setting
    /// a hook with no other rules is a complete policy on its own.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .on_tool_approval(|invocation| async move {
    ///         if invocation.tool_name == "bash" {
    ///             ApprovalDecision::deny("shell access needs a human")
    ///         } else {
    ///             ApprovalDecision::Approve
    ///         }
    ///     })
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn on_tool_approval<H>(mut self, hook: H) -> Self
    where
        H: crate::policy::ApprovalHookFn + 'static,
    {
        self.tool_approval_hook = Some(Arc::new(hook));
        self
    }

    /// Records every tool invocation to a tamper-evident, hash-chained trail.
    ///
    /// Replaces any `[audit]` section wholesale. Off unless one of the two is
    /// present: no trail is written and no actor is spawned.
    ///
    /// Every invocation produces an entry, including refused and failed ones —
    /// a trail that recorded only successes would answer the wrong question.
    /// Arguments are redacted before they are written.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ai = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .audit(AuditConfig::new("/var/log/acton-ai/audit.jsonl"))
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn audit(mut self, config: crate::audit::AuditConfig) -> Self {
        self.audit = Some(config);
        self
    }

    /// Records every tool invocation to a trail at `path`, with the default
    /// redaction patterns.
    ///
    /// A one-liner for the common case; [`audit`](Self::audit) takes the full
    /// settings.
    #[must_use]
    pub fn audit_to(self, path: impl Into<std::path::PathBuf>) -> Self {
        self.audit(crate::audit::AuditConfig::new(path))
    }

    /// Listens for control commands on a socket at `path`.
    ///
    /// As [`introspection`](Self::introspection), but at an address chosen by
    /// the caller — which is what a systemd unit or a deployment script needs,
    /// since neither can guess a PID.
    ///
    /// The path must be absolute; a relative one fails the launch, because a
    /// process that changes its working directory would otherwise leave its
    /// control socket somewhere no client can find it.
    #[must_use]
    pub fn introspection_at(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.introspection = Some(crate::introspection::IntrospectionConfig {
            socket_path: Some(path.into()),
            ..crate::introspection::IntrospectionConfig::default()
        });
        self
    }

    /// Starts a drain when the process receives `SIGTERM`.
    ///
    /// On the signal, the runtime stops admitting new turns and tells systemd
    /// it is `STOPPING=1`. Turns already running are not interrupted — that is
    /// the whole point, and it is what makes a rolling restart lose no work.
    ///
    /// This does **not** exit the process. Deciding when the process is done
    /// belongs to whoever owns `main`: they know what else is outstanding, and
    /// they are holding the turns this can only count. A typical `main` awaits
    /// its work and then calls [`ActonAI::shutdown`].
    ///
    /// Independent of [`introspection`](Self::introspection): a process can
    /// drain on `SIGTERM` without opening a socket, and vice versa. Unix only;
    /// elsewhere it is accepted and does nothing, because there is no
    /// `SIGTERM` to hear.
    #[must_use]
    pub fn drain_on_sigterm(mut self) -> Self {
        self.drain_on_sigterm = true;
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
                self.context_window_per_provider
                    .insert(name.clone(), tokens);
            }
            // Dollars become integer micro-USD here, once, at the config
            // boundary — every figure computed downstream is integer.
            if let Some(ref pricing) = provider_config.pricing {
                self.pricing
                    .insert(name.clone(), pricing.to_model_pricing());
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
        if let Some(defaults) = config.defaults {
            if self.default_max_tool_rounds.is_none() {
                self.default_max_tool_rounds = defaults.max_tool_rounds;
            }
            if self.usage_tracking.is_none() {
                self.usage_tracking = defaults.usage_tracking;
            }
        }

        // Merge `[skills] paths` onto whatever was staged programmatically so
        // CLI flags and config union (matching how providers are merged
        // above). Builder-stage paths come first; config-stage paths append.
        if let Some(skills_cfg) = config.skills {
            self.skill_paths.extend(skills_cfg.paths);
        }

        // Stash `[budget]` rather than converting it here: a `.budget(..)`
        // call may still be coming, and it replaces this section wholesale.
        if let Some(budget) = config.budget {
            self.budget_file = Some(budget);
        }

        // Same reasoning for `[telemetry]`, and additionally: stashing it
        // unconditionally is what lets a build without the `otel` feature
        // notice the section at launch and say so, rather than drop it.
        if let Some(telemetry) = config.telemetry {
            self.telemetry_file = Some(telemetry);
        }

        // And again for `[introspection]`: stashed unconditionally so a build
        // without the `ipc` feature notices the section at launch and says so.
        if let Some(introspection) = config.introspection {
            self.introspection_file = Some(introspection);
        }

        // `[tool_policy]` and `[audit]` follow the same rule: stashed, not
        // resolved. Resolution happens at launch so `.tool_policy(..)` wins
        // regardless of whether it was called before or after `from_config`.
        if let Some(tool_policy) = config.tool_policy {
            self.tool_policy_file = Some(tool_policy);
        }

        if let Some(audit) = config.audit {
            self.audit_file = Some(audit);
        }

        // Stash `[context]` settings — resolved at launch after the default
        // provider is known so per-provider overrides can win.
        if let Some(context_cfg) = config.context {
            self.context_config = Some(context_cfg);
        }

        // Merge `[mcp_servers.*]`. A builder entry with the same name wins:
        // an explicit `with_mcp_server` call is a deliberate override of what
        // the file says, not a duplicate of it.
        for (name, server) in config.mcp_servers {
            self.mcp_servers.entry(name).or_insert(server);
        }

        Ok(self)
    }

    /// Registers an external MCP server whose tools become available to every
    /// prompt.
    ///
    /// Repeatable; the name is the `{server}` segment in the tool names the
    /// LLM sees (`mcp__{server}__{tool}`). Calling this with a name that also
    /// appears in `[mcp_servers.*]` overrides the file entry.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-app")
    ///     .ollama("qwen2.5:7b")
    ///     .with_mcp_server(
    ///         "filesystem",
    ///         McpServerConfig::stdio("npx")
    ///             .with_args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
    ///     )
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_mcp_server(
        mut self,
        name: impl Into<String>,
        config: crate::config::McpServerConfig,
    ) -> Self {
        self.mcp_servers.insert(name.into(), config);
        self
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

    /// Registers a custom tool once, for every prompt and conversation this
    /// runtime produces.
    ///
    /// This exists for embedders — an agent daemon installing something like
    /// an `apply_patch` tool — that need a tool available everywhere without
    /// re-registering it on each [`PromptBuilder`] via
    /// [`with_tool`](crate::prompt::PromptBuilder::with_tool). The tool is
    /// injected alongside the built-ins, skill tools, and MCP tools, and runs
    /// through the same policy gate and audit trail they do.
    ///
    /// Names are validated at [`launch`](Self::launch): a custom tool that
    /// collides with an enabled built-in, a skill tool, an MCP tool, or
    /// another custom tool fails the launch rather than silently shadowing
    /// anything.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let runtime = ActonAI::builder()
    ///     .app_name("my-daemon")
    ///     .anthropic_from_env()
    ///     .with_tool(apply_patch_definition(), |args| async move {
    ///         apply_patch(args).await
    ///     })
    ///     .launch()
    ///     .await?;
    ///
    /// // Available on every prompt, no per-call wiring:
    /// runtime.prompt("fix the failing test").collect().await?;
    /// ```
    #[must_use]
    pub fn with_tool<F, Fut>(mut self, definition: ToolDefinition, executor: F) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<serde_json::Value, crate::tools::ToolError>>
            + Send
            + 'static,
    {
        self.custom_tools
            .push(crate::prompt::SharedToolSpec::from_closure(
                definition, executor,
            ));
        self
    }

    /// Registers a custom tool backed by a
    /// [`ToolExecutorTrait`](crate::tools::ToolExecutorTrait) executor, for
    /// every prompt and conversation this runtime produces.
    ///
    /// This is [`with_tool`](Self::with_tool) for callers whose tool is an
    /// executor *object* rather than a closure — the shape a reusable tool
    /// library exports. The executor's
    /// [`validate_args`](crate::tools::ToolExecutorTrait::validate_args) runs
    /// before every execution, and the same launch-time name validation and
    /// policy/audit coverage as [`with_tool`](Self::with_tool) apply.
    #[must_use]
    pub fn with_tool_executor<E>(mut self, definition: ToolDefinition, executor: E) -> Self
    where
        E: crate::tools::ToolExecutorTrait + 'static,
    {
        self.custom_tools
            .push(crate::prompt::SharedToolSpec::from_executor(
                definition,
                Arc::new(Box::new(executor) as crate::tools::BoxedToolExecutor),
            ));
        self
    }

    /// Registers a value implementing the [`Tool`](crate::tools::Tool) trait
    /// — typically generated by the `#[tool]` attribute macro — for every
    /// prompt and conversation this runtime produces.
    ///
    /// This is the runtime-wide counterpart of
    /// [`PromptBuilder::add_tool`](crate::prompt::PromptBuilder::add_tool),
    /// for embedders that define tools with `#[tool]` and want them installed
    /// once at build time. The same launch-time name validation and
    /// policy/audit coverage as [`with_tool`](Self::with_tool) apply.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// /// Adds two numbers.
    /// #[tool]
    /// async fn add(a: i64, b: i64) -> Result<serde_json::Value, ToolError> {
    ///     Ok(serde_json::json!({ "sum": a + b }))
    /// }
    ///
    /// let runtime = ActonAI::builder()
    ///     .anthropic_from_env()
    ///     .add_tool(Add)
    ///     .launch()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn add_tool<T>(mut self, tool: T) -> Self
    where
        T: crate::tools::Tool,
    {
        self.custom_tools
            .push(crate::prompt::SharedToolSpec::from_tool_impl(tool));
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
    /// This spawns the actor runtime and LLM providers.
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

        // Budgets are settled before anything is spawned. Every failure here
        // is a cap that would not have held, and discovering that after the
        // first request is discovering it too late.
        let budget = self.resolve_budget()?;
        if let Some(ref budget) = budget {
            if !self.usage_tracking.unwrap_or(true) {
                return Err(ActonAIError::configuration(
                    "budget",
                    "a budget needs the cost accountant, but usage tracking is off; drop \
                     usage_tracking(false) / `usage_tracking = false` under [defaults], or drop \
                     the budget",
                ));
            }
            validate_budget_coverage(budget, &self.providers, &self.pricing)?;
        }

        // Routing is settled before anything is spawned, for the same reason
        // budgets are: a chain that names a provider nobody configured would
        // otherwise surface as an unexplained failure on the day the primary
        // dies, which is the worst possible day to discover it.
        validate_failover(&self.providers)?;

        // Telemetry is resolved and installed before anything is spawned, so
        // spans and metrics from the launch itself have somewhere to land.
        // Resolution can fail (bad endpoint, missing feature), and a failure
        // here must stop the launch rather than produce a runtime that
        // silently exports nothing.
        let telemetry = self.resolve_telemetry()?;

        // Settled before anything is spawned, for the same reason budgets and
        // routing are: a socket path that cannot be bound, or a section this
        // build cannot honour, must stop the launch rather than produce a
        // process an operator believes they can talk to.
        let introspection_config = self.resolve_introspection()?;

        // The tool policy and the audit trail are settled here for the same
        // reason: both are refusals waiting to happen, and a rule that turns
        // out to be malformed on the first tool call is a rule that was not
        // enforcing anything in the meantime. `resolve_audit` also creates the
        // trail's directory, so an unwritable path fails the launch rather
        // than silently dropping every entry.
        let tool_policy = self.resolve_tool_policy()?;
        let audit_config = self.resolve_audit()?;

        // Idempotent, and here for embedders who never run this crate's
        // `main`: a FIPS build must have its provider installed before the
        // first provider actor can reach an API, and this is the last point
        // that is still true.
        crate::fips::install_crypto_provider()?;

        let app_name = self
            .app_name
            .take()
            .unwrap_or_else(|| "acton-ai".to_string());

        // Launch the actor runtime
        let mut runtime = ActonApp::launch_async().await;

        // Install journald logging if no other subscriber has claimed the
        // global slot yet. Non-Linux hosts and environments without a
        // journald socket silently fall through — the caller is expected to
        // install their own subscriber (e.g. the CLI's stderr layer).
        let logging_config = LoggingConfig::default().with_app_name(&app_name);
        match init_and_store_logging(&logging_config) {
            Ok(true) => {
                tracing::info!(app_name = %logging_config.app_name, "Journald logging initialized");
            }
            Ok(false) => {
                // Disabled, already installed, or journald unavailable — nothing to do.
            }
            Err(e) => {
                eprintln!("Warning: journald logging initialization failed: {e}");
            }
        }

        // Spawn all LLM providers
        let mut providers = HashMap::new();
        // Captured here because the config map is consumed by this loop, and
        // the prompt loop needs the model name to label round spans and
        // latency metrics — the provider actor knows it, but by then the
        // telemetry has already been recorded.
        let mut provider_models = HashMap::new();
        // Only providers that actually configured a chain go in here: the
        // prompt loop treats absence as "dispatch exactly as before".
        let mut provider_failover = HashMap::new();
        for (name, config) in std::mem::take(&mut self.providers) {
            provider_models.insert(name.clone(), config.model.clone());
            if !config.failover.is_empty() {
                provider_failover.insert(name.clone(), config.failover.clone());
            }
            // The provider is told its configured name so the usage reports
            // it broadcasts are keyed by it rather than by vendor.
            let handle = LLMProvider::spawn(&mut runtime, name.clone(), config).await;
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

        // Bring MCP servers up under supervision before handing the runtime
        // out. A server that cannot connect is a launch failure naming it:
        // silently starting without the tools the caller configured would
        // surface later as an unexplained "no such tool".
        let mcp_specs =
            crate::mcp::launch::specs_from_config(&self.mcp_servers).map_err(ActonAIError::from)?;
        let mcp = if mcp_specs.is_empty() {
            None
        } else {
            let tools = crate::mcp::launch::launch_mcp_servers(&mut runtime, &mcp_specs)
                .await
                .map_err(ActonAIError::from)?;
            tracing::info!(
                servers = mcp_specs.len(),
                tools = tools.len(),
                "MCP servers connected"
            );
            Some(tools)
        };

        // Custom tool names are validated here, once everything they could
        // collide with — built-ins, skill tools, MCP tools — is resolved. A
        // collision is a launch failure: two tools with one name would either
        // shadow silently or send the provider a duplicate definition, and
        // neither is something to discover mid-conversation.
        validate_custom_tool_names(
            &self.custom_tools,
            builtins.as_ref(),
            skills.is_some(),
            mcp.as_ref(),
        )?;

        // Tracking is on unless something explicitly turned it off. The
        // accountant is a plain top-level actor: no IO, no connection, so
        // nothing for supervision to repair — and a restart would zero the
        // very totals it exists to keep.
        let accountant = if self.usage_tracking.unwrap_or(true) {
            let handle = crate::accounting::CostAccountant::spawn(
                &mut runtime,
                self.pricing,
                budget.clone(),
            )
            .await;
            tracing::debug!(budget = budget.is_some(), "usage tracking enabled");
            Some(handle)
        } else {
            tracing::debug!("usage tracking disabled; ActonAI::usage() will report as much");
            None
        };

        // The one writer of the trail. Spawned before any prompt can run, and
        // only when a trail was configured: with no `[audit]` section and no
        // `.audit(..)` call there is no actor and nothing is written.
        let audit = match audit_config {
            Some(ref config) => Some(crate::audit::AuditLog::spawn(&mut runtime, config).await?),
            None => None,
        };

        // Spawned after the accountant so the subscription is in place before
        // anything can broadcast. Only when a callback was actually
        // registered — an actor whose closure is `None` is pure overhead.
        if let Some(callback) = self.budget_event_callback.take() {
            crate::accounting::BudgetEventListener::spawn(&mut runtime, callback).await;
        }

        // Same shape, different broadcast. Only when a callback was
        // registered — an actor whose closure is `None` is pure overhead.
        if let Some(callback) = self.failover_event_callback.take() {
            crate::llm::failover::FailoverEventListener::spawn(&mut runtime, callback).await;
        }

        // Only when telemetry is configured: with no providers installed the
        // actor would subscribe to three broadcasts and record every one of
        // them into a no-op instrument.
        #[cfg(feature = "otel")]
        if telemetry.is_some() {
            crate::telemetry::TelemetryActor::spawn(&mut runtime).await;
            tracing::debug!("telemetry enabled");
        }

        // The gate exists whether or not anything can drive it from outside:
        // the prompt loop reads it unconditionally, and an embedder gets
        // pause/resume/drain with no socket at all.
        let admission = crate::introspection::AdmissionGate::new();

        // Last of the actors, deliberately. Everything a status report reads —
        // providers, MCP supervision, the accountant — is up by now, so the
        // first `acton-ai status` to arrive describes a complete runtime
        // rather than a half-built one.
        let introspection = match introspection_config {
            Some(config) => Some(
                start_introspection(
                    &mut runtime,
                    &config,
                    &app_name,
                    admission.clone(),
                    IntrospectionCollaborators {
                        providers: &providers,
                        provider_models: &provider_models,
                        provider_failover: &provider_failover,
                        accountant: accountant.as_ref(),
                        mcp: mcp.as_ref(),
                    },
                )
                .await?,
            ),
            None => None,
        };

        if self.drain_on_sigterm {
            spawn_sigterm_drain(admission.clone());
        }

        let context_window = resolve_context_window(
            self.context_window_override.take(),
            self.context_window_disabled,
            self.context_config.as_ref(),
            self.context_window_per_provider
                .get(&default_provider_name)
                .copied(),
            &default_provider_model,
        );

        Ok(ActonAI {
            inner: Arc::new(ActonAIInner {
                runtime,
                providers,
                default_provider: default_provider_name,
                provider_models,
                provider_failover,
                builtins,
                auto_builtins: self.auto_builtins,
                skills,
                sandbox_factory,
                default_max_tool_rounds,
                context_window,
                accountant,
                budget,
                mcp,
                custom_tools: self.custom_tools,
                telemetry,
                admission,
                introspection,
                tool_policy,
                audit,
                audit_config,
                is_shutdown: AtomicBool::new(false),
            }),
        })
    }

    /// Resolves the effective introspection: builder first, then
    /// `[introspection]` TOML.
    ///
    /// Wholesale, never field-by-field — the same precedence `budget` and
    /// `telemetry` follow. `None` means nothing listens, which is what a
    /// runtime that configured neither gets.
    ///
    /// Without the `ipc` feature this is where a configured `[introspection]`
    /// section becomes a launch failure. Ignoring it would leave an operator
    /// debugging a socket that was never going to exist.
    fn resolve_introspection(
        &mut self,
    ) -> Result<Option<crate::introspection::IntrospectionConfig>, ActonAIError> {
        let resolved = match self.introspection.take() {
            // The builder path is already a validated config in shape only —
            // `introspection_at` takes any path — so it goes through the same
            // resolution the TOML path does rather than a second dialect of
            // "valid".
            Some(config) => Some(crate::introspection::IntrospectionConfig::resolve(
                config.socket_path,
                Some(config.socket_mode),
            )?),
            None => match self.introspection_file.take() {
                // `enabled = false` is a deployment switching the socket off
                // without deleting the settings it will want back, so it is
                // not an error and not a launch failure in a build without the
                // feature either: nothing was going to listen anyway.
                Some(file) if !file.is_enabled() => None,
                Some(file) => Some(file.to_introspection()?),
                None => None,
            },
        };

        #[cfg(not(feature = "ipc"))]
        if resolved.is_some() {
            return Err(crate::introspection::unsupported_error());
        }

        Ok(resolved)
    }

    /// Resolves the effective telemetry: builder first, then `[telemetry]`
    /// TOML, and installs the providers.
    ///
    /// Wholesale, never field-by-field — the same precedence `budget` and
    /// `usage_tracking` follow.
    ///
    /// Without the `otel` feature this is where a configured `[telemetry]`
    /// section becomes a launch failure. Ignoring it would leave an operator
    /// debugging a collector that was never going to receive anything.
    fn resolve_telemetry(&mut self) -> Result<Option<TelemetryRuntime>, ActonAIError> {
        #[cfg(feature = "otel")]
        {
            if let Some(setup) = self.telemetry.take() {
                let config = match setup {
                    TelemetrySetup::Otlp(telemetry) => telemetry.to_config()?,
                    // Already installed by the caller; we only take over the
                    // flushing and shutting down.
                    TelemetrySetup::Guard(guard) => {
                        return Ok(Some(TelemetryRuntime {
                            guard: std::sync::Mutex::new(Some(*guard)),
                        }))
                    }
                    // Nothing to build and nothing to own: the application
                    // installed the providers and owns their lifecycle.
                    TelemetrySetup::Globals => {
                        return Ok(Some(TelemetryRuntime {
                            guard: std::sync::Mutex::new(None),
                        }))
                    }
                };
                let guard = crate::telemetry::init_telemetry(&config)?;
                return Ok(Some(TelemetryRuntime {
                    guard: std::sync::Mutex::new(Some(guard)),
                }));
            }

            match self.telemetry_file.take() {
                Some(file) => {
                    let config = file.to_telemetry()?;
                    let guard = crate::telemetry::init_telemetry(&config)?;
                    Ok(Some(TelemetryRuntime {
                        guard: std::sync::Mutex::new(Some(guard)),
                    }))
                }
                None => Ok(None),
            }
        }
        #[cfg(not(feature = "otel"))]
        {
            if self.telemetry_file.is_some() {
                return Err(crate::telemetry::unsupported_error());
            }
            Ok(None)
        }
    }

    /// Resolves the effective budget: builder first, then `[budget]` TOML.
    ///
    /// Wholesale, never field-by-field — the same precedence
    /// `usage_tracking` follows.
    fn resolve_budget(&self) -> Result<Option<BudgetConfig>, ActonAIError> {
        if let Some(ref budget) = self.budget {
            return budget.to_config().map(Some);
        }
        self.budget_file
            .as_ref()
            .map(crate::config::BudgetFileConfig::to_budget)
            .transpose()
    }

    /// Resolves the effective tool policy: builder first, then
    /// `[tool_policy]` TOML, then the approval hook folded into whichever
    /// won.
    ///
    /// Wholesale, never field-by-field — the same precedence `budget` and
    /// `telemetry` follow. The hook is the one exception, and deliberately
    /// so: a callback cannot be written in TOML, so a config file that sets
    /// rules and a builder that sets a hook are describing two halves of one
    /// policy rather than competing for the same slot. A hook with no rules
    /// at all is still a policy — every call goes to the callback.
    fn resolve_tool_policy(&mut self) -> Result<Option<crate::policy::ToolPolicy>, ActonAIError> {
        let rules = match self.tool_policy.take() {
            Some(policy) => Some(policy),
            None => self
                .tool_policy_file
                .as_ref()
                .map(crate::config::ToolPolicyFileConfig::to_tool_policy)
                .transpose()?,
        };

        let hook = self.tool_approval_hook.take();

        match (rules, hook) {
            (Some(mut policy), Some(hook)) => {
                policy.set_hook(hook);
                Ok(Some(policy))
            }
            (Some(policy), None) => Ok(Some(policy)),
            (None, Some(hook)) => {
                let mut policy = crate::policy::ToolPolicy::new();
                policy.set_hook(hook);
                Ok(Some(policy))
            }
            (None, None) => Ok(None),
        }
    }

    /// Resolves the effective audit settings: builder first, then `[audit]`
    /// TOML.
    ///
    /// The parent directory is created here rather than on first append, so a
    /// path nobody can write fails the launch instead of dropping entries
    /// into the void for the rest of the process's life.
    fn resolve_audit(&mut self) -> Result<Option<crate::audit::AuditConfig>, ActonAIError> {
        let resolved = match self.audit.take() {
            Some(config) => Some(config),
            None => match self.audit_file.as_ref() {
                // `enabled = false` is a deployment switching the trail off
                // without deleting the settings it will want back.
                Some(file) if !file.is_enabled() => None,
                Some(file) => Some(file.to_audit()?),
                None => None,
            },
        };

        if let Some(ref config) = resolved {
            // Creates the trail as well as its directory: an armed trail that
            // has recorded nothing must be an empty file, or `audit verify`
            // reports a missing trail while `audit_head()` reports genesis.
            config.ensure_trail_exists()?;
        }

        Ok(resolved)
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

/// Rejects budgets that could not be enforced against the providers as
/// configured.
///
/// Two ways a cap silently stops being a cap, both caught here:
///
/// - it names a provider that was never configured, so nothing is ever
///   measured against it;
/// - a configured provider has no pricing, so part of the spend is invisible
///   and the ceiling can never be reached. [`Budget::allow_unpriced`] is the
///   deliberate opt-out, which counts that usage as `$0`.
///
/// Pure: everything it inspects is an argument.
fn validate_budget_coverage(
    budget: &BudgetConfig,
    providers: &HashMap<String, ProviderConfig>,
    pricing: &PricingTable,
) -> Result<(), ActonAIError> {
    let unknown: Vec<&str> = budget
        .capped_providers()
        .filter(|name| !providers.contains_key(*name))
        .collect();
    if !unknown.is_empty() {
        let mut configured: Vec<&str> = providers.keys().map(String::as_str).collect();
        configured.sort_unstable();
        return Err(ActonAIError::configuration(
            "budget.providers",
            format!(
                "capped provider(s) {} are not configured, so those caps would never apply; \
                 configured providers: {}",
                quoted(&unknown),
                quoted(&configured),
            ),
        ));
    }

    if budget.allow_unpriced() {
        return Ok(());
    }

    let mut unpriced: Vec<&str> = providers
        .keys()
        .map(String::as_str)
        .filter(|name| pricing.get(name).is_none())
        .collect();
    if unpriced.is_empty() {
        return Ok(());
    }
    unpriced.sort_unstable();

    let tables = unpriced
        .iter()
        .map(|name| {
            format!("[providers.{name}.pricing]\ninput_per_mtok = ...\noutput_per_mtok = ...")
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    Err(ActonAIError::configuration(
        "budget",
        format!(
            "a budget is set but provider(s) {} have no pricing, so their spending would be \
             invisible to the cap. Add:\n\n{tables}\n\nor call \
             ActonAIBuilder::pricing(name, ModelPricing::from_dollars_per_mtok(..)), or accept \
             the blind spot with Budget::allow_unpriced() / `allow_unpriced = true` under \
             [budget]",
            quoted(&unpriced),
        ),
    ))
}

/// Rejects failover and circuit-breaker settings that could not do what they
/// claim, before a single actor is spawned.
///
/// Every rule here catches a misconfiguration whose only other symptom is
/// silence on the day the primary provider dies — a chain pointing at a
/// provider nobody configured, a breaker that can never trip, a fallback model
/// identical to the one that just failed. Fail-closed at launch, exactly as
/// [`validate_budget_coverage`] does.
///
/// Pure: takes the configured providers and returns a verdict.
fn validate_failover(providers: &HashMap<String, ProviderConfig>) -> Result<(), ActonAIError> {
    // Deterministic order: a HashMap walk would report a different offender
    // each run for a config with several problems.
    let mut names: Vec<&str> = providers.keys().map(String::as_str).collect();
    names.sort_unstable();

    for name in names {
        let config = &providers[name];
        let field = format!("providers.{name}");

        let breaker = &config.circuit_breaker;
        if breaker.enabled && breaker.failure_threshold == 0 {
            return Err(ActonAIError::configuration(
                format!("{field}.circuit_breaker.failure_threshold"),
                "a threshold of 0 would open the circuit before any request had failed; \
                 use at least 1, or set `enabled = false` to switch the breaker off"
                    .to_string(),
            ));
        }
        if breaker.enabled && breaker.cooldown.is_zero() {
            return Err(ActonAIError::configuration(
                format!("{field}.circuit_breaker.cooldown_secs"),
                "a cooldown of 0 would reopen the circuit for probing immediately, which is \
                 the same as having no breaker at all; use at least 1 second, or set \
                 `enabled = false`"
                    .to_string(),
            ));
        }

        if config.fallback_model.as_deref() == Some(config.model.as_str()) {
            let fallback = &config.model;
            return Err(ActonAIError::configuration(
                format!("{field}.fallback_model"),
                format!(
                    "`{fallback}` is already this provider's model, so degrading to it on a \
                     rate limit would retry the model that was just throttled; name a \
                     different, cheaper model or drop the setting"
                ),
            ));
        }

        let mut seen: Vec<&str> = Vec::new();
        for target in &config.failover {
            if target == name {
                return Err(ActonAIError::configuration(
                    format!("{field}.failover"),
                    format!(
                        "the chain names `{name}` itself, which would re-dispatch to the \
                         provider that just failed; list only other providers"
                    ),
                ));
            }
            if seen.contains(&target.as_str()) {
                return Err(ActonAIError::configuration(
                    format!("{field}.failover"),
                    format!(
                        "`{target}` appears twice in the chain; a second attempt on the same \
                         provider would meet the same open circuit, so remove the duplicate"
                    ),
                ));
            }
            if !providers.contains_key(target.as_str()) {
                let mut configured: Vec<&str> = providers.keys().map(String::as_str).collect();
                configured.sort_unstable();
                return Err(ActonAIError::configuration(
                    format!("{field}.failover"),
                    format!(
                        "`{target}` is not a configured provider, so failing over to it would \
                         be impossible; configured providers: {}",
                        quoted(&configured),
                    ),
                ));
            }
            seen.push(target.as_str());
        }
    }

    Ok(())
}

/// Renders names as a comma-separated, backtick-quoted list.
fn quoted(names: &[&str]) -> String {
    names
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The actors a status report reads from.
///
/// Borrowed as a group rather than passed one by one, because they are all the
/// same thing — "everything already spawned that a status describes" — and a
/// nine-argument function is a place for two of them to be swapped by mistake.
struct IntrospectionCollaborators<'a> {
    providers: &'a HashMap<String, ActorHandle>,
    provider_models: &'a HashMap<String, String>,
    provider_failover: &'a HashMap<String, Vec<String>>,
    accountant: Option<&'a ActorHandle>,
    mcp: Option<&'a crate::mcp::McpTools>,
}

/// Spawns the introspection actor and binds its control socket.
///
/// The socket path is resolved here rather than at config time because the
/// default scheme needs the app name and the PID, neither of which the config
/// layer knows.
async fn start_introspection(
    runtime: &mut ActorRuntime,
    config: &crate::introspection::IntrospectionConfig,
    app_name: &str,
    admission: crate::introspection::AdmissionGate,
    collaborators: IntrospectionCollaborators<'_>,
) -> Result<IntrospectionRuntime, ActonAIError> {
    let IntrospectionCollaborators {
        providers,
        provider_models,
        provider_failover,
        accountant,
        mcp,
    } = collaborators;

    let socket_path = crate::introspection::resolve_socket_path(config, app_name);
    crate::introspection::server::ensure_socket_dir(&socket_path)?;

    #[cfg(feature = "ipc")]
    {
        let sources = crate::introspection::actor::StatusSources {
            providers: providers.clone(),
            provider_models: provider_models.clone(),
            provider_failover: provider_failover.clone(),
            accountant: accountant.cloned(),
            mcp: mcp
                .map(|mcp| {
                    mcp.servers()
                        .map(|(name, child)| (name.clone(), child.clone()))
                        .collect()
                })
                .unwrap_or_default(),
            app_name: app_name.to_string(),
        };

        let handle =
            crate::introspection::IntrospectionActor::spawn(runtime, admission, sources).await;

        let listener = crate::introspection::server::start_listener(
            runtime,
            handle,
            &socket_path,
            config.socket_mode,
        )
        .await?;

        Ok(IntrospectionRuntime {
            socket_path,
            listener: std::sync::Mutex::new(Some(listener)),
        })
    }

    // Unreachable in practice: `resolve_introspection` refuses a configured
    // section without the feature, so no config ever reaches here. Written out
    // rather than `unreachable!()` so a future caller cannot turn a wiring
    // mistake into a panic in someone's production process.
    #[cfg(not(feature = "ipc"))]
    {
        let _ = (
            runtime,
            admission,
            providers,
            provider_models,
            provider_failover,
            accountant,
            mcp,
        );
        Ok(IntrospectionRuntime { socket_path })
    }
}

/// Refuses a launch whose custom tool names collide with anything.
///
/// Pure: reads the resolved built-ins, the skill-tool names, and the MCP
/// tools, and reports the first custom tool whose name is already taken —
/// by one of those, or by an earlier custom tool. Built-ins count even under
/// `manual_builtins()`, because `PromptBuilder::use_builtins()` can put them
/// on any prompt later; a name that *might* be injected is not a name a
/// custom tool can safely hold.
fn validate_custom_tool_names(
    custom_tools: &[crate::prompt::SharedToolSpec],
    builtins: Option<&BuiltinTools>,
    skills_configured: bool,
    mcp: Option<&crate::mcp::McpTools>,
) -> Result<(), ActonAIError> {
    let mut taken: HashMap<String, &'static str> = HashMap::new();
    if let Some(builtins) = builtins {
        for (_, config) in builtins.configs() {
            taken.insert(config.definition.name.clone(), "a built-in tool");
        }
    }
    if skills_configured {
        use crate::tools::builtins::{ActivateSkillTool, ListSkillsTool};
        taken.insert(ListSkillsTool::config().definition.name, "a skill tool");
        taken.insert(ActivateSkillTool::config().definition.name, "a skill tool");
    }
    if let Some(mcp) = mcp {
        for (_, config) in mcp.configs() {
            taken.insert(config.definition.name.clone(), "an MCP tool");
        }
    }

    for spec in custom_tools {
        let name = spec.name();
        if let Some(owner) = taken.get(name) {
            return Err(ActonAIError::configuration(
                "tools",
                format!(
                    "custom tool '{name}' collides with {owner} of the same name; \
                     rename the custom tool or drop the conflicting registration"
                ),
            ));
        }
        taken.insert(name.to_string(), "another custom tool");
    }

    Ok(())
}

/// Flips the gate to draining when `SIGTERM` arrives.
///
/// Holds only the gate, never the runtime: a task holding an `ActonAI` would
/// keep the `Arc` alive for the life of the process and quietly defeat every
/// `shutdown()` the owner performed.
fn spawn_sigterm_drain(admission: crate::introspection::AdmissionGate) {
    #[cfg(unix)]
    tokio::spawn(async move {
        let mut term = match tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        ) {
            Ok(stream) => stream,
            Err(error) => {
                tracing::warn!(%error, "could not install the SIGTERM handler; drain_on_sigterm is inactive");
                return;
            }
        };

        // One shot: a second SIGTERM means the operator has stopped asking
        // politely, and the default disposition — terminate — is the right
        // answer to that. Staying installed would swallow it.
        if term.recv().await.is_some() {
            let state = admission.drain();
            crate::introspection::sd_notify::notify_stopping();
            tracing::info!(%state, "SIGTERM received; no longer admitting new turns");
        }
    });

    #[cfg(not(unix))]
    let _ = admission;
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
        ContextWindow, ContextWindowConfig, TiktokenEstimator, TokenEstimator, TruncationStrategy,
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
    use crate::llm::CircuitBreakerConfig;

    /// Builds a provider map from `(name, config)` pairs.
    fn provider_map(entries: Vec<(&str, ProviderConfig)>) -> HashMap<String, ProviderConfig> {
        entries
            .into_iter()
            .map(|(name, config)| (name.to_string(), config))
            .collect()
    }

    /// The validator's message for a config it should reject.
    fn failover_rejection(providers: &HashMap<String, ProviderConfig>) -> String {
        let err = validate_failover(providers)
            .expect_err("validate_failover accepted a configuration it should have rejected");
        err.to_string()
    }

    #[test]
    fn validate_failover_accepts_providers_with_no_chains() {
        let providers = provider_map(vec![
            ("primary", ProviderConfig::ollama("llama3.2")),
            ("backup", ProviderConfig::ollama("qwen3")),
        ]);
        assert!(validate_failover(&providers).is_ok());
    }

    #[test]
    fn validate_failover_accepts_a_chain_naming_configured_providers() {
        let providers = provider_map(vec![
            (
                "primary",
                ProviderConfig::ollama("llama3.2").with_failover(["backup", "last-resort"]),
            ),
            ("backup", ProviderConfig::ollama("qwen3")),
            ("last-resort", ProviderConfig::ollama("phi4")),
        ]);
        assert!(validate_failover(&providers).is_ok());
    }

    #[test]
    fn validate_failover_rejects_a_chain_naming_an_unconfigured_provider() {
        let providers = provider_map(vec![(
            "primary",
            ProviderConfig::ollama("llama3.2").with_failover(["typo"]),
        )]);
        let message = failover_rejection(&providers);
        assert!(message.contains("providers.primary.failover"), "{message}");
        assert!(message.contains("`typo`"), "{message}");
        // The remedy is only actionable if it says what *is* configured.
        assert!(message.contains("`primary`"), "{message}");
    }

    #[test]
    fn validate_failover_rejects_a_chain_naming_itself() {
        let providers = provider_map(vec![(
            "primary",
            ProviderConfig::ollama("llama3.2").with_failover(["primary"]),
        )]);
        let message = failover_rejection(&providers);
        assert!(message.contains("providers.primary.failover"), "{message}");
        assert!(message.contains("itself"), "{message}");
    }

    #[test]
    fn validate_failover_rejects_a_duplicated_chain_entry() {
        let providers = provider_map(vec![
            (
                "primary",
                ProviderConfig::ollama("llama3.2").with_failover(["backup", "backup"]),
            ),
            ("backup", ProviderConfig::ollama("qwen3")),
        ]);
        let message = failover_rejection(&providers);
        assert!(message.contains("twice"), "{message}");
        assert!(message.contains("`backup`"), "{message}");
    }

    #[test]
    fn validate_failover_rejects_a_zero_failure_threshold() {
        let providers = provider_map(vec![(
            "primary",
            ProviderConfig::ollama("llama3.2").with_circuit_breaker(CircuitBreakerConfig::new(
                0,
                std::time::Duration::from_secs(30),
            )),
        )]);
        let message = failover_rejection(&providers);
        assert!(
            message.contains("providers.primary.circuit_breaker.failure_threshold"),
            "{message}"
        );
    }

    #[test]
    fn validate_failover_rejects_a_zero_cooldown() {
        let providers = provider_map(vec![(
            "primary",
            ProviderConfig::ollama("llama3.2")
                .with_circuit_breaker(CircuitBreakerConfig::new(3, std::time::Duration::ZERO)),
        )]);
        let message = failover_rejection(&providers);
        assert!(
            message.contains("providers.primary.circuit_breaker.cooldown_secs"),
            "{message}"
        );
    }

    #[test]
    fn validate_failover_ignores_breaker_numbers_when_the_breaker_is_off() {
        // A disabled breaker never consults either number, so nonsense in them
        // is inert rather than a launch failure.
        let providers = provider_map(vec![(
            "primary",
            ProviderConfig::ollama("llama3.2")
                .with_circuit_breaker(CircuitBreakerConfig::new(0, std::time::Duration::ZERO))
                .without_circuit_breaker(),
        )]);
        assert!(validate_failover(&providers).is_ok());
    }

    #[test]
    fn validate_failover_rejects_a_fallback_model_equal_to_the_primary_model() {
        let providers = provider_map(vec![(
            "primary",
            ProviderConfig::ollama("llama3.2").with_fallback_model("llama3.2"),
        )]);
        let message = failover_rejection(&providers);
        assert!(
            message.contains("providers.primary.fallback_model"),
            "{message}"
        );
        assert!(message.contains("`llama3.2`"), "{message}");
    }

    #[test]
    fn validate_failover_accepts_a_distinct_fallback_model() {
        let providers = provider_map(vec![(
            "primary",
            ProviderConfig::ollama("llama3.2").with_fallback_model("llama3.2:1b"),
        )]);
        assert!(validate_failover(&providers).is_ok());
    }

    #[test]
    fn validate_failover_reports_the_same_offender_across_runs() {
        // Two broken providers, one HashMap: without a sorted walk the reported
        // offender would depend on hash order and the error would flap.
        let providers = provider_map(vec![
            (
                "alpha",
                ProviderConfig::ollama("llama3.2").with_failover(["nope"]),
            ),
            (
                "zulu",
                ProviderConfig::ollama("qwen3").with_failover(["nope"]),
            ),
        ]);
        let first = failover_rejection(&providers);
        assert!(first.contains("providers.alpha.failover"), "{first}");
        for _ in 0..8 {
            assert_eq!(failover_rejection(&providers), first);
        }
    }

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
            .with_provider("ollama", crate::config::NamedProviderConfig::ollama("test"))
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
            .with_provider("ollama", crate::config::NamedProviderConfig::ollama("test"))
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

    #[test]
    fn a_builder_mcp_server_overrides_the_config_file_entry() {
        use crate::config::McpServerConfig;

        let mut file_config = ActonAIConfig::default();
        file_config
            .mcp_servers
            .insert("fs".to_string(), McpServerConfig::stdio("from-file"));
        file_config
            .mcp_servers
            .insert("other".to_string(), McpServerConfig::stdio("only-in-file"));

        let builder = ActonAI::builder()
            .with_mcp_server("fs", McpServerConfig::stdio("from-builder"))
            .apply_config(file_config)
            .expect("apply_config");

        assert_eq!(builder.mcp_servers.len(), 2);
        assert_eq!(
            builder.mcp_servers["fs"].command.as_deref(),
            Some("from-builder"),
            "an explicit with_mcp_server call must win over the file"
        );
        assert_eq!(
            builder.mcp_servers["other"].command.as_deref(),
            Some("only-in-file")
        );
    }

    #[tokio::test]
    async fn launch_fails_when_an_mcp_server_names_no_transport() {
        use crate::config::McpServerConfig;

        let err = ActonAI::builder()
            .ollama("test")
            .with_mcp_server("broken", McpServerConfig::default())
            .launch()
            .await
            .expect_err("a server with neither command nor url must fail the launch");

        assert!(err.to_string().contains("broken"), "error = {err}");
    }

    #[tokio::test]
    async fn launch_without_mcp_servers_has_no_mcp_tools() {
        let runtime = ActonAI::builder()
            .ollama("test")
            .launch()
            .await
            .expect("launch");

        assert!(runtime.mcp().is_none());
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

    // -------------------------------------------------------------------
    // Budgets
    // -------------------------------------------------------------------

    /// A config whose one provider is priced, which is what a budget needs.
    fn priced_config() -> crate::config::ActonAIConfig {
        let mut provider = crate::config::NamedProviderConfig::ollama("test");
        provider.pricing = Some(crate::config::PricingFileConfig {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
            cache_read_per_mtok: None,
            cache_creation_per_mtok: None,
        });
        crate::config::ActonAIConfig::new()
            .with_provider("claude", provider)
            .with_default_provider("claude")
    }

    // -------------------------------------------------------------------
    // Telemetry
    // -------------------------------------------------------------------

    /// A config carrying a `[telemetry]` section pointing at a nonsense
    /// scheme, so a launch that reaches telemetry resolution must fail and
    /// one that ignores the section would silently succeed.
    #[cfg(feature = "otel")]
    fn config_with_bad_telemetry() -> crate::config::ActonAIConfig {
        crate::config::from_str(
            "[providers.local]\ntype = \"ollama\"\nmodel = \"test\"\n\
             [telemetry]\notlp_endpoint = \"grpc://localhost:4317\"\n",
        )
        .expect("the section parses even when it cannot be honoured")
    }

    #[cfg(feature = "otel")]
    #[tokio::test]
    async fn launch_fails_when_a_telemetry_endpoint_is_not_http() {
        let err = ActonAI::builder()
            .apply_config(config_with_bad_telemetry())
            .expect("apply_config")
            .launch()
            .await
            .expect_err("an OTLP/HTTP exporter cannot use a grpc endpoint");

        assert!(err.is_configuration(), "err = {err}");
        assert!(
            err.to_string().contains("telemetry.otlp_endpoint"),
            "err = {err}"
        );
    }

    #[cfg(feature = "otel")]
    #[tokio::test]
    async fn a_builder_telemetry_call_replaces_the_toml_section_wholesale() {
        // The TOML section is unusable. If the builder's telemetry replaced it
        // only field-by-field, the bad endpoint would survive and the launch
        // would fail — so a clean launch is the proof of wholesale precedence.
        let ai = ActonAI::builder()
            .apply_config(config_with_bad_telemetry())
            .expect("apply_config")
            .telemetry_from_globals()
            .launch()
            .await
            .expect("the builder's telemetry replaces the section entirely");

        assert!(format!("{ai:?}").contains("telemetry: true"));
    }

    /// Without the `otel` feature, a configured `[telemetry]` section must
    /// stop the launch and say why.
    ///
    /// This is the outcome the whole feature-off code path exists to produce:
    /// an operator who configured telemetry into a binary that cannot export
    /// it needs to be told, not left watching an empty collector.
    #[cfg(not(feature = "otel"))]
    #[tokio::test]
    async fn launch_fails_when_telemetry_is_configured_without_the_otel_feature() {
        let config = crate::config::from_str(
            "[providers.local]\ntype = \"ollama\"\nmodel = \"test\"\n\
             [telemetry]\notlp_endpoint = \"http://localhost:4318\"\n",
        )
        .expect("the section must parse even without the feature");

        let err = ActonAI::builder()
            .apply_config(config)
            .expect("apply_config")
            .launch()
            .await
            .expect_err("telemetry that cannot be exported must not be ignored");

        assert!(err.is_configuration(), "err = {err}");
        let rendered = err.to_string();
        assert!(rendered.contains("otel"), "err = {rendered}");
        assert!(
            rendered.contains("built without"),
            "the error must name the missing feature: {rendered}"
        );
    }

    /// A config file with a provider and nothing else.
    fn plain_config() -> crate::config::ActonAIConfig {
        crate::config::from_str("[providers.local]\ntype = \"ollama\"\nmodel = \"test\"\n")
            .expect("a minimal provider config parses")
    }

    #[tokio::test]
    async fn a_runtime_admits_turns_until_something_says_otherwise() {
        let ai = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        // The default has to be Running: a runtime that launched paused would
        // be a library that silently does nothing.
        assert_eq!(
            ai.admission_state(),
            crate::introspection::AdmissionState::Running
        );
        assert!(ai.admission_state().admits());

        ai.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn pause_and_resume_move_the_gate_without_a_socket() {
        // No `.introspection()` call anywhere: admission control is in-process
        // state, and an embedder with its own control plane must get it
        // without opening a Unix socket.
        let ai = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        assert_eq!(ai.pause(), crate::introspection::AdmissionState::Paused);
        assert!(!ai.admission_state().admits());

        assert_eq!(ai.drain(), crate::introspection::AdmissionState::Draining);
        assert!(!ai.admission_state().admits());

        // Resume lifts a drain too, for the operator who changed their mind.
        assert_eq!(ai.resume(), crate::introspection::AdmissionState::Running);
        assert!(ai.admission_state().admits());

        ai.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn a_paused_runtime_refuses_a_turn_before_dispatching_it() {
        // The provider is Ollama on its default port, which is not running in
        // CI. That is the point: if the refusal did not happen first, this
        // would fail with a connection error instead.
        let ai = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        ai.pause();

        let err = ai
            .prompt("this must never be sent")
            .collect()
            .await
            .expect_err("a paused runtime admits nothing");

        assert!(err.is_turns_not_admitted(), "err = {err}");
        let rendered = err.to_string();
        assert!(rendered.contains("paused"), "err = {rendered}");
        assert!(rendered.contains("acton-ai resume"), "err = {rendered}");

        ai.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn a_resumed_runtime_dispatches_again() {
        let ai = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        ai.pause();
        ai.resume();

        let err = ai
            .prompt("this is allowed to fail, but not by refusal")
            .collect()
            .await
            .expect_err("no Ollama is listening in a test run");

        // The distinction is the whole point: after a resume the turn is
        // admitted and fails on its merits, rather than being turned away.
        assert!(!err.is_turns_not_admitted(), "err = {err}");

        ai.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn no_socket_is_created_unless_introspection_is_configured() {
        let ai = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        // Compiling the `ipc` feature in must not open a control socket. A
        // library that listens by default is one nobody can safely embed.
        assert!(ai.introspection_socket().is_none());

        ai.shutdown().await.expect("shutdown");
    }

    #[cfg(feature = "ipc")]
    #[tokio::test]
    async fn an_introspection_socket_is_bound_owner_only_and_removed_on_shutdown() {
        let dir = tempfile::tempdir().expect("temp dir");
        let socket = dir.path().join("control.sock");

        let ai = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .introspection_at(&socket)
            .launch()
            .await
            .expect("launch");

        assert_eq!(ai.introspection_socket(), Some(socket.as_path()));
        assert!(
            socket.exists(),
            "the socket file must exist while listening"
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&socket)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777;
            // The permission bits are the access control, since the IPC layer
            // exposes no peer-credential hook. A regression here is a real
            // widening.
            assert_eq!(mode, crate::introspection::SOCKET_MODE, "{mode:#o}");
        }

        ai.shutdown().await.expect("shutdown");

        // A socket left behind would make the next launch treat a dead address
        // as possibly-live.
        assert!(!socket.exists(), "the socket must be gone after shutdown");

        // The operator-facing consequence, and the reason the file matters:
        // restarting in place at the same address has to work, immediately and
        // without a manual `rm`.
        let restarted = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .introspection_at(&socket)
            .launch()
            .await
            .expect("the same socket path is free once the first runtime is down");

        assert!(socket.exists());
        restarted.shutdown().await.expect("shutdown");
    }

    #[cfg(feature = "ipc")]
    #[tokio::test]
    async fn a_second_runtime_cannot_take_over_a_live_socket() {
        let dir = tempfile::tempdir().expect("temp dir");
        let socket = dir.path().join("contested.sock");

        let first = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .introspection_at(&socket)
            .launch()
            .await
            .expect("the first runtime binds the socket");

        let err = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .introspection_at(&socket)
            .launch()
            .await
            .expect_err("two runtimes must not share one control address");

        assert!(err.is_configuration(), "err = {err}");
        // Naming the path is the difference between a five-second fix and a
        // long afternoon: the operator has to know *which* socket.
        assert!(
            err.to_string().contains(&socket.display().to_string()),
            "err = {err}"
        );

        first.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn a_relative_introspection_socket_path_fails_the_launch() {
        let err = ActonAI::builder()
            .apply_config(plain_config())
            .expect("apply_config")
            .introspection_at("control.sock")
            .launch()
            .await
            .expect_err("a socket that moves with the cwd is unfindable");

        assert!(err.is_configuration(), "err = {err}");
        assert!(err.to_string().contains("absolute"), "err = {err}");
    }

    #[tokio::test]
    async fn a_disabled_introspection_section_listens_and_fails_at_nothing() {
        // `enabled = false` must be inert in *every* build, including one
        // without the `ipc` feature: nothing was going to listen, so there is
        // nothing to refuse over.
        let config = crate::config::from_str(
            "[providers.local]\ntype = \"ollama\"\nmodel = \"test\"\n\
             [introspection]\nenabled = false\nsocket_path = \"/nonexistent/dir/x.sock\"\n",
        )
        .expect("the section parses");

        let ai = ActonAI::builder()
            .apply_config(config)
            .expect("apply_config")
            .launch()
            .await
            .expect("a disabled section is not a launch failure");

        assert!(ai.introspection_socket().is_none());

        ai.shutdown().await.expect("shutdown");
    }

    #[cfg(feature = "ipc")]
    #[tokio::test]
    async fn a_builder_introspection_call_replaces_the_toml_section_wholesale() {
        // The TOML section names a socket mode that would be refused. If the
        // builder's call merged field-by-field instead of replacing, that mode
        // would survive and the launch would fail — so a clean launch at the
        // builder's own path is the proof.
        let config = crate::config::from_str(
            "[providers.local]\ntype = \"ollama\"\nmodel = \"test\"\n\
             [introspection]\nsocket_mode = 0o666\nsocket_path = \"/nonexistent/dir/x.sock\"\n",
        )
        .expect("the section parses even when it cannot be honoured");

        let dir = tempfile::tempdir().expect("temp dir");
        let socket = dir.path().join("builder-wins.sock");

        let ai = ActonAI::builder()
            .apply_config(config)
            .expect("apply_config")
            .introspection_at(&socket)
            .launch()
            .await
            .expect("the builder's introspection replaces the section entirely");

        assert_eq!(ai.introspection_socket(), Some(socket.as_path()));

        ai.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn a_world_writable_socket_mode_in_toml_fails_the_launch() {
        let config = crate::config::from_str(
            "[providers.local]\ntype = \"ollama\"\nmodel = \"test\"\n\
             [introspection]\nsocket_mode = 0o666\n",
        )
        .expect("the section parses");

        let err = ActonAI::builder()
            .apply_config(config)
            .expect("apply_config")
            .launch()
            .await
            .expect_err("a control socket anyone can write to is refused");

        assert!(err.is_configuration(), "err = {err}");
        assert!(err.to_string().contains("pause"), "err = {err}");
    }

    /// Without the `ipc` feature, a configured `[introspection]` section must
    /// stop the launch and say why.
    ///
    /// The mirror of the telemetry case, and for the same reason: an operator
    /// whose `acton-ai status` finds nothing needs to be told the binary
    /// cannot listen, not left debugging the socket.
    #[cfg(not(feature = "ipc"))]
    #[tokio::test]
    async fn launch_fails_when_introspection_is_configured_without_the_ipc_feature() {
        let config = crate::config::from_str(
            "[providers.local]\ntype = \"ollama\"\nmodel = \"test\"\n\
             [introspection]\nsocket_path = \"/run/agent.sock\"\n",
        )
        .expect("the section must parse even without the feature");

        let err = ActonAI::builder()
            .apply_config(config)
            .expect("apply_config")
            .launch()
            .await
            .expect_err("a socket that cannot be created must not be ignored");

        assert!(err.is_configuration(), "err = {err}");
        let rendered = err.to_string();
        assert!(rendered.contains("ipc"), "err = {rendered}");
        assert!(
            rendered.contains("built without"),
            "the error must name the missing feature: {rendered}"
        );
    }

    #[tokio::test]
    async fn launch_fails_when_a_budget_has_no_accountant_to_count_for_it() {
        let err = ActonAI::builder()
            .apply_config(priced_config())
            .expect("apply_config")
            .budget_usd(5.00)
            .usage_tracking(false)
            .launch()
            .await
            .expect_err("a budget with nothing counting is not a budget");

        assert!(err.is_configuration(), "err = {err}");
        assert!(err.to_string().contains("usage tracking"), "err = {err}");
    }

    #[tokio::test]
    async fn launch_fails_when_a_budget_covers_an_unpriced_provider() {
        // `ollama("test")` carries no pricing, so its spending would be
        // invisible to the cap.
        let err = ActonAI::builder()
            .ollama("test")
            .budget_usd(5.00)
            .launch()
            .await
            .expect_err("an unpriced provider under a budget must fail closed");

        let message = err.to_string();
        assert!(err.is_configuration(), "err = {err}");
        assert!(
            message.contains(DEFAULT_PROVIDER_NAME),
            "the error must name the offending provider: {message}"
        );
        assert!(
            message.contains("pricing"),
            "the error must name the TOML table to add: {message}"
        );
        assert!(
            message.contains("allow_unpriced"),
            "the error must name the opt-out: {message}"
        );
    }

    #[tokio::test]
    async fn builder_pricing_is_enough_to_launch_a_budget_without_a_config_file() {
        // The all-in-code path: without `pricing()` there would be no way to
        // price a provider outside TOML, and every programmatic budget would
        // fail the launch.
        let ai = ActonAI::builder()
            .ollama("test")
            .pricing(
                DEFAULT_PROVIDER_NAME,
                crate::accounting::ModelPricing::from_dollars_per_mtok(3.0, 15.0),
            )
            .budget_usd(5.00)
            .launch()
            .await
            .expect("a builder-priced provider satisfies the budget's pricing requirement");

        assert!(ai.is_budget_enforced());
        ai.shutdown().await.expect("clean shutdown");
    }

    #[tokio::test]
    async fn allow_unpriced_launches_over_an_unpriced_provider() {
        let ai = ActonAI::builder()
            .ollama("test")
            .budget(Budget::usd(5.00).allow_unpriced())
            .launch()
            .await
            .expect("an explicit blind spot is allowed");

        assert!(ai.is_budget_enforced());
        ai.shutdown().await.expect("clean shutdown");
    }

    #[tokio::test]
    async fn the_budget_usd_one_liner_launches_over_a_priced_provider() {
        let ai = ActonAI::builder()
            .apply_config(priced_config())
            .expect("apply_config")
            .budget_usd(5.00)
            .launch()
            .await
            .expect("a priced provider under a cap launches");

        assert!(ai.is_budget_enforced());
        let usage = ai.usage().await.expect("usage");
        let budget = usage.budget.expect("the snapshot must carry the cap");
        assert_eq!(budget.total.expect("a total cap").limit_microusd, 5_000_000);

        ai.shutdown().await.expect("clean shutdown");
    }

    #[tokio::test]
    async fn launching_without_a_budget_enforces_nothing() {
        let ai = ActonAI::builder()
            .ollama("test")
            .launch()
            .await
            .expect("launch");

        assert!(!ai.is_budget_enforced());
        assert!(
            ai.usage().await.expect("usage").budget.is_none(),
            "no budget must read as absent, not as one with room left"
        );

        ai.shutdown().await.expect("clean shutdown");
    }

    #[tokio::test]
    async fn launch_fails_when_a_cap_names_a_provider_that_was_never_configured() {
        // Otherwise the typo is a cap that silently never applies.
        let err = ActonAI::builder()
            .apply_config(priced_config())
            .expect("apply_config")
            .budget(Budget::for_provider("cluade", 2.00))
            .launch()
            .await
            .expect_err("a cap on a provider that does not exist must fail");

        let message = err.to_string();
        assert!(message.contains("cluade"), "err = {message}");
        assert!(message.contains("claude"), "err = {message}");
    }

    #[tokio::test]
    async fn a_builder_budget_replaces_the_toml_section_wholesale() {
        let config = crate::config::ActonAIConfig {
            budget: Some(crate::config::BudgetFileConfig {
                total_usd: Some(99.0),
                warn_at_percent: Some(10),
                allow_unpriced: Some(true),
                providers: Default::default(),
            }),
            ..priced_config()
        };

        // Applied after the builder call, to prove the precedence does not
        // depend on the order the two are written in.
        let ai = ActonAI::builder()
            .budget(Budget::usd(1.00))
            .apply_config(config)
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        let status = ai
            .usage()
            .await
            .expect("usage")
            .budget
            .expect("a budget is configured");
        assert_eq!(status.total.expect("a total cap").limit_microusd, 1_000_000);
        assert_eq!(
            status.warn_at_percent,
            crate::accounting::DEFAULT_WARN_AT_PERCENT,
            "the builder budget replaces the section, it does not merge with it"
        );
        assert!(!status.allow_unpriced);

        ai.shutdown().await.expect("clean shutdown");
    }

    #[tokio::test]
    async fn the_toml_budget_applies_when_the_builder_sets_none() {
        let config = crate::config::ActonAIConfig {
            budget: Some(crate::config::BudgetFileConfig {
                total_usd: Some(7.0),
                warn_at_percent: None,
                allow_unpriced: None,
                providers: Default::default(),
            }),
            ..priced_config()
        };

        let ai = ActonAI::builder()
            .apply_config(config)
            .expect("apply_config")
            .launch()
            .await
            .expect("launch");

        let status = ai.usage().await.expect("usage").budget.expect("a budget");
        assert_eq!(status.total.expect("a total cap").limit_microusd, 7_000_000);

        ai.shutdown().await.expect("clean shutdown");
    }

    #[test]
    fn an_invalid_budget_is_rejected_before_anything_is_spawned() {
        let builder = ActonAI::builder()
            .ollama("test")
            .budget(Budget::usd(5.0).warn_at_percent(200));

        assert!(builder.resolve_budget().is_err());
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
            .with_provider("ollama", crate::config::NamedProviderConfig::ollama("test"))
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
