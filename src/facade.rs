//! High-level facade for ActonAI.
//!
//! This module provides a simplified API for common use cases, hiding the
//! complexity of actor setup, kernel spawning, and provider configuration.
//!
//! # Example
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

use crate::error::{ActonAIError, ActonAIErrorKind};
use crate::kernel::{Kernel, KernelConfig};
use crate::llm::{LLMProvider, ProviderConfig};
use crate::prompt::PromptBuilder;
use acton_reactive::prelude::*;

/// High-level facade for interacting with ActonAI.
///
/// `ActonAI` encapsulates the runtime, kernel, and LLM provider, providing
/// a simplified API for common operations. It handles all the actor setup
/// and subscription management automatically.
///
/// # Example
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
pub struct ActonAI {
    /// The underlying actor runtime
    runtime: ActorRuntime,
    /// Handle to the LLM provider actor
    provider_handle: ActorHandle,
    /// Whether the runtime has been shut down
    is_shutdown: bool,
}

impl std::fmt::Debug for ActonAI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActonAI")
            .field("is_shutdown", &self.is_shutdown)
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
        PromptBuilder::new(self, content.into())
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

    /// Returns a clone of the LLM provider handle.
    ///
    /// This can be used to send requests directly to the provider
    /// for advanced use cases.
    #[must_use]
    pub fn provider_handle(&self) -> ActorHandle {
        self.provider_handle.clone()
    }

    /// Returns whether the runtime has been shut down.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown
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

/// Builder for configuring and launching ActonAI.
///
/// # Example
///
/// ```rust,ignore
/// let runtime = ActonAI::builder()
///     .app_name("my-chat-app")
///     .ollama("qwen2.5:7b")
///     .launch()
///     .await?;
/// ```
#[derive(Default)]
pub struct ActonAIBuilder {
    app_name: Option<String>,
    provider_config: Option<ProviderConfig>,
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

    /// Configures for Ollama with the specified model.
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
    pub fn ollama(mut self, model: impl Into<String>) -> Self {
        self.provider_config = Some(ProviderConfig::ollama(model));
        self
    }

    /// Configures for Ollama with a custom URL and model.
    ///
    /// Use this when Ollama is running on a non-default address.
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
    pub fn ollama_at(mut self, base_url: impl Into<String>, model: impl Into<String>) -> Self {
        self.provider_config = Some(ProviderConfig::openai_compatible(base_url, model));
        self
    }

    /// Configures for Anthropic Claude with the specified API key.
    ///
    /// Uses the default Claude model (claude-sonnet-4-20250514).
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
    pub fn anthropic(mut self, api_key: impl Into<String>) -> Self {
        self.provider_config = Some(ProviderConfig::anthropic(api_key));
        self
    }

    /// Configures for Anthropic Claude with a specific model.
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
    pub fn anthropic_model(mut self, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        self.provider_config = Some(ProviderConfig::anthropic(api_key).with_model(model));
        self
    }

    /// Configures for OpenAI with the specified API key.
    ///
    /// Uses the default GPT model (gpt-4o).
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
    pub fn openai(mut self, api_key: impl Into<String>) -> Self {
        self.provider_config = Some(ProviderConfig::openai(api_key));
        self
    }

    /// Configures for OpenAI with a specific model.
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
    pub fn openai_model(mut self, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        self.provider_config = Some(ProviderConfig::openai(api_key).with_model(model));
        self
    }

    /// Sets a custom provider configuration.
    ///
    /// Use this for advanced configuration or custom OpenAI-compatible providers.
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
    pub fn provider(mut self, config: ProviderConfig) -> Self {
        self.provider_config = Some(config);
        self
    }

    /// Launches the ActonAI runtime with the configured settings.
    ///
    /// This spawns the actor runtime, kernel, and LLM provider.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No provider is configured
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
        // Validate configuration
        let provider_config = self.provider_config.ok_or_else(|| {
            ActonAIError::new(ActonAIErrorKind::Configuration {
                field: "provider".to_string(),
                reason:
                    "no LLM provider configured; use ollama(), anthropic(), openai(), or provider()"
                        .to_string(),
            })
        })?;

        let app_name = self.app_name.unwrap_or_else(|| "acton-ai".to_string());

        // Launch the actor runtime
        let mut runtime = ActonApp::launch_async().await;

        // Spawn the kernel with the app name for logging
        let kernel_config = KernelConfig::default().with_app_name(&app_name);
        let _kernel = Kernel::spawn_with_config(&mut runtime, kernel_config).await;

        // Spawn the LLM provider
        let provider_handle = LLMProvider::spawn(&mut runtime, provider_config).await;

        Ok(ActonAI {
            runtime,
            provider_handle,
            is_shutdown: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_default_has_no_provider() {
        let builder = ActonAIBuilder::default();
        assert!(builder.provider_config.is_none());
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
        assert!(builder.provider_config.is_some());

        let config = builder.provider_config.unwrap();
        assert_eq!(config.model, "llama3.2");
        assert!(config.api_key.is_empty());
    }

    #[test]
    fn builder_ollama_at_sets_custom_url() {
        let builder = ActonAI::builder().ollama_at("http://custom:11434/v1", "llama3.2");
        assert!(builder.provider_config.is_some());

        let config = builder.provider_config.unwrap();
        assert_eq!(config.model, "llama3.2");
        assert_eq!(config.base_url, "http://custom:11434/v1");
    }

    #[test]
    fn builder_anthropic_sets_provider() {
        let builder = ActonAI::builder().anthropic("sk-ant-test");
        assert!(builder.provider_config.is_some());

        let config = builder.provider_config.unwrap();
        assert_eq!(config.api_key, "sk-ant-test");
        assert!(config.model.contains("claude"));
    }

    #[test]
    fn builder_anthropic_model_sets_custom_model() {
        let builder = ActonAI::builder().anthropic_model("sk-ant-test", "claude-3-haiku");
        assert!(builder.provider_config.is_some());

        let config = builder.provider_config.unwrap();
        assert_eq!(config.api_key, "sk-ant-test");
        assert_eq!(config.model, "claude-3-haiku");
    }

    #[test]
    fn builder_openai_sets_provider() {
        let builder = ActonAI::builder().openai("sk-test");
        assert!(builder.provider_config.is_some());

        let config = builder.provider_config.unwrap();
        assert_eq!(config.api_key, "sk-test");
        assert!(config.model.contains("gpt"));
    }

    #[test]
    fn builder_openai_model_sets_custom_model() {
        let builder = ActonAI::builder().openai_model("sk-test", "gpt-4-turbo");
        assert!(builder.provider_config.is_some());

        let config = builder.provider_config.unwrap();
        assert_eq!(config.api_key, "sk-test");
        assert_eq!(config.model, "gpt-4-turbo");
    }

    #[test]
    fn builder_provider_sets_custom_config() {
        let custom_config =
            ProviderConfig::openai_compatible("http://custom:8080/v1", "custom-model");
        let builder = ActonAI::builder().provider(custom_config);
        assert!(builder.provider_config.is_some());

        let config = builder.provider_config.unwrap();
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
}
