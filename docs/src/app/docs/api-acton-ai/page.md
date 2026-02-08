---
title: ActonAI & ActonAIBuilder
---

Complete API reference for the `ActonAI` runtime handle and the `ActonAIBuilder` used to configure and launch it.

---

## Constants

### `DEFAULT_PROVIDER_NAME`

```rust
pub const DEFAULT_PROVIDER_NAME: &str = "default";
```

The name assigned to providers registered via the single-provider convenience methods (`ollama()`, `anthropic()`, `openai()`, `provider()`). When only one provider is registered, it becomes the default automatically.

---

## ActonAI

```rust
pub struct ActonAI { /* fields omitted */ }
```

High-level facade for interacting with the ActonAI runtime. `ActonAI` encapsulates the actor runtime, kernel, and LLM providers, providing a simplified API for common operations. It handles actor setup and subscription management automatically.

`ActonAI` is **cheaply cloneable** via internal `Arc` -- cloning shares the same underlying runtime, providers, and configuration. All methods take `&self`.

### Creating an instance

#### `builder()`

```rust
pub fn builder() -> ActonAIBuilder
```

Creates a new builder for configuring ActonAI. This is the entry point for all ActonAI setup.

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("llama3.2")
    .launch()
    .await?;
```

### Sending prompts

#### `prompt()`

```rust
pub fn prompt(&self, content: impl Into<String>) -> PromptBuilder
```

Creates a [`PromptBuilder`](/docs/api-prompt-builder) for sending a message to the LLM. If built-in tools were configured with `with_builtins()` or `with_builtin_tools()`, they are automatically added to the prompt unless `manual_builtins()` was called.

```rust
runtime
    .prompt("What is 2 + 2?")
    .system("Be concise.")
    .on_token(|t| print!("{t}"))
    .collect()
    .await?;
```

#### `continue_with()`

```rust
pub fn continue_with(&self, messages: impl IntoIterator<Item = Message>) -> PromptBuilder
```

Continues a conversation from existing messages. This is a clearer alternative to `.prompt("").messages(...)` when you want to continue a conversation without adding a new user message. The provided messages become the conversation history.

```rust
let history = vec![
    Message::user("What is Rust?"),
    Message::assistant("Rust is a systems programming language..."),
    Message::user("How does ownership work?"),
];

let response = runtime
    .continue_with(history)
    .system("Be concise.")
    .on_token(|t| print!("{t}"))
    .collect()
    .await?;
```

### Conversations

#### `conversation()`

```rust
pub fn conversation(&self) -> ConversationBuilder
```

Starts a managed conversation session. Returns a [`ConversationBuilder`](/docs/api-conversation) that can be used to configure and create a `Conversation` with automatic history management.

```rust
let conv = runtime.conversation()
    .system("You are a helpful assistant.")
    .build()
    .await;

let response = conv.send("What is Rust?").await?;
println!("{}", response.text);
```

### Provider management

#### `provider_handle()`

```rust
pub fn provider_handle(&self) -> ActorHandle
```

Returns a clone of the default LLM provider's actor handle. Useful for sending requests directly to the provider in advanced use cases.

#### `provider_handle_named()`

```rust
pub fn provider_handle_named(&self, name: &str) -> Option<ActorHandle>
```

Returns a clone of a named LLM provider handle. Returns `None` if no provider with the given name exists.

#### `default_provider_name()`

```rust
pub fn default_provider_name(&self) -> &str
```

Returns the name of the default provider.

#### `provider_names()`

```rust
pub fn provider_names(&self) -> impl Iterator<Item = &str>
```

Returns an iterator over the names of all registered providers.

#### `provider_count()`

```rust
pub fn provider_count(&self) -> usize
```

Returns the number of registered providers.

#### `has_provider()`

```rust
pub fn has_provider(&self, name: &str) -> bool
```

Returns `true` if a provider with the given name exists.

### Built-in tools

#### `builtins()`

```rust
pub fn builtins(&self) -> Option<&BuiltinTools>
```

Returns a reference to the built-in tools, if enabled. Returns `None` if built-in tools were not configured. See [`BuiltinTools`](/docs/api-builtin-tools) for details.

#### `has_builtins()`

```rust
pub fn has_builtins(&self) -> bool
```

Returns whether built-in tools are enabled.

#### `is_auto_builtins()`

```rust
pub fn is_auto_builtins(&self) -> bool
```

Returns whether builtins are automatically enabled on each prompt. When `true`, `prompt()`, `continue_with()`, and `conversation()` automatically add builtins without requiring `use_builtins()`.

### Runtime access

#### `runtime()`

```rust
pub fn runtime(&self) -> &ActorRuntime
```

Returns a reference to the underlying actor runtime. This is an escape hatch for advanced use cases that need direct access to the actor system.

#### `runtime_mut()`

```rust
pub fn runtime_mut(&mut self) -> &mut ActorRuntime
```

Returns a mutable reference to the underlying actor runtime. **Panics** if there are other clones of this `ActonAI` handle.

### Lifecycle

#### `is_shutdown()`

```rust
pub fn is_shutdown(&self) -> bool
```

Returns whether the runtime has been shut down.

#### `shutdown()`

```rust
pub async fn shutdown(self) -> Result<(), ActonAIError>
```

Shuts down the runtime gracefully, stopping all actors and releasing resources. Consumes the `ActonAI` instance.

```rust
runtime.shutdown().await?;
```

---

## ActonAIBuilder

```rust
pub struct ActonAIBuilder { /* fields omitted */ }
```

Builder for configuring and launching ActonAI. Created via `ActonAI::builder()`.

### Application settings

#### `app_name()`

```rust
pub fn app_name(self, name: impl Into<String>) -> Self
```

Sets the application name for logging and identification.

### Provider configuration -- single provider

These convenience methods register a provider under the name `"default"`. For multi-provider setups, use `provider_named()` instead.

#### `ollama()`

```rust
pub fn ollama(self, model: impl Into<String>) -> Self
```

Configures for Ollama with the specified model. Ollama runs locally and does not require an API key.

```rust
ActonAI::builder()
    .app_name("my-app")
    .ollama("llama3.2")
    .launch()
    .await?;
```

#### `ollama_at()`

```rust
pub fn ollama_at(self, base_url: impl Into<String>, model: impl Into<String>) -> Self
```

Configures for Ollama at a custom URL.

```rust
ActonAI::builder()
    .ollama_at("http://192.168.1.100:11434/v1", "llama3.2")
    .launch()
    .await?;
```

#### `anthropic()`

```rust
pub fn anthropic(self, api_key: impl Into<String>) -> Self
```

Configures for Anthropic Claude with the specified API key and the default Claude model.

#### `anthropic_model()`

```rust
pub fn anthropic_model(self, api_key: impl Into<String>, model: impl Into<String>) -> Self
```

Configures for Anthropic Claude with a specific model.

```rust
ActonAI::builder()
    .anthropic_model("sk-ant-...", "claude-3-haiku-20240307")
    .launch()
    .await?;
```

#### `openai()`

```rust
pub fn openai(self, api_key: impl Into<String>) -> Self
```

Configures for OpenAI with the specified API key and the default GPT model.

#### `openai_model()`

```rust
pub fn openai_model(self, api_key: impl Into<String>, model: impl Into<String>) -> Self
```

Configures for OpenAI with a specific model.

#### `provider()`

```rust
pub fn provider(self, config: ProviderConfig) -> Self
```

Sets a custom provider configuration. Use this for advanced configuration or custom OpenAI-compatible providers.

```rust
let config = ProviderConfig::openai_compatible("http://localhost:8080/v1", "my-model")
    .with_timeout(Duration::from_secs(60));

ActonAI::builder()
    .provider(config)
    .launch()
    .await?;
```

### Provider configuration -- multi-provider

#### `provider_named()`

```rust
pub fn provider_named(self, name: impl Into<String>, config: ProviderConfig) -> Self
```

Registers a named provider configuration. Multiple providers can be configured, each with a unique name. Use `default_provider()` to set which one is used when none is specified on a prompt.

```rust
ActonAI::builder()
    .provider_named("claude", ProviderConfig::anthropic("sk-ant-..."))
    .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
    .default_provider("local")
    .launch()
    .await?;
```

#### `default_provider()`

```rust
pub fn default_provider(self, name: impl Into<String>) -> Self
```

Sets the name of the default provider. The default provider is used when no provider is specified on a prompt. If not set and only one provider exists, that provider becomes the default automatically.

### Configuration files

#### `from_config()`

```rust
pub fn from_config(self) -> Result<Self, ActonAIError>
```

Loads provider configurations from a config file. Searches in order:

1. `./acton-ai.toml` (project-local)
2. `~/.config/acton-ai/config.toml` (XDG config)

If no config file is found, this is a no-op. Providers loaded from config are merged with any already registered.

```rust
ActonAI::builder()
    .app_name("my-app")
    .from_config()?
    .launch()
    .await?;
```

#### `from_config_file()`

```rust
pub fn from_config_file(self, path: impl AsRef<Path>) -> Result<Self, ActonAIError>
```

Loads provider configurations from a specific file path.

#### `try_from_config()`

```rust
pub fn try_from_config(self) -> Result<Self, ActonAIError>
```

Attempts to load from a config file, ignoring errors if no config exists. Parse errors are still returned.

### Built-in tools configuration

#### `with_builtins()`

```rust
pub fn with_builtins(self) -> Self
```

Enables all [built-in tools](/docs/api-builtin-tools) with automatic enabling on each prompt. You do **not** need to call `use_builtins()` on each prompt.

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .with_builtins()
    .launch()
    .await?;

// Builtins are automatically available
runtime.prompt("List files in the current directory").collect().await?;
```

#### `with_builtin_tools()`

```rust
pub fn with_builtin_tools(self, tools: &[&str]) -> Self
```

Enables specific built-in tools by name with automatic enabling on each prompt.

```rust
ActonAI::builder()
    .ollama("qwen2.5:7b")
    .with_builtin_tools(&["read_file", "write_file", "glob"])
    .launch()
    .await?;
```

#### `manual_builtins()`

```rust
pub fn manual_builtins(self) -> Self
```

Disables auto-enabling of builtins on each prompt. Call this after `with_builtins()` or `with_builtin_tools()` to require explicit `use_builtins()` calls on individual prompts.

```rust
let runtime = ActonAI::builder()
    .ollama("qwen2.5:7b")
    .with_builtins()
    .manual_builtins()
    .launch()
    .await?;

// Must explicitly enable builtins per prompt
runtime.prompt("List files").use_builtins().collect().await?;
```

### Sandbox configuration

#### `with_hyperlight_sandbox()`

```rust
pub fn with_hyperlight_sandbox(self) -> Self
```

Enables Hyperlight sandbox for hardware-isolated tool execution with default settings. Requires a hypervisor (KVM on Linux, Hyper-V on Windows).

#### `with_hyperlight_sandbox_config()`

```rust
pub fn with_hyperlight_sandbox_config(self, config: SandboxConfig) -> Self
```

Enables Hyperlight sandbox with custom configuration.

#### `with_sandbox_pool()`

```rust
pub fn with_sandbox_pool(self, pool_size: usize) -> Self
```

Enables a pool of pre-warmed Hyperlight sandboxes. Recommended for high-throughput scenarios.

```rust
ActonAI::builder()
    .ollama("qwen2.5:7b")
    .with_sandbox_pool(4)  // Keep 4 sandboxes warm
    .launch()
    .await?;
```

#### `with_sandbox_pool_config()`

```rust
pub fn with_sandbox_pool_config(self, pool_size: usize, sandbox_config: SandboxConfig) -> Self
```

Enables a pool of pre-warmed Hyperlight sandboxes with custom configuration.

### Launch

#### `launch()`

```rust
pub async fn launch(self) -> Result<ActonAI, ActonAIError>
```

Launches the ActonAI runtime with the configured settings. This spawns the actor runtime, kernel, and all LLM providers.

**Errors:**

- No provider is configured
- Default provider is specified but does not exist
- Multiple providers exist but no default is specified
- The runtime fails to launch

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("llama3.2")
    .launch()
    .await?;
```

{% callout type="warning" title="At least one provider required" %}
`launch()` returns an error if no LLM provider has been configured. Use at least one of: `ollama()`, `anthropic()`, `openai()`, `provider()`, `provider_named()`, or `from_config()`.
{% /callout %}

---

## Common patterns

### Minimal setup with Ollama

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .launch()
    .await?;

runtime.prompt("Hello!").on_token(|t| print!("{t}")).collect().await?;
```

### Multi-provider with tools

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .provider_named("claude", ProviderConfig::anthropic("sk-ant-..."))
    .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
    .default_provider("local")
    .with_builtins()
    .launch()
    .await?;

// Simple tasks use the local model
runtime.prompt("Summarize this").collect().await?;

// Complex tasks use Claude
runtime.prompt("Analyze this codebase").provider("claude").collect().await?;
```

### Config file with selective tools

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .from_config()?
    .with_builtin_tools(&["read_file", "glob", "grep"])
    .launch()
    .await?;
```
