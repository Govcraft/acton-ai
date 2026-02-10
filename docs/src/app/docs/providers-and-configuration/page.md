---
title: Providers and Configuration
---

acton-ai supports multiple LLM providers and can be configured programmatically, via TOML config files, or both. Each provider is backed by its own [actor](/docs/actor-model-for-ai), giving you independent rate limiting, retry logic, and connection management per provider.

## Supported providers

| Provider | Constructor | API Key Required | Notes |
|---|---|---|---|
| **Anthropic** (Claude) | `ProviderConfig::anthropic(key)` | Yes | Default model: `claude-sonnet-4-20250514` |
| **OpenAI** (GPT) | `ProviderConfig::openai(key)` | Yes | Default model: `gpt-4o` |
| **Ollama** (local) | `ProviderConfig::ollama(model)` | No | Default URL: `http://localhost:11434/v1` |
| **OpenAI-compatible** | `ProviderConfig::openai_compatible(url, model)` | No | Works with vLLM, LocalAI, LiteLLM, etc. |

All providers use the same streaming and tool-calling interface. You can switch between them without changing your application code.

## Single-provider setup

For most applications, a single provider is all you need. The builder offers shorthand methods:

```rust
use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    // Ollama (local, no API key)
    let runtime = ActonAI::builder()
        .app_name("my-app")
        .ollama("qwen2.5:7b")
        .launch()
        .await?;

    runtime
        .prompt("What is the capital of France?")
        .on_token(|t| print!("{t}"))
        .collect()
        .await?;

    Ok(())
}
```

Other single-provider shortcuts:

```rust
// Anthropic Claude
ActonAI::builder().anthropic("sk-ant-...").launch().await?;

// Anthropic with specific model
ActonAI::builder().anthropic_model("sk-ant-...", "claude-3-haiku-20240307").launch().await?;

// OpenAI
ActonAI::builder().openai("sk-...").launch().await?;

// OpenAI with specific model
ActonAI::builder().openai_model("sk-...", "gpt-4-turbo").launch().await?;

// Ollama on a different host
ActonAI::builder().ollama_at("http://192.168.1.100:11434/v1", "llama3.2").launch().await?;

// Any OpenAI-compatible endpoint
ActonAI::builder()
    .provider(ProviderConfig::openai_compatible("http://localhost:8080/v1", "my-model"))
    .launch()
    .await?;
```

## Multi-provider setup

Register multiple providers with `provider_named()` and set which one is the default:

```rust
use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    let runtime = ActonAI::builder()
        .app_name("my-app")
        .provider_named("claude", ProviderConfig::anthropic("sk-ant-..."))
        .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
        .default_provider("local")
        .launch()
        .await?;

    // Uses the default provider ("local")
    runtime.prompt("Quick question").collect().await?;

    // Uses a specific provider
    runtime
        .prompt("Complex reasoning task")
        .provider("claude")
        .collect()
        .await?;

    Ok(())
}
```

{% callout type="note" title="Default provider resolution" %}
If only one provider is registered, it automatically becomes the default. If multiple providers exist and no default is specified, `launch()` returns an error. You can inspect available providers with `runtime.provider_names()` and `runtime.has_provider("name")`.
{% /callout %}

## Selecting a provider per-prompt

Use `.provider("name")` on a `PromptBuilder` to route a specific prompt to a named provider:

```rust
// Route expensive tasks to Claude, cheap tasks to local Ollama
let response = runtime
    .prompt("Analyze this complex codebase...")
    .provider("claude")
    .system("You are a senior software architect.")
    .on_token(|t| print!("{t}"))
    .collect()
    .await?;
```

If the named provider does not exist, `collect()` returns a configuration error with the list of available providers.

## ProviderConfig builder methods

`ProviderConfig` supports a fluent builder pattern for fine-tuning:

```rust
use std::time::Duration;
use acton_ai::prelude::*;

let config = ProviderConfig::anthropic("sk-ant-...")
    .with_model("claude-3-haiku-20240307")
    .with_max_tokens(1024)
    .with_timeout(Duration::from_secs(30))
    .with_rate_limit(RateLimitConfig::new(100, 100_000))
    .with_retry(RetryConfig::new(5));
```

Available builder methods:

| Method | Description | Default |
|---|---|---|
| `.with_model(model)` | Set the model name | Provider-dependent |
| `.with_max_tokens(n)` | Maximum tokens to generate | `4096` |
| `.with_timeout(duration)` | Request timeout | `120s` (Anthropic/OpenAI), `300s` (Ollama) |
| `.with_api_key(key)` | Set/override API key | From constructor |
| `.with_base_url(url)` | Override the API base URL | Provider-dependent |
| `.with_rate_limit(config)` | Rate limiting settings | 50 req/min, 40k tokens/min |
| `.with_retry(config)` | Retry settings | 3 retries, exponential backoff |
| `.with_sampling(params)` | Set all sampling parameters at once | None |
| `.with_temperature(f64)` | Set sampling temperature | Provider default |
| `.with_top_p(f64)` | Set nucleus sampling (top-p) | Provider default |
| `.with_top_k(u32)` | Set top-k sampling (Anthropic/Ollama) | Provider default |
| `.with_stop_sequences(vec)` | Set custom stop sequences | None |

## Sampling parameters

Sampling parameters control the randomness and diversity of LLM output. You can set defaults at the provider level, and optionally override them per-prompt via the [PromptBuilder](/docs/api-prompt-builder).

Provider-level defaults apply to every request sent through that provider. Per-prompt overrides (set on `PromptBuilder`) take precedence and are merged on top -- only the fields you set are overridden, the rest fall back to the provider default.

### Programmatic configuration

```rust
use acton_ai::prelude::*;

// Set defaults on the provider
let config = ProviderConfig::anthropic("sk-ant-...")
    .with_temperature(0.7)
    .with_top_p(0.9);

// Or set all parameters at once
let config = ProviderConfig::openai("sk-...")
    .with_sampling(SamplingParams {
        temperature: Some(0.8),
        top_p: Some(0.95),
        frequency_penalty: Some(0.5),
        ..Default::default()
    });
```

### TOML configuration

Sampling fields are set directly on the provider section:

```toml
[providers.claude]
type = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"
temperature = 0.7
top_p = 0.9
top_k = 40

[providers.openai]
type = "openai"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
temperature = 0.8
frequency_penalty = 0.5
presence_penalty = 0.3
```

### Available parameters

| Parameter | Type | Supported by | Description |
|---|---|---|---|
| `temperature` | `f64` | All providers | Controls randomness (0.0 = deterministic, higher = more random) |
| `top_p` | `f64` | All providers | Nucleus sampling -- consider tokens within this cumulative probability |
| `top_k` | `u32` | Anthropic, Ollama | Only consider the top-k most likely tokens |
| `frequency_penalty` | `f64` | OpenAI | Penalize tokens based on their frequency in the text so far |
| `presence_penalty` | `f64` | OpenAI | Penalize tokens that have appeared at all in the text so far |
| `seed` | `u64` | OpenAI | Seed for deterministic sampling (best-effort) |
| `stop_sequences` | `Vec<String>` | All providers | Custom sequences that cause the model to stop generating |

{% callout type="note" title="Provider-specific parameters" %}
Parameters not supported by a provider are silently ignored. For example, setting `top_k` on an OpenAI provider has no effect. This lets you share a `SamplingParams` across providers without errors.
{% /callout %}

## Rate limiting configuration

Each provider has its own rate limiter. The default settings match Anthropic's Tier 1 limits:

```rust
use acton_ai::prelude::*;

// Custom rate limits
let rate_limit = RateLimitConfig::new(100, 200_000) // 100 req/min, 200k tokens/min
    .with_max_queue_size(50);                        // Queue up to 50 when rate-limited

// Disable queueing (requests fail immediately when rate-limited)
let strict = RateLimitConfig::new(50, 40_000)
    .without_queueing();

let config = ProviderConfig::anthropic("sk-ant-...")
    .with_rate_limit(rate_limit);
```

When `queue_when_limited` is `true` (the default), requests that exceed the rate limit are queued and processed when the window resets. When `false`, they fail immediately with a rate limit error.

## Retry configuration

Failed requests are retried with exponential backoff:

```rust
use std::time::Duration;
use acton_ai::prelude::*;

let retry = RetryConfig::new(5)                              // Up to 5 retries
    .with_initial_backoff(Duration::from_secs(1))            // Start at 1 second
    .with_max_backoff(Duration::from_secs(30))               // Cap at 30 seconds
    .with_backoff_multiplier(2)                              // Double each retry
    .without_jitter();                                       // Disable random jitter

// Or disable retries entirely
let no_retry = RetryConfig::no_retries();
```

## Configuration via TOML file

Instead of hardcoding provider settings, you can load them from a TOML config file:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .from_config()?    // Searches for config file
    .with_builtins()
    .launch()
    .await?;
```

`from_config()` searches for configuration in this order:

1. `./acton-ai.toml` (project-local, checked first)
2. `~/.config/acton-ai/config.toml` (XDG config directory)

If no config file is found, `from_config()` is a no-op. If a file exists but cannot be parsed, it returns an error.

### Config file format

```toml
# Default provider when none is specified
default_provider = "ollama"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434/v1"

[providers.claude]
type = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"  # Read API key from environment variable

[providers.openai]
type = "openai"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"

[providers.custom]
type = "openai-compatible"
model = "my-model"
base_url = "http://localhost:8080/v1"
```

{% callout type="warning" title="Never put API keys directly in config files" %}
Use `api_key_env` to reference environment variables instead of storing keys in plain text. The config file should be safe to commit to version control.
{% /callout %}

### Loading from a specific path

```rust
// Load from a specific file
let runtime = ActonAI::builder()
    .app_name("my-app")
    .from_config_file("/etc/acton-ai/config.toml")?
    .launch()
    .await?;
```

### Merging config with programmatic settings

Config file providers are merged with any providers registered programmatically. Programmatic settings take precedence:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .from_config()?                                          // Load from file
    .provider_named("fast", ProviderConfig::ollama("phi3"))  // Add another
    .default_provider("fast")                                // Override default
    .launch()
    .await?;
```

## Timeout configuration

Timeouts control how long to wait for a response from the LLM API:

```rust
use std::time::Duration;

// Short timeout for fast models
let config = ProviderConfig::ollama("phi3")
    .with_timeout(Duration::from_secs(30));

// Longer timeout for complex reasoning
let config = ProviderConfig::anthropic("sk-ant-...")
    .with_timeout(Duration::from_secs(300));
```

Default timeouts:
- **Anthropic and OpenAI**: 120 seconds
- **Ollama and OpenAI-compatible**: 300 seconds (local inference can be slower)

## Next steps

- [Tools and Streaming](/docs/tools-and-streaming) -- Add tools and handle streamed responses
- [The Two API Levels](/docs/two-api-levels) -- Understand the high-level and low-level APIs
- [Conversation Management](/docs/conversation-management) -- Multi-turn conversations with automatic history
