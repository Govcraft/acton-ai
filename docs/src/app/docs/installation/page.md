---
title: Installation
---

Add acton-ai to your Rust project and configure your first LLM provider.

---

## Prerequisites

- **Rust** edition 2021 or later (install via [rustup](https://rustup.rs))
- **Tokio** async runtime (acton-ai depends on `acton-reactive`, which re-exports Tokio)
- An LLM provider: [Ollama](https://ollama.com) running locally, or an API key for Anthropic or OpenAI

{% callout type="note" title="Hypervisor for sandboxing" %}
If you plan to use sandboxed tool execution, you need a hypervisor: KVM on Linux or Hyper-V on Windows. This is optional -- acton-ai works fine without sandboxing for development and many production use cases.
{% /callout %}

---

## Add the dependency

```bash
cargo add acton-ai
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
acton-ai = "0.25"
```

You do not need to add `tokio` as a direct dependency. Acton AI re-exports it through `acton-reactive`, and the prelude includes everything you need. However, you still need the `#[tokio::main]` macro on your entry point, so add Tokio to your dev or direct dependencies if your IDE needs it for completion:

```toml
[dependencies]
acton-ai = "0.25"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

---

## Feature flags

| Feature | Default | Description |
| --- | --- | --- |
| `agent-skills` | off | Enables the agent skills system for loading and activating dynamic skill plugins. Adds `SkillRegistry`, `LoadedSkill`, `ActivateSkillTool`, and `ListSkillsTool` to the prelude. |

Enable a feature in `Cargo.toml`:

```toml
[dependencies]
acton-ai = { version = "0.25", features = ["agent-skills"] }
```

---

## Provider setup

Acton AI supports three provider types out of the box. You can configure them in a config file, in code, or with a combination of both.

### Ollama (local, no API key)

[Ollama](https://ollama.com) runs models locally. Install it, pull a model, and you are ready:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen2.5:7b
```

Then configure acton-ai to use it:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .launch()
    .await?;
```

If Ollama runs on a different host or port:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama_at("http://192.168.1.100:11434/v1", "qwen2.5:7b")
    .launch()
    .await?;
```

### Anthropic Claude

Set the `ANTHROPIC_API_KEY` environment variable, then:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .anthropic(std::env::var("ANTHROPIC_API_KEY").unwrap())
    .launch()
    .await?;
```

Or specify a particular model:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .anthropic_model(
        std::env::var("ANTHROPIC_API_KEY").unwrap(),
        "claude-sonnet-4-20250514",
    )
    .launch()
    .await?;
```

### OpenAI

Set the `OPENAI_API_KEY` environment variable, then:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .openai(std::env::var("OPENAI_API_KEY").unwrap())
    .launch()
    .await?;
```

### OpenAI-compatible endpoints

Any endpoint that speaks the OpenAI chat completions protocol works:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .provider(
        ProviderConfig::openai_compatible("http://localhost:8080/v1", "my-model")
    )
    .launch()
    .await?;
```

---

## Configuration file

For most projects, a config file is the preferred way to manage providers. It keeps API keys out of source code and lets you switch models without recompiling.

### File format

Configuration uses TOML. Here is a full example:

```toml
# Which provider to use when none is specified
default_provider = "ollama"

# --- Providers ---

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434/v1"
timeout_secs = 300

[providers.ollama.rate_limit]
requests_per_minute = 1000
tokens_per_minute = 1000000

[providers.claude]
type = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"

[providers.fast]
type = "openai"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"

# --- Sandbox (optional) ---

[sandbox]
pool_warmup = 4
pool_max_per_type = 32
max_executions_before_recycle = 1000

[sandbox.limits]
max_execution_ms = 30000
max_memory_mb = 64
```

### Provider fields

Each `[providers.<name>]` section supports these fields:

| Field | Required | Description |
| --- | --- | --- |
| `type` | yes | Provider type: `"anthropic"`, `"openai"`, or `"ollama"`. Unknown types are treated as OpenAI-compatible. |
| `model` | yes | Model identifier (e.g., `"claude-sonnet-4-20250514"`, `"gpt-4o"`, `"qwen2.5:7b"`). |
| `api_key_env` | no | Name of the environment variable holding the API key. Recommended over `api_key`. |
| `api_key` | no | Direct API key value. Discouraged -- use `api_key_env` instead. |
| `base_url` | no | Custom API base URL. Required for Ollama; optional for Anthropic and OpenAI. |
| `timeout_secs` | no | Request timeout in seconds. Defaults to 120 for cloud providers, 300 for local. |
| `max_tokens` | no | Maximum tokens to generate per response. Defaults to 4096. |

### API key resolution order

When resolving the API key for a provider, acton-ai checks in this order:

1. The environment variable named in `api_key_env`
2. The standard environment variable for the provider type (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`)
3. The `api_key` field in the config file
4. Empty string (appropriate for Ollama and other local providers)

{% callout type="warning" title="Never commit API keys" %}
Always use `api_key_env` to reference an environment variable rather than putting keys directly in your config file. If you must use `api_key`, make sure your config file is in `.gitignore`.
{% /callout %}

### Search paths

When you call `from_config()`, acton-ai looks for config files in this order:

1. **`./acton-ai.toml`** -- project-local config (checked first)
2. **`~/.config/acton-ai/config.toml`** -- XDG user config directory

The first file found wins. If no config file is found, `from_config()` returns an empty configuration (no error). If a file is found but contains invalid TOML, it returns an error.

You can also load from a specific path:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .from_config_file("/etc/acton-ai/config.toml")?
    .launch()
    .await?;
```

### Loading config in code

```rust
use acton_ai::prelude::*;

// Load from acton-ai.toml or ~/.config/acton-ai/config.toml
let runtime = ActonAI::builder()
    .app_name("my-app")
    .from_config()?
    .launch()
    .await?;
```

You can combine config-file providers with programmatic ones. Providers registered in code are merged with those from the config file:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .from_config()?                           // load ollama, claude from file
    .provider_named("custom",                 // add another in code
        ProviderConfig::openai_compatible("http://localhost:8080/v1", "my-model")
    )
    .default_provider("custom")               // override the file's default
    .launch()
    .await?;
```

---

## Environment variables

| Variable | Used by | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Anthropic provider | API key for Claude models. Checked automatically when provider type is `"anthropic"`. |
| `OPENAI_API_KEY` | OpenAI provider | API key for GPT models. Checked automatically when provider type is `"openai"`. |
| `RUST_LOG` | Logging | Controls log level via `tracing-subscriber`'s env filter (e.g., `RUST_LOG=acton_ai=debug`). |

You can also define custom environment variable names per provider using the `api_key_env` config field. For example, `api_key_env = "MY_CUSTOM_KEY"` tells acton-ai to read the key from `$MY_CUSTOM_KEY`.

---

## Verifying your setup

Here is a minimal program to verify everything works:

```rust
use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    let runtime = ActonAI::builder()
        .app_name("verify-setup")
        .from_config()?
        .launch()
        .await?;

    let response = runtime
        .prompt("Say hello in exactly three words.")
        .collect()
        .await?;

    println!("Response: {}", response.text);
    println!("Tokens: {}", response.token_count);

    runtime.shutdown().await?;
    Ok(())
}
```

```bash
cargo run
```

If you see a response, your provider is configured correctly.

---

## Next steps

- [Providers and Configuration](/docs/providers-and-configuration) -- rate limiting, timeouts, and multi-provider patterns
- [Tools and Streaming](/docs/tools-and-streaming) -- enable built-in tools and handle streaming responses
- [Conversation Management](/docs/conversation-management) -- multi-turn conversations with automatic history
- [Architecture Overview](/docs/architecture-overview) -- understand the actor-based runtime
