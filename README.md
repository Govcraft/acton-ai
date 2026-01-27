# acton-ai

**Build production-ready AI agents in Rust with minimal boilerplate.**

Acton-ai handles the hard problems—concurrency, fault tolerance, rate limiting, streaming, and tool execution—so you can focus on your application logic.

## At a Glance

```rust
use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    ActonAI::builder()
        .app_name("my-app")
        .ollama("qwen2.5:7b")
        .with_builtins()
        .launch()
        .await?
        .conversation()
        .run_chat()
        .await
}
```

Five lines to an interactive chat with file access and command execution.

## Features

- **Multi-provider support** — Anthropic Claude, OpenAI, Ollama, and any OpenAI-compatible API
- **Streaming responses** — Token-by-token callbacks for real-time output
- **Built-in tools** — File operations, bash, grep, glob, web fetch, and calculations
- **Tool execution loop** — Automatic tool calling and result handling until completion
- **Two API levels** — Simple facade for common cases, full actor access for advanced control
- **TOML configuration** — Define providers and settings in config files
- **Hyperlight sandboxing** — Hardware-isolated execution for untrusted code (KVM/Hyper-V)
- **Rate limiting** — Built-in request and token limits per provider
- **Actor-based architecture** — Fault-tolerant, concurrent design via [acton-reactive](https://docs.rs/acton-reactive)

## Installation

```bash
cargo add acton-ai
```

For Ollama (local), no API key is needed. For cloud providers, set environment variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

## Quick Start

Common patterns to get you started. Complete examples in [`examples/`](examples/).

### Simple Prompt

```rust
use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    let runtime = ActonAI::builder()
        .app_name("my-app")
        .ollama("qwen2.5:7b")
        .launch()
        .await?;

    let response = runtime
        .prompt("What is the capital of France?")
        .system("Be concise.")
        .collect()
        .await?;

    println!("{}", response.text);
    Ok(())
}
```

### Streaming Output

```rust
runtime
    .prompt("Explain Rust ownership in simple terms.")
    .on_token(|token| print!("{token}"))
    .collect()
    .await?;
```

### Multi-turn Conversation

```rust
let mut conv = runtime.conversation()
    .system("You are a helpful assistant.")
    .build();

let response = conv.send("What is Rust?").await?;
println!("{}", response.text);

// Context is maintained
let response = conv.send("How does ownership work?").await?;
println!("{}", response.text);
```

### Using Built-in Tools

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .with_builtins()  // Enable all built-in tools
    .launch()
    .await?;

runtime
    .prompt("List the Rust files in the current directory")
    .on_token(|t| print!("{t}"))
    .collect()
    .await?;
```

### Custom Tools

```rust
runtime
    .prompt("What is 42 * 17?")
    .tool(
        "calculator",
        "Evaluates math expressions",
        json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string" }
            },
            "required": ["expression"]
        }),
        |args| async move {
            let expr = args["expression"].as_str().unwrap();
            Ok(json!({ "result": evaluate(expr) }))
        },
    )
    .collect()
    .await?;
```

### Multiple Providers

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
    .provider_named("cloud", ProviderConfig::anthropic("sk-ant-..."))
    .default_provider("local")
    .launch()
    .await?;

// Quick tasks on local
runtime.prompt("Summarize this").collect().await?;

// Complex reasoning on cloud
runtime.prompt("Analyze this code").provider("cloud").collect().await?;
```

## Configuration

Configure providers via TOML files or programmatically.

### Config File

Create `acton-ai.toml` in your project root or `~/.config/acton-ai/config.toml`:

```toml
default_provider = "ollama"

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

# Optional: Hyperlight sandbox for tool isolation
[sandbox]
pool_warmup = 4
pool_max_per_type = 32

[sandbox.limits]
max_execution_ms = 30000
max_memory_mb = 64
```

Load the configuration:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .from_config()?
    .with_builtins()
    .launch()
    .await?;
```

### Programmatic Configuration

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .provider_named("claude",
        ProviderConfig::anthropic("sk-ant-...")
            .with_model("claude-sonnet-4-20250514")
            .with_max_tokens(4096))
    .provider_named("local",
        ProviderConfig::ollama("qwen2.5:7b"))
    .default_provider("local")
    .with_builtins()
    .with_sandbox_pool(4)  // Pre-warm 4 Hyperlight sandboxes
    .launch()
    .await?;
```

## Built-in Tools

Available when you call `.with_builtins()`:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line numbers |
| `write_file` | Write content to files |
| `edit_file` | Make targeted string replacements |
| `list_directory` | List directory contents with metadata |
| `glob` | Find files matching glob patterns |
| `grep` | Search file contents with regex |
| `bash` | Execute shell commands |
| `calculate` | Evaluate mathematical expressions |
| `web_fetch` | Fetch content from URLs |

Select specific tools with `.with_builtin_tools(&["read_file", "glob", "bash"])`.

## Architecture

Acton-ai uses the actor model for fault-tolerant, concurrent AI systems:

```
ActonAI (Facade)
    │
    ├── ActorRuntime (acton-reactive)
    │       │
    │       ├── Kernel ─────────── Central supervisor, agent lifecycle
    │       │
    │       ├── LLMProvider(s) ─── API calls, streaming, rate limiting
    │       │
    │       ├── Agent(s) ───────── Individual AI agents with reasoning
    │       │
    │       └── ToolRegistry ───── Tool registration and execution
    │
    └── BuiltinTools ──────────── File ops, bash, web fetch, etc.
```

**Two API levels:**

| Level | Use Case | Access |
|-------|----------|--------|
| **High-level** | Most applications | `ActonAI::builder()`, `PromptBuilder`, `Conversation` |
| **Low-level** | Custom agent topologies | Direct actor spawning, message routing, subscriptions |

The high-level API handles actor lifecycle, subscriptions, and message routing automatically. Drop down to the low-level API when you need custom supervision strategies or multi-agent coordination.

## Examples

```bash
# Interactive chat with tools
cargo run --example conversation

# Multiple LLM providers
cargo run --example multi_provider

# Custom tool definitions
cargo run --example ollama_tools

# Hyperlight sandboxed execution
cargo run --example bash_sandbox

# Per-agent tool configuration
cargo run --example per_agent_tools
```

## Documentation

- [API Documentation (docs.rs)](https://docs.rs/acton-ai)
- [acton-reactive](https://docs.rs/acton-reactive) — The underlying actor framework

## Contributing

Contributions welcome. Please open an issue to discuss significant changes before submitting a PR.

## License

MIT License. See [LICENSE](LICENSE) for details.
