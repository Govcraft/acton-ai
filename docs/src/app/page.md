---
title: Getting started
---

An agentic AI framework where each agent is an actor, built in Rust.

Acton AI combines the [actor model](https://en.wikipedia.org/wiki/Actor_model) with large language models to give you concurrent, fault-isolated AI agents with a simple, ergonomic API. Connect to Anthropic, OpenAI, or local models through Ollama -- all with streaming responses, built-in tool use, multi-turn conversations, and hardware-sandboxed code execution.

{% quick-links %}

{% quick-link title="Installation" icon="installation" href="/docs/installation" description="Add acton-ai to your project and configure your first LLM provider." /%}

{% quick-link title="Core Concepts" icon="presets" href="/docs/architecture-overview" description="Understand the actor model, kernels, agents, and how they fit together." /%}

{% quick-link title="API Reference" icon="plugins" href="/docs/api-acton-ai" description="Explore ActonAI, PromptBuilder, Conversation, and built-in tools." /%}

{% quick-link title="Guides" icon="theming" href="/docs/tools-and-streaming" description="Learn about tool use, streaming, conversations, and multi-agent collaboration." /%}

{% /quick-links %}

---

## Quick start

Here is the fastest way to go from zero to a working AI prompt. This example uses [Ollama](https://ollama.com) running locally, so no API key is needed.

### 1. Add the dependency

```bash
cargo add acton-ai
```

### 2. Create a config file

Save this as `acton-ai.toml` in your project root:

```toml
default_provider = "ollama"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434/v1"
```

### 3. Write your first prompt

```rust
use acton_ai::prelude::*;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    // Load provider settings from acton-ai.toml
    let runtime = ActonAI::builder()
        .app_name("hello-acton")
        .from_config()?
        .launch()
        .await?;

    // Send a prompt and stream tokens as they arrive
    let response = runtime
        .prompt("What is the capital of France? Answer in one sentence.")
        .system("You are a helpful assistant. Be concise.")
        .on_token(|token| {
            print!("{token}");
            std::io::stdout().flush().ok();
        })
        .collect()
        .await?;

    println!();
    println!("[{} tokens, {:?}]", response.token_count, response.stop_reason);

    runtime.shutdown().await?;
    Ok(())
}
```

Run it:

```bash
cargo run
```

{% callout type="note" title="No config file? No problem." %}
You can skip the config file entirely and configure the provider in code:

```rust
let runtime = ActonAI::builder()
    .app_name("hello-acton")
    .ollama("qwen2.5:7b")
    .launch()
    .await?;
```

See [Installation](/docs/installation) for all the ways to configure providers.
{% /callout %}

---

## Key features

### Multi-provider LLM support

Connect to Anthropic Claude, OpenAI GPT, Ollama, or any OpenAI-compatible endpoint. Register multiple providers and switch between them per-prompt:

```rust
let runtime = ActonAI::builder()
    .app_name("multi-provider")
    .provider_named("claude", ProviderConfig::anthropic("sk-ant-..."))
    .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
    .default_provider("local")
    .launch()
    .await?;

// Uses the default provider (local Ollama)
runtime.prompt("Quick question").collect().await?;

// Uses Claude for this specific prompt
runtime.prompt("Complex reasoning task").provider("claude").collect().await?;
```

### Streaming responses

Every prompt streams tokens by default. Attach a callback to process them as they arrive:

```rust
runtime
    .prompt("Explain the actor model")
    .on_token(|token| print!("{token}"))
    .collect()
    .await?;
```

### Built-in tools

Give your agents the ability to read files, run shell commands, search with glob and grep, fetch URLs, and more -- all with a single method call:

```rust
let runtime = ActonAI::builder()
    .app_name("tool-user")
    .from_config()?
    .with_builtins()   // enables read_file, bash, glob, grep, etc.
    .launch()
    .await?;

runtime
    .prompt("List all Rust source files in this project and count the lines")
    .collect()
    .await?;
```

### Multi-turn conversations

The `Conversation` API handles history management automatically. Each call to `send()` appends both the user message and the assistant response to the conversation history:

```rust
let conv = runtime.conversation()
    .system("You are a helpful Rust tutor.")
    .build()
    .await;

let r1 = conv.send("What is ownership in Rust?").await?;
println!("{}", r1.text);

// The conversation remembers the previous exchange
let r2 = conv.send("How does borrowing relate to that?").await?;
println!("{}", r2.text);
```

Or launch an interactive terminal chat in five lines:

```rust
ActonAI::builder()
    .app_name("chat")
    .from_config()?
    .with_builtins()
    .launch()
    .await?
    .conversation()
    .run_chat()
    .await
```

### Sandboxed tool execution

Tools run inside [Hyperlight](https://github.com/hyperlight-dev/hyperlight) micro-VMs for hardware-level isolation, with 1-2ms cold start times. Requires a hypervisor (KVM on Linux, Hyper-V on Windows).

```rust
let runtime = ActonAI::builder()
    .app_name("sandboxed")
    .from_config()?
    .with_builtins()
    .with_sandbox_pool(4)   // keep 4 micro-VMs warm
    .launch()
    .await?;
```

---

## Next steps

- [Installation](/docs/installation) -- add acton-ai to your project and set up providers
- [Providers and Configuration](/docs/providers-and-configuration) -- deep dive into multi-provider setup and config files
- [Tools and Streaming](/docs/tools-and-streaming) -- understand tool use and streaming responses
- [Conversation Management](/docs/conversation-management) -- multi-turn conversations and history
- [Architecture Overview](/docs/architecture-overview) -- how actors, kernels, and agents work together
