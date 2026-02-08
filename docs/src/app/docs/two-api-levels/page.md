---
title: The Two API Levels
---

acton-ai exposes two API levels that share the same actor-based runtime. The **high-level API** (`ActonAI`, `PromptBuilder`, `Conversation`) handles actor setup and message routing for you. The **low-level API** (`Kernel`, `Agent`, `ToolRegistry`, `MemoryStore`, `LLMProvider`) gives you direct control over every actor in the system.

## High-level API

The high-level API is the recommended starting point. It provides a facade that hides actor setup, subscription management, and message routing behind a fluent builder pattern.

### Key types

| Type | Role |
|---|---|
| `ActonAI` | The runtime facade. Owns the actor runtime and provider handles. `Clone + Send + 'static`. |
| `ActonAIBuilder` | Configures providers, tools, and sandboxing before launch. |
| `PromptBuilder` | Fluent builder for a single LLM request. Supports tools, streaming callbacks, and provider selection. |
| `Conversation` | Actor-backed multi-turn conversation handle. Manages history automatically. `Clone + Send + 'static`. |
| `CollectedResponse` | The result of `collect()` -- contains the full text, stop reason, token count, and executed tool calls. |

### A complete example

```rust
use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    let runtime = ActonAI::builder()
        .app_name("my-app")
        .ollama("qwen2.5:7b")
        .with_builtins()
        .launch()
        .await?;

    // Single prompt with streaming
    let response = runtime
        .prompt("List the files in the current directory")
        .system("Use tools when appropriate.")
        .on_token(|t| print!("{t}"))
        .collect()
        .await?;

    println!("\n[{} tokens]", response.token_count);

    // Multi-turn conversation
    let conv = runtime.conversation()
        .system("You are a helpful coding assistant.")
        .build()
        .await;

    let r1 = conv.send("What is Rust?").await?;
    println!("{}", r1.text);

    let r2 = conv.send("How does ownership work?").await?;
    println!("{}", r2.text);

    runtime.shutdown().await?;
    Ok(())
}
```

### What the high-level API handles for you

- Spawns the `ActorRuntime`, `Kernel`, and `LLMProvider` actors
- Creates temporary `StreamCollector` actors for each `collect()` call
- Subscribes collectors to the correct broadcast messages
- Runs the tool execution loop (send prompt, execute tools, send results, repeat)
- Manages conversation history in `Conversation` via an actor mailbox
- Resolves named providers when `.provider("name")` is used

## Low-level API

The low-level API gives you direct access to the actor primitives. Use it when you need:

- Custom actor topologies (e.g., multiple agents with different tool sets)
- Direct message handling with lifecycle hooks
- Custom stream processing with stateful actors
- Integration with your own actor-based system
- Multi-agent collaboration with task delegation

### Key types

| Type | Role |
|---|---|
| `Kernel` | Central supervisor. Manages agent lifecycles, routes inter-agent messages, capability discovery. |
| `Agent` | Individual AI agent actor. Owns conversation history, state machine, tool handles. |
| `LLMProvider` | LLM API client actor. Handles rate limiting, retries, streaming. |
| `ToolRegistry` | Central tool registration and execution dispatch actor. Supports sandboxed execution. |
| `MemoryStore` | Persistent conversation storage and semantic memory retrieval actor. |
| `LLMRequest` / `LLMStreamToken` / etc. | The raw messages that flow between actors. |

### A complete low-level example

This example is equivalent to the high-level version but uses actors directly:

```rust
use acton_ai::prelude::*;
use std::sync::Arc;
use tokio::sync::Notify;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Launch the actor runtime
    let mut runtime = ActonApp::launch_async().await;

    // 2. Spawn the kernel
    let kernel_config = KernelConfig::default().with_app_name("my-app");
    let _kernel = Kernel::spawn_with_config(&mut runtime, kernel_config).await;

    // 3. Spawn the LLM provider
    let provider_config = ProviderConfig::ollama("qwen2.5:7b");
    let provider_handle = LLMProvider::spawn(&mut runtime, provider_config).await;

    // 4. Set up a completion signal
    let done = Arc::new(Notify::new());
    let done_signal = done.clone();

    // 5. Create a custom actor to collect the response
    let mut collector = runtime.new_actor::<ResponseCollector>();

    // Handle stream tokens
    collector.mutate_on::<LLMStreamToken>(move |actor, envelope| {
        let token = &envelope.message().token;
        actor.model.buffer.push_str(token);
        print!("{token}");
        Reply::ready()
    });

    // Handle stream end
    collector.mutate_on::<LLMStreamEnd>(move |_actor, _envelope| {
        done_signal.notify_one();
        Reply::ready()
    });

    // Subscribe to streaming events
    collector.handle().subscribe::<LLMStreamToken>().await;
    collector.handle().subscribe::<LLMStreamEnd>().await;
    collector.handle().subscribe::<LLMStreamStart>().await;

    let _collector_handle = collector.start().await;

    // 6. Build and send the request
    let request = LLMRequest::builder()
        .system("You are a helpful assistant.")
        .user("What is the capital of France?")
        .build();

    provider_handle.send(request).await;

    // 7. Wait for completion
    done.notified().await;

    // 8. Shutdown
    runtime.shutdown_all().await?;
    Ok(())
}

#[acton_actor]
struct ResponseCollector {
    buffer: String,
}
```

{% callout type="warning" title="Low-level API requires more boilerplate" %}
With the low-level API, you are responsible for creating actors, subscribing to the correct messages, managing lifecycle, and handling the tool execution loop yourself. The high-level API does all of this for you. Only drop to the low-level API when you have a specific need.
{% /callout %}

## When to use which level

| Scenario | Recommended API |
|---|---|
| Single-agent chat application | High-level (`ActonAI` + `Conversation`) |
| One-shot prompt with tools | High-level (`PromptBuilder` + `collect()`) |
| Multiple providers, simple routing | High-level (`.provider("name")`) |
| Custom stateful stream processing | Low-level (`StreamHandler` trait or custom actor) |
| Multi-agent collaboration | Low-level (`Kernel` + multiple `Agent` actors) |
| Custom tool execution dispatch | Low-level (`ToolRegistry` + `Agent`) |
| Persistent memory and embeddings | Low-level (`MemoryStore`) |
| Integration with existing actor system | Low-level (share `ActorRuntime`) |

## How the high-level API wraps the low-level primitives

Understanding the mapping helps if you ever need to migrate:

| High-level | Low-level equivalent |
|---|---|
| `ActonAI::builder().launch()` | `ActonApp::launch_async()` + `Kernel::spawn()` + `LLMProvider::spawn()` |
| `.prompt("...").collect()` | Spawn `StreamCollector` actor, subscribe to `LLMStream*` messages, send `LLMRequest`, wait for `LLMStreamEnd` |
| `.tool(name, desc, schema, executor)` | Build `ToolDefinition`, execute in the `collect()` loop, send results as `Message::tool()` |
| `.with_builtins()` | `BuiltinTools::all()`, register each tool as a `ToolSpec` on the prompt |
| `.provider("claude")` | Resolve `ActorHandle` from the provider map, send `LLMRequest` to that specific handle |
| `runtime.conversation()` | Spawn `ConversationActor`, which uses `PromptBuilder` internally |

## Migration path: high to low level

If you start with the high-level API and later need low-level control, you can adopt incrementally:

### Step 1: Access the runtime

The `ActonAI` facade exposes the underlying `ActorRuntime`:

```rust
let actor_runtime = runtime.runtime();
// Now you can spawn your own actors on the same runtime
```

### Step 2: Access provider handles

Get the raw `ActorHandle` for any provider:

```rust
// Default provider
let handle = runtime.provider_handle();

// Named provider
let claude_handle = runtime.provider_handle_named("claude").unwrap();

// Send a raw LLMRequest
let request = LLMRequest::builder()
    .system("You are helpful.")
    .user("Hello")
    .build();

claude_handle.send(request).await;
```

### Step 3: Mix high and low level

You can use `PromptBuilder` for most prompts while using low-level actors for specialized processing:

```rust
// High-level for simple prompts
let response = runtime.prompt("Quick question").collect().await?;

// Low-level for custom stream processing
let mut my_actor = runtime.runtime().clone().new_actor::<MyProcessor>();
my_actor.handle().subscribe::<LLMStreamToken>().await;
let handle = my_actor.start().await;

// Send a request that your custom actor will handle
runtime.provider_handle().send(custom_request).await;
```

{% callout type="note" title="Shared message broker" %}
All actors on the same runtime share a single message broker. This means your custom low-level actors can subscribe to the same `LLMStreamToken` broadcasts that the high-level `PromptBuilder` uses. Be mindful of correlation IDs to filter messages intended for your actor.
{% /callout %}

## Next steps

- [Actor Model for AI](/docs/actor-model-for-ai) -- Understand the actor model foundation
- [Providers and Configuration](/docs/providers-and-configuration) -- Configure LLM providers
- [Tools and Streaming](/docs/tools-and-streaming) -- Tool execution and streaming details
