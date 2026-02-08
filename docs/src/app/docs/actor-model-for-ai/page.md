---
title: Actor Model for AI
---

acton-ai is built on the **actor model** -- a concurrency paradigm where independent actors communicate exclusively through asynchronous message-passing. This design is a natural fit for AI agent systems, where multiple components (LLM providers, tool registries, agents, memory stores) must operate concurrently without stepping on each other's state.

## What is the actor model?

In the actor model, each **actor** is a lightweight concurrent entity that:

- Owns private, mutable state that no other actor can access directly
- Communicates with other actors only by sending and receiving messages
- Processes messages one at a time from its **mailbox** (a queue)
- Can create new actors, send messages, and update its own state in response to a message

This stands in contrast to shared-state concurrency (mutexes, locks, atomics), which is error-prone and difficult to reason about -- especially in async Rust.

## Why actors for AI agents?

AI agent systems have several properties that make the actor model an excellent fit:

1. **Multiple concurrent concerns** -- An LLM is generating tokens, tools are executing, memory is being persisted, and the user is waiting for streamed output. These happen simultaneously and need coordination without blocking each other.

2. **Natural isolation** -- Each agent should have its own conversation history, state machine, and tool set. Actors enforce this isolation structurally rather than through discipline.

3. **Message-driven workflows** -- The LLM request/response cycle is inherently message-based: send a prompt, receive streamed tokens, receive tool call requests, send tool results back. Actors model this directly.

4. **No shared mutable state** -- When multiple agents and providers are running, shared state (protected by `Mutex` or `RwLock`) becomes a bottleneck and a source of deadlocks. Actors eliminate this class of bugs entirely.

## The acton-reactive foundation

acton-ai is built on [**acton-reactive**](https://crates.io/crates/acton-reactive), a Rust actor framework designed for async message-passing on top of Tokio. acton-reactive provides:

- **`#[acton_actor]`** -- A derive macro for defining actor state
- **`ManagedActor`** -- A builder for configuring message handlers and lifecycle hooks
- **`ActorHandle`** -- A cheap, cloneable handle for sending messages to an actor
- **Message broker** -- A pub/sub system for broadcasting messages to all subscribers
- **Lifecycle hooks** -- `before_start`, `after_start`, `before_stop`, `after_stop`

acton-ai uses these primitives to build its entire runtime. You never need to use acton-reactive directly unless you are working with the [low-level API](/docs/two-api-levels).

## How acton-ai uses actors internally

The system is composed of several actor types that communicate through the message broker:

### Kernel

The **Kernel** is the central supervisor. It manages agent lifecycles, routes inter-agent messages, and maintains a capability registry for agent discovery. There is one Kernel per runtime.

### LLMProvider

Each **LLMProvider** actor wraps a single LLM API connection (Anthropic, OpenAI, Ollama, or any OpenAI-compatible endpoint). It handles:

- Rate limiting with configurable requests-per-minute and tokens-per-minute
- Request queuing when rate-limited
- Streaming token delivery via broadcast messages
- Retry logic with exponential backoff

When the provider receives an `LLMRequest`, it calls the API and broadcasts `LLMStreamStart`, `LLMStreamToken`, `LLMStreamToolCall`, and `LLMStreamEnd` messages. Any actor that has subscribed to these message types receives them.

### Agent

Each **Agent** actor maintains its own conversation history, state machine (`Idle`, `Thinking`, `Executing`, `Completed`), and tool definitions. It processes `UserPrompt` messages by sending `LLMRequest` to the provider and handling the streamed response. When the LLM requests tool calls, the agent transitions to the `Executing` state and dispatches tool executions.

### ToolRegistry

The **ToolRegistry** actor manages tool registration, validation, and execution dispatch. It supports sandboxed execution through configurable sandbox factories.

### MemoryStore

The **MemoryStore** actor handles persistent conversation storage, context windows, and semantic memory retrieval. It uses libsql for database operations, spawning them as tokio tasks to avoid blocking the actor mailbox.

## Message-passing vs shared state

Here is the key difference in practice. With shared state, you might write:

```rust
// Shared-state approach (problematic)
let history = Arc::new(Mutex::new(Vec::new()));
let history_clone = history.clone();

// In the streaming handler:
let mut guard = history_clone.lock().await; // Can deadlock!
guard.push(new_message);
// guard held across .await -- undefined behavior risk
```

With the actor model, the same operation is safe by construction:

```rust
// Actor approach (used by acton-ai)
// The ConversationActor owns history -- no locks needed
actor.model.conversation.push(new_message);
// Only one message is processed at a time
```

{% callout type="note" title="Zero mutexes in Conversation" %}
The `Conversation` type in acton-ai v0.25.0 is backed by a `ConversationActor` that owns the history. Concurrency safety comes from mailbox serialization, not from locks. The public `Conversation` handle is `Clone + Send + 'static`, and all methods take `&self`.
{% /callout %}

## Mailbox serialization

Each actor processes messages **one at a time** from its mailbox queue. This is the fundamental safety guarantee:

1. A message arrives and is placed in the actor's mailbox
2. The actor picks up the message and runs its handler
3. The handler can mutate actor state freely -- no other code is running against this state
4. When the handler completes, the actor picks up the next message

This means:

- **No data races** -- Only one handler runs against actor state at a time
- **No deadlocks** -- There are no locks to contend over
- **Deterministic ordering** -- Messages from a single sender arrive in order
- **Async-safe** -- Handlers can use `Reply::pending(async { ... })` to perform async work without holding state across await points

```rust
// Inside an actor handler -- safe direct state mutation
collector.mutate_on::<LLMStreamToken>(move |actor, envelope| {
    let token = &envelope.message().token;

    // Direct mutation -- no locks, no contention
    actor.model.buffer.push_str(token);
    actor.model.token_count += 1;

    Reply::ready()
});
```

{% callout type="warning" title="Mailbox blocks during Reply::pending" %}
When a handler returns `Reply::pending(async { ... })`, the actor's mailbox is blocked until the future completes. This serializes message processing for that actor. For long-running operations, consider spawning a separate tokio task and communicating the result back via a message.
{% /callout %}

## The broadcast pattern

acton-ai uses the **broker** (a pub/sub message bus) extensively. When the LLM provider streams tokens, it broadcasts `LLMStreamToken` messages. Any actor that has subscribed to this message type receives a copy:

```text
LLMProvider --broadcast(LLMStreamToken)--> Broker
                                              |
                                    +---------+---------+
                                    |                   |
                               Agent actor        StreamCollector
                              (updates history)   (calls on_token callback)
```

This decoupling means the provider does not need to know who is listening. New subscribers can be added without modifying the provider.

## Next steps

- [Providers and Configuration](/docs/providers-and-configuration) -- Set up LLM providers
- [Tools and Streaming](/docs/tools-and-streaming) -- Understand tool execution and streaming
- [The Two API Levels](/docs/two-api-levels) -- Choose between high-level and low-level APIs
