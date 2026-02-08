---
title: Architecture Overview
---

Acton-ai is an agentic AI framework where each agent is an actor. This page describes the system architecture, module responsibilities, and how data flows through the system.

---

## Design philosophy

The framework is built on three principles:

1. **Actors all the way down.** Every long-lived component (kernel, agents, LLM providers, tool executors, memory store) is an actor managed by [acton-reactive](https://github.com/acton-reactive/acton-reactive). This gives you supervision, fault isolation, and message-based concurrency for free.

2. **Two API levels.** A high-level facade (`ActonAI`) hides actor plumbing for common use cases. The low-level API exposes the full actor system for advanced scenarios. Both levels are first-class citizens.

3. **Streaming-first.** LLM responses are streamed token-by-token through the actor system using a pub/sub broker. Consumers (agents, prompt builders, conversation handles) subscribe to the stream events they care about.

---

## High-level architecture

```text
+------------------------------------------------------------------+
|                         ActonAI (Facade)                         |
|   Builder pattern, PromptBuilder, ConversationBuilder            |
+------------------------------------------------------------------+
        |                    |                    |
        v                    v                    v
+----------------+  +------------------+  +-------------------+
| ActorRuntime   |  | LLM Provider(s)  |  | BuiltinTools      |
| (acton-reactive|  | (actor per       |  | (tool definitions |
|  tokio-based)  |  |  provider)       |  |  + executors)     |
+----------------+  +------------------+  +-------------------+
        |                    |
        v                    v
+----------------+  +------------------+
| Kernel         |  | Pub/Sub Broker   |
| (supervisor)   |  | (event routing)  |
+----------------+  +------------------+
        |
        v
+------------------------------------------------------------------+
|                        Agent(s)                                   |
|   State machine, conversation history, tool handles,             |
|   delegation tracker                                              |
+------------------------------------------------------------------+
        |                    |                    |
        v                    v                    v
+----------------+  +------------------+  +-------------------+
| Tool Actors    |  | Memory Store     |  | Sandbox Pool      |
| (per-agent,    |  | (libSQL/Turso    |  | (Hyperlight       |
|  supervised)   |  |  persistence)    |  |  micro-VMs)       |
+----------------+  +------------------+  +-------------------+
```

---

## Module organization

The crate is organized into the following modules. Each module maps to a directory under `src/` (or a single file for simpler modules).

### `facade` (`src/facade.rs`)

The high-level entry point. A single file containing `ActonAI` and `ActonAIBuilder`, which wrap the actor runtime, named LLM providers, and built-in tools behind a clean builder-pattern API. The facade is `Clone + Send + 'static` thanks to internal `Arc` sharing.

Key types:
- `ActonAI` -- the runtime handle returned by `ActonAIBuilder::launch()`
- `ActonAIBuilder` -- configures providers, builtins, and sandbox before launch
- `DEFAULT_PROVIDER_NAME` -- the name used for single-provider setups (`"default"`)

### `kernel` (`src/kernel/`)

The central supervisor. The Kernel actor manages agent lifecycles, routes inter-agent messages, maintains a capability registry for agent discovery, and tracks system-wide metrics.

Key types:
- `Kernel` -- the supervisor actor
- `KernelConfig` -- configuration (max agents, enable metrics, default system prompt, logging)
- `CapabilityRegistry` -- maps capabilities to agents for discovery
- `KernelMetrics` -- counters for agents spawned, stopped, and messages routed

### `agent` (`src/agent/`)

Individual AI agents. Each agent is an actor with its own conversation history, state machine, tool handles, and delegation tracker. The agent processes user prompts by sending LLM requests, handling streaming responses, and executing tool calls in a reasoning loop.

Key types:
- `Agent` -- the agent actor
- `AgentConfig` -- agent configuration (system prompt, name, tools)
- `AgentState` -- state machine (Idle, Thinking, Executing, etc.)
- `DelegationTracker` -- tracks delegated and incoming tasks for multi-agent workflows

### `llm` (`src/llm/`)

LLM provider actor and API clients. The provider actor manages rate limiting, request queuing, retry logic, and streaming. Two client implementations are included: `AnthropicClient` for the Anthropic Messages API and `OpenAIClient` for OpenAI-compatible chat completions endpoints (including Ollama).

Key types:
- `LLMProvider` -- the provider actor
- `LLMClient` -- trait (`Send + Sync + Debug`) abstracting provider-specific API calls
- `AnthropicClient`, `OpenAIClient` -- concrete implementations of `LLMClient`
- `ProviderConfig` -- model, API key, base URL, rate limits, timeouts
- `ProviderType` -- enum distinguishing Anthropic from OpenAI-compatible
- `LLMStreamEvent` -- token, tool call, start, end, and error events
- `StreamAccumulator` -- accumulates streaming tokens into complete responses

### `tools` (`src/tools/`)

The tool system. Provides infrastructure for tool registration, execution, and sandboxing. Tools can be registered globally (via `ToolRegistry`) or per-agent (via `ToolActor`). The per-agent approach is recommended because each tool actor is supervised as a child of its owning agent.

Key submodules:
- `builtins/` -- pre-built tools (read_file, write_file, edit_file, list_directory, glob, grep, bash, calculate, web_fetch, rust_code)
- `sandbox/` -- sandboxed execution via Hyperlight micro-VMs
- `compiler/` -- Rust code compilation, caching, and templates for the `rust_code` tool
- `security/` -- path validation and sanitization (`PathValidator`)
- `actor.rs` -- per-agent `ToolActor` implementation and the `ToolExecutor` trait
- `registry.rs` -- global `ToolRegistry` actor (legacy approach)
- `executor.rs` -- temporary executor actor for one-shot tool execution
- `definition.rs` -- `ToolConfig`, `ToolExecutorTrait`, and boxed executor types

Key types:
- `ToolDefinition` -- JSON schema describing a tool's name, description, and parameters
- `ToolConfig` -- tool definition plus executor
- `ToolExecutorTrait` -- trait for implementing custom tool execution
- `BuiltinTools` -- collection of pre-built tools

### `memory` (`src/memory/`)

Persistence and context window management. The `MemoryStore` actor handles all database operations asynchronously using libSQL (Turso's SQLite fork). Supports storing agent state snapshots, conversation histories, and semantic memories with optional vector embeddings.

Key types:
- `MemoryStore` -- the persistence actor
- `PersistenceConfig` -- database connection configuration
- `Memory`, `ScoredMemory` -- memory entries with optional embeddings
- `EmbeddingProvider` -- trait for embedding generation services
- `ContextWindow`, `ContextWindowConfig` -- context window management and truncation strategies
- `AgentStateSnapshot` -- serializable agent state for persistence

### `conversation` (`src/conversation.rs`)

Actor-backed conversation handle for multi-turn interactions. `Conversation` is `Clone + Send + 'static` with all methods taking `&self`. A `ConversationActor` owns the history internally, and the mailbox serializes sends -- no mutexes needed.

Key types:
- `Conversation` -- the public handle
- `ConversationBuilder` -- configures system prompt, history restoration, exit tool
- `ChatConfig` -- configuration for the interactive chat loop (`user_prompt`, `assistant_prompt`, `map_input`)
- `StreamToken` -- individual streaming token message

### `prompt` (`src/prompt.rs`)

The fluent `PromptBuilder` API for constructing single LLM requests. Supports system prompts, message history, tool definitions, streaming callbacks, and provider selection. The builder pattern makes it easy to construct complex requests step by step.

### `messages` (`src/messages/`)

All actor message types used throughout the framework. Contains `mod.rs` and `types.rs`. Messages are the currency of communication between actors. They implement the `ActonMessage` trait via the `#[acton_message]` derive macro, which requires `Any + Send + Sync + Debug + Clone + 'static`.

### `error` (`src/error.rs`)

The error type hierarchy. See the [Code Standards](/docs/code-standards) page for details on the error handling patterns.

### `types` (`src/types/`)

Core newtypes providing strong typing for identifiers. Each type has its own file: `AgentId`, `ConversationId`, `CorrelationId`, `MemoryId`, `MessageId`, `TaskId`, and `ToolName`. All implement `Clone`, `Debug`, `PartialEq`, `Eq`, `Hash`, `Serialize`, and `Deserialize` with validation on construction.

---

## Actor hierarchy

When you call `ActonAI::builder().launch()`, the following actors are spawned:

```text
ActorRuntime (tokio-based)
  |
  +-- Kernel (supervisor, one per runtime)
  |     |
  |     +-- Agent (one per spawned agent)
  |     |     |
  |     |     +-- ToolActor (one per tool registered with the agent)
  |     |
  |     +-- Agent ...
  |
  +-- LLMProvider "default" (one per registered provider)
  +-- LLMProvider "claude" ...
  +-- SandboxPool (optional, manages pre-warmed Hyperlight VMs)
  +-- MemoryStore (optional, for persistence)
```

The Kernel supervises agents. Each agent supervises its own tool actors. LLM providers and the sandbox pool are top-level actors managed directly by the runtime. This hierarchy ensures that when an agent fails, only its tools are affected -- other agents and the rest of the system continue operating.

---

## Message flow: prompt to response

Here is how a typical `runtime.prompt("Hello").collect().await` flows through the system:

### 1. PromptBuilder constructs the request

The `PromptBuilder` (returned by `ActonAI::prompt()`) assembles a `LLMRequest` message containing:
- The user message
- An optional system prompt
- Conversation history (if provided via `.messages()`)
- Tool definitions (if builtins or custom tools are configured)
- A unique `CorrelationId` for tracking this request through the system

### 2. PromptBuilder spawns a collector actor

Before sending the request, the builder spawns a temporary actor that subscribes to the pub/sub broker for:
- `LLMStreamStart` -- marks the beginning of a streaming response
- `LLMStreamToken` -- individual tokens as they arrive
- `LLMStreamEnd` -- marks the end of streaming
- `LLMStreamToolCall` -- tool call requests from the LLM
- `LLMResponse` -- the complete accumulated response

The collector filters events by `CorrelationId` so it only processes events for its own request.

### 3. Request is sent to the LLM Provider

The `LLMRequest` message is sent to the target LLM Provider actor. The provider:
1. Checks rate limits (queues if limited, rejects if queue is full)
2. Records the request in its metrics
3. Spawns a `tokio::spawn` task to make the HTTP API call (avoids blocking the actor mailbox)

### 4. LLM Provider streams the response

The spawned task calls the `LLMClient::send_streaming_request()` method, which returns an async stream of `LLMStreamEvent` items. As events arrive:
- Each `Token` event is broadcast via the pub/sub broker as `LLMStreamToken`
- Each `ToolCall` event is broadcast as `LLMStreamToolCall`
- The `End` event triggers `LLMStreamEnd` and the accumulated `LLMResponse`

### 5. Collector actor receives events

The collector actor's message handlers fire for each broadcast event:
- `on_token` callbacks are invoked for each `LLMStreamToken` (this is how `print!("{t}")` works in the streaming API)
- Tool calls are collected and optionally auto-executed
- When `LLMStreamEnd` arrives, the collector assembles the final `CollectedResponse`

### 6. Tool execution loop (if tools are present)

If the LLM response includes tool calls and the `StopReason` is `ToolUse`:
1. Each tool call is executed (either via a `ToolActor` or inline executor)
2. Tool results are appended to the message history
3. A new `LLMRequest` is sent with the updated history (continuing the conversation)
4. Steps 3-6 repeat until the LLM returns `EndTurn` or `MaxTokens`

### 7. Response is returned

The `collect().await` call resolves with a `CollectedResponse` containing:
- `text` -- the full accumulated text
- `stop_reason` -- why the LLM stopped (`EndTurn`, `MaxTokens`, `ToolUse`, `StopSequence`)
- `token_count` -- number of tokens received
- `tool_calls` -- details of any tool calls that were executed (as `Vec<ExecutedToolCall>`)

---

## How streaming works internally

Streaming in acton-ai is broker-based rather than channel-based. This is a key architectural decision:

1. The **LLM Provider** broadcasts stream events (`LLMStreamToken`, `LLMStreamStart`, `LLMStreamEnd`, `LLMStreamToolCall`) to the pub/sub broker.

2. Any actor that subscribes to these message types receives them. The `CorrelationId` field on each event allows receivers to filter for their specific request.

3. The **PromptBuilder** spawns a temporary collector actor that subscribes to these events, invokes user callbacks (like `on_token`), and accumulates the response.

4. The **Conversation** handle works similarly -- it spawns its own internal actor that subscribes to stream events.

This design means multiple consumers can observe the same stream simultaneously, and adding new stream consumers does not require changes to the provider or agent code.

---

## How tool execution flows

The tool system supports two architectures:

### Per-agent tool actors (recommended)

```text
Agent receives LLMStreamToolCall
  |
  +-- Looks up tool handle in self.tool_handles
  |
  +-- Sends ExecuteToolDirect to the ToolActor
  |
  +-- ToolActor executes the tool (via ToolExecutorTrait)
  |     |
  |     +-- (optional) Runs in Hyperlight sandbox
  |
  +-- ToolActor responds with ToolActorResponse
  |
  +-- Agent appends tool result to conversation history
  |
  +-- Agent sends new LLMRequest with updated history
```

Each tool actor is a supervised child of its owning agent. If a tool actor crashes, only that agent is affected.

### Inline executors (via PromptBuilder)

When tools are registered directly on a `PromptBuilder` via `.tool()`, they run as inline async closures within the collector actor. This is simpler but provides less isolation.

---

## The facade pattern

The `ActonAI` struct is a facade that wraps:

- An `ActorRuntime` (the tokio-based actor system)
- A `HashMap<String, ActorHandle>` of named LLM providers
- An optional `BuiltinTools` collection
- Shutdown state

The facade provides three main entry points:

| Method | Returns | Purpose |
| --- | --- | --- |
| `prompt()` | `PromptBuilder` | Single request with fluent API |
| `continue_with()` | `PromptBuilder` | Continue from existing message history |
| `conversation()` | `ConversationBuilder` | Multi-turn session with automatic history |

All three ultimately construct `LLMRequest` messages and send them to the appropriate `LLMProvider` actor. The facade insulates users from needing to understand actor handles, message types, or pub/sub subscription mechanics.

For users who need full control, the `runtime()` method provides an escape hatch to the underlying `ActorRuntime`, from which you can spawn custom actors, access the broker directly, or interact with the kernel.

---

## Next steps

- [Actor Model for AI](/docs/actor-model-for-ai) -- why the actor model is a good fit for AI systems
- [Two API Levels](/docs/two-api-levels) -- detailed comparison of the facade vs. low-level API
- [Tools and Streaming](/docs/tools-and-streaming) -- working with tools and streaming responses
- [Code Standards](/docs/code-standards) -- coding conventions and error handling patterns
