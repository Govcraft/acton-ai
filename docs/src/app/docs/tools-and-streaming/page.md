---
title: Tools and Streaming
---

acton-ai gives LLMs the ability to call **tools** (also called function calling) and delivers responses via **streaming** so your users see output as it is generated. This page covers the tool system, built-in tools, and the three streaming layers.

## How LLM tool calling works

When tools are registered on a prompt, their definitions (name, description, JSON schema) are sent to the LLM alongside the conversation. The LLM can then decide to call a tool instead of (or in addition to) generating text:

1. You send a prompt with tool definitions to the LLM
2. The LLM responds with a `ToolUse` stop reason and one or more tool call requests
3. acton-ai executes the tools and sends the results back to the LLM
4. The LLM generates a final text response incorporating the tool results
5. Steps 2-4 repeat until the LLM responds with `EndTurn` (or the max rounds limit is reached)

This loop is handled automatically by `PromptBuilder::collect()`.

## Registering tools with `.tool()`

The simplest way to add a tool is the `.tool()` method on `PromptBuilder`. You provide a name, description, JSON schema for the input, and an async closure that executes the tool:

```rust
use acton_ai::prelude::*;
use serde_json::json;

let response = runtime
    .prompt("What is 42 * 17?")
    .system("Use the calculator for math.")
    .tool(
        "calculator",
        "Evaluates mathematical expressions",
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate"
                }
            },
            "required": ["expression"]
        }),
        |args| async move {
            let expr = args["expression"].as_str().unwrap_or("0");
            // Your calculation logic here
            Ok(json!({"result": expr}))
        },
    )
    .on_token(|t| print!("{t}"))
    .collect()
    .await?;
```

## Registering tools with `.with_tool()`

When you have a pre-built `ToolDefinition`, use `.with_tool()`:

```rust
let calculator = ToolDefinition {
    name: "calculator".to_string(),
    description: "Evaluates mathematical expressions".to_string(),
    input_schema: json!({
        "type": "object",
        "properties": {
            "expression": { "type": "string" }
        },
        "required": ["expression"]
    }),
};

let response = runtime
    .prompt("What is 2 + 2?")
    .with_tool(calculator, |args| async move {
        let expr = args["expression"].as_str().unwrap_or("0");
        Ok(json!({"result": calculate(expr)}))
    })
    .collect()
    .await?;
```

## Tool result callbacks with `.with_tool_callback()`

To observe tool execution results (for logging, UI updates, or debugging), use `.with_tool_callback()`:

```rust
let response = runtime
    .prompt("What is 42 * 17?")
    .with_tool_callback(
        calculator_definition,
        |args| async move {
            let expr = args["expression"].as_str().unwrap_or("0");
            Ok(json!({"result": calculate(expr)}))
        },
        |result| {
            match result {
                Ok(value) => eprintln!("[Calculator returned: {value}]"),
                Err(e) => eprintln!("[Calculator error: {e}]"),
            }
        },
    )
    .collect()
    .await?;
```

The callback receives `Result<&serde_json::Value, &str>` -- either the successful JSON result or the error message.

## Tool execution rounds and `max_tool_rounds()`

By default, the tool execution loop runs for up to **10 rounds**. Each round is one LLM call that may request tool executions. If the LLM keeps requesting tools beyond this limit, `collect()` returns an error.

```rust
// Allow up to 20 tool rounds for complex tasks
let response = runtime
    .prompt("Analyze this project structure")
    .tool("read_file", ...)
    .tool("list_directory", ...)
    .max_tool_rounds(20)
    .collect()
    .await?;

// Limit to 3 rounds for simple tasks
let response = runtime
    .prompt("Quick calculation")
    .tool("calculator", ...)
    .max_tool_rounds(3)
    .collect()
    .await?;
```

{% callout type="note" title="Tool calls in CollectedResponse" %}
After `collect()` completes, `response.tool_calls` contains a `Vec<ExecutedToolCall>` with every tool call that was executed during the conversation, including the arguments passed and the results returned (or errors).
{% /callout %}

## Built-in tools

acton-ai ships with a set of ready-to-use tools. Enable them at the builder level:

```rust
// Enable all built-in tools
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .with_builtins()
    .launch()
    .await?;

// Enable only specific tools
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .with_builtin_tools(&["read_file", "write_file", "glob", "bash"])
    .launch()
    .await?;
```

### Available built-in tools

| Tool | Description |
|---|---|
| `read_file` | Read file contents with line numbers |
| `write_file` | Write content to files |
| `edit_file` | Make targeted string replacements in files |
| `list_directory` | List directory contents with metadata |
| `glob` | Find files matching glob patterns |
| `grep` | Search file contents with regex |
| `bash` | Execute shell commands |
| `calculate` | Evaluate mathematical expressions |
| `web_fetch` | Fetch content from URLs |
| `rust_code` | Execute compiler-verified Rust code (requires Rust toolchain) |

### Auto-builtins vs manual builtins

When you use `.with_builtins()` or `.with_builtin_tools()`, the tools are **automatically enabled on every prompt**. You do not need to call `.use_builtins()` on each prompt:

```rust
// Auto-builtins (default with .with_builtins())
runtime.prompt("List files in src/").collect().await?;
// Built-in tools are already available ^
```

If you want builtins available but not auto-enabled, use `.manual_builtins()`:

```rust
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .with_builtins()
    .manual_builtins()   // Opt out of auto-enable
    .launch()
    .await?;

// Now you must explicitly enable builtins per-prompt
runtime.prompt("List files")
    .use_builtins()      // Required when manual_builtins() is set
    .collect()
    .await?;
```

This is useful when you only want tools on certain prompts (e.g., coding tasks but not simple Q&A).

## The three streaming layers

acton-ai offers three ways to handle streamed LLM responses, from simplest to most powerful:

### Layer 1: Closure callbacks (PromptBuilder)

The simplest approach. Set callbacks directly on the `PromptBuilder`:

```rust
runtime
    .prompt("Tell me a story")
    .on_start(|| eprintln!("Thinking..."))
    .on_token(|token| print!("{token}"))
    .on_end(|reason| eprintln!("\n[Done: {reason:?}]"))
    .collect()
    .await?;
```

- `on_start(|| ...)` -- Called once when the LLM begins responding
- `on_token(|token| ...)` -- Called for each token as it arrives
- `on_end(|reason| ...)` -- Called when the response finishes, with the `StopReason`

This is sufficient for most applications, including terminal UIs and web streaming.

### Layer 2: Token target (actor messages)

Send tokens to another actor as `StreamToken` messages. This is used internally by `Conversation::send_streaming()` and is useful when you need to process tokens in an actor context:

```rust
runtime
    .prompt("Tell me a story")
    .token_target(my_actor_handle)   // Forward tokens as StreamToken messages
    .collect()
    .await?;
```

The target actor must have a handler registered for `StreamToken`.

### Layer 3: StreamHandler trait (custom actors)

For full control, implement the `StreamHandler` trait on your own actor. This gives you access to actor state during streaming:

```rust
use acton_ai::prelude::*;

#[acton_actor]
struct WordCounter {
    buffer: String,
    word_count: usize,
}

impl StreamHandler for WordCounter {
    fn on_start(&mut self, correlation_id: &CorrelationId) {
        self.buffer.clear();
        self.word_count = 0;
    }

    fn on_token(&mut self, token: &str) {
        self.buffer.push_str(token);
        self.word_count += token.split_whitespace().count();
    }

    fn on_end(&mut self, reason: StopReason) -> StreamAction {
        println!("Received {} words", self.word_count);
        StreamAction::Complete
    }
}
```

The `StreamAction` return value controls what happens after the stream:

| Action | Behavior |
|---|---|
| `StreamAction::Continue` | Keep the actor alive for more streams |
| `StreamAction::Complete` | Signal completion to any waiters |
| `StreamAction::Stop` | Stop the actor |

{% callout type="note" title="When to use which layer" %}
**Layer 1 (closures)** is right for 90% of use cases. Use **Layer 2 (token_target)** when you need actor-to-actor token forwarding. Use **Layer 3 (StreamHandler)** when you need custom stateful stream processing with full actor lifecycle control.
{% /callout %}

## Under the hood: broadcast-based streaming

All streaming in acton-ai works through the actor message broker. The `LLMProvider` actor broadcasts four message types:

- `LLMStreamStart` -- Stream has started (contains correlation ID)
- `LLMStreamToken` -- A single token (contains correlation ID and token text)
- `LLMStreamToolCall` -- A tool call request from the LLM
- `LLMStreamEnd` -- Stream has ended (contains correlation ID and stop reason)

When you call `.collect()`, a temporary `StreamCollector` actor is spawned that subscribes to these messages, collects tokens into a buffer, and signals completion. The collector owns all mutable state internally, so no external locks are needed during streaming.

## Next steps

- [Providers and Configuration](/docs/providers-and-configuration) -- Configure LLM providers and rate limits
- [The Two API Levels](/docs/two-api-levels) -- Understand when to use the high-level vs low-level API
- [Conversation Management](/docs/conversation-management) -- Automatic multi-turn history management
