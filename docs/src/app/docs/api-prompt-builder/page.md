---
title: PromptBuilder
---

Complete API reference for `PromptBuilder`, `ToolExecutorFn`, `ToolDefinition`, and related tool registration types.

---

## PromptBuilder

```rust
pub struct PromptBuilder { /* fields omitted */ }
```

A fluent builder for constructing and sending LLM prompts. Created via [`ActonAI::prompt()`](/docs/api-acton-ai) or [`ActonAI::continue_with()`](/docs/api-acton-ai), this builder lets you configure the request, register tools, set up streaming callbacks, and then execute the prompt.

`PromptBuilder` is **not** `Clone` or `Send` (it contains `FnMut` callbacks). It is designed to be built and consumed in a single expression chain ending with `.collect().await`.

### System prompt

#### `system()`

```rust
pub fn system(self, prompt: impl Into<String>) -> Self
```

Sets the system prompt for this request. The system prompt provides context and instructions to the LLM about how to respond.

```rust
runtime
    .prompt("What is the capital of France?")
    .system("Be concise. Answer in one word if possible.")
    .collect()
    .await?;
```

### Conversation history

#### `messages()`

```rust
pub fn messages(self, messages: impl IntoIterator<Item = Message>) -> Self
```

Sets conversation history for multi-turn conversations. When set, this replaces the initial user content passed to `prompt()`. The system prompt (if set via `.system()`) is automatically prepended.

```rust
let mut history = vec![
    Message::user("What is Rust?"),
    Message::assistant("Rust is a systems programming language..."),
];
history.push(Message::user("How does ownership work?"));

let response = runtime
    .prompt("")  // Ignored when messages() is set
    .system("You are a Rust expert.")
    .messages(history)
    .collect()
    .await?;
```

{% callout type="note" title="Prefer Conversation for multi-turn" %}
For multi-turn conversations, consider using [`Conversation`](/docs/api-conversation) instead. It manages history automatically, so you do not need to track and pass messages manually.
{% /callout %}

### Streaming callbacks

#### `on_start()`

```rust
pub fn on_start<F>(self, f: F) -> Self
where
    F: FnMut() + Send + 'static,
```

Sets a callback invoked when the LLM stream starts. Useful for displaying a "thinking" indicator.

```rust
runtime.prompt("Hello")
    .on_start(|| println!("Thinking..."))
    .collect()
    .await?;
```

#### `on_token()`

```rust
pub fn on_token<F>(self, f: F) -> Self
where
    F: FnMut(&str) + Send + 'static,
```

Sets a callback invoked for each token as it arrives from the LLM. Tokens are delivered in order. This is the primary way to stream output to the user.

```rust
runtime.prompt("Tell me a story.")
    .on_token(|token| print!("{token}"))
    .collect()
    .await?;
```

#### `on_end()`

```rust
pub fn on_end<F>(self, f: F) -> Self
where
    F: FnMut(StopReason) + Send + 'static,
```

Sets a callback invoked when the LLM stream ends. The callback receives the `StopReason` indicating why the LLM stopped generating.

`StopReason` variants: `EndTurn`, `MaxTokens`, `StopSequence`, `ToolUse`.

```rust
runtime.prompt("Hello")
    .on_end(|reason| println!("\n[Finished: {reason:?}]"))
    .collect()
    .await?;
```

### Provider selection

#### `provider()`

```rust
pub fn provider(self, name: impl Into<String>) -> Self
```

Selects which named provider handles this specific prompt. If not called, the default provider is used. Only relevant when multiple providers are configured.

```rust
// Use Claude for complex reasoning
runtime.prompt("Analyze this code...")
    .provider("claude")
    .collect()
    .await?;

// Use a fast provider for simple tasks
runtime.prompt("Summarize this text")
    .provider("fast")
    .collect()
    .await?;
```

### Sampling parameters

Control the randomness and diversity of LLM output on a per-prompt basis. These override any defaults set on the [provider configuration](/docs/providers-and-configuration#sampling-parameters).

#### `temperature()`

```rust
pub fn temperature(self, value: f64) -> Self
```

Sets the sampling temperature. Lower values (e.g. 0.0) make output more deterministic; higher values (e.g. 1.0) increase randomness.

```rust
runtime.prompt("Write a creative story")
    .temperature(0.9)
    .collect()
    .await?;
```

#### `top_p()`

```rust
pub fn top_p(self, value: f64) -> Self
```

Sets nucleus sampling. The model considers only tokens within this cumulative probability mass. For example, `0.9` means the model picks from tokens comprising the top 90% probability.

#### `top_k()`

```rust
pub fn top_k(self, value: u32) -> Self
```

Sets top-k sampling. The model considers only the `k` most likely tokens at each step. Supported by Anthropic and Ollama; ignored by OpenAI.

#### `frequency_penalty()`

```rust
pub fn frequency_penalty(self, value: f64) -> Self
```

Penalizes tokens based on how often they appear in the text so far. OpenAI only.

#### `presence_penalty()`

```rust
pub fn presence_penalty(self, value: f64) -> Self
```

Penalizes tokens that have appeared at all in the text so far. OpenAI only.

#### `seed()`

```rust
pub fn seed(self, value: u64) -> Self
```

Sets a seed for deterministic output (best-effort). OpenAI only.

#### `stop_sequences()`

```rust
pub fn stop_sequences(self, sequences: Vec<String>) -> Self
```

Sets custom stop sequences that cause the model to stop generating when encountered.

#### `sampling()`

```rust
pub fn sampling(self, params: SamplingParams) -> Self
```

Sets all sampling parameters at once using a `SamplingParams` struct. Useful when you have a pre-built configuration.

```rust
use acton_ai::prelude::*;

runtime.prompt("Analyze this data")
    .sampling(SamplingParams {
        temperature: Some(0.3),
        top_p: Some(0.9),
        ..Default::default()
    })
    .collect()
    .await?;
```

{% callout type="note" title="Override behavior" %}
Per-prompt sampling parameters are merged with provider defaults. Only the fields you set on the `PromptBuilder` override the provider -- unset fields fall back to the provider's configured defaults.
{% /callout %}

### Tool registration

There are three ways to register tools on a prompt, each offering a different level of control.

#### `tool()` -- inline closure

```rust
pub fn tool<F, Fut>(
    self,
    name: impl Into<String>,
    description: impl Into<String>,
    input_schema: serde_json::Value,
    executor: F,
) -> Self
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<serde_json::Value, ToolError>> + Send + 'static,
```

The most ergonomic way to add tools. You provide the tool name, description, JSON schema, and an async closure in one call.

```rust
runtime
    .prompt("What is 42 * 17?")
    .tool(
        "calculator",
        "Computes mathematical expressions",
        json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }),
        |args| async move {
            let expr = args["expression"].as_str().unwrap();
            Ok(json!({"result": calculate(expr)}))
        },
    )
    .collect()
    .await?;
```

#### `with_tool()` -- pre-built definition

```rust
pub fn with_tool<F, Fut>(self, definition: ToolDefinition, executor: F) -> Self
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<serde_json::Value, ToolError>> + Send + 'static,
```

Registers a tool using a pre-built `ToolDefinition`. Useful when you have tool definitions defined separately.

```rust
let calculator = ToolDefinition {
    name: "calculator".to_string(),
    description: "Evaluates math expressions".to_string(),
    input_schema: json!({
        "type": "object",
        "properties": {
            "expression": { "type": "string" }
        },
    }),
};

runtime.prompt("What is 2 + 2?")
    .with_tool(calculator, |args| async move {
        let expr = args["expression"].as_str().unwrap();
        Ok(json!({"result": calculate(expr)}))
    })
    .collect()
    .await?;
```

#### `with_tool_callback()` -- definition with result callback

```rust
pub fn with_tool_callback<F, Fut, C>(
    self,
    definition: ToolDefinition,
    executor: F,
    on_result: C,
) -> Self
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<serde_json::Value, ToolError>> + Send + 'static,
    C: FnMut(Result<&serde_json::Value, &str>) + Send + 'static,
```

Like `with_tool()`, but also registers a result callback that is invoked after the tool executes. The callback receives either the successful result value or an error message. Useful for logging, debugging, or updating UI state.

```rust
runtime.prompt("What is 2 + 2?")
    .with_tool_callback(
        calculator_def,
        |args| async move { Ok(json!({"result": 4})) },
        |result| {
            match result {
                Ok(value) => println!("Calculator returned: {value}"),
                Err(e) => println!("Calculator failed: {e}"),
            }
        },
    )
    .collect()
    .await?;
```

### Built-in tools

#### `use_builtins()`

```rust
pub fn use_builtins(self) -> Self
```

Enables the [built-in tools](/docs/api-builtin-tools) that were configured on the runtime via `with_builtins()` or `with_builtin_tools()`. This method is only needed when `manual_builtins()` was called on the builder. Otherwise, builtins are auto-enabled.

```rust
// Only needed with manual_builtins()
runtime.prompt("List files")
    .use_builtins()
    .collect()
    .await?;
```

### Tool execution limits

#### `max_tool_rounds()`

```rust
pub fn max_tool_rounds(self, max: usize) -> Self
```

Sets the maximum number of tool execution rounds. This prevents infinite loops if the LLM keeps requesting tools. Default is **10** rounds.

When the LLM requests a tool, the tool is executed, results are sent back, and the LLM continues. This constitutes one round. If the maximum is exceeded, `collect()` returns an error.

```rust
runtime.prompt("Complex multi-step task")
    .tool(...)
    .max_tool_rounds(5)
    .collect()
    .await?;
```

### Token forwarding

#### `token_target()`

```rust
pub fn token_target(self, handle: ActorHandle) -> Self
```

Sets a target actor to receive `StreamToken` messages during streaming. Used internally by `Conversation::send_streaming()`. The target actor must have a handler registered for `StreamToken`.

### Execution

#### `collect()`

```rust
pub async fn collect(self) -> Result<CollectedResponse, ActonAIError>
```

Sends the prompt and collects the complete response. This is the terminal method that consumes the builder and executes the request.

The method:
1. Creates a temporary actor to collect tokens
2. Subscribes to streaming events
3. Sends the request to the LLM provider
4. If tools are registered and the LLM requests them, executes the tools, sends results back, and repeats until the LLM completes or `max_tool_rounds` is exceeded
5. Returns the collected response

Callbacks (`on_start`, `on_token`, `on_end`) fire during streaming.

**Errors:**
- The runtime has been shut down
- The stream fails to complete
- Maximum tool rounds exceeded
- Named provider not found

```rust
let response = runtime
    .prompt("What is 2 + 2?")
    .on_token(|t| print!("{t}"))
    .collect()
    .await?;

println!("\nFull response: {}", response.text);
println!("Tokens: {}", response.token_count);
println!("Stop reason: {:?}", response.stop_reason);
```

---

## CollectedResponse

```rust
pub struct CollectedResponse {
    pub text: String,
    pub stop_reason: StopReason,
    pub token_count: usize,
    pub tool_calls: Vec<ExecutedToolCall>,
}
```

Response collected from a completed stream. Returned by `PromptBuilder::collect()`.

| Field | Type | Description |
|---|---|---|
| `text` | `String` | The complete text generated by the LLM |
| `stop_reason` | `StopReason` | Why the LLM stopped generating |
| `token_count` | `usize` | Number of tokens in the response |
| `tool_calls` | `Vec<ExecutedToolCall>` | Tool calls executed during the conversation loop |

### Methods

```rust
pub fn is_complete(&self) -> bool      // true if stop_reason is EndTurn
pub fn is_truncated(&self) -> bool     // true if stop_reason is MaxTokens
pub fn needs_tool_call(&self) -> bool  // true if stop_reason is ToolUse
pub fn has_tool_calls(&self) -> bool   // true if any tools were called
```

---

## ExecutedToolCall

```rust
pub struct ExecutedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    pub result: Result<serde_json::Value, String>,
}
```

Record of a tool call that was executed during the conversation. Captures the arguments passed and the result returned.

### Methods

```rust
pub fn is_success(&self) -> bool  // true if the tool execution succeeded
pub fn is_error(&self) -> bool    // true if the tool execution failed
```

---

## ToolDefinition

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}
```

Defines a tool that can be used by the LLM. The `input_schema` is a JSON Schema object describing the parameters the tool accepts.

| Field | Type | Description |
|---|---|---|
| `name` | `String` | The tool name (used by the LLM to invoke it) |
| `description` | `String` | Human-readable description of what the tool does |
| `input_schema` | `serde_json::Value` | JSON Schema for the tool's input parameters |

---

## ToolExecutorFn

```rust
pub trait ToolExecutorFn: Send + Sync {
    fn call(&self, args: serde_json::Value) -> ToolFuture;
}
```

Trait for tool execution functions. This trait allows both closures and custom executors to be used as tool handlers. You typically do not implement this directly -- instead use the `tool()`, `with_tool()`, or `with_tool_callback()` methods on `PromptBuilder`, which wrap closures automatically.

The return type `ToolFuture` is:

```rust
type ToolFuture = Pin<Box<dyn Future<Output = Result<serde_json::Value, ToolError>> + Send>>;
```

---

## StopReason

```rust
pub enum StopReason {
    EndTurn,       // Normal completion
    MaxTokens,     // Reached token limit
    StopSequence,  // Hit a stop sequence
    ToolUse,       // Model wants to call tools
}
```

Indicates why the LLM stopped generating tokens.

---

## Common patterns

### Streaming with tool use

```rust
let response = runtime
    .prompt("What files are in the current directory?")
    .system("Use tools to interact with the filesystem.")
    .tool(
        "list_files",
        "Lists files in a directory",
        json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        |args| async move {
            let path = args["path"].as_str().unwrap_or(".");
            let entries: Vec<String> = std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            Ok(json!({"files": entries}))
        },
    )
    .on_token(|t| print!("{t}"))
    .max_tool_rounds(3)
    .collect()
    .await?;

// Inspect what tools were called
for tc in &response.tool_calls {
    println!("Tool: {} -> {:?}", tc.name, tc.result);
}
```

### Multi-provider routing

```rust
// Route different prompts to different providers
let quick = runtime.prompt("Quick question").provider("fast").collect().await?;
let deep = runtime.prompt("Deep analysis").provider("claude").collect().await?;
```
