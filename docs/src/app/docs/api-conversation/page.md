---
title: Conversation
---

Complete API reference for `Conversation`, `ConversationBuilder`, `ChatConfig`, `StreamToken`, and the `DEFAULT_SYSTEM_PROMPT` constant.

---

## Constants

### `DEFAULT_SYSTEM_PROMPT`

```rust
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant with access to various tools. \
Use tools when appropriate to help the user. \
When the user wants to end the conversation (says goodbye, bye, quit, exit, etc.), \
use the exit_conversation tool.";
```

The default system prompt used by `Conversation::run_chat()` when no system prompt has been set. It instructs the LLM to use tools and detect when the user wants to exit.

---

## Conversation

```rust
pub struct Conversation { /* fields omitted */ }
```

A managed conversation with automatic history tracking. `Conversation` is backed by an actor that serializes all operations through its mailbox, making it `Clone + Send + 'static` with all methods taking `&self`. This enables safe sharing across tasks.

Each call to `send()` automatically:
1. Adds the user message to history
2. Sends the full history to the LLM
3. Adds the assistant's response to history
4. Returns the response

Create a `Conversation` using [`ActonAI::conversation()`](/docs/api-acton-ai) and `ConversationBuilder::build().await`.

{% callout type="note" title="Actor-backed concurrency" %}
`Conversation` uses zero mutexes. All state is owned by an internal actor whose mailbox serializes access. The public handle uses atomics and watch channels for lock-free reads of history length and exit state.
{% /callout %}

### Sending messages

#### `send()`

```rust
pub async fn send(&self, content: impl Into<String>) -> Result<CollectedResponse, ActonAIError>
```

Sends a message and receives a response, automatically managing history. This is the primary method for interacting with a conversation. If builtins were configured with `with_builtins()`, they are automatically available.

```rust
let response = conv.send("What is the capital of France?").await?;
println!("{}", response.text);

// Conversation remembers context
let response = conv.send("What about Germany?").await?;
println!("{}", response.text);
```

#### `send_streaming()`

```rust
pub async fn send_streaming(
    &self,
    content: impl Into<String>,
    token_handle: &ActorHandle,
) -> Result<CollectedResponse, ActonAIError>
```

Sends a message with streaming tokens delivered to a user-provided actor. The `token_handle` actor must have a handler registered for [`StreamToken`](#streamtoken) messages. Tokens are delivered in order as they arrive.

```rust
// Create a token-handling actor
let mut printer = runtime.runtime().clone().new_actor::<MyPrinter>();
printer.mutate_on::<StreamToken>(|_actor, ctx| {
    print!("{}", ctx.message().text);
    Reply::ready()
});
let printer_handle = printer.start().await;

let response = conv.send_streaming("Tell me a story", &printer_handle).await?;
```

### History management

#### `history()`

```rust
pub fn history(&self) -> Vec<Message>
```

Returns a snapshot of the conversation history. Useful for serializing, debugging, or building custom UIs.

```rust
for message in conv.history() {
    println!("{}: {}", message.role, message.content);
}
```

#### `clear()`

```rust
pub fn clear(&self)
```

Clears the conversation history while keeping the system prompt. The clear is sent as a fire-and-forget message and will be processed after any in-flight sends complete.

```rust
conv.send("Topic A discussion...").await?;
conv.clear();  // Start fresh
conv.send("Topic B discussion...").await?;
```

#### `len()`

```rust
pub fn len(&self) -> usize
```

Returns the number of messages in the conversation history. This is a lock-free atomic read.

#### `is_empty()`

```rust
pub fn is_empty(&self) -> bool
```

Returns `true` if the conversation history is empty.

### System prompt

#### `system_prompt()`

```rust
pub fn system_prompt(&self) -> Option<String>
```

Returns the current system prompt, if set.

#### `set_system_prompt()`

```rust
pub fn set_system_prompt(&self, prompt: impl Into<String>)
```

Sets or updates the system prompt. The change is fire-and-forget and takes effect on the next `send()` call.

#### `clear_system_prompt()`

```rust
pub fn clear_system_prompt(&self)
```

Clears the system prompt.

### Exit detection

#### `should_exit()`

```rust
pub fn should_exit(&self) -> bool
```

Returns `true` if the `exit_conversation` tool has been called by the LLM. Only meaningful when the conversation was built with `with_exit_tool()` or is running via `run_chat()`.

```rust
loop {
    let response = conv.send(&input).await?;

    if conv.should_exit() {
        println!("Goodbye!");
        break;
    }
}
```

#### `clear_exit()`

```rust
pub fn clear_exit(&self)
```

Clears the exit flag. Use this if you want to continue the conversation after the exit tool was called (for example, after asking for confirmation).

```rust
if conv.should_exit() {
    println!("Are you sure you want to leave?");
    let confirmation = read_input();
    if confirmation != "yes" {
        conv.clear_exit();
        continue;
    }
}
```

#### `exit_requested()`

```rust
pub fn exit_requested(&self) -> Arc<AtomicBool>
```

Returns a clone of the exit flag `Arc<AtomicBool>`. Useful for sharing the exit state with other components.

#### `is_exit_tool_enabled()`

```rust
pub fn is_exit_tool_enabled(&self) -> bool
```

Returns whether the exit tool is enabled for this conversation.

### Chat loops

#### `run_chat()`

```rust
pub async fn run_chat(&self) -> Result<(), ActonAIError>
```

Runs a complete terminal chat loop. Handles reading from stdin, streaming responses to stdout, and checking for exit. The exit tool is automatically enabled. If no system prompt is set, `DEFAULT_SYSTEM_PROMPT` is used.

The loop:
- Prints "You: " and reads a line from stdin
- Prints "Assistant: " and streams the response
- Checks `should_exit()` after each exchange
- Exits on EOF (Ctrl+D) or when the exit tool fires

```rust
runtime.conversation()
    .system("You are a helpful coding assistant.")
    .build()
    .await
    .run_chat()
    .await?;
```

#### `run_chat_with()`

```rust
pub async fn run_chat_with(&self, config: ChatConfig) -> Result<(), ActonAIError>
```

Runs a complete terminal chat loop with custom configuration. Like `run_chat()`, but allows customization of prompts and input transformation via `ChatConfig`.

```rust
conv.run_chat_with(
    ChatConfig::new()
        .user_prompt(">>> ")
        .assistant_prompt("AI: ")
        .map_input(|s| format!("[context] {}", s))
).await?;
```

---

## ConversationBuilder

```rust
pub struct ConversationBuilder { /* fields omitted */ }
```

Builder for creating a `Conversation`. Created via [`ActonAI::conversation()`](/docs/api-acton-ai).

### Configuration

#### `system()`

```rust
pub fn system(self, prompt: impl Into<String>) -> Self
```

Sets the system prompt for the conversation. Applied to every message exchange.

```rust
runtime.conversation()
    .system("You are a concise assistant. Answer in one sentence.")
    .build()
    .await;
```

#### `restore()`

```rust
pub fn restore(self, messages: impl IntoIterator<Item = Message>) -> Self
```

Restores conversation history from a previous session. Use this to continue a conversation that was previously saved or to inject context from an external source.

```rust
let saved: Vec<Message> = load_from_database();

let conv = runtime.conversation()
    .system("Be helpful.")
    .restore(saved)
    .build()
    .await;
```

#### `with_exit_tool()`

```rust
pub fn with_exit_tool(self) -> Self
```

Enables the built-in `exit_conversation` tool for this conversation. When the LLM calls this tool (typically when the user says goodbye), the conversation's exit flag is set. Check the flag with `Conversation::should_exit()`.

```rust
let conv = runtime.conversation()
    .system("Be helpful. Use exit_conversation when the user says goodbye.")
    .with_exit_tool()
    .build()
    .await;

loop {
    let response = conv.send(&input).await?;
    if conv.should_exit() {
        break;
    }
}
```

#### `without_exit_tool()`

```rust
pub fn without_exit_tool(self) -> Self
```

Explicitly disables the exit tool. This is the default unless `with_exit_tool()` is called.

{% callout type="note" title="run_chat auto-enables exit tool" %}
If you later call `run_chat()` or `run_chat_with()`, the exit tool is automatically enabled regardless of this setting, because the chat loop depends on it for graceful termination.
{% /callout %}

### Building and running

#### `build()`

```rust
pub async fn build(self) -> Conversation
```

Builds the conversation by spawning a `ConversationActor`. The returned `Conversation` is `Clone + Send + 'static`.

```rust
let conv = runtime.conversation()
    .system("You are a helpful assistant.")
    .build()
    .await;
```

#### `run_chat()`

```rust
pub async fn run_chat(self) -> Result<(), ActonAIError>
```

Convenience method that combines `build()` and `Conversation::run_chat()` into a single call.

```rust
// Minimal chat -- build and run in one step
runtime.conversation()
    .system("You are a coding assistant.")
    .run_chat()
    .await?;
```

#### `run_chat_with()`

```rust
pub async fn run_chat_with(self, config: ChatConfig) -> Result<(), ActonAIError>
```

Convenience method that combines `build()` and `Conversation::run_chat_with()`.

```rust
runtime.conversation()
    .system("You are helpful.")
    .run_chat_with(
        ChatConfig::new()
            .user_prompt(">>> ")
            .assistant_prompt("AI: ")
    )
    .await?;
```

---

## ChatConfig

```rust
pub struct ChatConfig { /* fields omitted */ }
```

Configuration for the terminal chat loop. Created via `ChatConfig::new()` and passed to `Conversation::run_chat_with()`.

### Constructor

#### `new()`

```rust
pub fn new() -> Self
```

Creates a new `ChatConfig` with default settings:

| Setting | Default |
|---|---|
| User prompt | `"You: "` |
| Assistant prompt | `"Assistant: "` |
| Input mapper | None |

### Methods

#### `user_prompt()`

```rust
pub fn user_prompt(self, prompt: impl Into<String>) -> Self
```

Sets the prompt shown before user input.

#### `assistant_prompt()`

```rust
pub fn assistant_prompt(self, prompt: impl Into<String>) -> Self
```

Sets the prefix shown before assistant responses.

#### `map_input()`

```rust
pub fn map_input<F>(self, f: F) -> Self
where
    F: FnMut(&str) -> String + Send + 'static,
```

Sets a function to transform user input before sending to the LLM. Use this to add context, preprocess input, or inject system information.

```rust
ChatConfig::new()
    .user_prompt(">>> ")
    .assistant_prompt("AI: ")
    .map_input(|s| format!("[User context: admin] {}", s))
```

---

## StreamToken

```rust
#[derive(Clone, Debug)]
pub struct StreamToken {
    pub text: String,
}
```

A streamed token message sent from the conversation to a user-provided actor. Register a handler for this message type on your actor to receive individual tokens during `send_streaming()`.

| Field | Type | Description |
|---|---|---|
| `text` | `String` | The token text |

```rust
actor.mutate_on::<StreamToken>(|_actor, ctx| {
    print!("{}", ctx.message().text);
    std::io::stdout().flush().ok();
    Reply::ready()
});
```

---

## Common patterns

### Basic multi-turn conversation

```rust
let conv = runtime.conversation()
    .system("You are a helpful Rust tutor.")
    .build()
    .await;

let r1 = conv.send("What is ownership in Rust?").await?;
println!("{}", r1.text);

let r2 = conv.send("How does borrowing relate to that?").await?;
println!("{}", r2.text);

// Inspect full history
println!("Messages so far: {}", conv.len());
```

### Restoring a saved conversation

```rust
let saved_history = vec![
    Message::user("What is Rust?"),
    Message::assistant("Rust is a systems programming language..."),
];

let conv = runtime.conversation()
    .system("You are a Rust tutor.")
    .restore(saved_history)
    .build()
    .await;

// Continues from saved context
let response = conv.send("Tell me more about lifetimes.").await?;
```

### Five-line terminal chat

```rust
ActonAI::builder()
    .app_name("chat")
    .ollama("qwen2.5:7b")
    .with_builtins()
    .launch()
    .await?
    .conversation()
    .run_chat()
    .await?;
```

### Custom chat loop with streaming

```rust
let conv = runtime.conversation()
    .system("You are a helpful assistant.")
    .with_exit_tool()
    .build()
    .await;

loop {
    let input = read_user_input();

    // Stream tokens to a custom actor
    let response = conv.send_streaming(&input, &my_token_actor).await?;

    if conv.should_exit() {
        println!("Goodbye!");
        break;
    }
}
```

### Sharing a conversation across tasks

```rust
let conv = runtime.conversation()
    .system("You are helpful.")
    .build()
    .await;

// Conversation is Clone + Send -- safe to share
let conv2 = conv.clone();
tokio::spawn(async move {
    // Messages are serialized by the actor mailbox
    conv2.send("Background task question").await.ok();
});

conv.send("Main task question").await?;
```
