---
title: Conversation Management
---

The `Conversation` API provides automatic multi-turn history management backed by an actor. Every `Conversation` is `Clone + Send + 'static`, uses zero mutexes, and can be safely shared across async tasks.

---

## Architecture overview

`Conversation` is a thin handle to a `ConversationActor` that owns the conversation history. All mutations are serialized through the actor's mailbox, which means:

- **No mutexes** -- state is protected by the actor mailbox, not locks
- **Atomic reads** -- `len()`, `is_empty()`, and `should_exit()` use atomics for lock-free access
- **Watch channels** -- `history()` and `system_prompt()` use `tokio::sync::watch` for efficient snapshots
- **Clone + Send + 'static** -- the handle can be cloned and shared freely across tasks

```text
                  +-------------------+
  conv.send() -> | ConversationActor |  (owns Vec<Message>)
                  |  - push user msg  |
                  |  - call LLM       |
                  |  - push assistant  |
                  +-------------------+
                          |
                    watch channel
                          |
                  conv.history()  (snapshot)
```

---

## Creating a conversation

Use the `ConversationBuilder` returned by `ActonAI::conversation()`:

```rust
use acton_ai::prelude::*;

let runtime = ActonAI::builder()
    .app_name("my-app")
    .from_config()?
    .with_builtins()
    .launch()
    .await?;

let conv = runtime.conversation()
    .system("You are a helpful Rust tutor.")
    .build()
    .await;
```

### Builder methods

| Method | Description |
|---|---|
| `.system("prompt")` | Set the system prompt for all messages |
| `.restore(messages)` | Restore history from a previous session |
| `.with_exit_tool()` | Enable the built-in exit detection tool |
| `.without_exit_tool()` | Explicitly disable the exit tool |
| `.build().await` | Spawn the actor and return a `Conversation` |
| `.run_chat().await` | Build and immediately start an interactive chat loop |
| `.run_chat_with(config).await` | Build and start a chat loop with custom config |

---

## Sending messages and getting responses

The `send()` method is the primary way to interact with a conversation. It automatically:

1. Adds the user message to the history
2. Sends the full history to the LLM
3. Adds the assistant's response to the history
4. Returns the collected response

```rust
let response = conv.send("What is ownership in Rust?").await?;
println!("Assistant: {}", response.text);

// The conversation remembers context
let response = conv.send("How does borrowing relate to that?").await?;
println!("Assistant: {}", response.text);
// The LLM sees both the ownership question and its answer as context
```

The returned `CollectedResponse` includes:

- `text` -- the full response text
- `token_count` -- number of tokens used
- `stop_reason` -- why the LLM stopped generating
- `tool_calls` -- any tools the LLM invoked

---

## Streaming within conversations

For real-time token delivery, use `send_streaming()` with a token-handling actor:

```rust
use acton_ai::prelude::*;
use acton_ai::conversation::StreamToken;
use std::io::Write;

// Create a token-handling actor
let mut actor_runtime = runtime.runtime().clone();
let mut token_actor = actor_runtime.new_actor::<MyTokenPrinter>();
token_actor.mutate_on::<StreamToken>(|_actor, ctx| {
    print!("{}", ctx.message().text);
    std::io::stdout().flush().ok();
    Reply::ready()
});
let token_handle = token_actor.start().await;

// Stream tokens to the actor
let response = conv.send_streaming("Tell me about Rust's type system", &token_handle).await?;
println!(); // Newline after streaming
```

{% callout type="note" title="StreamToken message type" %}
The `StreamToken` message has a single field `text: String` containing the token. Register a handler for this message type on any actor to receive streaming tokens from conversations.
{% /callout %}

---

## History management

### Getting the history

`history()` returns a snapshot of the current conversation:

```rust
let messages = conv.history();
for msg in &messages {
    println!("{:?}: {}", msg.role, msg.content);
}
```

### Checking history size

Use the lock-free atomic accessors:

```rust
println!("Messages: {}", conv.len());
println!("Empty: {}", conv.is_empty());
```

### Clearing history

Reset the conversation to start fresh while keeping the system prompt:

```rust
conv.send("Topic A discussion...").await?;
conv.clear();  // Fire-and-forget, processed after in-flight sends
conv.send("Topic B discussion...").await?;
// The LLM only sees Topic B, not Topic A
```

### Restoring history

Load a previously saved conversation when building:

```rust
use acton_ai::messages::Message;

let saved_history = vec![
    Message::user("What is Rust?"),
    Message::assistant("Rust is a systems programming language..."),
];

let conv = runtime.conversation()
    .system("You are a Rust tutor.")
    .restore(saved_history)
    .build()
    .await;

// The conversation continues from where it left off
let response = conv.send("Tell me more about its memory model.").await?;
```

---

## System prompt management

### Setting the system prompt at build time

```rust
let conv = runtime.conversation()
    .system("You are a concise assistant. Answer in one sentence.")
    .build()
    .await;
```

### Changing the system prompt mid-conversation

```rust
// Read the current prompt
if let Some(prompt) = conv.system_prompt() {
    println!("Current: {}", prompt);
}

// Change it (takes effect on the next send)
conv.set_system_prompt("You are now a creative writing assistant.");

// Clear it entirely
conv.clear_system_prompt();
```

{% callout type="note" title="Fire-and-forget updates" %}
`set_system_prompt()` and `clear_system_prompt()` are fire-and-forget operations. They send a message to the actor and return immediately. The change takes effect on the next `send()` call.
{% /callout %}

---

## Exit tool and interactive chat loops

### The exit tool

The `Conversation` includes a built-in `exit_conversation` tool that the LLM can call when it detects the user wants to leave. When called, it sets an atomic flag you can check with `should_exit()`.

```rust
let conv = runtime.conversation()
    .system("Help the user. Use exit_conversation when they say goodbye.")
    .with_exit_tool()
    .build()
    .await;

loop {
    let input = read_user_input();
    let response = conv.send(&input).await?;
    println!("{}", response.text);

    if conv.should_exit() {
        println!("Goodbye!");
        break;
    }
}
```

You can also clear the exit flag for confirmation flows:

```rust
if conv.should_exit() {
    println!("Are you sure you want to leave? (yes/no)");
    let answer = read_user_input();
    if answer != "yes" {
        conv.clear_exit();  // Reset and continue
        continue;
    }
}
```

### `run_chat()` -- the minimal chat loop

`run_chat()` handles stdin reading, streaming, exit detection, and EOF in a single call:

```rust
use acton_ai::prelude::*;

ActonAI::builder()
    .app_name("chat")
    .from_config()?
    .with_builtins()
    .launch()
    .await?
    .conversation()
    .run_chat()
    .await?;
```

This is equivalent to building a conversation and calling `run_chat()` on it. The exit tool is automatically enabled, and a default system prompt is used if none was set.

### `run_chat_with()` -- customized chat loops

Use `ChatConfig` to customize prompts and input processing:

```rust
use acton_ai::prelude::*;
use acton_ai::conversation::ChatConfig;

let conv = runtime.conversation()
    .system("You are a coding assistant.")
    .build()
    .await;

conv.run_chat_with(
    ChatConfig::new()
        .user_prompt(">>> ")           // Custom input prompt
        .assistant_prompt("AI: ")      // Custom response prefix
        .map_input(|s| {               // Transform input before sending
            format!("[user:admin] {}", s)
        })
).await?;
```

### `ChatConfig` options

| Method | Default | Description |
|---|---|---|
| `.user_prompt(">>> ")` | `"You: "` | Prompt shown before user input |
| `.assistant_prompt("AI: ")` | `"Assistant: "` | Prefix before assistant responses |
| `.map_input(fn)` | None | Transform user input before sending to LLM |

The `map_input` callback is useful for injecting context, adding metadata, or preprocessing user input:

```rust
ChatConfig::new()
    .map_input(|input| {
        let timestamp = chrono::Local::now().format("%H:%M:%S");
        format!("[{}] {}", timestamp, input)
    })
```

### Default system prompt

When `run_chat()` or `run_chat_with()` is called without a system prompt, this default is used:

```text
You are a helpful assistant with access to various tools.
Use tools when appropriate to help the user.
When the user wants to end the conversation (says goodbye, bye, quit, exit, etc.),
use the exit_conversation tool.
```

---

## Zero-mutex design

The `Conversation` handle achieves thread safety without any `Mutex` or `RwLock`:

| Data | Synchronization | Access pattern |
|---|---|---|
| Conversation history | Actor mailbox + `watch::channel` | Writes serialized by mailbox; reads via `watch` snapshot |
| History length | `AtomicUsize` | Lock-free read with `Ordering::SeqCst` |
| Exit flag | `AtomicBool` | Lock-free read/write |
| Exit tool enabled | `AtomicBool` | Lock-free read/write |
| System prompt | `watch::channel` | Reads via `watch` snapshot |

This design means:
- `send()` blocks the mailbox during the LLM call, guaranteeing ordering
- `history()` returns an instant snapshot without waiting for in-flight sends
- `len()`, `is_empty()`, and `should_exit()` are always non-blocking

---

## Sharing conversations across tasks

Because `Conversation` is `Clone + Send + 'static`, you can share it across tokio tasks:

```rust
let conv = runtime.conversation()
    .system("You are helpful.")
    .build()
    .await;

// Clone for use in another task
let conv_clone = conv.clone();

let handle = tokio::spawn(async move {
    let response = conv_clone.send("Background question").await?;
    Ok::<_, ActonAIError>(response.text)
});

// Meanwhile, use the original
let response = conv.send("Foreground question").await?;
```

{% callout type="warning" title="Serialized sends" %}
While `Conversation` is safe to share across tasks, sends are serialized through the actor mailbox. Two concurrent `send()` calls will execute one after the other, not in parallel. This is by design -- it guarantees history consistency.
{% /callout %}

---

## Next steps

- [Multi-Agent Collaboration](/docs/multi-agent-collaboration) -- coordinate multiple conversations across agents
- [Error Handling](/docs/error-handling) -- handle `ActonAIError` from conversation operations
- [Testing Your Agents](/docs/testing) -- test conversation flows with mock providers
