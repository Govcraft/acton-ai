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
conv.clear().await;  // Enqueued before anything you send next
conv.send("Topic B discussion...").await?;
// The LLM only sees Topic B, not Topic A
```

Awaiting the clear is what makes the ordering above a guarantee: mailboxes are
FIFO per sender, so the clear is in the actor's mailbox before the call returns
and cannot be overtaken by the `send()` that follows it.

If you need to *read* the cleared history rather than just order against it,
follow up with `sync()`:

```rust
conv.clear().await;
conv.sync().await?;
assert!(conv.is_empty());
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

## Context window management

Every turn of a `Conversation` sends the **entire accumulated history** to the LLM. Without bounds, a long-lived session eventually exceeds the provider's context window and fails at the model boundary, and token cost per turn grows linearly with session length.

By default, acton-ai truncates the history on each turn to fit a configurable token budget using the `KeepRecent` strategy — the newest user message is always kept; older turns are dropped until everything fits. The system prompt is carried out-of-band by the prompt builder and is never subject to truncation.

### Default budget

The runtime resolves the budget at `launch()` time with this precedence:

1. Per-provider `context_window_tokens` for the default provider (if set).
2. Global `[context] max_tokens` in `acton-ai.toml`.
3. Built-in default of 8192 tokens (with 1024 reserved for the response).

The default token estimator is `tiktoken-rs`: `o200k_base` for GPT-4o / o-series models, `cl100k_base` for GPT-4 / GPT-3.5, and `cl100k_base` as a fallback for Anthropic and Ollama models (accurate ±10% — sufficient for budgeting).

### Configuring via TOML

```toml
[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
context_window_tokens = 32000    # native limit for this model

[providers.claude]
type = "anthropic"
model = "claude-sonnet-4-20250514"
context_window_tokens = 200000   # Claude's native limit

[context]
max_tokens = 8192                # fallback when provider doesn't set it
reserved_for_response = 1024
strategy = "keep-recent"         # "keep-recent" | "keep-system-and-recent" | "keep-ends"
```

### Overriding or opting out in code

```rust
use acton_ai::prelude::*;
use acton_ai::memory::{ContextWindow, ContextWindowConfig, TruncationStrategy};

// Custom window for the whole runtime
let cw = ContextWindow::new(
    ContextWindowConfig {
        max_tokens: 16_384,
        truncation_strategy: TruncationStrategy::KeepEnds,
        reserved_for_response: 2048,
        tokens_per_char: 0.25,
    }
);

let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .context_window(cw)
    .launch()
    .await?;

// Or opt out entirely (unbounded history per turn — the pre-wiring behavior)
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .without_context_window()
    .launch()
    .await?;
```

Per-`Conversation` overrides work the same way:

```rust
let conv = runtime.conversation()
    .system("You are brief.")
    .without_context_window()    // this conversation ships full history
    .build()
    .await;
```

### Observing truncation

When history is actually clipped, a `tracing::warn!` fires once per turn with the drop counts. With `-v` on the CLI:

```text
WARN acton_ai::conversation: truncated conversation history to fit context window
    dropped_messages=12 dropped_tokens=4231 kept_messages=8 kept_tokens=7801 max_tokens=8192
```

---

## Auto-compaction

Truncation runs **between** turns. It does nothing **inside** one, and inside one is where a history most easily runs away: a turn that works through tools appends an assistant turn and every tool result on each round, with nothing bounding the result. A long tool loop eventually exceeds the provider's window and fails mid-turn, after the earlier rounds are already paid for.

Auto-compaction bounds that. When the running history crosses a fraction of the available budget, the prompt loop — between rounds, never while a tool exchange is in flight — sends the older history back to the **same provider** with a fixed summarization prompt, and splices the model's own summary in where the elided messages were, keeping the last few exchanges verbatim:

```text
[system]   kept as-is, never elided
[user]     "[conversation compacted] Earlier messages were summarized…"
           followed by the provider-written summary of everything removed
[ ... ]    the last N exchanges, verbatim
```

This is deliberately different from truncation. Dropping the oldest exchanges silently erases the user's original request and the model has no way to know: it answers a question it can no longer see. The summary tells the model what it forgot — and because the model that will continue the conversation is the one that wrote it, the summary keeps what *it* considers load-bearing.

The summarization is a paid request like any other round: it goes to the same provider, under the same budget caps, and its usage folds into the turn's totals. If it fails — provider error, empty reply, budget refusal — the turn proceeds with its full history and takes its chances, which is strictly better than proceeding with a hole where its history used to be; a failed attempt also stops compaction for the rest of that turn rather than paying for a doomed retry every round.

### Enabling it

Off by default: an unconfigured runtime behaves exactly as before. The fallback `max_tokens` of 8192 is far below the native window of most providers, so compacting by default would summarize history that was never in danger — and every summary costs tokens. Set a realistic budget first — per-provider `context_window_tokens` is the best place — then switch it on.

```toml
[providers.claude]
type = "anthropic"
model = "claude-sonnet-4-20250514"
context_window_tokens = 200000

[context]
auto_compact = true        # off unless this says otherwise
compact_threshold = 0.8    # compact at 80% of the available budget
keep_recent_turns = 3      # trailing exchanges kept verbatim
```

Or in code:

```rust
use acton_ai::memory::{CompactionConfig, KeepRecentTurns};

let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .compaction(
        CompactionConfig::default()
            .with_keep_recent_turns(KeepRecentTurns::new(5)?)
    )
    .launch()
    .await?;

// Or refuse it, including an `auto_compact = true` in TOML that would
// otherwise switch it on.
let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .without_compaction()
    .launch()
    .await?;
```

Compaction needs a context window to measure against. With `without_context_window()` there is no budget, so a policy is inert rather than an error.

### What it will not do

- **Split an exchange.** `keep_recent_turns` counts whole exchanges: an assistant turn carrying tool calls travels with the tool results answering it. A `tool_use` block with no matching `tool_result` is rejected by every provider for the rest of the conversation, not just for the request that introduced it.
- **Elide the system message.** It is held aside and re-emitted first.
- **Compact when it would not help.** If the summary would be no smaller than the text it replaces — short histories summarize to more than themselves — the pass declines and nothing changes. This is also what stops an already-compacted history from being rewritten, and paid for, on every round.
- **Compact a history that fits.** A policy in force is not a licence to rewrite a history that was never in danger, or to spend money summarizing it.
- **Interrupt the caller's stream.** The summarization round fires none of the turn's streaming callbacks; it is framework traffic, not the model's answer.

### Observing it

Compaction rewrites what the model sees, so it is loud about every pass — a framework that silently deletes context is indistinguishable, from the outside, from a model that ignores it.

Every pass logs at `info` and broadcasts `TurnLifecycle::ContextCompacted`:

```text
INFO acton_ai::prompt: compacted conversation history to stay within the context window
    messages_before=23 messages_after=4 messages_elided=20
    tokens_before=7904 tokens_after=1220 max_tokens=8192
```

The `CollectedResponse` a turn returns carries one `CompactionRecord` per pass, with the summary text and the measured effect; `record.as_message()` is the exact marked message the model saw, ready to store. The CLI's `chat` and `run-job` sessions persist those rows automatically, so a stored session records that — and what — the model was told it forgot.

`acton-ai status` reports the running total once it is nonzero:

```text
Compaction: 7 histor(ies) compacted mid-turn
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

// Change it (takes effect on every subsequently enqueued send)
conv.set_system_prompt("You are now a creative writing assistant.").await;

// Clear it entirely
conv.clear_system_prompt().await;
```

{% callout type="note" title="Enqueued, not yet applied" %}
`set_system_prompt()` and `clear_system_prompt()` return once the change is in the actor's mailbox. That is enough to guarantee it applies to every `send()` you issue afterwards, but the actor may not have processed it yet — so `system_prompt()` can still report the old value immediately after. Call `sync()` first if you need to read it back.
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
