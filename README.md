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
- **MCP client** — Consume tools from external Model Context Protocol servers (stdio or streamable HTTP) under supervised, self-reconnecting connections
- **Tool execution loop** — Automatic tool calling and result handling until completion
- **Typed structured output** — `extract::<T>()` returns a schema-validated Rust value, with automatic repair rounds when the model gets the shape wrong
- **Two API levels** — Simple facade for common cases, full actor access for advanced control
- **TOML configuration** — Define providers and settings in config files
- **Process sandboxing** — Portable subprocess isolation for tool execution with rlimits, timeouts, and optional Linux hardening (landlock + seccomp)
- **Rate limiting** — Built-in request and token limits per provider
- **Actor-based architecture** — Fault-tolerant, concurrent design via [acton-reactive](https://docs.rs/acton-reactive)

## Installation

### CLI (Arch Linux)

```bash
yay -S acton-ai-bin
```

### CLI (from source)

```bash
cargo install acton-ai
```

### Library

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

# Optional: ProcessSandbox for tool isolation
# Runs sandboxed tools in a subprocess with rlimits, timeouts, and
# (on Linux) best-effort landlock + seccomp hardening.
[sandbox]
hardening = "besteffort"    # "off" | "besteffort" | "enforce"

[sandbox.limits]
max_execution_ms = 30000
max_memory_mb = 256
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
    .with_process_sandbox()  // Isolate sandboxed tools in a subprocess
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

## MCP Servers

Acton-ai is an MCP **client**: tools exposed by external Model Context Protocol
servers become available to the LLM alongside the built-ins, under the name
`mcp__{server}__{tool}`.

Declare servers in `acton-ai.toml`. A server uses either `command` (a stdio
child process) or `url` (a streamable-HTTP endpoint) — setting both, or
neither, is a launch error naming the server.

```toml
[mcp_servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
env = { RUST_LOG = "info" }     # optional
cwd = "/srv"                    # optional
tool_timeout_secs = 60          # optional, default 60
enabled_tools = ["read_file"]   # optional allowlist of remote tool names

[mcp_servers.linear]
url = "https://mcp.linear.app/mcp"
auth_token_env = "LINEAR_MCP_TOKEN"   # optional bearer token, read from the environment
```

Or configure them programmatically — builder entries win over same-named TOML
entries:

```rust,ignore
use acton_ai::prelude::*;

let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .with_mcp_server(
        "filesystem",
        McpServerConfig::stdio("npx")
            .with_args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
    )
    .launch()
    .await?;

// MCP tools are injected into every prompt automatically.
let response = runtime.prompt("List the files in /tmp").collect().await?;
```

Every server connection is owned by a supervised actor. If a connection dies —
the child process exits, the HTTP transport drops — the in-flight tool call
returns an error the model can retry, the supervisor restarts the actor, and
the restarted actor reconnects. Tool executors resolve the live actor per call,
so nothing downstream has to notice.

`ActonAI::shutdown()` stops those actors, which closes the sessions and kills
any stdio child processes. Dropping the runtime without calling `shutdown()`
leaves that cleanup to a best-effort drop guard — call `shutdown()`.

## Structured Output

Get a typed Rust value back from a prompt instead of prose:

```rust
use acton_ai::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Invoice {
    vendor: String,
    total_cents: u64,
    line_items: Vec<LineItem>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct LineItem {
    description: String,
    cents: u64,
}

let invoice: Invoice = runtime
    .prompt("Extract the invoice from this email: ...")
    .extract::<Invoice>()
    .await?;
```

`extract::<T>()` does not ask the model for JSON and hope it parses. It appends
a synthetic `structured_output` tool whose input schema is the JSON Schema of
`T` — generated by [schemars](https://docs.rs/schemars) with subschemas inlined,
since provider support for `$ref` is inconsistent — and constrains the request
with `tool_choice` so the model has to call it. The arguments of that call are
deserialized into `T`. If they don't fit, the path-qualified serde error
(`line_items[0].cents: invalid type: string, expected u64`) goes back as a tool
result and the model is asked to correct itself, twice at most; after that you
get an `Extraction` error carrying the serde error and a truncated dump of what
the model actually produced.

Real tools still work. Anything from `.tool()`, `.use_builtins()`, or MCP may
run first, and extraction is the terminal step; if a round ends in prose with
no answer recorded, the model is asked once more with the choice forced. Add
`schemars` to your own dependencies to derive `JsonSchema`.

## CLI

Acton-ai ships a scriptable CLI with persistent sessions, autonomous task execution, and stdin/stdout piping.

### Chat

```bash
# Single message
acton-ai chat -m "What is Rust?"

# Pipe from stdin
echo "Explain ownership" | acton-ai chat

# Persistent sessions — context carries across invocations
acton-ai chat --session work --create -m "Start a new project plan"
acton-ai chat --session work -m "Add a testing section"

# JSON output for scripting
acton-ai chat -m "List 3 colors" --json | jq .text

# Interactive terminal chat
acton-ai chat
```

### Jobs

Define reusable jobs in `acton-ai.toml` with template substitution and agentic tool loops:

```toml
[jobs.summarize]
system_prompt = "You are a summarization expert. Be concise."
message_template = "Summarize:\n\n{{input}}"

[jobs.translate]
system_prompt = "Translate to the requested language. Output ONLY the translation."
message_template = "Translate to {{lang}}: {{input}}"
```

```bash
cat document.txt | acton-ai run-job summarize
echo "Hello" | acton-ai run-job translate --param lang=Spanish
```

### Heartbeat

Autonomous wake-up cycle for scheduled tasks. During chat, the agent can create heartbeat entries (recurring tasks). A systemd timer triggers `acton-ai heartbeat` to review and execute due tasks:

```bash
# Run all due heartbeat entries
acton-ai heartbeat

# Run entries for a specific session only
acton-ai heartbeat --session main
```

Output is a JSON activity report to stdout, suitable for monitoring pipelines.

### Session Management

```bash
acton-ai session list                  # List all sessions
acton-ai session show work             # Session metadata + recent messages
acton-ai session delete work --force   # Delete session and history
```

### Global Options

```
--json         Machine-readable JSON output
--config PATH  Override config file path
--provider NAME Override default LLM provider
-v / -vv / -vvv Increase verbosity
-q             Suppress stderr output
```

## Architecture

Acton-ai uses the actor model for fault-tolerant, concurrent AI systems:

```
ActonAI (Facade)
    │
    ├── ActorRuntime (acton-reactive)
    │       │
    │       ├── LLMProvider(s) ─── API calls, streaming, rate limiting
    │       │
    │       ├── Agent(s) ───────── Individual AI agents with reasoning
    │       │
    │       ├── ToolRegistry ───── Tool registration and execution
    │       │
    │       ├── MemoryStore ───── Persistent sessions, memories, embeddings
    │       │
    │       └── McpSupervisor ─── One supervised actor per MCP server connection
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

# Process-sandboxed execution
cargo run --example process_sandbox

# Per-agent tool configuration
cargo run --example per_agent_tools

# Typed structured output
cargo run --example structured_output
```

## Documentation

- [API Documentation (docs.rs)](https://docs.rs/acton-ai)
- [acton-reactive](https://docs.rs/acton-reactive) — The underlying actor framework

## Contributing

Contributions welcome. Please open an issue to discuss significant changes before submitting a PR.

## License

MIT License. See [LICENSE](LICENSE) for details.
