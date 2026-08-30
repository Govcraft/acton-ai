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
- **Project instructions** — Discover and hierarchically layer cross-vendor `AGENTS.md` files with auditable structured sources
- **Tool execution loop** — Automatic tool calling and result handling until completion
- **Derived tools** — `#[tool]` turns an `async fn` into a tool: name, description, and JSON Schema all come from the signature and doc comment
- **Typed structured output** — `extract::<T>()` returns a schema-validated Rust value, with automatic repair rounds when the model gets the shape wrong
- **Two API levels** — Simple facade for common cases, full actor access for advanced control
- **TOML configuration** — Define providers and settings in config files
- **Process sandboxing** — Portable subprocess isolation for tool execution with rlimits, timeouts, and optional Linux hardening (landlock + seccomp)
- **Usage & cost tracking** — Token usage from every provider tallied per provider and model, priced from your own rate table; on by default
- **Spending budgets** — Hard process-wide and per-provider caps checked before every request, with warnings on the way up and a typed refusal at the ceiling
- **OpenTelemetry export** — Traces spanning each prompt loop (turn → rounds → tools) plus token, latency, and reliability metrics, over OTLP to any collector
- **Failover & circuit breaking** — Named provider chains tried in order, a per-provider circuit breaker with half-open recovery, and model degradation when a vendor throttles instead of dies
- **Tool-approval policy** — Allowlists, denylists, and per-turn invocation caps over every tool the model can reach, plus an async approval hook for a human in the loop; a refusal is fed back to the model and the turn carries on
- **Tamper-evident audit trail** — Every tool invocation appended to a BLAKE3 hash-chained JSONL log with secrets redacted, verified by `acton-ai audit verify`; the trail carries its own identity and admits exactly one writer
- **Durable audit and writer health** — `durability = "strict"` acknowledges every entry before the next tool runs, and a trail that has started failing refuses mutating tools through the ordinary refusal path instead of quietly losing evidence; the writer's state is readable as `AuditHealth`
- **Persistent sessions** — Named sessions with their messages, an opaque metadata column for the embedder's own per-session state, and checkpoint recovery that lists the turns a crash interrupted
- **FIPS mode** — An optional `fips` build routes every TLS connection through the FIPS 140-3 validated AWS-LC module
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

The default features are `sandbox-hardening` (Linux landlock + seccomp for the
process sandbox; a no-op elsewhere), `derive` (the [`#[tool]`](#deriving-tools-with-tool)
attribute macro), and `otel` ([OpenTelemetry export](#observability)). All three are
independent — `--no-default-features --features derive` gets you the macro with no OS
hardening and no telemetry stack.

The minimum supported Rust is **1.89**. That is where `std::fs::File::try_lock`
stabilized, and the audit trail's single-writer claim uses it, so an older
toolchain fails at resolution rather than part-way through a build.

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

### Discovering Project Instructions

Hosts can discover the instruction stack for a session and either inject its
rendered context or inspect each source before deciding what to trust:

```rust
use acton_ai::prelude::*;

let instructions = AgentInstructions::discover("./packages/api")?;

for layer in instructions.layers() {
    println!("{:?}: {}", layer.scope, layer.path.display());
}

let turn_start_context = instructions.context_fragment();
# Ok::<(), InstructionsError>(())
```

Discovery walks from the nearest checkout root to the working directory and
loads only files named exactly `AGENTS.md`. Deeper files appear later, and the
user-level `~/.agents/AGENTS.md` is last. Use
`AgentInstructions::discover_with_root` when the host supplies its own
workspace or trust boundary.

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

### Deriving Tools with `#[tool]`

Writing that by hand means restating the function signature four times — as a
name, a description, a JSON Schema, and a closure that plucks arguments back
out of a `Value` — and keeping all four in sync forever. The `#[tool]`
attribute derives them from the signature instead:

```rust
use acton_ai::prelude::*;

/// Evaluates math expressions.
///
/// Set `precision` to control decimal places; it defaults to 2.
#[tool]
async fn calculator(expression: String, precision: Option<u8>) -> Result<String, ToolError> {
    let places = usize::from(precision.unwrap_or(2));
    Ok(format!("{:.places$}", evaluate(&expression)))
}

runtime
    .prompt("What is 42 * 17?")
    .add_tool(Calculator)   // PascalCase unit struct, generated by the macro
    .collect()
    .await?;
```

The tool's name is the function name, its description is the doc comment, and
its schema has one property per parameter — with `Option<T>` parameters left
out of `required`. Struct parameters are inlined, so the schema carries no
`$ref` for a provider to choke on. Arguments are deserialized individually, so
a model that gets one wrong is told which one:
`parameter 'precision': invalid type: string "lots", expected u8`.

The annotated function is emitted unchanged, so `calculator(...)` is still an
ordinary function you can call and unit-test without a model in the loop.

A missing doc comment is a compile error, not a warning: the description is
the only thing the model reads when deciding whether to call your tool.

Available under the default `derive` feature. See `cargo run --example
tool_macro`.

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

# The hardened child reaches the system directories, $TMPDIR and the session
# root, and nothing else — so a toolchain installed under $HOME is found by
# the shell on PATH and then refused by the kernel. Declare what it needs.
[sandbox.paths]
read_exec = ["~/.local/bin", "~/.local/share/uv"]
read_write = ["~/.cache/uv"]

# Optional: rates for usage costs. Dollars per million tokens, copied
# straight off a vendor pricing page. Required for budgets.
[providers.claude.pricing]
input_per_mtok = 3.0
output_per_mtok = 15.0

# Optional: spending caps, checked before every request.
[budget]
total_usd = 5.00
warn_at_percent = 80        # default 80; 0 disables warnings

[budget.providers]
claude = 2.00               # per configured provider name

# Optional: where a round goes when this provider cannot serve it, and a
# cheaper model to use when it is merely rate limited.
failover = ["claude-backup", "local"]
fallback_model = "claude-haiku-4-5"

# The circuit breaker is ON by default (5 failures, 30s). This block only
# changes the numbers, or switches it off.
[providers.claude.circuit_breaker]
failure_threshold = 5
cooldown_secs = 30
enabled = true
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

### Spending Budgets

A budget refuses requests once a ceiling is reached. The check is pre-flight,
so a refusal costs nothing:

```rust
use acton_ai::prelude::*;

let runtime = ActonAI::builder()
    .anthropic_from_env()
    .pricing(DEFAULT_PROVIDER_NAME, ModelPricing::from_dollars_per_mtok(3.0, 15.0))
    .budget_usd(5.00)                                 // or .budget(Budget::usd(5.00)
    .on_budget_event(|e| eprintln!("budget: {e}"))    //      .provider("claude", 2.00)
    .launch()                                         //      .warn_at_percent(50))
    .await?;

match runtime.prompt("hello").collect().await {
    Err(e) if e.is_budget_exceeded() => println!("{e}"),
    other => { other?; }
}

// Where the caps stand, alongside the usual token tallies.
let usage = runtime.usage().await?;
if let Some(budget) = &usage.budget {
    println!("{:?} left", budget.remaining_usd());
}
```

Budgets need pricing: a provider whose tokens cannot be priced spends
invisibly, so it fails the launch unless `Budget::allow_unpriced()` says the
blind spot is acceptable. A cap is a circuit breaker, not an exact meter — a
request already in flight when the ceiling is crossed still completes.

### Failover and circuit breaking

A chain is tried in order when a provider cannot serve a round. The
re-dispatch happens inside the same round, so the caller gets an answer rather
than an error, and billing follows the provider that actually served:

```rust
use acton_ai::prelude::*;
use std::time::Duration;

let runtime = ActonAI::builder()
    .provider_named("claude",
        ProviderConfig::anthropic(&key)
            .with_failover(["claude-backup", "local"])
            .with_fallback_model("claude-haiku-4-5")   // used when rate limited
            .with_circuit_breaker(CircuitBreakerConfig::new(5, Duration::from_secs(30))))
    .provider_named("claude-backup", ProviderConfig::anthropic(&backup_key))
    .provider_named("local", ProviderConfig::ollama("qwen2.5:7b"))
    .default_provider("claude")
    .on_failover_event(|e| eprintln!("failover: {e}"))
    .launch()
    .await?;

match runtime.prompt("hello").collect().await {
    // Every candidate refused: the error names each one and why.
    Err(e) if e.is_all_providers_failed() => println!("{e}"),
    other => { other?; }
}
```

The circuit breaker is **on by default** — five consecutive failures open a
provider's circuit for thirty seconds, after which the next real request is
allowed through as a probe. Probes are never synthetic, so recovery costs
nothing extra. Opt out with `.without_circuit_breaker()` or
`enabled = false`. Failover chains and `fallback_model` change routing, so
they are always explicit.

### Observability

Point the runtime at an OTLP collector and every prompt becomes a trace —
`acton_ai.turn` with a child span per provider round and per tool call —
alongside metrics for tokens, latency, and rate limits:

```rust
use acton_ai::prelude::*;

let runtime = ActonAI::builder()
    .anthropic_from_env()
    .telemetry_otlp("http://localhost:4318")
    .launch()
    .await?;

// ... prompts ...

// Flushes telemetry after the actors stop, so the final batch is exported
// rather than dropped on the way out.
runtime.shutdown().await?;
```

The TOML twin, with the knobs the one-liner defaults:

```toml
[telemetry]
otlp_endpoint = "http://localhost:4318"   # required; presence enables export
service_name = "my-agent"                 # default "acton-ai"
metrics_interval_secs = 60                # default 60

[telemetry.headers]                       # optional; authenticated collectors
authorization = "Bearer ..."
```

To see the traces, run a collector with a UI and browse
<http://localhost:16686>:

```bash
docker run --rm -p 4318:4318 -p 16686:16686 jaegertracing/jaeger:latest
```

Tool arguments and results are **never** recorded on spans — they are user
data and unbounded in size. Correlation and agent IDs go on spans only, never
on metrics, where one series per request would melt the backend. The exporter
is OTLP over HTTP/protobuf; no gRPC stack is pulled in.

Already running OpenTelemetry in your application? `.telemetry_from_globals()`
emits into the providers you installed instead of installing competing ones.

### Live introspection

Traces and metrics tell you what happened. Introspection answers a different
question: what is this process doing *right now*, and can I safely restart it?

```rust
use acton_ai::prelude::*;

let runtime = ActonAI::builder()
    .anthropic_from_env()
    .introspection_at("/run/my-agent/control.sock")
    .drain_on_sigterm()
    .launch()
    .await?;
```

From another terminal:

```bash
acton-ai status --socket /run/my-agent/control.sock          # what is running
acton-ai status --socket /run/my-agent/control.sock --json   # the same, for jq
acton-ai pause  --socket /run/my-agent/control.sock          # refuse new turns
acton-ai resume --socket /run/my-agent/control.sock          # take them again
acton-ai drain  --socket /run/my-agent/control.sock --wait   # and wait for the last one
```

`status` reports admission state, turns and tool calls in flight, per-provider
circuit-breaker health, MCP server generations and restart counts, and token
and cost totals. It is assembled on demand from the live actors, so nothing in
it can be stale, and it keeps answering when a provider is wedged — which is
when you need it.

**Pausing never interrupts a turn that has started.** A half-finished turn has
usually already been paid for, and a tool it launched may have already changed
the world. `drain` is therefore "finish what you started, take nothing new",
which is what makes it safe to wire to `SIGTERM` and to an `ExecStop`. Callers
that try to start a turn get a distinguishable error
(`ActonAIError::is_turns_not_admitted`), not a generic failure.

The TOML twin:

```toml
[introspection]
enabled = true                          # default true when the section exists
socket_path = "/run/my-agent/ctl.sock"  # default $XDG_RUNTIME_DIR/acton-ai/<app>-<pid>.sock
socket_mode = 0o600                     # default 0o600; wider than owner-only is refused
```

Access control is the socket's permission bits, so they are owner-only and
anything laxer fails the launch: `pause` and `drain` are levers over your
process. Leave `socket_path` unset and each process gets its own PID-suffixed
address, which the CLI discovers by scanning; set it when something outside the
process — a systemd `ExecStop`, a deploy script — needs a predictable one.

Under systemd, `Type=notify` works out of the box: the runtime sends `READY=1`
once providers, tools, and MCP servers are up, so "started" means "serving",
and `STOPPING=1` when a drain begins. See `examples/introspection.rs` for a
complete unit file.

The socket server lives behind the `ipc` feature (on by default) but never
listens unless configured. Pause, resume, and drain are *not* behind it —
`--no-default-features` builds keep `ActonAI::pause()` and friends, since an
embedder driving the library from its own control plane needs them just as
much.

### Tool-approval policy

Every tool the model can reach — built-ins, `#[tool]` functions, MCP tools —
passes one gate before it runs. Nothing is configured by default, and an
unconfigured runtime behaves exactly as it always did.

```rust
use acton_ai::prelude::*;

let ai = ActonAI::builder()
    .anthropic_from_env()
    .tool_policy(
        ToolPolicy::new()
            .allow(["read_file", "glob", "grep", "mcp__docs__*"])
            .deny(["bash"])
            .cap_per_turn("read_file", 20),
    )
    .on_tool_approval(|invocation| async move {
        if invocation.tool_name == "write_file" {
            ApprovalDecision::deny("writes need a change ticket")
        } else {
            ApprovalDecision::Approve
        }
    })
    .launch()
    .await?;
```

Patterns are exact names or a single trailing `*`, which is enough to name a
whole MCP server (`mcp__docs__*`) without inviting something subtle enough to
be misread. Precedence is denylist, then allowlist, then cap; the denylist
wins over the allowlist because when a tool is named by both, refusing is the
reading that cannot cause harm.

The hook is consulted only for calls the rules already admitted, and it can
approve, refuse with a reason, or rewrite the arguments
(`ApprovalDecision::approve_with(..)`). It is `async`, so "ask a human" is a
legitimate implementation.

**A refusal is an outcome, not an error.** The tool does not run, the reason
goes back to the model as that call's tool result, and the turn continues —
the same shape the loop already uses for a schema-validation failure. Nothing
aborts, and the model is told not to retry.

The TOML twin, for the rules half:

```toml
[tool_policy]
allow = ["read_file", "glob", "grep", "mcp__docs__*"]
deny = ["bash"]

[tool_policy.per_turn_caps]
read_file = 20
```

A config file that sets rules and a builder that sets a hook are describing
two halves of one policy, so they combine rather than compete. Two sets of
*rules* do compete, and the builder wins.

### Audit trail

An append-only, hash-chained record of every tool invocation — allowed,
refused, or failed:

```toml
[audit]
path = "/var/log/acton-ai/audit.jsonl"
user = "acct:alice"
redact_patterns = ["password", "token", "api_key", "authorization"]
```

or `.audit_to("/var/log/acton-ai/audit.jsonl")` on the builder. Off unless one
of the two is present.

Each line is one entry: timestamp, acting user, correlation / conversation /
turn IDs, tool name, the arguments with secret-bearing keys redacted *before*
they are written, a bounded result summary, the complete response size, who
approved or refused it and under which rule, and how long it took. Every entry
carries the BLAKE3 hash of the one before it, so editing an entry invalidates
it, and re-sealing that entry invalidates its successor. The chain starts at a
fixed genesis hash and is resumed, never restarted, when the process restarts.

```bash
acton-ai audit verify                                   # the configured trail
acton-ai audit verify --file /var/log/acton-ai/audit.jsonl
acton-ai audit verify --json                            # for a monitoring check
```

Exit code is `0` when the chain verifies and `3` when it does not, so a cron
job can tell "the evidence has been altered" from "the check could not run".
A break is reported at the first entry that does not add up, with the expected
and found hashes.

Redaction is by key, not by value, and matches case-insensitively anywhere in
a key name; a matched key has its whole subtree replaced. Setting
`redact_patterns` replaces the defaults rather than extending them, so the list
in your config is the list in force.

One actor owns the chain and is the only thing that writes the file, which is
what makes the ordering guarantee real rather than hopeful.

**Every trail has an identity.** The first time an audit log opens a trail it
mints a `TrailId` (a TypeID with prefix `trail`), keeps it in a sidecar beside
the file (`audit.jsonl.trail`), and seals it into every entry's hash. An entry
cannot be relabelled as some other trail's, and a chain cannot change identity
part-way: `verify_chain` reports that as `ChainBreakKind::TrailMismatch`.
`audit verify` prints the identity on a `trail:` line and carries `trail_id` in
its JSON report. Trails written before identities existed keep verifying, and
gain one on their next spawn; a sidecar that disagrees with the chain refuses
the spawn.

**One writer, enforced by the kernel.** `AuditLog::spawn` claims the trail with
an exclusive advisory lock before it reads the chain head, and holds it for the
actor's lifetime. A second process opening the same trail fails to launch with
a configuration error rather than forking the chain, and on Linux the refusal
names the pid holding the lock. There is no pid file to go stale: the kernel
releases the claim on shutdown or on `SIGKILL`. Read-only verification of a
live trail is unaffected.

**Durability is a choice.** The default, `best_effort`, is fire-and-forget: a
turn is never blocked on a disk.

```toml
[audit]
durability = "strict"
```

Under `strict`, every entry is fsynced and acknowledged before the prompt loop
considers the next tool call, and once an append has failed the loop refuses
every tool not declared idempotent. That refusal travels the ordinary path, so
the attempt is itself recorded as denied and the model is told that mutating
tools stay refused for the rest of the session. An audit actor that does not
answer the guard at all is treated the same way, because the guard fails
closed. Read the writer's state with `ActonAI::audit_health()`: healthy,
degraded or disabled, with appended and failed counts, the first sequence that
failed, and the last error. The healthy-to-degraded transition is broadcast
once as `AuditHealthChanged`.

### FIPS mode

An optional, non-default `fips` feature routes every TLS connection through
the FIPS 140-3 validated AWS-LC module instead of *ring*:

```bash
cargo build --release --no-default-features \
    --features "fips,sandbox-hardening,derive,otel,ipc"
```

`acton_ai::fips::install_crypto_provider()` installs it as the process-wide
rustls default before anything can open a connection; the CLI calls it at the
top of `main`, and `launch()` calls it again idempotently for embedders with
their own entry point. A build that claims FIPS refuses to start if a non-FIPS
provider got there first.

Two things to know before you reach for it. It needs **CMake and Go** on the
build host, because aws-lc-fips-sys builds the module from source — which is
why it is not in the default set and not gated in CI. And it must be a
**release** build: AWS-LC's power-on integrity test hashes its own loaded text
segment, and a debug build's relocations change that hash between runs, so the
module aborts the process at startup. That is the FIPS module working as
specified.

One honest caveat: this covers the TLS the framework itself opens — LLM
providers, streamable-HTTP MCP servers, OTLP export — all of which go through
reqwest. libsql, used for persistence, carries its own pinned rustls and
*ring* for remote-database connections; a deployment that needs the whole
process free of non-validated cryptography should use a local database file,
which is the default.

`acton_ai::fips::is_fips_build()` reports which binary you have.

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
| `update_plan` | Record the plan for the current task and its progress |

Select specific tools with `.with_builtin_tools(&["read_file", "glob", "bash"])`.

`update_plan` is the odd one out: it runs no code and touches nothing. The
model sends its whole plan — `{steps: [{title, status}], note?}`, two or more
steps, at most one of them `in_progress` — the plan is validated, recorded as
the turn's state, and broadcast as a `PlanUpdated` message. Single-step plans
are refused with a corrective message, and the tool's description tells the
model to skip planning for trivial tasks. The chat REPL renders each update as
an inline checklist; the turn's final plan is on `CollectedResponse::plan`; and
any actor can watch live:

```rust
builder.handle().subscribe::<PlanUpdated>().await;   // on the builder, before start()
```

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

### Status, Pause, Resume, Drain

The only commands that do not start a runtime: they talk to one that is already
running, over its introspection socket.

```bash
acton-ai status                        # what a running process is doing
acton-ai status --json                 # the same, machine-readable
acton-ai pause                         # stop admitting new turns
acton-ai resume                        # admit them again
acton-ai drain                         # stop admitting, report what is left
acton-ai drain --wait --timeout 0      # and block until the last turn finishes
```

The socket is found from `--socket`, then `[introspection] socket_path` in the
config file, then a scan of the default runtime directory. The scan is what
makes a bare `acton-ai status` work, since the default socket name carries a
PID nobody can guess; it refuses to choose when it finds more than one, because
draining the wrong process is worse than being asked which one you meant.

`drain --wait` is what belongs in an `ExecStop` or a deploy script. It returns
as soon as the last in-flight turn finishes, and `--timeout 0` waits
indefinitely so systemd's `TimeoutStopSec` is the only deadline in play.

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
    │       ├── MemoryStore ───── Persistent sessions, memories, embeddings
    │       │
    │       ├── CostAccountant ── Tallies broadcast usage reports, prices snapshots
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

# Spending caps and budget events
cargo run --example budget

# Deriving tools from functions with #[tool]
cargo run --example tool_macro

# OpenTelemetry traces and metrics over OTLP
cargo run --example telemetry
```

## Documentation

- [API Documentation (docs.rs)](https://docs.rs/acton-ai)
- [acton-reactive](https://docs.rs/acton-reactive) — The underlying actor framework

## Contributing

Contributions welcome. Please open an issue to discuss significant changes before submitting a PR.

## License

MIT License. See [LICENSE](LICENSE) for details.
