---
title: Development Setup
---

Everything you need to clone, build, test, and contribute to acton-ai.

---

## Prerequisites

### Rust toolchain

Install Rust via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Acton-ai targets **Rust edition 2021**. Any stable toolchain that supports edition 2021 will work. Verify your installation:

```bash
rustc --version
cargo --version
```

### System dependencies

**Linux (Ubuntu / Debian):**

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev
```

**macOS:**

Xcode Command Line Tools are sufficient:

```bash
xcode-select --install
```

### Optional: Hypervisor for sandbox testing

The tool sandbox system uses [Hyperlight](https://github.com/hyperlight-dev/hyperlight) micro-VMs. If you plan to work on sandbox-related code, you need a hypervisor:

- **Linux**: KVM (`sudo apt-get install qemu-kvm` or verify with `ls /dev/kvm`)
- **Windows**: Hyper-V (enable via Windows Features)

{% callout type="note" title="Sandbox is optional for most development" %}
If you do not have a hypervisor available, the rest of the framework builds and tests fine. Sandbox-specific tests will be skipped or will use the stub implementation.
{% /callout %}

### Optional: Ollama for integration testing

Many examples and integration tests use [Ollama](https://ollama.com) for local LLM access:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a small model for testing
ollama pull qwen2.5:7b
```

---

## Cloning the repository

```bash
git clone https://github.com/rodzilla/acton-ai.git
cd acton-ai
```

The repository is a single-crate Rust project (no workspace). The main source lives under `src/`, with examples in `examples/` and documentation in `docs/`.

---

## Building the project

### Standard build

```bash
cargo build
```

### Build with all features

```bash
cargo build --all-features
```

The `agent-skills` feature is the only optional feature flag. It enables the agent skills system for loading dynamic skill plugins. See the [Installation](/docs/installation) page for details on feature flags.

### Build the documentation

```bash
cargo doc --open
```

This generates rustdoc output and opens it in your browser.

---

## Running tests

### Run the full test suite

```bash
cargo test
```

This runs all unit tests embedded in source files and any integration tests. The crate has extensive unit tests co-located with each module.

### Run tests for a specific module

```bash
cargo test --lib kernel
cargo test --lib llm
cargo test --lib tools
cargo test --lib memory
cargo test --lib error
```

### Run tests with output visible

```bash
cargo test -- --nocapture
```

### Run a specific test by name

```bash
cargo test builder_ollama_sets_provider
```

---

## Running examples

The `examples/` directory contains runnable examples demonstrating various features:

| Example | Description |
| --- | --- |
| `ollama_chat` | Basic chat with Ollama |
| `ollama_chat_advanced` | Advanced chat with streaming and system prompts |
| `ollama_tools` | Tool usage with Ollama |
| `conversation` | Multi-turn conversation management |
| `multi_provider` | Using multiple LLM providers |
| `multi_agent` | Multi-agent collaboration |
| `per_agent_tools` | Per-agent tool configuration |
| `bash_sandbox` | Sandboxed bash execution |
| `agent_skills` | Agent skills system (requires `agent-skills` feature) |

Run an example:

```bash
cargo run --example ollama_chat
```

For examples that require the `agent-skills` feature:

```bash
cargo run --example agent_skills --features agent-skills
```

{% callout type="note" title="LLM provider required" %}
Most examples require a running Ollama instance or an API key for Anthropic/OpenAI. Check the example source for provider configuration details.
{% /callout %}

---

## Development workflow tips

### Clippy

Always run Clippy before submitting changes:

```bash
cargo clippy --all-targets --all-features
```

See the [Code Standards](/docs/code-standards) page for Clippy configuration details and known warnings.

### Formatting

Format your code with rustfmt:

```bash
cargo fmt
```

Check formatting without modifying files:

```bash
cargo fmt -- --check
```

### Watch mode

For rapid iteration, use `cargo-watch` to automatically rebuild on file changes:

```bash
cargo install cargo-watch
cargo watch -x check
```

Or run tests on every save:

```bash
cargo watch -x test
```

### Logging during development

Acton-ai uses `tracing` for structured logging. Set the `RUST_LOG` environment variable to control log output:

```bash
# See debug output from acton-ai
RUST_LOG=acton_ai=debug cargo run --example ollama_chat

# See trace-level output for a specific module
RUST_LOG=acton_ai::llm=trace cargo run --example ollama_chat

# See all logs at info level
RUST_LOG=info cargo test
```

### Useful cargo commands

```bash
# Check for compilation errors without building (faster)
cargo check

# Check with all features enabled
cargo check --all-features

# Build in release mode (for benchmarking)
cargo build --release

# View the dependency tree
cargo tree

# Check for outdated dependencies
cargo install cargo-outdated
cargo outdated
```

---

## Project structure

```text
acton-ai/
  src/
    lib.rs            # Crate root, module declarations, prelude
    facade.rs         # High-level ActonAI facade (builder pattern)
    kernel/           # Kernel actor (central supervisor)
    agent/            # Agent actor (individual AI agents)
    llm/              # LLM provider actor and API clients
    tools/            # Tool registry, executors, builtins, sandbox
    memory/           # Persistence and context window management
    conversation.rs   # Actor-backed Conversation handle
    prompt.rs         # Fluent PromptBuilder API
    stream.rs         # Stream handling traits
    messages.rs       # Actor message definitions
    types.rs          # Core type aliases (AgentId, CorrelationId, etc.)
    error.rs          # Error type hierarchy
    config.rs         # Configuration file loading
  examples/           # Runnable example programs
  docs/               # Documentation site (Next.js + Markdoc)
  Cargo.toml          # Package manifest
```

For a deeper look at the module responsibilities and how they interact, see the [Architecture Overview](/docs/architecture-overview) page.

---

## Next steps

- [Architecture Overview](/docs/architecture-overview) -- understand the actor hierarchy and message flow
- [Code Standards](/docs/code-standards) -- coding conventions, error handling patterns, and PR process
- [Two API Levels](/docs/two-api-levels) -- understand the high-level facade vs. low-level actor API
