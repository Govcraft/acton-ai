# Project Overview

This repository contains a small example crate demonstrating how to build
agents with the `acton-reactive` framework.

## Purpose

Use `acton-reactive` to create a simple agentic AI framework so we can start
creating autonomous AI agents.

## Structure

- `src/main.rs` launches a few example agents and shuts them down.
- `src/agents/` contains individual agents such as an AI agent, model agent
  and memory agent.
- `src/messages.rs` defines the message types shared between agents.
- `src/persistence/` provides a minimal in-memory storage implementation.
- `src/llm_clients/` and `src/tools/` contain placeholders for future
  integrations.

## Development

Ensure the project builds and is lint-free with:

```bash
cargo check
cargo test
cargo clippy -- -D warnings
```

Clippy must run with `-D warnings` and pass with no lints.
