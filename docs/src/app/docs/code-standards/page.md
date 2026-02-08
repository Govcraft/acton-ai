---
title: Code Standards
---

Conventions, patterns, and processes for contributing to acton-ai.

---

## Rust coding conventions

### General style

- **Edition 2021.** All code targets Rust edition 2021.
- **No external error crates.** The project deliberately avoids `thiserror`, `anyhow`, and `eyre`. All error types are hand-written to keep the public API explicit and dependency-light.
- **No external derive macros for errors.** Error types manually implement `Display`, `Debug`, `Clone`, `PartialEq`, `Eq`, and `std::error::Error`.
- **`#[must_use]` on constructors and query methods.** Any function that returns a value the caller should not silently discard is annotated with `#[must_use]`.
- **Builder pattern for configuration.** Public-facing configuration uses the builder pattern (`ActonAIBuilder`, `ConversationBuilder`, `PromptBuilder`). Builders consume `self` and return `Self` for chaining.
- **Derive `Clone` generously.** Configuration types, error types, and message types all derive `Clone`. This aligns with the actor model where messages are sent by value.

### Actor conventions

- Actor state structs use the `#[acton_actor]` derive macro.
- Actor messages use the `#[acton_message]` derive macro, which requires `Any + Send + Sync + Debug + Clone + 'static`. In practice, `#[derive(Clone, Debug)]` on your message struct is sufficient since the blanket impl covers the rest.
- The public field for user state in actor handlers is `actor.model`.
- `Reply::pending(async move { ... })` blocks the mailbox -- use it intentionally to serialize message processing. Use `tokio::spawn` inside `Reply::pending` if you need to release the mailbox while awaiting.
- Keep handler closures focused. Extract complex logic into standalone `async fn` helpers (as seen in `process_streaming_request` in the LLM provider).

### Naming conventions

- Actor modules follow a consistent structure: `mod.rs` (public exports), `actor.rs` (actor implementation), `config.rs` (configuration types).
- Error types are named `{Module}Error` with a `{Module}ErrorKind` enum (e.g., `LLMError` / `LLMErrorKind`).
- Message types describe the action or event: `SpawnAgent`, `LLMStreamToken`, `InitKernel`.
- ID types use the `{Entity}Id` pattern: `AgentId`, `CorrelationId`, `TaskId`.

### Function parameter limits

Group function parameters into structs when a function would take more than 7 arguments. This avoids Clippy's `too_many_arguments` lint and improves readability.

---

## Error handling patterns

### Error type hierarchy

The project uses a consistent pattern across all modules. Each error type is a struct with a `kind` field containing an enum of specific error variants:

```text
ActonAIError          -- High-level facade API errors
  ActonAIErrorKind
    Configuration
    LaunchFailed
    PromptFailed
    StreamError
    ProviderError
    RuntimeShutdown

KernelError           -- Kernel actor errors
  KernelErrorKind
    AgentNotFound
    SpawnFailed
    AgentAlreadyExists
    ShuttingDown
    InvalidConfig

AgentError            -- Agent actor errors
  AgentErrorKind
    InvalidState
    ProcessingFailed
    LLMRequestFailed
    ToolExecutionFailed
    Stopping
    InvalidConfig

LLMError              -- LLM provider errors
  LLMErrorKind
    Network
    RateLimited
    ApiError
    AuthenticationFailed
    InvalidRequest
    StreamError
    ParseError
    ShuttingDown
    InvalidConfig
    ModelOverloaded
    Timeout

ToolError             -- Tool system errors
  ToolErrorKind
    NotFound
    AlreadyRegistered
    ExecutionFailed
    Timeout
    ValidationFailed
    SandboxError
    ShuttingDown
    Internal

MultiAgentError       -- Multi-agent coordination errors
  MultiAgentErrorKind
    AgentNotFound
    TaskNotFound
    TaskAlreadyAccepted
    NoCapableAgent
    DelegationFailed
    RoutingFailed

PersistenceError      -- Database/storage errors
  PersistenceErrorKind
    DatabaseOpen
    SchemaInit
    QueryFailed
    NotFound
    SerializationFailed

EmbeddingError        -- Embedding generation errors
```

### Conventions for error types

Every error type follows these rules:

1. **Struct + Kind enum.** The struct holds optional context (like an `AgentId` or `CorrelationId`) plus a `kind` field. The enum lists all possible error variants with their data.

2. **Named constructors.** Each variant has a corresponding constructor method on the struct (e.g., `LLMError::network("connection refused")`). Constructors accept `impl Into<String>` for convenience.

3. **Query methods.** Boolean helpers like `is_retriable()`, `is_not_found()`, and `is_shutting_down()` make it easy to branch on error types without matching on the kind enum directly.

4. **Actionable Display messages.** Error messages describe both what happened and what the user can do about it. For example: `"agent 'abc123' not found; verify the agent ID is correct and the agent is running"`.

5. **All standard traits.** Every error type implements `Debug`, `Clone`, `PartialEq`, `Eq`, `Display`, and `std::error::Error`.

6. **Size optimization.** Large error kinds (like `ToolErrorKind` and `PersistenceErrorKind`) are boxed inside the error struct to keep the `Result` size small.

### Using errors in handlers

Actor message handlers use `try_mutate_on` / `try_act_on` with typed errors:

```rust
builder.try_mutate_on::<LLMRequest, (), LLMError>(|actor, envelope| {
    if actor.model.shutting_down {
        return Reply::try_err(LLMError::shutting_down());
    }
    // ... process request ...
    Reply::try_ok(())
});
```

---

## Testing requirements

### Test co-location

All unit tests live alongside the code they test, inside `#[cfg(test)] mod tests` blocks at the bottom of each source file. This is the standard Rust pattern and makes it easy to find tests for any given module.

### What to test

- **Constructors and defaults.** Verify that `Default` implementations produce expected values and that builder methods set the right fields.
- **Error formatting.** Every error variant should have a test verifying its `Display` output contains the relevant information.
- **Query methods.** Test boolean helpers like `is_retriable()` for both positive and negative cases.
- **Clone and Eq.** Verify that error types and message types correctly implement `Clone` and `PartialEq`.
- **Actor behavior.** For actor-specific logic, use `tokio::test` with the actual actor runtime to verify message handling.

### Test utilities

- `tokio-test` is available as a dev dependency for async test utilities.
- `tempfile` is available for tests that need temporary directories or files.
- `StubEmbeddingProvider` and `StubSandbox` / `StubSandboxFactory` are available in test builds for mocking external dependencies.

### Running tests

```bash
# Full test suite
cargo test

# Tests for a single module
cargo test --lib llm::error

# Single test
cargo test rate_limiter_allows_initial_request

# With output
cargo test -- --nocapture
```

---

## Documentation standards

### Rustdoc

- Every public type, trait, method, and function must have a doc comment.
- Use `///` for item-level documentation and `//!` for module-level documentation.
- Include a `# Example` section with `rust,ignore` code blocks for complex APIs.
- Use `# Errors`, `# Panics`, and `# Arguments` sections where applicable.
- Link to related types using `[`backtick`]` syntax.

### Module-level docs

Each module's `mod.rs` (or top-level file) should include:
- A one-line summary of the module's purpose
- An `## Architecture` section for modules with internal structure
- A `## Usage` section with code examples
- Re-exports of the module's public API

---

## Clippy configuration

The project does not use a `clippy.toml` or `.clippy.toml` file. All Clippy configuration is implicit via the defaults.

Run Clippy with:

```bash
cargo clippy --all-targets --all-features
```

### Known warnings

There are two pre-existing `derivable_impls` warnings that are intentionally left in place:

1. `src/llm/config.rs` -- a `Default` impl that could be derived but is kept explicit for clarity
2. `src/tools/sandbox/hyperlight/config.rs` -- same reason

These warnings are tracked and should not be "fixed" by replacing them with derives, as the explicit implementations serve as documentation of the default values.

{% callout type="warning" title="Do not suppress Clippy warnings" %}
Do not add `#[allow(clippy::...)]` attributes without discussion. If Clippy flags something in new code, fix it. If you believe a lint is a false positive, raise it in the PR.
{% /callout %}

---

## Semver policy

Acton-ai follows [Semantic Versioning](https://semver.org/) with the pre-1.0 convention:

- **Patch** (0.25.x): Bug fixes and non-breaking changes
- **Minor** (0.x.0): Breaking changes (since we are pre-1.0)
- **Major**: Reserved for 1.0 release

{% callout type="note" title="Pre-1.0 breaking changes" %}
While the version is below 1.0, breaking changes bump the minor version. This is standard semver practice for pre-release software. After 1.0, breaking changes will require a major version bump.
{% /callout %}

### Release process

Releases are managed with `cargo release`:

```bash
# Dry run
cargo release patch --dry-run

# Actual release (bumps version, tags, publishes)
cargo release patch
```

Commits are signed with SSH (`-S` flag), and tags are signed (`tag -s`). GitHub verifies signatures automatically.

---

## Commit message conventions

Write commit messages that explain **why** the change was made, not just what changed. Use the imperative mood in the subject line.

### Format

```text
type: short description

Optional longer explanation of why this change was made,
what problem it solves, and any relevant context.
```

### Types

| Type | When to use |
| --- | --- |
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code restructuring without behavior change |
| `docs` | Documentation changes |
| `test` | Test additions or modifications |
| `chore` | Build, CI, release, or tooling changes |
| `perf` | Performance improvements |

### Breaking changes

Prefix breaking changes with `refactor!` or `feat!` (note the `!`):

```text
refactor!: make Conversation an actor for concurrency safety
```

---

## PR process

### Before submitting

1. **Run the full check suite:**

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features
cargo test
```

2. **Verify documentation builds:**

```bash
cargo doc --no-deps
```

3. **Keep PRs focused.** One logical change per PR. If you find unrelated issues while working, file them separately.

4. **Add tests.** New functionality should include tests. Bug fixes should include a regression test.

### PR structure

- **Title**: Short, descriptive, in imperative mood (e.g., "Add retry logic to LLM provider")
- **Description**: Explain the motivation, approach, and any trade-offs. Reference related issues.
- **Test plan**: Describe how to verify the change works.

### Review expectations

- All Clippy warnings must be resolved (except the two known `derivable_impls` warnings)
- All tests must pass
- Public API additions should include rustdoc with examples
- Error types should follow the struct + kind enum pattern described above

---

## Next steps

- [Development Setup](/docs/development-setup) -- get the project building and running
- [Architecture Overview](/docs/architecture-overview) -- understand the system design
- [Error Handling](/docs/error-handling) -- detailed guide to error types and patterns
- [Testing](/docs/testing) -- comprehensive testing guide
