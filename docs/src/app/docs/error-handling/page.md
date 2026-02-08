---
title: Error Handling
---

Acton AI uses a custom error hierarchy with no external error crates. Every error type implements `Debug`, `Clone`, `PartialEq`, `Eq`, and `std::error::Error`, giving you full control over matching, comparison, and propagation.

---

## Error type hierarchy

The framework is organized into domain-specific error types, each with a `kind` field you can pattern-match on:

```text
ActonAIError          -- High-level API (builder, prompt, stream)
  |
KernelError           -- Kernel operations (spawn, routing)
  |
AgentError            -- Agent lifecycle and processing
  |
MultiAgentError       -- Cross-agent delegation and routing
  |
LLMError              -- LLM provider communication
  |
ToolError             -- Tool registration, execution, sandbox
  |
PersistenceError      -- Database, serialization, memory store
```

---

## ActonAIError

The top-level error type for the high-level `ActonAI` facade API. This is the error you will encounter most often when using `ActonAI::builder()`, `runtime.prompt()`, and `Conversation::send()`.

### Error kinds

```rust
use acton_ai::error::{ActonAIError, ActonAIErrorKind};

match error.kind {
    ActonAIErrorKind::Configuration { field, reason } => {
        // Builder setup problem (e.g., missing app_name)
        eprintln!("Config error in '{}': {}", field, reason);
    }
    ActonAIErrorKind::LaunchFailed { reason } => {
        // Runtime failed to start (e.g., provider spawn failed)
        eprintln!("Launch failed: {}", reason);
    }
    ActonAIErrorKind::PromptFailed { reason } => {
        // A prompt execution failed
        eprintln!("Prompt failed: {}", reason);
    }
    ActonAIErrorKind::StreamError { reason } => {
        // Streaming token delivery failed
        eprintln!("Stream error: {}", reason);
    }
    ActonAIErrorKind::ProviderError { reason } => {
        // LLM provider returned an error
        eprintln!("Provider error: {}", reason);
    }
    ActonAIErrorKind::RuntimeShutdown => {
        // Runtime was shut down before the operation completed
        eprintln!("Runtime is shut down");
    }
}
```

### Convenience predicates

```rust
if error.is_configuration() {
    // Fix configuration and retry
}

if error.is_runtime_shutdown() {
    // Runtime is gone, cannot recover
}
```

### Constructors

```rust
let err = ActonAIError::configuration("app_name", "cannot be empty");
let err = ActonAIError::launch_failed("provider not configured");
let err = ActonAIError::prompt_failed("timeout waiting for response");
let err = ActonAIError::stream_error("connection reset");
let err = ActonAIError::provider_error("rate limit exceeded");
let err = ActonAIError::runtime_shutdown();
```

---

## LLMError

Errors from LLM provider communication -- network issues, rate limits, API errors, and parse failures.

### Error kinds

```rust
use acton_ai::llm::error::{LLMError, LLMErrorKind};

match error.kind {
    LLMErrorKind::Network { message } => {
        // Connection failed, DNS resolution, etc.
    }
    LLMErrorKind::RateLimited { retry_after } => {
        // Wait and retry after the specified duration
        tokio::time::sleep(retry_after).await;
    }
    LLMErrorKind::ApiError { status_code, message, error_type } => {
        // HTTP error from the API
        if status_code >= 500 {
            // Server error -- retry
        } else {
            // Client error -- fix the request
        }
    }
    LLMErrorKind::AuthenticationFailed { reason } => {
        // API key is invalid or expired
    }
    LLMErrorKind::InvalidRequest { reason } => {
        // Bad request parameters
    }
    LLMErrorKind::StreamError { message } => {
        // Error during token streaming
    }
    LLMErrorKind::ParseError { message } => {
        // Failed to parse JSON response
    }
    LLMErrorKind::ModelOverloaded { model } => {
        // Model is temporarily unavailable
    }
    LLMErrorKind::Timeout { duration } => {
        // Request exceeded timeout
    }
    LLMErrorKind::ShuttingDown => {
        // Provider is shutting down
    }
    LLMErrorKind::InvalidConfig { field, reason } => {
        // Provider configuration error
    }
}
```

### Retry logic

`LLMError` has built-in retry classification:

```rust
if error.is_retriable() {
    // These errors are transient and may succeed on retry:
    // - Network errors
    // - Rate limits
    // - Model overloaded
    // - Timeouts
    // - Server errors (5xx)
}

// For rate limits, get the suggested wait time:
if let Some(wait) = error.retry_after() {
    tokio::time::sleep(wait).await;
}
```

---

## ToolError

Errors from tool registration, validation, execution, and sandbox operations.

### Error kinds

```rust
use acton_ai::tools::error::{ToolError, ToolErrorKind};

match error.kind() {
    ToolErrorKind::NotFound { tool_name } => {
        // Tool is not registered
    }
    ToolErrorKind::AlreadyRegistered { tool_name } => {
        // Tool name collision
    }
    ToolErrorKind::ExecutionFailed { tool_name, reason } => {
        // Tool ran but failed
    }
    ToolErrorKind::Timeout { tool_name, duration } => {
        // Tool exceeded its time limit
    }
    ToolErrorKind::ValidationFailed { tool_name, reason } => {
        // Invalid arguments passed to the tool
    }
    ToolErrorKind::SandboxError { message } => {
        // Sandbox creation or execution failed
    }
    ToolErrorKind::ShuttingDown => {
        // Tool registry is shutting down
    }
    ToolErrorKind::Internal { message } => {
        // Unexpected internal error
    }
}
```

### Retry and classification

```rust
// Timeout and sandbox errors are retriable
assert!(ToolError::timeout("slow_tool", Duration::from_secs(30)).is_retriable());
assert!(ToolError::sandbox_error("transient failure").is_retriable());

// Not-found and validation errors are not
assert!(!ToolError::not_found("missing_tool").is_retriable());
assert!(!ToolError::validation_failed("tool", "bad args").is_retriable());

// Quick checks
assert!(ToolError::not_found("x").is_not_found());
assert!(ToolError::already_registered("x").is_already_registered());
```

### Correlation IDs

`ToolError` supports optional correlation IDs for request tracking:

```rust
use acton_ai::types::CorrelationId;

let corr_id = CorrelationId::new();
let error = ToolError::with_correlation(
    corr_id,
    ToolErrorKind::ExecutionFailed {
        tool_name: "bash".to_string(),
        reason: "command not found".to_string(),
    },
);
// Display: "[corr-id] tool 'bash' execution failed: command not found"
```

---

## KernelError

Errors from the Kernel actor that manages agent lifecycles.

### Error kinds

```rust
use acton_ai::error::{KernelError, KernelErrorKind};

match error.kind {
    KernelErrorKind::AgentNotFound { agent_id } => {
        // Agent with this ID is not running
    }
    KernelErrorKind::SpawnFailed { reason } => {
        // Could not create the agent
    }
    KernelErrorKind::AgentAlreadyExists { agent_id } => {
        // An agent with this ID is already running
    }
    KernelErrorKind::ShuttingDown => {
        // Kernel is shutting down
    }
    KernelErrorKind::InvalidConfig { field, reason } => {
        // Bad kernel configuration
    }
}
```

### Convenience predicates

```rust
if error.is_not_found() {
    // Agent doesn't exist -- spawn it first
}

if error.is_shutting_down() {
    // Cannot recover -- system is stopping
}
```

---

## AgentError

Errors from individual agent operations.

### Error kinds

```rust
use acton_ai::error::{AgentError, AgentErrorKind};

match error.kind {
    AgentErrorKind::InvalidState { current, expected } => {
        // Agent is in the wrong state for the requested operation
        eprintln!("Agent is '{}', expected '{}'", current, expected);
    }
    AgentErrorKind::ProcessingFailed { reason } => {
        // Message processing failed
    }
    AgentErrorKind::LLMRequestFailed { reason } => {
        // LLM call from the agent failed
    }
    AgentErrorKind::ToolExecutionFailed { tool_name, reason } => {
        // A tool the agent called failed
    }
    AgentErrorKind::Stopping => {
        // Agent is shutting down
    }
    AgentErrorKind::InvalidConfig { field, reason } => {
        // Agent configuration error
    }
}
```

`AgentError` includes an optional `agent_id` field so you can identify which agent failed:

```rust
if let Some(ref id) = error.agent_id {
    eprintln!("Agent {} failed: {}", id, error);
} else {
    eprintln!("Unknown agent failed: {}", error);
}
```

---

## MultiAgentError

Errors from multi-agent operations -- delegation, routing, and capability lookup.

### Error kinds

```rust
use acton_ai::error::{MultiAgentError, MultiAgentErrorKind};

match error.kind {
    MultiAgentErrorKind::AgentNotFound { agent_id } => {
        // Target agent is not running
    }
    MultiAgentErrorKind::TaskNotFound { task_id } => {
        // Task ID is not recognized
    }
    MultiAgentErrorKind::TaskAlreadyAccepted { task_id } => {
        // Cannot accept a task twice
    }
    MultiAgentErrorKind::NoCapableAgent { capability } => {
        // No running agent has the required capability
    }
    MultiAgentErrorKind::DelegationFailed { task_id, reason } => {
        // Task delegation failed
    }
    MultiAgentErrorKind::RoutingFailed { to, reason } => {
        // Message could not be delivered to the target agent
    }
}
```

---

## PersistenceError

Errors from database and memory store operations.

### Error kinds

```rust
use acton_ai::memory::error::{PersistenceError, PersistenceErrorKind};

match error.kind() {
    PersistenceErrorKind::DatabaseOpen { path, message } => {
        // Failed to open or create the database file
    }
    PersistenceErrorKind::SchemaInit { message } => {
        // Database schema migration failed
    }
    PersistenceErrorKind::QueryFailed { operation, message } => {
        // A database query failed
    }
    PersistenceErrorKind::NotFound { entity, id } => {
        // Record not found
    }
    PersistenceErrorKind::SerializationFailed { message } => {
        // Could not serialize data for storage
    }
    PersistenceErrorKind::DeserializationFailed { message } => {
        // Could not read stored data
    }
    PersistenceErrorKind::TransactionFailed { message } => {
        // Transaction rolled back
    }
    PersistenceErrorKind::ConnectionError { message } => {
        // Database connection issue
    }
    PersistenceErrorKind::EmbeddingFailed { provider, message } => {
        // Embedding generation failed
    }
    PersistenceErrorKind::EmbeddingDimensionMismatch { expected, actual } => {
        // Embedding dimensions don't match the index
    }
    PersistenceErrorKind::VectorSearchFailed { message } => {
        // Vector similarity search failed
    }
    PersistenceErrorKind::ShuttingDown => {
        // Store is shutting down
    }
}
```

### Retry classification

```rust
// Connection errors and transaction failures are retriable
assert!(PersistenceError::connection_error("timeout").is_retriable());
assert!(PersistenceError::transaction_failed("deadlock").is_retriable());

// Not-found errors are not retriable
assert!(!PersistenceError::not_found("conversation", "id").is_retriable());
```

---

## Common error scenarios

### Scenario: Provider not configured

```rust
let result = ActonAI::builder()
    .app_name("my-app")
    // Forgot to configure a provider!
    .launch()
    .await;

match result {
    Err(e) if e.is_configuration() => {
        eprintln!("Setup error: {}", e);
        // "configuration error for 'provider': no default provider configured"
    }
    _ => {}
}
```

### Scenario: Rate limit with retry

```rust
use std::time::Duration;

async fn prompt_with_retry(
    runtime: &ActonAI,
    text: &str,
    max_retries: usize,
) -> Result<CollectedResponse, ActonAIError> {
    let mut attempts = 0;
    loop {
        match runtime.prompt(text).collect().await {
            Ok(response) => return Ok(response),
            Err(e) => {
                attempts += 1;
                if attempts >= max_retries {
                    return Err(e);
                }
                // Back off before retrying
                let wait = Duration::from_secs(2u64.pow(attempts as u32));
                eprintln!("Attempt {} failed: {}. Retrying in {:?}...", attempts, e, wait);
                tokio::time::sleep(wait).await;
            }
        }
    }
}
```

### Scenario: Handling tool execution failures

```rust
let response = runtime
    .prompt("List files in the current directory using bash")
    .use_builtins()
    .collect()
    .await?;

for tc in &response.tool_calls {
    match &tc.result {
        Ok(value) => {
            println!("Tool '{}' succeeded: {}", tc.name, value);
        }
        Err(error_msg) => {
            eprintln!("Tool '{}' failed: {}", tc.name, error_msg);
        }
    }
}
```

### Scenario: Graceful shutdown

```rust
match runtime.shutdown().await {
    Ok(()) => println!("Clean shutdown"),
    Err(e) if e.is_runtime_shutdown() => {
        // Already shut down -- this is fine
    }
    Err(e) => {
        eprintln!("Shutdown error: {}", e);
    }
}
```

---

## Best practices

### 1. Match on error kinds, not strings

Every error type has a structured `kind` field. Match on it instead of parsing display strings:

```rust
// Good
match error.kind {
    ActonAIErrorKind::ProviderError { reason } => handle_provider_error(&reason),
    ActonAIErrorKind::RuntimeShutdown => handle_shutdown(),
    _ => handle_other(error),
}

// Avoid
if error.to_string().contains("rate limit") {
    // Fragile -- display format may change
}
```

### 2. Use `is_retriable()` for retry decisions

Both `LLMError` and `ToolError` provide `is_retriable()` to classify transient vs. permanent failures:

```rust
if llm_error.is_retriable() {
    // Network timeout, rate limit, server error -- retry
} else {
    // Auth failure, bad request -- fix the code
}
```

### 3. Use convenience predicates for common checks

```rust
if kernel_error.is_not_found() { /* agent not registered */ }
if kernel_error.is_shutting_down() { /* system stopping */ }
if agent_error.is_stopping() { /* agent stopping */ }
if tool_error.is_not_found() { /* tool not registered */ }
if persistence_error.is_not_found() { /* record missing */ }
```

### 4. All errors are Clone + PartialEq

You can clone errors for logging, compare them in tests, and store them in collections:

```rust
let error1 = KernelError::shutting_down();
let error2 = error1.clone();
assert_eq!(error1, error2);
```

### 5. Propagate with `?` using From conversions

`ActonAIError` is the return type for high-level API functions. Lower-level errors are converted automatically when they surface through the facade:

```rust
// This works because lower-level errors convert to ActonAIError
async fn my_function(runtime: &ActonAI) -> Result<(), ActonAIError> {
    let response = runtime.prompt("Hello").collect().await?;
    Ok(())
}
```

---

## Next steps

- [Multi-Agent Collaboration](/docs/multi-agent-collaboration) -- handle `MultiAgentError` in delegation workflows
- [Secure Tool Execution](/docs/secure-tool-execution) -- understand `SandboxErrorKind` and `ToolError`
- [Testing Your Agents](/docs/testing) -- assert on specific error kinds in tests
