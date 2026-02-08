---
title: Testing Your Agents
---

AI agent code presents unique testing challenges: LLM responses are non-deterministic, tool execution has side effects, and multi-agent workflows involve complex async interactions. This guide provides practical patterns for testing each layer of an Acton AI application.

---

## Testing strategies

Testing AI agent systems effectively requires a layered approach:

| Layer | What to test | How |
|---|---|---|
| **Configuration** | Agent configs, tool selection, builder setup | Unit tests with direct struct construction |
| **Tool execution** | Individual tool behavior, input validation | Unit tests with mock inputs |
| **Error handling** | Error classification, retry logic, error propagation | Unit tests with constructed errors |
| **Delegation** | Task tracking, state transitions, cleanup | Unit tests with `DelegationTracker` |
| **Conversation flow** | History management, system prompts | Integration tests with a running LLM |
| **End-to-end** | Full prompt-to-response pipeline | Integration tests with Ollama or a mock server |

---

## Unit testing agent configuration

`AgentConfig` is a plain data struct that supports serialization. Test it without any async runtime or LLM:

```rust
use acton_ai::agent::AgentConfig;

#[test]
fn file_reader_agent_has_correct_tools() {
    let config = AgentConfig::new(
        "You are a file reader assistant.",
    )
    .with_tools(&["read_file", "glob"])
    .with_name("FileReader");

    assert_eq!(config.tools.len(), 2);
    assert!(config.tools.contains(&"read_file".to_string()));
    assert!(config.tools.contains(&"glob".to_string()));
    assert_eq!(config.name, Some("FileReader".to_string()));
}

#[test]
fn power_agent_has_all_builtins() {
    let config = AgentConfig::new("Power user")
        .with_all_builtins();

    // Should have all available builtin tools
    assert!(config.tools.len() >= 9);
    assert!(config.tools.contains(&"bash".to_string()));
    assert!(config.tools.contains(&"read_file".to_string()));
    assert!(config.tools.contains(&"calculate".to_string()));
}

#[test]
fn agent_config_serialization_roundtrip() {
    let config = AgentConfig::new("Test agent")
        .with_name("TestBot")
        .with_tools(&["read_file", "bash"])
        .with_max_conversation_length(50);

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config, deserialized);
}

#[test]
fn default_agent_has_no_tools() {
    let config = AgentConfig::default();
    assert!(config.tools.is_empty());
    assert_eq!(config.max_conversation_length, 100);
    assert!(config.enable_streaming);
}
```

---

## Testing agent state transitions

`AgentState` is a simple enum. Test its transition logic directly:

```rust
use acton_ai::agent::AgentState;

#[test]
fn idle_agent_can_accept_prompts() {
    assert!(AgentState::Idle.can_accept_prompt());
    assert!(AgentState::Completed.can_accept_prompt());
}

#[test]
fn active_agents_reject_prompts() {
    assert!(!AgentState::Thinking.can_accept_prompt());
    assert!(!AgentState::Executing.can_accept_prompt());
    assert!(!AgentState::Waiting.can_accept_prompt());
    assert!(!AgentState::Stopping.can_accept_prompt());
}

#[test]
fn only_stopping_is_terminal() {
    assert!(!AgentState::Idle.is_terminal());
    assert!(!AgentState::Completed.is_terminal());
    assert!(AgentState::Stopping.is_terminal());
}

#[test]
fn active_states_are_identified() {
    assert!(AgentState::Thinking.is_active());
    assert!(AgentState::Executing.is_active());
    assert!(AgentState::Waiting.is_active());
    assert!(!AgentState::Idle.is_active());
}
```

---

## Testing delegation tracking

The `DelegationTracker` manages task state without any async dependencies, making it straightforward to test:

```rust
use acton_ai::agent::delegation::{
    DelegatedTask, DelegatedTaskState, DelegationTracker,
};
use acton_ai::types::{AgentId, TaskId};
use std::time::Duration;

#[test]
fn track_and_complete_outgoing_task() {
    let mut tracker = DelegationTracker::new();
    let task_id = TaskId::new();
    let agent_id = AgentId::new();

    let task = DelegatedTask::new(
        task_id.clone(),
        agent_id,
        "code_review".to_string(),
    );
    tracker.track_outgoing(task);

    assert_eq!(tracker.pending_outgoing_count(), 1);

    // Complete the task
    let task = tracker.get_outgoing_mut(&task_id).unwrap();
    task.complete(serde_json::json!({"approved": true}));

    assert_eq!(tracker.pending_outgoing_count(), 0);
    assert!(task.is_terminal());
}

#[test]
fn track_incoming_task_acceptance() {
    let mut tracker = DelegationTracker::new();
    let task_id = TaskId::new();
    let from_agent = AgentId::new();

    tracker.track_incoming(
        task_id.clone(),
        from_agent,
        "analysis".to_string(),
    );

    assert_eq!(tracker.pending_incoming_count(), 1);
    assert!(tracker.accept_incoming(&task_id));
    assert_eq!(tracker.pending_incoming_count(), 0);
}

#[test]
fn overdue_detection() {
    let task_id = TaskId::new();
    let agent_id = AgentId::new();
    let task = DelegatedTask::new(task_id, agent_id, "test".to_string())
        .with_deadline(Duration::from_millis(1));

    std::thread::sleep(Duration::from_millis(5));
    assert!(task.is_overdue());
}

#[test]
fn cleanup_removes_only_terminal_tasks() {
    let mut tracker = DelegationTracker::new();
    let agent_id = AgentId::new();

    let task1_id = TaskId::new();
    let task2_id = TaskId::new();

    let task1 = DelegatedTask::new(
        task1_id.clone(), agent_id.clone(), "done".to_string()
    );
    let task2 = DelegatedTask::new(
        task2_id.clone(), agent_id, "pending".to_string()
    );

    tracker.track_outgoing(task1);
    tracker.track_outgoing(task2);

    // Complete task1
    tracker.get_outgoing_mut(&task1_id).unwrap()
        .complete(serde_json::json!({}));

    tracker.cleanup_completed();

    assert!(tracker.get_outgoing(&task1_id).is_none());  // Removed
    assert!(tracker.get_outgoing(&task2_id).is_some());   // Still tracked
}
```

---

## Testing error handling

All error types are `Clone + PartialEq`, making them easy to construct and assert against:

```rust
use acton_ai::error::{
    ActonAIError, ActonAIErrorKind,
    KernelError, KernelErrorKind,
    AgentError, AgentErrorKind,
};
use acton_ai::tools::error::{ToolError, ToolErrorKind};
use acton_ai::llm::error::{LLMError, LLMErrorKind};
use std::time::Duration;

#[test]
fn acton_ai_error_classification() {
    let config_err = ActonAIError::configuration("app_name", "cannot be empty");
    assert!(config_err.is_configuration());
    assert!(!config_err.is_runtime_shutdown());

    let shutdown_err = ActonAIError::runtime_shutdown();
    assert!(shutdown_err.is_runtime_shutdown());
}

#[test]
fn llm_error_retry_classification() {
    // Retriable errors
    assert!(LLMError::network("timeout").is_retriable());
    assert!(LLMError::rate_limited(Duration::from_secs(10)).is_retriable());
    assert!(LLMError::model_overloaded("gpt-4").is_retriable());
    assert!(LLMError::timeout(Duration::from_secs(30)).is_retriable());
    assert!(LLMError::api_error(500, "internal error", None).is_retriable());
    assert!(LLMError::api_error(503, "unavailable", None).is_retriable());

    // Non-retriable errors
    assert!(!LLMError::authentication_failed("bad key").is_retriable());
    assert!(!LLMError::api_error(400, "bad request", None).is_retriable());
    assert!(!LLMError::invalid_request("missing field").is_retriable());
}

#[test]
fn rate_limit_provides_retry_after() {
    let err = LLMError::rate_limited(Duration::from_secs(60));
    assert_eq!(err.retry_after(), Some(Duration::from_secs(60)));

    let other = LLMError::network("timeout");
    assert_eq!(other.retry_after(), None);
}

#[test]
fn tool_error_retry_classification() {
    assert!(ToolError::timeout("slow", Duration::from_secs(30)).is_retriable());
    assert!(ToolError::sandbox_error("transient").is_retriable());
    assert!(!ToolError::not_found("missing").is_retriable());
    assert!(!ToolError::validation_failed("tool", "bad args").is_retriable());
}

#[test]
fn errors_support_equality_comparison() {
    let err1 = KernelError::shutting_down();
    let err2 = KernelError::shutting_down();
    assert_eq!(err1, err2);

    let err3 = KernelError::spawn_failed("reason");
    assert_ne!(err1, err3);
}

#[test]
fn error_display_messages_are_actionable() {
    let err = ToolError::not_found("calculator");
    let msg = err.to_string();
    assert!(msg.contains("calculator"));
    assert!(msg.contains("not found"));
    assert!(msg.contains("verify"));  // Actionable guidance
}
```

---

## Testing tool execution

### Testing tool definitions

Verify tool definitions are correctly structured:

```rust
use acton_ai::tools::builtins::get_tool_definition;

#[test]
fn bash_tool_definition_is_valid() {
    let def = get_tool_definition("bash").unwrap();
    assert_eq!(def.name, "bash");
    assert!(!def.description.is_empty());

    // Verify the schema has the expected structure
    let schema = &def.input_schema;
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["command"].is_object());
}

#[test]
fn all_builtin_tools_have_valid_definitions() {
    use acton_ai::tools::builtins::BuiltinTools;

    for tool_name in BuiltinTools::available() {
        let def = get_tool_definition(tool_name);
        assert!(def.is_ok(), "Tool '{}' has no definition", tool_name);

        let def = def.unwrap();
        assert!(!def.name.is_empty());
        assert!(!def.description.is_empty());
    }
}
```

### Testing path validation

Path validation is pure logic with no async dependencies:

```rust
use acton_ai::tools::security::{PathValidator, PathValidationError};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

#[test]
fn validates_paths_within_allowed_root() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("test.txt");
    std::fs::write(&file, "content").unwrap();

    let validator = PathValidator::new()
        .clear_allowed_roots()
        .with_allowed_root(dir.path().to_path_buf());

    assert!(validator.validate(&file).is_ok());
}

#[test]
fn rejects_path_traversal() {
    let validator = PathValidator::new();
    let result = validator.validate(Path::new("/some/../../../etc/passwd"));
    assert!(matches!(
        result,
        Err(PathValidationError::DeniedPattern { pattern, .. }) if pattern == ".."
    ));
}

#[test]
fn rejects_git_directory_access() {
    let dir = TempDir::new().unwrap();
    let git_dir = dir.path().join(".git");
    std::fs::create_dir(&git_dir).unwrap();

    let validator = PathValidator::new()
        .clear_allowed_roots()
        .with_allowed_root(dir.path().to_path_buf());

    let result = validator.validate(&git_dir);
    assert!(matches!(
        result,
        Err(PathValidationError::DeniedPattern { pattern, .. }) if pattern == ".git"
    ));
}

#[test]
fn rejects_paths_outside_allowed_roots() {
    let allowed = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    let file = outside.path().join("secret.txt");
    std::fs::write(&file, "secret").unwrap();

    let validator = PathValidator::new()
        .clear_allowed_roots()
        .with_allowed_root(allowed.path().to_path_buf());

    assert!(matches!(
        validator.validate(&file),
        Err(PathValidationError::OutsideAllowedRoots { .. })
    ));
}

#[test]
fn validates_parent_for_new_files() {
    let dir = TempDir::new().unwrap();
    let new_file = dir.path().join("new_file.txt");

    let validator = PathValidator::new()
        .clear_allowed_roots()
        .with_allowed_root(dir.path().to_path_buf());

    // File doesn't exist, but parent does and is allowed
    assert!(validator.validate_parent(&new_file).is_ok());
}
```

---

## Using StubSandbox for tests

The `StubSandbox` and `StubSandboxFactory` are test-only implementations that do not require a hypervisor. They return placeholder responses instead of executing code.

{% callout type="warning" title="Test-only" %}
`StubSandbox` is only available in `#[cfg(test)]` builds. It does NOT sandbox code and must never be used in production.
{% /callout %}

```rust
#[cfg(test)]
mod sandbox_tests {
    use acton_ai::tools::sandbox::{
        Sandbox, SandboxFactory,
        StubSandbox, StubSandboxFactory,
    };

    #[tokio::test]
    async fn stub_sandbox_returns_placeholder() {
        let sandbox = StubSandbox::new();
        let result = sandbox
            .execute("echo hello", serde_json::json!({}))
            .await;

        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["status"], "stub");
    }

    #[tokio::test]
    async fn destroyed_sandbox_rejects_execution() {
        let mut sandbox = StubSandbox::new();
        assert!(sandbox.is_alive());

        sandbox.destroy();
        assert!(!sandbox.is_alive());

        let result = sandbox.execute("code", serde_json::json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn stub_factory_creates_sandboxes() {
        let factory = StubSandboxFactory::new();
        assert!(factory.is_available());

        let sandbox = factory.create().await.unwrap();
        assert!(sandbox.is_alive());
    }

    #[test]
    fn stub_sandbox_sync_execution() {
        let sandbox = StubSandbox::new();
        let result = sandbox.execute_sync(
            "some code",
            serde_json::json!({"arg": 1}),
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["status"], "stub");
    }
}
```

---

## Testing sandbox configuration

`SandboxConfig` and `PoolConfig` have validation methods that can be tested without a hypervisor:

```rust
use acton_ai::tools::sandbox::SandboxConfig;
use acton_ai::tools::sandbox::hyperlight::PoolConfig;
use std::time::Duration;

#[test]
fn valid_sandbox_config_passes_validation() {
    let config = SandboxConfig::new()
        .with_memory_limit(128 * 1024 * 1024)
        .with_timeout(Duration::from_secs(60))
        .with_pool_size(Some(8));

    assert!(config.validate().is_ok());
}

#[test]
fn rejects_memory_below_1mb() {
    let config = SandboxConfig::new().with_memory_limit(1000);
    assert!(config.validate().is_err());
}

#[test]
fn rejects_zero_timeout() {
    let config = SandboxConfig::new().with_timeout(Duration::ZERO);
    assert!(config.validate().is_err());
}

#[test]
fn rejects_zero_pool_size() {
    let config = SandboxConfig::new().with_pool_size(Some(0));
    assert!(config.validate().is_err());
}

#[test]
fn none_pool_size_is_valid() {
    let config = SandboxConfig::new().without_pool();
    assert!(config.validate().is_ok());
}

#[test]
fn pool_config_validation() {
    let config = PoolConfig::new()
        .with_warmup_count(8)
        .with_max_per_type(64)
        .with_max_executions_before_recycle(500);

    assert!(config.validate().is_ok());

    // Zero max_per_type is invalid
    let bad = PoolConfig::new().with_max_per_type(0);
    assert!(bad.validate().is_err());
}
```

---

## Integration testing with a running LLM

For end-to-end tests that verify actual LLM interaction, use a local Ollama instance:

```rust
use acton_ai::prelude::*;

/// Integration test requiring Ollama running locally.
/// Run with: cargo test -- --ignored
#[tokio::test]
#[ignore = "requires running Ollama instance"]
async fn basic_prompt_returns_response() {
    let runtime = ActonAI::builder()
        .app_name("integration-test")
        .ollama("qwen2.5:7b")
        .launch()
        .await
        .expect("Failed to launch runtime");

    let response = runtime
        .prompt("What is 2 + 2? Answer with just the number.")
        .collect()
        .await
        .expect("Prompt failed");

    assert!(!response.text.is_empty());
    assert!(response.text.contains("4"));
    assert!(response.token_count > 0);

    runtime.shutdown().await.unwrap();
}

#[tokio::test]
#[ignore = "requires running Ollama instance"]
async fn conversation_maintains_context() {
    let runtime = ActonAI::builder()
        .app_name("conv-test")
        .ollama("qwen2.5:7b")
        .launch()
        .await
        .unwrap();

    let conv = runtime.conversation()
        .system("You are a math tutor. Be concise.")
        .build()
        .await;

    let r1 = conv.send("What is 5 + 3?").await.unwrap();
    assert!(!r1.text.is_empty());

    // Verify history is maintained
    assert_eq!(conv.len(), 2); // user + assistant

    let r2 = conv.send("Now multiply that by 2").await.unwrap();
    assert!(!r2.text.is_empty());
    assert_eq!(conv.len(), 4); // 2 user + 2 assistant

    runtime.shutdown().await.unwrap();
}

#[tokio::test]
#[ignore = "requires running Ollama instance"]
async fn tool_execution_works() {
    let runtime = ActonAI::builder()
        .app_name("tool-test")
        .ollama("qwen2.5:7b")
        .with_builtin_tools(&["calculate"])
        .launch()
        .await
        .unwrap();

    let response = runtime
        .prompt("Use the calculate tool to compute 42 * 17")
        .use_builtins()
        .collect()
        .await
        .unwrap();

    // The LLM should have called the calculate tool
    assert!(!response.tool_calls.is_empty());
    assert!(response.tool_calls.iter().any(|tc| tc.name == "calculate"));

    runtime.shutdown().await.unwrap();
}
```

{% callout type="note" title="Running integration tests" %}
Integration tests that require Ollama are marked with `#[ignore]`. Run them explicitly with:

```bash
cargo test -- --ignored
```

Make sure Ollama is running locally on port 11434 with the `qwen2.5:7b` model pulled.
{% /callout %}

---

## Testing conversation history

Test history management without an LLM by verifying the builder and structural properties:

```rust
use acton_ai::messages::Message;

#[test]
fn message_construction() {
    let user_msg = Message::user("Hello");
    let assistant_msg = Message::assistant("Hi there!");

    // Messages can be serialized for persistence testing
    let json = serde_json::to_string(&user_msg).unwrap();
    let deserialized: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(user_msg.content, deserialized.content);
}

#[test]
fn chat_config_defaults() {
    use acton_ai::conversation::ChatConfig;

    let config = ChatConfig::new();
    // Check defaults match documentation
    let debug = format!("{:?}", config);
    assert!(debug.contains("You: "));
    assert!(debug.contains("Assistant: "));
}

#[test]
fn chat_config_input_mapper() {
    use acton_ai::conversation::ChatConfig;

    let mut config = ChatConfig::new()
        .map_input(|s| format!("[admin] {}", s));

    // The mapper is stored as Option<Box<dyn FnMut>>
    let debug = format!("{:?}", config);
    assert!(debug.contains("has_input_mapper"));
}
```

---

## Testing patterns summary

| What you are testing | Async? | LLM needed? | Pattern |
|---|---|---|---|
| `AgentConfig` construction | No | No | Direct struct creation and assertion |
| `AgentState` transitions | No | No | Call methods and assert results |
| `DelegationTracker` | No | No | Track/complete tasks and check counts |
| Error types and classification | No | No | Construct errors and test predicates |
| `PathValidator` | No | No | Create temp dirs and validate paths |
| `SandboxConfig` / `PoolConfig` | No | No | Builder methods and `validate()` |
| `StubSandbox` execution | Yes | No | `#[tokio::test]` with stub |
| Tool definitions | No | No | `get_tool_definition()` and schema checks |
| Prompt execution | Yes | Yes | `#[ignore]` with running Ollama |
| Conversation flow | Yes | Yes | `#[ignore]` with running Ollama |
| Tool invocation | Yes | Yes | `#[ignore]` with running Ollama |

---

## Next steps

- [Error Handling](/docs/error-handling) -- understand all error types for test assertions
- [Secure Tool Execution](/docs/secure-tool-execution) -- learn about `StubSandbox` vs `HyperlightSandbox`
- [Multi-Agent Collaboration](/docs/multi-agent-collaboration) -- test delegation and agent coordination
