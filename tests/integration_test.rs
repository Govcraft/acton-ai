//! Integration tests for the Acton-AI framework.
//!
//! These tests verify that the core components work together correctly:
//! - Kernel spawning
//! - Agent spawning via Kernel
//! - Message passing between actors

use acton_ai::prelude::*;
use std::time::Duration;

/// Test that we can spawn the kernel actor.
#[tokio::test]
async fn test_spawn_kernel() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn the kernel
    let kernel_handle = Kernel::spawn(&mut runtime).await;

    // The kernel should have a valid handle - check that root is "kernel"
    let id_str = kernel_handle.id().root.to_string();
    assert!(id_str.contains("kernel"), "Expected kernel in id: {}", id_str);

    // Graceful shutdown
    runtime.shutdown_all().await.expect("Shutdown failed");
}

/// Test spawning an agent via the kernel.
#[tokio::test]
async fn test_spawn_agent_via_kernel() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn the kernel
    let kernel_handle = Kernel::spawn(&mut runtime).await;

    // Create agent configuration
    let agent_config = AgentConfig::new("You are a helpful test assistant.")
        .with_name("TestAgent")
        .with_max_conversation_length(10);

    // Send spawn request to kernel
    kernel_handle
        .send(SpawnAgent {
            config: agent_config,
        })
        .await;

    // Give some time for message processing
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Graceful shutdown
    runtime.shutdown_all().await.expect("Shutdown failed");
}

/// Test creating an agent directly and sending a user prompt.
#[tokio::test]
async fn test_agent_receives_user_prompt() {
    let mut runtime = ActonApp::launch_async().await;

    // Create an agent directly
    let agent_builder = Agent::create(&mut runtime);
    let agent_handle = agent_builder.start().await;

    // Initialize the agent with config
    let agent_id = AgentId::new();
    let agent_config = AgentConfig::new("You are a test assistant.").with_id(agent_id.clone());

    agent_handle
        .send(InitAgent {
            config: agent_config,
        })
        .await;

    // Send a user prompt
    let prompt = UserPrompt::new("Hello, agent!");
    agent_handle.send(prompt).await;

    // Give some time for message processing
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Graceful shutdown
    runtime.shutdown_all().await.expect("Shutdown failed");
}

/// Test that agent IDs are unique.
#[test]
fn test_agent_ids_are_unique() {
    let id1 = AgentId::new();
    let id2 = AgentId::new();
    let id3 = AgentId::new();

    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_ne!(id1, id3);
}

/// Test that correlation IDs are unique.
#[test]
fn test_correlation_ids_are_unique() {
    let id1 = CorrelationId::new();
    let id2 = CorrelationId::new();
    let id3 = CorrelationId::new();

    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_ne!(id1, id3);
}

/// Test agent configuration builder.
#[test]
fn test_agent_config_builder() {
    let agent_id = AgentId::new();
    let config = AgentConfig::new("Test prompt")
        .with_id(agent_id.clone())
        .with_name("TestBot")
        .with_max_conversation_length(50)
        .with_streaming(false);

    assert_eq!(config.id, Some(agent_id));
    assert_eq!(config.name, Some("TestBot".to_string()));
    assert_eq!(config.system_prompt, "Test prompt");
    assert_eq!(config.max_conversation_length, 50);
    assert!(!config.enable_streaming);
}

/// Test kernel configuration builder.
#[test]
fn test_kernel_config_builder() {
    let config = KernelConfig::new()
        .with_max_agents(25)
        .with_metrics(false)
        .with_default_system_prompt("Default prompt");

    assert_eq!(config.max_agents, 25);
    assert!(!config.enable_metrics);
    assert_eq!(
        config.default_system_prompt,
        Some("Default prompt".to_string())
    );
}

/// Test message creation helpers.
#[test]
fn test_message_creation() {
    let user_msg = Message::user("Hello");
    assert_eq!(user_msg.role, MessageRole::User);
    assert_eq!(user_msg.content, "Hello");

    let assistant_msg = Message::assistant("Hi there");
    assert_eq!(assistant_msg.role, MessageRole::Assistant);
    assert_eq!(assistant_msg.content, "Hi there");

    let system_msg = Message::system("Be helpful");
    assert_eq!(system_msg.role, MessageRole::System);
    assert_eq!(system_msg.content, "Be helpful");

    let tool_msg = Message::tool("tc_123", "Result");
    assert_eq!(tool_msg.role, MessageRole::Tool);
    assert_eq!(tool_msg.tool_call_id, Some("tc_123".to_string()));
}

/// Test agent state transitions.
#[test]
fn test_agent_state_transitions() {
    // Idle state can accept prompts
    assert!(AgentState::Idle.can_accept_prompt());
    assert!(AgentState::Completed.can_accept_prompt());

    // Busy states cannot
    assert!(!AgentState::Thinking.can_accept_prompt());
    assert!(!AgentState::Executing.can_accept_prompt());
    assert!(!AgentState::Waiting.can_accept_prompt());
    assert!(!AgentState::Stopping.can_accept_prompt());

    // Active states
    assert!(AgentState::Thinking.is_active());
    assert!(AgentState::Executing.is_active());
    assert!(AgentState::Waiting.is_active());

    // Not active states
    assert!(!AgentState::Idle.is_active());
    assert!(!AgentState::Completed.is_active());
    assert!(!AgentState::Stopping.is_active());
}

/// Test error types.
#[test]
fn test_kernel_error_display() {
    let agent_id = AgentId::new();
    let error = KernelError::agent_not_found(agent_id.clone());

    let msg = error.to_string();
    assert!(msg.contains(&agent_id.to_string()));
    assert!(msg.contains("not found"));
}

/// Test error types for agent.
#[test]
fn test_agent_error_display() {
    let agent_id = AgentId::new();
    let error = AgentError::invalid_state(Some(agent_id.clone()), "Idle", "Thinking or Executing");

    let msg = error.to_string();
    assert!(msg.contains(&agent_id.to_string()));
    assert!(msg.contains("invalid state"));
}

/// Test UserPrompt automatically generates correlation ID.
#[test]
fn test_user_prompt_generates_correlation_id() {
    let prompt1 = UserPrompt::new("Test 1");
    let prompt2 = UserPrompt::new("Test 2");

    // Each prompt should have a unique correlation ID
    assert_ne!(prompt1.correlation_id, prompt2.correlation_id);

    // Correlation IDs should have the correct prefix
    assert!(prompt1.correlation_id.to_string().starts_with("corr_"));
    assert!(prompt2.correlation_id.to_string().starts_with("corr_"));
}

// ============================================================================
// Phase 2: LLM Integration Tests
// ============================================================================

/// Test LLM provider configuration.
#[test]
fn test_llm_provider_config() {
    let config = ProviderConfig::new("test-api-key")
        .with_model("claude-3-haiku-20240307")
        .with_max_tokens(2048);

    assert_eq!(config.api_key, "test-api-key");
    assert_eq!(config.model, "claude-3-haiku-20240307");
    assert_eq!(config.max_tokens, 2048);
}

/// Test LLM error types.
#[test]
fn test_llm_error_types() {
    use std::time::Duration;

    // Network error is retriable
    let net_err = LLMError::network("connection refused");
    assert!(net_err.is_retriable());

    // Rate limited error is retriable with retry-after
    let rate_err = LLMError::rate_limited(Duration::from_secs(30));
    assert!(rate_err.is_retriable());
    assert_eq!(rate_err.retry_after(), Some(Duration::from_secs(30)));

    // Auth error is not retriable
    let auth_err = LLMError::authentication_failed("invalid key");
    assert!(!auth_err.is_retriable());
}

/// Test rate limit configuration.
#[test]
fn test_rate_limit_config() {
    let config = RateLimitConfig::new(60, 50_000)
        .with_max_queue_size(200);

    assert_eq!(config.requests_per_minute, 60);
    assert_eq!(config.tokens_per_minute, 50_000);
    assert_eq!(config.max_queue_size, 200);
    assert!(config.queue_when_limited);

    let no_queue_config = config.without_queueing();
    assert!(!no_queue_config.queue_when_limited);
}

/// Test LLM request message creation.
#[test]
fn test_llm_request_creation() {
    let corr_id = CorrelationId::new();
    let agent_id = AgentId::new();

    let request = LLMRequest {
        correlation_id: corr_id.clone(),
        agent_id: agent_id.clone(),
        messages: vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
        ],
        tools: None,
    };

    assert_eq!(request.correlation_id, corr_id);
    assert_eq!(request.agent_id, agent_id);
    assert_eq!(request.messages.len(), 2);
    assert!(request.tools.is_none());
}

/// Test LLM response message handling.
#[test]
fn test_llm_response_handling() {
    let corr_id = CorrelationId::new();

    let response = LLMResponse {
        correlation_id: corr_id.clone(),
        content: "Hello! How can I help you today?".to_string(),
        tool_calls: None,
        stop_reason: StopReason::EndTurn,
    };

    assert_eq!(response.correlation_id, corr_id);
    assert_eq!(response.content, "Hello! How can I help you today?");
    assert!(response.tool_calls.is_none());
    assert_eq!(response.stop_reason, StopReason::EndTurn);
}

/// Test LLM response with tool calls.
#[test]
fn test_llm_response_with_tools() {
    let corr_id = CorrelationId::new();

    let tool_call = ToolCall {
        id: "tc_123".to_string(),
        name: "search".to_string(),
        arguments: serde_json::json!({"query": "rust programming"}),
    };

    let response = LLMResponse {
        correlation_id: corr_id.clone(),
        content: "Let me search for that.".to_string(),
        tool_calls: Some(vec![tool_call.clone()]),
        stop_reason: StopReason::ToolUse,
    };

    assert!(response.tool_calls.is_some());
    let tool_calls = response.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "search");
    assert_eq!(response.stop_reason, StopReason::ToolUse);
}

/// Test streaming message flow.
#[test]
fn test_streaming_messages() {
    let corr_id = CorrelationId::new();

    // Stream start
    let start = LLMStreamStart {
        correlation_id: corr_id.clone(),
    };
    assert_eq!(start.correlation_id, corr_id);

    // Stream tokens
    let token1 = LLMStreamToken {
        correlation_id: corr_id.clone(),
        token: "Hello".to_string(),
    };
    let token2 = LLMStreamToken {
        correlation_id: corr_id.clone(),
        token: " World".to_string(),
    };
    assert_eq!(token1.token, "Hello");
    assert_eq!(token2.token, " World");

    // Stream end
    let end = LLMStreamEnd {
        correlation_id: corr_id.clone(),
        stop_reason: StopReason::EndTurn,
    };
    assert_eq!(end.stop_reason, StopReason::EndTurn);
}

/// Test stream accumulator functionality.
#[test]
fn test_stream_accumulator() {
    use acton_ai::llm::StreamAccumulator;

    let mut accumulator = StreamAccumulator::new();
    let corr_id = CorrelationId::new();

    // Start a stream
    accumulator.start_stream(&corr_id);
    assert_eq!(accumulator.active_count(), 1);

    // Append tokens
    accumulator.append_token(&corr_id, "Hello");
    accumulator.append_token(&corr_id, " ");
    accumulator.append_token(&corr_id, "World");

    // End the stream
    let stream = accumulator.end_stream(&corr_id, StopReason::EndTurn).unwrap();
    assert_eq!(stream.content, "Hello World");
    assert!(stream.is_ended());
    assert!(accumulator.is_empty());
}

/// Test stop reason variants.
#[test]
fn test_stop_reason_variants() {
    assert_eq!(StopReason::EndTurn, StopReason::EndTurn);
    assert_ne!(StopReason::EndTurn, StopReason::MaxTokens);
    assert_ne!(StopReason::EndTurn, StopReason::ToolUse);
    assert_ne!(StopReason::EndTurn, StopReason::StopSequence);

    // Test serialization
    let json = serde_json::to_string(&StopReason::EndTurn).unwrap();
    assert!(json.contains("end_turn") || json.contains("EndTurn"));
}

/// Test tool definition structure.
#[test]
fn test_tool_definition() {
    let tool = ToolDefinition {
        name: "calculator".to_string(),
        description: "Performs mathematical calculations".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }),
    };

    assert_eq!(tool.name, "calculator");
    assert!(tool.input_schema.is_object());
}

// ============================================================================
// Phase 3: Tool System Integration Tests
// ============================================================================

use acton_ai::tools::{
    RegisterTool, ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait, ToolRegistry,
};
use std::sync::Arc;

/// A simple echo tool for testing.
#[derive(Debug)]
struct EchoTool;

impl ToolExecutorTrait for EchoTool {
    fn execute(&self, args: serde_json::Value) -> ToolExecutionFuture {
        Box::pin(async move { Ok(args) })
    }
}

/// A tool that always fails for testing error handling.
#[derive(Debug)]
struct FailingTool;

impl ToolExecutorTrait for FailingTool {
    fn execute(&self, _args: serde_json::Value) -> ToolExecutionFuture {
        Box::pin(async move { Err(ToolError::execution_failed("failing_tool", "always fails")) })
    }
}

/// Test spawning the Tool Registry actor.
#[tokio::test]
async fn test_spawn_tool_registry() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn the registry
    let registry = ToolRegistry::spawn(&mut runtime).await;

    // The registry should have a valid handle - check that root is "tool_registry"
    let id_str = registry.id().root.to_string();
    assert!(
        id_str.contains("tool_registry"),
        "Expected tool_registry in id: {}",
        id_str
    );

    // Graceful shutdown
    runtime.shutdown_all().await.expect("Shutdown failed");
}

/// Test registering a tool with the registry.
#[tokio::test]
async fn test_register_tool() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn the registry
    let registry = ToolRegistry::spawn(&mut runtime).await;

    // Create a tool definition
    let tool_def = ToolDefinition {
        name: "echo".to_string(),
        description: "Echoes input back".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            }
        }),
    };

    // Register the tool
    registry
        .send(RegisterTool {
            config: ToolConfig::new(tool_def),
            executor: Arc::new(Box::new(EchoTool) as Box<dyn ToolExecutorTrait>),
        })
        .await;

    // Give time for message processing
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Graceful shutdown
    runtime.shutdown_all().await.expect("Shutdown failed");
}

/// Test tool error types.
#[test]
fn test_tool_error_not_found() {
    let error = ToolError::not_found("missing_tool");
    let msg = error.to_string();
    assert!(msg.contains("missing_tool"));
    assert!(msg.contains("not found"));
    assert!(error.is_not_found());
}

/// Test tool error already registered.
#[test]
fn test_tool_error_already_registered() {
    let error = ToolError::already_registered("duplicate_tool");
    let msg = error.to_string();
    assert!(msg.contains("duplicate_tool"));
    assert!(msg.contains("already registered"));
    assert!(error.is_already_registered());
}

/// Test tool error execution failed.
#[test]
fn test_tool_error_execution_failed() {
    let error = ToolError::execution_failed("buggy_tool", "null pointer");
    let msg = error.to_string();
    assert!(msg.contains("buggy_tool"));
    assert!(msg.contains("execution failed"));
    assert!(msg.contains("null pointer"));
}

/// Test tool error timeout.
#[test]
fn test_tool_error_timeout() {
    let error = ToolError::timeout("slow_tool", Duration::from_secs(30));
    let msg = error.to_string();
    assert!(msg.contains("slow_tool"));
    assert!(msg.contains("timed out"));
    assert!(msg.contains("30"));
    assert!(error.is_retriable());
}

/// Test tool error validation failed.
#[test]
fn test_tool_error_validation_failed() {
    let error = ToolError::validation_failed("strict_tool", "missing required field");
    let msg = error.to_string();
    assert!(msg.contains("strict_tool"));
    assert!(msg.contains("validation failed"));
    assert!(msg.contains("missing required field"));
}

/// Test tool config builder.
#[test]
fn test_tool_config_builder() {
    let tool_def = ToolDefinition {
        name: "test".to_string(),
        description: "Test tool".to_string(),
        input_schema: serde_json::json!({"type": "object"}),
    };

    let config = ToolConfig::new(tool_def)
        .with_sandbox(true)
        .with_timeout(Duration::from_secs(60));

    assert!(config.sandboxed);
    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.definition.name, "test");
}

/// Test ToolName type.
#[test]
fn test_tool_name_creation() {
    use acton_ai::types::ToolName;

    let name1 = ToolName::new();
    let name2 = ToolName::new();

    // Each name should be unique
    assert_ne!(name1, name2);

    // Names should have correct prefix
    assert!(name1.to_string().starts_with("tool_"));
    assert!(name2.to_string().starts_with("tool_"));
}

/// Test ToolName parsing.
#[test]
fn test_tool_name_parsing() {
    use acton_ai::types::{InvalidToolName, ToolName};

    // Valid tool name should parse
    let name = ToolName::new();
    let parsed = ToolName::parse(&name.to_string());
    assert!(parsed.is_ok());
    assert_eq!(name, parsed.unwrap());

    // Wrong prefix should fail
    let result = ToolName::parse("agent_01h455vb4pex5vsknk084sn02q");
    assert!(matches!(
        result,
        Err(InvalidToolName::WrongPrefix {
            expected: "tool",
            ..
        })
    ));

    // Invalid format should fail
    let result = ToolName::parse("not-a-valid-typeid");
    assert!(matches!(result, Err(InvalidToolName::Parse(_))));
}

/// Test tool execution via executor trait.
#[tokio::test]
async fn test_echo_tool_execution() {
    let tool = EchoTool;
    let input = serde_json::json!({"message": "hello world"});
    let result = tool.execute(input.clone()).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), input);
}

/// Test failing tool execution.
#[tokio::test]
async fn test_failing_tool_execution() {
    let tool = FailingTool;
    let result = tool.execute(serde_json::json!({})).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("always fails"));
}

// ============================================================================
// Phase 4: Persistence Tests
// ============================================================================

use acton_ai::memory::{
    AgentStateSnapshot, InitMemoryStore, MemoryStore, PersistenceConfig, PersistenceError,
};
use acton_ai::types::{ConversationId, InvalidConversationId, InvalidMemoryId, InvalidMessageId, MemoryId, MessageId};

/// Test ConversationId type.
#[test]
fn test_conversation_id_creation() {
    let id1 = ConversationId::new();
    let id2 = ConversationId::new();

    // Each ID should be unique
    assert_ne!(id1, id2);

    // IDs should have correct prefix
    assert!(id1.to_string().starts_with("conv_"));
    assert!(id2.to_string().starts_with("conv_"));
}

/// Test ConversationId parsing.
#[test]
fn test_conversation_id_parsing() {
    // Valid conversation ID should parse
    let id = ConversationId::new();
    let parsed = ConversationId::parse(&id.to_string());
    assert!(parsed.is_ok());
    assert_eq!(id, parsed.unwrap());

    // Wrong prefix should fail
    let result = ConversationId::parse("agent_01h455vb4pex5vsknk084sn02q");
    assert!(matches!(
        result,
        Err(InvalidConversationId::WrongPrefix {
            expected: "conv",
            ..
        })
    ));

    // Invalid format should fail
    let result = ConversationId::parse("not-a-valid-typeid");
    assert!(matches!(result, Err(InvalidConversationId::Parse(_))));
}

/// Test MessageId type.
#[test]
fn test_message_id_creation() {
    let id1 = MessageId::new();
    let id2 = MessageId::new();

    // Each ID should be unique
    assert_ne!(id1, id2);

    // IDs should have correct prefix
    assert!(id1.to_string().starts_with("msg_"));
    assert!(id2.to_string().starts_with("msg_"));
}

/// Test MessageId parsing.
#[test]
fn test_message_id_parsing() {
    // Valid message ID should parse
    let id = MessageId::new();
    let parsed = MessageId::parse(&id.to_string());
    assert!(parsed.is_ok());
    assert_eq!(id, parsed.unwrap());

    // Wrong prefix should fail
    let result = MessageId::parse("agent_01h455vb4pex5vsknk084sn02q");
    assert!(matches!(
        result,
        Err(InvalidMessageId::WrongPrefix {
            expected: "msg",
            ..
        })
    ));

    // Invalid format should fail
    let result = MessageId::parse("not-a-valid-typeid");
    assert!(matches!(result, Err(InvalidMessageId::Parse(_))));
}

/// Test MemoryId type.
#[test]
fn test_memory_id_creation() {
    let id1 = MemoryId::new();
    let id2 = MemoryId::new();

    // Each ID should be unique
    assert_ne!(id1, id2);

    // IDs should have correct prefix
    assert!(id1.to_string().starts_with("mem_"));
    assert!(id2.to_string().starts_with("mem_"));
}

/// Test MemoryId parsing.
#[test]
fn test_memory_id_parsing() {
    // Valid memory ID should parse
    let id = MemoryId::new();
    let parsed = MemoryId::parse(&id.to_string());
    assert!(parsed.is_ok());
    assert_eq!(id, parsed.unwrap());

    // Wrong prefix should fail
    let result = MemoryId::parse("agent_01h455vb4pex5vsknk084sn02q");
    assert!(matches!(
        result,
        Err(InvalidMemoryId::WrongPrefix {
            expected: "mem",
            ..
        })
    ));

    // Invalid format should fail
    let result = MemoryId::parse("not-a-valid-typeid");
    assert!(matches!(result, Err(InvalidMemoryId::Parse(_))));
}

/// Test PersistenceConfig.
#[test]
fn test_persistence_config() {
    // Default config
    let config = PersistenceConfig::default();
    assert_eq!(config.db_path, "acton-ai.db");
    assert!(!config.is_in_memory());

    // In-memory config
    let mem_config = PersistenceConfig::in_memory();
    assert!(mem_config.is_in_memory());

    // Custom path config
    let custom_config = PersistenceConfig::new("/path/to/db.sqlite");
    assert_eq!(custom_config.db_path, "/path/to/db.sqlite");
}

/// Test PersistenceError types.
#[test]
fn test_persistence_error_database_open() {
    let error = PersistenceError::database_open("/path/to/db", "permission denied");
    let msg = error.to_string();
    assert!(msg.contains("/path/to/db"));
    assert!(msg.contains("permission denied"));
}

/// Test PersistenceError retriable.
#[test]
fn test_persistence_error_retriable() {
    // Connection errors are retriable
    let conn_err = PersistenceError::connection_error("timeout");
    assert!(conn_err.is_retriable());

    // Transaction errors are retriable
    let tx_err = PersistenceError::transaction_failed("deadlock");
    assert!(tx_err.is_retriable());

    // Not found errors are not retriable
    let not_found = PersistenceError::not_found("conversation", "conv_123");
    assert!(!not_found.is_retriable());
    assert!(not_found.is_not_found());
}

/// Test PersistenceError shutting down.
#[test]
fn test_persistence_error_shutting_down() {
    let error = PersistenceError::shutting_down();
    assert!(error.is_shutting_down());
    let msg = error.to_string();
    assert!(msg.contains("shutting down"));
}

/// Test AgentStateSnapshot serialization.
#[test]
fn test_agent_state_snapshot_serialization() {
    let snapshot = AgentStateSnapshot {
        agent_id: AgentId::new(),
        conversation_id: Some(ConversationId::new()),
        conversation: vec![Message::user("Hello"), Message::assistant("Hi there")],
        system_prompt: "You are a helpful assistant".to_string(),
    };

    // Serialize
    let json = serde_json::to_string(&snapshot).unwrap();
    assert!(json.contains("agent_"));
    assert!(json.contains("conv_"));

    // Deserialize
    let deserialized: AgentStateSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(snapshot.agent_id, deserialized.agent_id);
    assert_eq!(snapshot.system_prompt, deserialized.system_prompt);
    assert_eq!(snapshot.conversation.len(), deserialized.conversation.len());
}

/// Test spawning MemoryStore actor.
#[tokio::test]
async fn test_spawn_memory_store() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn the memory store
    let store_handle = MemoryStore::spawn(&mut runtime).await;

    // The store should have a valid handle - check that root is "memory_store"
    let id_str = store_handle.id().root.to_string();
    assert!(
        id_str.contains("memory_store"),
        "Expected memory_store in id: {}",
        id_str
    );

    // Graceful shutdown
    runtime.shutdown_all().await.expect("Shutdown failed");
}

/// Test initializing MemoryStore with in-memory database.
#[tokio::test]
async fn test_memory_store_initialization() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn the memory store
    let store_handle = MemoryStore::spawn(&mut runtime).await;

    // Initialize with in-memory database
    store_handle
        .send(InitMemoryStore {
            config: PersistenceConfig::in_memory(),
        })
        .await;

    // Give time for initialization
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Graceful shutdown
    runtime.shutdown_all().await.expect("Shutdown failed");
}

/// Test ConversationId serialization roundtrip.
#[test]
fn test_conversation_id_serialization() {
    let id = ConversationId::new();
    let json = serde_json::to_string(&id).unwrap();
    let deserialized: ConversationId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, deserialized);
}

/// Test MessageId serialization roundtrip.
#[test]
fn test_message_id_serialization() {
    let id = MessageId::new();
    let json = serde_json::to_string(&id).unwrap();
    let deserialized: MessageId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, deserialized);
}

/// Test MemoryId serialization roundtrip.
#[test]
fn test_memory_id_serialization() {
    let id = MemoryId::new();
    let json = serde_json::to_string(&id).unwrap();
    let deserialized: MemoryId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, deserialized);
}
