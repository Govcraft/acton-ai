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
