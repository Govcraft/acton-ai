//! # Acton-AI: Agentic AI Framework
//!
//! An agentic AI framework where each agent is an actor, leveraging acton-reactive's
//! supervision, pub/sub, and fault tolerance to create resilient, concurrent AI systems.
//!
//! ## Architecture
//!
//! - **Kernel**: Central supervisor managing all agents
//! - **Agent**: Individual AI agents with reasoning loops
//! - **LLM Provider**: Manages streaming LLM API calls with rate limiting
//! - **Tool Registry**: Registers and executes tools via supervised child actors
//! - **Memory Store**: Persistence via Turso/libSQL
//!
//! ## Quick Start (High-Level API)
//!
//! The simplest way to use acton-ai is via the `ActonAI` facade:
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), ActonAIError> {
//!     let runtime = ActonAI::builder()
//!         .app_name("my-app")
//!         .ollama("qwen2.5:7b")
//!         .launch()
//!         .await?;
//!
//!     runtime
//!         .prompt("What is the capital of France?")
//!         .system("Be concise.")
//!         .on_token(|t| print!("{t}"))
//!         .collect()
//!         .await?;
//!
//!     println!();
//!     Ok(())
//! }
//! ```
//!
//! ## Advanced Usage (Low-Level API)
//!
//! For full control over the actor system:
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut app = ActonApp::launch_async().await;
//!     let kernel = Kernel::spawn(&mut app).await;
//!
//!     let agent_id = kernel.spawn_agent(AgentConfig::default()).await;
//!     kernel.send_prompt(agent_id, "Hello, agent!").await;
//!
//!     app.shutdown_all().await.unwrap();
//! }
//! ```

pub mod agent;
pub mod config;
pub mod conversation;
pub mod error;
pub mod facade;
pub mod kernel;
pub mod llm;
pub mod memory;
pub mod messages;
pub mod prompt;
pub mod stream;
pub mod tools;
pub mod types;

#[cfg(feature = "agent-skills")]
pub mod skills;

/// Prelude module for convenient imports
pub mod prelude {
    // High-level API (recommended for most use cases)
    pub use crate::config::{ActonAIConfig, NamedProviderConfig, RateLimitFileConfig};
    pub use crate::conversation::{ChatConfig, Conversation, ConversationBuilder, DEFAULT_SYSTEM_PROMPT};
    pub use crate::error::{ActonAIError, ActonAIErrorKind};
    pub use crate::facade::{ActonAI, ActonAIBuilder, DEFAULT_PROVIDER_NAME};
    pub use crate::stream::{CollectedResponse, StreamAction, StreamHandler};

    // Low-level API (for advanced use cases)
    pub use crate::agent::{
        Agent, AgentConfig, AgentState, DelegatedTask, DelegatedTaskState, DelegationTracker,
        IncomingTaskInfo, InitAgent,
    };
    pub use crate::error::{AgentError, KernelError, MultiAgentError, MultiAgentErrorKind};
    pub use crate::kernel::{
        get_log_dir, init_and_store_logging, init_file_logging, CapabilityRegistry, InitKernel,
        Kernel, KernelConfig, KernelMetrics, LogLevel, LoggingConfig, LoggingError,
        LoggingErrorKind, LoggingGuard,
    };
    pub use crate::llm::{
        AnthropicClient, InitLLMProvider, LLMClient, LLMClientResponse, LLMError, LLMErrorKind,
        LLMEventStream, LLMProvider, LLMStreamEvent, OpenAIClient, ProviderConfig, ProviderType,
        RateLimitConfig,
    };
    pub use crate::memory::{
        AgentStateSnapshot, ContextStats, ContextWindow, ContextWindowConfig,
        ContextWindowResponse, Embedding, EmbeddingError, EmbeddingProvider, GetContextWindow,
        InitMemoryStore, LoadMemories, MemoriesLoaded, Memory, MemorySearchResults, MemoryStore,
        MemoryStoreMetrics, MemoryStored, PersistenceConfig, PersistenceError, ScoredMemory,
        SearchMemories, StoreMemory, StubEmbeddingProvider, TruncationStrategy,
    };
    pub use crate::messages::*;
    pub use crate::tools::builtins::BuiltinTools;
    pub use crate::tools::{
        RegisterTool, ToolConfig, ToolDefinition, ToolError, ToolErrorKind, ToolExecutorTrait,
        ToolRegistry,
    };
    pub use crate::types::{
        AgentId, ConversationId, CorrelationId, InvalidTaskId, MemoryId, MessageId, TaskId,
        ToolName,
    };

    // Re-export acton-reactive prelude
    pub use acton_reactive::prelude::*;

    // Agent Skills (feature-gated)
    #[cfg(feature = "agent-skills")]
    pub use crate::skills::{LoadedSkill, SkillInfo, SkillRegistry, SkillsError};
    #[cfg(feature = "agent-skills")]
    pub use crate::tools::builtins::{
        spawn_skill_tool_actors, skill_tool_names, ActivateSkillTool, ActivateSkillToolActor,
        ListSkillsTool, ListSkillsToolActor,
    };
}
