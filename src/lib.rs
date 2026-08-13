//! # Acton-AI: Agentic AI Framework
//!
//! An agentic AI framework where each agent is an actor, leveraging acton-reactive's
//! supervision, pub/sub, and fault tolerance to create resilient, concurrent AI systems.
//!
//! ## Architecture
//!
//! - **Agent**: Individual AI agents with reasoning loops
//! - **LLM Provider**: Manages streaming LLM API calls with rate limiting
//! - **Tool Registry**: Registers and executes tools via supervised child actors
//! - **MCP Client**: Consumes tools from external Model Context Protocol servers,
//!   one supervised actor per connection (see [`mcp`])
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
//!
//!     let provider = LLMProvider::spawn(&mut app, ProviderConfig::ollama("qwen2.5:7b")).await;
//!
//!     let mut agent = Agent::create(&mut app);
//!     let agent_handle = agent.start().await;
//!     agent_handle.send(InitAgent::default()).await;
//!
//!     app.shutdown_all().await.unwrap();
//! }
//! ```

pub mod agent;
pub mod cli;
pub mod config;
pub mod conversation;
pub mod error;
pub mod extract;
pub mod facade;
pub mod llm;
pub mod logging;
pub mod mcp;
pub mod memory;
pub mod messages;
pub mod prompt;
pub mod stream;
pub mod tools;
pub mod types;

pub mod skills;

/// Re-export of the [`schemars`] crate that generated schemas are built with.
///
/// [`PromptBuilder::extract`](crate::prompt::PromptBuilder::extract) requires
/// its target type to implement [`schemars::JsonSchema`]. Normally you add
/// `schemars` to your own `Cargo.toml` and derive it there; this re-export
/// exists so the version in use is discoverable and so documentation can link
/// to it. `JsonSchema` is deliberately **not** in the [`prelude`] — the name
/// collides too easily with application types.
pub use schemars;

/// Prelude module for convenient imports
pub mod prelude {
    // High-level API (recommended for most use cases)
    pub use crate::config::{
        ActonAIConfig, ActonAIDefaults, McpServerConfig, NamedProviderConfig, RateLimitFileConfig,
    };
    pub use crate::conversation::{
        ChatConfig, Conversation, ConversationBuilder, StreamToken, DEFAULT_SYSTEM_PROMPT,
    };
    pub use crate::error::{ActonAIError, ActonAIErrorKind};
    pub use crate::extract::STRUCTURED_OUTPUT_TOOL;
    pub use crate::facade::{ActonAI, ActonAIBuilder, DEFAULT_PROVIDER_NAME};
    pub use crate::mcp::{McpError, McpErrorKind, McpTools};
    pub use crate::stream::{CollectedResponse, StreamAction, StreamHandler};

    // Low-level API (for advanced use cases)
    pub use crate::agent::{
        Agent, AgentConfig, AgentState, DelegatedTask, DelegatedTaskState, DelegationTracker,
        IncomingTaskInfo, InitAgent,
    };
    pub use crate::error::AgentError;
    pub use crate::llm::{
        AnthropicClient, InitLLMProvider, LLMClient, LLMClientResponse, LLMError, LLMErrorKind,
        LLMEventStream, LLMProvider, LLMStreamEvent, OpenAIClient, ProviderConfig, ProviderType,
        RateLimitConfig, SamplingParams,
    };
    pub use crate::logging::{
        init_and_store_logging, init_journald_logging, journald_layer, mark_subscriber_installed,
        LogLevel, LoggingConfig, LoggingError, LoggingErrorKind,
    };
    pub use crate::memory::{
        AgentStateSnapshot, ContextStats, ContextWindow, ContextWindowConfig, ContextWindowData,
        ContextWindowResponse, Embedding, EmbeddingError, EmbeddingProvider, GetContextWindow,
        InitMemoryStore, LoadMemories, MemoriesLoaded, Memory, MemorySearchResults, MemoryStore,
        MemoryStoreMetrics, MemoryStoreMetricsSnapshot, MemoryStored, OperationCompleted,
        PersistenceConfig, PersistenceError, ScoredMemory, SearchMemories, StoreMemory,
        StubEmbeddingProvider, TruncationStrategy,
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

    // Agent Skills
    pub use crate::skills::{LoadedSkill, SkillInfo, SkillRegistry, SkillsError};
    pub use crate::tools::builtins::{
        skill_tool_names, spawn_skill_tool_actors, ActivateSkillTool, ActivateSkillToolActor,
        ListSkillsTool, ListSkillsToolActor,
    };
}
