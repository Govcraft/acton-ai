//! Memory and persistence module for Acton-AI.
//!
//! This module provides persistent storage capabilities using libSQL (Turso's SQLite fork).
//! It enables agents to save and restore conversation history, state, and memories with
//! optional vector embeddings for semantic search.
//!
//! ## Architecture
//!
//! - [`MemoryStore`]: Actor that manages all database operations asynchronously
//! - [`PersistenceConfig`]: Configuration for database connections
//! - [`AgentStateSnapshot`]: Serializable agent state for persistence
//! - [`Embedding`]: Vector embeddings for semantic memory search
//! - [`EmbeddingProvider`]: Trait for embedding generation services
//! - [`Memory`]: A memory entry with optional embedding
//! - [`ContextWindow`]: Context window management for LLM interactions
//!
//! ## Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//! use acton_ai::memory::{
//!     MemoryStore, InitMemoryStore, PersistenceConfig,
//!     StoreMemory, SearchMemories, StubEmbeddingProvider, EmbeddingProvider,
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut runtime = ActonApp::launch_async().await;
//!
//!     // Spawn memory store
//!     let store = MemoryStore::spawn(&mut runtime).await;
//!
//!     // Initialize with in-memory database for testing
//!     store.send(InitMemoryStore {
//!         config: PersistenceConfig::in_memory(),
//!     }).await;
//!
//!     // Store a memory with embedding
//!     let provider = StubEmbeddingProvider::default();
//!     let embedding = provider.embed("User prefers dark mode").await.unwrap();
//!
//!     let agent_id = AgentId::new();
//!     store.send(StoreMemory {
//!         agent_id: agent_id.clone(),
//!         content: "User prefers dark mode".to_string(),
//!         embedding: Some(embedding),
//!     }).await;
//!
//!     runtime.shutdown_all().await.unwrap();
//! }
//! ```

mod context;
mod embeddings;
mod error;
mod persistence;
mod store;

// Re-export context window types
pub use context::{ContextStats, ContextWindow, ContextWindowConfig, TruncationStrategy};

// Re-export embedding types
pub use embeddings::{
    Embedding, EmbeddingError, EmbeddingProvider, Memory, ScoredMemory, StubEmbeddingProvider,
};

// Re-export error types
pub use error::{PersistenceError, PersistenceErrorKind};

// Re-export persistence types
pub use persistence::{
    count_memories_for_agent, delete_agent_state, delete_memories_for_agent, delete_memory,
    load_memories_for_agent, save_memory, search_memories_by_embedding, AgentStateSnapshot,
    PersistenceConfig, SCHEMA_VERSION,
};

// Re-export store types and messages
pub use store::{
    // Agent state messages
    AgentStateLoaded,
    // Context window messages
    ContextWindowResponse,
    // Conversation messages
    ConversationCreated,
    ConversationList,
    ConversationLoaded,
    CreateConversation,
    // Memory messages
    DeleteAgentMemories,
    DeleteConversation,
    DeleteMemory,
    GetContextWindow,
    GetLatestConversation,
    InitMemoryStore,
    LatestConversationResponse,
    ListConversations,
    LoadAgentState,
    LoadConversation,
    LoadMemories,
    MemoriesLoaded,
    MemorySearchResults,
    // Core store types
    MemoryStore,
    MemoryStoreMetrics,
    MemoryStored,
    // Message store messages
    MessageSaved,
    SaveAgentState,
    SaveMessage,
    SearchMemories,
    SharedConnection,
    StoreMemory,
};
