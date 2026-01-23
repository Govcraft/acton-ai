//! Memory and persistence module for Acton-AI.
//!
//! This module provides persistent storage capabilities using libSQL (Turso's SQLite fork).
//! It enables agents to save and restore conversation history and state across restarts.
//!
//! ## Architecture
//!
//! - [`MemoryStore`]: Actor that manages all database operations asynchronously
//! - [`PersistenceConfig`]: Configuration for database connections
//! - [`AgentStateSnapshot`]: Serializable agent state for persistence
//!
//! ## Example
//!
//! ```rust,ignore
//! use acton_ai::prelude::*;
//! use acton_ai::memory::{MemoryStore, InitMemoryStore, PersistenceConfig};
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
//!     runtime.shutdown_all().await.unwrap();
//! }
//! ```

mod error;
mod persistence;
mod store;

pub use error::{PersistenceError, PersistenceErrorKind};
pub use persistence::{
    delete_agent_state, AgentStateSnapshot, PersistenceConfig, SCHEMA_VERSION,
};
pub use store::{
    AgentStateLoaded, ConversationCreated, ConversationList, ConversationLoaded,
    CreateConversation, DeleteConversation, GetLatestConversation, InitMemoryStore,
    LatestConversationResponse, ListConversations, LoadAgentState, LoadConversation, MemoryStore,
    MemoryStoreMetrics, MessageSaved, SaveAgentState, SaveMessage, SharedConnection,
};
