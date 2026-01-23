//! Database schema and query functions for persistence.
//!
//! This module provides pure functions for database operations, taking
//! connections as parameters and returning results. All functions are
//! async and use libSQL for database access.

use crate::memory::embeddings::{Embedding, Memory, ScoredMemory};
use crate::memory::error::PersistenceError;
use crate::messages::{Message, MessageRole};
use crate::types::{AgentId, ConversationId, MemoryId, MessageId};
use libsql::{Connection, Database};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Database schema version for migrations.
pub const SCHEMA_VERSION: u32 = 1;

/// SQL statements for schema creation.
const CREATE_SCHEMA: &str = r"
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_conversations_agent_id ON conversations(agent_id);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT,
    tool_call_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id);

CREATE TABLE IF NOT EXISTS agent_state (
    agent_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
";

/// Configuration for the persistence layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistenceConfig {
    /// Path to the database file
    pub db_path: String,
}

impl PersistenceConfig {
    /// Creates a new persistence config with the given database path.
    #[must_use]
    pub fn new(db_path: impl Into<String>) -> Self {
        Self {
            db_path: db_path.into(),
        }
    }

    /// Creates a config for an in-memory database (for testing).
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            db_path: ":memory:".to_string(),
        }
    }

    /// Creates a config for a specific agent's database.
    #[must_use]
    pub fn for_agent(agent_id: &AgentId, base_path: &Path) -> Self {
        let db_path = base_path
            .join(format!("{}.db", agent_id))
            .to_string_lossy()
            .to_string();
        Self { db_path }
    }

    /// Returns true if this is an in-memory database.
    #[must_use]
    pub fn is_in_memory(&self) -> bool {
        self.db_path == ":memory:"
    }
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self::new("acton-ai.db")
    }
}

/// Agent state snapshot for persistence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentStateSnapshot {
    /// The agent ID
    pub agent_id: AgentId,
    /// Current conversation ID (if any)
    pub conversation_id: Option<ConversationId>,
    /// Current conversation messages
    pub conversation: Vec<Message>,
    /// The agent's system prompt
    pub system_prompt: String,
}

/// Opens a database connection.
///
/// # Arguments
///
/// * `config` - The persistence configuration
///
/// # Returns
///
/// A database handle on success.
///
/// # Errors
///
/// Returns an error if the database cannot be opened.
pub async fn open_database(config: &PersistenceConfig) -> Result<Database, PersistenceError> {
    let builder = if config.is_in_memory() {
        libsql::Builder::new_local(":memory:")
    } else {
        libsql::Builder::new_local(&config.db_path)
    };

    let db = builder
        .build()
        .await
        .map_err(|e| PersistenceError::database_open(&config.db_path, e.to_string()))?;

    Ok(db)
}

/// Initializes the database schema.
///
/// # Arguments
///
/// * `conn` - The database connection
///
/// # Errors
///
/// Returns an error if the schema cannot be created.
pub async fn initialize_schema(conn: &Connection) -> Result<(), PersistenceError> {
    conn.execute_batch(CREATE_SCHEMA)
        .await
        .map_err(|e| PersistenceError::schema_init(e.to_string()))?;

    // Set schema version
    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version) VALUES (?1)",
        [SCHEMA_VERSION],
    )
    .await
    .map_err(|e| PersistenceError::schema_init(e.to_string()))?;

    Ok(())
}

/// Creates a new conversation record.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `agent_id` - The agent owning the conversation
///
/// # Returns
///
/// The ID of the newly created conversation.
///
/// # Errors
///
/// Returns an error if the insert fails.
pub async fn create_conversation(
    conn: &Connection,
    agent_id: &AgentId,
) -> Result<ConversationId, PersistenceError> {
    let conv_id = ConversationId::new();

    conn.execute(
        "INSERT INTO conversations (id, agent_id) VALUES (?1, ?2)",
        [conv_id.to_string(), agent_id.to_string()],
    )
    .await
    .map_err(|e| PersistenceError::query_failed("create_conversation", e.to_string()))?;

    Ok(conv_id)
}

/// Saves a message to a conversation.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `conversation_id` - The conversation to add the message to
/// * `message` - The message to save
///
/// # Returns
///
/// The ID of the newly saved message.
///
/// # Errors
///
/// Returns an error if the insert fails or serialization fails.
pub async fn save_message(
    conn: &Connection,
    conversation_id: &ConversationId,
    message: &Message,
) -> Result<MessageId, PersistenceError> {
    let msg_id = MessageId::new();

    let tool_calls = message
        .tool_calls
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|e| PersistenceError::serialization_failed(e.to_string()))?;

    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, tool_calls, tool_call_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        libsql::params![
            msg_id.to_string(),
            conversation_id.to_string(),
            message.role.to_string(),
            message.content.clone(),
            tool_calls,
            message.tool_call_id.clone(),
        ],
    )
    .await
    .map_err(|e| PersistenceError::query_failed("save_message", e.to_string()))?;

    Ok(msg_id)
}

/// Loads all messages for a conversation.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `conversation_id` - The conversation to load
///
/// # Returns
///
/// A vector of messages in chronological order.
///
/// # Errors
///
/// Returns an error if the query fails or deserialization fails.
pub async fn load_conversation_messages(
    conn: &Connection,
    conversation_id: &ConversationId,
) -> Result<Vec<Message>, PersistenceError> {
    let mut rows = conn
        .query(
            "SELECT role, content, tool_calls, tool_call_id FROM messages
             WHERE conversation_id = ?1 ORDER BY created_at ASC",
            [conversation_id.to_string()],
        )
        .await
        .map_err(|e| PersistenceError::query_failed("load_conversation_messages", e.to_string()))?;

    let mut messages = Vec::new();

    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| PersistenceError::query_failed("load_conversation_messages", e.to_string()))?
    {
        let role_str: String = row
            .get(0)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let content: String = row
            .get(1)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let tool_calls_json: Option<String> = row
            .get(2)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let tool_call_id: Option<String> = row
            .get(3)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;

        let role = parse_message_role(&role_str)?;
        let tool_calls = tool_calls_json
            .map(|json| serde_json::from_str(&json))
            .transpose()
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;

        messages.push(Message {
            role,
            content,
            tool_calls,
            tool_call_id,
        });
    }

    Ok(messages)
}

/// Gets the most recent conversation for an agent.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `agent_id` - The agent to query
///
/// # Returns
///
/// The conversation ID if one exists, None otherwise.
///
/// # Errors
///
/// Returns an error if the query fails.
pub async fn get_latest_conversation(
    conn: &Connection,
    agent_id: &AgentId,
) -> Result<Option<ConversationId>, PersistenceError> {
    let mut rows = conn
        .query(
            "SELECT id FROM conversations WHERE agent_id = ?1 ORDER BY created_at DESC LIMIT 1",
            [agent_id.to_string()],
        )
        .await
        .map_err(|e| PersistenceError::query_failed("get_latest_conversation", e.to_string()))?;

    if let Some(row) = rows
        .next()
        .await
        .map_err(|e| PersistenceError::query_failed("get_latest_conversation", e.to_string()))?
    {
        let id_str: String = row
            .get(0)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let conv_id = ConversationId::parse(&id_str)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        Ok(Some(conv_id))
    } else {
        Ok(None)
    }
}

/// Saves agent state snapshot.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `snapshot` - The agent state snapshot to save
///
/// # Errors
///
/// Returns an error if the insert fails or serialization fails.
pub async fn save_agent_state(
    conn: &Connection,
    snapshot: &AgentStateSnapshot,
) -> Result<(), PersistenceError> {
    let state_json = serde_json::to_string(snapshot)
        .map_err(|e| PersistenceError::serialization_failed(e.to_string()))?;

    conn.execute(
        "INSERT OR REPLACE INTO agent_state (agent_id, state, updated_at)
         VALUES (?1, ?2, datetime('now'))",
        [snapshot.agent_id.to_string(), state_json],
    )
    .await
    .map_err(|e| PersistenceError::query_failed("save_agent_state", e.to_string()))?;

    Ok(())
}

/// Loads agent state snapshot.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `agent_id` - The agent to load state for
///
/// # Returns
///
/// The agent state snapshot if one exists, None otherwise.
///
/// # Errors
///
/// Returns an error if the query fails or deserialization fails.
pub async fn load_agent_state(
    conn: &Connection,
    agent_id: &AgentId,
) -> Result<Option<AgentStateSnapshot>, PersistenceError> {
    let mut rows = conn
        .query(
            "SELECT state FROM agent_state WHERE agent_id = ?1",
            [agent_id.to_string()],
        )
        .await
        .map_err(|e| PersistenceError::query_failed("load_agent_state", e.to_string()))?;

    if let Some(row) = rows
        .next()
        .await
        .map_err(|e| PersistenceError::query_failed("load_agent_state", e.to_string()))?
    {
        let state_json: String = row
            .get(0)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let snapshot: AgentStateSnapshot = serde_json::from_str(&state_json)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        Ok(Some(snapshot))
    } else {
        Ok(None)
    }
}

/// Deletes a conversation and all its messages.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `conversation_id` - The conversation to delete
///
/// # Errors
///
/// Returns an error if the delete fails.
pub async fn delete_conversation(
    conn: &Connection,
    conversation_id: &ConversationId,
) -> Result<(), PersistenceError> {
    // Messages will be deleted via CASCADE
    conn.execute(
        "DELETE FROM conversations WHERE id = ?1",
        [conversation_id.to_string()],
    )
    .await
    .map_err(|e| PersistenceError::query_failed("delete_conversation", e.to_string()))?;

    Ok(())
}

/// Deletes agent state.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `agent_id` - The agent whose state to delete
///
/// # Errors
///
/// Returns an error if the delete fails.
pub async fn delete_agent_state(
    conn: &Connection,
    agent_id: &AgentId,
) -> Result<(), PersistenceError> {
    conn.execute(
        "DELETE FROM agent_state WHERE agent_id = ?1",
        [agent_id.to_string()],
    )
    .await
    .map_err(|e| PersistenceError::query_failed("delete_agent_state", e.to_string()))?;

    Ok(())
}

/// Lists all conversations for an agent.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `agent_id` - The agent to query
///
/// # Returns
///
/// A vector of conversation IDs in reverse chronological order.
///
/// # Errors
///
/// Returns an error if the query fails.
pub async fn list_conversations(
    conn: &Connection,
    agent_id: &AgentId,
) -> Result<Vec<ConversationId>, PersistenceError> {
    let mut rows = conn
        .query(
            "SELECT id FROM conversations WHERE agent_id = ?1 ORDER BY created_at DESC",
            [agent_id.to_string()],
        )
        .await
        .map_err(|e| PersistenceError::query_failed("list_conversations", e.to_string()))?;

    let mut conversations = Vec::new();

    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| PersistenceError::query_failed("list_conversations", e.to_string()))?
    {
        let id_str: String = row
            .get(0)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let conv_id = ConversationId::parse(&id_str)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        conversations.push(conv_id);
    }

    Ok(conversations)
}

/// Helper function to parse MessageRole from string.
fn parse_message_role(s: &str) -> Result<MessageRole, PersistenceError> {
    match s {
        "system" => Ok(MessageRole::System),
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "tool" => Ok(MessageRole::Tool),
        _ => Err(PersistenceError::deserialization_failed(format!(
            "unknown message role: {}",
            s
        ))),
    }
}

// =============================================================================
// Memory Functions
// =============================================================================

/// Saves a memory with optional embedding.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `memory` - The memory to save
///
/// # Returns
///
/// The ID of the saved memory.
///
/// # Errors
///
/// Returns an error if the insert fails.
pub async fn save_memory(conn: &Connection, memory: &Memory) -> Result<MemoryId, PersistenceError> {
    let embedding_bytes = memory.embedding.as_ref().map(Embedding::to_bytes);

    conn.execute(
        "INSERT INTO memories (id, agent_id, content, embedding, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        libsql::params![
            memory.id.to_string(),
            memory.agent_id.to_string(),
            memory.content.clone(),
            embedding_bytes,
            memory.created_at.clone(),
        ],
    )
    .await
    .map_err(|e| PersistenceError::query_failed("save_memory", e.to_string()))?;

    Ok(memory.id.clone())
}

/// Searches memories by embedding similarity.
///
/// Uses in-application cosine similarity calculation since libSQL
/// vector extension requires specific setup. Loads all memories with
/// embeddings for the agent and computes similarity in Rust.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `agent_id` - The agent to search within
/// * `query_embedding` - The query embedding
/// * `limit` - Maximum results to return
/// * `min_similarity` - Optional minimum similarity threshold
///
/// # Returns
///
/// Memories ranked by similarity score (highest first).
///
/// # Errors
///
/// Returns an error if the query fails.
pub async fn search_memories_by_embedding(
    conn: &Connection,
    agent_id: &AgentId,
    query_embedding: &Embedding,
    limit: usize,
    min_similarity: Option<f32>,
) -> Result<Vec<ScoredMemory>, PersistenceError> {
    // Load all memories with embeddings for this agent
    let mut rows = conn
        .query(
            "SELECT id, content, embedding, created_at FROM memories
             WHERE agent_id = ?1 AND embedding IS NOT NULL
             ORDER BY created_at DESC",
            [agent_id.to_string()],
        )
        .await
        .map_err(|e| PersistenceError::query_failed("search_memories", e.to_string()))?;

    let mut scored: Vec<ScoredMemory> = Vec::new();
    let threshold = min_similarity.unwrap_or(-1.0); // Default to include all

    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| PersistenceError::query_failed("search_memories", e.to_string()))?
    {
        let id_str: String = row
            .get(0)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let content: String = row
            .get(1)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let embedding_bytes: Vec<u8> = row
            .get(2)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let created_at: String = row
            .get(3)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;

        let memory_id = MemoryId::parse(&id_str)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;

        let embedding = Embedding::from_bytes(&embedding_bytes)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;

        let similarity = query_embedding
            .cosine_similarity(&embedding)
            .map_err(|e| PersistenceError::vector_search_failed(e.to_string()))?;

        if similarity >= threshold {
            scored.push(ScoredMemory {
                memory: Memory {
                    id: memory_id,
                    agent_id: agent_id.clone(),
                    content,
                    embedding: Some(embedding),
                    created_at,
                },
                score: similarity,
            });
        }
    }

    // Sort by score descending
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Limit results
    scored.truncate(limit);

    Ok(scored)
}

/// Loads all memories for an agent.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `agent_id` - The agent to load memories for
/// * `limit` - Optional limit on results
///
/// # Returns
///
/// List of memories in reverse chronological order.
///
/// # Errors
///
/// Returns an error if the query fails.
pub async fn load_memories_for_agent(
    conn: &Connection,
    agent_id: &AgentId,
    limit: Option<usize>,
) -> Result<Vec<Memory>, PersistenceError> {
    let query = match limit {
        Some(l) => format!(
            "SELECT id, content, embedding, created_at FROM memories
             WHERE agent_id = ?1 ORDER BY created_at DESC LIMIT {}",
            l
        ),
        None => "SELECT id, content, embedding, created_at FROM memories
             WHERE agent_id = ?1 ORDER BY created_at DESC"
            .to_string(),
    };

    let mut rows = conn
        .query(&query, [agent_id.to_string()])
        .await
        .map_err(|e| PersistenceError::query_failed("load_memories", e.to_string()))?;

    let mut memories = Vec::new();

    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| PersistenceError::query_failed("load_memories", e.to_string()))?
    {
        let id_str: String = row
            .get(0)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let content: String = row
            .get(1)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let embedding_bytes: Option<Vec<u8>> = row
            .get(2)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        let created_at: String = row
            .get(3)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;

        let memory_id = MemoryId::parse(&id_str)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;

        let embedding = match embedding_bytes {
            Some(bytes) if !bytes.is_empty() => Some(
                Embedding::from_bytes(&bytes)
                    .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?,
            ),
            _ => None,
        };

        memories.push(Memory {
            id: memory_id,
            agent_id: agent_id.clone(),
            content,
            embedding,
            created_at,
        });
    }

    Ok(memories)
}

/// Deletes a memory by ID.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `memory_id` - The memory to delete
///
/// # Errors
///
/// Returns an error if the delete fails.
pub async fn delete_memory(
    conn: &Connection,
    memory_id: &MemoryId,
) -> Result<(), PersistenceError> {
    conn.execute(
        "DELETE FROM memories WHERE id = ?1",
        [memory_id.to_string()],
    )
    .await
    .map_err(|e| PersistenceError::query_failed("delete_memory", e.to_string()))?;

    Ok(())
}

/// Deletes all memories for an agent.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `agent_id` - The agent whose memories to delete
///
/// # Errors
///
/// Returns an error if the delete fails.
pub async fn delete_memories_for_agent(
    conn: &Connection,
    agent_id: &AgentId,
) -> Result<(), PersistenceError> {
    conn.execute(
        "DELETE FROM memories WHERE agent_id = ?1",
        [agent_id.to_string()],
    )
    .await
    .map_err(|e| PersistenceError::query_failed("delete_memories_for_agent", e.to_string()))?;

    Ok(())
}

/// Counts memories for an agent.
///
/// # Arguments
///
/// * `conn` - The database connection
/// * `agent_id` - The agent to count memories for
///
/// # Returns
///
/// The number of memories for this agent.
///
/// # Errors
///
/// Returns an error if the query fails.
pub async fn count_memories_for_agent(
    conn: &Connection,
    agent_id: &AgentId,
) -> Result<usize, PersistenceError> {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM memories WHERE agent_id = ?1",
            [agent_id.to_string()],
        )
        .await
        .map_err(|e| PersistenceError::query_failed("count_memories", e.to_string()))?;

    if let Some(row) = rows
        .next()
        .await
        .map_err(|e| PersistenceError::query_failed("count_memories", e.to_string()))?
    {
        let count: i64 = row
            .get(0)
            .map_err(|e| PersistenceError::deserialization_failed(e.to_string()))?;
        Ok(count as usize)
    } else {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_config_new() {
        let config = PersistenceConfig::new("test.db");
        assert_eq!(config.db_path, "test.db");
        assert!(!config.is_in_memory());
    }

    #[test]
    fn persistence_config_in_memory() {
        let config = PersistenceConfig::in_memory();
        assert_eq!(config.db_path, ":memory:");
        assert!(config.is_in_memory());
    }

    #[test]
    fn persistence_config_for_agent() {
        let agent_id = AgentId::new();
        let config = PersistenceConfig::for_agent(&agent_id, Path::new("/data"));
        assert!(config.db_path.contains(&agent_id.to_string()));
        assert!(config.db_path.ends_with(".db"));
    }

    #[test]
    fn persistence_config_default() {
        let config = PersistenceConfig::default();
        assert_eq!(config.db_path, "acton-ai.db");
    }

    #[test]
    fn parse_message_role_valid() {
        assert_eq!(parse_message_role("system").unwrap(), MessageRole::System);
        assert_eq!(parse_message_role("user").unwrap(), MessageRole::User);
        assert_eq!(
            parse_message_role("assistant").unwrap(),
            MessageRole::Assistant
        );
        assert_eq!(parse_message_role("tool").unwrap(), MessageRole::Tool);
    }

    #[test]
    fn parse_message_role_invalid() {
        let result = parse_message_role("unknown");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unknown"));
    }

    #[test]
    fn agent_state_snapshot_serialization() {
        let snapshot = AgentStateSnapshot {
            agent_id: AgentId::new(),
            conversation_id: Some(ConversationId::new()),
            conversation: vec![Message::user("Hello")],
            system_prompt: "You are helpful".to_string(),
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let deserialized: AgentStateSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(snapshot.agent_id, deserialized.agent_id);
        assert_eq!(snapshot.system_prompt, deserialized.system_prompt);
        assert_eq!(snapshot.conversation.len(), deserialized.conversation.len());
    }
}
