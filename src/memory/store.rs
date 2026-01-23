//! Memory Store actor for persistence operations.
//!
//! The `MemoryStore` actor manages all database operations asynchronously,
//! spawning tokio tasks for database operations to avoid Sync constraints.

use crate::memory::error::PersistenceError;
use crate::memory::persistence::{self, AgentStateSnapshot, PersistenceConfig};
use crate::messages::Message;
use crate::types::{AgentId, ConversationId, MessageId};
use acton_reactive::prelude::*;
use libsql::{Connection, Database};
use std::sync::Arc;
use tokio::sync::Mutex;

// =============================================================================
// Messages
// =============================================================================

/// Message to initialize the Memory Store with configuration.
#[acton_message]
pub struct InitMemoryStore {
    /// The persistence configuration
    pub config: PersistenceConfig,
}

/// Request to create a new conversation.
#[acton_message]
pub struct CreateConversation {
    /// The agent creating the conversation
    pub agent_id: AgentId,
}

/// Response with newly created conversation ID.
#[acton_message]
pub struct ConversationCreated {
    /// The ID of the new conversation
    pub conversation_id: ConversationId,
}

/// Request to save a message.
#[acton_message]
pub struct SaveMessage {
    /// The conversation to add the message to
    pub conversation_id: ConversationId,
    /// The message to save
    pub message: Message,
}

/// Response with saved message ID.
#[acton_message]
pub struct MessageSaved {
    /// The ID of the saved message
    pub message_id: MessageId,
}

/// Request to load conversation messages.
#[acton_message]
pub struct LoadConversation {
    /// The conversation to load
    pub conversation_id: ConversationId,
}

/// Response with loaded messages.
#[acton_message]
pub struct ConversationLoaded {
    /// The conversation ID
    pub conversation_id: ConversationId,
    /// The messages in the conversation
    pub messages: Vec<Message>,
}

/// Request to get an agent's latest conversation.
#[acton_message]
pub struct GetLatestConversation {
    /// The agent to query
    pub agent_id: AgentId,
}

/// Response with optional conversation ID.
#[acton_message]
pub struct LatestConversationResponse {
    /// The latest conversation ID if one exists
    pub conversation_id: Option<ConversationId>,
}

/// Request to save agent state snapshot.
#[acton_message]
pub struct SaveAgentState {
    /// The state snapshot to save
    pub snapshot: AgentStateSnapshot,
}

/// Request to load agent state snapshot.
#[acton_message]
pub struct LoadAgentState {
    /// The agent to load state for
    pub agent_id: AgentId,
}

/// Response with optional agent state.
#[acton_message]
pub struct AgentStateLoaded {
    /// The loaded state snapshot if one exists
    pub snapshot: Option<AgentStateSnapshot>,
}

/// Request to delete a conversation.
#[acton_message]
pub struct DeleteConversation {
    /// The conversation to delete
    pub conversation_id: ConversationId,
}

/// Request to list all conversations for an agent.
#[acton_message]
pub struct ListConversations {
    /// The agent to query
    pub agent_id: AgentId,
}

/// Response with conversation list.
#[acton_message]
pub struct ConversationList {
    /// The list of conversation IDs
    pub conversations: Vec<ConversationId>,
}

// =============================================================================
// Metrics
// =============================================================================

/// Metrics for the Memory Store.
#[derive(Debug, Clone, Default)]
pub struct MemoryStoreMetrics {
    /// Number of conversations created
    pub conversations_created: u64,
    /// Number of messages saved
    pub messages_saved: u64,
    /// Number of conversations loaded
    pub conversations_loaded: u64,
    /// Number of state saves
    pub state_saves: u64,
    /// Number of state loads
    pub state_loads: u64,
}

// =============================================================================
// Shared Connection
// =============================================================================

/// Thread-safe shared connection wrapper.
#[derive(Clone, Default)]
pub struct SharedConnection(Arc<Mutex<Option<Connection>>>);

impl std::fmt::Debug for SharedConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedConnection")
            .field("initialized", &"<check async>")
            .finish()
    }
}

impl SharedConnection {
    /// Creates a new shared connection wrapper.
    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    /// Sets the connection.
    pub async fn set(&self, conn: Connection) {
        let mut guard = self.0.lock().await;
        *guard = Some(conn);
    }

    /// Gets a clone of the connection if initialized.
    pub async fn get(&self) -> Option<Connection> {
        let guard = self.0.lock().await;
        guard.clone()
    }

    /// Returns true if the connection has been initialized.
    pub async fn is_initialized(&self) -> bool {
        let guard = self.0.lock().await;
        guard.is_some()
    }
}

// =============================================================================
// Actor
// =============================================================================

/// The Memory Store actor state.
#[acton_actor]
pub struct MemoryStore {
    /// Configuration for persistence
    pub config: Option<PersistenceConfig>,
    /// Database handle (initialized after start)
    pub database: Option<Database>,
    /// Shared connection for async operations
    pub shared_connection: SharedConnection,
    /// Whether the store is shutting down
    pub shutting_down: bool,
    /// Metrics
    pub metrics: MemoryStoreMetrics,
}

impl MemoryStore {
    /// Spawns the Memory Store actor.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The actor runtime
    ///
    /// # Returns
    ///
    /// A handle to the spawned Memory Store actor.
    pub async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder =
            runtime.new_actor_with_name::<MemoryStore>("memory_store".to_string());

        // Set up lifecycle hooks
        builder
            .before_start(|_actor| {
                tracing::debug!("Memory Store initializing");
                Reply::ready()
            })
            .after_start(|actor| {
                tracing::info!(config = ?actor.model.config, "Memory Store ready");
                Reply::ready()
            })
            .before_stop(|actor| {
                tracing::info!(
                    conversations_created = actor.model.metrics.conversations_created,
                    messages_saved = actor.model.metrics.messages_saved,
                    "Memory Store shutting down"
                );
                Reply::ready()
            });

        // Configure message handlers
        configure_handlers(&mut builder);

        builder.start().await
    }
}

/// Configures message handlers for the Memory Store actor.
fn configure_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    configure_init_handler(builder);
    configure_conversation_handlers(builder);
    configure_message_handlers(builder);
    configure_state_handlers(builder);
}

/// Configures the initialization handler.
fn configure_init_handler(builder: &mut ManagedActor<Idle, MemoryStore>) {
    builder.mutate_on::<InitMemoryStore>(|actor, envelope| {
        let config = envelope.message().config.clone();
        let shared_conn = actor.model.shared_connection.clone();
        actor.model.config = Some(config.clone());

        // Spawn a tokio task to do the async database work
        // The JoinHandle is Send + Sync, so we can use it in Reply::pending
        let handle = tokio::spawn(async move {
            match initialize_database(&config).await {
                Ok((_db, conn)) => {
                    shared_conn.set(conn).await;
                    tracing::info!(db_path = %config.db_path, "Memory Store initialized with database");
                }
                Err(e) => {
                    tracing::error!(error = %e, "Memory Store initialization failed");
                }
            }
        });

        Reply::pending(async move {
            let _ = handle.await;
        })
    });
}

/// Helper function to initialize database outside of actor context.
async fn initialize_database(
    config: &PersistenceConfig,
) -> Result<(Database, Connection), PersistenceError> {
    let db = persistence::open_database(config).await?;
    let conn = db
        .connect()
        .map_err(|e| PersistenceError::connection_error(e.to_string()))?;
    persistence::initialize_schema(&conn).await?;
    Ok((db, conn))
}

/// Configures conversation-related handlers.
fn configure_conversation_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    // Handle conversation creation
    builder.mutate_on::<CreateConversation>(|actor, envelope| {
        if actor.model.shutting_down {
            tracing::warn!("Rejecting CreateConversation - store is shutting down");
            return Reply::ready();
        }

        let shared_conn = actor.model.shared_connection.clone();
        let agent_id = envelope.message().agent_id.clone();
        let reply = envelope.reply_envelope();
        actor.model.metrics.conversations_created += 1;

        let handle = tokio::spawn(async move {
            let Some(conn) = shared_conn.get().await else {
                tracing::error!("Memory Store not initialized");
                return;
            };

            match persistence::create_conversation(&conn, &agent_id).await {
                Ok(conversation_id) => {
                    reply.send(ConversationCreated { conversation_id }).await;
                }
                Err(e) => {
                    tracing::error!(agent_id = %agent_id, error = %e, "Failed to create conversation");
                }
            }
        });

        Reply::pending(async move {
            let _ = handle.await;
        })
    });

    // Handle conversation loading
    builder.mutate_on::<LoadConversation>(|actor, envelope| {
        if actor.model.shutting_down {
            tracing::warn!("Rejecting LoadConversation - store is shutting down");
            return Reply::ready();
        }

        let shared_conn = actor.model.shared_connection.clone();
        let conversation_id = envelope.message().conversation_id.clone();
        let reply = envelope.reply_envelope();
        actor.model.metrics.conversations_loaded += 1;

        let handle = tokio::spawn(async move {
            let Some(conn) = shared_conn.get().await else {
                tracing::error!("Memory Store not initialized");
                return;
            };

            match persistence::load_conversation_messages(&conn, &conversation_id).await {
                Ok(messages) => {
                    reply
                        .send(ConversationLoaded {
                            conversation_id,
                            messages,
                        })
                        .await;
                }
                Err(e) => {
                    tracing::error!(conversation_id = %conversation_id, error = %e, "Failed to load conversation");
                }
            }
        });

        Reply::pending(async move {
            let _ = handle.await;
        })
    });

    // Handle get latest conversation
    builder.mutate_on::<GetLatestConversation>(|actor, envelope| {
        if actor.model.shutting_down {
            tracing::warn!("Rejecting GetLatestConversation - store is shutting down");
            return Reply::ready();
        }

        let shared_conn = actor.model.shared_connection.clone();
        let agent_id = envelope.message().agent_id.clone();
        let reply = envelope.reply_envelope();

        let handle = tokio::spawn(async move {
            let Some(conn) = shared_conn.get().await else {
                tracing::error!("Memory Store not initialized");
                return;
            };

            match persistence::get_latest_conversation(&conn, &agent_id).await {
                Ok(conversation_id) => {
                    reply
                        .send(LatestConversationResponse { conversation_id })
                        .await;
                }
                Err(e) => {
                    tracing::error!(agent_id = %agent_id, error = %e, "Failed to get latest conversation");
                }
            }
        });

        Reply::pending(async move {
            let _ = handle.await;
        })
    });

    // Handle conversation deletion
    builder.mutate_on::<DeleteConversation>(|actor, envelope| {
        if actor.model.shutting_down {
            tracing::warn!("Rejecting DeleteConversation - store is shutting down");
            return Reply::ready();
        }

        let shared_conn = actor.model.shared_connection.clone();
        let conversation_id = envelope.message().conversation_id.clone();

        let handle = tokio::spawn(async move {
            let Some(conn) = shared_conn.get().await else {
                tracing::error!("Memory Store not initialized");
                return;
            };

            if let Err(e) = persistence::delete_conversation(&conn, &conversation_id).await {
                tracing::error!(conversation_id = %conversation_id, error = %e, "Failed to delete conversation");
            }
        });

        Reply::pending(async move {
            let _ = handle.await;
        })
    });

    // Handle list conversations
    builder.mutate_on::<ListConversations>(|actor, envelope| {
        if actor.model.shutting_down {
            tracing::warn!("Rejecting ListConversations - store is shutting down");
            return Reply::ready();
        }

        let shared_conn = actor.model.shared_connection.clone();
        let agent_id = envelope.message().agent_id.clone();
        let reply = envelope.reply_envelope();

        let handle = tokio::spawn(async move {
            let Some(conn) = shared_conn.get().await else {
                tracing::error!("Memory Store not initialized");
                return;
            };

            match persistence::list_conversations(&conn, &agent_id).await {
                Ok(conversations) => {
                    reply.send(ConversationList { conversations }).await;
                }
                Err(e) => {
                    tracing::error!(agent_id = %agent_id, error = %e, "Failed to list conversations");
                }
            }
        });

        Reply::pending(async move {
            let _ = handle.await;
        })
    });
}

/// Configures message-related handlers.
fn configure_message_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    builder.mutate_on::<SaveMessage>(|actor, envelope| {
        if actor.model.shutting_down {
            tracing::warn!("Rejecting SaveMessage - store is shutting down");
            return Reply::ready();
        }

        let shared_conn = actor.model.shared_connection.clone();
        let msg = envelope.message();
        let conversation_id = msg.conversation_id.clone();
        let message = msg.message.clone();
        let reply = envelope.reply_envelope();
        actor.model.metrics.messages_saved += 1;

        let handle = tokio::spawn(async move {
            let Some(conn) = shared_conn.get().await else {
                tracing::error!("Memory Store not initialized");
                return;
            };

            match persistence::save_message(&conn, &conversation_id, &message).await {
                Ok(message_id) => {
                    reply.send(MessageSaved { message_id }).await;
                }
                Err(e) => {
                    tracing::error!(conversation_id = %conversation_id, error = %e, "Failed to save message");
                }
            }
        });

        Reply::pending(async move {
            let _ = handle.await;
        })
    });
}

/// Configures agent state handlers.
fn configure_state_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    // Handle save agent state
    builder.mutate_on::<SaveAgentState>(|actor, envelope| {
        if actor.model.shutting_down {
            tracing::warn!("Rejecting SaveAgentState - store is shutting down");
            return Reply::ready();
        }

        let shared_conn = actor.model.shared_connection.clone();
        let snapshot = envelope.message().snapshot.clone();
        actor.model.metrics.state_saves += 1;

        let handle = tokio::spawn(async move {
            let Some(conn) = shared_conn.get().await else {
                tracing::error!("Memory Store not initialized");
                return;
            };

            if let Err(e) = persistence::save_agent_state(&conn, &snapshot).await {
                tracing::error!(agent_id = %snapshot.agent_id, error = %e, "Failed to save agent state");
            }
        });

        Reply::pending(async move {
            let _ = handle.await;
        })
    });

    // Handle load agent state
    builder.mutate_on::<LoadAgentState>(|actor, envelope| {
        if actor.model.shutting_down {
            tracing::warn!("Rejecting LoadAgentState - store is shutting down");
            return Reply::ready();
        }

        let shared_conn = actor.model.shared_connection.clone();
        let agent_id = envelope.message().agent_id.clone();
        let reply = envelope.reply_envelope();
        actor.model.metrics.state_loads += 1;

        let handle = tokio::spawn(async move {
            let Some(conn) = shared_conn.get().await else {
                tracing::error!("Memory Store not initialized");
                return;
            };

            match persistence::load_agent_state(&conn, &agent_id).await {
                Ok(snapshot) => {
                    reply.send(AgentStateLoaded { snapshot }).await;
                }
                Err(e) => {
                    tracing::error!(agent_id = %agent_id, error = %e, "Failed to load agent state");
                }
            }
        });

        Reply::pending(async move {
            let _ = handle.await;
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_metrics_default() {
        let metrics = MemoryStoreMetrics::default();
        assert_eq!(metrics.conversations_created, 0);
        assert_eq!(metrics.messages_saved, 0);
        assert_eq!(metrics.conversations_loaded, 0);
        assert_eq!(metrics.state_saves, 0);
        assert_eq!(metrics.state_loads, 0);
    }

    #[tokio::test]
    async fn shared_connection_initialization() {
        let shared = SharedConnection::new();
        assert!(!shared.is_initialized().await);
    }
}
