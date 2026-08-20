//! Memory Store actor for persistence operations.
//!
//! The `MemoryStore` actor manages all database operations asynchronously,
//! spawning tokio tasks for database operations to avoid Sync constraints.
//!
//! # Replies are `ask`-oriented
//!
//! Every request in this module has a matching response type carrying a
//! `Result`, and every failure path replies. Use [`ActorHandle::ask`] to issue
//! them:
//!
//! ```rust,ignore
//! let loaded = store.ask(LoadMemories { agent_id, limit: None }).await?;
//! let memories = loaded.result?;
//! ```
//!
//! A plain `send` discards the reply. acton-reactive points a handler's reply
//! envelope back at the *receiving* actor unless the caller used `ask`, so a
//! reply to a `send` goes nowhere.

use crate::checkpoint::{CheckpointRecord, CheckpointStatus};
use crate::memory::context::{ContextStats, ContextWindow, ContextWindowConfig};
use crate::memory::embeddings::{Embedding, Memory, ScoredMemory};
use crate::memory::error::PersistenceError;
use crate::memory::persistence::{self, AgentStateSnapshot, PersistenceConfig};
use crate::messages::Message;
use crate::types::{AgentId, CheckpointId, ConversationId, MemoryId, MessageId};
use acton_reactive::prelude::*;
use libsql::{Connection, Database};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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

/// Response with the newly created conversation ID.
#[acton_message]
pub struct ConversationCreated {
    /// The new conversation ID, or why one could not be created
    pub result: Result<ConversationId, PersistenceError>,
}

impl Request for CreateConversation {
    type Response = ConversationCreated;
}

/// Request to save a message.
#[acton_message]
pub struct SaveMessage {
    /// The conversation to add the message to
    pub conversation_id: ConversationId,
    /// The message to save
    pub message: Message,
}

/// Response with the saved message ID.
#[acton_message]
pub struct MessageSaved {
    /// The ID of the saved message, or why it could not be saved
    pub result: Result<MessageId, PersistenceError>,
}

impl Request for SaveMessage {
    type Response = MessageSaved;
}

/// Request to load conversation messages.
#[acton_message]
pub struct LoadConversation {
    /// The conversation to load
    pub conversation_id: ConversationId,
}

/// Response with the loaded messages.
#[acton_message]
pub struct ConversationLoaded {
    /// The conversation that was requested
    pub conversation_id: ConversationId,
    /// The messages in the conversation, or why they could not be loaded
    pub result: Result<Vec<Message>, PersistenceError>,
}

impl Request for LoadConversation {
    type Response = ConversationLoaded;
}

/// Request to get an agent's latest conversation.
#[acton_message]
pub struct GetLatestConversation {
    /// The agent to query
    pub agent_id: AgentId,
}

/// Response with the agent's most recent conversation.
#[acton_message]
pub struct LatestConversationResponse {
    /// The latest conversation ID, `Ok(None)` if the agent has none, or why
    /// the lookup failed
    pub result: Result<Option<ConversationId>, PersistenceError>,
}

impl Request for GetLatestConversation {
    type Response = LatestConversationResponse;
}

/// Request to save an agent state snapshot.
#[acton_message]
pub struct SaveAgentState {
    /// The state snapshot to save
    pub snapshot: AgentStateSnapshot,
}

impl Request for SaveAgentState {
    type Response = OperationCompleted;
}

/// Request to load an agent state snapshot.
#[acton_message]
pub struct LoadAgentState {
    /// The agent to load state for
    pub agent_id: AgentId,
}

/// Response with the agent's stored state.
#[acton_message]
pub struct AgentStateLoaded {
    /// The stored snapshot, `Ok(None)` if the agent has none, or why the load
    /// failed
    pub result: Result<Option<AgentStateSnapshot>, PersistenceError>,
}

impl Request for LoadAgentState {
    type Response = AgentStateLoaded;
}

/// Request to delete a conversation.
#[acton_message]
pub struct DeleteConversation {
    /// The conversation to delete
    pub conversation_id: ConversationId,
}

impl Request for DeleteConversation {
    type Response = OperationCompleted;
}

/// Request to list all conversations for an agent.
#[acton_message]
pub struct ListConversations {
    /// The agent to query
    pub agent_id: AgentId,
}

/// Response with the agent's conversations.
#[acton_message]
pub struct ConversationList {
    /// The conversation IDs, most recent first, or why they could not be listed
    pub result: Result<Vec<ConversationId>, PersistenceError>,
}

impl Request for ListConversations {
    type Response = ConversationList;
}

/// Response for a request whose only outcome is success or failure.
///
/// Used by the delete and state-save requests, which have nothing to return
/// but still need to report failure rather than fail silently.
#[acton_message]
pub struct OperationCompleted {
    /// The operation that was attempted, for logging and correlation
    pub operation: &'static str,
    /// Whether it succeeded, and why not if it did not
    pub result: Result<(), PersistenceError>,
}

// -----------------------------------------------------------------------------
// Memory Messages
// -----------------------------------------------------------------------------

/// Request to store a memory with an optional embedding.
#[acton_message]
pub struct StoreMemory {
    /// The agent this memory belongs to
    pub agent_id: AgentId,
    /// The content to store
    pub content: String,
    /// Optional pre-computed embedding for semantic search
    pub embedding: Option<Embedding>,
}

/// Response with the stored memory ID.
#[acton_message]
pub struct MemoryStored {
    /// The ID of the stored memory, or why it could not be stored
    pub result: Result<MemoryId, PersistenceError>,
}

impl Request for StoreMemory {
    type Response = MemoryStored;
}

/// Request to search memories by semantic similarity.
#[acton_message]
pub struct SearchMemories {
    /// The agent to search within
    pub agent_id: AgentId,
    /// The query embedding to match against
    pub query_embedding: Embedding,
    /// Maximum number of results
    pub limit: usize,
    /// Minimum similarity threshold (0.0 to 1.0)
    pub min_similarity: Option<f32>,
}

/// Response with ranked memory results.
#[acton_message]
pub struct MemorySearchResults {
    /// Memories ranked by similarity (highest first), or why the search failed
    pub result: Result<Vec<ScoredMemory>, PersistenceError>,
}

impl Request for SearchMemories {
    type Response = MemorySearchResults;
}

/// Request to load memories for an agent.
#[acton_message]
pub struct LoadMemories {
    /// The agent to load memories for
    pub agent_id: AgentId,
    /// Optional limit on results
    pub limit: Option<usize>,
}

/// Response with the loaded memories.
#[acton_message]
pub struct MemoriesLoaded {
    /// The loaded memories, or why they could not be loaded
    pub result: Result<Vec<Memory>, PersistenceError>,
}

impl Request for LoadMemories {
    type Response = MemoriesLoaded;
}

/// Request to delete a memory.
#[acton_message]
pub struct DeleteMemory {
    /// The memory to delete
    pub memory_id: MemoryId,
}

impl Request for DeleteMemory {
    type Response = OperationCompleted;
}

/// Request to delete all memories for an agent.
#[acton_message]
pub struct DeleteAgentMemories {
    /// The agent whose memories to delete
    pub agent_id: AgentId,
}

impl Request for DeleteAgentMemories {
    type Response = OperationCompleted;
}

/// Internal message to set the database connection after async initialization.
#[acton_message]
struct SetConnection {
    /// The initialized database connection
    conn: Connection,
}

/// Request an optimized context window.
#[acton_message]
pub struct GetContextWindow {
    /// The agent requesting context
    pub agent_id: AgentId,
    /// The agent's system prompt
    pub system_prompt: String,
    /// The current conversation messages
    pub conversation: Vec<Message>,
    /// Query embedding for memory retrieval (optional)
    pub query_embedding: Option<Embedding>,
    /// Maximum tokens for context
    pub max_tokens: usize,
    /// Number of memories to retrieve
    pub memory_limit: usize,
}

/// A built context window.
#[derive(Debug, Clone)]
pub struct ContextWindowData {
    /// Optimized messages for LLM context
    pub messages: Vec<Message>,
    /// Statistics about the context window
    pub stats: ContextStats,
    /// Number of memories included in the context
    pub included_memories: usize,
}

/// Response with the optimized context.
#[acton_message]
pub struct ContextWindowResponse {
    /// The built context window, or why it could not be built
    pub result: Result<ContextWindowData, PersistenceError>,
}

impl Request for GetContextWindow {
    type Response = ContextWindowResponse;
}

// -----------------------------------------------------------------------------
// Checkpoint Messages
// -----------------------------------------------------------------------------

/// Request to save a turn checkpoint, replacing any earlier one under the same
/// ID.
///
/// Handled on a mutable handler so writes for one turn land in the order they
/// were issued: the later checkpoint is the one that must survive, and
/// overlapping writes could otherwise leave the earlier progress in place.
#[acton_message]
pub struct SaveCheckpoint {
    /// The record to persist.
    pub record: CheckpointRecord,
}

/// Reply to [`SaveCheckpoint`].
#[acton_message]
pub struct CheckpointSaved {
    /// Whether the write landed, and why not if it did not.
    pub result: Result<(), PersistenceError>,
}

impl Request for SaveCheckpoint {
    type Response = CheckpointSaved;
}

/// Request to load a turn checkpoint by ID.
#[acton_message]
pub struct LoadCheckpoint {
    /// The checkpoint to look up.
    pub id: CheckpointId,
}

/// Reply to [`LoadCheckpoint`].
#[acton_message]
pub struct CheckpointLoaded {
    /// The record, `None` when no checkpoint is stored under that ID, or why
    /// the lookup failed.
    pub result: Result<Option<CheckpointRecord>, PersistenceError>,
}

impl Request for LoadCheckpoint {
    type Response = CheckpointLoaded;
}

/// Request to list stored checkpoints, newest first.
#[acton_message]
pub struct ListCheckpoints {
    /// Narrow the listing to one status, or `None` for every checkpoint.
    pub status: Option<CheckpointStatus>,
}

/// Reply to [`ListCheckpoints`].
#[acton_message]
pub struct CheckpointList {
    /// The matching records, or why they could not be listed.
    pub result: Result<Vec<CheckpointRecord>, PersistenceError>,
}

impl Request for ListCheckpoints {
    type Response = CheckpointList;
}

/// Request to delete a turn checkpoint.
///
/// Deleting one that is not there succeeds: the caller wanted it gone, and it
/// is gone.
#[acton_message]
pub struct DeleteCheckpoint {
    /// The checkpoint to remove.
    pub id: CheckpointId,
}

impl Request for DeleteCheckpoint {
    type Response = OperationCompleted;
}

/// Request to claim a checkpoint ID for the turn about to run under it.
///
/// The claim registry is in-process, in-memory state owned by this actor: a
/// checkpoint ID has exactly one live owner, and the prompt loop claims its ID
/// before planning a resume so two loops in the same process — a live turn and
/// an operator's `resume_turn`, or a caller's retry racing the `resume_auto`
/// background task — can never both settle the same pending tool calls. The
/// registry dies with the process, deliberately: a crashed process's claims
/// must not outlive it, or nothing could ever be resumed after a crash.
#[acton_message]
pub struct ClaimCheckpoint {
    /// The checkpoint to claim.
    pub id: CheckpointId,
}

/// Reply to [`ClaimCheckpoint`].
#[acton_message]
pub struct CheckpointClaimed {
    /// Whether the claim was granted. `false` means another turn in this
    /// process holds the ID right now.
    pub granted: bool,
}

impl Request for ClaimCheckpoint {
    type Response = CheckpointClaimed;
}

/// Request to release a claim taken with [`ClaimCheckpoint`].
///
/// Releasing an ID nobody holds succeeds: the caller wanted it free, and it
/// is free.
#[acton_message]
pub struct ReleaseCheckpoint {
    /// The checkpoint to release.
    pub id: CheckpointId,
}

/// Reply to [`ReleaseCheckpoint`]. Answered so a releasing turn can use the
/// reply as a barrier: once it lands, the next claim on the ID will succeed.
#[acton_message]
pub struct CheckpointReleased;

impl Request for ReleaseCheckpoint {
    type Response = CheckpointReleased;
}

/// Request for the checkpoint IDs currently claimed by running turns.
#[acton_message]
pub struct ListCheckpointClaims;

/// Reply to [`ListCheckpointClaims`].
#[acton_message]
pub struct CheckpointClaims {
    /// Every ID a turn in this process holds right now.
    pub ids: Vec<CheckpointId>,
}

impl Request for ListCheckpointClaims {
    type Response = CheckpointClaims;
}

// =============================================================================
// Metrics
// =============================================================================

/// Counters behind [`MemoryStoreMetrics`].
///
/// Held behind an `Arc` so read-only (`act_on`) handlers, which receive the
/// actor by shared reference, can still record what they did.
#[derive(Debug, Default)]
struct MemoryStoreCounters {
    conversations_created: AtomicU64,
    messages_saved: AtomicU64,
    conversations_loaded: AtomicU64,
    state_saves: AtomicU64,
    state_loads: AtomicU64,
    memories_stored: AtomicU64,
    memory_searches: AtomicU64,
    context_windows_built: AtomicU64,
}

/// A handle to the Memory Store's counters.
///
/// Cloning shares the same counters rather than copying their values; use
/// [`snapshot`](Self::snapshot) for a point-in-time copy.
#[derive(Debug, Clone, Default)]
pub struct MemoryStoreMetrics {
    counters: Arc<MemoryStoreCounters>,
}

/// A point-in-time copy of [`MemoryStoreMetrics`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryStoreMetricsSnapshot {
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
    /// Number of memories stored
    pub memories_stored: u64,
    /// Number of memory searches performed
    pub memory_searches: u64,
    /// Number of context windows built
    pub context_windows_built: u64,
}

impl MemoryStoreMetrics {
    /// Number of conversations created.
    #[must_use]
    pub fn conversations_created(&self) -> u64 {
        self.counters.conversations_created.load(Ordering::Relaxed)
    }

    /// Number of messages saved.
    #[must_use]
    pub fn messages_saved(&self) -> u64 {
        self.counters.messages_saved.load(Ordering::Relaxed)
    }

    /// Number of conversations loaded.
    #[must_use]
    pub fn conversations_loaded(&self) -> u64 {
        self.counters.conversations_loaded.load(Ordering::Relaxed)
    }

    /// Number of agent state saves.
    #[must_use]
    pub fn state_saves(&self) -> u64 {
        self.counters.state_saves.load(Ordering::Relaxed)
    }

    /// Number of agent state loads.
    #[must_use]
    pub fn state_loads(&self) -> u64 {
        self.counters.state_loads.load(Ordering::Relaxed)
    }

    /// Number of memories stored.
    #[must_use]
    pub fn memories_stored(&self) -> u64 {
        self.counters.memories_stored.load(Ordering::Relaxed)
    }

    /// Number of memory searches performed.
    #[must_use]
    pub fn memory_searches(&self) -> u64 {
        self.counters.memory_searches.load(Ordering::Relaxed)
    }

    /// Number of context windows built.
    #[must_use]
    pub fn context_windows_built(&self) -> u64 {
        self.counters.context_windows_built.load(Ordering::Relaxed)
    }

    /// Takes a point-in-time copy of every counter.
    #[must_use]
    pub fn snapshot(&self) -> MemoryStoreMetricsSnapshot {
        MemoryStoreMetricsSnapshot {
            conversations_created: self.conversations_created(),
            messages_saved: self.messages_saved(),
            conversations_loaded: self.conversations_loaded(),
            state_saves: self.state_saves(),
            state_loads: self.state_loads(),
            memories_stored: self.memories_stored(),
            memory_searches: self.memory_searches(),
            context_windows_built: self.context_windows_built(),
        }
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
    /// Database connection (initialized after start)
    pub connection: Option<Connection>,
    /// Whether the store is shutting down
    pub shutting_down: bool,
    /// Checkpoint IDs claimed by turns running in this process right now.
    ///
    /// Owned here — by the one actor every checkpoint read and write already
    /// goes through — rather than shared behind a lock, so claim and release
    /// are ordinary messages and the registry has exactly one owner.
    pub active_checkpoints: std::collections::HashSet<CheckpointId>,
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
        let mut builder = runtime.new_actor_with_name::<MemoryStore>("memory_store".to_string());

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
                let metrics = actor.model.metrics.snapshot();
                tracing::info!(
                    conversations_created = metrics.conversations_created,
                    messages_saved = metrics.messages_saved,
                    "Memory Store shutting down"
                );
                Reply::ready()
            });

        // Configure message handlers
        configure_handlers(&mut builder);

        builder.start().await
    }

    /// Returns a usable connection, or the reason there is not one.
    ///
    /// Callers turn the error into a reply so that a request issued before
    /// initialization completes, or during shutdown, gets a definite answer
    /// instead of silence.
    fn ready_connection(&self) -> Result<Connection, PersistenceError> {
        if self.shutting_down {
            return Err(PersistenceError::shutting_down());
        }

        self.connection
            .clone()
            .ok_or_else(PersistenceError::not_initialized)
    }
}

// =============================================================================
// Handler plumbing
// =============================================================================

/// The future type acton-reactive expects back from a handler.
///
/// Mirrors the crate's internal `FutureBox`, which is not publicly exported.
/// The `Sync` bound is what forces database work onto a task: libsql's futures
/// are `Send` but not `Sync`.
type ReplyFuture = Pin<Box<dyn Future<Output = ()> + Send + Sync + 'static>>;

/// Runs database work on a task while keeping it attached to the handler.
///
/// libsql's futures are not `Sync`, but acton-reactive's reply futures must be
/// `Send + Sync`. A `JoinHandle` is both, so the work is spawned and the handle
/// awaited inside the reply future — the work stays tied to the handler's
/// lifetime rather than being detached and forgotten.
fn attached<F>(work: F) -> ReplyFuture
where
    F: Future<Output = ()> + Send + 'static,
{
    let task = tokio::spawn(work);

    Reply::pending(async move {
        if let Err(e) = task.await {
            tracing::error!(error = %e, "Memory store task did not complete");
        }
    })
}

/// Sends a reply that is already known without touching the database.
fn reply_now<M>(reply: OutboundEnvelope, message: M) -> ReplyFuture
where
    M: ActonMessage + 'static,
{
    Reply::pending(async move {
        reply.send(message).await;
    })
}

/// Logs a failed operation, leaving successes silent.
fn log_failure<T>(operation: &'static str, result: &Result<T, PersistenceError>) {
    if let Err(e) = result {
        tracing::error!(operation, error = %e, "Memory store operation failed");
    }
}

/// Configures message handlers for the Memory Store actor.
fn configure_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    configure_init_handler(builder);
    configure_conversation_write_handlers(builder);
    configure_conversation_read_handlers(builder);
    configure_state_handlers(builder);
    configure_memory_write_handlers(builder);
    configure_memory_read_handlers(builder);
    configure_checkpoint_write_handlers(builder);
    configure_checkpoint_read_handlers(builder);
}

// -----------------------------------------------------------------------------
// Checkpoint handlers
//
// Writes are `mutate_on` for the same reason conversation writes are: a turn
// checkpoints after every round, and the newest record has to be the one left
// standing. Reads are `act_on` and overlap freely.
// -----------------------------------------------------------------------------

/// Configures the checkpoint handlers that write.
fn configure_checkpoint_write_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    builder.mutate_on::<SaveCheckpoint>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let record = envelope.message().record.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, CheckpointSaved { result: Err(e) }),
        };

        attached(async move {
            let result = persistence::save_checkpoint(&conn, &record).await;
            log_failure("save_checkpoint", &result);
            reply.send(CheckpointSaved { result }).await;
        })
    });

    builder.mutate_on::<ClaimCheckpoint>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let granted = actor
            .model
            .active_checkpoints
            .insert(envelope.message().id.clone());
        reply_now(reply, CheckpointClaimed { granted })
    });

    builder.mutate_on::<ReleaseCheckpoint>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        actor
            .model
            .active_checkpoints
            .remove(&envelope.message().id);
        reply_now(reply, CheckpointReleased)
    });

    builder.mutate_on::<DeleteCheckpoint>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let id = envelope.message().id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => {
                return reply_now(
                    reply,
                    OperationCompleted {
                        operation: "delete_checkpoint",
                        result: Err(e),
                    },
                );
            }
        };

        attached(async move {
            let result = persistence::delete_checkpoint(&conn, &id).await;
            log_failure("delete_checkpoint", &result);
            reply
                .send(OperationCompleted {
                    operation: "delete_checkpoint",
                    result,
                })
                .await;
        })
    });
}

/// Configures the checkpoint handlers that only read.
fn configure_checkpoint_read_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    builder.act_on::<LoadCheckpoint>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let id = envelope.message().id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, CheckpointLoaded { result: Err(e) }),
        };

        attached(async move {
            let result = persistence::load_checkpoint(&conn, &id).await;
            log_failure("load_checkpoint", &result);
            reply.send(CheckpointLoaded { result }).await;
        })
    });

    builder.act_on::<ListCheckpointClaims>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let ids: Vec<CheckpointId> = actor.model.active_checkpoints.iter().cloned().collect();
        reply_now(reply, CheckpointClaims { ids })
    });

    builder.act_on::<ListCheckpoints>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let status = envelope.message().status;

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, CheckpointList { result: Err(e) }),
        };

        attached(async move {
            let result = persistence::list_checkpoints(&conn, status).await;
            log_failure("list_checkpoints", &result);
            reply.send(CheckpointList { result }).await;
        })
    });
}

/// Configures the initialization handler.
fn configure_init_handler(builder: &mut ManagedActor<Idle, MemoryStore>) {
    // Handle SetConnection (internal message for async init completion)
    builder.mutate_on::<SetConnection>(|actor, envelope| {
        actor.model.connection = Some(envelope.message().conn.clone());
        tracing::info!("Memory Store connection established");
        Reply::ready()
    });

    builder.mutate_on::<InitMemoryStore>(|actor, envelope| {
        let config = envelope.message().config.clone();
        let actor_handle = actor.handle().clone();
        actor.model.config = Some(config.clone());

        attached(async move {
            match initialize_database(&config).await {
                Ok((_db, conn)) => {
                    // Send connection back to actor via message
                    actor_handle.send(SetConnection { conn }).await;
                    tracing::info!(db_path = %config.db_path, "Memory Store initialized with database");
                }
                Err(e) => {
                    tracing::error!(error = %e, "Memory Store initialization failed");
                }
            }
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

// -----------------------------------------------------------------------------
// Conversation writes
//
// These stay on `mutate_on` deliberately. acton-reactive does not guarantee
// ordering between read-only handlers, and a conversation log is order
// sensitive: two racing `SaveMessage`s would persist a user turn and the
// assistant turn that answered it in an arbitrary order.
// -----------------------------------------------------------------------------

/// Configures the conversation handlers that write.
fn configure_conversation_write_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    builder.mutate_on::<CreateConversation>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let agent_id = envelope.message().agent_id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, ConversationCreated { result: Err(e) }),
        };

        actor
            .model
            .metrics
            .counters
            .conversations_created
            .fetch_add(1, Ordering::Relaxed);

        attached(async move {
            let result = persistence::create_conversation(&conn, &agent_id).await;
            log_failure("create_conversation", &result);
            reply.send(ConversationCreated { result }).await;
        })
    });

    builder.mutate_on::<SaveMessage>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let msg = envelope.message();
        let conversation_id = msg.conversation_id.clone();
        let message = msg.message.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, MessageSaved { result: Err(e) }),
        };

        actor
            .model
            .metrics
            .counters
            .messages_saved
            .fetch_add(1, Ordering::Relaxed);

        attached(async move {
            let result = persistence::save_message(&conn, &conversation_id, &message).await;
            log_failure("save_message", &result);
            reply.send(MessageSaved { result }).await;
        })
    });

    builder.mutate_on::<DeleteConversation>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let conversation_id = envelope.message().conversation_id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => {
                return reply_now(
                    reply,
                    OperationCompleted {
                        operation: "delete_conversation",
                        result: Err(e),
                    },
                );
            }
        };

        attached(async move {
            let result = persistence::delete_conversation(&conn, &conversation_id).await;
            log_failure("delete_conversation", &result);
            reply
                .send(OperationCompleted {
                    operation: "delete_conversation",
                    result,
                })
                .await;
        })
    });
}

// -----------------------------------------------------------------------------
// Conversation reads
//
// Read-only against the actor's own state, so they run on `act_on` and overlap
// with each other instead of serialising the whole store behind one query.
// -----------------------------------------------------------------------------

/// Configures the conversation handlers that only read.
fn configure_conversation_read_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    builder.act_on::<LoadConversation>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let conversation_id = envelope.message().conversation_id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => {
                return reply_now(
                    reply,
                    ConversationLoaded {
                        conversation_id,
                        result: Err(e),
                    },
                );
            }
        };

        actor
            .model
            .metrics
            .counters
            .conversations_loaded
            .fetch_add(1, Ordering::Relaxed);

        attached(async move {
            let result = persistence::load_conversation_messages(&conn, &conversation_id).await;
            log_failure("load_conversation", &result);
            reply
                .send(ConversationLoaded {
                    conversation_id,
                    result,
                })
                .await;
        })
    });

    builder.act_on::<GetLatestConversation>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let agent_id = envelope.message().agent_id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, LatestConversationResponse { result: Err(e) }),
        };

        attached(async move {
            let result = persistence::get_latest_conversation(&conn, &agent_id).await;
            log_failure("get_latest_conversation", &result);
            reply.send(LatestConversationResponse { result }).await;
        })
    });

    builder.act_on::<ListConversations>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let agent_id = envelope.message().agent_id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, ConversationList { result: Err(e) }),
        };

        attached(async move {
            let result = persistence::list_conversations(&conn, &agent_id).await;
            log_failure("list_conversations", &result);
            reply.send(ConversationList { result }).await;
        })
    });
}

// -----------------------------------------------------------------------------
// Agent state
// -----------------------------------------------------------------------------

/// Configures agent state handlers.
fn configure_state_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    // A write, and an upsert at that: concurrent saves for one agent would make
    // last-writer-wins nondeterministic, so this stays on the serial path.
    builder.mutate_on::<SaveAgentState>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let snapshot = envelope.message().snapshot.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => {
                return reply_now(
                    reply,
                    OperationCompleted {
                        operation: "save_agent_state",
                        result: Err(e),
                    },
                );
            }
        };

        actor
            .model
            .metrics
            .counters
            .state_saves
            .fetch_add(1, Ordering::Relaxed);

        attached(async move {
            let result = persistence::save_agent_state(&conn, &snapshot).await;
            log_failure("save_agent_state", &result);
            reply
                .send(OperationCompleted {
                    operation: "save_agent_state",
                    result,
                })
                .await;
        })
    });

    builder.act_on::<LoadAgentState>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let agent_id = envelope.message().agent_id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, AgentStateLoaded { result: Err(e) }),
        };

        actor
            .model
            .metrics
            .counters
            .state_loads
            .fetch_add(1, Ordering::Relaxed);

        attached(async move {
            let result = persistence::load_agent_state(&conn, &agent_id).await;
            log_failure("load_agent_state", &result);
            reply.send(AgentStateLoaded { result }).await;
        })
    });
}

// -----------------------------------------------------------------------------
// Memory writes
// -----------------------------------------------------------------------------

/// Configures the memory handlers that write.
fn configure_memory_write_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    builder.mutate_on::<StoreMemory>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let msg = envelope.message();
        let agent_id = msg.agent_id.clone();
        let content = msg.content.clone();
        let embedding = msg.embedding.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, MemoryStored { result: Err(e) }),
        };

        actor
            .model
            .metrics
            .counters
            .memories_stored
            .fetch_add(1, Ordering::Relaxed);

        attached(async move {
            let memory = match embedding {
                Some(emb) => Memory::with_embedding(agent_id, content, emb),
                None => Memory::new(agent_id, content),
            };

            let result = persistence::save_memory(&conn, &memory).await;
            log_failure("store_memory", &result);
            reply.send(MemoryStored { result }).await;
        })
    });

    builder.mutate_on::<DeleteMemory>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let memory_id = envelope.message().memory_id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => {
                return reply_now(
                    reply,
                    OperationCompleted {
                        operation: "delete_memory",
                        result: Err(e),
                    },
                );
            }
        };

        attached(async move {
            let result = persistence::delete_memory(&conn, &memory_id).await;
            log_failure("delete_memory", &result);
            reply
                .send(OperationCompleted {
                    operation: "delete_memory",
                    result,
                })
                .await;
        })
    });

    builder.mutate_on::<DeleteAgentMemories>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let agent_id = envelope.message().agent_id.clone();

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => {
                return reply_now(
                    reply,
                    OperationCompleted {
                        operation: "delete_agent_memories",
                        result: Err(e),
                    },
                );
            }
        };

        attached(async move {
            let result = persistence::delete_memories_for_agent(&conn, &agent_id).await;
            log_failure("delete_agent_memories", &result);
            reply
                .send(OperationCompleted {
                    operation: "delete_agent_memories",
                    result,
                })
                .await;
        })
    });
}

// -----------------------------------------------------------------------------
// Memory reads
// -----------------------------------------------------------------------------

/// Configures the memory handlers that only read.
fn configure_memory_read_handlers(builder: &mut ManagedActor<Idle, MemoryStore>) {
    builder.act_on::<SearchMemories>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let msg = envelope.message();
        let agent_id = msg.agent_id.clone();
        let query_embedding = msg.query_embedding.clone();
        let limit = msg.limit;
        let min_similarity = msg.min_similarity;

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, MemorySearchResults { result: Err(e) }),
        };

        actor
            .model
            .metrics
            .counters
            .memory_searches
            .fetch_add(1, Ordering::Relaxed);

        attached(async move {
            let result = persistence::search_memories_by_embedding(
                &conn,
                &agent_id,
                &query_embedding,
                limit,
                min_similarity,
            )
            .await;
            log_failure("search_memories", &result);
            reply.send(MemorySearchResults { result }).await;
        })
    });

    builder.act_on::<LoadMemories>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let msg = envelope.message();
        let agent_id = msg.agent_id.clone();
        let limit = msg.limit;

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, MemoriesLoaded { result: Err(e) }),
        };

        attached(async move {
            let result = persistence::load_memories_for_agent(&conn, &agent_id, limit).await;
            log_failure("load_memories", &result);
            reply.send(MemoriesLoaded { result }).await;
        })
    });

    builder.act_on::<GetContextWindow>(|actor, envelope| {
        let reply = envelope.reply_envelope();
        let request = ContextWindowRequest::from(envelope.message());

        let conn = match actor.model.ready_connection() {
            Ok(conn) => conn,
            Err(e) => return reply_now(reply, ContextWindowResponse { result: Err(e) }),
        };

        actor
            .model
            .metrics
            .counters
            .context_windows_built
            .fetch_add(1, Ordering::Relaxed);

        attached(async move {
            let result = build_context_window(&conn, request).await;
            log_failure("get_context_window", &result);
            reply.send(ContextWindowResponse { result }).await;
        })
    });
}

/// Owned copy of the inputs needed to build a context window.
struct ContextWindowRequest {
    agent_id: AgentId,
    system_prompt: String,
    conversation: Vec<Message>,
    query_embedding: Option<Embedding>,
    max_tokens: usize,
    memory_limit: usize,
}

impl From<&GetContextWindow> for ContextWindowRequest {
    fn from(msg: &GetContextWindow) -> Self {
        Self {
            agent_id: msg.agent_id.clone(),
            system_prompt: msg.system_prompt.clone(),
            conversation: msg.conversation.clone(),
            query_embedding: msg.query_embedding.clone(),
            max_tokens: msg.max_tokens,
            memory_limit: msg.memory_limit,
        }
    }
}

/// Retrieves relevant memories and assembles them into a context window.
///
/// A memory-retrieval failure is fatal here rather than silently degrading to
/// an empty memory set: a caller asking for context that includes memories
/// should learn that the memories are missing.
async fn build_context_window(
    conn: &Connection,
    request: ContextWindowRequest,
) -> Result<ContextWindowData, PersistenceError> {
    let memories = match request.query_embedding {
        Some(ref embedding) => persistence::search_memories_by_embedding(
            conn,
            &request.agent_id,
            embedding,
            request.memory_limit,
            Some(0.0), // Include all matches
        )
        .await?
        .into_iter()
        .map(|scored| scored.memory)
        .collect(),
        None => Vec::new(),
    };

    let included_memories = memories.len();

    let config = ContextWindowConfig::with_max_tokens(request.max_tokens);
    let window = ContextWindow::new(config);

    let messages = window.build_context(&request.system_prompt, &memories, &request.conversation);
    let stats = window.get_context_stats(&messages);

    Ok(ContextWindowData {
        messages,
        stats,
        included_memories,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_metrics_start_at_zero() {
        let metrics = MemoryStoreMetrics::default();

        assert_eq!(metrics.snapshot(), MemoryStoreMetricsSnapshot::default());
    }

    #[test]
    fn cloned_metrics_share_counters() {
        let metrics = MemoryStoreMetrics::default();
        let clone = metrics.clone();

        clone
            .counters
            .messages_saved
            .fetch_add(3, Ordering::Relaxed);

        assert_eq!(
            metrics.messages_saved(),
            3,
            "a clone must observe the same counters, not a copy"
        );
    }

    #[test]
    fn snapshot_captures_every_counter() {
        let metrics = MemoryStoreMetrics::default();
        let counters = &metrics.counters;

        counters
            .conversations_created
            .fetch_add(1, Ordering::Relaxed);
        counters.messages_saved.fetch_add(2, Ordering::Relaxed);
        counters
            .conversations_loaded
            .fetch_add(3, Ordering::Relaxed);
        counters.state_saves.fetch_add(4, Ordering::Relaxed);
        counters.state_loads.fetch_add(5, Ordering::Relaxed);
        counters.memories_stored.fetch_add(6, Ordering::Relaxed);
        counters.memory_searches.fetch_add(7, Ordering::Relaxed);
        counters
            .context_windows_built
            .fetch_add(8, Ordering::Relaxed);

        assert_eq!(
            metrics.snapshot(),
            MemoryStoreMetricsSnapshot {
                conversations_created: 1,
                messages_saved: 2,
                conversations_loaded: 3,
                state_saves: 4,
                state_loads: 5,
                memories_stored: 6,
                memory_searches: 7,
                context_windows_built: 8,
            }
        );
    }

    #[test]
    fn uninitialized_store_reports_not_initialized() {
        let store = MemoryStore::default();

        let error = store
            .ready_connection()
            .expect_err("a store with no connection cannot serve requests");

        assert!(error.is_not_initialized());
    }

    #[test]
    fn shutting_down_takes_precedence_over_missing_connection() {
        let store = MemoryStore {
            shutting_down: true,
            ..MemoryStore::default()
        };

        let error = store
            .ready_connection()
            .expect_err("a shutting-down store cannot serve requests");

        assert!(error.is_shutting_down());
    }
}
