//! Persistence error types.
//!
//! Custom error types for all persistence operations, providing actionable
//! error messages and support for error classification.

use crate::types::AgentId;
use std::fmt;

/// Errors that can occur in persistence operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistenceError {
    /// Optional agent ID if error is agent-specific
    pub agent_id: Option<AgentId>,
    /// The specific error that occurred (boxed for size efficiency)
    kind: Box<PersistenceErrorKind>,
}

/// Specific persistence error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistenceErrorKind {
    /// Failed to open or create database
    DatabaseOpen {
        /// Path to the database file
        path: String,
        /// Error message from database
        message: String,
    },
    /// Failed to initialize schema
    SchemaInit {
        /// Error message from database
        message: String,
    },
    /// Query execution failed
    QueryFailed {
        /// The operation that failed
        operation: String,
        /// Error message from database
        message: String,
    },
    /// Record not found
    NotFound {
        /// The type of entity not found
        entity: String,
        /// The ID of the entity
        id: String,
    },
    /// Serialization failed
    SerializationFailed {
        /// Error message
        message: String,
    },
    /// Deserialization failed
    DeserializationFailed {
        /// Error message
        message: String,
    },
    /// Store is shutting down
    ShuttingDown,
    /// Transaction failed
    TransactionFailed {
        /// Error message
        message: String,
    },
    /// Connection error
    ConnectionError {
        /// Error message
        message: String,
    },
}

impl PersistenceError {
    /// Creates a new persistence error with the given kind.
    #[must_use]
    pub fn new(kind: PersistenceErrorKind) -> Self {
        Self {
            agent_id: None,
            kind: Box::new(kind),
        }
    }

    /// Creates a new persistence error associated with a specific agent.
    #[must_use]
    pub fn with_agent(agent_id: AgentId, kind: PersistenceErrorKind) -> Self {
        Self {
            agent_id: Some(agent_id),
            kind: Box::new(kind),
        }
    }

    /// Returns a reference to the error kind.
    #[must_use]
    pub fn kind(&self) -> &PersistenceErrorKind {
        &self.kind
    }

    /// Creates a database open error.
    #[must_use]
    pub fn database_open(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(PersistenceErrorKind::DatabaseOpen {
            path: path.into(),
            message: message.into(),
        })
    }

    /// Creates a schema initialization error.
    #[must_use]
    pub fn schema_init(message: impl Into<String>) -> Self {
        Self::new(PersistenceErrorKind::SchemaInit {
            message: message.into(),
        })
    }

    /// Creates a query failed error.
    #[must_use]
    pub fn query_failed(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(PersistenceErrorKind::QueryFailed {
            operation: operation.into(),
            message: message.into(),
        })
    }

    /// Creates a not found error.
    #[must_use]
    pub fn not_found(entity: impl Into<String>, id: impl Into<String>) -> Self {
        Self::new(PersistenceErrorKind::NotFound {
            entity: entity.into(),
            id: id.into(),
        })
    }

    /// Creates a serialization failed error.
    #[must_use]
    pub fn serialization_failed(message: impl Into<String>) -> Self {
        Self::new(PersistenceErrorKind::SerializationFailed {
            message: message.into(),
        })
    }

    /// Creates a deserialization failed error.
    #[must_use]
    pub fn deserialization_failed(message: impl Into<String>) -> Self {
        Self::new(PersistenceErrorKind::DeserializationFailed {
            message: message.into(),
        })
    }

    /// Creates a shutting down error.
    #[must_use]
    pub fn shutting_down() -> Self {
        Self::new(PersistenceErrorKind::ShuttingDown)
    }

    /// Creates a transaction failed error.
    #[must_use]
    pub fn transaction_failed(message: impl Into<String>) -> Self {
        Self::new(PersistenceErrorKind::TransactionFailed {
            message: message.into(),
        })
    }

    /// Creates a connection error.
    #[must_use]
    pub fn connection_error(message: impl Into<String>) -> Self {
        Self::new(PersistenceErrorKind::ConnectionError {
            message: message.into(),
        })
    }

    /// Returns true if this error is retriable.
    ///
    /// Connection errors and transaction failures are typically transient
    /// and may succeed on retry.
    #[must_use]
    pub fn is_retriable(&self) -> bool {
        matches!(
            *self.kind,
            PersistenceErrorKind::ConnectionError { .. }
                | PersistenceErrorKind::TransactionFailed { .. }
        )
    }

    /// Returns true if this error indicates a record was not found.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        matches!(*self.kind, PersistenceErrorKind::NotFound { .. })
    }

    /// Returns true if the store is shutting down.
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        matches!(*self.kind, PersistenceErrorKind::ShuttingDown)
    }
}

impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref id) = self.agent_id {
            write!(f, "agent '{}': ", id)?;
        }

        match self.kind.as_ref() {
            PersistenceErrorKind::DatabaseOpen { path, message } => {
                write!(
                    f,
                    "failed to open database at '{}': {}; check file permissions and path",
                    path, message
                )
            }
            PersistenceErrorKind::SchemaInit { message } => {
                write!(
                    f,
                    "failed to initialize database schema: {}; database may be corrupted",
                    message
                )
            }
            PersistenceErrorKind::QueryFailed { operation, message } => {
                write!(f, "{} query failed: {}", operation, message)
            }
            PersistenceErrorKind::NotFound { entity, id } => {
                write!(f, "{} with id '{}' not found", entity, id)
            }
            PersistenceErrorKind::SerializationFailed { message } => {
                write!(f, "failed to serialize data: {}", message)
            }
            PersistenceErrorKind::DeserializationFailed { message } => {
                write!(f, "failed to deserialize data: {}", message)
            }
            PersistenceErrorKind::ShuttingDown => {
                write!(
                    f,
                    "memory store is shutting down; cannot accept new requests"
                )
            }
            PersistenceErrorKind::TransactionFailed { message } => {
                write!(f, "transaction failed: {}; retry may succeed", message)
            }
            PersistenceErrorKind::ConnectionError { message } => {
                write!(f, "database connection error: {}", message)
            }
        }
    }
}

impl std::error::Error for PersistenceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_error_database_open_display() {
        let error = PersistenceError::database_open("/path/to/db", "permission denied");
        let msg = error.to_string();
        assert!(msg.contains("/path/to/db"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn persistence_error_connection_is_retriable() {
        let error = PersistenceError::connection_error("timeout");
        assert!(error.is_retriable());
    }

    #[test]
    fn persistence_error_transaction_is_retriable() {
        let error = PersistenceError::transaction_failed("deadlock");
        assert!(error.is_retriable());
    }

    #[test]
    fn persistence_error_not_found_is_not_retriable() {
        let error = PersistenceError::not_found("conversation", "conv_123");
        assert!(!error.is_retriable());
        assert!(error.is_not_found());
    }

    #[test]
    fn persistence_error_schema_init_display() {
        let error = PersistenceError::schema_init("table already exists");
        let msg = error.to_string();
        assert!(msg.contains("schema"));
        assert!(msg.contains("table already exists"));
    }

    #[test]
    fn persistence_error_query_failed_display() {
        let error = PersistenceError::query_failed("INSERT", "constraint violation");
        let msg = error.to_string();
        assert!(msg.contains("INSERT"));
        assert!(msg.contains("constraint violation"));
    }

    #[test]
    fn persistence_error_not_found_display() {
        let error = PersistenceError::not_found("conversation", "conv_abc");
        let msg = error.to_string();
        assert!(msg.contains("conversation"));
        assert!(msg.contains("conv_abc"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn persistence_error_serialization_display() {
        let error = PersistenceError::serialization_failed("invalid utf-8");
        let msg = error.to_string();
        assert!(msg.contains("serialize"));
        assert!(msg.contains("invalid utf-8"));
    }

    #[test]
    fn persistence_error_deserialization_display() {
        let error = PersistenceError::deserialization_failed("missing field");
        let msg = error.to_string();
        assert!(msg.contains("deserialize"));
        assert!(msg.contains("missing field"));
    }

    #[test]
    fn persistence_error_shutting_down() {
        let error = PersistenceError::shutting_down();
        assert!(error.is_shutting_down());
        let msg = error.to_string();
        assert!(msg.contains("shutting down"));
    }

    #[test]
    fn persistence_error_with_agent_display() {
        let agent_id = AgentId::new();
        let error = PersistenceError::with_agent(
            agent_id.clone(),
            PersistenceErrorKind::NotFound {
                entity: "state".to_string(),
                id: "state_123".to_string(),
            },
        );
        let msg = error.to_string();
        assert!(msg.contains(&agent_id.to_string()));
        assert!(msg.contains("state"));
    }

    #[test]
    fn persistence_error_kind_accessor() {
        let error = PersistenceError::connection_error("timeout");
        assert!(matches!(
            error.kind(),
            PersistenceErrorKind::ConnectionError { .. }
        ));
    }
}
