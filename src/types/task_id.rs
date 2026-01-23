//! Task identifier type using TypeID format.
//!
//! TaskId provides a human-readable, time-sortable, globally unique identifier
//! for delegated tasks in the system. Format: `task_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated task identifier.
///
/// Uses TypeID format for human-readable, time-sortable, globally unique IDs.
/// Example: `task_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(MagicTypeId);

/// Error returned when attempting to create an invalid task ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidTaskId {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "task")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidTaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid task ID: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidTaskId {}

impl TaskId {
    /// The TypeID prefix for task identifiers.
    pub const PREFIX: &'static str = "task";

    /// Creates a new task ID with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses a task ID from a string, validating the prefix.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse
    ///
    /// # Returns
    ///
    /// A valid `TaskId` if the string is a valid TypeID with the "task" prefix.
    ///
    /// # Errors
    ///
    /// Returns `InvalidTaskId::Parse` if the string is not a valid TypeID format.
    /// Returns `InvalidTaskId::WrongPrefix` if the TypeID has a different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidTaskId> {
        let id = MagicTypeId::from_str(s).map_err(|e| InvalidTaskId::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidTaskId::WrongPrefix {
                expected: Self::PREFIX,
                actual: prefix.to_string(),
            });
        }

        Ok(Self(id))
    }

    /// Returns a reference to the underlying MagicTypeId.
    #[must_use]
    pub fn inner(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for TaskId {
    type Err = InvalidTaskId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for TaskId {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for TaskId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TaskId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_valid_task_id() {
        let id = TaskId::new();
        assert!(id.to_string().starts_with("task_"));
    }

    #[test]
    fn parse_valid_task_id() {
        let id_str = TaskId::new().to_string();
        let parsed = TaskId::parse(&id_str);
        assert!(parsed.is_ok());
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = TaskId::parse("agent_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidTaskId::WrongPrefix {
                expected: "task",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        let result = TaskId::parse("not-a-valid-typeid");
        assert!(matches!(result, Err(InvalidTaskId::Parse(_))));
    }

    #[test]
    fn task_ids_are_unique() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn task_id_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let id = TaskId::new();
        set.insert(id.clone());

        assert!(set.contains(&id));
    }

    #[test]
    fn display_format() {
        let id = TaskId::new();
        let s = id.to_string();
        assert!(s.starts_with("task_"));
    }

    #[test]
    fn default_creates_new_id() {
        let id1 = TaskId::default();
        let id2 = TaskId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn serialization_roundtrip() {
        let id = TaskId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: TaskId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
