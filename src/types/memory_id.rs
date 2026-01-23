//! Memory identifier type using TypeID format.
//!
//! MemoryId provides a human-readable, time-sortable, globally unique identifier
//! for persisted memory entries. Format: `mem_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated memory identifier.
///
/// Uses TypeID format for human-readable, time-sortable, globally unique IDs.
/// Example: `mem_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryId(MagicTypeId);

/// Error returned when attempting to create an invalid memory ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidMemoryId {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "mem")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidMemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid memory ID: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidMemoryId {}

impl MemoryId {
    /// The TypeID prefix for memory identifiers.
    pub const PREFIX: &'static str = "mem";

    /// Creates a new memory ID with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses a memory ID from a string, validating the prefix.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse
    ///
    /// # Returns
    ///
    /// A valid `MemoryId` if the string is a valid TypeID with the "mem" prefix.
    ///
    /// # Errors
    ///
    /// Returns `InvalidMemoryId::Parse` if the string is not a valid TypeID format.
    /// Returns `InvalidMemoryId::WrongPrefix` if the TypeID has a different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidMemoryId> {
        let id = MagicTypeId::from_str(s).map_err(|e| InvalidMemoryId::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidMemoryId::WrongPrefix {
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

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for MemoryId {
    type Err = InvalidMemoryId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for MemoryId {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for MemoryId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MemoryId {
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
    fn new_creates_valid_memory_id() {
        let id = MemoryId::new();
        assert!(id.to_string().starts_with("mem_"));
    }

    #[test]
    fn parse_valid_memory_id() {
        let id_str = MemoryId::new().to_string();
        let parsed = MemoryId::parse(&id_str);
        assert!(parsed.is_ok());
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = MemoryId::parse("agent_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidMemoryId::WrongPrefix {
                expected: "mem",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        let result = MemoryId::parse("not-a-valid-typeid");
        assert!(matches!(result, Err(InvalidMemoryId::Parse(_))));
    }

    #[test]
    fn memory_ids_are_unique() {
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn memory_id_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let id = MemoryId::new();
        set.insert(id.clone());

        assert!(set.contains(&id));
    }

    #[test]
    fn display_format() {
        let id = MemoryId::new();
        let s = id.to_string();
        assert!(s.starts_with("mem_"));
    }

    #[test]
    fn default_creates_new_id() {
        let id1 = MemoryId::default();
        let id2 = MemoryId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn serialization_roundtrip() {
        let id = MemoryId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: MemoryId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
