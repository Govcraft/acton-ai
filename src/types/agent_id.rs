//! Agent identifier type using TypeID format.
//!
//! AgentId provides a human-readable, time-sortable, globally unique identifier
//! for agents in the system. Format: `agent_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated agent identifier.
///
/// Uses TypeID format for human-readable, time-sortable, globally unique IDs.
/// Example: `agent_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AgentId(MagicTypeId);

/// Error returned when attempting to create an invalid agent ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidAgentId {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "agent")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidAgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid agent ID: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidAgentId {}

impl AgentId {
    /// The TypeID prefix for agent identifiers.
    pub const PREFIX: &'static str = "agent";

    /// Creates a new agent ID with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses an agent ID from a string, validating the prefix.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse
    ///
    /// # Returns
    ///
    /// A valid `AgentId` if the string is a valid TypeID with the "agent" prefix.
    ///
    /// # Errors
    ///
    /// Returns `InvalidAgentId::Parse` if the string is not a valid TypeID format.
    /// Returns `InvalidAgentId::WrongPrefix` if the TypeID has a different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidAgentId> {
        let id = MagicTypeId::from_str(s).map_err(|e| InvalidAgentId::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidAgentId::WrongPrefix {
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

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for AgentId {
    type Err = InvalidAgentId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for AgentId {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for AgentId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AgentId {
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
    fn new_creates_valid_agent_id() {
        let id = AgentId::new();
        assert!(id.to_string().starts_with("agent_"));
    }

    #[test]
    fn parse_valid_agent_id() {
        let id_str = AgentId::new().to_string();
        let parsed = AgentId::parse(&id_str);
        assert!(parsed.is_ok());
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = AgentId::parse("user_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidAgentId::WrongPrefix {
                expected: "agent",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        let result = AgentId::parse("not-a-valid-typeid");
        assert!(matches!(result, Err(InvalidAgentId::Parse(_))));
    }

    #[test]
    fn agent_ids_are_unique() {
        let id1 = AgentId::new();
        let id2 = AgentId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn agent_id_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let id = AgentId::new();
        set.insert(id.clone());

        assert!(set.contains(&id));
    }

    #[test]
    fn display_format() {
        let id = AgentId::new();
        let s = id.to_string();
        assert!(s.starts_with("agent_"));
    }

    #[test]
    fn default_creates_new_id() {
        let id1 = AgentId::default();
        let id2 = AgentId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn serialization_roundtrip() {
        let id = AgentId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: AgentId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
