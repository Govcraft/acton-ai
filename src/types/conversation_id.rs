//! Conversation identifier type using TypeID format.
//!
//! ConversationId provides a human-readable, time-sortable, globally unique identifier
//! for conversations in the system. Format: `conv_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated conversation identifier.
///
/// Uses TypeID format for human-readable, time-sortable, globally unique IDs.
/// Example: `conv_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConversationId(MagicTypeId);

/// Error returned when attempting to create an invalid conversation ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidConversationId {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "conv")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidConversationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid conversation ID: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidConversationId {}

impl ConversationId {
    /// The TypeID prefix for conversation identifiers.
    pub const PREFIX: &'static str = "conv";

    /// Creates a new conversation ID with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses a conversation ID from a string, validating the prefix.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse
    ///
    /// # Returns
    ///
    /// A valid `ConversationId` if the string is a valid TypeID with the "conv" prefix.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConversationId::Parse` if the string is not a valid TypeID format.
    /// Returns `InvalidConversationId::WrongPrefix` if the TypeID has a different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidConversationId> {
        let id =
            MagicTypeId::from_str(s).map_err(|e| InvalidConversationId::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidConversationId::WrongPrefix {
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

impl Default for ConversationId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConversationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for ConversationId {
    type Err = InvalidConversationId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for ConversationId {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for ConversationId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ConversationId {
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
    fn new_creates_valid_conversation_id() {
        let id = ConversationId::new();
        assert!(id.to_string().starts_with("conv_"));
    }

    #[test]
    fn parse_valid_conversation_id() {
        let id_str = ConversationId::new().to_string();
        let parsed = ConversationId::parse(&id_str);
        assert!(parsed.is_ok());
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = ConversationId::parse("agent_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidConversationId::WrongPrefix {
                expected: "conv",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        let result = ConversationId::parse("not-a-valid-typeid");
        assert!(matches!(result, Err(InvalidConversationId::Parse(_))));
    }

    #[test]
    fn conversation_ids_are_unique() {
        let id1 = ConversationId::new();
        let id2 = ConversationId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn conversation_id_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let id = ConversationId::new();
        set.insert(id.clone());

        assert!(set.contains(&id));
    }

    #[test]
    fn display_format() {
        let id = ConversationId::new();
        let s = id.to_string();
        assert!(s.starts_with("conv_"));
    }

    #[test]
    fn default_creates_new_id() {
        let id1 = ConversationId::default();
        let id2 = ConversationId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn serialization_roundtrip() {
        let id = ConversationId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: ConversationId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
