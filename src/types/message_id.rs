//! Message identifier type using TypeID format.
//!
//! MessageId provides a human-readable, time-sortable, globally unique identifier
//! for persisted messages. Format: `msg_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated message identifier.
///
/// Uses TypeID format for human-readable, time-sortable, globally unique IDs.
/// Example: `msg_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MessageId(MagicTypeId);

/// Error returned when attempting to create an invalid message ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidMessageId {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "msg")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidMessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid message ID: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidMessageId {}

impl MessageId {
    /// The TypeID prefix for message identifiers.
    pub const PREFIX: &'static str = "msg";

    /// Creates a new message ID with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses a message ID from a string, validating the prefix.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse
    ///
    /// # Returns
    ///
    /// A valid `MessageId` if the string is a valid TypeID with the "msg" prefix.
    ///
    /// # Errors
    ///
    /// Returns `InvalidMessageId::Parse` if the string is not a valid TypeID format.
    /// Returns `InvalidMessageId::WrongPrefix` if the TypeID has a different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidMessageId> {
        let id = MagicTypeId::from_str(s).map_err(|e| InvalidMessageId::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidMessageId::WrongPrefix {
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

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for MessageId {
    type Err = InvalidMessageId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for MessageId {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for MessageId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MessageId {
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
    fn new_creates_valid_message_id() {
        let id = MessageId::new();
        assert!(id.to_string().starts_with("msg_"));
    }

    #[test]
    fn parse_valid_message_id() {
        let id_str = MessageId::new().to_string();
        let parsed = MessageId::parse(&id_str);
        assert!(parsed.is_ok());
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = MessageId::parse("agent_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidMessageId::WrongPrefix {
                expected: "msg",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        let result = MessageId::parse("not-a-valid-typeid");
        assert!(matches!(result, Err(InvalidMessageId::Parse(_))));
    }

    #[test]
    fn message_ids_are_unique() {
        let id1 = MessageId::new();
        let id2 = MessageId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn message_id_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let id = MessageId::new();
        set.insert(id.clone());

        assert!(set.contains(&id));
    }

    #[test]
    fn display_format() {
        let id = MessageId::new();
        let s = id.to_string();
        assert!(s.starts_with("msg_"));
    }

    #[test]
    fn default_creates_new_id() {
        let id1 = MessageId::default();
        let id2 = MessageId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn serialization_roundtrip() {
        let id = MessageId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: MessageId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
