//! Checkpoint identifier type using TypeID format.
//!
//! A *checkpoint* names one resumable turn. Unlike
//! [`TurnId`](crate::types::TurnId), which is minted fresh every time the
//! prompt loop runs, a `CheckpointId` is chosen by the caller and outlives the
//! process: it is the key a resume looks the saved progress up by. Two runs
//! that pass the same `CheckpointId` are two attempts at the *same* turn.
//!
//! Format: `ckpt_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated identifier for one resumable turn.
///
/// Uses TypeID format for human-readable, time-sortable, globally unique IDs.
/// Example: `ckpt_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CheckpointId(MagicTypeId);

/// Error returned when attempting to create an invalid checkpoint ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidCheckpointId {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "ckpt")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidCheckpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid checkpoint ID: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidCheckpointId {}

impl CheckpointId {
    /// The TypeID prefix for checkpoint identifiers.
    pub const PREFIX: &'static str = "ckpt";

    /// Creates a new checkpoint ID with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses a checkpoint ID from a string, validating the prefix.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidCheckpointId::Parse`] if the string is not a valid
    /// TypeID, and [`InvalidCheckpointId::WrongPrefix`] if it carries a
    /// different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidCheckpointId> {
        let id = MagicTypeId::from_str(s).map_err(|e| InvalidCheckpointId::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidCheckpointId::WrongPrefix {
                expected: Self::PREFIX,
                actual: prefix.to_string(),
            });
        }

        Ok(Self(id))
    }

    /// Returns a reference to the underlying `MagicTypeId`.
    #[must_use]
    pub fn inner(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Default for CheckpointId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for CheckpointId {
    type Err = InvalidCheckpointId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for CheckpointId {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for CheckpointId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CheckpointId {
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
    fn new_creates_valid_checkpoint_id() {
        let id = CheckpointId::new();
        assert!(id.to_string().starts_with("ckpt_"));
    }

    #[test]
    fn parse_round_trips_a_generated_id() {
        let id = CheckpointId::new();
        assert_eq!(CheckpointId::parse(&id.to_string()), Ok(id));
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = CheckpointId::parse("turn_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidCheckpointId::WrongPrefix {
                expected: "ckpt",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        assert!(matches!(
            CheckpointId::parse("not-a-valid-typeid"),
            Err(InvalidCheckpointId::Parse(_))
        ));
    }

    #[test]
    fn checkpoint_ids_are_unique() {
        assert_ne!(CheckpointId::new(), CheckpointId::new());
    }

    #[test]
    fn checkpoint_id_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let id = CheckpointId::new();
        set.insert(id.clone());

        assert!(set.contains(&id));
    }

    #[test]
    fn serialization_roundtrip() {
        let id = CheckpointId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: CheckpointId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn invalid_checkpoint_id_display_names_the_expected_prefix() {
        let err = CheckpointId::parse("turn_01h455vb4pex5vsknk084sn02q").unwrap_err();
        let text = err.to_string();
        assert!(text.contains("ckpt"), "{text}");
        assert!(text.contains("turn"), "{text}");
    }
}
