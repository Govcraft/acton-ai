//! Turn identifier type using TypeID format.
//!
//! A *turn* is one `collect()` / `extract()` call: the whole prompt/tool loop,
//! however many provider rounds it takes. That is deliberately coarser than
//! [`CorrelationId`](crate::types::CorrelationId), which names a single
//! request-response cycle — a turn that drives four rounds has one `TurnId`
//! and four correlation IDs.
//!
//! Format: `turn_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated identifier for one turn of the prompt loop.
///
/// Uses TypeID format for human-readable, time-sortable, globally unique IDs.
/// Example: `turn_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TurnId(MagicTypeId);

/// Error returned when attempting to create an invalid turn ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidTurnId {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "turn")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidTurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid turn ID: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidTurnId {}

impl TurnId {
    /// The TypeID prefix for turn identifiers.
    pub const PREFIX: &'static str = "turn";

    /// Creates a new turn ID with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses a turn ID from a string, validating the prefix.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidTurnId::Parse`] if the string is not a valid TypeID,
    /// and [`InvalidTurnId::WrongPrefix`] if it carries a different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidTurnId> {
        let id = MagicTypeId::from_str(s).map_err(|e| InvalidTurnId::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidTurnId::WrongPrefix {
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

impl Default for TurnId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for TurnId {
    type Err = InvalidTurnId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for TurnId {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for TurnId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TurnId {
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
    fn new_creates_valid_turn_id() {
        let id = TurnId::new();
        assert!(id.to_string().starts_with("turn_"));
    }

    #[test]
    fn parse_round_trips_a_generated_id() {
        let id = TurnId::new();
        assert_eq!(TurnId::parse(&id.to_string()), Ok(id));
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = TurnId::parse("corr_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidTurnId::WrongPrefix {
                expected: "turn",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        assert!(matches!(
            TurnId::parse("not-a-valid-typeid"),
            Err(InvalidTurnId::Parse(_))
        ));
    }

    #[test]
    fn turn_ids_are_unique() {
        assert_ne!(TurnId::new(), TurnId::new());
    }

    #[test]
    fn turn_id_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let id = TurnId::new();
        set.insert(id.clone());

        assert!(set.contains(&id));
    }

    #[test]
    fn serialization_roundtrip() {
        let id = TurnId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: TurnId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn invalid_turn_id_display_names_the_expected_prefix() {
        let err = TurnId::parse("corr_01h455vb4pex5vsknk084sn02q").unwrap_err();
        let text = err.to_string();
        assert!(text.contains("turn"), "{text}");
        assert!(text.contains("corr"), "{text}");
    }
}
