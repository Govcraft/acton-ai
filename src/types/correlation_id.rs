//! Correlation identifier type using TypeID format.
//!
//! CorrelationId provides a unique identifier for correlating requests with responses
//! across the actor system. Format: `corr_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated correlation identifier for request-response tracking.
///
/// Uses TypeID format for human-readable, time-sortable, globally unique IDs.
/// Example: `corr_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CorrelationId(MagicTypeId);

/// Error returned when attempting to create an invalid correlation ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidCorrelationId {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "corr")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidCorrelationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid correlation ID: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidCorrelationId {}

impl CorrelationId {
    /// The TypeID prefix for correlation identifiers.
    pub const PREFIX: &'static str = "corr";

    /// Creates a new correlation ID with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses a correlation ID from a string, validating the prefix.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse
    ///
    /// # Returns
    ///
    /// A valid `CorrelationId` if the string is a valid TypeID with the "corr" prefix.
    ///
    /// # Errors
    ///
    /// Returns `InvalidCorrelationId::Parse` if the string is not a valid TypeID format.
    /// Returns `InvalidCorrelationId::WrongPrefix` if the TypeID has a different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidCorrelationId> {
        let id =
            MagicTypeId::from_str(s).map_err(|e| InvalidCorrelationId::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidCorrelationId::WrongPrefix {
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

impl Default for CorrelationId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CorrelationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for CorrelationId {
    type Err = InvalidCorrelationId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for CorrelationId {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for CorrelationId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CorrelationId {
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
    fn new_creates_valid_correlation_id() {
        let id = CorrelationId::new();
        assert!(id.to_string().starts_with("corr_"));
    }

    #[test]
    fn parse_valid_correlation_id() {
        let id_str = CorrelationId::new().to_string();
        let parsed = CorrelationId::parse(&id_str);
        assert!(parsed.is_ok());
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = CorrelationId::parse("agent_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidCorrelationId::WrongPrefix {
                expected: "corr",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        let result = CorrelationId::parse("not-a-valid-typeid");
        assert!(matches!(result, Err(InvalidCorrelationId::Parse(_))));
    }

    #[test]
    fn correlation_ids_are_unique() {
        let id1 = CorrelationId::new();
        let id2 = CorrelationId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn correlation_id_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let id = CorrelationId::new();
        set.insert(id.clone());

        assert!(set.contains(&id));
    }

    #[test]
    fn display_format() {
        let id = CorrelationId::new();
        let s = id.to_string();
        assert!(s.starts_with("corr_"));
    }

    #[test]
    fn default_creates_new_id() {
        let id1 = CorrelationId::default();
        let id2 = CorrelationId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn serialization_roundtrip() {
        let id = CorrelationId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: CorrelationId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
