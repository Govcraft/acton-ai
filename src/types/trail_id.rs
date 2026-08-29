//! Trail identifier type using TypeID format.
//!
//! A *trail* is one hash chain in one file. The identifier is minted once,
//! the first time an audit log opens a trail, and is then stamped into every
//! entry's hash pre-image and kept beside the file in a sidecar. That is what
//! stops one trail's entries from being passed off as another's: the chain
//! verifies only under the identity it was sealed with.
//!
//! Format: `trail_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated identifier for one audit trail.
///
/// Uses TypeID format for human-readable, time-sortable, globally unique IDs.
/// Example: `trail_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrailId(MagicTypeId);

/// Error returned when attempting to create an invalid trail ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidTrailId {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "trail")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidTrailId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid trail ID: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidTrailId {}

impl TrailId {
    /// The TypeID prefix for trail identifiers.
    pub const PREFIX: &'static str = "trail";

    /// Creates a new trail ID with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses a trail ID from a string, validating the prefix.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidTrailId::Parse`] if the string is not a valid TypeID,
    /// and [`InvalidTrailId::WrongPrefix`] if it carries a different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidTrailId> {
        let id = MagicTypeId::from_str(s).map_err(|e| InvalidTrailId::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidTrailId::WrongPrefix {
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

impl Default for TrailId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TrailId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for TrailId {
    type Err = InvalidTrailId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for TrailId {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for TrailId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TrailId {
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
    fn new_creates_valid_trail_id() {
        let id = TrailId::new();
        assert!(id.to_string().starts_with("trail_"));
    }

    #[test]
    fn parse_round_trips_a_generated_id() {
        let id = TrailId::new();
        assert_eq!(TrailId::parse(&id.to_string()), Ok(id));
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = TrailId::parse("corr_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidTrailId::WrongPrefix {
                expected: "trail",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        assert!(matches!(
            TrailId::parse("not-a-valid-typeid"),
            Err(InvalidTrailId::Parse(_))
        ));
    }

    #[test]
    fn trail_ids_are_unique() {
        assert_ne!(TrailId::new(), TrailId::new());
    }

    #[test]
    fn trail_id_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let id = TrailId::new();
        set.insert(id.clone());

        assert!(set.contains(&id));
    }

    #[test]
    fn serialization_roundtrip() {
        let id = TrailId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: TrailId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn invalid_trail_id_display_names_the_expected_prefix() {
        let err = TrailId::parse("corr_01h455vb4pex5vsknk084sn02q").unwrap_err();
        let text = err.to_string();
        assert!(text.contains("trail"), "{text}");
        assert!(text.contains("corr"), "{text}");
    }
}
