//! Tool name identifier type using TypeID format.
//!
//! ToolName provides a validated identifier for tools in the registry.
//! Format: `tool_01h455vb4pex5vsknk084sn02q`

use mti::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A validated tool name identifier.
///
/// Uses TypeID format for human-readable, globally unique tool identifiers.
/// Example: `tool_01h455vb4pex5vsknk084sn02q`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolName(MagicTypeId);

/// Error returned when attempting to create an invalid tool name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidToolName {
    /// TypeID parsing failed
    Parse(String),
    /// Wrong prefix (expected "tool")
    WrongPrefix {
        /// The expected prefix
        expected: &'static str,
        /// The actual prefix found
        actual: String,
    },
}

impl fmt::Display for InvalidToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid tool name: {e}"),
            Self::WrongPrefix { expected, actual } => {
                write!(f, "expected prefix '{expected}', got '{actual}'")
            }
        }
    }
}

impl std::error::Error for InvalidToolName {}

impl ToolName {
    /// The TypeID prefix for tool name identifiers.
    pub const PREFIX: &'static str = "tool";

    /// Creates a new tool name with a fresh UUIDv7 (time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Self::PREFIX.create_type_id::<V7>())
    }

    /// Parses a tool name from a string, validating the prefix.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse
    ///
    /// # Returns
    ///
    /// A valid `ToolName` if the string is a valid TypeID with the "tool" prefix.
    ///
    /// # Errors
    ///
    /// Returns `InvalidToolName::Parse` if the string is not a valid TypeID format.
    /// Returns `InvalidToolName::WrongPrefix` if the TypeID has a different prefix.
    pub fn parse(s: &str) -> Result<Self, InvalidToolName> {
        let id = MagicTypeId::from_str(s).map_err(|e| InvalidToolName::Parse(e.to_string()))?;

        let prefix = id.prefix().as_str();
        if prefix != Self::PREFIX {
            return Err(InvalidToolName::WrongPrefix {
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

impl Default for ToolName {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for ToolName {
    type Err = InvalidToolName;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl AsRef<MagicTypeId> for ToolName {
    fn as_ref(&self) -> &MagicTypeId {
        &self.0
    }
}

impl Serialize for ToolName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ToolName {
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
    fn new_creates_valid_tool_name() {
        let name = ToolName::new();
        assert!(name.to_string().starts_with("tool_"));
    }

    #[test]
    fn parse_valid_tool_name() {
        let name_str = ToolName::new().to_string();
        let parsed = ToolName::parse(&name_str);
        assert!(parsed.is_ok());
    }

    #[test]
    fn parse_wrong_prefix_fails() {
        let result = ToolName::parse("agent_01h455vb4pex5vsknk084sn02q");
        assert!(matches!(
            result,
            Err(InvalidToolName::WrongPrefix {
                expected: "tool",
                ..
            })
        ));
    }

    #[test]
    fn parse_invalid_format_fails() {
        let result = ToolName::parse("not-a-valid-typeid");
        assert!(matches!(result, Err(InvalidToolName::Parse(_))));
    }

    #[test]
    fn tool_names_are_unique() {
        let name1 = ToolName::new();
        let name2 = ToolName::new();
        assert_ne!(name1, name2);
    }

    #[test]
    fn tool_name_can_be_used_as_hash_key() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let name = ToolName::new();
        set.insert(name.clone());

        assert!(set.contains(&name));
    }

    #[test]
    fn display_format() {
        let name = ToolName::new();
        let s = name.to_string();
        assert!(s.starts_with("tool_"));
    }

    #[test]
    fn default_creates_new_name() {
        let name1 = ToolName::default();
        let name2 = ToolName::default();
        assert_ne!(name1, name2);
    }

    #[test]
    fn serialization_roundtrip() {
        let name = ToolName::new();
        let json = serde_json::to_string(&name).unwrap();
        let deserialized: ToolName = serde_json::from_str(&json).unwrap();
        assert_eq!(name, deserialized);
    }

    #[test]
    fn error_display_for_parse() {
        let err = InvalidToolName::Parse("some error".to_string());
        let display = err.to_string();
        assert!(display.contains("invalid tool name"));
        assert!(display.contains("some error"));
    }

    #[test]
    fn error_display_for_wrong_prefix() {
        let err = InvalidToolName::WrongPrefix {
            expected: "tool",
            actual: "agent".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("tool"));
        assert!(display.contains("agent"));
    }
}
