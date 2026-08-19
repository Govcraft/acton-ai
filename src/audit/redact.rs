//! Redaction of secret-bearing arguments before they reach the trail.
//!
//! Pure, and applied at the boundary: the prompt loop redacts before it sends
//! the record, so an unredacted argument never enters the writer's state and
//! cannot be written by a later change to the actor.
//!
//! This is key-based, not value-based. A value is replaced when its *key*
//! looks secret-bearing, because guessing at values is how a redactor either
//! misses a token that does not look like one or mangles ordinary prose.

use serde_json::{Map, Value};

/// What a redacted value is replaced with.
///
/// The same literal `acton-ai config` already uses, so a reader who has seen
/// one has seen both.
pub const REDACTED: &str = "[redacted]";

/// The key fragments treated as secret-bearing when none are configured.
pub const DEFAULT_REDACT_PATTERNS: &[&str] = &[
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "private_key",
];

/// Replaces secret-bearing values in tool arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Redactor {
    /// Lowercased key fragments. A key containing any of them is redacted.
    patterns: Vec<String>,
}

impl Default for Redactor {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl Redactor {
    /// A redactor using [`DEFAULT_REDACT_PATTERNS`].
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            patterns: DEFAULT_REDACT_PATTERNS
                .iter()
                .map(|p| (*p).to_string())
                .collect(),
        }
    }

    /// A redactor using exactly these key fragments, replacing the defaults.
    ///
    /// Matching is case-insensitive and by substring, so `key` catches
    /// `api_key`, `KeyMaterial` and `key`.
    #[must_use]
    pub fn new<I, S>(patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self {
            patterns: patterns
                .into_iter()
                .map(|p| p.as_ref().to_lowercase())
                .collect(),
        }
    }

    /// The key fragments this redactor matches on.
    #[must_use]
    pub fn patterns(&self) -> &[String] {
        &self.patterns
    }

    /// Whether a key names something that should be redacted.
    #[must_use]
    pub fn matches_key(&self, key: &str) -> bool {
        let lowered = key.to_lowercase();
        self.patterns
            .iter()
            .any(|pattern| lowered.contains(pattern.as_str()))
    }

    /// Returns `value` with every secret-bearing field replaced.
    ///
    /// Recurses through objects and arrays, so a secret nested inside a
    /// structured argument is caught too. Non-object values are returned
    /// unchanged: with no key to judge, there is nothing to go on.
    #[must_use]
    pub fn redact(&self, value: &Value) -> Value {
        match value {
            Value::Object(fields) => Value::Object(self.redact_object(fields)),
            Value::Array(items) => {
                Value::Array(items.iter().map(|item| self.redact(item)).collect())
            }
            other => other.clone(),
        }
    }

    /// Redacts one object's fields.
    fn redact_object(&self, fields: &Map<String, Value>) -> Map<String, Value> {
        fields
            .iter()
            .map(|(key, value)| {
                if self.matches_key(key) {
                    (key.clone(), Value::String(REDACTED.to_string()))
                } else {
                    (key.clone(), self.redact(value))
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_secret_bearing_key_has_its_value_replaced() {
        let redactor = Redactor::with_defaults();
        let redacted = redactor.redact(&json!({"api_key": "sk-live-123", "model": "gpt-4"}));

        assert_eq!(redacted, json!({"api_key": REDACTED, "model": "gpt-4"}));
    }

    #[test]
    fn matching_is_case_insensitive_and_by_substring() {
        let redactor = Redactor::with_defaults();
        let redacted = redactor.redact(&json!({
            "API_KEY": "a",
            "userToken": "b",
            "Authorization": "c",
        }));

        assert_eq!(
            redacted,
            json!({"API_KEY": REDACTED, "userToken": REDACTED, "Authorization": REDACTED})
        );
    }

    #[test]
    fn nested_objects_and_arrays_are_reached() {
        let redactor = Redactor::with_defaults();
        let redacted = redactor.redact(&json!({
            "outer": {"inner": {"password": "hunter2", "keep": 1}},
            "list": [{"secret": "s"}, {"fine": "f"}],
        }));

        assert_eq!(
            redacted,
            json!({
                "outer": {"inner": {"password": REDACTED, "keep": 1}},
                "list": [{"secret": REDACTED}, {"fine": "f"}],
            })
        );
    }

    #[test]
    fn a_redacted_key_does_not_have_its_subtree_walked_into() {
        // The whole value goes, not just the secret-looking leaves inside it.
        let redactor = Redactor::with_defaults();
        let redacted = redactor.redact(&json!({"credentials": {"user": "alice", "pass": "p"}}));

        assert_eq!(redacted, json!({"credentials": REDACTED}));
    }

    #[test]
    fn configured_patterns_replace_the_defaults() {
        let redactor = Redactor::new(["ssn"]);
        let redacted = redactor.redact(&json!({"ssn": "1", "api_key": "2"}));

        assert_eq!(
            redacted,
            json!({"ssn": REDACTED, "api_key": "2"}),
            "a configured pattern list is exhaustive, not additive"
        );
    }

    #[test]
    fn an_empty_pattern_list_redacts_nothing() {
        let redactor = Redactor::new(Vec::<String>::new());
        let value = json!({"api_key": "sk-live-123"});

        assert_eq!(redactor.redact(&value), value);
    }

    #[test]
    fn values_without_keys_are_left_alone() {
        let redactor = Redactor::with_defaults();
        assert_eq!(redactor.redact(&json!("token")), json!("token"));
        assert_eq!(redactor.redact(&json!(42)), json!(42));
        assert_eq!(redactor.redact(&Value::Null), Value::Null);
    }
}
