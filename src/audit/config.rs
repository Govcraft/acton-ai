//! Resolved audit settings.
//!
//! The TOML-facing shape lives in [`crate::config::AuditFileConfig`]; this is
//! what it resolves into once the path is settled and the patterns are fixed.

use crate::audit::redact::Redactor;
use crate::error::ActonAIError;
use std::path::{Path, PathBuf};

/// The audit trail's file name under the data directory, when no path is set.
pub const DEFAULT_AUDIT_FILE: &str = "audit.jsonl";

/// Audit settings in force.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditConfig {
    /// Where the JSONL trail is appended.
    path: PathBuf,
    /// What gets redacted out of arguments before they are written.
    redactor: Redactor,
}

impl AuditConfig {
    /// Settings writing to `path`, redacting the default key fragments.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            redactor: Redactor::with_defaults(),
        }
    }

    /// Replaces the redaction patterns.
    #[must_use]
    pub fn with_redactor(mut self, redactor: Redactor) -> Self {
        self.redactor = redactor;
        self
    }

    /// Where the trail is written.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The redactor applied to arguments before they are recorded.
    #[must_use]
    pub fn redactor(&self) -> &Redactor {
        &self.redactor
    }

    /// Makes sure the trail's directory exists.
    ///
    /// Called once at launch rather than per append, so a misconfigured path
    /// fails the launch instead of quietly dropping every entry.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if the parent directory cannot be
    /// created.
    pub fn ensure_parent_dir(&self) -> Result<(), ActonAIError> {
        let Some(parent) = self.path.parent() else {
            return Ok(());
        };
        if parent.as_os_str().is_empty() || parent.exists() {
            return Ok(());
        }

        std::fs::create_dir_all(parent).map_err(|error| {
            ActonAIError::configuration(
                "audit.path",
                format!(
                    "could not create the audit trail's directory {}: {error}",
                    parent.display()
                ),
            )
        })
    }
}

/// Where the trail goes when `[audit]` names no path.
///
/// `$XDG_DATA_HOME/acton-ai/audit.jsonl`, matching where the CLI already puts
/// its database, falling back to the current directory when there is no data
/// directory to speak of.
#[must_use]
pub fn default_audit_path() -> PathBuf {
    dirs::data_dir().map_or_else(
        || PathBuf::from(DEFAULT_AUDIT_FILE),
        |dir| dir.join("acton-ai").join(DEFAULT_AUDIT_FILE),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_config_defaults_to_redacting_the_usual_suspects() {
        let config = AuditConfig::new("/tmp/audit.jsonl");
        assert!(config.redactor().matches_key("api_key"));
        assert_eq!(config.path(), Path::new("/tmp/audit.jsonl"));
    }

    #[test]
    fn ensure_parent_dir_creates_a_missing_directory() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("nested").join("deeper").join("audit.jsonl");
        let config = AuditConfig::new(&path);

        config
            .ensure_parent_dir()
            .expect("creating the parent must succeed");

        assert!(path.parent().expect("a parent").is_dir());
    }

    #[test]
    fn ensure_parent_dir_is_content_with_a_bare_file_name() {
        let config = AuditConfig::new("audit.jsonl");
        assert!(config.ensure_parent_dir().is_ok());
    }

    #[test]
    fn the_default_path_names_the_trail_file() {
        assert!(default_audit_path().ends_with(DEFAULT_AUDIT_FILE));
    }
}
