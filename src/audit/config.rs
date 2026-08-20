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

    /// Makes sure the trail itself exists, creating an empty one if it does
    /// not.
    ///
    /// Called once at launch, and it is what makes an *absent* trail
    /// meaningful. Without it, "the file is not there" is ambiguous between
    /// an armed runtime that has simply not recorded anything yet and a trail
    /// somebody deleted — and the two readers of the chain disagree about
    /// which: [`ActonAI::audit_head`](crate::ActonAI::audit_head) answers with
    /// the genesis head while `acton-ai audit verify` cannot find a file to
    /// read. Creating it up front collapses that ambiguity: an armed but idle
    /// trail is an empty file that verifies as genesis, and a missing file
    /// means somebody removed it.
    ///
    /// It also moves an unwritable destination to launch time, which is the
    /// same reason [`ensure_parent_dir`](Self::ensure_parent_dir) runs there:
    /// a trail that cannot be written is a configuration failure, not
    /// something to discover one entry at a time.
    ///
    /// An existing trail is left exactly as it is — this never truncates.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if the parent directory cannot be
    /// created or the trail cannot be opened for appending.
    pub fn ensure_trail_exists(&self) -> Result<(), ActonAIError> {
        self.ensure_parent_dir()?;

        std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)
            .map(|_| ())
            .map_err(|error| {
                ActonAIError::configuration(
                    "audit.path",
                    format!(
                        "could not open the audit trail {} for appending: {error}",
                        self.path.display()
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
    fn ensure_trail_exists_creates_an_empty_trail() {
        // An armed-but-idle trail must be an empty file rather than no file,
        // or `audit verify` reports a missing trail while the runtime reports
        // a perfectly good genesis head.
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("nested").join("audit.jsonl");
        let config = AuditConfig::new(&path);

        config.ensure_trail_exists().expect("arming must succeed");

        assert!(path.is_file(), "the trail must exist after arming");
        assert_eq!(
            std::fs::read_to_string(&path).expect("readable"),
            "",
            "a fresh trail must be empty, not seeded"
        );
    }

    #[test]
    fn ensure_trail_exists_never_truncates_an_existing_trail() {
        // The chain is append-only. Re-arming on restart must not erase what
        // the previous run recorded — that would be the framework itself
        // destroying the evidence.
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("audit.jsonl");
        std::fs::write(&path, "existing entry\n").expect("writes");
        let config = AuditConfig::new(&path);

        config
            .ensure_trail_exists()
            .expect("re-arming must succeed");

        assert_eq!(
            std::fs::read_to_string(&path).expect("readable"),
            "existing entry\n"
        );
    }

    #[test]
    fn ensure_trail_exists_reports_an_unwritable_destination() {
        // A directory where the trail should be is a configuration mistake
        // that must fail the launch, not every append.
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("audit.jsonl");
        std::fs::create_dir(&path).expect("a directory in the trail's place");
        let config = AuditConfig::new(&path);

        let error = config
            .ensure_trail_exists()
            .expect_err("a directory cannot be appended to");

        assert!(
            error.to_string().contains("audit trail"),
            "the error must name what failed, got: {error}"
        );
    }

    #[test]
    fn the_default_path_names_the_trail_file() {
        assert!(default_audit_path().ends_with(DEFAULT_AUDIT_FILE));
    }
}
