//! Resolved audit settings.
//!
//! The TOML-facing shape lives in [`crate::config::AuditFileConfig`]; this is
//! what it resolves into once the path is settled and the patterns are fixed.

use crate::audit::redact::Redactor;
use crate::error::ActonAIError;
use crate::types::TrailId;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

/// The audit trail's file name under the data directory, when no path is set.
pub const DEFAULT_AUDIT_FILE: &str = "audit.jsonl";

/// What an append promises before the turn moves on.
///
/// The choice is between two failure stories. Under `BestEffort` a tool that
/// ran is a tool that ran, and a trail that could not record it is an error
/// to log. Under `Strict` the trail is the authority: an entry is synced to
/// disk and acknowledged before the loop continues, and once one append has
/// failed, every tool that could change the world is refused until the
/// process is restarted over a repaired trail. Tools declared idempotent keep
/// running either way — refusing a read protects nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AuditDurability {
    /// Append and flush; a failed append is logged and the turn continues.
    #[default]
    BestEffort,
    /// Append, fsync, and acknowledge; a failed append degrades the writer
    /// and non-idempotent tool calls are refused for the rest of the process.
    Strict,
}

impl AuditDurability {
    /// Whether every append must be acknowledged and mutations are refused
    /// once one has failed.
    #[must_use]
    pub fn is_strict(self) -> bool {
        matches!(self, Self::Strict)
    }
}

impl fmt::Display for AuditDurability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::BestEffort => "best_effort",
            Self::Strict => "strict",
        };
        f.write_str(name)
    }
}

/// Audit settings in force.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditConfig {
    /// Where the JSONL trail is appended.
    path: PathBuf,
    /// What gets redacted out of arguments before they are written.
    redactor: Redactor,
    /// The principal on whose behalf tool calls run.
    user: Option<String>,
    /// What an append promises.
    durability: AuditDurability,
}

impl AuditConfig {
    /// Settings writing to `path`, redacting the default key fragments.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            redactor: Redactor::with_defaults(),
            user: None,
            durability: AuditDurability::default(),
        }
    }

    /// Sets what an append promises. Best effort unless this is called.
    #[must_use]
    pub fn with_durability(mut self, durability: AuditDurability) -> Self {
        self.durability = durability;
        self
    }

    /// What an append promises.
    #[must_use]
    pub fn durability(&self) -> AuditDurability {
        self.durability
    }

    /// Whether the trail is strict: appends are acknowledged, and a failed
    /// one refuses every later non-idempotent tool call.
    #[must_use]
    pub fn is_strict(&self) -> bool {
        self.durability.is_strict()
    }

    /// Replaces the redaction patterns.
    #[must_use]
    pub fn with_redactor(mut self, redactor: Redactor) -> Self {
        self.redactor = redactor;
        self
    }

    /// Stamps `user` onto every entry written by this audit session.
    #[must_use]
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
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

    /// The principal stamped onto entries, when configured.
    #[must_use]
    pub fn user(&self) -> Option<&str> {
        self.user.as_deref()
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

    /// Where the trail's identity is kept: the trail path with `.trail`
    /// appended, so `audit.jsonl` is identified by `audit.jsonl.trail`.
    ///
    /// A sidecar rather than a header line in the trail itself, so the trail
    /// stays plain JSONL of entries and an empty trail stays empty. The chain
    /// carries the identity too — every entry is sealed under it — so the
    /// sidecar is the identity's *first* home, from before the first entry,
    /// and the two are checked against each other at spawn.
    #[must_use]
    pub fn trail_id_path(&self) -> PathBuf {
        let mut name = self.path.file_name().map_or_else(
            || std::ffi::OsString::from(DEFAULT_AUDIT_FILE),
            std::ffi::OsStr::to_os_string,
        );
        name.push(TRAIL_ID_SUFFIX);
        self.path.with_file_name(name)
    }
}

/// What is appended to the trail's file name to name its identity sidecar.
pub const TRAIL_ID_SUFFIX: &str = ".trail";

/// Reads a trail's identity sidecar.
///
/// # Errors
///
/// Returns `Ok(None)` when there is no sidecar — a trail from before
/// identities, or a first run. Any other IO failure is returned as is, and a
/// sidecar that does not hold a trail id is `InvalidData`: a file that claims
/// to be the identity and is not one is a finding, not something to overwrite.
pub fn read_trail_id(path: &Path) -> Result<Option<TrailId>, std::io::Error> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };

    TrailId::parse(contents.trim())
        .map(Some)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))
}

/// Writes a trail's identity sidecar, once.
///
/// The id is written to a temporary file beside the destination and then
/// hard-linked into place, so a reader never sees a partial file and an
/// existing sidecar is never replaced: two identities for one trail is
/// exactly the confusion the sidecar exists to rule out.
///
/// # Errors
///
/// Returns `AlreadyExists` if a sidecar is already there, and any other IO
/// failure as is.
pub fn write_trail_id(path: &Path, id: &TrailId) -> Result<(), std::io::Error> {
    use std::io::Write as _;

    let mut staged_name = path.as_os_str().to_os_string();
    staged_name.push(format!(".{}.tmp", std::process::id()));
    let staged = PathBuf::from(staged_name);

    let mut file = std::fs::File::create(&staged)?;
    writeln!(file, "{id}")?;
    file.sync_all()?;
    drop(file);

    let linked = std::fs::hard_link(&staged, path);
    // The staged copy is done with either way; a failure to remove it is not
    // a failure to write the identity.
    let _ = std::fs::remove_file(&staged);
    linked
}

/// A trail whose two homes for its identity disagree.
///
/// The ids are carried rendered: this is a report, and what matters is that
/// the refusal names both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrailIdConflict {
    /// What the sidecar says.
    pub sidecar: String,
    /// What the chain's entries are sealed under.
    pub chain: String,
}

impl fmt::Display for TrailIdConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the chain is sealed under trail {} but the sidecar says {}",
            self.chain, self.sidecar
        )
    }
}

impl std::error::Error for TrailIdConflict {}

/// Settles which identity a trail has, from its sidecar and its chain.
///
/// Pure. The four agreeing cases: both name the same trail, only the sidecar
/// names one (a trail that has not been appended to since it was identified,
/// or a legacy chain that gained a sidecar), only the chain names one (the
/// sidecar was lost; it is rewritten from the chain), or neither does (a
/// fresh trail, or a legacy one, which is given a new identity). Two
/// different identities is a conflict: the trail cannot be both, and the
/// caller must refuse rather than pick.
///
/// # Errors
///
/// Returns [`TrailIdConflict`] when the sidecar and the chain name different
/// trails.
pub fn resolve_trail_id(
    sidecar: Option<TrailId>,
    chain: Option<TrailId>,
) -> Result<TrailId, TrailIdConflict> {
    match (sidecar, chain) {
        (Some(sidecar), Some(chain)) if sidecar == chain => Ok(sidecar),
        (Some(sidecar), Some(chain)) => Err(TrailIdConflict {
            sidecar: sidecar.to_string(),
            chain: chain.to_string(),
        }),
        (Some(sidecar), None) => Ok(sidecar),
        (None, Some(chain)) => Ok(chain),
        (None, None) => Ok(TrailId::new()),
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
        assert_eq!(config.user(), None);
        assert_eq!(config.durability(), AuditDurability::BestEffort);
        assert!(!config.is_strict());
    }

    #[test]
    fn a_config_can_be_made_strict() {
        let config = AuditConfig::new("/tmp/audit.jsonl").with_durability(AuditDurability::Strict);
        assert!(config.is_strict());
        assert_eq!(config.durability(), AuditDurability::Strict);
    }

    #[test]
    fn durability_reads_and_writes_as_snake_case() {
        assert_eq!(
            serde_json::from_str::<AuditDurability>("\"strict\"").expect("parses"),
            AuditDurability::Strict
        );
        assert_eq!(
            serde_json::to_string(&AuditDurability::BestEffort).expect("serializes"),
            "\"best_effort\""
        );
        assert_eq!(AuditDurability::Strict.to_string(), "strict");
        assert_eq!(AuditDurability::default(), AuditDurability::BestEffort);
    }

    #[test]
    fn a_config_can_name_the_acting_user() {
        let config = AuditConfig::new("/tmp/audit.jsonl").with_user("acct:alice");
        assert_eq!(config.user(), Some("acct:alice"));
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

    #[test]
    fn the_sidecar_sits_beside_the_trail_with_a_trail_suffix() {
        let config = AuditConfig::new("/var/log/acton-ai/audit.jsonl");
        assert_eq!(
            config.trail_id_path(),
            PathBuf::from("/var/log/acton-ai/audit.jsonl.trail")
        );

        let bare = AuditConfig::new("audit.jsonl");
        assert_eq!(bare.trail_id_path(), PathBuf::from("audit.jsonl.trail"));
    }

    #[test]
    fn a_missing_sidecar_reads_as_no_identity() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar = dir.path().join("audit.jsonl.trail");
        assert_eq!(read_trail_id(&sidecar).unwrap(), None);
    }

    #[test]
    fn a_written_sidecar_reads_back_and_is_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar = dir.path().join("audit.jsonl.trail");
        let id = TrailId::new();

        write_trail_id(&sidecar, &id).expect("the first write succeeds");
        assert_eq!(
            std::fs::read_to_string(&sidecar).unwrap(),
            format!("{id}\n"),
            "one line, newline-terminated, nothing else"
        );
        assert_eq!(read_trail_id(&sidecar).unwrap(), Some(id.clone()));
        assert!(
            std::fs::read_dir(dir.path()).unwrap().count() == 1,
            "no staging file is left behind"
        );

        let error = write_trail_id(&sidecar, &TrailId::new())
            .expect_err("a second identity must not replace the first");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(read_trail_id(&sidecar).unwrap(), Some(id));
    }

    #[test]
    fn a_sidecar_that_is_not_a_trail_id_is_invalid_data() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar = dir.path().join("audit.jsonl.trail");
        std::fs::write(&sidecar, "turn_01h455vb4pex5vsknk084sn02q\n").unwrap();

        let error = read_trail_id(&sidecar).expect_err("a wrong prefix is not an identity");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn identities_that_agree_resolve_to_themselves() {
        let id = TrailId::new();
        assert_eq!(
            resolve_trail_id(Some(id.clone()), Some(id.clone())),
            Ok(id.clone())
        );
        assert_eq!(resolve_trail_id(Some(id.clone()), None), Ok(id.clone()));
        assert_eq!(resolve_trail_id(None, Some(id.clone())), Ok(id));
    }

    #[test]
    fn a_trail_with_no_identity_anywhere_is_given_a_fresh_one() {
        let first = resolve_trail_id(None, None).unwrap();
        let second = resolve_trail_id(None, None).unwrap();
        assert_ne!(first, second, "each fresh trail is its own");
    }

    #[test]
    fn disagreeing_identities_are_a_conflict_naming_both() {
        let sidecar = TrailId::new();
        let chain = TrailId::new();

        let conflict = resolve_trail_id(Some(sidecar.clone()), Some(chain.clone()))
            .expect_err("two identities for one trail must be refused");

        assert_eq!(
            conflict,
            TrailIdConflict {
                sidecar: sidecar.to_string(),
                chain: chain.to_string()
            }
        );
        let text = conflict.to_string();
        assert!(text.contains(&sidecar.to_string()), "{text}");
        assert!(text.contains(&chain.to_string()), "{text}");
    }
}
