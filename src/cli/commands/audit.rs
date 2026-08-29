//! The `audit` command — verify the tamper-evident tool-invocation trail.
//!
//! `audit verify` walks the hash chain from its genesis entry and reports the
//! first broken link, or the head it verified. It reads the file and nothing
//! else: no runtime is launched, no provider is contacted, and the trail is
//! never written to. That matters, because the whole point of the check is
//! that it can be run by somebody who is not trusted to modify anything.

use crate::audit::{parse_entries, verify_chain, AuditEntry, ChainBreak, ChainHead};
use crate::cli::error::{CliError, CliErrorKind};
use crate::cli::output::{OutputMode, OutputWriter};
use crate::config;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Audit trail subcommands.
#[derive(Debug, clap::Args)]
pub struct AuditArgs {
    #[command(subcommand)]
    pub command: AuditCommand,
}

/// Available audit subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum AuditCommand {
    /// Verify the trail's hash chain and report the first broken link.
    #[command(long_about = "Walk the audit trail's hash chain from its first entry \
                      and report where, if anywhere, it stops adding up.\n\n\
                      The trail is found from --file, then the `[audit] path` \
                      config key, then the default data directory. Exit code \
                      is 0 when the chain verifies and 3 when it does not, so \
                      this is safe to put in a cron job or a compliance check.")]
    Verify {
        /// Path to the trail (overrides the configured path).
        #[arg(long)]
        file: Option<PathBuf>,
    },
}

const SCHEMA_VERSION: u32 = 1;

/// What `audit verify` reports when the chain holds.
#[derive(Debug, Serialize)]
struct VerifyReport {
    schema_version: u32,
    path: String,
    verified: bool,
    entries: u64,
    /// The sequence number of the last entry, `0` for an empty trail.
    head_sequence: u64,
    /// The hash of the last entry, or the genesis hash for an empty trail.
    head_hash: String,
    /// The identity the chain is sealed under; `null` for a trail written
    /// entirely before trails had identities, or one with no entries yet.
    trail_id: Option<String>,
}

impl VerifyReport {
    fn new(path: &Path, head: &ChainHead) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            path: path.to_string_lossy().into_owned(),
            verified: true,
            entries: head.entries,
            head_sequence: head.sequence,
            head_hash: head.hash.clone(),
            trail_id: head.trail_id.as_ref().map(ToString::to_string),
        }
    }
}

/// The identity as `audit verify` prints it: the id, or `unidentified`.
fn describe_trail(head: &ChainHead) -> String {
    head.trail_id
        .as_ref()
        .map_or_else(|| "unidentified".to_string(), ToString::to_string)
}

/// Execute the audit command.
///
/// # Errors
///
/// Returns a configuration error if the trail cannot be found or read, and
/// [`CliErrorKind::AuditChainBroken`] — exit code 3 — if it reads fine but
/// does not verify.
pub fn execute(
    args: &AuditArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
) -> Result<(), CliError> {
    match &args.command {
        AuditCommand::Verify { file } => verify(file.as_ref(), output, config_path),
    }
}

fn verify(
    file: Option<&PathBuf>,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
) -> Result<(), CliError> {
    let path = resolve_trail_path(file, config_path);
    let entries = read_trail(&path)?;

    match verify_chain(&entries) {
        Ok(head) => report_intact(output, &path, &head),
        Err(break_found) => Err(chain_broken(path, break_found)),
    }
}

/// Finds the trail: `--file`, then `[audit] path`, then the default location.
///
/// A config file that cannot be parsed is not fatal here. Verification is the
/// one command that has to keep working when the deployment around it does
/// not, so a broken config falls back to the default path rather than
/// refusing to look.
fn resolve_trail_path(file: Option<&PathBuf>, config_path: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = file {
        return path.clone();
    }

    let configured = config_path
        .cloned()
        .or_else(|| config::search_paths().into_iter().find(|p| p.exists()))
        .and_then(|path| config::from_path(&path).ok())
        .and_then(|cfg| cfg.audit)
        .and_then(|audit| audit.path);

    configured.unwrap_or_else(crate::audit::default_audit_path)
}

/// Reads the trail, turning both "no file" and "not a trail" into advice.
fn read_trail(path: &Path) -> Result<Vec<AuditEntry>, CliError> {
    let contents = std::fs::read_to_string(path).map_err(|error| {
        CliError::configuration(format!(
            "could not read the audit trail at {}: {error}",
            path.display()
        ))
    })?;

    parse_entries(&contents).map_err(|error| {
        CliError::configuration(format!(
            "the audit trail at {} is not readable as a trail: {error}",
            path.display()
        ))
    })
}

fn report_intact(output: &OutputWriter, path: &Path, head: &ChainHead) -> Result<(), CliError> {
    match output.mode() {
        OutputMode::Json => output.write_json(&VerifyReport::new(path, head))?,
        OutputMode::Plain => {
            output.write_line(&format!("audit trail: {}", path.display()))?;
            output.write_line(&format!("trail:       {}", describe_trail(head)))?;
            output.write_line(&format!("entries:     {}", head.entries))?;
            output.write_line(&format!("head:        {}", head.hash))?;
            output.write_line("chain:       verified")?;
        }
    }

    Ok(())
}

fn chain_broken(path: PathBuf, break_found: ChainBreak) -> CliError {
    CliError {
        kind: CliErrorKind::AuditChainBroken { path, break_found },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrailId;

    #[test]
    fn the_report_names_the_trail_or_says_it_is_unidentified() {
        let legacy = ChainHead::empty();
        assert_eq!(describe_trail(&legacy), "unidentified");
        let report = VerifyReport::new(Path::new("/tmp/audit.jsonl"), &legacy);
        assert_eq!(report.trail_id, None);

        let trail = TrailId::new();
        let identified = ChainHead {
            trail_id: Some(trail.clone()),
            ..ChainHead::empty()
        };
        assert_eq!(describe_trail(&identified), trail.to_string());
        let report = VerifyReport::new(Path::new("/tmp/audit.jsonl"), &identified);
        assert_eq!(report.trail_id, Some(trail.to_string()));
    }

    #[test]
    fn an_explicit_file_wins_over_everything_else() {
        let explicit = PathBuf::from("/tmp/explicit.jsonl");
        assert_eq!(resolve_trail_path(Some(&explicit), None), explicit);
    }

    #[test]
    fn a_missing_trail_is_reported_as_a_configuration_error() {
        let error = read_trail(Path::new("/nonexistent/audit.jsonl"))
            .expect_err("a missing trail must be reported");

        assert!(matches!(error.kind, CliErrorKind::Configuration(_)));
    }

    #[test]
    fn a_file_that_is_not_a_trail_says_so_rather_than_claiming_tampering() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("audit.jsonl");
        std::fs::write(&path, "this is not JSON\n").expect("writes");

        let error = read_trail(&path).expect_err("a non-trail must be reported");

        // Deliberately not `AuditChainBroken`: "I cannot read this" and "this
        // has been altered" are different findings and only one of them is an
        // incident.
        assert!(matches!(error.kind, CliErrorKind::Configuration(_)));
    }
}
