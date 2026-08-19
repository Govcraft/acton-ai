//! The `status`, `pause`, `resume`, and `drain` commands — talking to a
//! running process over its control socket.
//!
//! These are the only commands that do not start a runtime of their own. They
//! connect to one that is already running, which is what makes them usable
//! during an incident: a wedged provider or an exhausted budget does not stop
//! `acton-ai status` from answering.
//!
//! # Finding the socket
//!
//! In order: `--socket`, then `socket_path` under `[introspection]` in the
//! config file, then a scan of the default directory. The scan exists because
//! the default socket name carries the PID, which nobody can guess — and it
//! refuses to choose when it finds more than one, since picking the wrong
//! process to `drain` is worse than asking.

use std::path::{Path, PathBuf};

use crate::cli::error::CliError;
use crate::cli::output::OutputWriter;
use crate::config;

/// Options shared by every command that talks to a control socket.
#[derive(Debug, Clone, clap::Args)]
pub struct SocketArgs {
    /// Path to the process's introspection socket.
    ///
    /// Defaults to the `[introspection] socket_path` in the config file, or
    /// to the single socket found in the default runtime directory.
    #[arg(long, value_name = "PATH")]
    pub socket: Option<PathBuf>,
}

/// Options for `acton-ai status`.
#[derive(Debug, clap::Args)]
pub struct StatusArgs {
    #[command(flatten)]
    pub socket: SocketArgs,
}

/// Options for `acton-ai pause`.
#[derive(Debug, clap::Args)]
pub struct PauseArgs {
    #[command(flatten)]
    pub socket: SocketArgs,
}

/// Options for `acton-ai resume`.
#[derive(Debug, clap::Args)]
pub struct ResumeArgs {
    #[command(flatten)]
    pub socket: SocketArgs,
}

/// Options for `acton-ai drain`.
#[derive(Debug, clap::Args)]
pub struct DrainArgs {
    #[command(flatten)]
    pub socket: SocketArgs,

    /// Wait until every in-flight turn has finished before returning.
    ///
    /// Without this, `drain` returns as soon as the process has stopped
    /// admitting new turns, which is what a supervisor's `ExecStop` wants.
    #[arg(long)]
    pub wait: bool,

    /// Seconds to wait with `--wait` before giving up. 0 waits indefinitely.
    #[arg(long, value_name = "SECONDS", default_value_t = DEFAULT_DRAIN_TIMEOUT_SECS)]
    pub timeout: u64,
}

/// How long `drain --wait` waits before reporting the turns still running.
///
/// Five minutes is longer than almost any single turn and shorter than a
/// deploy window. A caller who knows their own worst case passes `--timeout`.
pub const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 300;

/// How often `drain --wait` re-asks for the in-flight count.
///
/// Polling, because the wire protocol is request/response — there is no
/// "tell me when you are done" message, and adding one would put a
/// long-lived subscription in the path of a shutdown.
#[cfg(feature = "ipc")]
const DRAIN_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

/// Executes `acton-ai status`.
pub async fn execute_status(
    args: &StatusArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
) -> Result<(), CliError> {
    let socket = resolve_socket(args.socket.socket.as_deref(), config_path)?;
    let report = ask_status(&socket).await?;
    write_status(output, &report)
}

/// Executes `acton-ai pause`.
pub async fn execute_pause(
    args: &PauseArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
) -> Result<(), CliError> {
    let socket = resolve_socket(args.socket.socket.as_deref(), config_path)?;
    let ack = ask_pause(&socket).await?;
    write_ack(output, &ack, "no longer admitting new turns")
}

/// Executes `acton-ai resume`.
pub async fn execute_resume(
    args: &ResumeArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
) -> Result<(), CliError> {
    let socket = resolve_socket(args.socket.socket.as_deref(), config_path)?;
    let ack = ask_resume(&socket).await?;
    write_ack(output, &ack, "admitting new turns again")
}

/// Executes `acton-ai drain`.
pub async fn execute_drain(
    args: &DrainArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
) -> Result<(), CliError> {
    let socket = resolve_socket(args.socket.socket.as_deref(), config_path)?;
    let ack = ask_drain(&socket).await?;

    if !args.wait {
        return write_ack(output, &ack, "no longer admitting new turns");
    }

    let ack = wait_for_drain(&socket, ack, args.timeout, output).await?;
    write_ack(output, &ack, "drained")
}

/// Resolves which socket to talk to.
///
/// # Errors
///
/// Returns a configuration error naming everything that was tried when no
/// socket can be chosen — an explicit path that is not there, a config file
/// that names none, an empty runtime directory, or several candidates where
/// only one can be meant.
fn resolve_socket(
    explicit: Option<&Path>,
    config_path: Option<&PathBuf>,
) -> Result<PathBuf, CliError> {
    resolve_socket_in(
        explicit,
        config_path,
        &crate::introspection::default_socket_dir(),
    )
}

/// The whole cascade, with the scan directory supplied.
///
/// Split out so the tests can exercise the three outcomes of the scan against
/// a directory they control. Reading the real `$XDG_RUNTIME_DIR` would make
/// them pass or fail on whether the developer happens to have an agent
/// running, which is not a property of this code.
fn resolve_socket_in(
    explicit: Option<&Path>,
    config_path: Option<&PathBuf>,
    dir: &Path,
) -> Result<PathBuf, CliError> {
    // An explicit path is used even when nothing is there: the operator said
    // where to look, and "no such socket at the path you gave me" is a more
    // useful answer than quietly looking somewhere else.
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }

    if let Some(path) = socket_from_config(config_path) {
        return Ok(path);
    }

    let mut found = list_sockets(dir);
    found.sort();

    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(CliError::configuration(format!(
            "no introspection socket found. Looked at: --socket (not given), \
             `[introspection] socket_path` in {}, and {} (no *.sock entries). \
             A process only listens when it is configured to — add an \
             [introspection] section or call .introspection() on the builder — \
             and if it is running elsewhere, point at it with --socket PATH",
            describe_config_source(config_path),
            dir.display(),
        ))),
        _ => Err(CliError::configuration(format!(
            "{} introspection sockets are live in {}: {}. Each is a different \
             process, so pick one with --socket PATH rather than have this \
             command guess which to pause",
            found.len(),
            dir.display(),
            found
                .iter()
                .map(|p| p
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned())
                .collect::<Vec<_>>()
                .join(", "),
        ))),
    }
}

/// The `[introspection] socket_path` from the config file, if there is one.
fn socket_from_config(config_path: Option<&PathBuf>) -> Option<PathBuf> {
    let path = match config_path {
        Some(path) => path.clone(),
        None => config::search_paths().into_iter().find(|p| p.exists())?,
    };
    let cfg = config::from_path(&path).ok()?;
    cfg.introspection?.socket_path
}

/// How to describe where the config would have been read from, for an error
/// an operator has to act on.
fn describe_config_source(config_path: Option<&PathBuf>) -> String {
    match config_path {
        Some(path) => path.display().to_string(),
        None => match config::search_paths().into_iter().find(|p| p.exists()) {
            Some(path) => path.display().to_string(),
            None => "no config file found on the search path".to_string(),
        },
    }
}

/// Socket-looking entries directly inside `dir`.
///
/// Non-recursive and extension-matched rather than probing each candidate:
/// connecting to every file in a directory to see what answers is a rude
/// thing for a status command to do.
fn list_sockets(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "sock"))
        .collect()
}

/// Renders an acknowledgement of a state change.
#[cfg(feature = "ipc")]
fn write_ack(
    output: &OutputWriter,
    ack: &crate::introspection::AdmissionAck,
    summary: &str,
) -> Result<(), CliError> {
    use crate::cli::output::OutputMode;

    match output.mode() {
        OutputMode::Json => output.write_json(ack)?,
        OutputMode::Plain => {
            output.write_line(&format!(
                "{summary} (admission: {}, {} turn(s) still running)",
                ack.admission, ack.in_flight_turns
            ))?;
        }
    }
    Ok(())
}

/// Renders a status report in whichever form was asked for.
#[cfg(feature = "ipc")]
fn write_status(
    output: &OutputWriter,
    report: &crate::introspection::StatusReport,
) -> Result<(), CliError> {
    use crate::cli::output::OutputMode;

    match output.mode() {
        // The wire type *is* the JSON contract, so `--json` prints it
        // verbatim rather than through a second, drifting representation.
        OutputMode::Json => output.write_json(report)?,
        OutputMode::Plain => write_status_plain(output, report)?,
    }
    Ok(())
}

/// Renders a status report for a human.
#[cfg(feature = "ipc")]
fn write_status_plain(
    output: &OutputWriter,
    report: &crate::introspection::StatusReport,
) -> Result<(), CliError> {
    output.write_line(&format!(
        "{} (pid {}, acton-ai {}), up {}",
        report.app_name,
        report.pid,
        report.crate_version,
        format_uptime(report.uptime_secs),
    ))?;
    output.write_line(&format!(
        "Admission:  {}  ({} turn(s) in flight, {} tool call(s) running)",
        report.admission, report.active_turns, report.in_flight_tool_calls,
    ))?;
    output.write_line(&format!(
        "Turns:      {} started, {} refused",
        report.turns_started, report.turns_refused,
    ))?;
    // Only shown once it has happened: a "0 compacted" line on every status
    // is noise for the majority of runtimes, which never configure compaction.
    if report.turns_compacted > 0 {
        output.write_line(&format!(
            "Compaction: {} histor(ies) compacted mid-turn",
            report.turns_compacted,
        ))?;
    }

    if !report.providers.is_empty() {
        output.write_line("")?;
        output.write_line("Providers:")?;
        for provider in &report.providers {
            let breaker = if provider.circuit_open {
                "  [CIRCUIT OPEN]"
            } else {
                ""
            };
            output.write_line(&format!(
                "  {} ({}): {}{breaker}",
                provider.name, provider.model, provider.health,
            ))?;
            if !provider.failover.is_empty() {
                output.write_line(&format!("    failover: {}", provider.failover.join(" -> ")))?;
            }
        }
    }

    if !report.mcp_servers.is_empty() {
        output.write_line("")?;
        output.write_line("MCP servers:")?;
        for server in &report.mcp_servers {
            // The generation is the restart count, and an operator chasing a
            // flapping server needs it far more than the up/down bit alone.
            output.write_line(&format!(
                "  {}: {} ({}, generation {}, {} restart(s) in window)",
                server.name,
                if server.up { "up" } else { "DOWN" },
                server.state,
                server.generation,
                server.restarts_in_window,
            ))?;
        }
    }

    if let Some(usage) = &report.usage {
        output.write_line("")?;
        output.write_line("Usage:")?;
        output.write_line(&format!(
            "  {} request(s), {} in / {} out tokens",
            usage.requests, usage.input_tokens, usage.output_tokens,
        ))?;
        match usage.total_cost_usd {
            Some(cost) => output.write_line(&format!("  cost: ${cost:.4}"))?,
            // Never "$0.00": unpriced is not free, it is unknown, and the
            // difference matters to whoever is watching the bill.
            None => output.write_line("  cost: unpriced (no [pricing] configured)")?,
        }
        if let Some(budget) = &usage.budget {
            let spent = budget.total_spent_microusd as f64 / 1_000_000.0;
            match budget.total_limit_microusd {
                Some(limit) => output.write_line(&format!(
                    "  budget: ${spent:.4} of ${:.4}{}",
                    limit as f64 / 1_000_000.0,
                    if budget.exceeded { "  [EXCEEDED]" } else { "" },
                ))?,
                None => output.write_line(&format!("  budget: ${spent:.4} spent"))?,
            }
        }
    }

    Ok(())
}

/// Renders seconds as something a human reads at a glance.
///
/// Only the plain renderer needs this; `--json` reports raw seconds, because
/// a machine consumer should never have to parse "1d 1h 1m" back apart.
#[cfg(any(feature = "ipc", test))]
fn format_uptime(secs: u64) -> String {
    let days = secs / 86_400;
    let hours = (secs % 86_400) / 3_600;
    let minutes = (secs % 3_600) / 60;
    let seconds = secs % 60;

    if days > 0 {
        format!("{days}d {hours}h {minutes}m")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

/// The client half of the wire protocol.
#[cfg(feature = "ipc")]
mod client {
    use super::{CliError, Path, DRAIN_POLL_INTERVAL};
    use crate::cli::output::{OutputMode, OutputWriter};
    use crate::introspection::wire::{
        AdmissionAck, Drain, GetStatus, Pause, Resume, StatusReport, INTROSPECTION_ACTOR_NAME,
        SCHEMA_VERSION,
    };
    use acton_reactive::ipc::IpcClient;
    // Via the actor framework's re-export: `tokio` is not a direct dependency
    // of this crate, and adding one just for a sleep would be a second copy of
    // the runtime's version to keep in step.
    use acton_reactive::prelude::tokio;

    /// Connects, or explains why not.
    ///
    /// The two ways this fails need different answers, so they get different
    /// messages: nothing at the path at all usually means the process is not
    /// running, while a path that exists but refuses usually means it died
    /// without cleaning up.
    async fn connect(socket: &Path) -> Result<IpcClient, CliError> {
        IpcClient::connect(socket).await.map_err(|error| {
            if socket.exists() {
                CliError::configuration(format!(
                    "the introspection socket at {} exists but is not answering ({error}). \
                     The process that created it has probably exited without cleaning up; \
                     a new run will replace the file",
                    socket.display(),
                ))
            } else {
                CliError::configuration(format!(
                    "no introspection socket at {} ({error}). Check the process is running, \
                     and that it was configured to listen — an [introspection] section in \
                     its config, or .introspection() on its builder",
                    socket.display(),
                ))
            }
        })
    }

    /// Turns a remote `ask` failure into something actionable.
    fn remote_error(socket: &Path, command: &str, error: &impl std::fmt::Display) -> CliError {
        CliError::configuration(format!(
            "the process at {} did not answer `{command}` ({error}). If it is an older \
             build, it may not expose the introspection surface this client speaks \
             (schema version {SCHEMA_VERSION})",
            socket.display(),
        ))
    }

    /// Warns when the far end speaks a newer protocol than this client.
    ///
    /// Deliberately a warning and not a refusal: the wire types are
    /// additive-only, so a newer server's report still deserializes into what
    /// this build knows. What it cannot do is *show* the newer fields, and an
    /// operator staring at a report missing the field they came for deserves
    /// to know why.
    fn warn_on_newer_schema(output: Option<&OutputWriter>, seen: u32) {
        if seen <= SCHEMA_VERSION {
            return;
        }
        let message = format!(
            "the process speaks introspection schema {seen}; this client knows {SCHEMA_VERSION}, \
             so newer fields are not shown. Upgrade the acton-ai CLI to see them"
        );
        match output {
            Some(output) if output.mode() == OutputMode::Plain => {
                let _ = output.status(&message);
            }
            _ => tracing::warn!("{message}"),
        }
    }

    /// Asks a running process for its status.
    pub(super) async fn ask_status(socket: &Path) -> Result<StatusReport, CliError> {
        let client = connect(socket).await?;
        let report: StatusReport = client
            .actor(INTROSPECTION_ACTOR_NAME)
            .ask(GetStatus)
            .await
            .map_err(|error| remote_error(socket, "status", &error))?;
        warn_on_newer_schema(None, report.schema_version);
        Ok(report)
    }

    /// Tells a running process to stop admitting turns.
    pub(super) async fn ask_pause(socket: &Path) -> Result<AdmissionAck, CliError> {
        let client = connect(socket).await?;
        client
            .actor(INTROSPECTION_ACTOR_NAME)
            .ask(Pause)
            .await
            .map_err(|error| remote_error(socket, "pause", &error))
    }

    /// Tells a running process to admit turns again.
    pub(super) async fn ask_resume(socket: &Path) -> Result<AdmissionAck, CliError> {
        let client = connect(socket).await?;
        client
            .actor(INTROSPECTION_ACTOR_NAME)
            .ask(Resume)
            .await
            .map_err(|error| remote_error(socket, "resume", &error))
    }

    /// Tells a running process to drain.
    pub(super) async fn ask_drain(socket: &Path) -> Result<AdmissionAck, CliError> {
        let client = connect(socket).await?;
        client
            .actor(INTROSPECTION_ACTOR_NAME)
            .ask(Drain)
            .await
            .map_err(|error| remote_error(socket, "drain", &error))
    }

    /// Polls until no turns are in flight, or the timeout expires.
    ///
    /// `timeout_secs` of 0 waits indefinitely, which is what a supervisor with
    /// its own `TimeoutStopSec` wants: two competing deadlines is one too
    /// many, and systemd's is the one that can actually kill the process.
    pub(super) async fn wait_for_drain(
        socket: &Path,
        first: AdmissionAck,
        timeout_secs: u64,
        output: &OutputWriter,
    ) -> Result<AdmissionAck, CliError> {
        if first.is_drained() {
            return Ok(first);
        }

        let deadline = (timeout_secs > 0)
            .then(|| std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs));

        let mut latest = first;
        loop {
            if let Some(deadline) = deadline {
                if std::time::Instant::now() >= deadline {
                    return Err(CliError::configuration(format!(
                        "the process at {} still has {} turn(s) running after {timeout_secs}s. \
                         It is not admitting new ones, so waiting longer is safe — re-run with \
                         --timeout 0 to wait indefinitely, or leave it and let the turns finish",
                        socket.display(),
                        latest.in_flight_turns,
                    )));
                }
            }

            tokio::time::sleep(DRAIN_POLL_INTERVAL).await;

            // Re-asking `Drain` rather than `GetStatus` keeps the process
            // draining even if something raced a `resume` in between, and
            // costs less to assemble than a full report.
            latest = ask_drain(socket).await?;
            if latest.is_drained() {
                return Ok(latest);
            }
            let _ = output.status(&format!(
                "waiting for {} turn(s) to finish...",
                latest.in_flight_turns
            ));
        }
    }
}

#[cfg(feature = "ipc")]
use client::{ask_drain, ask_pause, ask_resume, ask_status, wait_for_drain};

/// The stand-ins a build without the `ipc` feature gets.
///
/// The commands stay in `--help` rather than vanishing, because an operator
/// who types `acton-ai status` and is told "unknown subcommand" learns
/// nothing, while one told the binary was built without the feature knows
/// exactly what to do next.
#[cfg(not(feature = "ipc"))]
mod stub {
    use super::{CliError, OutputWriter, Path};

    /// The one message every command in this module gives without `ipc`.
    fn unsupported(command: &str) -> CliError {
        CliError::configuration(format!(
            "`acton-ai {command}` talks to a running process over a Unix socket, but this \
             binary was built without the `ipc` feature (it is on by default), so it has no \
             client to talk with; rebuild with the `ipc` feature enabled"
        ))
    }

    pub(super) async fn ask_status(_socket: &Path) -> Result<Unsupported, CliError> {
        Err(unsupported("status"))
    }

    pub(super) async fn ask_pause(_socket: &Path) -> Result<Unsupported, CliError> {
        Err(unsupported("pause"))
    }

    pub(super) async fn ask_resume(_socket: &Path) -> Result<Unsupported, CliError> {
        Err(unsupported("resume"))
    }

    pub(super) async fn ask_drain(_socket: &Path) -> Result<Unsupported, CliError> {
        Err(unsupported("drain"))
    }

    pub(super) async fn wait_for_drain(
        _socket: &Path,
        _first: Unsupported,
        _timeout_secs: u64,
        _output: &OutputWriter,
    ) -> Result<Unsupported, CliError> {
        Err(unsupported("drain"))
    }

    /// Inhabits the success type of calls that can never succeed here.
    ///
    /// An empty enum rather than a unit struct, so the renderers below are
    /// statically unreachable rather than merely unused.
    pub(super) enum Unsupported {}
}

#[cfg(not(feature = "ipc"))]
use stub::{ask_drain, ask_pause, ask_resume, ask_status, wait_for_drain, Unsupported};

/// Statically unreachable: `ask_*` never returns an [`Unsupported`], so this
/// exists only to keep the command bodies identical across both builds.
#[cfg(not(feature = "ipc"))]
fn write_ack(_output: &OutputWriter, ack: &Unsupported, _summary: &str) -> Result<(), CliError> {
    match *ack {}
}

/// Statically unreachable, for the same reason as [`write_ack`].
#[cfg(not(feature = "ipc"))]
fn write_status(_output: &OutputWriter, report: &Unsupported) -> Result<(), CliError> {
    match *report {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config file with providers but no `[introspection]` section.
    fn config_without_introspection(dir: &Path) -> PathBuf {
        let path = dir.join("acton-ai.toml");
        std::fs::write(
            &path,
            "[providers.local]\ntype = \"ollama\"\nmodel = \"m\"\n",
        )
        .expect("write config");
        path
    }

    #[test]
    fn an_explicit_socket_wins_over_everything() {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = dir.path().join("acton-ai.toml");
        std::fs::write(
            &config,
            "[providers.local]\ntype = \"ollama\"\nmodel = \"m\"\n\
             [introspection]\nsocket_path = \"/run/from-config.sock\"\n",
        )
        .expect("write config");
        std::fs::write(dir.path().join("scanned.sock"), b"").expect("write");

        // Both of the lower-precedence sources would answer, and neither gets
        // to: `--socket` is the operator being explicit.
        let chosen = resolve_socket_in(
            Some(Path::new("/tmp/explicit.sock")),
            Some(&config),
            dir.path(),
        )
        .expect("an explicit path is always usable");
        assert_eq!(chosen, PathBuf::from("/tmp/explicit.sock"));
    }

    #[test]
    fn an_explicit_socket_is_used_even_when_nothing_is_there() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("live.sock"), b"").expect("write");

        // Silently falling back to a discovered socket would be the worst
        // possible outcome: `pause --socket a` must never pause b.
        let chosen = resolve_socket_in(
            Some(Path::new("/definitely/not/here.sock")),
            None,
            dir.path(),
        )
        .expect("the operator said where to look");
        assert_eq!(chosen, PathBuf::from("/definitely/not/here.sock"));
    }

    #[test]
    fn the_config_socket_path_is_used_when_no_flag_is_given() {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = dir.path().join("acton-ai.toml");
        std::fs::write(
            &config,
            "[providers.local]\ntype = \"ollama\"\nmodel = \"m\"\n\
             [introspection]\nsocket_path = \"/run/from-config.sock\"\n",
        )
        .expect("write config");
        // A socket in the scan directory too, so this pins precedence rather
        // than merely "the config was read at all".
        std::fs::write(dir.path().join("scanned.sock"), b"").expect("write");

        let chosen =
            resolve_socket_in(None, Some(&config), dir.path()).expect("the config names a socket");
        assert_eq!(chosen, PathBuf::from("/run/from-config.sock"));
    }

    #[test]
    fn a_single_socket_in_the_runtime_directory_is_chosen() {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = config_without_introspection(dir.path());
        let only = dir.path().join("acton-ai-42.sock");
        std::fs::write(&only, b"").expect("write");

        // The default naming scheme carries a PID nobody can guess, so the
        // scan is the only thing that makes a bare `acton-ai status` work.
        let chosen = resolve_socket_in(None, Some(&config), dir.path()).expect("exactly one match");
        assert_eq!(chosen, only);
    }

    #[test]
    fn several_live_sockets_are_refused_rather_than_guessed_between() {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = config_without_introspection(dir.path());
        std::fs::write(dir.path().join("acton-ai-1.sock"), b"").expect("write");
        std::fs::write(dir.path().join("acton-ai-2.sock"), b"").expect("write");

        let error = resolve_socket_in(None, Some(&config), dir.path())
            .expect_err("two candidates is not a choice this command may make");
        let text = error.to_string();

        // Guessing here would let `drain` stop the wrong process, so the error
        // has to list what it found rather than pick.
        assert!(text.contains("acton-ai-1.sock"), "{text}");
        assert!(text.contains("acton-ai-2.sock"), "{text}");
        assert!(text.contains("--socket"), "{text}");
    }

    #[test]
    fn a_config_without_an_introspection_section_does_not_choose_a_socket() {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = config_without_introspection(dir.path());

        // Falls through to the directory scan, which is empty — the point is
        // that it does not invent a path.
        let error = resolve_socket_in(None, Some(&config), dir.path())
            .expect_err("nothing names a socket")
            .to_string();
        assert!(error.contains("no introspection socket found"), "{error}");
    }

    #[test]
    fn the_not_found_error_names_everything_that_was_tried() {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = config_without_introspection(dir.path());

        let error = resolve_socket_in(None, Some(&config), dir.path())
            .expect_err("nothing names a socket")
            .to_string();

        // Three o'clock in the morning: the error has to carry the whole
        // story, because the operator cannot ask it a follow-up question.
        assert!(error.contains("--socket"), "{error}");
        assert!(error.contains(&config.display().to_string()), "{error}");
        assert!(error.contains(&dir.path().display().to_string()), "{error}");
        assert!(error.contains("introspection"), "{error}");
    }

    #[test]
    fn the_default_scan_directory_is_the_one_the_server_binds_into() {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = config_without_introspection(dir.path());

        // `resolve_socket` is the thin wrapper; the one thing it adds is
        // *which* directory gets scanned, and a client scanning a directory
        // the server never binds into would make discovery silently never
        // work. Asserted as an equivalence rather than by inspecting the real
        // runtime directory, so the result does not depend on whether this
        // machine happens to have an agent running right now.
        let via_wrapper = resolve_socket(None, Some(&config)).map_err(|e| e.to_string());
        let via_default = resolve_socket_in(
            None,
            Some(&config),
            &crate::introspection::default_socket_dir(),
        )
        .map_err(|e| e.to_string());

        assert_eq!(via_wrapper, via_default);
    }

    #[test]
    fn list_sockets_finds_only_socket_named_entries() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("one.sock"), b"").expect("write");
        std::fs::write(dir.path().join("two.sock"), b"").expect("write");
        std::fs::write(dir.path().join("notes.txt"), b"").expect("write");
        std::fs::create_dir(dir.path().join("nested")).expect("mkdir");
        std::fs::write(dir.path().join("nested").join("deep.sock"), b"").expect("write");

        let mut found = list_sockets(dir.path());
        found.sort();

        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found.iter().all(|p| p.extension().unwrap() == "sock"));
        // Non-recursive: a socket one level down belongs to something else.
        assert!(!found.iter().any(|p| p.ends_with("deep.sock")), "{found:?}");
    }

    #[test]
    fn list_sockets_on_a_missing_directory_is_empty_not_an_error() {
        assert!(list_sockets(Path::new("/definitely/not/a/directory")).is_empty());
    }

    #[test]
    fn uptime_is_rendered_at_the_scale_a_human_reads() {
        assert_eq!(format_uptime(0), "0s");
        assert_eq!(format_uptime(45), "45s");
        assert_eq!(format_uptime(90), "1m 30s");
        assert_eq!(format_uptime(3_661), "1h 1m");
        assert_eq!(format_uptime(90_061), "1d 1h 1m");
    }

    #[test]
    fn the_default_drain_timeout_is_bounded_but_generous() {
        // Long enough that a normal turn finishes, short enough that a stuck
        // deploy reports rather than hangs.
        assert_eq!(DEFAULT_DRAIN_TIMEOUT_SECS, 300);
    }
}
