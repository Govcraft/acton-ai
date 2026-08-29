//! Live introspection: ask a running process what it is doing, and tell it to
//! stop taking new work.
//!
//! A long-running agent process is otherwise opaque. Logs say what happened,
//! metrics say how much, but neither answers "what is this process doing *now*,
//! and can I safely restart it?". This module answers both over a Unix socket:
//!
//! ```text
//! acton-ai status            what is running, how much has been spent
//! acton-ai pause             refuse new turns; in-flight turns finish
//! acton-ai resume            accept new turns again
//! acton-ai drain             refuse new turns and report when the last finishes
//! ```
//!
//! # The shape of it
//!
//! - [`AdmissionGate`] is a single atomic byte the prompt loop reads before it
//!   starts a turn. It is **not** feature-gated: `pause`/`resume`/`drain` are
//!   just as useful to an embedder driving the library from its own control
//!   plane as they are over a socket.
//! - [`IntrospectionActor`] tracks turns passively, by subscribing to the
//!   [`TurnLifecycle`](crate::messages::TurnLifecycle) broadcast the prompt
//!   loop already publishes, and assembles a [`wire::StatusReport`] on demand
//!   by asking the cost accountant, the providers, and the MCP supervisor.
//! - [`wire`] is the versioned protocol, shared verbatim by the server here
//!   and the CLI client — one definition, so the two cannot disagree.
//! - [`sd_notify`] tells systemd when the process is ready and when it has
//!   begun stopping.
//!
//! # What in-flight means
//!
//! A *turn* is one `collect()`/`extract()` call, from the first request to the
//! final answer, spanning however many provider rounds and tool executions it
//! takes. Pausing refuses turns that have not started. It never interrupts one
//! that has: a half-finished turn has usually already been paid for, and a
//! tool it launched may have already changed the world. `drain` is therefore
//! "finish what you started, take nothing new", which is what makes it safe to
//! wire to `SIGTERM`.
//!
//! Drain is reported as complete when the active-turn count reaches zero. That
//! is a statement about turns, not about the process: background work an
//! embedder started outside a turn is not tracked here and will not hold a
//! drain open.
//!
//! # Feature gating
//!
//! The socket server ([`server`], [`wire`], [`actor`]) sits behind the `ipc`
//! feature, which is on by default. Admission control does not, so a
//! `--no-default-features` build keeps [`AdmissionGate`] and the builder's
//! pause/resume/drain.
//!
//! Configuring `[introspection]` in a build without `ipc` **fails the launch**
//! naming the missing feature — see `unsupported_error`. An operator whose
//! socket was silently never created would debug the wrong thing entirely.
//!
//! # It is off unless asked for
//!
//! Compiling the feature in creates no socket. Nothing listens until either
//! `[introspection]` appears in the config file or
//! [`ActonAIBuilder::introspection`](crate::facade::ActonAIBuilder::introspection)
//! is called. A library whose default behaviour is to open a control socket
//! would be a surprising one to depend on.

use std::path::PathBuf;

pub mod admission;
pub mod sd_notify;
pub mod server;

#[cfg(all(feature = "ipc", unix))]
pub mod actor;
#[cfg(all(feature = "ipc", unix))]
pub mod wire;

pub use admission::{AdmissionGate, AdmissionState};
pub use server::{default_socket_dir, resolve_socket_path, SOCKET_MODE};

#[cfg(all(feature = "ipc", unix))]
pub use actor::IntrospectionActor;
#[cfg(all(feature = "ipc", unix))]
pub use wire::{
    AdmissionAck, BudgetStanding, Drain, GetStatus, McpServerStatus, Pause, ProviderStatus, Resume,
    StatusReport, UsageSummary, INTROSPECTION_ACTOR_NAME, SCHEMA_VERSION,
};

/// Validated introspection settings.
///
/// Unconditional, exactly as
/// [`TelemetryConfig`](crate::telemetry::TelemetryConfig) is: a build without
/// the `ipc` feature still has to parse `[introspection]` and then refuse to
/// launch, which it cannot do with a type that does not exist.
///
/// Holding one of these means the socket mode is a plausible permission bits
/// value, so nothing downstream re-validates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntrospectionConfig {
    /// Where the control socket is bound.
    ///
    /// `None` means the default scheme — see [`resolve_socket_path`] — whose
    /// PID suffix keeps two processes for one user from colliding. Set this
    /// when something outside the process needs a *predictable* address, such
    /// as a systemd unit with a matching `ExecStop`.
    pub socket_path: Option<PathBuf>,

    /// Permission bits for the socket file. Defaults to [`SOCKET_MODE`].
    pub socket_mode: u32,
}

impl Default for IntrospectionConfig {
    fn default() -> Self {
        Self {
            socket_path: None,
            socket_mode: SOCKET_MODE,
        }
    }
}

impl IntrospectionConfig {
    /// Validates raw parts and fills in defaults.
    ///
    /// Shared by the TOML path and the builder path so both refuse the same
    /// inputs for the same reasons.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when `socket_path` is relative (the
    /// process may change directory, and a socket that moves with the cwd is
    /// one no client can find), or when `socket_mode` sets bits outside the
    /// twelve permission bits, or when it grants access to other users.
    pub(crate) fn resolve(
        socket_path: Option<PathBuf>,
        socket_mode: Option<u32>,
    ) -> Result<Self, crate::error::ActonAIError> {
        if let Some(path) = socket_path.as_ref() {
            // `has_root` alongside `is_absolute`: on Unix the two are the same
            // test, but a Unix socket path like `/run/user/1000/agent.sock` is
            // not "absolute" to Windows (no drive prefix), and a config shared
            // across platforms must still parse on a Windows build, where the
            // socket is never created anyway.
            if !(path.is_absolute() || path.has_root()) {
                return Err(crate::error::ActonAIError::configuration(
                    "introspection.socket_path",
                    format!(
                        "`{}` is a relative path; the socket must be given as an absolute path, \
                         because a process that changes its working directory would otherwise \
                         leave its control socket somewhere no client can find it",
                        path.display()
                    ),
                ));
            }
        }

        let socket_mode = socket_mode.unwrap_or(SOCKET_MODE);
        if socket_mode & !0o7777 != 0 {
            return Err(crate::error::ActonAIError::configuration(
                "introspection.socket_mode",
                format!(
                    "`{socket_mode:#o}` sets bits outside the permission bits; write it in octal, \
                     e.g. socket_mode = 0o600"
                ),
            ));
        }

        // `pause` and `drain` change what the process will do. A socket other
        // users can write to hands them that lever, and nothing about
        // introspection needs it, so this is refused rather than warned about.
        if socket_mode & 0o077 != 0 {
            return Err(crate::error::ActonAIError::configuration(
                "introspection.socket_mode",
                format!(
                    "`{socket_mode:#o}` grants access beyond the owner; the introspection socket \
                     accepts pause and drain commands, so group and other permissions would let \
                     another account stop this process taking work. Use 0o600, or 0o700 if a \
                     directory-style bit is wanted"
                ),
            ));
        }

        Ok(Self {
            socket_path,
            socket_mode,
        })
    }
}

/// The error a build without the `ipc` feature raises when a runtime is
/// nonetheless configured for introspection.
///
/// Kept here rather than inline at the call site so the message stays in the
/// module that owns the feature, and so the feature-off test can assert on
/// exactly what an operator will read.
#[cfg(not(all(feature = "ipc", unix)))]
pub(crate) fn unsupported_error() -> crate::error::ActonAIError {
    crate::error::ActonAIError::configuration(
        "introspection",
        "introspection is configured, but this binary was built without the `ipc` feature, so no \
         control socket would be created and `acton-ai status` would find nothing; rebuild with \
         the `ipc` feature (it is on by default) or remove the [introspection] section",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_socket_mode_is_owner_only() {
        assert_eq!(
            IntrospectionConfig::default().socket_mode,
            SOCKET_MODE,
            "the default must not widen without a deliberate change"
        );
    }

    #[test]
    fn an_absolute_socket_path_is_kept() {
        let resolved = IntrospectionConfig::resolve(Some(PathBuf::from("/run/a.sock")), None)
            .expect("absolute paths are accepted");
        assert_eq!(resolved.socket_path, Some(PathBuf::from("/run/a.sock")));
        assert_eq!(resolved.socket_mode, SOCKET_MODE);
    }

    #[test]
    fn a_relative_socket_path_is_refused() {
        let error = IntrospectionConfig::resolve(Some(PathBuf::from("agent.sock")), None)
            .expect_err("relative paths are refused");
        let text = error.to_string();
        assert!(text.contains("agent.sock"), "{text}");
        assert!(text.contains("absolute"), "{text}");
    }

    #[test]
    fn a_group_readable_socket_mode_is_refused() {
        let error =
            IntrospectionConfig::resolve(None, Some(0o660)).expect_err("group access is refused");
        // The reason matters more than the refusal: an operator who set 0o660
        // deliberately needs to know which command they were handing out.
        assert!(error.to_string().contains("pause"), "{error}");
    }

    #[test]
    fn a_world_readable_socket_mode_is_refused() {
        assert!(IntrospectionConfig::resolve(None, Some(0o606)).is_err());
    }

    #[test]
    fn an_owner_only_socket_mode_is_accepted() {
        let resolved =
            IntrospectionConfig::resolve(None, Some(0o700)).expect("owner-only bits are accepted");
        assert_eq!(resolved.socket_mode, 0o700);
    }

    #[test]
    fn a_socket_mode_outside_the_permission_bits_is_refused() {
        let error = IntrospectionConfig::resolve(None, Some(0o10600))
            .expect_err("stray high bits are refused");
        assert!(error.to_string().contains("octal"), "{error}");
    }
}
