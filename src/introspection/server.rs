//! Where the control socket lives, and how it is brought up and taken down.

use std::path::{Path, PathBuf};

use crate::error::ActonAIError;
use crate::introspection::IntrospectionConfig;

/// The directory every default socket path is created under, relative to the
/// runtime dir.
pub const SOCKET_DIR_NAME: &str = "acton-ai";

/// Permissions on the socket file: owner read/write, nobody else.
///
/// Tighter than acton-reactive's own `0o660` default, deliberately. That
/// default suits a socket other services in the same group are meant to talk
/// to; this one accepts `pause` and `drain`, so group write is a wider door
/// than the feature needs.
pub const SOCKET_MODE: u32 = 0o600;

/// Permissions on the containing directory: owner only, so the socket cannot
/// be reached by traversing in from outside.
const SOCKET_DIR_MODE: u32 = 0o700;

/// Resolves the socket path for a process, given its configuration.
///
/// # The default scheme
///
/// `$XDG_RUNTIME_DIR/acton-ai/<app-name>-<pid>.sock`, falling back to
/// `/tmp/acton-ai-<uid>/<app-name>-<pid>.sock` when `XDG_RUNTIME_DIR` is
/// unset.
///
/// The PID suffix is what makes the default *safe*: two `acton-ai chat`
/// sessions for one user must not fight over one path, and the loser of that
/// fight would either fail to launch or — far worse — answer control commands
/// meant for the other process. The cost is that the default path is not
/// predictable from outside, which is why `acton-ai status` discovers sockets
/// by scanning the directory, and why a process that wants a *stable* address
/// sets `socket_path` explicitly.
///
/// The fallback directory is per-uid rather than shared, because `/tmp` is
/// world-writable and a shared `acton-ai` directory there would be a name
/// another user could claim first.
#[must_use]
pub fn resolve_socket_path(config: &IntrospectionConfig, app_name: &str) -> PathBuf {
    if let Some(path) = config.socket_path.as_ref() {
        return path.clone();
    }
    default_socket_path(runtime_dir(), app_name, std::process::id())
}

/// The directory default socket paths are created in.
///
/// `$XDG_RUNTIME_DIR/acton-ai` when the variable is set, else the per-uid
/// `/tmp` fallback. This is also what the CLI scans when no `--socket` was
/// given.
#[must_use]
pub fn default_socket_dir() -> PathBuf {
    socket_dir_in(runtime_dir())
}

/// `$XDG_RUNTIME_DIR`, or `None` when it is unset or empty.
fn runtime_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The directory portion of the default scheme.
///
/// Pure: it decides from the runtime dir it is handed and touches no
/// filesystem, so both branches are testable without mutating the process
/// environment.
fn socket_dir_in(runtime_dir: Option<PathBuf>) -> PathBuf {
    match runtime_dir {
        Some(dir) => dir.join(SOCKET_DIR_NAME),
        None => std::env::temp_dir().join(format!("{SOCKET_DIR_NAME}-{}", current_uid())),
    }
}

/// The full default scheme. Pure, for the same reason.
fn default_socket_path(runtime_dir: Option<PathBuf>, app_name: &str, pid: u32) -> PathBuf {
    socket_dir_in(runtime_dir).join(format!("{}-{pid}.sock", sanitize(app_name)))
}

/// The current user's uid, or 0 where the platform has no such concept.
#[cfg(unix)]
fn current_uid() -> u32 {
    nix::unistd::getuid().as_raw()
}

#[cfg(not(unix))]
fn current_uid() -> u32 {
    0
}

/// Reduces `name` to characters that are unambiguous in a path.
///
/// An `app_name` is caller-supplied and otherwise unconstrained, so it can
/// contain a `/` — which would silently redirect the socket into another
/// directory — or a NUL, which is not representable in a socket address.
/// Anything outside the safe set becomes `-`, and an empty result becomes
/// the crate name so the path is never a bare `-1234.sock`.
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '-'
            }
        })
        .collect();
    // `.` and `..` name directories rather than files, and a name of nothing
    // but dots is one of those two. Dots elsewhere are left alone: with every
    // separator already gone, `a..b` is an ordinary filename.
    if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
        "acton-ai".to_string()
    } else {
        cleaned
    }
}

/// Creates the socket's parent directory with owner-only permissions.
///
/// # Errors
///
/// Returns a configuration error naming the directory when it cannot be
/// created — an unwritable runtime dir is a launch failure, not something to
/// discover when the first `status` fails to connect.
pub(crate) fn ensure_socket_dir(socket_path: &Path) -> Result<(), ActonAIError> {
    let Some(parent) = socket_path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }

    std::fs::create_dir_all(parent).map_err(|error| {
        ActonAIError::configuration(
            "introspection.socket_path",
            format!(
                "could not create the directory for the introspection socket at {}: {error}",
                parent.display()
            ),
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Best effort: a pre-existing directory with looser permissions is
        // tightened, but failing to tighten it must not stop a launch — the
        // socket file's own 0o600 is the load-bearing check.
        if let Err(error) =
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(SOCKET_DIR_MODE))
        {
            tracing::debug!(
                path = %parent.display(),
                %error,
                "could not tighten permissions on the introspection socket directory"
            );
        }
    }

    Ok(())
}

/// Registers the wire types, exposes the actor, and starts the listener.
///
/// Ordering matters and is not incidental: the types must be registered and
/// the actor exposed *before* the listener starts accepting, or the first
/// request to arrive can find a name that is not yet bound and be refused
/// with `ACTOR_NOT_FOUND` — a race that would show up as a flaky `status`
/// immediately after launch.
///
/// A socket file left behind by a process that was killed rather than shut
/// down is handled by acton-reactive's listener, which tries to connect before
/// it removes anything: a socket something answers on is a live one and fails
/// the bind, while a dead file is cleared. That is the behaviour this feature
/// needs, so there is nothing to duplicate here.
///
/// # Errors
///
/// Returns a configuration error when the exposed name is already taken, or
/// when the socket cannot be bound — most often because another process is
/// already listening on the same path.
#[cfg(feature = "ipc")]
pub(crate) async fn start_listener(
    runtime: &mut acton_reactive::prelude::ActorRuntime,
    handle: acton_reactive::prelude::ActorHandle,
    socket_path: &Path,
    socket_mode: u32,
) -> Result<acton_reactive::ipc::IpcListenerHandle, ActonAIError> {
    use crate::introspection::wire::{
        AdmissionAck, Drain, GetStatus, Pause, Resume, StatusReport, INTROSPECTION_ACTOR_NAME,
    };
    use acton_reactive::prelude::RemoteRequest;

    let registry = runtime.ipc_registry();
    // Both directions of every exchange: the request type so the server can
    // decode what arrives, the response type so it can encode what it sends.
    // Registering only the requests produces a server that parses commands
    // and then cannot answer them.
    registry.register::<GetStatus>(GetStatus::MESSAGE_TYPE);
    registry.register::<Pause>(Pause::MESSAGE_TYPE);
    registry.register::<Resume>(Resume::MESSAGE_TYPE);
    registry.register::<Drain>(Drain::MESSAGE_TYPE);
    registry.register::<StatusReport>("acton_ai.StatusReport");
    registry.register::<AdmissionAck>("acton_ai.AdmissionAck");

    runtime
        .ipc_expose(INTROSPECTION_ACTOR_NAME, handle)
        .map_err(|error| {
            ActonAIError::configuration(
                "introspection",
                format!(
                    "the IPC name '{INTROSPECTION_ACTOR_NAME}' is already exposed in this \
                     process ({error}); acton-ai reserves it for the introspection surface, so \
                     an embedder must expose its own actors under a different name"
                ),
            )
        })?;

    let mut config = acton_reactive::ipc::IpcConfig::default();
    config.socket.path = Some(socket_path.to_path_buf());
    #[cfg(unix)]
    {
        config.socket.mode = socket_mode;
    }
    #[cfg(not(unix))]
    let _ = socket_mode;

    let listener = runtime
        .start_ipc_listener_with_config(config)
        .await
        .map_err(|error| {
            ActonAIError::configuration(
                "introspection.socket_path",
                format!(
                    "could not listen on the introspection socket at {}: {error}. Another \
                     process may already be listening there — check with `acton-ai status \
                     --socket {}` — or set a different `socket_path` under [introspection]",
                    socket_path.display(),
                    socket_path.display(),
                ),
            )
        })?;

    tracing::info!(
        socket = %socket_path.display(),
        mode = format_args!("{socket_mode:#o}"),
        "introspection listening"
    );

    Ok(listener)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_socket_path_wins() {
        let config = IntrospectionConfig {
            socket_path: Some(PathBuf::from("/run/user/1000/fixed.sock")),
            ..IntrospectionConfig::default()
        };
        assert_eq!(
            resolve_socket_path(&config, "whatever"),
            PathBuf::from("/run/user/1000/fixed.sock")
        );
    }

    #[test]
    fn the_default_path_lives_under_the_runtime_dir() {
        let path = default_socket_path(Some(PathBuf::from("/run/user/1000")), "my-agent", 4321);
        assert_eq!(
            path,
            PathBuf::from("/run/user/1000/acton-ai/my-agent-4321.sock")
        );
    }

    #[test]
    fn the_default_path_is_pid_suffixed() {
        let one = default_socket_path(Some(PathBuf::from("/run/user/1000")), "agent", 1);
        let two = default_socket_path(Some(PathBuf::from("/run/user/1000")), "agent", 2);
        // Two processes for the same user must never resolve to one path:
        // the loser would answer control commands meant for the winner.
        assert_ne!(one, two);
    }

    #[test]
    fn without_a_runtime_dir_the_fallback_is_per_uid() {
        let path = default_socket_path(None, "agent", 7);
        let text = path.to_string_lossy().into_owned();
        // Per-uid, because /tmp is world-writable and a shared name there is
        // one another user could claim first.
        assert!(
            text.contains(&format!("{SOCKET_DIR_NAME}-{}", current_uid())),
            "{text}"
        );
        assert!(text.ends_with("agent-7.sock"), "{text}");
    }

    #[test]
    fn a_path_separator_in_the_app_name_cannot_redirect_the_socket() {
        let path = default_socket_path(Some(PathBuf::from("/run/user/1000")), "../../etc/evil", 9);
        assert_eq!(
            path,
            PathBuf::from("/run/user/1000/acton-ai/..-..-etc-evil-9.sock")
        );
        // The decisive property: the socket stays inside the directory the
        // scheme chose.
        assert!(path.starts_with("/run/user/1000/acton-ai"));
    }

    #[test]
    fn an_empty_app_name_still_produces_a_named_socket() {
        let path = default_socket_path(Some(PathBuf::from("/run")), "", 3);
        assert_eq!(path, PathBuf::from("/run/acton-ai/acton-ai-3.sock"));
    }

    #[test]
    fn an_app_name_of_dots_does_not_become_a_directory_reference() {
        assert_eq!(sanitize(".."), "acton-ai");
        assert_eq!(sanitize("."), "acton-ai");
    }

    #[test]
    fn sanitize_keeps_ordinary_names_intact() {
        // The common case must not be mangled: an operator should recognise
        // their own app name in the path.
        assert_eq!(sanitize("my-agent_v2.1"), "my-agent_v2.1");
    }

    #[test]
    fn the_socket_mode_is_owner_only() {
        // 0o600 is the security boundary the credential check cannot be, so
        // a regression here is a real widening of access.
        assert_eq!(SOCKET_MODE, 0o600);
        assert_eq!(SOCKET_MODE & 0o077, 0, "no group or other permissions");
    }

    #[test]
    fn ensure_socket_dir_creates_a_missing_parent() {
        let dir = tempfile::tempdir().expect("temp dir");
        let socket = dir.path().join("nested").join("deeper").join("ipc.sock");

        ensure_socket_dir(&socket).expect("directory is created");

        assert!(socket.parent().expect("has a parent").is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn ensure_socket_dir_tightens_the_parent_to_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("temp dir");
        let socket = dir.path().join("nested").join("ipc.sock");

        ensure_socket_dir(&socket).expect("directory is created");

        let mode = std::fs::metadata(socket.parent().expect("has a parent"))
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, SOCKET_DIR_MODE, "{mode:#o}");
    }
}
