//! A hand-rolled sd_notify client: ~40 lines instead of a dependency.
//!
//! The systemd notification protocol is a newline-separated `KEY=VALUE`
//! datagram sent to the socket named by `$NOTIFY_SOCKET`. That is the whole
//! specification for what this crate needs, so the `libsystemd`/`sd-notify`
//! crates would be a dependency to carry, audit, and license for two string
//! constants.
//!
//! # What is sent
//!
//! - `READY=1` once launch has completed and the process can serve.
//! - `STOPPING=1` when a drain or shutdown begins.
//!
//! With `Type=notify` in the unit file, systemd holds `systemctl start` open
//! until `READY=1` arrives, so "started" means "actually serving" rather than
//! "forked". `STOPPING=1` tells systemd the process heard the signal and is
//! working on it, so `TimeoutStopSec` is a backstop rather than the plan.
//!
//! # When nothing happens
//!
//! Unset `$NOTIFY_SOCKET` means the process was not started by systemd, which
//! is the common case (a shell, a test, a container). That is not an error
//! and never surfaces as one — every function here is a silent no-op.
//!
//! # Address forms
//!
//! systemd names either a filesystem path (`/run/systemd/notify`) or an
//! abstract socket, written with a leading `@`. The abstract form is a Linux
//! extension whose address is the NUL byte followed by the name, which is why
//! the `@` cannot simply be passed through as a path.

/// The environment variable systemd sets to name its notification socket.
pub(crate) const NOTIFY_SOCKET_ENV: &str = "NOTIFY_SOCKET";

/// The state line meaning "startup finished, the process is serving".
pub(crate) const READY: &str = "READY=1";

/// The state line meaning "shutdown has begun".
pub(crate) const STOPPING: &str = "STOPPING=1";

/// Tells systemd the process has finished starting up.
///
/// A no-op when `$NOTIFY_SOCKET` is unset.
pub fn notify_ready() {
    notify(READY);
}

/// Tells systemd the process has begun shutting down.
///
/// A no-op when `$NOTIFY_SOCKET` is unset.
pub fn notify_stopping() {
    notify(STOPPING);
}

/// Reads `$NOTIFY_SOCKET` and sends `state` to it.
///
/// Split from [`notify_to`] so that everything testable is testable without
/// touching the environment: env mutation is process-global and racy under a
/// parallel test runner, so the tests drive `notify_to` with a path instead.
/// This wrapper is the thin, untested part, and it is thin on purpose.
fn notify(state: &str) {
    let Ok(address) = std::env::var(NOTIFY_SOCKET_ENV) else {
        return;
    };
    if let Err(error) = notify_to(&address, state) {
        // A failed notification must never take the process down with it: at
        // worst systemd falls back to its own timeouts. Logged, because a
        // unit that mysteriously times out on start is exactly the thing an
        // operator needs a breadcrumb for.
        tracing::debug!(%error, address, state, "sd_notify datagram not delivered");
    }
}

/// Sends `state` to the socket at `address`.
///
/// `address` is a systemd notify address: a filesystem path, or an
/// abstract-namespace name written with a leading `@`.
///
/// # Errors
///
/// Returns the underlying IO error when the socket cannot be opened or the
/// datagram cannot be sent. Callers treat this as advisory.
#[cfg(unix)]
pub(crate) fn notify_to(address: &str, state: &str) -> std::io::Result<()> {
    use std::os::unix::net::UnixDatagram;

    if address.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "NOTIFY_SOCKET is set but empty",
        ));
    }

    let socket = UnixDatagram::unbound()?;

    match address.strip_prefix('@') {
        Some(name) => send_to_abstract(&socket, name, state),
        None => {
            socket.send_to(state.as_bytes(), address)?;
            Ok(())
        }
    }
}

/// Sends to an abstract-namespace socket, whose address is a NUL byte
/// followed by the name.
///
/// The abstract namespace is a Linux extension. systemd runs on Linux, so
/// every real `@`-prefixed address arrives here on a Linux build; the
/// fallback exists so the crate still compiles on the other Unixes rather
/// than for any address it expects to see.
#[cfg(all(unix, target_os = "linux"))]
fn send_to_abstract(
    socket: &std::os::unix::net::UnixDatagram,
    name: &str,
    state: &str,
) -> std::io::Result<()> {
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::SocketAddr;

    let addr = SocketAddr::from_abstract_name(name.as_bytes())?;
    socket.send_to_addr(state.as_bytes(), &addr)?;
    Ok(())
}

/// Stand-in for the Unixes without an abstract socket namespace.
#[cfg(all(unix, not(target_os = "linux")))]
fn send_to_abstract(
    _socket: &std::os::unix::net::UnixDatagram,
    _name: &str,
    _state: &str,
) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "abstract-namespace sockets are a Linux extension; NOTIFY_SOCKET named one on a platform \
         that has none",
    ))
}

/// Non-Unix stand-in. systemd is a Unix service manager; there is nothing to
/// notify, and pretending otherwise would be a lie in the type signature.
#[cfg(not(unix))]
pub(crate) fn notify_to(_address: &str, _state: &str) -> std::io::Result<()> {
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::net::UnixDatagram;

    /// Binds a listener at a temp path and returns it with that path.
    fn listener() -> (UnixDatagram, tempfile::TempDir, String) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("notify.sock");
        let socket = UnixDatagram::bind(&path).expect("bind notify socket");
        let address = path.to_string_lossy().into_owned();
        (socket, dir, address)
    }

    /// Reads one datagram. The send is synchronous and already returned, so
    /// the datagram is queued before this runs — there is nothing to wait for
    /// and nothing to sleep on.
    fn recv(socket: &UnixDatagram) -> String {
        let mut buf = [0_u8; 64];
        let read = socket.recv(&mut buf).expect("datagram queued");
        String::from_utf8_lossy(&buf[..read]).into_owned()
    }

    #[test]
    fn ready_is_delivered_verbatim_over_a_path_socket() {
        let (socket, _dir, address) = listener();

        notify_to(&address, READY).expect("send succeeds");

        // Verbatim: systemd parses `KEY=VALUE`, so a stray newline or a
        // different spelling is a datagram systemd silently ignores.
        assert_eq!(recv(&socket), "READY=1");
    }

    #[test]
    fn stopping_is_delivered_verbatim() {
        let (socket, _dir, address) = listener();

        notify_to(&address, STOPPING).expect("send succeeds");

        assert_eq!(recv(&socket), "STOPPING=1");
    }

    #[test]
    fn each_notification_is_its_own_datagram() {
        let (socket, _dir, address) = listener();

        notify_to(&address, READY).expect("send succeeds");
        notify_to(&address, STOPPING).expect("send succeeds");

        // Two sends must not coalesce into one datagram: systemd reads one
        // message at a time, so a merged "READY=1STOPPING=1" would lose the
        // second state entirely.
        assert_eq!(recv(&socket), "READY=1");
        assert_eq!(recv(&socket), "STOPPING=1");
    }

    #[test]
    fn an_empty_address_is_rejected_rather_than_sent() {
        let error = notify_to("", READY).expect_err("empty address is not a socket");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        // The OS refuses an empty path too, with the same kind and a generic
        // message. The reason for checking first is the message: this one gets
        // logged, and "NOTIFY_SOCKET is set but empty" names the variable an
        // operator has to go and fix.
        assert!(
            error.to_string().contains(NOTIFY_SOCKET_ENV),
            "the error must name the variable: {error}"
        );
    }

    #[test]
    fn a_missing_socket_reports_an_error_instead_of_panicking() {
        let dir = tempfile::tempdir().expect("temp dir");
        let missing = dir.path().join("nobody-is-listening.sock");

        assert!(notify_to(&missing.to_string_lossy(), READY).is_err());
    }

    #[test]
    fn the_state_constants_are_the_protocol_spellings() {
        // These strings are the wire protocol. A typo here is a silent
        // failure at the far end, so they are asserted rather than assumed.
        assert_eq!(READY, "READY=1");
        assert_eq!(STOPPING, "STOPPING=1");
        assert_eq!(NOTIFY_SOCKET_ENV, "NOTIFY_SOCKET");
    }

    #[test]
    fn notify_is_a_no_op_when_the_environment_names_no_socket() {
        // `notify` reads the environment, so this asserts only the property
        // that needs no mutation: it returns rather than panicking whatever
        // the ambient value is. The delivery itself is covered above.
        notify_ready();
        notify_stopping();
    }
}
