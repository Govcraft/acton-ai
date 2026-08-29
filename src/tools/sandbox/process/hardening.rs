//! Linux-specific OS hardening for the process sandbox child.
//!
//! This module is only compiled on Linux. Callers on other platforms get a
//! no-op `apply` shim from [`super`] so they can invoke
//! `super::hardening::apply(cfg)` unconditionally.
//!
//! When the `sandbox-hardening` feature is enabled we install:
//!
//! - a **landlock** ruleset that confines the child to read-only access on
//!   standard system paths (`/usr`, `/lib`, `/lib64`, `/bin`, `/sbin`,
//!   `/etc`, `/sys/kernel`, `/proc`), read/write access on a handful of
//!   character devices (`/dev/null` and friends, without which no
//!   subprocess can be spawned at all), plus read/write access on `$TMPDIR`
//!   and the current working directory — which for a confined call is the
//!   root the parent handed it, and otherwise a throwaway directory, plus
//!   whatever [`ProcessSandboxConfig::read_exec_paths`] and
//!   [`ProcessSandboxConfig::read_write_paths`] declare;
//! - a **seccomp** filter that returns `EPERM` for a small set of dangerous
//!   syscalls (`ptrace`, `keyctl`, `mount`, `umount2`, `reboot`,
//!   `kexec_load`, `init_module`, `finit_module`, `delete_module`, `bpf`,
//!   `perf_event_open`).
//!
//! Each step is best-effort. When [`HardeningMode::BestEffort`] is in
//! effect, individual failures are logged and execution continues. When
//! [`HardeningMode::Enforce`] is in effect, any failure propagates as a
//! `ToolError::sandbox_error`. [`HardeningMode::Off`] short-circuits the
//! entire routine.
//!
//! Note what the built-in list does *not* cover: anything a user installed
//! under their home directory. A `uv`, `cargo` or `pnpm` on `PATH` at
//! `~/.local/bin` is found by the shell and then refused by the kernel,
//! surfacing as a bare `Permission denied` with no hint of landlock in it.
//! That is what the declared path lists are for, and why they are honest
//! configuration rather than an inferred default: widening the boundary is
//! the operator's call.
//!
//! This module is reachable only when `target_os = "linux"`: its parent
//! declares it as `#[cfg(target_os = "linux")] pub mod hardening;`. On
//! other platforms, [`super`] provides a no-op `apply` stub with the same
//! signature.

#[cfg(feature = "sandbox-hardening")]
use super::config::HardeningMode;
use super::config::ProcessSandboxConfig;
use crate::tools::ToolError;

#[cfg(feature = "sandbox-hardening")]
pub fn apply(cfg: &ProcessSandboxConfig) -> Result<(), ToolError> {
    if cfg.hardening == HardeningMode::Off {
        return Ok(());
    }
    apply_landlock(cfg)?;
    apply_seccomp(cfg)?;
    Ok(())
}

#[cfg(not(feature = "sandbox-hardening"))]
pub fn apply(_cfg: &ProcessSandboxConfig) -> Result<(), ToolError> {
    Ok(())
}

#[cfg(feature = "sandbox-hardening")]
fn apply_landlock(cfg: &ProcessSandboxConfig) -> Result<(), ToolError> {
    use std::path::Path;

    use landlock::{Access, AccessFs, Ruleset, RulesetAttr, RulesetStatus, ABI};

    let abi = ABI::V1;
    let read_paths: &[&str] = &[
        "/usr",
        "/lib",
        "/lib64",
        "/bin",
        "/sbin",
        "/etc",
        "/sys/kernel",
    ];
    // Character devices a spawned process cannot do without: `/dev/null`
    // alone is opened by every `Stdio::null()` redirect, so without it the
    // child cannot start a subprocess at all. These are granted file-level
    // read+write, which for a character device is the whole of its use.
    let device_paths: &[&str] = &[
        "/dev/null",
        "/dev/zero",
        "/dev/full",
        "/dev/random",
        "/dev/urandom",
    ];

    // Process metadata a shell and the tools it runs read routinely
    // (`/proc/self/*`, `/proc/meminfo`); read-only, and only where it exists.
    let proc_paths: &[&str] = &["/proc"];

    let read_access = AccessFs::from_read(abi);
    let device_access = AccessFs::ReadFile | AccessFs::WriteFile;
    let rw_access = AccessFs::from_all(abi);

    let mut ruleset = match Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .and_then(Ruleset::create)
    {
        Ok(rs) => rs,
        Err(err) => {
            return hardening_failure(cfg, format!("landlock: failed to create ruleset: {err}"));
        }
    };

    for path in read_paths.iter().chain(proc_paths) {
        let path = Path::new(path);
        if !path.exists() {
            continue;
        }
        add_landlock_rule(cfg, &mut ruleset, path, read_access, "read")?;
    }

    for path in device_paths {
        let path = Path::new(path);
        if !path.exists() {
            continue;
        }
        add_landlock_rule(cfg, &mut ruleset, path, device_access, "device")?;
    }

    // Directories the operator declared. A read grant carries `Execute` with
    // it under `AccessFs::from_read`, so one entry covers both finding a
    // binary and running it.
    for path in &cfg.read_exec_paths {
        if !declared_path_exists(path, "read_exec_paths") {
            continue;
        }
        add_landlock_rule(cfg, &mut ruleset, path, read_access, "read-exec")?;
    }

    let mut rw_paths: Vec<String> = Vec::new();
    if let Ok(tmp) = std::env::var("TMPDIR") {
        rw_paths.push(tmp);
    } else {
        rw_paths.push("/tmp".to_string());
    }
    if let Ok(cwd) = std::env::current_dir() {
        rw_paths.push(cwd.to_string_lossy().into_owned());
    }

    for path in &rw_paths {
        let path = Path::new(path);
        if !path.exists() {
            continue;
        }
        add_landlock_rule(cfg, &mut ruleset, path, rw_access, "rw")?;
    }

    for path in &cfg.read_write_paths {
        if !declared_path_exists(path, "read_write_paths") {
            continue;
        }
        add_landlock_rule(cfg, &mut ruleset, path, rw_access, "read-write")?;
    }

    match ruleset.restrict_self() {
        Ok(status) => {
            if status.ruleset == RulesetStatus::NotEnforced {
                return hardening_failure(
                    cfg,
                    "landlock: kernel reports ruleset not enforced".to_string(),
                );
            }
            Ok(())
        }
        Err(err) => hardening_failure(cfg, format!("landlock: restrict_self failed: {err}")),
    }
}

/// Whether a path an operator declared is actually there.
///
/// A missing entry is a typo or a tool that was never installed, not a
/// hardening failure: the ruleset that results is *narrower* than asked for,
/// so it cannot widen the boundary. It is warned about rather than skipped
/// in silence, because an unexplained `Permission denied` later is exactly
/// the failure this configuration exists to prevent — and it is warned about
/// in every mode, so `enforce` does not abort a whole deployment over a
/// directory that has not been created yet.
#[cfg(feature = "sandbox-hardening")]
fn declared_path_exists(path: &std::path::Path, field: &str) -> bool {
    if path.exists() {
        return true;
    }
    tracing::warn!(
        target: "acton_ai::sandbox::process",
        "landlock: {field} names '{}', which does not exist; skipping the rule",
        path.display(),
    );
    false
}

#[cfg(feature = "sandbox-hardening")]
fn add_landlock_rule<A>(
    cfg: &ProcessSandboxConfig,
    ruleset: &mut landlock::RulesetCreated,
    path: &std::path::Path,
    access: A,
    label: &str,
) -> Result<(), ToolError>
where
    A: Into<landlock::BitFlags<landlock::AccessFs>>,
{
    use landlock::{PathBeneath, PathFd, RulesetCreatedAttr};

    let display = path.display();
    match PathFd::new(path) {
        Ok(fd) => {
            if let Err(err) = ruleset.add_rule(PathBeneath::new(fd, access)) {
                hardening_failure(
                    cfg,
                    format!("landlock: failed to add {label} rule for {display}: {err}"),
                )?;
            }
            Ok(())
        }
        Err(err) => hardening_failure(cfg, format!("landlock: failed to open {display}: {err}")),
    }
}

#[cfg(feature = "sandbox-hardening")]
fn apply_seccomp(cfg: &ProcessSandboxConfig) -> Result<(), ToolError> {
    use std::convert::TryInto;

    use seccompiler::{apply_filter, BpfProgram, SeccompAction, SeccompFilter, TargetArch};

    let arch: TargetArch = match std::env::consts::ARCH {
        "x86_64" => TargetArch::x86_64,
        "aarch64" => TargetArch::aarch64,
        other => {
            return hardening_failure(
                cfg,
                format!("seccomp: unsupported target architecture: {other}"),
            );
        }
    };

    // Syscall numbers are architecture-dependent. Resolve them via libc's
    // pre-resolved constants. `libc` is a Linux-only, feature-gated dep.
    let denied: &[i64] = &[
        libc::SYS_ptrace,
        libc::SYS_keyctl,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_reboot,
        libc::SYS_kexec_load,
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
    ];

    let rules = denied
        .iter()
        .map(|nr| (*nr, Vec::new()))
        .collect::<std::collections::BTreeMap<_, _>>();

    let filter = match SeccompFilter::new(
        rules,
        SeccompAction::Allow,
        SeccompAction::Errno(libc::EPERM as u32),
        arch,
    ) {
        Ok(f) => f,
        Err(err) => {
            return hardening_failure(cfg, format!("seccomp: failed to build filter: {err}"));
        }
    };

    let program: BpfProgram = match filter.try_into() {
        Ok(p) => p,
        Err(err) => {
            return hardening_failure(cfg, format!("seccomp: failed to compile filter: {err}"));
        }
    };

    if let Err(err) = apply_filter(&program) {
        return hardening_failure(cfg, format!("seccomp: apply_filter failed: {err}"));
    }

    Ok(())
}

#[cfg(feature = "sandbox-hardening")]
fn hardening_failure(cfg: &ProcessSandboxConfig, message: String) -> Result<(), ToolError> {
    match cfg.hardening {
        HardeningMode::Enforce => Err(ToolError::sandbox_error(message)),
        HardeningMode::BestEffort => {
            tracing::warn!(target: "acton_ai::sandbox::process", "{}", message);
            Ok(())
        }
        HardeningMode::Off => Ok(()),
    }
}

#[cfg(all(test, feature = "sandbox-hardening"))]
mod tests {
    use super::*;

    #[test]
    fn apply_off_is_ok() {
        let cfg = ProcessSandboxConfig::new().with_hardening(HardeningMode::Off);
        assert!(apply(&cfg).is_ok());
    }

    #[test]
    fn apply_does_not_panic_when_off() {
        // Installing landlock/seccomp in the test process would permanently
        // restrict the test runner. HardeningMode::Off exercises the
        // dispatch path without touching the kernel. End-to-end coverage
        // lives in an integration test that spawns a child process.
        let cfg = ProcessSandboxConfig::new().with_hardening(HardeningMode::Off);
        assert!(apply(&cfg).is_ok());
    }
}
