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
//!   `/etc`, `/sys/kernel`) plus read/write access on `$TMPDIR` and the
//!   current working directory;
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
    let read_access = AccessFs::from_read(abi);
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

    for path in read_paths {
        if !Path::new(path).exists() {
            continue;
        }
        add_landlock_rule(cfg, &mut ruleset, path, read_access, "read")?;
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
        if !Path::new(path).exists() {
            continue;
        }
        add_landlock_rule(cfg, &mut ruleset, path, rw_access, "rw")?;
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

#[cfg(feature = "sandbox-hardening")]
fn add_landlock_rule<A>(
    cfg: &ProcessSandboxConfig,
    ruleset: &mut landlock::RulesetCreated,
    path: &str,
    access: A,
    label: &str,
) -> Result<(), ToolError>
where
    A: Into<landlock::BitFlags<landlock::AccessFs>>,
{
    use landlock::{PathBeneath, PathFd, RulesetCreatedAttr};

    match PathFd::new(path) {
        Ok(fd) => {
            if let Err(err) = ruleset.add_rule(PathBeneath::new(fd, access)) {
                hardening_failure(
                    cfg,
                    format!("landlock: failed to add {label} rule for {path}: {err}"),
                )?;
            }
            Ok(())
        }
        Err(err) => hardening_failure(cfg, format!("landlock: failed to open {path}: {err}")),
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
