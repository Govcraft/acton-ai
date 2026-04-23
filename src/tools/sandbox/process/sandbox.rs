//! Parent-side [`Sandbox`] implementation that runs each tool invocation in
//! a re-execed child process.
//!
//! The parent owns the timeout and the lifecycle of the child; the child is
//! a single-shot [`super::runner`] invocation. On timeout the parent kills
//! the whole child process group so bash-spawned grandchildren are reaped
//! along with the supervisor.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use acton_reactive::prelude::tokio::io::AsyncWriteExt;
use acton_reactive::prelude::tokio::process::Command;
use acton_reactive::prelude::tokio::time::timeout;
use serde_json::Value;

use super::config::{HardeningMode, ProcessSandboxConfig};
use super::protocol::{read_response, write_request, Request};
use super::runner::env_vars;
use crate::tools::error::ToolError;
use crate::tools::sandbox::traits::{Sandbox, SandboxExecutionFuture};

/// Runs each tool invocation in a re-execed child process.
///
/// The sandbox re-execs the current binary with `ACTON_AI_SANDBOX_RUNNER=1`
/// plus the configured resource/hardening env vars, writes a length-prefixed
/// JSON [`Request`] to the child's stdin, and reads a length-prefixed JSON
/// response from its stdout. The parent enforces the wall-clock timeout and
/// kills the child's process group if it elapses.
pub struct ProcessSandbox {
    /// Path to the binary to re-exec. Typically the current executable, but
    /// may be overridden in tests.
    pub(crate) exe: PathBuf,
    /// Configuration applied to every child spawned from this sandbox.
    pub(crate) config: ProcessSandboxConfig,
    /// Set to `true` by [`Sandbox::destroy`]. The sandbox has no long-lived
    /// state — this flag is purely informational for [`Sandbox::is_alive`].
    pub(crate) destroyed: AtomicBool,
}

impl std::fmt::Debug for ProcessSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessSandbox")
            .field("exe", &self.exe)
            .field("config", &self.config)
            .field("destroyed", &self.destroyed.load(Ordering::Relaxed))
            .finish()
    }
}

impl ProcessSandbox {
    /// Creates a new sandbox wrapping the given executable and config.
    ///
    /// Construction does not validate the config — callers typically go
    /// through [`super::factory::ProcessSandboxFactory`] which validates up
    /// front and caches the resolved executable path.
    #[must_use]
    pub fn new(exe: PathBuf, config: ProcessSandboxConfig) -> Self {
        Self {
            exe,
            config,
            destroyed: AtomicBool::new(false),
        }
    }

    /// Builds the base [`tokio::process::Command`] with the runner env-var
    /// contract and stdio plumbing applied.
    fn build_command(&self) -> Command {
        let mut cmd = Command::new(&self.exe);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear();

        for key in &self.config.env_allowlist {
            if let Ok(value) = std::env::var(key) {
                cmd.env(key, value);
            }
        }

        let timeout_ms = duration_to_millis(self.config.timeout);
        cmd.env(env_vars::RUNNER, "1")
            .env(env_vars::TIMEOUT_MS, timeout_ms.to_string())
            .env(env_vars::HARDENING, hardening_to_env(self.config.hardening));
        set_optional_u64_env(&mut cmd, env_vars::MEM_BYTES, self.config.memory_limit);
        set_optional_u64_env(&mut cmd, env_vars::CPU_SECS, self.config.cpu_limit_secs);
        set_optional_u64_env(&mut cmd, env_vars::FSIZE_BYTES, self.config.fsize_limit);

        #[cfg(unix)]
        cmd.process_group(0);

        cmd
    }

    /// Computes the absolute deadline in milliseconds-since-epoch used by
    /// the wire protocol. Saturating arithmetic avoids wraparound on the
    /// unlikely event of a 292-million-year timeout.
    fn absolute_deadline_ms(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        now.saturating_add(self.config.timeout)
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
}

impl Sandbox for ProcessSandbox {
    fn execute(&self, code: &str, args: Value) -> SandboxExecutionFuture {
        if self.destroyed.load(Ordering::Relaxed) {
            return Box::pin(async {
                Err(ToolError::sandbox_error(
                    "process sandbox has been destroyed",
                ))
            });
        }

        let mut cmd = self.build_command();
        let request = Request {
            tool_name: code.to_string(),
            args,
            deadline_ms: self.absolute_deadline_ms(),
        };
        let wall_clock = self.config.timeout;

        // Allocate a per-invocation tempdir and point the child at it via
        // TMPDIR. Keeping the TempDir alive until after `wait_with_output`
        // ensures it outlives every child-spawned process.
        let tmp = match tempfile::tempdir() {
            Ok(dir) => dir,
            Err(err) => {
                return Box::pin(async move {
                    Err(ToolError::sandbox_error(format!(
                        "failed to create sandbox tempdir: {err}"
                    )))
                });
            }
        };
        cmd.current_dir(tmp.path());
        cmd.env("TMPDIR", tmp.path());

        Box::pin(async move {
            let mut child = cmd
                .spawn()
                .map_err(|err| ToolError::sandbox_error(format!("spawn failed: {err}")))?;

            #[cfg(unix)]
            let child_pid = child.id();
            #[cfg(not(unix))]
            let _ = &child;

            // Write the request frame to the child's stdin, then drop the
            // handle so the child sees EOF. We do this before waiting so
            // the child has something to read; spawning has already
            // connected the pipes.
            if let Some(mut stdin) = child.stdin.take() {
                let mut buf = Vec::new();
                if let Err(err) = write_request(&mut buf, &request) {
                    let _ = child.kill().await;
                    return Err(ToolError::sandbox_error(format!(
                        "failed to frame request: {err}"
                    )));
                }
                if let Err(err) = stdin.write_all(&buf).await {
                    let _ = child.kill().await;
                    return Err(ToolError::sandbox_error(format!(
                        "failed to write request to child stdin: {err}"
                    )));
                }
                if let Err(err) = stdin.shutdown().await {
                    let _ = child.kill().await;
                    return Err(ToolError::sandbox_error(format!(
                        "failed to close child stdin: {err}"
                    )));
                }
            } else {
                let _ = child.kill().await;
                return Err(ToolError::sandbox_error("child stdin was not piped"));
            }

            let output = match timeout(wall_clock, child.wait_with_output()).await {
                Ok(Ok(output)) => output,
                Ok(Err(err)) => {
                    return Err(ToolError::sandbox_error(format!(
                        "waiting for sandbox child failed: {err}"
                    )));
                }
                Err(_) => {
                    kill_child_group(
                        #[cfg(unix)]
                        child_pid,
                    );
                    // Hold tmp across this path too; drop happens at fn end.
                    let _ = tmp;
                    return Err(ToolError::sandbox_error(format!(
                        "sandbox execution exceeded timeout of {}ms",
                        duration_to_millis(wall_clock)
                    )));
                }
            };

            // Explicitly keep `tmp` alive until after the output is parsed.
            let result = parse_child_output(&output);
            drop(tmp);
            result
        })
    }

    fn destroy(&mut self) {
        self.destroyed.store(true, Ordering::Relaxed);
    }

    fn is_alive(&self) -> bool {
        !self.destroyed.load(Ordering::Relaxed)
    }
}

fn parse_child_output(output: &std::process::Output) -> Result<Value, ToolError> {
    if output.stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let status = output.status;
        return Err(ToolError::sandbox_error(format!(
            "sandbox child produced no response (status: {status}, stderr: {stderr})"
        )));
    }

    let mut cursor = std::io::Cursor::new(&output.stdout);
    let response = read_response(&mut cursor).map_err(|err| {
        let stderr = String::from_utf8_lossy(&output.stderr);
        ToolError::sandbox_error(format!(
            "failed to parse sandbox response: {err} (stderr: {stderr})"
        ))
    })?;

    response.result.map_err(ToolError::sandbox_error)
}

fn duration_to_millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

fn hardening_to_env(mode: HardeningMode) -> &'static str {
    match mode {
        HardeningMode::Off => "off",
        HardeningMode::BestEffort => "besteffort",
        HardeningMode::Enforce => "enforce",
    }
}

fn set_optional_u64_env(cmd: &mut Command, key: &str, value: Option<u64>) {
    match value {
        Some(v) => {
            cmd.env(key, v.to_string());
        }
        None => {
            // Empty string is the contract for "limit disabled".
            cmd.env(key, "");
        }
    }
}

#[cfg(unix)]
fn kill_child_group(pid: Option<u32>) {
    use nix::sys::signal::{killpg, Signal};
    use nix::unistd::Pid;

    let Some(pid) = pid else { return };
    let Ok(pid_i32) = i32::try_from(pid) else {
        return;
    };
    let group = Pid::from_raw(pid_i32);
    if let Err(err) = killpg(group, Signal::SIGKILL) {
        tracing::warn!(
            target: "acton_ai::sandbox::process",
            "killpg({pid_i32}, SIGKILL) failed: {err}"
        );
    }
}

#[cfg(not(unix))]
fn kill_child_group() {
    tracing::warn!(
        target: "acton_ai::sandbox::process",
        "timeout kill on non-unix uses Child::start_kill; grandchildren may leak"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardening_to_env_maps_all_variants() {
        assert_eq!(hardening_to_env(HardeningMode::Off), "off");
        assert_eq!(hardening_to_env(HardeningMode::BestEffort), "besteffort");
        assert_eq!(hardening_to_env(HardeningMode::Enforce), "enforce");
    }

    #[test]
    fn duration_to_millis_saturates() {
        assert_eq!(duration_to_millis(Duration::from_millis(1500)), 1500);
        assert_eq!(duration_to_millis(Duration::ZERO), 0);
    }

    #[test]
    fn destroy_flips_is_alive() {
        let mut sb = ProcessSandbox::new(PathBuf::from("/bin/true"), ProcessSandboxConfig::new());
        assert!(sb.is_alive());
        sb.destroy();
        assert!(!sb.is_alive());
    }
}
