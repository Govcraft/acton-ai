//! Child-side entry point for the process sandbox.
//!
//! The parent spawns the same binary with `ACTON_AI_SANDBOX_RUNNER=1` set; the
//! binary's `main` detects that env var and calls [`main`] before the normal
//! CLI runtime is initialized. This keeps the actor runtime out of the child
//! entirely — the child is a bounded, one-shot tool executor.
//!
//! Contract between parent and child:
//!
//! - Parent sets `ACTON_AI_SANDBOX_RUNNER=1`, plus the `ACTON_AI_SANDBOX_*`
//!   env vars documented in [`env_vars`].
//! - Parent writes one length-prefixed [`Request`] to stdin, then closes
//!   stdin.
//! - Child applies resource limits, then OS hardening, then dispatches the
//!   request to the in-process tool executor.
//! - Child writes one length-prefixed [`Response`] to stdout and exits.
//!   Exit code `0` means the child completed its lifecycle (regardless of
//!   whether the tool succeeded); `1` means the child itself failed before
//!   it could produce a response.
//!
//! ## Testability
//!
//! Everything tied to stdin/stdout/env-vars/exit lives in [`main`]. The core
//! dispatch logic is exposed as [`dispatch`] so unit tests can exercise it
//! without spawning a subprocess.

use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Duration;

use acton_reactive::prelude::tokio::runtime::Builder;
use serde_json::Value;

use super::config::{HardeningMode, ProcessSandboxConfig, DEFAULT_ENV_ALLOWLIST};
use super::hardening;
use super::protocol::{read_request, write_response, Request, Response};
use crate::tools::builtins::{BashTool, EditFileTool, WriteFileTool};
use crate::tools::{ToolError, ToolExecutorTrait};

/// Names of the environment variables the parent sets for the child.
pub mod env_vars {
    /// Signals to the binary entry point that it should run in sandbox mode.
    pub const RUNNER: &str = "ACTON_AI_SANDBOX_RUNNER";
    /// Wall-clock timeout in milliseconds. Informational — the parent owns
    /// the true timeout via `tokio::time::timeout`.
    pub const TIMEOUT_MS: &str = "ACTON_AI_SANDBOX_TIMEOUT_MS";
    /// Address-space / virtual-memory ceiling in bytes.
    pub const MEM_BYTES: &str = "ACTON_AI_SANDBOX_MEM_BYTES";
    /// CPU-time ceiling in seconds.
    pub const CPU_SECS: &str = "ACTON_AI_SANDBOX_CPU_SECS";
    /// Maximum file size the child may create, in bytes.
    pub const FSIZE_BYTES: &str = "ACTON_AI_SANDBOX_FSIZE_BYTES";
    /// Hardening mode: `"off"`, `"besteffort"`, or `"enforce"`.
    pub const HARDENING: &str = "ACTON_AI_SANDBOX_HARDENING";
    /// The single directory the child's tools may read and write.
    ///
    /// Absent means unconfined, which for the child amounts to the
    /// throwaway directory it was started in.
    pub const ROOT: &str = "ACTON_AI_SANDBOX_ROOT";
}

/// Detours into the sandbox runner when this process was re-exec'd as a
/// sandbox child, and returns immediately otherwise.
///
/// The process sandbox re-execs the *current binary* and relies on that
/// binary routing into [`main`] when `ACTON_AI_SANDBOX_RUNNER=1` is set —
/// `acton-ai`'s own entry point does exactly this. A downstream embedder
/// whose binary hosts the framework as a library must uphold the same
/// contract, or the re-exec'd child will run their whole application against
/// the sandbox protocol. This helper is that contract in one call: put it at
/// the very top of `main()`, before any runtime, socket, or CLI setup.
///
/// ```rust,ignore
/// fn main() {
///     // Must run first: the sandbox child is a bounded one-shot executor
///     // and must never boot the application around it.
///     acton_ai::tools::sandbox::process::runner::run_if_sandbox_child();
///
///     my_daemon::main();
/// }
/// ```
pub fn run_if_sandbox_child() {
    let is_child = std::env::var(env_vars::RUNNER).ok().as_deref() == Some("1");
    if is_child {
        main();
    }
}

/// Whether the sandbox runner can dispatch `tool_name`.
///
/// The child executes a fixed set of built-ins — the sandboxed ones — and
/// nothing else: custom `#[tool]` functions and MCP tools cannot cross the
/// process boundary. Embedders routing work through
/// [`SandboxedExecution`](crate::tools::sandbox::SandboxedExecution) use this
/// to decide, before spawning a child, whether a name will resolve inside it.
#[must_use]
pub fn supports(tool_name: &str) -> bool {
    sandbox_tool(tool_name, None).is_some()
}

/// The single source of truth for what the child can run.
///
/// [`supports`] and [`dispatch_async`] both consult this, so the advertised
/// set and the dispatched set cannot drift apart.
///
/// `root`, when the parent named one, is the only directory these tools may
/// touch — the same boundary the caller's session is held to, carried across
/// the process edge rather than re-derived from wherever the child landed.
fn sandbox_tool(tool_name: &str, root: Option<&Path>) -> Option<Box<dyn ToolExecutorTrait>> {
    let root = root.map(Path::to_path_buf);
    match tool_name {
        "bash" => Some(Box::new(match root {
            Some(root) => BashTool::new().with_root(root),
            None => BashTool::new(),
        }) as Box<dyn ToolExecutorTrait>),
        "write_file" => Some(Box::new(match root {
            Some(root) => WriteFileTool::new().with_root(root),
            None => WriteFileTool::new(),
        }) as Box<dyn ToolExecutorTrait>),
        "edit_file" => Some(Box::new(match root {
            Some(root) => EditFileTool::new().with_root(root),
            None => EditFileTool::new(),
        }) as Box<dyn ToolExecutorTrait>),
        _ => None,
    }
}

/// The directory the parent confined this child to, if any.
fn root_from_env() -> Option<PathBuf> {
    std::env::var_os(env_vars::ROOT).map(PathBuf::from)
}

/// Child entry point.
///
/// Reads one [`Request`] from stdin, executes it against the in-process tool
/// registry, writes one [`Response`] to stdout, and terminates the process.
///
/// # Panics
///
/// This function never returns. Fatal errors before the response is emitted
/// cause the process to exit with code `1` after best-effort emitting an
/// error response.
pub fn main() -> ! {
    let cfg = load_config_from_env();

    if let Err(err) = apply_rlimits(&cfg) {
        exit_with_error(&format!("sandbox: failed to apply resource limits: {err}"));
    }

    if let Err(err) = hardening::apply(&cfg) {
        // hardening::apply already consults cfg.hardening to decide between
        // error-out and warn-and-continue; if it returns Err, we must abort.
        exit_with_error(&format!("sandbox: hardening failed: {err}"));
    }

    let request = match read_request(&mut io::stdin().lock()) {
        Ok(req) => req,
        Err(err) => exit_with_error(&format!("sandbox: failed to read request: {err}")),
    };

    let response = dispatch(request);
    emit_response(&response);

    std::process::exit(0);
}

/// Pure dispatcher: maps a [`Request`] to a [`Response`] by running the
/// named tool against its in-process executor.
///
/// Unknown tool names yield an `Err` response; tool-level failures are also
/// mapped to `Err` responses rather than propagating up the stack.
///
/// This function owns a minimal `current_thread` tokio runtime so it can
/// drive the async `ToolExecutorTrait::execute` without requiring the caller
/// to already be inside an async context. The runtime is built once per
/// invocation and dropped before the function returns.
#[must_use]
pub fn dispatch(req: Request) -> Response {
    let runtime = match Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(err) => {
            return Response::err(format!("sandbox: failed to build tokio runtime: {err}"));
        }
    };

    runtime.block_on(dispatch_async(req))
}

/// Async core of [`dispatch`]. Split out so callers that already have a
/// tokio runtime (notably tests) can `await` it directly.
pub async fn dispatch_async(req: Request) -> Response {
    let Request {
        tool_name, args, ..
    } = req;

    let result: Result<Value, ToolError> =
        match sandbox_tool(&tool_name, root_from_env().as_deref()) {
            Some(tool) => tool.execute(args).await,
            None => Err(ToolError::not_found(&tool_name)),
        };

    match result {
        Ok(value) => Response::ok(value),
        Err(err) => Response::err(err.to_string()),
    }
}

/// Reconstructs a [`ProcessSandboxConfig`] from the env-var contract set by
/// the parent.
///
/// Missing values fall through to the defaults defined in [`super::config`];
/// malformed values are treated as missing with a log line so the child can
/// still make forward progress. The true enforcement source-of-truth is the
/// parent — the child uses these only for its own resource limits + hardening.
fn load_config_from_env() -> ProcessSandboxConfig {
    let mut cfg = ProcessSandboxConfig::default();

    if let Some(ms) = parse_u64(env_vars::TIMEOUT_MS) {
        cfg.timeout = Duration::from_millis(ms);
    }
    cfg.memory_limit = parse_optional_u64(env_vars::MEM_BYTES, cfg.memory_limit);
    cfg.cpu_limit_secs = parse_optional_u64(env_vars::CPU_SECS, cfg.cpu_limit_secs);
    cfg.fsize_limit = parse_optional_u64(env_vars::FSIZE_BYTES, cfg.fsize_limit);

    if let Ok(raw) = std::env::var(env_vars::HARDENING) {
        cfg.hardening = parse_hardening_mode(&raw).unwrap_or(cfg.hardening);
    }

    // The env allowlist was used by the parent to decide what to forward;
    // the child just defaults it back to the canonical list for any code
    // paths that inspect it. (Hardening does not consume it.)
    cfg.env_allowlist = DEFAULT_ENV_ALLOWLIST
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    cfg
}

fn parse_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse::<u64>().ok())
}

/// Reads an `Option<u64>` from the environment. An unset variable leaves the
/// existing value untouched; an explicitly empty value means "disable the
/// limit" (`None`); any other value is parsed as `Some(u64)` with malformed
/// inputs falling back to the existing value.
fn parse_optional_u64(key: &str, fallback: Option<u64>) -> Option<u64> {
    match std::env::var(key) {
        Ok(raw) if raw.is_empty() => None,
        Ok(raw) => raw.parse::<u64>().ok().map(Some).unwrap_or(fallback),
        Err(_) => fallback,
    }
}

fn parse_hardening_mode(raw: &str) -> Option<HardeningMode> {
    match raw {
        "off" => Some(HardeningMode::Off),
        "besteffort" => Some(HardeningMode::BestEffort),
        "enforce" => Some(HardeningMode::Enforce),
        _ => None,
    }
}

/// Applies OS resource limits to the current process.
///
/// On Windows this is a no-op (no `setrlimit`).
#[cfg(unix)]
fn apply_rlimits(cfg: &ProcessSandboxConfig) -> Result<(), String> {
    use rlimit::{setrlimit, Resource};

    if let Some(bytes) = cfg.memory_limit {
        // RLIMIT_AS covers the full virtual-memory footprint on Linux and is
        // enforced there, so failure to apply it is fatal. macOS rejects any
        // finite RLIMIT_AS with EINVAL; RLIMIT_DATA is the closest knob it
        // has and is applied best-effort — the wall-clock deadline remains
        // the hard backstop on that platform.
        #[cfg(target_os = "macos")]
        let _ = setrlimit(Resource::DATA, bytes, bytes);
        #[cfg(not(target_os = "macos"))]
        set_limit(Resource::AS, bytes, "RLIMIT_AS")?;
    }
    if let Some(secs) = cfg.cpu_limit_secs {
        set_limit(Resource::CPU, secs, "RLIMIT_CPU")?;
    }
    if let Some(bytes) = cfg.fsize_limit {
        set_limit(Resource::FSIZE, bytes, "RLIMIT_FSIZE")?;
    }
    let _ = setrlimit;
    Ok(())
}

#[cfg(unix)]
fn set_limit(resource: rlimit::Resource, value: u64, label: &str) -> Result<(), String> {
    rlimit::setrlimit(resource, value, value).map_err(|err| format!("{label}: {err}"))
}

#[cfg(not(unix))]
fn apply_rlimits(_cfg: &ProcessSandboxConfig) -> Result<(), String> {
    Ok(())
}

fn emit_response(resp: &Response) {
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    if let Err(err) = write_response(&mut writer, resp) {
        // Nothing to do other than crash — the parent will see a missing
        // frame and surface it as a sandbox error.
        eprintln!("sandbox: failed to write response: {err}");
        std::process::exit(1);
    }
}

fn exit_with_error(message: &str) -> ! {
    let resp = Response::err(message.to_string());
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    // Best-effort: if writing fails we've already lost the channel.
    let _ = write_response(&mut writer, &resp);
    eprintln!("{message}");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dispatch_unknown_tool_returns_err() {
        let req = Request {
            tool_name: "nope".to_string(),
            args: json!({}),
            deadline_ms: 0,
        };
        let resp = dispatch(req);
        let err = resp.result.expect_err("unknown tool must fail");
        assert!(
            err.contains("nope"),
            "error should mention the unknown tool, got: {err}"
        );
    }

    // Needs a real POSIX bash on PATH, which Windows does not guarantee.
    #[cfg(unix)]
    #[test]
    fn dispatch_bash_echo_succeeds() {
        let req = Request {
            tool_name: "bash".to_string(),
            args: json!({"command": "echo hello"}),
            deadline_ms: 0,
        };
        let resp = dispatch(req);
        let value = resp
            .result
            .expect("bash echo should succeed in a healthy environment");
        let stdout = value
            .get("stdout")
            .and_then(Value::as_str)
            .expect("bash response must include stdout");
        assert!(
            stdout.contains("hello"),
            "stdout should contain echoed text, got: {stdout}"
        );
    }

    #[test]
    fn dispatch_write_file_creates_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("note.txt");
        let req = Request {
            tool_name: "write_file".to_string(),
            args: json!({
                "path": path.to_str().unwrap(),
                "content": "sandbox ok",
            }),
            deadline_ms: 0,
        };
        let resp = dispatch(req);
        resp.result
            .as_ref()
            .expect("write_file should succeed when path is writable");
        let written = std::fs::read_to_string(&path).expect("file should exist");
        assert_eq!(written, "sandbox ok");
    }

    #[test]
    fn parse_optional_u64_empty_means_none() {
        // Exercise the pure parser without touching process env; we use
        // temp-env-style direct std::env calls scoped to a unique key.
        let key = "ACTON_AI_TEST_PARSE_OPTIONAL";
        // SAFETY: single-threaded test; no other readers of this env var.
        unsafe {
            std::env::set_var(key, "");
        }
        assert_eq!(parse_optional_u64(key, Some(42)), None);
        unsafe {
            std::env::set_var(key, "123");
        }
        assert_eq!(parse_optional_u64(key, Some(42)), Some(123));
        unsafe {
            std::env::set_var(key, "not-a-number");
        }
        assert_eq!(parse_optional_u64(key, Some(42)), Some(42));
        unsafe {
            std::env::remove_var(key);
        }
        assert_eq!(parse_optional_u64(key, Some(42)), Some(42));
    }

    #[test]
    fn parse_hardening_mode_handles_all_variants() {
        assert_eq!(parse_hardening_mode("off"), Some(HardeningMode::Off));
        assert_eq!(
            parse_hardening_mode("besteffort"),
            Some(HardeningMode::BestEffort)
        );
        assert_eq!(
            parse_hardening_mode("enforce"),
            Some(HardeningMode::Enforce)
        );
        assert_eq!(parse_hardening_mode("BOGUS"), None);
    }
}
