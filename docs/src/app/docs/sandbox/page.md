---
title: Process Sandbox
---

Acton AI isolates dangerous tool calls -- shell commands, file writes, and untrusted code paths -- inside a `ProcessSandbox`. The sandbox is a portable, subprocess-based isolation layer with timeouts, resource limits, and (on Linux) best-effort kernel hardening. It replaces the earlier Hyperlight micro-VM experiment, which pinned releases to `x86_64-linux` and did not actually contain untrusted work: Hyperlight's guest shims just RPC'd back to the host.

---

## How it works

Sandboxed tool calls flow through three steps:

1. **Re-exec.** The parent process spawns a copy of the current binary (`std::env::current_exe()`) with the environment variable `ACTON_AI_SANDBOX_RUNNER=1` and a minimal allowlisted environment.
2. **Protocol.** The child sees the env-var at the top of `main()` and hands control to the sandbox runner. A length-prefixed JSON request is piped in on stdin: `{ tool_name, args, deadline_ms }`. The response is a length-prefixed JSON envelope on stdout: `{ Ok: value } | { Err: message }`.
3. **Enforcement.** Before touching the request, the child applies `setrlimit` ceilings for address space, CPU time, and file size. On Linux, if the `sandbox-hardening` feature is built in, it also installs a `landlock` ruleset (filesystem access filter) and a `seccompiler` filter (syscall allowlist) before dispatching the tool. The parent enforces a wall-clock deadline via `tokio::time::timeout` and kills the child's entire process group on overrun, so shell grandchildren cannot survive the deadline.

The abstraction -- `Sandbox` and `SandboxFactory` -- is unchanged from the previous implementation. The `ToolRegistry` calls `factory.create() -> sandbox.execute() -> sandbox.destroy()` as before; only the implementation swapped.

---

## Hardening modes

`HardeningMode` controls what the Linux child does before running the tool. All modes always apply rlimits; the mode governs the landlock/seccomp layer.

| Mode | Behavior |
|---|---|
| `Off` | rlimits only. Useful for local debugging when landlock/seccomp are interfering with a legitimate tool. Unsafe for production. |
| `BestEffort` *(default)* | Try to install landlock + seccomp. On kernels without support (pre-5.13) or in containers where the sandbox syscalls are blocked, log a warning and continue with rlimits-only. |
| `Enforce` | Install landlock + seccomp; if either setup step fails, abort the child before the tool runs. Recommended for production workloads on supported kernels. |

On macOS and Windows the hardening layer is compiled out. The child still applies rlimits where the platform supports them (`RLIMIT_DATA` on macOS; Windows currently enforces timeout + process termination only).

---

## What the sandbox protects against

- **Accidental deletion or modification of host files outside the working tree.** The default landlock ruleset gives the child read access to `/usr`, `/lib`, `/bin`, `/etc`, and read-write on `$TMPDIR`. Writes outside those roots are rejected by the kernel.
- **Runaway resource use.** Address space, CPU time, wall-clock time, and output file size all have ceilings; exceeding any of them kills the child.
- **Process escape via backgrounded shell children.** The child is launched in its own process group; timeout kills the group, not just the direct child.
- **Most ambient syscall abuse.** The seccomp filter denies `ptrace`, `keyctl`, `mount`, `reboot`, `kexec_load`, and similar capabilities.

## What the sandbox does **not** protect against

Be honest about the threat model:

- **Network isolation is not applied by default.** If your tool reaches the network (e.g., `bash` running `curl`), it can still reach any host the parent can. Use the `env_allowlist` to strip proxy variables if that matters, and consider running the whole process inside a network namespace if you need hard network isolation.
- **Kernel exploits.** A ProcessSandbox is not a VM. A Linux kernel bug reachable from userspace is reachable from the child. If you need VM-level isolation for truly adversarial workloads, run the whole agent inside a VM or microVM yourself.
- **The host filesystem you explicitly allow.** Tools that need to read your source tree will read your source tree. The sandbox limits blast radius; it does not sign or audit the tool call.
- **Covert timing/side channels.** CPU-time and memory limits stop runaway loops, not careful side-channel probes.

The goal is to make an LLM issuing `rm -rf ~/` expensive and obvious, not to run Pegasus in a cage.

---

## Configuration via TOML

The `[sandbox]` section in `acton-ai.toml` maps to `ProcessSandboxConfig`:

```toml
[sandbox]
hardening = "besteffort"    # "off" | "besteffort" | "enforce"

[sandbox.limits]
max_execution_ms = 30000    # wall-clock deadline, default 30s
max_memory_mb = 256         # RLIMIT_AS / RLIMIT_DATA, default 256 MB
```

Unknown keys are ignored, so TOMLs carrying the retired Hyperlight-pool fields (`pool_warmup`, `pool_max_per_type`, `max_executions_before_recycle`) still load cleanly. They are no-ops under the process sandbox.

---

## Rust API

Enable the sandbox through the builder. The default configuration is appropriate for most applications:

```rust
use acton_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    let runtime = ActonAI::builder()
        .app_name("my-app")
        .ollama("qwen2.5:7b")
        .with_builtin_tools(&["bash"])
        .with_process_sandbox()
        .launch()
        .await?;

    runtime
        .prompt("What is today's date?")
        .system("Use the bash tool to run `date`.")
        .collect()
        .await?;

    Ok(())
}
```

For explicit control over limits or hardening:

```rust
use acton_ai::prelude::*;
use acton_ai::tools::sandbox::{HardeningMode, ProcessSandboxConfig};
use std::time::Duration;

let cfg = ProcessSandboxConfig::new()
    .with_timeout(Duration::from_secs(60))
    .with_memory_limit(Some(512 * 1024 * 1024))
    .with_cpu_limit_secs(Some(30))
    .with_hardening(HardeningMode::Enforce);

let runtime = ActonAI::builder()
    .app_name("my-app")
    .ollama("qwen2.5:7b")
    .with_builtins()
    .with_process_sandbox_config(cfg)
    .launch()
    .await?;
```

---

## Platform support

| Platform | Rlimits | Landlock | Seccomp | Process-group kill |
|---|---|---|---|---|
| Linux x86_64 / aarch64 | Yes | Yes (5.13+, best-effort) | Yes (best-effort) | Yes |
| macOS x86_64 / aarch64 | Partial (`RLIMIT_DATA`) | No | No | Yes |
| Windows x86_64 | No | No | No | Timeout + taskkill |

`ProcessSandbox` builds and runs on all five targets. Hardening features are compiled out on non-Linux platforms via `cfg` gates, so no code paths are unreachable.

---

## Next steps

- [Secure Tool Execution](/docs/secure-tool-execution) -- path validation for filesystem tools
- [Error Handling](/docs/error-handling) -- handle `ToolError::SandboxError` variants
- [Testing Your Agents](/docs/testing) -- use `StubSandbox` for deterministic tests
