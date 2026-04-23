# Changelog

All notable changes to this project are documented in this file. The project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Breaking changes

- Removed the Hyperlight sandbox. The sandbox abstraction is now backed by
  `ProcessSandbox`, a portable subprocess-based sandbox with OS-level
  hardening where available (landlock + seccomp on Linux kernels 5.13+).
- Removed builder methods `with_hyperlight_sandbox`,
  `with_hyperlight_sandbox_config`, `with_sandbox_pool`, and
  `with_sandbox_pool_config`. Use `with_process_sandbox` or
  `with_process_sandbox_config` instead.
- Removed the `rust_code` builtin tool. It relied on Hyperlight's hardware
  isolation to execute user-generated compiled Rust; no coherent portable
  replacement exists yet. Track the follow-up issue if you need it.
- Sandboxed builtin tools now actually execute through the sandbox factory
  when one is configured. Previously the sandbox plumbing existed but was
  never reached through the facade API — `.with_hyperlight_sandbox()`
  silently ran tools in-process.
- Replaced `SandboxFileConfig` TOML keys `pool_warmup`, `pool_max_per_type`,
  and `max_executions_before_recycle` with `hardening` (values: `"off"`,
  `"besteffort"`, `"enforce"`). Old keys are ignored rather than rejected,
  so existing TOMLs still parse.

### Added

- `ProcessSandbox` implementation under `src/tools/sandbox/process/`. The
  parent re-execs the current binary as a child with `ACTON_AI_SANDBOX_RUNNER=1`,
  exchanges length-prefixed JSON over stdin/stdout, enforces a wall-clock
  timeout, and kills the child's process group on overrun. The child
  applies `setrlimit` ceilings (address space, CPU, file size) before
  dispatching the requested tool.
- New `sandbox-hardening` Cargo feature (default-enabled on Linux,
  compiled out elsewhere). When active, the child additionally applies a
  best-effort `landlock` ruleset and a `seccompiler` filter before running
  user-provided tool arguments. On kernels without landlock/seccomp
  support the feature logs a warning and falls back to rlimits-only.
- Builder methods `with_process_sandbox()` and
  `with_process_sandbox_config(ProcessSandboxConfig)` on `ActonAIBuilder`.
- `examples/process_sandbox.rs` replaces the retired `bash_sandbox`
  example and demonstrates sandboxed bash tool calls end-to-end.

### Changed

- Release and CI workflows now target Linux (x86_64 + aarch64), macOS
  (Intel + Apple Silicon), and Windows x86_64. The previous `x86_64-linux`
  hard-scoping (required by Hyperlight's KVM dependency) is gone.

### Internal

- Deleted `guests/` workspace (hyperlight no_std guest binaries:
  `shell_guest`, `http_guest`).
- Deleted `src/tools/sandbox/hyperlight/` and `src/tools/compiler/`.
- Collapsed `build.rs` to a no-op; guest compilation is no longer part of
  the build.
- Dropped the `hyperlight-host = "0.12"` dependency.
