# Changelog

All notable changes to this project are documented in this file. The project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Breaking changes

- `CostAccountant::spawn` takes a third argument: the optional `BudgetConfig`
  to enforce. Facade users are unaffected.
- `UsageSnapshot` gains a `budget` field, so struct literals of it need
  updating. Field access and the accessor methods are unchanged.

### Added

- OpenTelemetry export behind the new `otel` feature, which joins the default
  set. Every prompt loop run becomes an `acton_ai.turn` span with an
  `acton_ai.llm_round` child per provider dispatch and an `acton_ai.tool`
  child per tool execution, parented explicitly rather than through ambient
  context. Metrics cover tokens (`acton_ai.tokens`, split by kind), requests,
  rate limits, budget events, round latency, time to first token, and tool
  duration.
- `ActonAIBuilder::telemetry_otlp("http://localhost:4318")` for the common
  case, `ActonAIBuilder::telemetry(Telemetry::otlp(..).service_name(..)
  .metrics_interval_secs(..).header(..))` for the rest, and a `[telemetry]`
  TOML section with the same shape. A builder-set telemetry replaces the TOML
  section wholesale.
- `ActonAIBuilder::telemetry_from_globals()` emits into OpenTelemetry
  providers the surrounding application already installed, instead of
  installing competing ones. That application keeps the lifecycle, so
  `shutdown()` deliberately neither flushes nor shuts those providers down.
- `ActonAIBuilder::telemetry_guard(guard)` plus the public
  `telemetry::install_with_exporters` assemble the providers around any
  exporters you supply — a different protocol, a file, an in-memory recorder —
  with this runtime owning their lifecycle.
- `ActonAI::shutdown` flushes telemetry **after** the actors stop, so the
  final broadcasts are recorded rather than dropped with the last batch.
- `ActonAI::provider_model(name)` exposes the model a configured provider
  serves.
- Tool arguments and results are never recorded on spans, and correlation and
  agent IDs are recorded on spans only, never as metric attributes. Both are
  deliberate: the first is unbounded user data, the second would create one
  time series per request.
- The OTLP transport is HTTP/protobuf over reqwest's blocking client. No gRPC
  or tonic stack is pulled in, and the blocking client is required rather than
  incidental — `BatchSpanProcessor` and `PeriodicReader` export from a plain
  `std::thread` where an async client would have no reactor.
- A `[telemetry]` section parses in builds without the `otel` feature and
  fails the launch naming the missing feature, rather than being silently
  ignored.
- Spending budgets. `Budget` sets a process-wide cap and/or per-provider caps;
  the prompt loop asks the accountant before every provider dispatch and fails
  with `ActonAIErrorKind::BudgetExceeded` once a ceiling is reached. The check
  is pre-flight, so a refused request costs nothing — but a request already in
  flight when the ceiling is crossed still completes. A cap is a circuit
  breaker, not an exact meter.
- `ActonAIBuilder::budget_usd(5.00)` for the common case,
  `ActonAIBuilder::budget(Budget::usd(5.00).provider("claude", 2.00)
  .warn_at_percent(50).allow_unpriced())` for the rest, and a `[budget]` TOML
  section with the same shape. A builder-set budget replaces the TOML section
  wholesale.
- `ActonAIBuilder::on_budget_event` runs a callback for every `BudgetEvent`;
  `ThresholdCrossed` (default 80%, configurable, 0 disables) and `Exceeded`
  are broadcast on the broker, so low-level users can subscribe directly. Each
  fires once per scope.
- `UsageSnapshot::budget` carries a `BudgetStatus`: limit, spend, remaining
  and percent used per scope, with `*_usd()` display helpers. `None` when no
  budget is configured, never a budget with nothing spent against it.
- `ActonAIBuilder::pricing(name, ModelPricing)` — the programmatic twin of
  `[providers.<name>.pricing]`, and what makes a budget possible without a
  config file.
- `accounting::dollars_to_microusd`, the shared dollar boundary;
  `dollars_per_mtok_to_microusd` now delegates to it so caps and prices round
  identically.
- `ActonAIError::budget_exceeded` and `ActonAIError::is_budget_exceeded`.

### Changed

- Budgets fail closed. A budget with `usage_tracking(false)`, a budget
  alongside a configured provider with no pricing, or a cap naming a provider
  that was never configured all fail the launch, naming the exact knob to
  change. `Budget::allow_unpriced()` accepts the pricing blind spot
  explicitly, counting that usage as $0. Unpriced usage that reaches the
  accountant at runtime refuses the next request.

## 0.30.0 - 2026-08-12

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
- Removed the `ConfigureSandbox` message and the `sandbox_factory` field on
  `ToolRegistry`. The registry-routed sandbox wiring was never reachable —
  nothing sent `ConfigureSandbox` and the factory was always `None`. The
  facade drives sandboxing directly through `PromptBuilder::use_builtins()`
  now. `ToolRegistry::ExecuteTool` runs tools inline; the `ToolConfig::sandboxed`
  flag is advisory metadata for the facade and is ignored by the registry.

### Added

- `#[tool]` attribute macro: annotate an `async fn` and get a `Tool`
  implementation derived from its signature. The tool's name is the function
  name, its description is the doc comment, and its JSON Schema has one
  property per parameter, with `Option<T>` parameters left out of `required`.
  Register with `PromptBuilder::add_tool`. The annotated function is emitted
  unchanged, so it stays independently callable and testable. Available under
  the new `derive` feature, which is on by default and independent of
  `sandbox-hardening`. A missing doc comment is a compile error.
- `Tool` trait (`acton_ai::tools::Tool`) and `PromptBuilder::add_tool`, for
  registering a tool that carries its own name, description, schema, and
  executor. `add_tool` rather than `with_tool` because the latter already
  means "a `ToolDefinition` plus a closure" and Rust cannot overload on arity.
- `acton-ai-macros` companion crate, backing the `#[tool]` macro. The
  repository is now a two-member workspace. **Release order: `acton-ai-macros`
  must be published before `acton-ai`.**
- Crate-root re-export of `serde_json`, so code generated by `#[tool]` resolves
  it without the downstream crate needing its own dependency.

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
- `[defaults]` TOML block and `ActonAIDefaults` type, currently carrying
  `max_tool_rounds`. Framework-wide default for the agentic tool-call
  loop now cascades through `DEFAULT_MAX_TOOL_ROUNDS` (10) → `[defaults]`
  → `ActonAIBuilder::max_tool_rounds(n)` → per-prompt
  `PromptBuilder::max_tool_rounds(n)`. Previously the cap was hardcoded
  at 10 with no way to raise it for chat/conversation flows.
- `ActonAI::default_max_tool_rounds()` getter exposes the resolved value
  for introspection. `PromptBuilder::current_max_tool_rounds()` returns
  the value that will actually be enforced for this request.

### Changed

- Release and CI workflows now target Linux (x86_64 + aarch64), macOS
  (Intel + Apple Silicon), and Windows x86_64. The previous `x86_64-linux`
  hard-scoping (required by Hyperlight's KVM dependency) is gone.
- Minimum `acton-reactive` is now 9.1 (was 9.0), for the scheduled-send
  facility (`send_after` / `send_at` / `send_every`).

### Internal

- The LLM provider's rate-limit drain timer now uses acton-reactive 9.1's
  `send_at` instead of a detached `tokio::spawn` + `sleep`. The schedule is
  owned by the runtime, ends with the provider actor, and the
  `drain_scheduled` guard flag became the held `ScheduledSend` itself.
- `Cargo.toml` declares an explicit single-package `[workspace]`, so a
  checkout nested under another checkout (e.g. a git worktree in
  `.claude/worktrees/`) resolves as its own workspace root.

- Deleted `guests/` workspace (hyperlight no_std guest binaries:
  `shell_guest`, `http_guest`).
- Deleted `src/tools/sandbox/hyperlight/` and `src/tools/compiler/`.
- Collapsed `build.rs` to a no-op; guest compilation is no longer part of
  the build.
- Dropped the `hyperlight-host = "0.12"` dependency.
