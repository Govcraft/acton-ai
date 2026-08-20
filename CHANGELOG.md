# Changelog

All notable changes to this project are documented in this file. The project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.32.0 - 2026-08-19

### Added

- **Tool-approval policy gate.** One choke point between "the model asked for
  a tool" and "the tool ran", covering built-ins, `#[tool]` functions and MCP
  tools uniformly. Rules — allowlist, denylist, per-turn invocation caps —
  come from `ToolPolicy` or a `[tool_policy]` section; an async
  `.on_tool_approval(..)` hook can approve, refuse with a reason, or rewrite
  the arguments, which makes a human in the loop possible. Patterns are exact
  names or a single trailing `*`, enough to name a whole MCP server. A refusal
  is an outcome, not an error: the tool does not run, the reason goes back to
  the model as that call's tool result, and the turn continues. Nothing is
  configured by default and an unconfigured runtime behaves exactly as before.
- **Tamper-evident audit trail.** Every tool invocation — allowed, refused, or
  failed — is appended to a JSONL file as one BLAKE3 hash-chained entry
  carrying the timestamp, correlation / conversation / turn IDs, tool name,
  redacted arguments, a bounded result summary, the approval decision and who
  made it, and the duration. One actor owns the chain and is the only writer.
  Secrets are redacted by key before an entry ever leaves the prompt loop, and
  the result summary is redacted too, so a tool that echoes its input cannot
  launder a secret past the argument redaction. A restarted process resumes
  the chain it already has, and refuses to start on a file that does not
  verify. Configured with `[audit]` or `.audit_to(..)`; off by default.
- **`acton-ai audit verify`.** Walks the chain and reports the first broken
  link, with `--file` and the global `--json`. Exit code 3 when the chain does
  not hold, distinct from the generic runtime error, so a monitoring check can
  tell "the evidence has been altered" from "the check could not run".
- **`ConversationId` on the wire.** `Conversation::id()` names a conversation
  and every audit entry its turns produce carries it, so the tool calls of one
  exchange can be grouped out of a shared trail.
- **`fips` cargo feature.** Routes every TLS connection the framework opens
  through the FIPS 140-3 validated AWS-LC module instead of *ring*, installed
  as the process-wide rustls default before anything can connect. Off by
  default and not gated in CI: it builds AWS-LC from source and needs CMake
  and Go, and it must be a release build because the module's power-on
  integrity self-test does not survive a debug build's relocations.

- **An interrupted turn now finishes its lifecycle.** Dropping the
  `collect()` / `extract()` future mid-turn — a user pressing cancel, a
  `select!` taking the other arm — used to leave its `TurnStarted` broadcast
  unmatched forever, so the in-flight count never returned to zero and a
  drain waiting on it never completed. A drop guard in the prompt loop now
  publishes the balancing `TurnFinished` from cancellation, carrying the new
  `TurnOutcome::Interrupted` so observers can tell a cancelled turn from a
  completed or failed one. `TurnLifecycle::TurnFinished` now carries a
  `TurnOutcome` (`completed` / `failed` / `interrupted`).
- **`ToolPolicy::classify`.** A pure, public query answering what the gate
  would do with a tool call — `AutoAllow`, `NeedsApproval`, or `Deny` with
  the refusing rule and reason — so an embedder can render "this tool will
  ask" UI without reimplementing the allowlist/denylist/cap rules. It is not
  a parallel implementation: the gate's own `decide` is built on the same
  classification, so the two cannot drift.
- **`PromptBuilder` (and its turn future) is `Send + Sync`.** acton-reactive
  handler futures must be `Send + Sync`, so an embedder driving turns from
  inside its own actors previously had to spawn a detached task per turn.
  Callback setters are unchanged (`FnMut + Send` still suffices); the
  callbacks are stored pre-wrapped and the loop's `dyn Future + Send` awaits
  go through `sync_wrapper::SyncFuture`. `tests/prompt_builder_sync.rs`
  makes a regression a compile error.

### Changed

- TLS backend selection moved behind features. The ordinary *ring* stack is
  now the `tls-ring` feature and is part of the default set, so a default
  build is unchanged. **A `--no-default-features` build that relied on TLS
  must now name `tls-ring` (or `fips`) explicitly**, where it previously got
  the *ring* stack implicitly.
- `TurnLifecycle::TurnFinished` gained an `outcome` field. A subscriber
  matching the variant by its full field list must add `outcome` (or `..`);
  a subscriber only balancing starts against finishes can ignore it.

## 0.31.0 - 2026-08-16

### Fixed

- **A malformed model response no longer wedges a conversation.** Several
  shapes a response can produce were rejected by the provider on *every*
  later turn rather than only the one that introduced them, because the bad
  message was replayed each round: a `tool_use` with no matching
  `tool_result`, a tool result answering no call, an empty assistant turn, a
  history that no longer opened on a user turn. A new provider-agnostic
  repair pass runs before serialization and returns a well-formed history
  unchanged. Recovering previously meant discarding the whole history.
- **Streaming tool calls work on the Anthropic client.** `content_block_start`
  and `content_block_stop` were discarded and `input_json_delta` ignored, so
  the streaming path — the default — dropped every tool call the model made
  and ended the turn as though it had chosen to answer instead. Fragments are
  now reassembled and the call is emitted when its block closes.
- Parallel tool calls no longer fail against Anthropic. Each result was sent
  as its own user message, but every result answering one assistant turn must
  ride in a single message, so any multi-tool round was refused.
- The empty text block emitted beside a `tool_use` is gone. Anthropic rejects
  it, and a tool call with no preamble text is the ordinary case.
- **A broken stream is reported as a failed round.** A stream dying mid-flight
  set `StopReason::EndTurn` and reported success, presenting a partial answer
  as a finished turn and leaving the failover chain and circuit breaker with
  nothing to act on. It now sets `StopReason::Error` and fails the round, so
  failover engages.
- A tool call whose arguments will not parse is an error on the OpenAI client
  rather than being dropped or defaulted to `{}`. A stream cut mid-JSON ran
  the tool against inputs the model never sent.
- Context-window truncation keeps tool exchanges intact. It could previously
  separate an assistant turn from the results answering it, or keep results
  whose call was gone — both refused by the providers, and again for the rest
  of the conversation.

### Behavior changes

- `ContextWindow::fit_messages` returns a repaired history: consecutive
  same-role turns are coalesced (their content joined), and structurally
  unsendable messages are dropped. It can therefore return fewer messages
  than before.
- `web_fetch` returns extracted text rather than raw markup for HTML
  responses, capped at 120k characters independently of the 5 MB download
  limit, and reports `extracted_as_text`. A tool result is replayed on every
  later round of a turn, so an unbounded one grew each request until the
  provider refused it or dropped the connection — the failure that motivated
  most of this release.
- **The circuit breaker is on by default.** Five consecutive failures on a
  provider open its circuit for thirty seconds, during which requests to it
  are refused with `LLMErrorKind::CircuitOpen` instead of being sent. A
  provider failing five times in a row should fail fast rather than be
  hammered, but this is new behavior for existing configurations: opt out per
  provider with `ProviderConfig::without_circuit_breaker()` or
  `enabled = false` under `[providers.<name>.circuit_breaker]`. Failover
  chains and `fallback_model` change routing and stay opt-in.

### Breaking changes

- `LLMStreamEnd` gains a `model` field carrying the model that actually served
  the round, which is not the configured one when a rate limit degraded it.
  Struct literals of it need updating; field access is unchanged.
- `ActonAIErrorKind` gains an `AllProvidersFailed` variant, so exhaustive
  matches on it need a new arm.
- `CostAccountant::spawn` takes a third argument: the optional `BudgetConfig`
  to enforce. Facade users are unaffected.
- `UsageSnapshot` gains a `budget` field, so struct literals of it need
  updating. Field access and the accessor methods are unchanged.

### Added

- `WebFetchTool::with_max_body_chars` to tune how much extracted text reaches
  the model, separately from the download cap.
- **Live introspection** over a Unix control socket, behind the new `ipc`
  feature, which joins the default set. `acton-ai status` asks a running
  process what it is doing — uptime, PID, each provider's model, health,
  circuit and failover chain, each MCP server's incarnation and restart count,
  turns started, turns refused, turns and tool calls in flight, spend against
  budget, and the current admission state — and `acton-ai pause`, `resume`,
  and `drain`
  change whether it takes new turns. Answering never goes through the prompt
  loop, so a wedged provider or an exhausted budget does not stop you finding
  out what is wrong.
- Admission control is compiled in whether or not `ipc` is: `ActonAI::pause`,
  `resume`, `drain`, and `admission_state()` are ordinary in-process state.
  A refused turn fails with `ActonAIErrorKind::TurnsNotAdmitted`, distinct from
  a provider failure, so a caller can tell "you paused me" from "it broke".
  Closing admission never interrupts a turn already running.
- `ActonAIBuilder::introspection()` / `introspection_at(path)` and an
  `[introspection]` TOML section arm the listener. Compile-time on is not
  runtime on: a control socket is a security surface, so nothing binds until
  one of these says so. The socket is created `0o600` under a `0700`
  directory, and a `socket_mode` granting anything beyond the owner is refused
  at launch.
- `ActonAIBuilder::drain_on_sigterm()` closes admission on `SIGTERM` and lets
  the turn in flight finish, and `introspection::sd_notify::notify_ready()`
  reports readiness to systemd under `Type=notify` — a hand-rolled datagram,
  no new dependency. `acton-ai heartbeat` reports how many due entries it left
  untouched in a new `entries_deferred` field rather than failing them.
- Named **failover chains**: `ProviderConfig::with_failover(["backup", "local"])`
  and `failover = [...]` in TOML. When a provider cannot serve a round the
  prompt loop re-dispatches the same round to the next candidate — new
  correlation ID, its own budget check, its own round span and latency sample.
  Only the entry provider's chain is consulted, so there are no transitive
  chains and no cycles to reason about. When every candidate is exhausted the
  turn fails with `ActonAIError::all_providers_failed`, whose message names
  each candidate and why it was skipped or how it failed.
- A **per-provider circuit breaker**, held as pure state inside each
  `LLMProvider` actor rather than in a shared registry. Configure it with
  `ProviderConfig::with_circuit_breaker(CircuitBreakerConfig::new(5,
  Duration::from_secs(30)))` or `[providers.<name>.circuit_breaker]`. Open →
  half-open is computed lazily from the clock, never timed, and the half-open
  probe is the next real request — no synthetic traffic is ever sent.
- **Model degradation on rate limits**: `ProviderConfig::with_fallback_model`
  / `fallback_model` re-dispatches to a cheaper model on the *same* provider
  while an API-reported rate limit is in force, and returns to the primary
  model on its own once the limit clears. `UsageReport.model` and the round
  span both carry the model that actually served.
- `FailoverEvent` broadcasts — `CircuitOpened`, `CircuitClosed`, `FailedOver`,
  `ModelDegraded` — surfaced through `ActonAIBuilder::on_failover_event` and
  counted by the telemetry actor as `acton_ai.failover.events` with `kind` and
  `provider` attributes.
- `CheckHealth` asks a provider for its `ProviderHealth`. The prompt loop only
  asks when a chain is configured, so a runtime without one performs no extra
  round trips.
- Launch-time validation for chains: unknown or self-referencing members,
  duplicates, a zero failure threshold or cooldown on an enabled breaker, and
  a `fallback_model` identical to the provider's own model are all refused
  before anything is spawned.
- `ActonAI::provider_failover(name)` exposes the chain configured for a
  provider, and `acton-ai config` renders the resolved failover, breaker, and
  fallback-model settings.
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
