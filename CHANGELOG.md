# Changelog

All notable changes to this project are documented in this file. The project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.37.0 - 2026-08-30

### Added

- **Host-owned turn refusals reach the audit chain.**
  `ActonAI::record_refused_turn` gives embedders a public path through the
  runtime's single audit writer for turns rejected before `run_prompt_loop`,
  including entitlement, policy, control-plane, and audit-backlog decisions.
  It returns an `AppendReceipt` after the configured durability guarantee is
  met and never calls a model provider.

- **Instruction provenance is sealed with each turn.** `ContextSource`
  records an instruction layer's scope, resolved path, and BLAKE3 content
  hash without recording its content. `AgentInstructions::context_sources`
  derives those fingerprints and `PromptBuilder::context_sources` attaches
  them to the terminal turn entry. Empty source lists remain absent from both
  serialized entries and hash preimages, so existing trails still verify.

### Fixed

- **The single-writer audit claim works on Windows.** Windows now locks a
  sibling `<trail>.lock` file because locking the trail itself prevents the
  writer from reopening that file for appends on that platform. The lock is
  still kernel-released on shutdown or process death, and Unix continues to
  lock the trail directly.

## 0.36.0 - 2026-08-30

### Added

- **Every attempted turn reaches the audit chain.** An armed audit trail now
  appends one metadata-only terminal entry for completed, failed, interrupted,
  and admission-refused turns, including turns where the model answered with
  text and called no tool. Turn entries carry the existing user/timestamp and
  correlation/conversation/turn identities plus prompt and response byte
  counts, configured provider/model, and aggregate input/output tokens. They
  deliberately contain no prompt or response content. `TurnRefused` now
  carries the attempted `TurnId` and rendered reason. Strict durability waits
  for the terminal entry before returning; the existing finish guard records
  cancellation from its `Drop` path. `AuditEntryKind`, `TurnRecord`, and
  `TurnAuditOutcome` expose the structured form. Existing invocation JSON and
  hash preimages remain byte-identical: their discriminator and turn-only
  fields are absent, and legacy v0.33 fixtures still verify.

- **Hierarchical `AGENTS.md` discovery.** `AgentInstructions::discover`
  finds the nearest checkout boundary, walks from that workspace root to the
  session working directory, and loads each exact-name `AGENTS.md` in
  increasing precedence order. A nested checkout is its own boundary and
  never inherits instructions from an outer checkout. The canonical
  user-level `~/.agents/AGENTS.md` is layered last. Hosts that already own
  workspace and trust boundaries can use `discover_with_root`, including an
  explicit user file or no user layer at all. The structured result exposes
  every `InstructionLayer` with its scope, source path, content, and
  precedence for auditing or filtering; `context_fragment()` renders the
  approved stack for injection at turn start. Vendor-specific and similarly
  named files are intentionally ignored.

## 0.35.0 - 2026-08-29

The embedder release: everything an agent host needs to run one governed
daemon over one audit trail and to own its sessions. One trail has one
writer, one identity, and a health it can report; a session survives the
process; a compaction can be replayed by whoever keeps the history.

### Added

- **Audit durability and writer health.** `[audit] durability = "strict"`
  (or `AuditConfig::with_durability(AuditDurability::Strict)`) makes every
  entry fsync and be acknowledged before the prompt loop considers the next
  tool call, and once an append has failed the loop refuses every tool not
  declared idempotent — through the ordinary refusal path, so the attempt is
  recorded as denied by the new `Decider::AuditGuard` with
  `DenialReason::AuditDegraded`, and the model is told mutating tools stay
  refused for the session. An audit actor that does not answer the guard is
  refused the same way (`DenialReason::AuditUnreachable`): the guard fails
  closed. The default stays `best_effort`, so nothing changes for an
  existing embedder. The writer's state is readable as `AuditHealth`
  (`healthy`, `degraded`, or `disabled`; appended and failed counts; the
  first failed sequence; the last error; when it degraded) through
  `ActonAI::audit_health()` and the `GetAuditHealth` request, and the
  healthy-to-degraded transition is broadcast once as `AuditHealthChanged`.
  `RecordInvocationDurably` is the acknowledged form of `RecordInvocation`,
  answering with an `AppendReceipt`; `ActonAI::audit_durability()` reports
  what the trail promises.

- **Compaction as a strict prefix elision.** The prompt loop has always
  compacted from the front — a leading system message held aside, the
  oldest exchanges replaced by the summary — but an embedder that keeps its
  own copy of the history had to infer how much of it the summary stood for.
  `CompactionRecord.elided_prefix_len` now states it: the number of leading
  non-system messages the summary replaced, and `CompactionRecord::adopt`
  replays that onto the embedder's copy so the next turn sends the summary
  instead of the elided span and does not pay for the same summary twice.
  `CompactionPlan::elided_prefix_len()` is where the number comes from.

- **Session persistence for embedders.** The `MemoryStore` actor now answers
  the named-session messages the CLI used to reach only through free
  functions: `CreateSession` (-> `SessionCreated`), `ResolveSession`
  (-> `SessionResolved`), `ListSessions` (-> `SessionList`), `TouchSession`,
  `UpdateSessionMetadata` and `DeleteSession` (-> `OperationCompleted`).
  Sessions carry an opaque `metadata` column (`SessionInfo::metadata`,
  `update_session_metadata`) for the embedder's own per-session state;
  the database schema is version 3 and a version-2 file gains the column in
  place on its next open. `ActonAI::checkpoint_policy()` reports the
  `ResumePolicy` a runtime launched under, so an embedder can refuse
  `resume_auto`. A turn whose future is dropped mid-flight — an embedder's
  cancel — now releases its checkpoint claim from `Drop`, so the record it
  left is listed by `interrupted_turns()` and can be resumed or abandoned
  by id instead of staying claimed until the process dies. A turn driven by
  `continue_with` fingerprints its checkpoint over the history's last user
  message rather than the empty placeholder, so the turns of one session no
  longer all look like the same turn; `continue_with` still attaches no
  sink of its own, which is now documented on both it and
  `PromptBuilder::checkpoint`.

- **Trail identity bound into the chain.** Every audit trail now has a
  `TrailId` (TypeID, prefix `trail`), minted the first time an audit log
  opens it, kept in a sidecar beside the file (`audit.jsonl.trail`,
  `AuditConfig::trail_id_path()`), and sealed into every entry's hash as
  `AuditEntry.trail_id`. An entry can no longer be relabelled as another
  trail's, and a chain cannot change identity part-way: `verify_chain`
  reports the new `ChainBreakKind::TrailMismatch`. The per-link rule is
  exposed as `verify_next(head, entry, line)` so a verifier holding the head
  elsewhere applies exactly the file walk's rule. `ChainHead.trail_id`
  carries the identity through `ActonAI::audit_head()` and `audit verify`
  (`trail:` line, `trail_id` in the JSON report). Trails written before
  identities keep verifying — the unidentified prefix is allowed — and gain
  an identity on their next spawn; a sidecar that disagrees with the chain
  refuses the spawn. `ActonAI::audit_config()` exposes the resolved audit
  settings so an embedder can find the trail and its sidecar. The sidecar
  helpers `read_trail_id`, `write_trail_id` and the pure `resolve_trail_id`
  (with `TrailIdConflict`) are public under `audit`.

### Changed

- **`create_session` takes the session metadata.** The persistence function
  is now `create_session(conn, name, agent_id, system_prompt, metadata:
  Option<&str>)`; pass `None` to keep the old behaviour.

- **`AuditEntry::seal` takes the trail identity.** The signature is now
  `seal(record, sequence, prev_hash, trail_id: Option<&TrailId>)`; `None`
  reproduces the legacy pre-identity form. Any embedder sealing entries
  itself must adapt.

- **One writer per audit trail, enforced.** `AuditLog::spawn` now claims the
  trail with an exclusive advisory lock (`std::fs::File::try_lock`) before it
  reads the chain head, and holds it for the actor's lifetime; the kernel
  releases it on shutdown or on `SIGKILL`, so there is no pid file to go
  stale. A second process opening the same trail fails to launch with a
  configuration error instead of forking the chain. The claim is exposed as
  `audit::claim_trail` and its typed refusal as `audit::TrailClaimError`,
  whose `Busy` variant names the holder's pid where the platform can tell
  (`/proc/locks` on Linux). Read-only verification of a live trail is
  unaffected (Govcraft/acton-ai#14).

- **`CompactionRecord` carries `elided_prefix_len`.** The struct gained a
  public field, so an embedder that builds records itself must set it;
  the prompt loop is the only in-tree constructor.

- **Minimum supported Rust is 1.89.** The trail lock uses
  `std::fs::File::try_lock`, stabilized in 1.89; `rust-version` now says
  so, so an older toolchain fails at resolution rather than mid-build.

## 0.34.0 - 2026-08-28

### Added

- **Audit attribution and response sizing.** `[audit] user` (or
  `AuditConfig::with_user`) stamps the acting principal onto every tool entry,
  and successful calls record the byte size of the complete serialized result
  before its audit summary is bounded and redacted.
- **Declared paths for the hardened sandbox.** The landlock ruleset granted
  the system directories, `$TMPDIR` and the session root, and nothing else,
  which left out every toolchain installed under a home directory. A `uv` at
  `~/.local/bin` was found by the shell on `PATH` and then refused by the
  kernel — reported as a bare `Permission denied`, exit code 126, with no
  mention of landlock in it, and no configuration short of turning hardening
  off entirely to resolve it. `[sandbox.paths]` now declares what the child
  may additionally reach: `read_exec` for directories it may read and run
  binaries from, `read_write` for the caches those binaries must write to. A
  leading `~/` expands against `HOME`; the same lists exist on
  `ProcessSandboxConfig` as `with_read_exec_paths` / `with_read_write_paths`.
  Both are empty by default, so nothing widens unless a deployment says so.
- **`env_allowlist` in `[sandbox]`.** The set of variables forwarded into the
  sandbox child was reachable from Rust and not from a config file, so a tool
  needing `UV_CACHE_DIR` or `CARGO_HOME` had no way to be given it. The key
  replaces the default list rather than extending it — name every variable the
  child still needs, `PATH` and `HOME` included.

### Changed

- **Declared sandbox paths are validated up front.** `validate()` refuses a
  relative entry, which would otherwise resolve against whichever directory
  the child happened to start in and fail later as an unexplained denial. An
  entry that simply does not exist is warned about and skipped in every
  hardening mode, `enforce` included: it narrows the ruleset rather than
  widening it, and a cache directory a tool has not created yet must not abort
  a deployment.

## 0.33.0 - 2026-08-20

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
  verify. An armed trail exists on disk from launch — empty, verifying as
  the genesis head — so a missing file always means somebody removed it.
  Configured with `[audit]` or `.audit_to(..)`; off by default.
- **`acton-ai audit verify`.** Walks the chain and reports the first broken
  link, with `--file` and the global `--json`. Exit code 3 when the chain does
  not hold, distinct from the generic runtime error, so a monitoring check can
  tell "the evidence has been altered" from "the check could not run".
- **`ConversationId` on the wire.** `Conversation::id()` names a conversation
  and every audit entry its turns produce carries it, so the tool calls of one
  exchange can be grouped out of a shared trail.
- **Embedder access to the sandbox execution path.** A downstream crate
  wrapping builtin execution — an approval flow, a protocol adapter — could
  not keep the sandbox while doing it: the factory getter and the executor
  adapter were crate-private. `SandboxedExecution` is now the public handle
  over the configured factory, returned by `ActonAI::sandboxed_execution()`,
  and it owns the one-shot create → execute → destroy lifecycle so nobody
  re-derives it. `ActonAI::builtin_executor(name)` returns the named
  builtin's `BuiltinExecutor` with the sandbox-or-in-process decision
  already made; it is the very value `.use_builtins()` registers, so what an
  embedder wraps is literally what the prompt loop runs. Because the process
  sandbox re-execs the current binary, an embedder's `main` installs
  `runner::run_if_sandbox_child()` first thing — the same guard `acton-ai`'s
  own entry point now uses — and `runner::supports(name)` says which tools
  can cross the process boundary at all.
- **Caller-supplied turn identity.** `PromptBuilder::turn_id(..)` lets an
  embedder that already answered a session (an ACP daemon, say) name the turn
  before the loop starts, and `CollectedResponse::turn_id` reports the turn's
  ID back either way — supplied or minted — so no claim/bind side table is
  needed to attribute a response.
- **`StreamContext` on the stream callbacks.** `.on_start(..)` and
  `.on_end(..)` now receive a `&StreamContext` carrying the turn ID and the
  round's correlation ID, so a streaming consumer can attribute every round
  to its turn without out-of-band bookkeeping. Exported from the prelude.
- **Enriched tool lifecycle events.** `TurnLifecycle::ToolStarted` carries the
  arguments the model proposed, verbatim; `ToolFinished` carries `success` and
  a bounded `summary`; `LLMStreamToolResult` carries the `turn_id`. The enum
  and its variants are now `#[non_exhaustive]`.
- **The tool bracket is total.** `ToolStarted` is broadcast *before* the
  policy gate deliberates, so every `ToolFinished` and `LLMStreamToolResult`
  — including a call the policy refused — is preceded by exactly one
  `ToolStarted` with the same `tool_call_id`, and a human approval hook
  deliberates on a call the client has already been shown.
- **`tool_call_id` end to end.** `policy::ToolInvocation` and every audit
  entry now carry the provider's own ID for the call — the same ID the
  lifecycle events broadcast — so an approval prompt and a trail read after
  the fact both join cleanly against a session watched live.
- **`fips` cargo feature.** Routes every TLS connection the framework opens
  through the FIPS 140-3 validated AWS-LC module instead of *ring*, installed
  as the process-wide rustls default before anything can connect. Off by
  default and not gated in CI: it builds AWS-LC from source and needs CMake
  and Go, and it must be a release build because the module's power-on
  integrity self-test does not survive a debug build's relocations.
- **Runtime-wide custom tool registration.** `ActonAIBuilder::with_tool`
  (definition + async closure), `with_tool_executor` (a `ToolExecutorTrait`
  object; its `validate_args` runs before every execution), and `add_tool`
  (a `Tool` value, the shape `#[tool]` generates) register a tool once and
  inject it into every prompt and every conversation turn, alongside the
  built-ins, skill tools, and MCP tools. Custom tools execute on the same
  path as the built-ins — behind the tool-approval policy gate and onto the
  audit trail when those are configured. Names are validated at `launch()`:
  a custom tool that collides with an enabled built-in, a skill tool, an MCP
  tool, or another custom tool fails the launch instead of silently
  shadowing anything. Downstream embedders (an agent daemon installing an
  `apply_patch` tool, say) previously had to re-register such tools on every
  `PromptBuilder` — and could not reach `Conversation::send` at all.
- **Per-conversation tools.** `ConversationBuilder::with_tool` /
  `with_tool_executor` / `add_tool` register a tool for every turn one
  conversation runs, closing the gap where `Conversation` rebuilt its prompt
  internally and per-prompt registration could never reach it. Collisions —
  with an injected runtime tool, another conversation tool, or the reserved
  `exit_conversation` name — are refused at registration time with a
  configuration error, never discovered mid-conversation.

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
- **`api_key_file` as a provider key source.** A named provider can point at a
  file holding its key: `api_key_file = "~/.config/app/anthropic-key"`.
  Resolution slots between the environment variables and the inline
  `api_key` — env stays authoritative, a file beats a value baked into the
  config. Contents are trimmed, `~/` expands, and an unreadable or blank file
  warns and falls through. This suits keys provisioned as secret files
  (systemd credentials, a login command writing an 0600 file), where an
  environment variable would leak into every child process.
- **One filesystem boundary per caller.** Every filesystem-capable builtin can
  now be built for exactly one directory: `ActonAI::builtin_executor_in(name,
  root)` and `PromptBuilder::use_builtins_in(root)` hand back tools that reach
  `root` and nothing else — not the process working directory, not the system
  temp directory, both of which the unconfined defaults allow. This is what a
  host serving several workspaces from one runtime needs, and it could not be
  expressed before: `ToolExecutorTrait::execute` takes no per-call context, so
  confinement is a property of construction. `glob` and `grep` default their
  base path to the root instead of the process cwd, and `bash` starts there
  and validates any `cwd` it is handed. The boundary crosses the process edge
  too — `Sandbox::execute_in` carries the root, the process sandbox makes it
  the child's working directory and passes it as `ACTON_AI_SANDBOX_ROOT`, and
  the child rebuilds its tools around it — so the in-process and sandboxed
  paths cannot enforce different rules. The default `execute_in` fails closed:
  a sandbox that cannot confine says so rather than silently running
  unconfined.
- **The sandbox says what it is enforcing.** `ActonAI::sandbox_config()`
  returns the `ProcessSandboxConfig` in force — limits and
  [`HardeningMode`] — `Some` exactly when `sandboxed_execution()` is. A
  governed deployment is usually required to report whether isolation and OS
  hardening are active, and an operator who has to infer that from a config
  file is reading an intention rather than a fact.

### Fixed

- Launch-time custom tool validation now reserves the prompt loop's own tool
  names. A runtime-wide custom tool named `structured_output` or
  `exit_conversation` used to launch fine and misbehave later — every
  `.extract::<T>()` failed at call time, and a conversation's exit tool was
  silently shadowed so a chat could never end through it. Both names now fail
  `launch()` like any other collision, and they are published as
  `extract::STRUCTURED_OUTPUT_TOOL` and `conversation::EXIT_CONVERSATION_TOOL`
  so an embedder can avoid them without hardcoding strings.
- `TurnLifecycle::ToolStarted` no longer broadcasts secret-bearing arguments
  past the audit redactor. Trail entries were redacted at the boundary, but
  the lifecycle broadcast carried the model's arguments raw into every
  subscriber's mailbox — an embedder forwarding lifecycle events to a UI or a
  log received exactly what the redaction config promised to withhold. With a
  trail configured, `ToolStarted` arguments now pass through the same
  redactor; without one they are unchanged.
- A turn future dropped from a thread outside any Tokio context now still
  publishes its balancing `TurnFinished`. The drop guard used to look up the
  ambient runtime at drop time and silently give up without one, so an
  embedder storing the `Send + Sync` `collect()` future in a session table
  and dropping it from a UI thread, C-FFI callback, or watchdog thread left
  the turn counted in-flight forever and wedged `acton-ai drain --wait`. The
  guard now captures the runtime handle at construction and spawns the
  broadcast on it, falling back to the ambient handle as before.
- Reused caller-supplied turn ids no longer corrupt the in-flight accounting.
  The introspection actor kept live turns in a set, so two concurrent turns
  sharing a `PromptBuilder::turn_id` — or a cancel-and-retry whose
  interrupted finish landed after the retry started — let one finish erase
  the other's live turn and sweep its in-flight tools, and a drain could
  report drained mid-turn. Starts and finishes are now counted per id, an
  unmatched finish is a no-op, and the tool sweep runs only when an id goes
  fully dead.
- A dropped sandbox execution no longer leaks the child process. Cancelling
  the execute future mid-flight — a caller-side timeout, an aborted turn —
  left the re-exec'd child running detached until its own resource limits
  bit: the destroy step lives after the await, and a dropped future never
  reaches it. The child is now killed when the future is dropped.
  Grandchildren in the child's process group can still outlive it on this
  path, exactly as on the non-unix timeout path.
- A hardened sandbox child could not spawn a process at all. Every `bash` call
  under landlock failed with "Permission denied" before the command was
  parsed: `Stdio::null()` opens `/dev/null`, and `/dev` was in no rule. The
  ruleset now grants file-level read+write on `/dev/null`, `/dev/zero`,
  `/dev/full`, `/dev/random` and `/dev/urandom`, plus read-only `/proc`, which
  shells and the tools they run read routinely. Confinement is unchanged, and
  a test now pins it: a hardened child runs a command inside its root and is
  refused the sibling directory by the kernel, with no tool argument naming
  it.
- `hardening = "best-effort"` (and `best_effort`) are accepted in TOML. The
  serde rename made `besteffort` the only spelling, which is the one nobody
  writes by hand.
- SSE lines split across network chunks are reassembled. Both streaming
  clients split each chunk into lines in isolation, so a `data:` line
  straddling a boundary parsed as truncated JSON in the OpenAI client
  (failing the turn) and was silently dropped by the Anthropic client (losing
  tokens); a UTF-8 character split across chunks was corrupted by per-chunk
  lossy decoding. Localhost streams deliver one event per chunk and never
  exposed this; fast TLS streams split lines constantly. A new
  `llm::sse::LineAssembler` carries the unterminated tail between chunks and
  reassembles at the byte level.
- An in-band stream error from an OpenAI-compatible provider now surfaces as
  the provider's message. Providers that hit trouble after the 200 header
  report it as a `data: {"error": ...}` event, which failed to deserialize as
  a chunk, so the turn died on our parse error instead. The error shape is
  detected before the chunk parse, whose defaulted fields would otherwise
  absorb it as an empty chunk. Chunks without an `id` now parse instead of
  failing.
- A round whose stream died after it started is retried instead of killing the
  turn. The provider actor classifies the failure where the error kind is
  still known (`LLMError::is_transient`), `LLMStreamEnd` carries a transient
  note when re-dispatch may succeed, and the prompt loop re-dispatches with a
  fresh correlation ID, linear backoff, and at most two retries per candidate.
  Failures *before* any stream started are deliberately untouched: those
  already belong to retry-after, the circuit breaker, and the failover chain.
- A hallucinated tool call is now corrected on the retry. Hosts that validate
  tool calls server-side kill the stream with an in-band error when the model
  calls a tool that is not in `request.tools`, so the call never reaches the
  tool loop where an unknown tool would be answered with an error result the
  model can read. The blind retry re-sent identical context and the model
  hallucinated identically until the budget ran out. The retry now carries the
  feedback the tool result would have carried — a user message naming the
  invented tool and pointing at the real list, appended once.
- A provider-rejected tool call no longer counts against the host. Every
  server-side rejection was recorded as a provider failure, walking the
  circuit breaker open after five of the model's own mistakes on a perfectly
  healthy host. A shared classifier now gates both the correction and the
  breaker outcome, so a round the provider killed over what the model wrote
  reports as succeeded. Corrected retries also draw from their own budget
  rather than the transient-retry budget a model needing two nudges exhausts,
  and the `bash` schema no longer declares a `maximum` for `timeout` that a
  validating host rejects while the executor clamps it anyway.
- An oversized tool result no longer ends the turn. The fit budget reserved a
  fixed 1024 tokens for the response while the request asked for the
  provider's full `max_tokens`, so any prompt in the gap was rejected as too
  long; `reserved_for_response` now defaults to the default provider's
  `max_tokens`, clamped to half the window. Truncation also dropped whole
  exchanges, so a newest exchange that alone exceeded the window emptied the
  history: an exchange too large to fit now sheds its tool results
  largest-first, replacing each with a placeholder telling the model to re-run
  the tool with a narrower query, and call/result pairing stays intact.
- Every request is now bounded by the context window. The prompt loop resolved
  a window and reported it to tools through `get_context_remaining` but never
  enforced it — only the `Conversation` API called `fit_messages` — so one
  oversized round left the loop sending a raw over-window request the provider
  rejected. The loop now fits the history right before each request is built,
  after the compaction gate so a configured summarizer still runs first, and
  only on estimated overflow.

### Changed

- **The stream callbacks take an identity.** `.on_start(..)` now receives
  `&StreamContext` where it previously took no arguments, and `.on_end(..)`
  receives `(&StreamContext, StopReason)` where it previously took only the
  stop reason. **Existing callers must add the parameter**, typically
  `.on_start(|_ctx| ..)` and `.on_end(|_ctx, reason| ..)`. `.on_token(..)` is
  deliberately unchanged and still takes `&str`: it fires per token, where an
  identity that is constant for the whole round would be repeated noise.
- `TurnLifecycle` and its struct variants are now `#[non_exhaustive]`, so a
  downstream `match` needs a wildcard arm. These are observation events and
  later additions should not be breaking changes.
- TLS backend selection moved behind features. The ordinary *ring* stack is
  now the `tls-ring` feature and is part of the default set, so a default
  build is unchanged. **A `--no-default-features` build that relied on TLS
  must now name `tls-ring` (or `fips`) explicitly**, where it previously got
  the *ring* stack implicitly.
- `TurnLifecycle::TurnFinished` gained an `outcome` field. A subscriber
  matching the variant by its full field list must add `outcome` (or `..`);
  a subscriber only balancing starts against finishes can ignore it.

### Removed

- **The orphaned `ToolRegistry` path.** `ToolRegistry`, `RegisterTool`,
  `UnregisterTool`, `ListTools`, `ToolListResponse`, `RegisteredTool`,
  `InitToolRegistry`, `RegistryMetrics`, `RegistryMetricsSnapshot`, the
  one-shot `ToolExecutor` actor (`Execute`, `InitExecutor`), and the
  `ExecuteTool` / `ToolResponse` messages are gone. Nothing in the facade,
  the prompt loop, or the agents ever routed through them — the registry was
  a dead end that *looked* like the way to register a global tool while the
  real path did not exist. It is replaced, not merely deleted:
  `ActonAIBuilder::with_tool` and `ConversationBuilder::with_tool` (above)
  are the supported global and per-conversation registration paths, and both
  actually reach every prompt. Per-agent tool actors (`ToolActor`,
  `ExecuteToolDirect`, `ToolActorResponse`) are unchanged.

## 0.32.0 - 2026-08-19

### Added

- **Auto-compaction.** A turn that works through tools appends an assistant
  turn and every tool result on each round, and nothing bounded that growth:
  the per-turn truncation a `Conversation` applies runs *between* turns, not
  inside one, so a long tool loop eventually exceeded the provider's context
  window and failed mid-turn, after the earlier rounds were already paid for.
  When the history reaches a configurable fraction of the available budget the
  prompt loop now — between rounds, never mid-exchange — sends the older
  history back to the **same provider** with a fixed summarization prompt and
  splices the model's own summary in where the elided messages were, keeping
  the last few exchanges verbatim. This is deliberately not truncation:
  dropping the oldest exchanges silently erases the user's original request
  and the model has no way to know, whereas the summary tells it what it
  forgot. Configured with `auto_compact = true` (plus optional
  `compact_threshold` and `keep_recent_turns`) under `[context]`, or
  `.compaction(..)` on the builder; **off by default**, so an unconfigured
  runtime behaves exactly as before. A summarization that fails or is refused
  by the budget stalls compaction for the turn, which proceeds with its full
  history rather than a hole where its history used to be.
- **`acton_ai::memory::compaction`.** The policy as pure functions —
  `plan_compaction`, `summarization_messages`, `finish_compaction` — over
  validated newtypes (`CompactionThreshold`, `KeepRecentTurns`), so every
  decision is testable without a provider. Compaction never splits an
  exchange: an assistant turn carrying tool calls travels with the tool
  results answering it, because a `tool_use` with no matching `tool_result` is
  rejected by every provider for the rest of the conversation. It also
  declines whenever the summary would be no smaller than the text it replaces,
  which is what stops an already-compacted history from being rewritten — and
  paid for — on every round.
- **Compaction is transparent, everywhere it can be seen.**
  `TurnLifecycle::ContextCompacted` is broadcast on every pass with the token
  counts and the number of messages elided, alongside an `info` log;
  `CollectedResponse` gained `compactions` carrying a `CompactionRecord` per
  pass; the CLI's session store persists each record as a clearly marked
  summary message, so a stored session records that — and what — the model was
  told it forgot; and `acton-ai status` reports the running total once it is
  nonzero.

- **Checkpoint and resume.** A turn that dies with the process no longer
  loses its work. With a `[checkpoint]` section (or `.checkpoint(..)` on the
  builder) every prompt records its progress into a Turso/libSQL database:
  the conversation as the next round would send it, the rounds spent, the
  tools already executed — and, mid-round, a per-call ledger written before
  each tool starts and rewritten when it completes, one single-row upsert per
  change so the state and the record of it can never disagree. On restart the
  configured `ResumePolicy` decides what happens to interrupted turns:
  `abandon` (the default) marks them as a recorded outcome and runs nothing,
  `resume_on_request` leaves them for `interrupted_turns()` /
  `resume_turn(..)`, and `resume_auto` picks them back up in the background.
  A resume settles the interrupted round first — completed calls keep their
  stored results, unstarted calls execute — and then re-prompts with the
  accumulated results. Without the section nothing is recorded, no store is
  opened, and behavior is exactly what it was.
- **Per-tool idempotency declarations.** `ToolDefinition` carries an
  `idempotent` flag (default `false`; the pure and read-only built-ins —
  `read_file`, `glob`, `grep`, `list_directory`, `calculate`,
  `get_context_remaining`, `update_plan` — declare `true`). Crash
  recovery turns on it: a call that was mid-execution when the process died
  re-runs only if its tool is idempotent; otherwise it is NOT re-run, and the
  model is told its outcome is uncertain as that call's tool result.
- **`resumed` marker in the audit trail.** Every tool call executed by a
  resumed turn is stamped `resumed: true` in its audit entry, covered by the
  entry's hash. First-run entries keep their exact pre-existing byte shape,
  so trails written before the field existed still verify.
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
- **`get_context_remaining` built-in tool.** Lets the model ask how much of
  the context window is spent before it commits to something large.
  Registered with the other builtins, so `with_builtins()` offers it, and it
  answers `{total_tokens, used_tokens, remaining_tokens, percent_used}` —
  measured by the same estimator that decides truncation, against the window
  resolved at launch from per-provider `context_window_tokens`, then
  `[context] max_tokens`, then the built-in default. The live message state
  has exactly one owner, the prompt loop, so the loop measures at call time
  and injects the two numbers into the call's arguments; the tool itself is
  pure arithmetic, and the injected state never enters the conversation
  history.
- **`update_plan` built-in tool.** The model maintains a structured plan for
  the current task — `{steps: [{title, status}], note?}` with statuses
  `pending`/`in_progress`/`completed` — and every accepted update is broadcast
  on the broker as `PlanUpdated` (in `acton_ai::messages`, alongside the other
  stream and lifecycle events), carrying the turn ID, the round's correlation
  ID, the tool call ID, and the validated `Plan`. Subscribe an actor to it (on
  the builder, before `start()`) to render the model's progress; the chat REPL
  already does, drawing each update as an inline checklist. The prompt round
  loop owns the turn's current plan: it carries across every round of the tool
  loop and the turn's final plan is returned on `CollectedResponse::plan`. The
  tool runs no code and touches nothing: 2–32 steps, each title trimmed,
  non-empty, at most 200 characters, and distinct; at most one step
  `in_progress`. A single-step plan is refused outright — the tool's
  description tells the model to skip planning for trivial tasks, and the
  refusal tells it to split the work or not plan at all. Any refused plan goes
  back to the model as that call's tool result so it can correct itself on the
  next round; nothing is broadcast and the recorded plan stands. Included in
  `with_builtins()`, or named on its own with
  `.with_builtin_tools(&["update_plan"])`. The plan types and validation live
  in the new public `acton_ai::tools::plan` module.

### Changed

- `StatusReport` gained `turns_compacted`. Additive and `#[serde(default)]`,
  so an older client still deserializes a newer server's reply and
  `SCHEMA_VERSION` is unchanged.
- TLS backend selection moved behind features. The ordinary *ring* stack is
  now the `tls-ring` feature and is part of the default set, so a default
  build is unchanged. **A `--no-default-features` build that relied on TLS
  must now name `tls-ring` (or `fips`) explicitly**, where it previously got
  the *ring* stack implicitly.

### Fixed

- **Windows builds no longer break on the default feature set.** The `ipc`
  feature (the Unix-socket introspection transport) is now a no-op on
  non-Unix targets instead of a hard compile error from acton-reactive:
  the upstream `ipc` feature is enabled through a `cfg(unix)` dependency
  entry, and every `ipc`-gated code site now gates on
  `all(feature = "ipc", unix)`. Same philosophy as `sandbox-hardening`,
  which has always been a no-op off Linux. Relatedly, socket-path validation
  now accepts Unix-style absolute paths on Windows (`Path::is_absolute`
  demands a drive prefix there), so a config file shared across platforms
  still parses on a Windows build.
- **A refused resume can no longer reopen a finished turn.** Marking a failed
  turn's checkpoint now leaves terminal records — `Completed`, `Abandoned` —
  exactly as they are, so a pre-flight refusal (changed inputs, an abandoned
  record) no longer downgrades them to `Failed`, which would have made a
  finished answer re-executable and an operator's abandonment resumable.
- **A seeded resume keeps the record's original fingerprint.** Progress
  written by `resume_turn(..)` / `resume_auto` no longer stamps a fingerprint
  computed from the resumed builder's synthetic inputs, so the documented
  retry — re-issuing `.prompt(P).checkpoint(store, id)` after a crash-resume —
  still matches and replays instead of being refused as changed inputs.
- **One live owner per checkpoint.** The prompt loop claims its checkpoint ID
  (an in-process registry owned by the `MemoryStore` actor) before reading
  the record and releases it when the turn ends; a concurrent resume of the
  same ID — an operator's `resume_turn` against a live turn, a retry racing
  the `resume_auto` background task — is refused with `AlreadyRunning`
  instead of double-executing pending tool calls, and `interrupted_turns()`
  filters out records claimed by turns still running in this process.
- **The unattended resume sweep is bounded.** Each failed attempt increments
  the record's `resume_attempts`; once a `Failed` record reaches
  `max_resume_attempts` (default 3, settable on `CheckpointConfig` or as
  `max_resume_attempts` under `[checkpoint]`), `resume_interrupted()` — and
  therefore `resume_auto` — marks it `Abandoned` instead of re-dispatching,
  and re-paying for, the same failure on every process start. A deliberate
  per-turn `resume_turn(..)` is never subject to the ceiling.
- **Uncertain calls reach the audit trail.** A started, non-idempotent call
  the resume settlement declines to re-run is now recorded as an
  `AuditOutcome::Uncertain` entry (decider `settlement`), so the trail
  accounts for the one call whose first attempt died before it could write —
  previously the response's `tool_calls` showed a call the chain had no
  entry for.
- **Settlement re-execution matches the main loop.** A `Pending`
  `get_context_remaining` settled on resume now gets the loop-injected
  context state instead of erroring with "live conversation state is
  unavailable", and a settled `update_plan` updates the turn's plan and
  broadcasts `PlanUpdated`, exactly as the main round loop does.
- Removed the two `#[allow(clippy::cast_possible_truncation)]` suppressions
  in `memory::compaction`: `CompactionThreshold` now stores `f64` (so the
  TOML fraction is never narrowed at all — `get()` returns `f64`), and
  `trigger_tokens` clamps to the budget before a cast that is exact by
  construction.
- `PlanError` and `PlanTextError` are now `#[non_exhaustive]`, like every
  other error enum in the crate, so adding a validation rule stays a
  semver-compatible change.

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
