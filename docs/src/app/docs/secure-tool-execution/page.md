---
title: Secure Tool Execution
---

When AI agents run tools like bash commands or file writes, those operations need isolation from the host system. Acton AI provides two complementary layers: a portable process sandbox for tool execution, and path validation for filesystem tools.

---

## Why sandboxing matters

LLM-powered agents generate tool calls based on user prompts and model reasoning. Without isolation, a model could:

- Execute arbitrary shell commands on the host
- Read sensitive files outside the project directory
- Modify system configuration
- Exfiltrate data through network requests

Acton AI addresses these risks at two layers:

1. **Process sandbox** -- sandboxed tool calls run in a subprocess with rlimits, a wall-clock timeout, and (on Linux) best-effort `landlock` + `seccomp` filters applied before the tool sees the request.
2. **Path validation** -- filesystem tools are restricted to allowed directories, with blocked patterns for sensitive paths.

---

## Process sandbox integration

The [Process Sandbox](/docs/sandbox) page has the full model, hardening modes, and threat model. The short version:

### Requirements

- **Any supported target:** Linux (x86_64 + aarch64), macOS (Intel + Apple Silicon), Windows x86_64. No hypervisor required.
- **Optional Linux hardening:** the `sandbox-hardening` Cargo feature (default-enabled on Linux) pulls in `landlock` and `seccompiler`. On kernels older than 5.13, `BestEffort` mode transparently falls back to rlimits-only.

### Enabling sandboxing (high-level API)

```rust
use acton_ai::prelude::*;

let runtime = ActonAI::builder()
    .app_name("sandboxed-app")
    .from_config()?
    .with_builtin_tools(&["bash"])
    .with_process_sandbox()   // Enable ProcessSandbox with defaults
    .launch()
    .await?;

// Sandboxed tools now execute inside a subprocess with rlimits
// and (on Linux) best-effort landlock + seccomp filters.
let response = runtime
    .prompt("What is the current date and time?")
    .system("Use the bash tool to run commands.")
    .use_builtins()
    .on_token(|token| {
        print!("{token}");
        std::io::stdout().flush().ok();
    })
    .collect()
    .await?;
```

---

## Sandbox configuration

`ProcessSandboxConfig` controls timeouts, resource limits, the environment allowlist, and the hardening mode.

### Default values

| Setting | Default | Description |
|---|---|---|
| `timeout` | 30 seconds | Wall-clock deadline enforced by the parent |
| `memory_limit` | 256 MB | `RLIMIT_AS` / `RLIMIT_DATA` ceiling |
| `cpu_limit_secs` | 30 | `RLIMIT_CPU` ceiling |
| `fsize_limit` | 128 MB | `RLIMIT_FSIZE` ceiling |
| `env_allowlist` | `PATH`, `LANG`, `LC_ALL`, `HOME`, `TMPDIR` | Env vars forwarded to the child |
| `hardening` | `BestEffort` | Landlock + seccomp policy |

### Custom configuration

```rust
use acton_ai::prelude::*;
use acton_ai::tools::sandbox::{HardeningMode, ProcessSandboxConfig};
use std::time::Duration;

let config = ProcessSandboxConfig::new()
    .with_timeout(Duration::from_secs(60))
    .with_memory_limit(Some(128 * 1024 * 1024))   // 128 MB
    .with_cpu_limit_secs(Some(30))
    .with_hardening(HardeningMode::Enforce);

let runtime = ActonAI::builder()
    .app_name("custom-sandbox")
    .from_config()?
    .with_builtin_tools(&["bash"])
    .with_process_sandbox_config(config)
    .launch()
    .await?;
```

### Validation

Call `validate()` to check configuration before launching:

```rust
let config = ProcessSandboxConfig::new()
    .with_timeout(Duration::from_secs(60));

config.validate()?;  // Returns Err on invalid values (zero timeout, empty allowlist, etc.)
```

---

## Configuration via TOML

Sandbox settings can also be specified in `acton-ai.toml`:

```toml
default_provider = "ollama"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434/v1"

[sandbox]
hardening = "besteffort"    # "off" | "besteffort" | "enforce"

[sandbox.limits]
max_execution_ms = 30000
max_memory_mb = 256
```

{% callout type="note" title="Old TOMLs still parse" %}
The retired Hyperlight-era keys (`pool_warmup`, `pool_max_per_type`, `max_executions_before_recycle`) are silently ignored. You can leave them in a config file while migrating; they are no-ops.
{% /callout %}

---

## Sandbox error handling

When sandbox operations fail, errors are reported through `SandboxErrorKind` and bubble up as `ToolError::SandboxError`:

| Variant | Cause |
|---|---|
| `CreationFailed` | Unable to spawn or validate the child process |
| `ExecutionTimeout` | Wall-clock deadline exceeded; process group was killed |
| `MemoryLimitExceeded` | Child hit `RLIMIT_AS` / `RLIMIT_DATA` |
| `GuestCallFailed` | The child exited non-zero or returned a malformed response |
| `AlreadyDestroyed` | Sandbox handle used after `destroy()` |
| `InvalidConfiguration` | `ProcessSandboxConfig::validate()` rejected the settings |

```rust
use acton_ai::tools::error::ToolError;

match result {
    Err(ref e) if e.is_retriable() => {
        // Transient sandbox errors are retriable
        println!("Retrying: {}", e);
    }
    Err(e) => {
        println!("Permanent failure: {}", e);
    }
    Ok(value) => { /* success */ }
}
```

---

## Path validation and security

Beyond sandboxing, Acton AI restricts which filesystem paths tools can access through the `PathValidator`. Path validation applies to all filesystem builtins whether or not the process sandbox is enabled.

### Default behavior

By default, `PathValidator` allows access to:
- The current working directory
- The system temp directory

And blocks paths containing:
- `..` (path traversal)
- `.git` (repository internals)
- `.env` (environment/secrets files)

### Using PathValidator

```rust
use acton_ai::tools::security::PathValidator;
use std::path::{Path, PathBuf};

let validator = PathValidator::new()
    .with_allowed_root(PathBuf::from("/home/user/project"));

// Validate a file path
match validator.validate(Path::new("/home/user/project/src/main.rs")) {
    Ok(canonical) => println!("Allowed: {}", canonical.display()),
    Err(e) => eprintln!("Blocked: {}", e),
}

// Validate for file creation (parent must exist and be allowed)
match validator.validate_parent(Path::new("/home/user/project/output/result.txt")) {
    Ok(path) => println!("Can create: {}", path.display()),
    Err(e) => eprintln!("Blocked: {}", e),
}
```

### Customizing validation rules

```rust
let validator = PathValidator::new()
    .clear_allowed_roots()                          // Remove defaults
    .with_allowed_root(PathBuf::from("/data"))      // Only allow /data
    .with_allowed_root(PathBuf::from("/tmp"))       // And /tmp
    .with_denied_pattern("secrets")                 // Block "secrets" in paths
    .with_denied_pattern("credentials");            // Block "credentials" too
```

### Validation methods

| Method | Use case |
|---|---|
| `validate(path)` | General path validation |
| `validate_file(path)` | Validates path exists and is a file |
| `validate_directory(path)` | Validates path exists and is a directory |
| `validate_parent(path)` | For file creation -- validates the parent directory |

### Symlink protection

`PathValidator` resolves symlinks before checking allowed roots. A symlink inside an allowed directory that points outside will be rejected:

```rust
// Even if /home/user/project/link.txt is a symlink to /etc/passwd,
// validation will reject it because the canonical path is outside
// the allowed root.
let result = validator.validate(Path::new("/home/user/project/link.txt"));
// Returns Err(OutsideAllowedRoots { ... })
```

### Error types

Path validation returns `PathValidationError` with three variants:

```rust
use acton_ai::tools::security::PathValidationError;

match validator.validate(some_path) {
    Ok(canonical) => { /* use canonical path */ }
    Err(PathValidationError::CanonicalizeError { path, reason }) => {
        // Path doesn't exist or can't be resolved
    }
    Err(PathValidationError::OutsideAllowedRoots { path, allowed_roots }) => {
        // Path is outside permitted directories
    }
    Err(PathValidationError::DeniedPattern { path, pattern }) => {
        // Path contains a blocked pattern like ".git" or ".env"
    }
}
```

---

## Next steps

- [Process Sandbox](/docs/sandbox) -- detailed sandbox model, hardening modes, and honest threat model
- [Multi-Agent Collaboration](/docs/multi-agent-collaboration) -- configure per-agent tool access
- [Error Handling](/docs/error-handling) -- handle `ToolError` and `SandboxErrorKind`
- [Testing Your Agents](/docs/testing) -- use `StubSandbox` for deterministic tests
