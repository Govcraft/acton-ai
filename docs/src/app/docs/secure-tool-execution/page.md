---
title: Secure Tool Execution
---

When AI agents run tools like bash commands or code snippets, those operations need isolation from the host system. Acton AI provides hardware-level sandboxing through Hyperlight micro-VMs and path validation to restrict filesystem access.

---

## Why sandboxing matters

LLM-powered agents generate tool calls based on user prompts and model reasoning. Without isolation, a model could:

- Execute arbitrary shell commands on the host
- Read sensitive files outside the project directory
- Modify system configuration
- Exfiltrate data through network requests

Acton AI addresses these risks at two layers:

1. **Hyperlight micro-VM sandboxing** -- tool execution runs inside hardware-isolated virtual machines with restricted memory and time limits
2. **Path validation** -- filesystem tools are restricted to allowed directories, with blocked patterns for sensitive paths

---

## Hyperlight sandbox integration

[Hyperlight](https://github.com/hyperlight-dev/hyperlight) is a lightweight hypervisor library that creates micro-VMs with 1-2ms cold start times. Acton AI integrates Hyperlight to sandbox `bash` and `rust_code` tool execution.

### Requirements

- **Linux**: KVM support (check with `ls /dev/kvm`)
- **Windows**: Hyper-V enabled
- **Architecture**: x86_64 with hardware virtualization

### Enabling sandboxing (high-level API)

The simplest way to enable sandboxing is through the `ActonAI` builder:

```rust
use acton_ai::prelude::*;

let runtime = ActonAI::builder()
    .app_name("sandboxed-app")
    .from_config()?
    .with_builtin_tools(&["bash"])
    .with_hyperlight_sandbox()   // Enable Hyperlight with defaults
    .launch()
    .await?;

// Commands now execute inside a micro-VM
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

The `SandboxConfig` struct controls memory limits, timeouts, guest binary source, and pool sizing.

### Default values

| Setting | Default | Description |
|---|---|---|
| `memory_limit` | 64 MB | Maximum memory for the guest VM |
| `timeout` | 30 seconds | Maximum execution time |
| `guest_binary` | Embedded | Pre-built shell executor |
| `pool_size` | 4 | Number of pre-warmed sandboxes |
| `debug_output` | false | Forward guest print statements to host |

### Custom configuration

```rust
use acton_ai::prelude::*;
use acton_ai::tools::sandbox::SandboxConfig;
use std::time::Duration;

let config = SandboxConfig::new()
    .with_memory_limit(128 * 1024 * 1024)  // 128 MB
    .with_timeout(Duration::from_secs(60))   // 60 second timeout
    .with_pool_size(Some(8))                  // 8 pre-warmed sandboxes
    .with_debug_output(true);                 // Enable debug output

let runtime = ActonAI::builder()
    .app_name("custom-sandbox")
    .from_config()?
    .with_builtin_tools(&["bash", "rust_code"])
    .with_hyperlight_sandbox_config(config)
    .launch()
    .await?;
```

### Disabling the pool

For development or low-traffic scenarios, you can disable pooling so each request creates a fresh sandbox:

```rust
let config = SandboxConfig::new()
    .without_pool();  // Equivalent to .with_pool_size(None)
```

### Validation

Call `validate()` to check your configuration before launching:

```rust
let config = SandboxConfig::new()
    .with_memory_limit(128 * 1024 * 1024)
    .with_timeout(Duration::from_secs(60));

config.validate()?;  // Returns Err if invalid
```

Validation checks:
- Memory limit must be at least 1 MB
- Timeout must be greater than zero
- Pool size, if `Some`, must be greater than zero

---

## Sandbox pool for performance

Creating a micro-VM takes 1-2ms. For latency-sensitive applications, the sandbox pool pre-creates VMs so tool calls can acquire a ready sandbox instantly.

### Pool configuration with `PoolConfig`

```rust
use acton_ai::tools::sandbox::hyperlight::PoolConfig;

let pool_config = PoolConfig::new()
    .with_warmup_count(8)                      // Pre-warm 8 sandboxes per guest type
    .with_max_per_type(64)                     // Maximum 64 sandboxes per guest type
    .with_max_executions_before_recycle(500);   // Recycle after 500 executions
```

| Setting | Default | Description |
|---|---|---|
| `warmup_count` | 4 | Sandboxes to pre-create per guest type |
| `max_per_type` | 32 | Maximum sandboxes per guest type |
| `max_executions_before_recycle` | 1000 | Execution count before sandbox is replaced |

### Pre-warming the pool

Use `with_sandbox_pool` for the simplest pool configuration:

```rust
let runtime = ActonAI::builder()
    .app_name("pooled-sandbox")
    .from_config()?
    .with_builtins()
    .with_sandbox_pool(4)   // Keep 4 micro-VMs warm
    .launch()
    .await?;
```

For full control, use `with_sandbox_pool_config`:

```rust
let sandbox_config = SandboxConfig::new()
    .with_memory_limit(128 * 1024 * 1024);

let runtime = ActonAI::builder()
    .app_name("custom-pool")
    .from_config()?
    .with_builtins()
    .with_sandbox_pool_config(8, sandbox_config)
    .launch()
    .await?;
```

### How pooling works

1. At startup, the pool pre-creates sandboxes for each guest type (Shell, Http)
2. When a tool needs to execute, it acquires a sandbox from the pool
3. After execution, the sandbox is returned to the pool for reuse
4. Sandboxes are automatically recycled after `max_executions_before_recycle` uses
5. Dead or exhausted sandboxes are replaced with fresh instances

### Pool metrics

The pool tracks detailed metrics per guest type:

```rust
use acton_ai::tools::sandbox::{SandboxPool, GetPoolMetrics, PoolMetrics};

// Pool metrics include:
// - available: Number of ready sandboxes
// - in_use: Number currently executing
// - total_created: Lifetime creation count
// - pool_hits: Times a pre-warmed sandbox was acquired
// - pool_misses: Times a new sandbox had to be created
// - avg_creation_ms: Average sandbox creation time
```

---

## Guest binary sources

Hyperlight sandboxes execute a specially compiled guest binary inside the micro-VM. Acton AI provides three options:

```rust
use acton_ai::tools::sandbox::hyperlight::GuestBinarySource;

// Default: use the embedded shell executor (recommended)
let config = SandboxConfig::new();
// Equivalent to:
let config = SandboxConfig::new()
    .with_guest_binary(GuestBinarySource::Embedded);

// Load a custom guest binary from disk
let config = SandboxConfig::new()
    .with_guest_binary(GuestBinarySource::FromPath("/path/to/guest".into()));

// Use guest binary data already in memory
let binary_data: Vec<u8> = load_binary_somehow();
let config = SandboxConfig::new()
    .with_guest_binary(GuestBinarySource::FromBytes(binary_data));
```

{% callout type="note" title="Embedded guest is sufficient for most use cases" %}
The embedded shell executor handles bash commands and Rust code compilation. You only need a custom guest binary if you are extending the sandbox with additional execution capabilities.
{% /callout %}

---

## Path validation and security

Beyond sandboxing, Acton AI restricts which filesystem paths tools can access through the `PathValidator`.

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

## Sandbox error handling

When sandbox operations fail, errors are reported through `SandboxErrorKind`:

| Error | Cause |
|---|---|
| `HypervisorNotAvailable` | No KVM (Linux) or Hyper-V (Windows) |
| `CreationFailed` | Resource exhaustion, permission issues |
| `ExecutionTimeout` | Code exceeded the configured timeout |
| `MemoryLimitExceeded` | Code used more memory than allocated |
| `PoolExhausted` | All pooled sandboxes are in use |
| `GuestCallFailed` | Guest function invocation failed |
| `AlreadyDestroyed` | Sandbox used after `destroy()` |
| `InvalidConfiguration` | Bad configuration values |
| `ArchitectureNotSupported` | Non-x86_64 platform |

All sandbox errors convert to `ToolError::SandboxError` for consistent handling:

```rust
use acton_ai::tools::error::{ToolError, ToolErrorKind};

match result {
    Err(ref e) if e.is_retriable() => {
        // Sandbox errors are retriable -- try again
        println!("Retrying: {}", e);
    }
    Err(e) => {
        println!("Permanent failure: {}", e);
    }
    Ok(value) => { /* success */ }
}
```

---

## Configuration via TOML file

Sandbox settings can also be specified in your `acton-ai.toml` configuration file:

```toml
default_provider = "ollama"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434/v1"

[sandbox]
enabled = true
memory_limit_mb = 128
timeout_secs = 60
pool_size = 8
```

{% callout type="warning" title="Hypervisor required" %}
Hyperlight sandboxing requires hardware virtualization support. On Linux, verify KVM is available with `ls /dev/kvm`. On systems without a hypervisor, sandbox creation will fail with `SandboxErrorKind::HypervisorNotAvailable`. For development on unsupported platforms, omit the sandbox configuration -- tools will execute directly on the host.
{% /callout %}

---

## Next steps

- [Multi-Agent Collaboration](/docs/multi-agent-collaboration) -- configure per-agent tool access
- [Error Handling](/docs/error-handling) -- handle `ToolError` and `SandboxErrorKind`
- [Testing Your Agents](/docs/testing) -- use `StubSandbox` for tests without a hypervisor
