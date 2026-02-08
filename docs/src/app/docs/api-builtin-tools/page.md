---
title: Built-in Tools
---

Complete reference for the `BuiltinTools` registry and every built-in tool: its name, description, JSON schema parameters, and behavior.

---

## BuiltinTools

```rust
pub struct BuiltinTools { /* fields omitted */ }
```

Registry of built-in tools. Provides methods to access tool configurations and executors for registration with the tool system.

### Constructors

#### `all()`

```rust
pub fn all() -> Self
```

Creates a registry containing all built-in tools (except `rust_code`, which requires a Rust toolchain and is only available via `spawn_tool_actor()`). Returns 9 tools.

#### `select()`

```rust
pub fn select(tools: &[&str]) -> Result<Self, ToolError>
```

Creates a registry containing only the specified tools by name. Returns an error if an unknown tool name is provided.

```rust
let tools = BuiltinTools::select(&["read_file", "write_file", "glob"])?;
```

### Query methods

#### `available()`

```rust
pub fn available() -> Vec<&'static str>
```

Lists all available built-in tool names (10 total, including `rust_code`).

#### `get_config()`

```rust
pub fn get_config(&self, name: &str) -> Option<&ToolConfig>
```

Returns the configuration for a specific tool.

#### `get_executor()`

```rust
pub fn get_executor(&self, name: &str) -> Option<Arc<BoxedToolExecutor>>
```

Returns the executor for a specific tool.

#### `configs()`

```rust
pub fn configs(&self) -> impl Iterator<Item = (&String, &ToolConfig)>
```

Returns an iterator over all registered tool configurations.

#### `executors()`

```rust
pub fn executors(&self) -> impl Iterator<Item = (&String, &Arc<BoxedToolExecutor>)>
```

Returns an iterator over all registered tool executors.

#### `len()`

```rust
pub fn len(&self) -> usize
```

Returns the number of registered tools.

#### `is_empty()`

```rust
pub fn is_empty(&self) -> bool
```

Returns `true` if no tools are registered.

---

## Enabling built-in tools

There are several ways to make built-in tools available to the LLM.

### On the builder (recommended)

```rust
// All builtins, auto-enabled on every prompt
ActonAI::builder()
    .with_builtins()
    .launch().await?;

// Specific builtins, auto-enabled
ActonAI::builder()
    .with_builtin_tools(&["read_file", "glob", "grep", "bash"])
    .launch().await?;

// All builtins, but require manual opt-in per prompt
ActonAI::builder()
    .with_builtins()
    .manual_builtins()
    .launch().await?;
```

### On the prompt

When `manual_builtins()` is set, you must enable them on each prompt:

```rust
runtime.prompt("List files")
    .use_builtins()
    .collect()
    .await?;
```

See [`ActonAIBuilder`](/docs/api-acton-ai) and [`PromptBuilder`](/docs/api-prompt-builder) for full details.

---

## Tool reference

Each tool below lists its name (used by the LLM), description, JSON schema parameters, and behavior notes.

### read_file

Reads file contents with line numbers.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Absolute path to the file to read"
    },
    "offset": {
      "type": "integer",
      "description": "Line number to start from (1-indexed, default: 1)",
      "minimum": 1
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of lines to read (default: 2000)",
      "minimum": 1
    }
  },
  "required": ["path"]
}
```

**Returns:** `{ content, total_lines, start_line, end_line, truncated }`

**Behavior:**
- Path must be absolute. Relative paths are rejected.
- Content is returned with line numbers in `cat -n` style format.
- Lines longer than 2000 characters are truncated.
- Binary files are detected and rejected.
- Path is validated by `PathValidator` for security (prevents path traversal).

---

### write_file

Writes content to a file, creating parent directories if needed.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Absolute path to the file to write"
    },
    "content": {
      "type": "string",
      "description": "Content to write to the file"
    }
  },
  "required": ["path", "content"]
}
```

**Returns:** `{ success, path, bytes_written }`

**Behavior:**
- Path must be absolute.
- Parent directories are created automatically if they do not exist.
- Overwrites existing files.
- Marked as requiring sandbox (`sandboxed: true`).

---

### edit_file

Makes targeted string replacements in a file.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Absolute path to the file to edit"
    },
    "old_string": {
      "type": "string",
      "description": "Exact string to find and replace"
    },
    "new_string": {
      "type": "string",
      "description": "Replacement string"
    },
    "replace_all": {
      "type": "boolean",
      "description": "Replace all occurrences (default: false, which requires exactly one match)"
    }
  },
  "required": ["path", "old_string", "new_string"]
}
```

**Returns:** `{ success, path, replacements, diff }`

**Behavior:**
- Path must be absolute.
- When `replace_all` is `false` (default), the `old_string` must appear exactly once in the file. If it appears multiple times, the tool returns an error suggesting to use `replace_all: true` or provide more context.
- `old_string` and `new_string` must be different.
- Returns a simple diff-style output showing what changed.
- Marked as requiring sandbox (`sandboxed: true`).

---

### list_directory

Lists directory contents with metadata (type, size, modified time).

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Directory path to list"
    }
  },
  "required": ["path"]
}
```

**Returns:** `{ path, entries: [{ name, entry_type, size, modified }], count }`

**Behavior:**
- Path must be absolute.
- Each entry includes its type (`"file"`, `"dir"`, or `"symlink"`), size (for files), and last modified time in ISO 8601 format.
- Results are sorted alphabetically by name.

---

### glob

Finds files matching a glob pattern.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "pattern": {
      "type": "string",
      "description": "Glob pattern to match (e.g., '**/*.rs', 'src/**/*.ts')"
    },
    "path": {
      "type": "string",
      "description": "Base directory to search in (default: current working directory)"
    }
  },
  "required": ["pattern"]
}
```

**Returns:** `{ matches: [string], count, truncated, pattern, base_path }`

**Behavior:**
- Supports recursive patterns with `**`.
- Returns up to 1000 matching paths.
- Results are sorted alphabetically.
- If `path` is provided, it must be absolute.

---

### grep

Searches file contents using regex patterns.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "pattern": {
      "type": "string",
      "description": "Regex pattern to search for"
    },
    "path": {
      "type": "string",
      "description": "File or directory to search in (default: current directory)"
    },
    "glob": {
      "type": "string",
      "description": "File pattern to filter files (e.g., '*.rs', '*.{ts,tsx}')"
    },
    "context_lines": {
      "type": "integer",
      "description": "Number of context lines before and after each match",
      "minimum": 0,
      "maximum": 10
    },
    "ignore_case": {
      "type": "boolean",
      "description": "Case insensitive search (default: false)"
    }
  },
  "required": ["pattern"]
}
```

**Returns:** `{ matches: [{ file, line, content, before, after }], count, files_searched, truncated, pattern }`

**Behavior:**
- Uses Rust's `regex` crate for pattern matching.
- Returns up to 500 matches.
- Skips hidden files (starting with `.`), binary files, and files larger than 10MB.
- When searching a directory, walks the tree recursively.
- Context lines (before/after) are included when `context_lines > 0`.
- If `path` is provided, it must be absolute.

---

### bash

Executes shell commands and captures output.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "command": {
      "type": "string",
      "description": "The shell command to execute"
    },
    "timeout": {
      "type": "integer",
      "description": "Timeout in seconds (default: 120, max: 600)",
      "minimum": 1,
      "maximum": 600
    },
    "cwd": {
      "type": "string",
      "description": "Absolute path to working directory. Only specify if you need a different directory than the current one."
    }
  },
  "required": ["command"]
}
```

**Returns:** `{ exit_code, stdout, stderr, success, truncated }`

**Behavior:**
- Runs the command via `bash -c`.
- stdin is connected to `/dev/null` (no interactive input).
- Output is captured and truncated at 1MB per stream (stdout/stderr).
- If `cwd` is provided, it must be an absolute path to an existing directory.
- Rejects obviously dangerous commands (`rm -rf /`, fork bombs, `mkfs`, `dd` to devices).
- Marked as requiring sandbox (`sandboxed: true`).
- Killed after the timeout expires.

{% callout type="warning" title="Security" %}
The `bash` tool executes arbitrary shell commands. In production, consider enabling the Hyperlight sandbox (`with_hyperlight_sandbox()` or `with_sandbox_pool()`) for hardware-level isolation.
{% /callout %}

---

### calculate

Evaluates mathematical expressions using the `fasteval` crate.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "expression": {
      "type": "string",
      "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'abs(-5)', 'sin(pi()/2)')"
    },
    "variables": {
      "type": "object",
      "description": "Optional variable bindings (e.g., {\"x\": 5, \"y\": 10})",
      "additionalProperties": {
        "type": "number"
      }
    }
  },
  "required": ["expression"]
}
```

**Returns:** `{ result, formatted, expression, is_special }`

**Behavior:**
- Supports arithmetic operators: `+`, `-`, `*`, `/`, `^`, `%`.
- Built-in functions: `sin`, `cos`, `tan`, `log`, `abs`, `min`, `max`, `floor`, `ceil`, `round`, `sqrt`, and more.
- Constants via functions: `pi()`, `e()`.
- User-defined variables can be passed via the `variables` field.
- Results are formatted nicely: integer results show as integers, floats as floats.
- Special values (NaN, Infinity) are detected and flagged.
- Expression length is capped at 1000 characters.
- This is a safe math parser -- it does **not** execute arbitrary code.

---

### web_fetch

Fetches content from a URL. Supports GET and POST methods with custom headers.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "url": {
      "type": "string",
      "description": "URL to fetch (must be http or https)"
    },
    "method": {
      "type": "string",
      "enum": ["GET", "POST"],
      "description": "HTTP method (default: GET)"
    },
    "headers": {
      "type": "object",
      "description": "Optional HTTP headers",
      "additionalProperties": {
        "type": "string"
      }
    },
    "body": {
      "type": "string",
      "description": "Optional request body (for POST requests)"
    },
    "timeout": {
      "type": "integer",
      "description": "Timeout in seconds (default: 30, max: 120)",
      "minimum": 1,
      "maximum": 120
    }
  },
  "required": ["url"]
}
```

**Returns:** `{ status_code, success, content_type, body, body_length, truncated, headers }`

**Behavior:**
- Only `http` and `https` schemes are allowed.
- Blocks requests to localhost, `127.0.0.1`, `::1`, and private IP ranges (`192.168.*`, `10.*`, `172.16-31.*`) for security.
- Response body is truncated at 5MB.
- Default timeout is 30 seconds.
- User agent is set to `acton-ai/0.1`.

---

### rust_code

Executes Rust code in a secure Hyperlight sandbox. Code is compiler-verified before execution.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "string",
      "description": "Rust function body. Receives `input: String`, must return `String`. Example: `input.to_uppercase()`"
    },
    "input": {
      "type": "string",
      "description": "Input string passed to the function (default: empty string)"
    },
    "timeout_secs": {
      "type": "integer",
      "description": "Timeout in seconds (default: 30, max: 300)"
    }
  },
  "required": ["code"]
}
```

**Returns:** `{ output, success }`

**Behavior:**
- Code is wrapped in a `no_std` template with `#![forbid(unsafe_code)]`.
- Clippy runs with `-D warnings` before compilation.
- Compiled to `x86_64-unknown-none` target.
- Binary executes inside a Hyperlight micro-VM with hardware isolation.
- Defense-in-depth security: compile-time checks, unsafe forbidden, hardware sandbox.
- Requires the Rust toolchain to be installed.
- Not included in `BuiltinTools::all()`. Available via `spawn_tool_actor("rust_code")`.

{% callout type="note" title="Requires Hyperlight" %}
The `rust_code` tool requires a hypervisor (KVM on Linux, Hyper-V on Windows) and the Rust toolchain. It is designed for scenarios where you need safe, isolated code execution.
{% /callout %}

---

## Feature-gated tools

The following tools are only available when the `agent-skills` feature is enabled.

### list_skills

Lists available agent skills with their descriptions.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "filter": {
      "type": "string",
      "description": "Optional filter pattern for skill names (case-insensitive substring match)"
    }
  }
}
```

**Returns:** `{ skills: [{ name, description, tags }], count }`

**Behavior:**
- Returns all registered skills, optionally filtered by name.
- Use this to discover available skills before activating one.

### activate_skill

Activates a skill and receives its full instructions.

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Name of the skill to activate"
    }
  },
  "required": ["name"]
}
```

**Returns:** `{ name, description, instructions, path, tags }`

**Behavior:**
- Loads the full instructions for the named skill.
- Call `list_skills` first to see what is available.
- The `instructions` field contains the complete skill content that guides the agent.

### Enabling skill tools

```rust
use acton_ai::tools::builtins::{spawn_skill_tool_actors, skill_tool_names};
use acton_ai::skills::SkillRegistry;
use std::sync::Arc;

let registry = Arc::new(SkillRegistry::new());
let skill_tools = spawn_skill_tool_actors(&mut runtime, registry).await;

// skill_tool_names() returns ["list_skills", "activate_skill"]
```

---

## Tool summary table

| Tool | Sandboxed | Category | Description |
|---|---|---|---|
| `read_file` | No | Filesystem | Read file contents with line numbers |
| `write_file` | Yes | Filesystem | Write content to files |
| `edit_file` | Yes | Filesystem | Targeted string replacements |
| `list_directory` | No | Filesystem | List directory contents with metadata |
| `glob` | No | Filesystem | Find files matching glob patterns |
| `grep` | No | Filesystem | Search file contents with regex |
| `bash` | Yes | Execution | Execute shell commands |
| `calculate` | No | Computation | Evaluate math expressions |
| `rust_code` | Yes | Execution | Compiler-verified Rust code execution |
| `web_fetch` | No | Web | Fetch content from URLs |
| `list_skills` | No | Skills | List available agent skills |
| `activate_skill` | No | Skills | Activate a skill for the agent |

{% callout type="note" title="Sandboxed tools" %}
Tools marked as sandboxed (`write_file`, `edit_file`, `bash`, `rust_code`) are flagged for execution in a Hyperlight micro-VM when sandbox mode is enabled. Without sandbox mode, they still execute but without hardware isolation.
{% /callout %}
