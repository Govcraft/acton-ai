---
title: Multi-Agent Collaboration
---

Acton AI lets you orchestrate multiple specialized agents that collaborate to solve complex tasks. Because every agent is an actor, they run concurrently with built-in fault isolation -- one agent crashing never takes down another.

This guide covers the high-level prompt-chaining pattern, the low-level agent actor API with per-agent tool configuration, and the delegation tracking system for coordinating work between agents.

---

## How agents collaborate through the Kernel

Every agent in Acton AI is an actor managed by a central **Kernel**. The Kernel handles:

- **Agent lifecycle** -- spawning, initializing, and stopping agents
- **Message routing** -- delivering messages between agents
- **Tool registration** -- ensuring each agent only accesses its configured tools

Agents communicate by sending messages through the actor system. The Kernel routes these messages and tracks agent states, enabling patterns like sequential pipelines, fan-out/fan-in, and delegation hierarchies.

---

## Pattern 1: Sequential prompt chaining (high-level API)

The simplest multi-agent pattern chains prompts together, where each "agent" is a prompt with a distinct system prompt and tool set. The output of one feeds into the next.

```rust
use acton_ai::prelude::*;
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let runtime = ActonAI::builder()
        .app_name("multi-agent-research")
        .from_config()?
        .with_builtin_tools(&["web_fetch", "calculate"])
        .launch()
        .await?;

    let user_question = "What is the population of the USA, and what's 15% of that?";

    // === Agent 1: Researcher ===
    let research_result = runtime
        .prompt(&format!(
            "The user asked: \"{}\"\n\nFind the population data using web_fetch.",
            user_question
        ))
        .system(
            "You are a research specialist. Use the web_fetch tool to \
             retrieve factual data from URLs.",
        )
        .use_builtins()
        .on_token(|token| {
            print!("{token}");
            std::io::stdout().flush().ok();
        })
        .collect()
        .await?;

    // === Agent 2: Analyst ===
    let analysis_result = runtime
        .prompt(&format!(
            "Research findings: '{}'\n\nCalculate 15% of the population.",
            research_result.text
        ))
        .system(
            "You are a data analyst. ALWAYS use the calculate tool \
             for math operations.",
        )
        .use_builtins()
        .collect()
        .await?;

    // === Agent 3: Coordinator ===
    let final_answer = runtime
        .prompt(&format!(
            "Combine these findings into a clear answer:\n\n\
             Research: {}\nAnalysis: {}\nQuestion: {}",
            research_result.text, analysis_result.text, user_question
        ))
        .system(
            "You are a coordinator who synthesizes information from \
             specialists into concise, direct answers.",
        )
        .collect()
        .await?;

    println!("Final answer: {}", final_answer.text);

    // Token usage summary
    let total = research_result.token_count
        + analysis_result.token_count
        + final_answer.token_count;
    println!("Total tokens used: {}", total);

    Ok(())
}
```

{% callout type="note" title="Each prompt is stateless" %}
In this pattern, each prompt is independent. The coordinator only knows what previous agents found because their results are included in its prompt. For stateful multi-turn collaboration, use the Conversation API or the low-level agent actor API described below.
{% /callout %}

---

## Pattern 2: Per-agent tool configuration (low-level API)

For fine-grained control, use the agent actor API directly. Each agent gets its own `AgentConfig` that specifies exactly which tools it can access.

### Configuring agent tools

```rust
use acton_ai::agent::AgentConfig;

// Agent with specific tools
let reader_config = AgentConfig::new(
    "You are a file reader assistant. You can read files and search \
     for files using glob patterns.",
)
.with_tools(&["read_file", "glob"])
.with_name("FileReader");

// Agent with calculation and search tools
let researcher_config = AgentConfig::new(
    "You are a research assistant. You can search file contents \
     with grep and perform calculations.",
)
.with_tools(&["grep", "calculate"])
.with_name("Researcher");

// Agent with all builtin tools
let power_config = AgentConfig::new(
    "You are a power user with access to all available tools.",
)
.with_all_builtins()
.with_name("PowerUser");

// Agent with no tools (conversation only)
let conversational_config = AgentConfig::new(
    "You are a helpful conversational assistant with no tool access.",
)
.with_name("Conversational");
```

### Key `AgentConfig` methods

| Method | Description |
|---|---|
| `AgentConfig::new(system_prompt)` | Create config with a system prompt |
| `.with_tools(&["read_file", "bash"])` | Enable specific tools by name |
| `.with_tool("read_file")` | Add a single tool |
| `.with_all_builtins()` | Enable all available builtin tools |
| `.with_name("MyAgent")` | Set a display name |
| `.with_max_conversation_length(50)` | Limit conversation history |
| `.with_streaming(false)` | Disable streaming responses |

### Spawning agents with their tools

```rust
use acton_ai::agent::{Agent, AgentConfig, InitAgent, RegisterToolActors};
use acton_ai::tools::builtins::{spawn_tool_actor, BuiltinTools};
use acton_ai::messages::ToolDefinition;
use acton_ai::prelude::*;

// Launch the actor runtime
let mut runtime = ActonApp::launch_async().await;

// Create and start the agent
let reader_agent = Agent::create(&mut runtime);
let reader_handle = reader_agent.start().await;

// Initialize with configuration
reader_handle
    .send(InitAgent {
        config: reader_config.clone(),
    })
    .await;

// Spawn and register tool actors for this agent
let mut tools: Vec<(String, ActorHandle, ToolDefinition)> = Vec::new();
for tool_name in &reader_config.tools {
    if let Ok((handle, definition)) = spawn_tool_actor(&mut runtime, tool_name).await {
        tools.push((tool_name.clone(), handle, definition));
    }
}

reader_handle
    .send(RegisterToolActors { tools })
    .await;
```

{% callout type="warning" title="Tool isolation is per-agent" %}
Each agent only sees the tools registered with it via `RegisterToolActors`. An agent configured with `["read_file", "glob"]` cannot access `bash` or `write_file`, even if those tools are available in the runtime. This provides defense-in-depth for multi-agent systems.
{% /callout %}

---

## Agent delegation with DelegatedTask

For complex workflows where agents need to assign work to each other, Acton AI provides a formal delegation system through `DelegatedTask` and `DelegationTracker`.

### Task lifecycle

A delegated task moves through these states:

```text
Pending --> Accepted --> Completed
                    \-> Failed
```

| State | Meaning |
|---|---|
| `Pending` | Task sent, awaiting acceptance |
| `Accepted` | Target agent accepted the task |
| `Completed` | Task finished successfully with a result |
| `Failed` | Task failed with an error message |

### Creating and tracking tasks

```rust
use acton_ai::agent::delegation::{
    DelegatedTask, DelegatedTaskState, DelegationTracker,
};
use acton_ai::types::{AgentId, TaskId};
use std::time::Duration;

let mut tracker = DelegationTracker::new();

// Create a task delegated to another agent
let task_id = TaskId::new();
let target_agent = AgentId::new();
let task = DelegatedTask::new(
    task_id.clone(),
    target_agent,
    "code_review".to_string(),
)
.with_deadline(Duration::from_secs(60));

// Track the outgoing task
tracker.track_outgoing(task);

// Later, when the target agent completes the task:
if let Some(task) = tracker.get_outgoing_mut(&task_id) {
    task.accept();
    task.complete(serde_json::json!({
        "review": "Looks good, approved.",
        "issues": 0
    }));
}

// Check pending work
println!("Pending tasks: {}", tracker.pending_outgoing_count());
```

### Tracking incoming tasks

Agents can also track tasks delegated _to_ them:

```rust
// When this agent receives a task from another agent
let from_agent = AgentId::new();
let incoming_task_id = TaskId::new();

tracker.track_incoming(
    incoming_task_id.clone(),
    from_agent,
    "code_review".to_string(),
);

// Accept the task
tracker.accept_incoming(&incoming_task_id);

// After completing the work, remove it
tracker.remove_incoming(&incoming_task_id);
```

### Deadline monitoring

Tasks can have optional deadlines. Use `is_overdue()` to detect tasks that have exceeded their time limit:

```rust
let task = DelegatedTask::new(task_id, agent_id, "analysis".to_string())
    .with_deadline(Duration::from_secs(30));

// ... time passes ...

if task.is_overdue() {
    eprintln!("Task {} exceeded deadline!", task.task_type);
}
```

### Cleanup

Use `cleanup_completed` to remove terminal tasks from the tracker:

```rust
// Remove all completed and failed tasks
tracker.cleanup_completed();
```

---

## Agent states and lifecycle

Every agent transitions through a defined set of states during its lifecycle:

| State | Description | Can accept prompts? |
|---|---|---|
| `Idle` | Waiting for input | Yes |
| `Thinking` | Processing and reasoning about input | No |
| `Executing` | Running a tool | No |
| `Waiting` | Waiting for external input (tool result, user confirmation) | No |
| `Completed` | Task finished | Yes |
| `Stopping` | Agent is shutting down | No |

```rust
use acton_ai::agent::AgentState;

let state = AgentState::Idle;

assert!(state.can_accept_prompt());  // true for Idle and Completed
assert!(!state.is_active());          // true for Thinking, Executing, Waiting
assert!(!state.is_terminal());        // true only for Stopping
```

---

## Practical example: research team

Here is a complete multi-agent workflow that demonstrates sequential delegation with tool usage. You can find the full runnable version at `examples/multi_agent.rs`.

```rust
use acton_ai::prelude::*;
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let runtime = ActonAI::builder()
        .app_name("multi-agent-research")
        .from_config()?
        .with_builtin_tools(&["web_fetch", "calculate"])
        .launch()
        .await?;

    let question = "What is the population of the USA, and what's 15% of that?";

    // Step 1: Researcher fetches data
    let research = runtime
        .prompt("Find the USA population using web_fetch.")
        .system("You are a research specialist. Use web_fetch to find facts.")
        .use_builtins()
        .on_token(|t| { print!("{t}"); std::io::stdout().flush().ok(); })
        .collect()
        .await?;

    // Step 2: Analyst runs calculations
    let analysis = runtime
        .prompt(&format!(
            "Research: '{}'\nCalculate 15% of the population using calculate.",
            research.text
        ))
        .system("You are a data analyst. Always use the calculate tool.")
        .use_builtins()
        .collect()
        .await?;

    // Step 3: Coordinator synthesizes
    let answer = runtime
        .prompt(&format!(
            "Research: {}\nAnalysis: {}\nQuestion: {}\n\nSynthesize a final answer.",
            research.text, analysis.text, question
        ))
        .system("Combine specialist findings into a clear, concise answer.")
        .collect()
        .await?;

    println!("\nFinal: {}", answer.text);
    println!(
        "Tool calls: research={}, analysis={}, synthesis={}",
        research.tool_calls.len(),
        analysis.tool_calls.len(),
        answer.tool_calls.len()
    );

    Ok(())
}
```

---

## Next steps

- [Secure Tool Execution](/docs/secure-tool-execution) -- sandbox agent tool calls in Hyperlight micro-VMs
- [Conversation Management](/docs/conversation-management) -- manage multi-turn state across agents
- [Error Handling](/docs/error-handling) -- handle errors from multi-agent operations with `MultiAgentError`
- [Testing Your Agents](/docs/testing) -- test multi-agent workflows with mock providers
