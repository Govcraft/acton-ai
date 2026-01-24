//! Example: Per-Agent Tool Configuration
//!
//! This example demonstrates how to configure different builtin tools for
//! different agents using the per-agent tools API. It shows:
//!
//! 1. Using `AgentConfig::with_tools(&["read_file", "glob"])` for limited tools
//! 2. Using `AgentConfig::with_all_builtins()` for full tool access
//! 3. Spawning agents with different tool configurations
//! 4. Demonstrating that each agent only has access to its configured tools
//!
//! ## Architecture
//!
//! This example uses the low-level API to spawn Agent actors with specific
//! tool configurations. Each agent:
//! - Has its own `AgentConfig` specifying which tools it can use
//! - Gets tool actors spawned and registered via `RegisterToolActors`
//! - Only sees the tools it was configured with
//!
//! ## Configuration
//!
//! Create an `acton-ai.toml` file in the project root or at
//! `~/.config/acton-ai/config.toml`:
//!
//! ```toml
//! default_provider = "ollama"
//!
//! [providers.ollama]
//! type = "ollama"
//! model = "qwen2.5:7b"
//! base_url = "http://localhost:11434/v1"
//! ```
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example per_agent_tools
//! ```

use acton_ai::agent::{Agent, AgentConfig, InitAgent, RegisterToolActors};
use acton_ai::config;
use acton_ai::llm::LLMProvider;
use acton_ai::messages::ToolDefinition;
use acton_ai::prelude::*;
use acton_ai::tools::builtins::{get_tool_definition, spawn_tool_actor, BuiltinTools};
use colored::Colorize;
use std::time::Duration;

/// Print the list of tools available for an agent configuration.
fn print_tools(name: &str, config: &AgentConfig) {
    let tools = &config.tools;
    if tools.is_empty() {
        println!(
            "  {} {}",
            name.cyan().bold(),
            "(no tools configured)".dimmed()
        );
    } else {
        println!(
            "  {} {}",
            name.cyan().bold(),
            format!("({} tools)", tools.len()).dimmed()
        );
        for tool in tools {
            println!("    - {}", tool.green());
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "Per-Agent Tools Demo".cyan().bold());
    println!("{}", "====================".dimmed());
    eprintln!("{}", "Loading configuration from acton-ai.toml...".dimmed());
    println!();

    // =========================================================================
    // Part 1: Demonstrate AgentConfig tool configuration
    // =========================================================================

    println!(
        "{}",
        "Part 1: AgentConfig Tool Configuration".yellow().bold()
    );
    println!();

    // Agent with specific tools (FileReader)
    let reader_config = AgentConfig::new(
        "You are a file reader assistant. You can read files and search for files \
         using glob patterns. You do NOT have access to bash, write, or edit tools.",
    )
    .with_tools(&["read_file", "glob"])
    .with_name("FileReader");

    // Agent with calculation and grep tools (Researcher)
    let researcher_config = AgentConfig::new(
        "You are a research assistant. You can search file contents with grep \
         and perform calculations. You do NOT have access to file read/write tools.",
    )
    .with_tools(&["grep", "calculate"])
    .with_name("Researcher");

    // Agent with all builtin tools (PowerUser)
    let power_config = AgentConfig::new(
        "You are a power user assistant with access to all available tools. \
         You can read, write, edit files, run bash commands, search, and calculate.",
    )
    .with_all_builtins()
    .with_name("PowerUser");

    // Agent with no tools (Conversational)
    let conversational_config =
        AgentConfig::new("You are a helpful conversational assistant with no tool access.")
            .with_name("Conversational");

    println!("{}", "Configured agent tool access:".white().bold());
    print_tools("FileReader", &reader_config);
    print_tools("Researcher", &researcher_config);
    print_tools("PowerUser", &power_config);
    print_tools("Conversational", &conversational_config);
    println!();

    // =========================================================================
    // Part 2: Show available builtin tools
    // =========================================================================

    println!("{}", "Part 2: Available Builtin Tools".yellow().bold());
    println!();

    let all_builtins = BuiltinTools::available();
    println!(
        "{} {} {}",
        "Total available tools:".white(),
        all_builtins.len().to_string().green().bold(),
        "builtins".dimmed()
    );
    for tool_name in &all_builtins {
        let def = get_tool_definition(tool_name)?;
        // Truncate description for display
        let desc: String = def.description.chars().take(60).collect();
        let desc = if def.description.len() > 60 {
            format!("{}...", desc)
        } else {
            desc
        };
        println!("  {} - {}", tool_name.green().bold(), desc.dimmed());
    }
    println!();

    // =========================================================================
    // Part 3: Spawn agents with their tool configurations
    // =========================================================================

    println!("{}", "Part 3: Spawning Agents with Tools".yellow().bold());
    println!();

    // Launch the actor runtime
    let mut runtime = ActonApp::launch_async().await;

    // Load configuration from acton-ai.toml and spawn the LLM provider
    let acton_config = config::load()?;
    let default_name = acton_config
        .effective_default()
        .ok_or_else(|| anyhow::anyhow!("No default provider configured"))?;
    let named_config = acton_config
        .providers
        .get(default_name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{}' not found", default_name))?;

    let provider_config = named_config
        .to_provider_config()
        .with_timeout(Duration::from_secs(120))
        .with_max_tokens(512);
    let _provider_handle = LLMProvider::spawn(&mut runtime, provider_config).await;

    // Spawn the FileReader agent with its configured tools
    println!(
        "{} {}",
        "[FileReader]".blue().bold(),
        "Spawning with read_file and glob tools...".dimmed()
    );
    let reader_agent = Agent::create(&mut runtime);

    // Subscribe to streaming events BEFORE starting
    reader_agent.handle().subscribe::<LLMStreamStart>().await;
    reader_agent.handle().subscribe::<LLMStreamToken>().await;
    reader_agent.handle().subscribe::<LLMStreamToolCall>().await;
    reader_agent.handle().subscribe::<LLMStreamEnd>().await;

    let reader_handle = reader_agent.start().await;

    // Initialize the agent
    reader_handle
        .send(InitAgent {
            config: reader_config.clone(),
        })
        .await;

    // Spawn tool actors for the FileReader agent's configured tools
    let mut reader_tools: Vec<(String, ActorHandle, ToolDefinition)> = Vec::new();
    for tool_name in &reader_config.tools {
        match spawn_tool_actor(&mut runtime, tool_name).await {
            Ok((handle, definition)) => {
                println!("    {} {}", "Registered:".green(), definition.name.cyan());
                reader_tools.push((tool_name.clone(), handle, definition));
            }
            Err(e) => {
                eprintln!("    {} {} - {}", "Failed:".red(), tool_name, e);
            }
        }
    }

    // Register tools with the agent
    reader_handle
        .send(RegisterToolActors {
            tools: reader_tools,
        })
        .await;

    println!(
        "    {} FileReader ready with {} tools\n",
        "OK:".green().bold(),
        reader_config.tools.len()
    );

    // Spawn the PowerUser agent with ALL tools
    println!(
        "{} {}",
        "[PowerUser]".magenta().bold(),
        "Spawning with all builtin tools...".dimmed()
    );
    let power_agent = Agent::create(&mut runtime);

    // Subscribe to streaming events BEFORE starting
    power_agent.handle().subscribe::<LLMStreamStart>().await;
    power_agent.handle().subscribe::<LLMStreamToken>().await;
    power_agent.handle().subscribe::<LLMStreamToolCall>().await;
    power_agent.handle().subscribe::<LLMStreamEnd>().await;

    let power_handle = power_agent.start().await;

    // Initialize the agent
    power_handle
        .send(InitAgent {
            config: power_config.clone(),
        })
        .await;

    // Spawn all tool actors for the PowerUser agent
    let mut power_tools: Vec<(String, ActorHandle, ToolDefinition)> = Vec::new();
    for tool_name in &power_config.tools {
        match spawn_tool_actor(&mut runtime, tool_name).await {
            Ok((handle, definition)) => {
                println!("    {} {}", "Registered:".green(), definition.name.cyan());
                power_tools.push((tool_name.clone(), handle, definition));
            }
            Err(e) => {
                eprintln!("    {} {} - {}", "Failed:".red(), tool_name, e);
            }
        }
    }

    // Register tools with the agent
    power_handle
        .send(RegisterToolActors { tools: power_tools })
        .await;

    println!(
        "    {} PowerUser ready with {} tools\n",
        "OK:".green().bold(),
        power_config.tools.len()
    );

    // =========================================================================
    // Part 4: Summary
    // =========================================================================

    println!("{}", "Part 4: Summary".yellow().bold());
    println!();
    println!("{}", "Agent tool isolation achieved:".white().bold());
    println!(
        "  {} - {} tools: {}",
        "FileReader".blue().bold(),
        reader_config.tools.len(),
        reader_config.tools.join(", ").green()
    );
    println!(
        "  {} - {} tools (all builtins)",
        "PowerUser".magenta().bold(),
        power_config.tools.len()
    );
    println!(
        "  {} - {} tools (none configured)",
        "Conversational".dimmed(),
        conversational_config.tools.len()
    );
    println!();
    println!(
        "{}",
        "Each agent only has access to its configured tools!".cyan()
    );
    println!();

    // =========================================================================
    // Part 5: Key API Methods Reference
    // =========================================================================

    println!("{}", "Key API Methods:".yellow().bold());
    println!();
    println!(
        "  {} - Add specific tools by name",
        "AgentConfig::with_tools(&[...])".cyan()
    );
    println!(
        "  {} - Add a single tool",
        "AgentConfig::with_tool(\"name\")".cyan()
    );
    println!(
        "  {} - Enable all 9 builtin tools",
        "AgentConfig::with_all_builtins()".cyan()
    );
    println!(
        "  {} - List all available tools",
        "BuiltinTools::available()".cyan()
    );
    println!(
        "  {} - Get tool definition by name",
        "get_tool_definition(\"name\")".cyan()
    );
    println!(
        "  {} - Spawn a tool actor",
        "spawn_tool_actor(&runtime, \"name\")".cyan()
    );
    println!();

    // Shutdown
    runtime.shutdown_all().await?;

    Ok(())
}
