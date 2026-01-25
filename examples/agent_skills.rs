//! Example: Agent Skills with Dynamic Tool Discovery
//!
//! This example demonstrates the feature-gated agent skills functionality with a
//! code assistant scenario. It shows:
//!
//! 1. Loading skills from markdown files with YAML frontmatter
//! 2. Using `SkillRegistry::from_paths()` to load skills from a directory
//! 3. Registering skill tools (`list_skills`, `activate_skill`) with the high-level API
//! 4. Combining skill tools with standard builtin tools (read_file, glob)
//! 5. Interactive demo where the agent discovers and uses skills
//!
//! ## Architecture
//!
//! This example uses the high-level ActonAI facade API with custom skill tools
//! registered via the fluent `PromptBuilder::with_tool()` method. Skills are loaded
//! from `examples/skills/` directory containing markdown files with YAML frontmatter.
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
//! cargo run --example agent_skills --features agent-skills
//! ```

#[cfg(feature = "agent-skills")]
mod skills_example {
    use acton_ai::prelude::*;
    use acton_ai::tools::builtins::{ActivateSkillTool, ListSkillsTool};
    use acton_ai::tools::ToolExecutorTrait;
    use colored::Colorize;
    use std::io::Write;
    use std::path::Path;
    use std::sync::Arc;

    /// Summarize a tool result for display.
    fn summarize_tool_result(
        tool_name: &str,
        result: &Result<serde_json::Value, String>,
    ) -> String {
        match result {
            Ok(value) => match tool_name {
                "list_skills" => {
                    let count = value.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                    format!("(found {} skills)", count)
                }
                "activate_skill" => {
                    let name = value
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    format!("(activated '{}')", name)
                }
                "read_file" => {
                    let lines = value
                        .get("line_count")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    format!("({} lines)", lines)
                }
                "glob" => {
                    let count = value
                        .get("count")
                        .and_then(|v| v.as_i64())
                        .unwrap_or_else(|| {
                            value
                                .get("paths")
                                .and_then(|v| v.as_array())
                                .map(|a| a.len() as i64)
                                .unwrap_or(0)
                        });
                    format!("({} files)", count)
                }
                _ => "(OK)".to_string(),
            },
            Err(e) => format!("(Error: {})", e),
        }
    }

    pub async fn run() -> anyhow::Result<()> {
        println!("{}", "Agent Skills Demo".cyan().bold());
        println!("{}", "=================".dimmed());
        eprintln!("{}", "Loading configuration from acton-ai.toml...".dimmed());
        println!();

        // =========================================================================
        // Part 1: Setup - Load configuration and launch runtime
        // =========================================================================

        println!(
            "{}",
            "Part 1: Setup - Loading Configuration".yellow().bold()
        );
        println!();

        // Launch the high-level ActonAI runtime from config file
        let ai_runtime = ActonAI::builder()
            .app_name("skills-demo")
            .from_config()?
            .with_builtin_tools(&["read_file", "glob"]) // Include filesystem tools
            .launch()
            .await?;

        println!(
            "  {} Provider '{}' ready",
            "OK:".green().bold(),
            ai_runtime.default_provider_name().cyan()
        );
        println!();

        // =========================================================================
        // Part 2: Load Skills - Use SkillRegistry::from_paths()
        // =========================================================================

        println!("{}", "Part 2: Loading Skills".yellow().bold());
        println!();

        // Load skills from the examples/skills directory
        let skills_path = Path::new("examples/skills");
        let registry = SkillRegistry::from_paths(&[skills_path]).await?;

        // Wrap in Arc for thread-safe sharing with tool executors
        let skill_registry = Arc::new(registry);

        // Get skill tool definitions (used when registering tools on the prompt)
        let list_skills_def = ListSkillsTool::config().definition;
        let activate_skill_def = ActivateSkillTool::config().definition;

        // =========================================================================
        // Part 3: Display Available Skills
        // =========================================================================

        println!("{}", "Part 3: Available Skills".yellow().bold());
        println!();

        println!(
            "{} {} skills loaded from {}",
            "Skills:".white().bold(),
            skill_registry.len().to_string().green().bold(),
            skills_path.display().to_string().cyan()
        );

        for skill_info in skill_registry.list() {
            println!(
                "  {} - {}",
                skill_info.name.green().bold(),
                skill_info.description.dimmed()
            );
            if !skill_info.tags.is_empty() {
                println!(
                    "    {} {}",
                    "Tags:".dimmed(),
                    skill_info.tags.join(", ").cyan()
                );
            }
        }
        println!();

        // =========================================================================
        // Part 4: Display Registered Tools
        // =========================================================================

        println!("{}", "Part 4: Tools Available to Agent".yellow().bold());
        println!();

        println!(
            "    {} {}",
            "Skill tools:".green(),
            "list_skills, activate_skill".cyan()
        );
        println!(
            "    {} {}",
            "Builtin tools:".green(),
            "read_file, glob".cyan()
        );
        println!(
            "    {} Agent ready with 4 total tools\n",
            "OK:".green().bold()
        );

        // =========================================================================
        // Part 5: Interactive Demo - Send prompt, stream response
        // =========================================================================

        println!("{}", "Part 5: Interactive Skills Demo".yellow().bold());
        println!();

        let prompt = "List all available skills, then activate the one that would help \
                      me review code. Once activated, read the file at \
                      examples/skills/sample_code.rs and perform a code review using \
                      the skill's instructions.";

        println!("{} {}\n", "User:".white().bold(), prompt);
        println!(
            "{} {}",
            "[SkillsAgent]".blue().bold(),
            "Discovering and using skills...".dimmed()
        );

        // Build the system prompt with skill context
        let system_prompt = "You are a code assistant with access to skills. \
             You have access to these tools:\n\
             - list_skills: Discover available skills\n\
             - activate_skill: Load a skill's full instructions\n\
             - read_file: Read file contents\n\
             - glob: Find files by pattern\n\n\
             Use list_skills first to see what's available, then activate_skill \
             to load the detailed instructions for a skill.";

        // Send prompt with skill tools registered via with_tool()
        // This is the key fix: we register skill tools directly on the PromptBuilder
        // Clone the registry reference for each closure
        let registry_for_list = Arc::clone(&skill_registry);
        let registry_for_activate = Arc::clone(&skill_registry);

        let result = ai_runtime
            .prompt(prompt)
            .system(system_prompt)
            .use_builtins() // Adds read_file, glob (configured in builder)
            .with_tool(list_skills_def, move |args| {
                let tool = ListSkillsTool::new(Arc::clone(&registry_for_list));
                async move { tool.execute(args).await }
            })
            .with_tool(activate_skill_def, move |args| {
                let tool = ActivateSkillTool::new(Arc::clone(&registry_for_activate));
                async move { tool.execute(args).await }
            })
            .on_token(|token| {
                print!("{token}");
                std::io::stdout().flush().ok();
            })
            .collect()
            .await?;

        println!("\n");

        // =========================================================================
        // Part 6: Summary - Show tokens and tool usage
        // =========================================================================

        println!("{}", "=========================".dimmed());
        println!("{}", "Summary".yellow().bold());
        println!();

        // Tool usage
        println!("{}", "Tool usage:".cyan().bold());
        println!(
            "  {} {} tool calls",
            "Total:".white(),
            result.tool_calls.len().to_string().green()
        );
        for tc in &result.tool_calls {
            let result_summary = summarize_tool_result(&tc.name, &tc.result);
            println!("    - {} {}", tc.name.yellow(), result_summary.dimmed());
        }
        println!();

        // Token usage
        println!("{}", "Token usage:".cyan().bold());
        println!(
            "  {} {} tokens",
            "Total:".white(),
            result.token_count.to_string().green().bold()
        );
        println!();

        // Key API Methods
        println!("{}", "Key API Methods:".yellow().bold());
        println!();
        println!(
            "  {} - Load skills from files/directories",
            "SkillRegistry::from_paths()".cyan()
        );
        println!(
            "  {} - Thread-safe registry wrapper",
            "Arc<SkillRegistry>".cyan()
        );
        println!(
            "  {} - Register custom tools on prompts",
            "PromptBuilder::with_tool()".cyan()
        );
        println!("  {} - List available skills (tool)", "list_skills".cyan());
        println!(
            "  {} - Load skill instructions (tool)",
            "activate_skill".cyan()
        );
        println!();

        // Shutdown
        ai_runtime.shutdown().await?;

        Ok(())
    }
}

#[cfg(feature = "agent-skills")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    skills_example::run().await
}

#[cfg(not(feature = "agent-skills"))]
fn main() {
    eprintln!("This example requires the 'agent-skills' feature.");
    eprintln!("Run with: cargo run --example agent_skills --features agent-skills");
    std::process::exit(1);
}
