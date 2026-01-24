//! Example: Multi-Agent Research Team
//!
//! A complete end-to-end example demonstrating:
//! - Multi-agent collaboration with role-based specialists
//! - Tool usage by specialist agents (building on `ollama_tools.rs`)
//! - Sequential task delegation and result collection
//! - LLM-powered task processing and synthesis
//!
//! ## Scenario
//!
//! A user asks a question. The coordinator orchestrates specialists:
//! - **Researcher**: Uses `search_web` tool to find information
//! - **Analyst**: Uses `calculate` tool for numerical analysis
//! - **Coordinator**: Synthesizes findings into a final answer
//!
//! # Configuration
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
//! # Usage
//!
//! ```bash
//! cargo run --example multi_agent
//! ```

use acton_ai::prelude::*;
use colored::Colorize;
use std::io::Write;

/// Mock web search function.
///
/// In production, this would call a real search API. For this example,
/// it returns canned responses for known queries.
fn search_web(query: &str) -> String {
    let q = query.to_lowercase();
    if q.contains("population") && q.contains("france") {
        "France has a population of approximately 67.75 million people (2024).".into()
    } else if q.contains("population") && q.contains("germany") {
        "Germany has a population of approximately 84.4 million people (2024).".into()
    } else if q.contains("population") && q.contains("japan") {
        "Japan has a population of approximately 123.3 million people (2024).".into()
    } else {
        format!("Search results for: {query}")
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "Multi-Agent Research Team".cyan().bold());
    println!("{}", "=========================".dimmed());
    eprintln!("{}", "Loading configuration from acton-ai.toml...".dimmed());
    eprintln!();

    // Launch ActonAI runtime from config file
    // Register the calculate builtin for the analyst agent
    let runtime = ActonAI::builder()
        .app_name("multi-agent-research")
        .from_config()?
        .with_builtin_tools(&["calculate"])
        .launch()
        .await?;

    // Define the search tool for the researcher agent
    let search_tool = ToolDefinition {
        name: "search_web".to_string(),
        description: "Search the web for factual information. Use this to find data like \
                      population statistics, facts, and current information."
            .to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find information"
                }
            },
            "required": ["query"]
        }),
    };

    // The user's question
    let user_question = "What is the population of France, and what's 15% of that number?";
    println!("{} {}\n", "User:".white().bold(), user_question);

    // === Step 1: Researcher Agent ===
    // The researcher uses the search_web tool to find factual information
    println!(
        "{} {}",
        "[Researcher]".blue().bold(),
        "Searching for population data...".dimmed()
    );
    let research_result = runtime
        .prompt("Find the current population of France using the search_web tool.")
        .system(
            "You are a research specialist. Your job is to find factual information. \
             ALWAYS use the search_web tool to look up data. \
             After getting the search results, summarize the key facts found.",
        )
        .with_tool_callback(
            search_tool,
            |args| async move {
                let query = args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("population France");
                let result = search_web(query);
                Ok(serde_json::json!({ "result": result }))
            },
            |result| {
                if let Ok(value) = result {
                    eprintln!(
                        "  {} {}",
                        "[search_web]".yellow().dimmed(),
                        value.to_string().yellow().dimmed()
                    );
                }
            },
        )
        .on_token(|token| {
            print!("{token}");
            std::io::stdout().flush().ok();
        })
        .collect()
        .await?;
    println!("\n");

    // === Step 2: Analyst Agent ===
    // The analyst uses the calculate tool to perform numerical analysis
    println!(
        "{} {}",
        "[Analyst]".magenta().bold(),
        "Calculating 15% of the population...".dimmed()
    );
    let analysis_prompt = format!(
        "Based on this research: '{}'\n\
         Calculate 15% of France's population using the calculate tool. \
         Use the actual number (e.g., 67750000 * 0.15).",
        research_result.text
    );

    let analysis_result = runtime
        .prompt(&analysis_prompt)
        .system(
            "You are a data analyst. Your job is to perform calculations and interpret numbers. \
             ALWAYS use the calculate tool for any math operations. \
             After getting the result, explain what the number means.",
        )
        .use_builtins() // Use the built-in calculate tool
        .on_token(|token| {
            print!("{token}");
            std::io::stdout().flush().ok();
        })
        .collect()
        .await?;
    println!("\n");

    // === Step 3: Coordinator Agent ===
    // The coordinator synthesizes the findings into a coherent answer
    println!(
        "{} {}",
        "[Coordinator]".green().bold(),
        "Synthesizing final answer...".dimmed()
    );
    let synthesis_prompt = format!(
        "Combine these specialist findings into a clear, concise answer:\n\n\
         Research findings: {}\n\n\
         Analysis findings: {}\n\n\
         Original question: {}",
        research_result.text, analysis_result.text, user_question
    );

    let final_answer = runtime
        .prompt(&synthesis_prompt)
        .system(
            "You are a coordinator who synthesizes information from specialists. \
             Combine the research and analysis into a single, clear answer. \
             Be concise and direct. Include both the population and the calculated percentage.",
        )
        .on_token(|token| {
            print!("{token}");
            std::io::stdout().flush().ok();
        })
        .collect()
        .await?;
    println!("\n");

    // === Summary ===
    println!("{}", "=========================".dimmed());
    println!("{}", "Token usage:".cyan().bold());
    println!(
        "  {}  {} tokens",
        "Research:".blue(),
        research_result.token_count.to_string().green()
    );
    println!(
        "  {}  {} tokens",
        "Analysis:".magenta(),
        analysis_result.token_count.to_string().green()
    );
    println!(
        "  {} {} tokens",
        "Synthesis:".green(),
        final_answer.token_count.to_string().green()
    );
    let total =
        research_result.token_count + analysis_result.token_count + final_answer.token_count;
    println!(
        "  {}     {} tokens",
        "Total:".cyan().bold(),
        total.to_string().green().bold()
    );

    Ok(())
}
