//! Example: Multi-Agent Research Team
//!
//! A complete end-to-end example demonstrating:
//! - Multi-agent collaboration with role-based specialists
//! - Built-in tool usage (web_fetch, calculate)
//! - Sequential task delegation and result collection
//! - LLM-powered task processing and synthesis
//!
//! ## Scenario
//!
//! A user asks a question. The coordinator orchestrates specialists:
//! - **Researcher**: Uses `web_fetch` tool to find information
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

/// Summarize a tool result for display.
fn summarize_tool_result(tool_name: &str, result: &Result<serde_json::Value, String>) -> String {
    match result {
        Ok(value) => match tool_name {
            "web_fetch" => {
                let status = value
                    .get("status_code")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let size = value
                    .get("body_length")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                format!("(HTTP {} - {} bytes)", status, size)
            }
            "calculate" => {
                let expr = value
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let result = value
                    .get("formatted")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                format!("({} = {})", expr, result)
            }
            _ => "(OK)".to_string(),
        },
        Err(e) => format!("(Error: {})", e),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "Multi-Agent Research Team".cyan().bold());
    println!("{}", "=========================".dimmed());
    eprintln!("{}", "Loading configuration from acton-ai.toml...".dimmed());
    eprintln!();

    // Launch ActonAI runtime from config file
    // Register web_fetch and calculate builtins for the agents
    let runtime = ActonAI::builder()
        .app_name("multi-agent-research")
        .from_config()?
        .with_builtin_tools(&["web_fetch", "calculate"])
        .launch()
        .await?;

    // The user's question
    let user_question = "What is the population of USA, and what's 15% of that number?";
    println!("{} {}\n", "User:".white().bold(), user_question);

    // === Step 1: Researcher Agent ===
    // The researcher uses the web_fetch tool to find factual information
    println!(
        "{} {}",
        "[Researcher]".blue().bold(),
        "Fetching population data from the web...".dimmed()
    );

    // Build the research prompt dynamically from the user's question
    let research_prompt = format!(
        "The user asked: \"{}\"\n\n\
         Find the population data needed to answer this question. \
         Use the web_fetch tool to fetch from https://restcountries.com/v3.1/name/{{country}} \
         (replace {{country}} with the appropriate country name from the question). \
         Extract the population from the JSON response.",
        user_question
    );

    let research_result = runtime
        .prompt(&research_prompt)
        .system(
            "You are a research specialist. Your job is to find factual information. \
             Use the web_fetch tool to retrieve data from URLs. \
             After getting the response, extract and summarize the key facts found.",
        )
        .use_builtins()
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
        "The user's original question: \"{}\"\n\n\
         Research findings: '{}'\n\n\
         Calculate 15% of the population mentioned in the research using the calculate tool. \
         Use the actual number from the research (e.g., population * 0.15).",
        user_question, research_result.text
    );

    let analysis_result = runtime
        .prompt(&analysis_prompt)
        .system(
            "You are a data analyst. Your job is to perform calculations and interpret numbers. \
             ALWAYS use the calculate tool for any math operations. \
             After getting the result, explain what the number means.",
        )
        .use_builtins()
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

    // Tool usage summary
    println!("{}", "Tool usage:".cyan().bold());
    println!(
        "  {} {} tool calls",
        "Research:".blue(),
        research_result.tool_calls.len().to_string().green()
    );
    for tc in &research_result.tool_calls {
        let result_summary = summarize_tool_result(&tc.name, &tc.result);
        println!("    - {} {}", tc.name.yellow(), result_summary.dimmed());
    }
    println!(
        "  {} {} tool calls",
        "Analysis:".magenta(),
        analysis_result.tool_calls.len().to_string().green()
    );
    for tc in &analysis_result.tool_calls {
        let result_summary = summarize_tool_result(&tc.name, &tc.result);
        println!("    - {} {}", tc.name.yellow(), result_summary.dimmed());
    }
    println!(
        "  {} {} tool calls",
        "Synthesis:".green(),
        final_answer.tool_calls.len().to_string().green()
    );
    println!();

    // Token usage summary
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
