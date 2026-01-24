//! Example: Tool Usage with the High-Level API
//!
//! This example demonstrates how to use tools (function calling) with the
//! ActonAI high-level API. It shows:
//!
//! 1. Setting up the runtime with `ActonAI::builder()`
//! 2. Defining tools inline with closures
//! 3. The automatic tool execution loop
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
//! cargo run --example ollama_tools
//! ```

use acton_ai::prelude::*;
use std::io::Write;

/// Simple expression evaluator for basic math.
fn evaluate_expression(expr: &str) -> Result<f64, String> {
    let expr = expr.trim();

    // Try to parse as a simple number first
    if let Ok(num) = expr.parse::<f64>() {
        return Ok(num);
    }

    // Handle basic operations
    for op in ['+', '-', '*', '/'] {
        if let Some(pos) = expr.rfind(op) {
            if pos > 0 {
                let left = evaluate_expression(&expr[..pos])?;
                let right = evaluate_expression(&expr[pos + 1..])?;
                return match op {
                    '+' => Ok(left + right),
                    '-' => Ok(left - right),
                    '*' => Ok(left * right),
                    '/' => {
                        if right == 0.0 {
                            Err("division by zero".to_string())
                        } else {
                            Ok(left / right)
                        }
                    }
                    _ => unreachable!(),
                };
            }
        }
    }

    Err(format!("cannot evaluate: {expr}"))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    eprintln!("Loading configuration from acton-ai.toml...");

    // Build the ActonAI runtime from config file
    let runtime = ActonAI::builder()
        .app_name("ollama-tools")
        .from_config()?
        .launch()
        .await?;

    // Define the calculator tool
    let calculator = ToolDefinition {
        name: "calculator".to_string(),
        description: "Evaluates mathematical expressions. Use this for any math calculations."
            .to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }),
    };

    // Define the time tool
    let get_time = ToolDefinition {
        name: "get_current_time".to_string(),
        description: "Returns the current UTC time.".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    };

    eprintln!("Tools available: calculator, get_current_time\n");

    // Send the prompt with tools - the high-level API handles:
    // - Tool registration with result callbacks
    // - Tool execution loop
    // - Returning results to the LLM
    // - Collecting the final response
    let response = runtime
        .prompt("What is 42 multiplied by 17? Please use the calculator tool.")
        .system(
            "You are a helpful assistant with access to tools. \
             Use the calculator tool for any math operations. \
             Always show your work.",
        )
        .with_tool_callback(
            calculator,
            |args| async move {
                let expression =
                    args.get("expression")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            ToolError::validation_failed("calculator", "missing 'expression'")
                        })?;

                let result = evaluate_expression(expression)
                    .map_err(|e| ToolError::execution_failed("calculator", e))?;

                Ok(serde_json::json!({
                    "expression": expression,
                    "result": result
                }))
            },
            |result| {
                // This callback is invoked when the calculator tool returns
                match result {
                    Ok(value) => eprintln!("\n[Calculator returned: {value}]"),
                    Err(e) => eprintln!("\n[Calculator error: {e}]"),
                }
            },
        )
        .with_tool_callback(
            get_time,
            |_args| async move {
                let now = chrono::Utc::now();
                Ok(serde_json::json!({
                    "utc_time": now.to_rfc3339(),
                    "unix_timestamp": now.timestamp()
                }))
            },
            |result| match result {
                Ok(value) => eprintln!("\n[Time returned: {value}]"),
                Err(e) => eprintln!("\n[Time error: {e}]"),
            },
        )
        .on_token(|token| {
            print!("{token}");
            std::io::stdout().flush().ok();
        })
        .collect()
        .await?;

    println!("\n");
    println!("Total tokens: {}", response.token_count);
    println!("Stop reason: {:?}", response.stop_reason);

    Ok(())
}
