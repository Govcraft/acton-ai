//! Example: Deriving Tools with `#[tool]`
//!
//! This example demonstrates the `#[tool]` attribute macro, which turns an
//! ordinary `async fn` into a tool the model can call. It shows:
//!
//! 1. A tool with a required and an optional parameter
//! 2. A tool with a struct parameter, whose schema is inlined automatically
//! 3. A zero-parameter tool
//! 4. Registering them on a prompt with `.add_tool(...)`
//!
//! Compare with `examples/ollama_tools.rs`, which registers the same kind of
//! tool by hand: there, the name, the description, the JSON Schema, and the
//! argument-plucking closure are four separate things that all restate the
//! function signature. Here they are derived from it.
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
//! cargo run --example tool_macro
//! ```

use acton_ai::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::io::Write;

// =============================================================================
// Tools
// =============================================================================

/// Evaluates a basic arithmetic expression.
///
/// Supports `+`, `-`, `*`, and `/` over decimal numbers, for example
/// `42 * 17`. Set `precision` to control how many decimal places are
/// returned; it defaults to 2.
#[tool]
async fn calculator(expression: String, precision: Option<u8>) -> Result<String, ToolError> {
    let value = evaluate(&expression)
        .map_err(|reason| ToolError::execution_failed("calculator", reason))?;
    let places = usize::from(precision.unwrap_or(2));
    Ok(format!("{value:.places$}"))
}

/// A city and the units to report its weather in.
///
/// A struct parameter like this is inlined into the tool's schema — the model
/// sees `{"city": …, "units": …}` nested under `location`, with no `$ref` to
/// resolve.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Location {
    /// The city name, for example "Austin".
    city: String,
    /// Either "celsius" or "fahrenheit".
    units: String,
}

/// Reports the current weather for a location.
///
/// This is a stub that always reports the same conditions — the point is the
/// shape of the call, not the forecast.
#[tool]
async fn get_weather(location: Location) -> Result<serde_json::Value, ToolError> {
    let degrees = match location.units.as_str() {
        "fahrenheit" => 72,
        "celsius" => 22,
        other => {
            return Err(ToolError::validation_failed(
                "get_weather",
                format!("units must be \"celsius\" or \"fahrenheit\", got {other:?}"),
            ))
        }
    };

    Ok(serde_json::json!({
        "city": location.city,
        "temperature": degrees,
        "units": location.units,
        "conditions": "clear",
    }))
}

/// Lists the tools this assistant can call.
///
/// Takes no arguments, which is fine: the generated schema is an empty object.
#[tool]
async fn list_capabilities() -> Result<Vec<String>, ToolError> {
    Ok(vec![
        "calculator".to_string(),
        "get_weather".to_string(),
        "list_capabilities".to_string(),
    ])
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    let runtime = ActonAI::builder().app_name("tool-macro-example").launch().await?;

    // Each `#[tool]` function generated a PascalCase unit struct alongside it:
    // `calculator` -> `Calculator`, `get_weather` -> `GetWeather`.
    println!("Registered tools:");
    for tool in [
        &Calculator as &dyn Tool,
        &GetWeather as &dyn Tool,
        &ListCapabilities as &dyn Tool,
    ] {
        println!("  {} — {}", tool.name(), first_line(tool.description()));
    }

    println!("\nSchema derived for `calculator`:");
    println!(
        "{}",
        serde_json::to_string_pretty(&Calculator.input_schema())
            .expect("a generated schema must serialize")
    );

    println!("\n--- Asking the model ---\n");

    let response = runtime
        .prompt("What is 42 * 17, and what's the weather in Austin in fahrenheit?")
        .system("You are a helpful assistant. Use the tools you have.")
        .add_tool(Calculator)
        .add_tool(GetWeather)
        .add_tool(ListCapabilities)
        .on_token(|token| {
            print!("{token}");
            let _ = std::io::stdout().flush();
        })
        .collect()
        .await?;

    println!("\n\n--- Done ({} tokens) ---", response.token_count);

    // The annotated functions are emitted unchanged, so they stay ordinary
    // functions you can call and test without a model in the loop.
    let direct = calculator("2 + 2".to_string(), Some(0))
        .await
        .expect("2 + 2 is computable");
    println!("Called directly, no model involved: 2 + 2 = {direct}");

    Ok(())
}

// =============================================================================
// Helpers
// =============================================================================

/// The first line of a description, for one-line display.
fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or(text)
}

/// Evaluates a basic arithmetic expression, left-associatively.
///
/// Deliberately small: this example is about the macro, not about parsing.
fn evaluate(expr: &str) -> Result<f64, String> {
    let expr = expr.trim();

    if let Ok(number) = expr.parse::<f64>() {
        return Ok(number);
    }

    for op in ['+', '-', '*', '/'] {
        if let Some(position) = expr.rfind(op) {
            if position == 0 {
                continue;
            }
            let left = evaluate(&expr[..position])?;
            let right = evaluate(&expr[position + 1..])?;
            return match op {
                '+' => Ok(left + right),
                '-' => Ok(left - right),
                '*' => Ok(left * right),
                '/' if right == 0.0 => Err("division by zero".to_string()),
                '/' => Ok(left / right),
                _ => unreachable!("the loop only yields the four operators"),
            };
        }
    }

    Err(format!("could not evaluate {expr:?}"))
}
