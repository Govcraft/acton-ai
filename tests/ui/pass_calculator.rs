//! The happy path, compiled through trybuild so ordinary expansion is pinned.

use acton_ai::prelude::*;

/// Computes the result of a math expression.
#[tool]
async fn calculator(expr: String, precision: Option<u8>) -> Result<serde_json::Value, ToolError> {
    let places = usize::from(precision.unwrap_or(2));
    Ok(serde_json::json!({ "expr": expr, "places": places }))
}

/// A tool that takes nothing.
#[tool]
async fn ping() -> Result<&'static str, ToolError> {
    Ok("pong")
}

fn main() {
    assert_eq!(Calculator.name(), "calculator");
    assert_eq!(Ping.name(), "ping");
    assert_eq!(Calculator.input_schema()["required"], serde_json::json!(["expr"]));
}
