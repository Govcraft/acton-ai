use acton_ai::prelude::*;

/// Computes the result of a math expression.
#[tool(name = "calc")]
async fn calculator(expr: String) -> Result<serde_json::Value, ToolError> {
    Ok(serde_json::json!({ "expr": expr }))
}

fn main() {}
