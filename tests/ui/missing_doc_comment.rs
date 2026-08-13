use acton_ai::prelude::*;

#[tool]
async fn calculator(expr: String) -> Result<serde_json::Value, ToolError> {
    Ok(serde_json::json!({ "expr": expr }))
}

fn main() {}
