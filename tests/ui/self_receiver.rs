use acton_ai::prelude::*;

struct Calculators;

impl Calculators {
    /// Computes the result of a math expression.
    #[tool]
    async fn calculator(&self, expr: String) -> Result<serde_json::Value, ToolError> {
        Ok(serde_json::json!({ "expr": expr }))
    }
}

fn main() {}
