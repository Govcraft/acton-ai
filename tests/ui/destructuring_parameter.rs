use acton_ai::prelude::*;

/// Adds a pair of numbers.
#[tool]
async fn add_pair((a, b): (i64, i64)) -> Result<i64, ToolError> {
    Ok(a + b)
}

fn main() {}
