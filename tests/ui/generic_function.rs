use acton_ai::prelude::*;

/// Echoes any serializable value.
#[tool]
async fn echo<T: serde::Serialize>(value: T) -> Result<T, ToolError> {
    Ok(value)
}

fn main() {}
