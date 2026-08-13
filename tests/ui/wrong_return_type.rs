use acton_ai::prelude::*;

/// Returns a plain string, not a Result.
#[tool]
async fn plain(expr: String) -> String {
    expr
}

/// Returns a Result with the wrong error type.
#[tool]
async fn foreign_error(expr: String) -> Result<String, std::io::Error> {
    Ok(expr)
}

/// Returns nothing at all.
#[tool]
async fn unit(expr: String) {
    let _ = expr;
}

fn main() {}
