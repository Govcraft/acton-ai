//! Calculate mathematical expressions built-in tool.
//!
//! Evaluates mathematical expressions safely using fasteval.
//! Note: fasteval is a safe math expression parser, not arbitrary code execution.

use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use fasteval::ez_eval;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Calculate tool executor.
///
/// Evaluates mathematical expressions with optional variables.
/// Uses fasteval which only supports mathematical operations, not arbitrary code.
#[derive(Debug, Default, Clone)]
pub struct CalculateTool;

/// Arguments for the calculate tool.
#[derive(Debug, Deserialize)]
struct CalculateArgs {
    /// Mathematical expression to evaluate
    expression: String,
    /// Optional variable bindings (name -> value)
    #[serde(default)]
    variables: Option<BTreeMap<String, f64>>,
}

impl CalculateTool {
    /// Creates a new calculate tool.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        use crate::messages::ToolDefinition;

        ToolConfig::new(ToolDefinition {
            name: "calculate".to_string(),
            description: "Evaluate mathematical expressions. Supports arithmetic (+, -, *, /, ^, %), comparison, and built-in functions (sin, cos, tan, log, abs, min, max, floor, ceil, round, etc.).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'abs(-5)', 'sin(pi()/2)')"
                    },
                    "variables": {
                        "type": "object",
                        "description": "Optional variable bindings (e.g., {\"x\": 5, \"y\": 10})",
                        "additionalProperties": {
                            "type": "number"
                        }
                    }
                },
                "required": ["expression"]
            }),
        })
    }
}

impl ToolExecutorTrait for CalculateTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let args: CalculateArgs = serde_json::from_value(args)
                .map_err(|e| ToolError::validation_failed("calculate", format!("invalid arguments: {e}")))?;

            // Validate empty expression early
            if args.expression.is_empty() {
                return Err(ToolError::validation_failed(
                    "calculate",
                    "expression cannot be empty",
                ));
            }

            // Create namespace with user variables
            let user_vars = args.variables.unwrap_or_default();

            // Create a callback that provides user variables
            // fasteval has built-in functions (sin, cos, log, etc.) by default
            let mut namespace = |name: &str, _args: Vec<f64>| -> Option<f64> {
                user_vars.get(name).copied()
            };

            // Evaluate the expression
            let result = ez_eval(&args.expression, &mut namespace)
                .map_err(|e| ToolError::validation_failed("calculate", format!("failed to evaluate expression: {e}")))?;

            // Check for special values
            let (result_str, is_special) = if result.is_nan() {
                ("NaN".to_string(), true)
            } else if result.is_infinite() {
                if result.is_sign_positive() {
                    ("Infinity".to_string(), true)
                } else {
                    ("-Infinity".to_string(), true)
                }
            } else {
                // Format the result nicely
                let formatted = if result.fract() == 0.0 && result.abs() < 1e15 {
                    format!("{}", result as i64)
                } else {
                    format!("{result}")
                };
                (formatted, false)
            };

            Ok(json!({
                "result": result,
                "formatted": result_str,
                "expression": args.expression,
                "is_special": is_special
            }))
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: CalculateArgs = serde_json::from_value(args.clone())
            .map_err(|e| ToolError::validation_failed("calculate", format!("invalid arguments: {e}")))?;

        if args.expression.is_empty() {
            return Err(ToolError::validation_failed(
                "calculate",
                "expression cannot be empty",
            ));
        }

        // Validate that the expression doesn't contain obviously problematic patterns
        if args.expression.len() > 1000 {
            return Err(ToolError::validation_failed(
                "calculate",
                "expression is too long (max 1000 characters)",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn calculate_basic_arithmetic() {
        let tool = CalculateTool::new();

        // Addition
        let result = tool.execute(json!({"expression": "2 + 2"})).await.unwrap();
        assert_eq!(result["result"], 4.0);

        // Subtraction
        let result = tool.execute(json!({"expression": "10 - 3"})).await.unwrap();
        assert_eq!(result["result"], 7.0);

        // Multiplication
        let result = tool.execute(json!({"expression": "6 * 7"})).await.unwrap();
        assert_eq!(result["result"], 42.0);

        // Division
        let result = tool.execute(json!({"expression": "20 / 4"})).await.unwrap();
        assert_eq!(result["result"], 5.0);
    }

    #[tokio::test]
    async fn calculate_operator_precedence() {
        let tool = CalculateTool::new();

        let result = tool
            .execute(json!({"expression": "2 + 3 * 4"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 14.0);

        let result = tool
            .execute(json!({"expression": "(2 + 3) * 4"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 20.0);
    }

    #[tokio::test]
    async fn calculate_power() {
        let tool = CalculateTool::new();

        let result = tool
            .execute(json!({"expression": "2 ^ 10"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 1024.0);
    }

    #[tokio::test]
    async fn calculate_functions() {
        let tool = CalculateTool::new();

        // abs (built-in function)
        let result = tool
            .execute(json!({"expression": "abs(-5)"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 5.0);

        // floor (built-in function)
        let result = tool
            .execute(json!({"expression": "floor(3.7)"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 3.0);

        // ceil (built-in function)
        let result = tool
            .execute(json!({"expression": "ceil(3.2)"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 4.0);
    }

    #[tokio::test]
    async fn calculate_trigonometry() {
        let tool = CalculateTool::new();

        // sin(0) = 0
        let result = tool.execute(json!({"expression": "sin(0)"})).await.unwrap();
        assert!((result["result"].as_f64().unwrap() - 0.0).abs() < 1e-10);

        // cos(0) = 1
        let result = tool.execute(json!({"expression": "cos(0)"})).await.unwrap();
        assert!((result["result"].as_f64().unwrap() - 1.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn calculate_constants() {
        let tool = CalculateTool::new();

        // fasteval uses pi() and e() as functions
        let result = tool.execute(json!({"expression": "pi()"})).await.unwrap();
        assert!((result["result"].as_f64().unwrap() - std::f64::consts::PI).abs() < 1e-10);

        let result = tool.execute(json!({"expression": "e()"})).await.unwrap();
        assert!((result["result"].as_f64().unwrap() - std::f64::consts::E).abs() < 1e-10);
    }

    #[tokio::test]
    async fn calculate_with_variables() {
        let tool = CalculateTool::new();

        let result = tool
            .execute(json!({
                "expression": "x + y * 2",
                "variables": {"x": 5.0, "y": 10.0}
            }))
            .await
            .unwrap();
        assert_eq!(result["result"], 25.0);
    }

    #[tokio::test]
    async fn calculate_division_by_zero() {
        let tool = CalculateTool::new();

        let result = tool.execute(json!({"expression": "1 / 0"})).await.unwrap();
        assert!(result["is_special"].as_bool().unwrap());
        assert_eq!(result["formatted"], "Infinity");
    }

    #[tokio::test]
    async fn calculate_invalid_expression() {
        let tool = CalculateTool::new();

        // This should fail because it's syntactically invalid
        let result = tool.execute(json!({"expression": "("})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn calculate_empty_expression() {
        let tool = CalculateTool::new();

        let result = tool.execute(json!({"expression": ""})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn calculate_formats_integers() {
        let tool = CalculateTool::new();

        let result = tool
            .execute(json!({"expression": "100 + 200"}))
            .await
            .unwrap();
        assert_eq!(result["formatted"], "300");
    }

    #[test]
    fn config_has_correct_schema() {
        let config = CalculateTool::config();
        assert_eq!(config.definition.name, "calculate");
        assert!(config.definition.description.contains("mathematical"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["expression"].is_object());
        assert!(schema["properties"]["variables"].is_object());
    }
}
