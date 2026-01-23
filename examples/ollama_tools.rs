//! Example: Tool Usage with Ollama
//!
//! This example demonstrates how to use tools (function calling) with the
//! ActonAI framework. It shows:
//!
//! 1. Defining tools with JSON schemas
//! 2. Implementing tool executors
//! 3. Registering tools with the ToolRegistry
//! 4. Handling tool calls from the LLM
//! 5. Returning tool results to continue the conversation
//!
//! # Configuration
//!
//! Set environment variables or create a `.env` file:
//!
//! ```bash
//! export OLLAMA_URL="http://localhost:11434/v1"
//! export OLLAMA_MODEL="qwen2.5:7b"
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example ollama_tools
//! ```

use acton_ai::prelude::*;
use acton_ai::tools::{RegisterTool, ToolConfig, ToolExecutionFuture, ToolExecutorTrait};
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

/// A simple calculator tool that evaluates basic math expressions.
#[derive(Debug)]
struct CalculatorTool;

impl ToolExecutorTrait for CalculatorTool {
    fn execute(&self, args: serde_json::Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let expression = args
                .get("expression")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ToolError::validation_failed("calculator", "missing 'expression' field")
                })?;

            // Simple evaluation (in production, use a proper expression parser)
            let result = evaluate_expression(expression)
                .map_err(|e| ToolError::execution_failed("calculator", e))?;

            Ok(serde_json::json!({
                "expression": expression,
                "result": result
            }))
        })
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
}

/// Simple expression evaluator for basic math.
fn evaluate_expression(expr: &str) -> Result<f64, String> {
    // Very basic evaluation - just handles single operations
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

    Err(format!("cannot evaluate: {}", expr))
}

/// A tool that returns the current time.
#[derive(Debug)]
struct TimeTool;

impl ToolExecutorTrait for TimeTool {
    fn execute(&self, _args: serde_json::Value) -> ToolExecutionFuture {
        Box::pin(async move {
            let now = chrono::Utc::now();
            Ok(serde_json::json!({
                "utc_time": now.to_rfc3339(),
                "unix_timestamp": now.timestamp()
            }))
        })
    }
}

/// Load environment variables from .env file if present
fn load_dotenv() {
    if let Ok(contents) = std::fs::read_to_string(".env") {
        for line in contents.lines() {
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"');
                if !key.is_empty() && !key.starts_with('#') {
                    std::env::set_var(key, value);
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    load_dotenv();

    // Launch the actor runtime
    let mut runtime = ActonApp::launch_async().await;

    // Spawn the kernel
    let kernel_config = KernelConfig::default().with_app_name("ollama-tools");
    let _kernel = Kernel::spawn_with_config(&mut runtime, kernel_config).await;

    // Get configuration from environment
    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());

    eprintln!("Connecting to Ollama at {ollama_url} with model {model}");

    // Configure for Ollama
    let provider_config = ProviderConfig::openai_compatible(&ollama_url, &model)
        .with_timeout(Duration::from_secs(120))
        .with_max_tokens(512);

    // Spawn the LLM provider
    let provider_handle = LLMProvider::spawn(&mut runtime, provider_config).await;

    // Spawn the tool registry
    let registry = ToolRegistry::spawn(&mut runtime).await;

    // Define and register the calculator tool
    let calculator_def = ToolDefinition {
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

    registry
        .send(RegisterTool {
            config: ToolConfig::new(calculator_def.clone()),
            executor: Arc::new(Box::new(CalculatorTool) as Box<dyn ToolExecutorTrait>),
        })
        .await;

    // Define and register the time tool
    let time_def = ToolDefinition {
        name: "get_current_time".to_string(),
        description: "Returns the current UTC time.".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    };

    registry
        .send(RegisterTool {
            config: ToolConfig::new(time_def.clone()),
            executor: Arc::new(Box::new(TimeTool) as Box<dyn ToolExecutorTrait>),
        })
        .await;

    eprintln!("Tools registered: calculator, get_current_time");

    // Create a response collector that handles both tokens and tool responses
    let stream_done = Arc::new(Notify::new());
    let stream_done_signal = stream_done.clone();

    let mut collector = runtime.new_actor::<ResponseCollector>();

    // Handle stream start
    collector.mutate_on::<LLMStreamStart>(|_actor, envelope| {
        eprintln!("[Stream started: {}]", envelope.message().correlation_id);
        Reply::ready()
    });

    // Handle streaming tokens
    collector.mutate_on::<LLMStreamToken>(|actor, envelope| {
        let token = &envelope.message().token;
        actor.model.buffer.push_str(token);
        print!("{token}");
        std::io::stdout().flush().ok();
        Reply::ready()
    });

    // Handle stream end - check if we need to execute tools
    collector.mutate_on::<LLMStreamEnd>(move |_actor, envelope| {
        let stop_reason = envelope.message().stop_reason;

        eprintln!("\n[Stream ended: {stop_reason:?}]");

        if stop_reason == StopReason::ToolUse {
            // The LLM wants to use tools - we need to execute them
            // In a real app, you'd extract tool calls from the stream accumulator
            // For this example, we'll show the concept
            eprintln!("[Tool use requested - in production, execute tools and continue]");
        }

        // For this example, signal completion regardless
        // In production, you'd loop until stop_reason is EndTurn
        stream_done_signal.notify_one();
        Reply::ready()
    });

    // Handle tool responses
    collector.mutate_on::<ToolResponse>(|_actor, envelope| {
        let response = envelope.message();
        match &response.result {
            Ok(result) => {
                eprintln!("[Tool {} returned: {}]", response.tool_call_id, result);
            }
            Err(err) => {
                eprintln!("[Tool {} error: {}]", response.tool_call_id, err);
            }
        }
        Reply::ready()
    });

    // Subscribe to events BEFORE starting
    collector.handle().subscribe::<LLMStreamStart>().await;
    collector.handle().subscribe::<LLMStreamToken>().await;
    collector.handle().subscribe::<LLMStreamEnd>().await;
    collector.handle().subscribe::<ToolResponse>().await;

    let _collector_handle = collector.start().await;

    // Build the request with tools
    let request = LLMRequest::builder()
        .system(
            "You are a helpful assistant with access to tools. \
             Use the calculator tool for any math operations. \
             Always show your work.",
        )
        .user("What is 42 multiplied by 17? Please use the calculator tool.")
        .tools(vec![calculator_def, time_def])
        .build();

    eprintln!("\nSending prompt with tools...\n");

    // Send the request
    provider_handle.send(request).await;

    // Wait for completion
    stream_done.notified().await;

    // Graceful shutdown
    runtime.shutdown_all().await?;

    Ok(())
}

/// Actor to collect streamed responses
#[acton_actor]
struct ResponseCollector {
    buffer: String,
}
