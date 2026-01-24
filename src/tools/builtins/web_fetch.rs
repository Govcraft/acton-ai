//! Web fetch built-in tool.
//!
//! Fetches content from URLs.

use crate::messages::ToolDefinition;
use crate::tools::actor::{ExecuteToolDirect, ToolActor, ToolActorResponse};
use crate::tools::{ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait};
use acton_reactive::prelude::*;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;
use url::Url;

/// Web fetch tool executor.
///
/// Fetches content from URLs with configurable method and headers.
#[derive(Debug, Clone)]
pub struct WebFetchTool {
    /// HTTP client
    client: reqwest::Client,
    /// Maximum response size in bytes
    max_response_size: usize,
}

/// Web fetch tool actor state.
///
/// This actor wraps the `WebFetchTool` executor for per-agent tool spawning.
#[acton_actor]
pub struct WebFetchToolActor;

impl Default for WebFetchTool {
    fn default() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("acton-ai/0.1")
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            max_response_size: 5 * 1024 * 1024, // 5MB
        }
    }
}

/// Arguments for the web_fetch tool.
#[derive(Debug, Deserialize)]
struct WebFetchArgs {
    /// URL to fetch
    url: String,
    /// HTTP method (GET or POST)
    #[serde(default = "default_method")]
    method: String,
    /// Optional HTTP headers
    #[serde(default)]
    headers: Option<HashMap<String, String>>,
    /// Optional request body (for POST)
    #[serde(default)]
    body: Option<String>,
    /// Timeout in seconds (default: 30)
    #[serde(default)]
    timeout: Option<u64>,
}

fn default_method() -> String {
    "GET".to_string()
}

impl WebFetchTool {
    /// Creates a new web fetch tool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a web fetch tool with custom settings.
    #[must_use]
    pub fn with_config(timeout: Duration, max_response_size: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("acton-ai/0.1")
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            max_response_size,
        }
    }

    /// Returns the tool configuration for registration.
    #[must_use]
    pub fn config() -> ToolConfig {
        use crate::messages::ToolDefinition;

        ToolConfig::new(ToolDefinition {
            name: "web_fetch".to_string(),
            description:
                "Fetch content from a URL. Supports GET and POST methods with custom headers."
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch (must be http or https)"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "HTTP method (default: GET)"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers",
                        "additionalProperties": {
                            "type": "string"
                        }
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional request body (for POST requests)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30, max: 120)",
                        "minimum": 1,
                        "maximum": 120
                    }
                },
                "required": ["url"]
            }),
        })
    }

    /// Validates and normalizes the URL.
    fn validate_url(url: &str) -> Result<String, ToolError> {
        // Parse the URL
        let parsed = Url::parse(url)
            .map_err(|e| ToolError::validation_failed("web_fetch", format!("invalid URL: {e}")))?;

        // Only allow http and https
        match parsed.scheme() {
            "http" | "https" => {}
            scheme => {
                return Err(ToolError::validation_failed(
                    "web_fetch",
                    format!("unsupported URL scheme: {scheme}; only http and https are allowed"),
                ));
            }
        }

        // Block localhost and private IPs for security
        if let Some(host) = parsed.host_str() {
            let is_local = host == "localhost"
                || host == "127.0.0.1"
                || host == "::1"
                || host.starts_with("192.168.")
                || host.starts_with("10.")
                || host.starts_with("172.16.")
                || host.starts_with("172.17.")
                || host.starts_with("172.18.")
                || host.starts_with("172.19.")
                || host.starts_with("172.2")
                || host.starts_with("172.30.")
                || host.starts_with("172.31.");

            if is_local {
                return Err(ToolError::validation_failed(
                    "web_fetch",
                    "cannot fetch from localhost or private IP addresses",
                ));
            }
        }

        Ok(parsed.to_string())
    }
}

impl ToolExecutorTrait for WebFetchTool {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        let client = self.client.clone();
        let max_size = self.max_response_size;

        Box::pin(async move {
            let args: WebFetchArgs = serde_json::from_value(args).map_err(|e| {
                ToolError::validation_failed("web_fetch", format!("invalid arguments: {e}"))
            })?;

            // Validate empty URL early
            if args.url.is_empty() {
                return Err(ToolError::validation_failed(
                    "web_fetch",
                    "url cannot be empty",
                ));
            }

            // Validate URL
            let url = Self::validate_url(&args.url)?;

            // Build request
            let method = args.method.to_uppercase();
            let mut request = match method.as_str() {
                "GET" => client.get(&url),
                "POST" => client.post(&url),
                _ => {
                    return Err(ToolError::validation_failed(
                        "web_fetch",
                        format!("unsupported method: {method}; use GET or POST"),
                    ));
                }
            };

            // Add custom headers
            if let Some(headers) = args.headers {
                let mut header_map = HeaderMap::new();
                for (key, value) in headers {
                    let name = HeaderName::try_from(key.as_str()).map_err(|e| {
                        ToolError::validation_failed(
                            "web_fetch",
                            format!("invalid header name: {e}"),
                        )
                    })?;
                    let val = HeaderValue::try_from(value.as_str()).map_err(|e| {
                        ToolError::validation_failed(
                            "web_fetch",
                            format!("invalid header value: {e}"),
                        )
                    })?;
                    header_map.insert(name, val);
                }
                request = request.headers(header_map);
            }

            // Add body for POST
            if let Some(body) = args.body {
                if method == "POST" {
                    request = request.body(body);
                }
            }

            // Set timeout
            if let Some(timeout_secs) = args.timeout {
                let timeout = Duration::from_secs(timeout_secs.min(120));
                request = request.timeout(timeout);
            }

            // Execute request
            let response = request.send().await.map_err(|e| {
                if e.is_timeout() {
                    ToolError::timeout("web_fetch", Duration::from_secs(args.timeout.unwrap_or(30)))
                } else if e.is_connect() {
                    ToolError::execution_failed("web_fetch", format!("connection failed: {e}"))
                } else {
                    ToolError::execution_failed("web_fetch", format!("request failed: {e}"))
                }
            })?;

            let status = response.status();
            let status_code = status.as_u16();
            let headers: HashMap<String, String> = response
                .headers()
                .iter()
                .filter_map(|(k, v)| {
                    v.to_str()
                        .ok()
                        .map(|s| (k.as_str().to_string(), s.to_string()))
                })
                .collect();

            // Get content type
            let content_type = headers
                .get("content-type")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            // Read body with size limit
            let bytes = response.bytes().await.map_err(|e| {
                ToolError::execution_failed("web_fetch", format!("failed to read response: {e}"))
            })?;

            let (body, truncated) = if bytes.len() > max_size {
                (
                    String::from_utf8_lossy(&bytes[..max_size]).to_string(),
                    true,
                )
            } else {
                (String::from_utf8_lossy(&bytes).to_string(), false)
            };

            Ok(json!({
                "status_code": status_code,
                "success": status.is_success(),
                "content_type": content_type,
                "body": body,
                "body_length": bytes.len(),
                "truncated": truncated,
                "headers": headers
            }))
        })
    }

    fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        let args: WebFetchArgs = serde_json::from_value(args.clone()).map_err(|e| {
            ToolError::validation_failed("web_fetch", format!("invalid arguments: {e}"))
        })?;

        if args.url.is_empty() {
            return Err(ToolError::validation_failed(
                "web_fetch",
                "url cannot be empty",
            ));
        }

        // Validate URL format
        Self::validate_url(&args.url)?;

        Ok(())
    }
}

impl ToolActor for WebFetchToolActor {
    fn name() -> &'static str {
        "web_fetch"
    }

    fn definition() -> ToolDefinition {
        WebFetchTool::config().definition
    }

    async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut builder = runtime.new_actor_with_name::<Self>("web_fetch_tool".to_string());

        builder.act_on::<ExecuteToolDirect>(|actor, envelope| {
            let msg = envelope.message();
            let correlation_id = msg.correlation_id.clone();
            let tool_call_id = msg.tool_call_id.clone();
            let args = msg.args.clone();
            let broker = actor.broker().clone();

            Reply::pending(async move {
                let tool = WebFetchTool::new();
                let result = tool.execute(args).await;

                let response = match result {
                    Ok(value) => {
                        let result_str = serde_json::to_string(&value)
                            .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e));
                        ToolActorResponse::success(correlation_id, tool_call_id, result_str)
                    }
                    Err(e) => ToolActorResponse::error(correlation_id, tool_call_id, e.to_string()),
                };

                broker.broadcast(response).await;
            })
        });

        builder.start().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_url_accepts_https() {
        let result = WebFetchTool::validate_url("https://example.com/path");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_url_accepts_http() {
        let result = WebFetchTool::validate_url("http://example.com/path");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_url_rejects_ftp() {
        let result = WebFetchTool::validate_url("ftp://example.com/file");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unsupported"));
    }

    #[test]
    fn validate_url_rejects_localhost() {
        let result = WebFetchTool::validate_url("http://localhost/api");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("localhost"));
    }

    #[test]
    fn validate_url_rejects_private_ip() {
        let result = WebFetchTool::validate_url("http://192.168.1.1/api");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("private"));

        let result = WebFetchTool::validate_url("http://10.0.0.1/api");
        assert!(result.is_err());
    }

    #[test]
    fn validate_url_rejects_invalid() {
        let result = WebFetchTool::validate_url("not a url");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid"));
    }

    #[tokio::test]
    async fn web_fetch_empty_url_rejected() {
        let tool = WebFetchTool::new();
        let result = tool.execute(json!({"url": ""})).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn web_fetch_invalid_method_rejected() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({
                "url": "https://example.com",
                "method": "DELETE"
            }))
            .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unsupported method"));
    }

    #[test]
    fn config_has_correct_schema() {
        let config = WebFetchTool::config();
        assert_eq!(config.definition.name, "web_fetch");
        assert!(config.definition.description.contains("Fetch"));

        let schema = &config.definition.input_schema;
        assert!(schema["properties"]["url"].is_object());
        assert!(schema["properties"]["method"].is_object());
        assert!(schema["properties"]["headers"].is_object());
        assert!(schema["properties"]["body"].is_object());
        assert!(schema["properties"]["timeout"].is_object());
    }

    #[test]
    fn default_method_is_get() {
        assert_eq!(default_method(), "GET");
    }
}
