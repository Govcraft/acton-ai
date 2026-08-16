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
    /// Maximum response size read off the socket, in bytes
    max_response_size: usize,
    /// Maximum extracted text handed back to the model, in characters
    max_body_chars: usize,
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
            max_response_size: DEFAULT_MAX_RESPONSE_SIZE,
            max_body_chars: DEFAULT_MAX_BODY_CHARS,
        }
    }
}

/// How much of a response is read off the socket.
///
/// Generous, because the raw download is cheap and markup compresses hard
/// once it is reduced to text.
const DEFAULT_MAX_RESPONSE_SIZE: usize = 5 * 1024 * 1024;

/// How much extracted text reaches the model, in characters.
///
/// This is the figure that matters. A tool result is replayed in full on
/// every subsequent round of the turn, so an unbounded one does not cost a
/// context window once — it costs it repeatedly, and grows the request until
/// the provider refuses it or drops the connection mid-response. Roughly
/// 30k tokens, which is a large but survivable share of any modern window.
const DEFAULT_MAX_BODY_CHARS: usize = 120_000;

/// Reduces an HTML document to the text a model can actually read.
///
/// Drops `script` and `style` element *contents* outright — they are pure
/// noise that routinely outweighs the prose — then removes tags, decodes the
/// handful of entities that matter, and collapses the whitespace that markup
/// leaves behind.
///
/// This is deliberately a reducer, not a parser: it never has to round-trip
/// or preserve structure, only to stop a page of markup from consuming a
/// context window. Malformed input degrades to "slightly worse text".
fn html_to_text(html: &str) -> String {
    let mut text = String::with_capacity(html.len() / 4);
    let mut rest = html;

    while let Some(open) = rest.find('<') {
        text.push_str(&rest[..open]);
        let after = &rest[open..];

        // `2 < 3` is arithmetic, not an element. Only a name character can
        // open a tag; anything else makes the '<' ordinary text, and treating
        // it as a tag would swallow the prose up to the next '>'.
        if !after[1..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '/' || c == '!' || c == '?')
        {
            text.push('<');
            rest = &after[1..];
            continue;
        }

        // Elements whose content is markup for machines, not prose.
        let skipped = ["script", "style", "head", "svg"]
            .iter()
            .find(|tag| starts_with_tag(after, tag))
            .and_then(|tag| skip_element(after, tag));

        if let Some(remainder) = skipped {
            rest = remainder;
            continue;
        }

        match after.find('>') {
            Some(close) => {
                // A block-level tag is a word boundary; without this, text
                // either side of it runs together into one nonsense token.
                if is_block_tag(&after[..close]) {
                    text.push('\n');
                }
                rest = &after[close + 1..];
            }
            // An unterminated '<' is literal text, not a tag.
            None => {
                text.push_str(after);
                rest = "";
            }
        }
    }
    text.push_str(rest);

    collapse_whitespace(&decode_entities(&text))
}

/// Whether `input` opens the named element.
fn starts_with_tag(input: &str, tag: &str) -> bool {
    input
        .strip_prefix('<')
        .map(str::trim_start)
        .is_some_and(|rest| {
            rest.len() >= tag.len()
                && rest[..tag.len()].eq_ignore_ascii_case(tag)
                && rest[tag.len()..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_whitespace() || c == '>' || c == '/')
        })
}

/// Skips past the closing tag of the element `input` opens, if there is one.
fn skip_element<'a>(input: &'a str, tag: &str) -> Option<&'a str> {
    let closing = format!("</{tag}");
    let lowered = input.to_ascii_lowercase();
    let close_start = lowered.find(&closing)?;
    let close_end = input[close_start..].find('>')?;
    Some(&input[close_start + close_end + 1..])
}

/// Whether a tag body names an element that implies a line break.
fn is_block_tag(tag_body: &str) -> bool {
    let name = tag_body
        .trim_start_matches(['<', '/'])
        .split([' ', '\t', '\n', '/', '>'])
        .next()
        .unwrap_or_default();

    matches!(
        name.to_ascii_lowercase().as_str(),
        "p" | "br"
            | "div"
            | "li"
            | "tr"
            | "td"
            | "th"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "section"
            | "article"
            | "header"
            | "footer"
            | "nav"
            | "table"
            | "ul"
            | "ol"
            | "blockquote"
            | "pre"
    )
}

/// Decodes the named and numeric entities that appear in ordinary prose.
fn decode_entities(input: &str) -> String {
    if !input.contains('&') {
        return input.to_string();
    }

    let mut out = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let after = &rest[start..];

        // Entities are short; a distant ';' belongs to something else.
        let Some(end) = after[..after.len().min(12)].find(';') else {
            out.push('&');
            rest = &after[1..];
            continue;
        };

        let entity = &after[1..end];
        let decoded = match entity {
            "amp" => Some("&".to_string()),
            "lt" => Some("<".to_string()),
            "gt" => Some(">".to_string()),
            "quot" => Some("\"".to_string()),
            "apos" | "#39" => Some("'".to_string()),
            "nbsp" => Some(" ".to_string()),
            _ => entity
                .strip_prefix('#')
                .and_then(|digits| match digits.strip_prefix(['x', 'X']) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok(),
                    None => digits.parse().ok(),
                })
                .and_then(char::from_u32)
                .map(String::from),
        };

        match decoded {
            Some(text) => {
                out.push_str(&text);
                rest = &after[end + 1..];
            }
            None => {
                out.push('&');
                rest = &after[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Collapses runs of whitespace, keeping paragraph breaks.
fn collapse_whitespace(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut blank_run = 0_usize;

    for line in input.lines() {
        let trimmed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if trimmed.is_empty() {
            blank_run += 1;
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
            if blank_run > 0 {
                out.push('\n');
            }
        }
        blank_run = 0;
        out.push_str(&trimmed);
    }

    out
}

/// Truncates on a character boundary, reporting whether it had to.
fn truncate_chars(input: &str, max_chars: usize) -> (String, bool) {
    match input.char_indices().nth(max_chars) {
        Some((byte_index, _)) => (input[..byte_index].to_string(), true),
        None => (input.to_string(), false),
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
            max_body_chars: DEFAULT_MAX_BODY_CHARS,
        }
    }

    /// Sets how much extracted text is handed back to the model.
    ///
    /// Separate from the download cap because the two limits guard different
    /// things: the download cap bounds what crosses the network, this bounds
    /// what is replayed into the model's context on every later round.
    #[must_use]
    pub fn with_max_body_chars(mut self, max_body_chars: usize) -> Self {
        self.max_body_chars = max_body_chars;
        self
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
        let max_body_chars = self.max_body_chars;

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

            let (raw, size_truncated) = if bytes.len() > max_size {
                (
                    String::from_utf8_lossy(&bytes[..max_size]).to_string(),
                    true,
                )
            } else {
                (String::from_utf8_lossy(&bytes).to_string(), false)
            };

            // Markup is stripped before the cap is applied, so the budget is
            // spent on prose rather than on attributes and inline scripts.
            let is_html = content_type.to_ascii_lowercase().contains("html");
            let extracted = if is_html { html_to_text(&raw) } else { raw };

            let (body, length_truncated) = truncate_chars(&extracted, max_body_chars);
            let truncated = size_truncated || length_truncated;

            Ok(json!({
                "status_code": status_code,
                "success": status.is_success(),
                "content_type": content_type,
                "body": body,
                "body_length": bytes.len(),
                "extracted_as_text": is_html,
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
    fn html_to_text_keeps_prose_and_drops_markup() {
        let html = "<html><body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>";

        let text = html_to_text(html);

        assert!(text.contains("Title"));
        assert!(text.contains("Hello world."));
        assert!(!text.contains('<'));
    }

    #[test]
    fn html_to_text_drops_script_and_style_contents() {
        let html = "<html><head><style>.a{color:red}</style></head>\
                    <body><script>var x = 1 < 2;</script><p>Real content</p></body></html>";

        let text = html_to_text(html);

        assert!(text.contains("Real content"));
        assert!(!text.contains("color:red"));
        assert!(!text.contains("var x"));
    }

    #[test]
    fn html_to_text_separates_block_elements() {
        let text = html_to_text("<p>first</p><p>second</p>");

        // Without a boundary these would run together into "firstsecond".
        assert!(text.contains("first"));
        assert!(text.contains("second"));
        assert!(!text.contains("firstsecond"));
    }

    #[test]
    fn html_to_text_decodes_common_entities() {
        let text = html_to_text("<p>Tom &amp; Jerry &lt;3 &#39;quotes&#39; &nbsp;here</p>");

        assert!(text.contains("Tom & Jerry"));
        assert!(text.contains("<3"));
        assert!(text.contains("'quotes'"));
    }

    #[test]
    fn html_to_text_leaves_a_bare_angle_bracket_alone() {
        let text = html_to_text("<p>2 < 3 is true</p>");

        assert!(text.contains('3'), "unterminated tag ate the text: {text}");
    }

    #[test]
    fn html_to_text_collapses_markup_whitespace() {
        let text = html_to_text("<p>a</p>\n\n\n\n     \n<p>b</p>");

        assert!(!text.contains("\n\n\n"));
    }

    #[test]
    fn html_extraction_shrinks_a_page_dramatically() {
        // A page whose markup outweighs its prose, which is the ordinary case.
        let html = format!(
            "<html><head><style>{}</style></head><body><p>The answer is 42.</p></body></html>",
            ".cls { margin: 0 } ".repeat(500)
        );

        let text = html_to_text(&html);

        assert!(text.contains("The answer is 42."));
        assert!(
            text.len() < html.len() / 10,
            "extraction barely helped: {} -> {}",
            html.len(),
            text.len()
        );
    }

    #[test]
    fn truncate_chars_respects_character_boundaries() {
        let (text, truncated) = truncate_chars("héllo wörld", 4);

        assert!(truncated);
        assert_eq!(text.chars().count(), 4);
        assert_eq!(text, "héll");
    }

    #[test]
    fn truncate_chars_leaves_short_input_alone() {
        let (text, truncated) = truncate_chars("short", 100);

        assert!(!truncated);
        assert_eq!(text, "short");
    }

    #[test]
    fn default_body_cap_is_a_survivable_share_of_a_context_window() {
        let tool = WebFetchTool::default();

        // The failure this guards against is a tool result replayed on every
        // round until the request outgrows the provider's limit.
        assert!(tool.max_body_chars <= 200_000);
        assert!(tool.max_body_chars < tool.max_response_size);
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
