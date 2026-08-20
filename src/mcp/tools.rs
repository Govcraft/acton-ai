//! Turning MCP tools into acton-ai tools.
//!
//! The conversions here are pure functions over rmcp's wire types, kept
//! separate from the actor so every branch is unit testable without a running
//! connection. [`McpToolExecutor`] is the one impure piece: it resolves the
//! *current* incarnation of a supervised server actor and asks it to make the
//! call.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use acton_reactive::prelude::{ActorHandleInterface, SupervisedChild};
use rmcp::model::{CallToolResult, ContentBlock, Tool as RmcpTool};
use serde_json::{Map, Value};

use crate::mcp::actor::{CallMcpTool, McpCallOutcome};
use crate::mcp::connector::detached;
use crate::messages::ToolDefinition;
use crate::tools::{
    BoxedToolExecutor, ToolConfig, ToolError, ToolExecutionFuture, ToolExecutorTrait,
};

/// The longest tool name OpenAI's function-calling API accepts.
///
/// Anthropic allows more, so exceeding this is a warning rather than an error:
/// truncating would create collisions, and rejecting would make a server
/// unusable on providers that would have accepted it.
const OPENAI_TOOL_NAME_LIMIT: usize = 64;

/// Replaces every character outside `[A-Za-z0-9_-]` with `_`.
fn sanitize_segment(segment: &str) -> String {
    segment
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Builds the name an MCP tool is exposed to the LLM under.
///
/// Both segments are sanitized to `[A-Za-z0-9_-]` so a server or tool name
/// containing dots, slashes, or spaces still produces a name every provider
/// accepts.
///
/// ```rust
/// use acton_ai::mcp::prefixed_tool_name;
///
/// assert_eq!(prefixed_tool_name("fs", "read_file"), "mcp__fs__read_file");
/// assert_eq!(prefixed_tool_name("my.server", "a b"), "mcp__my_server__a_b");
/// ```
#[must_use]
pub fn prefixed_tool_name(server: &str, tool: &str) -> String {
    format!(
        "mcp__{}__{}",
        sanitize_segment(server),
        sanitize_segment(tool)
    )
}

/// Converts an rmcp tool into the definition sent to the LLM.
///
/// The description defaults to empty rather than to a placeholder: an empty
/// description is what the server actually said, and inventing text would put
/// words in its mouth.
#[must_use]
pub fn tool_definition(server: &str, tool: &RmcpTool) -> ToolDefinition {
    ToolDefinition {
        idempotent: false,
        name: prefixed_tool_name(server, &tool.name),
        description: tool
            .description
            .as_ref()
            .map(|d| d.to_string())
            .unwrap_or_default(),
        input_schema: Value::Object((*tool.input_schema).clone()),
    }
}

/// Joins the text of every text block, ignoring non-text blocks.
fn joined_text(content: &[ContentBlock]) -> Option<String> {
    let texts: Vec<&str> = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();

    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

/// Converts a `tools/call` result into the JSON value handed back to the LLM.
///
/// Branches, in order:
/// 1. `is_error == Some(true)` — a tool-level failure, surfaced as a
///    [`ToolError`] so the model sees it as a failed tool rather than as a
///    successful result whose content happens to describe a failure.
/// 2. `structured_content` present — used verbatim.
/// 3. every block is text — the texts joined with newlines, as a JSON string.
/// 4. otherwise — the content array serialized as-is, so images and embedded
///    resources reach the model rather than being dropped.
///
/// # Errors
///
/// [`ToolError::execution_failed`] for branch 1, or if branch 4's content
/// cannot be serialized.
pub fn call_result_to_value(
    display_name: &str,
    result: CallToolResult,
) -> Result<Value, ToolError> {
    if result.is_error == Some(true) {
        let message = joined_text(&result.content)
            .or_else(|| {
                result
                    .structured_content
                    .as_ref()
                    .map(std::string::ToString::to_string)
            })
            .unwrap_or_else(|| "the MCP server reported an error with no message".to_string());
        return Err(ToolError::execution_failed(display_name, message));
    }

    if let Some(structured) = result.structured_content {
        return Ok(structured);
    }

    let all_text = result
        .content
        .iter()
        .all(|block| matches!(block, ContentBlock::Text(_)));

    if all_text {
        return Ok(Value::String(
            joined_text(&result.content).unwrap_or_default(),
        ));
    }

    serde_json::to_value(&result.content).map_err(|e| {
        ToolError::execution_failed(
            display_name,
            format!("could not serialize tool content: {e}"),
        )
    })
}

/// Executes one MCP tool by asking the server's supervised actor.
///
/// Holds a [`SupervisedChild`], never an `ActorHandle`: a handle names one
/// incarnation and goes stale the moment the supervisor restarts the actor,
/// which is exactly the case this design exists to survive. The live handle is
/// resolved per call.
#[derive(Debug, Clone)]
pub struct McpToolExecutor {
    server_name: String,
    remote_name: String,
    display_name: String,
    child: SupervisedChild,
    timeout: Duration,
}

impl McpToolExecutor {
    /// Creates an executor for one remote tool on one server.
    #[must_use]
    pub fn new(
        server_name: impl Into<String>,
        remote_name: impl Into<String>,
        child: SupervisedChild,
        timeout: Duration,
    ) -> Self {
        let server_name = server_name.into();
        let remote_name = remote_name.into();
        let display_name = prefixed_tool_name(&server_name, &remote_name);
        Self {
            server_name,
            remote_name,
            display_name,
            child,
            timeout,
        }
    }

    /// The name this tool is exposed to the LLM under.
    #[must_use]
    pub fn display_name(&self) -> &str {
        &self.display_name
    }
}

/// Coerces tool arguments into the JSON object MCP requires.
///
/// `null` and absent arguments become an empty object, which is what a
/// zero-parameter tool call looks like coming from most providers.
fn arguments_object(display_name: &str, args: Value) -> Result<Map<String, Value>, ToolError> {
    match args {
        Value::Object(map) => Ok(map),
        Value::Null => Ok(Map::new()),
        other => Err(ToolError::validation_failed(
            display_name,
            format!(
                "MCP tool arguments must be a JSON object, got {}",
                match other {
                    Value::Array(_) => "an array",
                    Value::String(_) => "a string",
                    Value::Bool(_) => "a boolean",
                    _ => "a number",
                }
            ),
        )),
    }
}

impl ToolExecutorTrait for McpToolExecutor {
    fn execute(&self, args: Value) -> ToolExecutionFuture {
        let display_name = self.display_name.clone();
        let server_name = self.server_name.clone();
        let remote_name = self.remote_name.clone();
        // Resolved now, not cached at construction: after a restart the
        // previous incarnation's handle is dead and silently swallows sends.
        let handle = self.child.current();
        let timeout = self.timeout;

        Box::pin(async move {
            let arguments = arguments_object(&display_name, args)?;
            let Some(handle) = handle else {
                return Err(ToolError::execution_failed(
                    &display_name,
                    format!("MCP server '{server_name}' is not running"),
                ));
            };

            let request = CallMcpTool {
                remote_name,
                arguments,
            };

            // The actor applies the real per-call deadline; this outer bound
            // exists so a wedged mailbox cannot hang the tool round forever.
            let ask_timeout = timeout.saturating_add(Duration::from_secs(5));
            let reply =
                detached(async move { handle.ask_with_timeout(request, ask_timeout).await }).await;

            match reply {
                Some(Ok(McpCallOutcome { outcome })) => outcome,
                Some(Err(err)) => Err(ToolError::execution_failed(
                    &display_name,
                    format!("MCP server '{server_name}' did not answer: {err}"),
                )),
                None => Err(ToolError::execution_failed(
                    &display_name,
                    format!("MCP server '{server_name}' call task was cancelled"),
                )),
            }
        })
    }

    fn requires_sandbox(&self) -> bool {
        false
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// The MCP tools available to every prompt, mirroring
/// [`BuiltinTools`](crate::tools::builtins::BuiltinTools).
///
/// Also owns the per-server [`SupervisedChild`] handles, which is what keeps
/// the executors pointed at whichever incarnation is currently alive.
#[derive(Debug, Default)]
pub struct McpTools {
    configs: HashMap<String, ToolConfig>,
    executors: HashMap<String, Arc<BoxedToolExecutor>>,
    servers: HashMap<String, SupervisedChild>,
}

impl McpTools {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the supervised actor owning one server's connection.
    pub fn add_server(&mut self, server_name: impl Into<String>, child: SupervisedChild) {
        self.servers.insert(server_name.into(), child);
    }

    /// Registers one discovered tool.
    ///
    /// Logs a warning when the prefixed name exceeds OpenAI's 64-character
    /// limit; the tool is still registered, because Anthropic accepts it.
    pub fn add_tool(&mut self, definition: ToolDefinition, executor: McpToolExecutor) {
        let name = definition.name.clone();
        if name.len() > OPENAI_TOOL_NAME_LIMIT {
            tracing::warn!(
                tool = %name,
                length = name.len(),
                limit = OPENAI_TOOL_NAME_LIMIT,
                "MCP tool name exceeds the OpenAI function-name limit; OpenAI-compatible providers will reject it"
            );
        }

        let config = ToolConfig::new(definition)
            .with_sandbox(false)
            .with_timeout(executor.timeout());

        self.configs.insert(name.clone(), config);
        self.executors
            .insert(name, Arc::new(Box::new(executor) as BoxedToolExecutor));
    }

    /// Returns the configuration for a registered tool.
    #[must_use]
    pub fn get_config(&self, name: &str) -> Option<&ToolConfig> {
        self.configs.get(name)
    }

    /// Returns the executor for a registered tool.
    #[must_use]
    pub fn get_executor(&self, name: &str) -> Option<Arc<BoxedToolExecutor>> {
        self.executors.get(name).cloned()
    }

    /// Iterates every registered tool configuration.
    pub fn configs(&self) -> impl Iterator<Item = (&String, &ToolConfig)> {
        self.configs.iter()
    }

    /// Iterates every registered executor.
    pub fn executors(&self) -> impl Iterator<Item = (&String, &Arc<BoxedToolExecutor>)> {
        self.executors.iter()
    }

    /// Returns the supervised actor for a server by its configured name.
    #[must_use]
    pub fn server(&self, server_name: &str) -> Option<&SupervisedChild> {
        self.servers.get(server_name)
    }

    /// Iterates the configured servers.
    pub fn servers(&self) -> impl Iterator<Item = (&String, &SupervisedChild)> {
        self.servers.iter()
    }

    /// The number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Whether no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::TextContent;
    use serde_json::json;
    use std::borrow::Cow;

    fn rmcp_tool(name: &'static str, description: Option<&'static str>) -> RmcpTool {
        let schema: Map<String, Value> = json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        })
        .as_object()
        .cloned()
        .unwrap();

        RmcpTool::new_with_raw(
            Cow::Borrowed(name),
            description.map(Cow::Borrowed),
            Arc::new(schema),
        )
    }

    fn text_block(text: &str) -> ContentBlock {
        ContentBlock::Text(TextContent::new(text))
    }

    #[test]
    fn prefixes_server_and_tool() {
        assert_eq!(
            prefixed_tool_name("filesystem", "read_file"),
            "mcp__filesystem__read_file"
        );
    }

    #[test]
    fn sanitizes_illegal_characters_in_both_segments() {
        assert_eq!(
            prefixed_tool_name("my.server/one", "do this!"),
            "mcp__my_server_one__do_this_"
        );
    }

    #[test]
    fn keeps_hyphens_and_underscores() {
        assert_eq!(
            prefixed_tool_name("my-server", "get_thing-2"),
            "mcp__my-server__get_thing-2"
        );
    }

    #[test]
    fn converts_tool_to_definition() {
        let definition = tool_definition("fs", &rmcp_tool("read_file", Some("Reads a file")));

        assert_eq!(definition.name, "mcp__fs__read_file");
        assert_eq!(definition.description, "Reads a file");
        assert_eq!(definition.input_schema["type"], json!("object"));
        assert_eq!(definition.input_schema["required"], json!(["path"]));
    }

    #[test]
    fn missing_description_becomes_empty() {
        let definition = tool_definition("fs", &rmcp_tool("read_file", None));
        assert_eq!(definition.description, "");
    }

    #[test]
    fn is_error_becomes_a_tool_error_carrying_the_text() {
        let result = CallToolResult::error(vec![text_block("file not found")]);
        let err = call_result_to_value("mcp__fs__read", result).unwrap_err();
        assert!(err.to_string().contains("file not found"));
        assert!(err.to_string().contains("mcp__fs__read"));
    }

    #[test]
    fn is_error_without_text_still_produces_a_message() {
        let mut result = CallToolResult::success(vec![]);
        result.is_error = Some(true);
        let err = call_result_to_value("mcp__fs__read", result).unwrap_err();
        assert!(err.to_string().contains("no message"));
    }

    #[test]
    fn structured_content_wins_over_text() {
        let mut result = CallToolResult::success(vec![text_block("ignored")]);
        result.structured_content = Some(json!({"count": 3}));

        let value = call_result_to_value("mcp__fs__read", result).unwrap();
        assert_eq!(value, json!({"count": 3}));
    }

    #[test]
    fn a_single_text_block_becomes_a_json_string() {
        let result = CallToolResult::success(vec![text_block("hello")]);
        let value = call_result_to_value("mcp__fs__read", result).unwrap();
        assert_eq!(value, json!("hello"));
    }

    #[test]
    fn several_text_blocks_join_with_newlines() {
        let result = CallToolResult::success(vec![text_block("one"), text_block("two")]);
        let value = call_result_to_value("mcp__fs__read", result).unwrap();
        assert_eq!(value, json!("one\ntwo"));
    }

    #[test]
    fn empty_content_becomes_an_empty_string() {
        let result = CallToolResult::success(vec![]);
        let value = call_result_to_value("mcp__fs__read", result).unwrap();
        assert_eq!(value, json!(""));
    }

    #[test]
    fn mixed_content_is_serialized_as_the_content_array() {
        let result = CallToolResult::success(vec![
            text_block("caption"),
            ContentBlock::image("aGk=", "image/png"),
        ]);

        let value = call_result_to_value("mcp__fs__read", result).unwrap();
        let array = value.as_array().expect("mixed content serializes to array");
        assert_eq!(array.len(), 2);
        assert_eq!(array[0]["type"], json!("text"));
        assert_eq!(array[1]["type"], json!("image"));
    }

    #[test]
    fn object_arguments_pass_through() {
        let map = arguments_object("mcp__fs__read", json!({"path": "/tmp"})).unwrap();
        assert_eq!(map["path"], json!("/tmp"));
    }

    #[test]
    fn null_arguments_become_an_empty_object() {
        let map = arguments_object("mcp__fs__read", Value::Null).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn non_object_arguments_are_rejected() {
        let err = arguments_object("mcp__fs__read", json!([1, 2, 3])).unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"));
        assert!(err.to_string().contains("an array"));
    }

    #[test]
    fn empty_registry_reports_empty() {
        let tools = McpTools::new();
        assert!(tools.is_empty());
        assert_eq!(tools.len(), 0);
        assert!(tools.get_config("anything").is_none());
    }
}
