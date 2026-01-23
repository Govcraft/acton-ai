//! Streaming response handling.
//!
//! Types for accumulating streaming tokens and managing active streams.

use crate::messages::{StopReason, ToolCall};
use crate::types::CorrelationId;
use std::collections::HashMap;

/// An active streaming response being accumulated.
#[derive(Debug, Clone)]
pub struct ActiveStream {
    /// Correlation ID for this stream
    pub correlation_id: CorrelationId,
    /// Accumulated text content
    pub content: String,
    /// Tool calls collected during streaming
    pub tool_calls: Vec<ToolCall>,
    /// Whether the stream has started
    pub started: bool,
    /// The stop reason when stream ends
    pub stop_reason: Option<StopReason>,
}

impl ActiveStream {
    /// Creates a new active stream for the given correlation ID.
    #[must_use]
    pub fn new(correlation_id: CorrelationId) -> Self {
        Self {
            correlation_id,
            content: String::new(),
            tool_calls: Vec::new(),
            started: false,
            stop_reason: None,
        }
    }

    /// Marks the stream as started.
    pub fn mark_started(&mut self) {
        self.started = true;
    }

    /// Appends a token to the accumulated content.
    pub fn append_token(&mut self, token: &str) {
        self.content.push_str(token);
    }

    /// Adds a tool call to the stream.
    pub fn add_tool_call(&mut self, tool_call: ToolCall) {
        self.tool_calls.push(tool_call);
    }

    /// Marks the stream as ended with the given stop reason.
    pub fn mark_ended(&mut self, stop_reason: StopReason) {
        self.stop_reason = Some(stop_reason);
    }

    /// Returns true if the stream has ended.
    #[must_use]
    pub fn is_ended(&self) -> bool {
        self.stop_reason.is_some()
    }

    /// Returns true if this stream has tool calls.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Returns the accumulated content length.
    #[must_use]
    pub fn content_length(&self) -> usize {
        self.content.len()
    }
}

/// Accumulator for managing multiple active streams.
#[derive(Debug, Clone, Default)]
pub struct StreamAccumulator {
    /// Active streams by correlation ID
    streams: HashMap<String, ActiveStream>,
}

impl StreamAccumulator {
    /// Creates a new stream accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a new stream for the given correlation ID.
    pub fn start_stream(&mut self, correlation_id: &CorrelationId) {
        let key = correlation_id.to_string();
        let mut stream = ActiveStream::new(correlation_id.clone());
        stream.mark_started();
        self.streams.insert(key, stream);
    }

    /// Gets a mutable reference to an active stream.
    pub fn get_stream_mut(&mut self, correlation_id: &CorrelationId) -> Option<&mut ActiveStream> {
        let key = correlation_id.to_string();
        self.streams.get_mut(&key)
    }

    /// Gets an immutable reference to an active stream.
    #[must_use]
    pub fn get_stream(&self, correlation_id: &CorrelationId) -> Option<&ActiveStream> {
        let key = correlation_id.to_string();
        self.streams.get(&key)
    }

    /// Appends a token to the specified stream.
    pub fn append_token(&mut self, correlation_id: &CorrelationId, token: &str) {
        if let Some(stream) = self.get_stream_mut(correlation_id) {
            stream.append_token(token);
        }
    }

    /// Adds a tool call to the specified stream.
    pub fn add_tool_call(&mut self, correlation_id: &CorrelationId, tool_call: ToolCall) {
        if let Some(stream) = self.get_stream_mut(correlation_id) {
            stream.add_tool_call(tool_call);
        }
    }

    /// Ends a stream and returns it.
    pub fn end_stream(
        &mut self,
        correlation_id: &CorrelationId,
        stop_reason: StopReason,
    ) -> Option<ActiveStream> {
        let key = correlation_id.to_string();
        if let Some(mut stream) = self.streams.remove(&key) {
            stream.mark_ended(stop_reason);
            Some(stream)
        } else {
            None
        }
    }

    /// Removes a stream without marking it as ended (e.g., on error).
    pub fn remove_stream(&mut self, correlation_id: &CorrelationId) -> Option<ActiveStream> {
        let key = correlation_id.to_string();
        self.streams.remove(&key)
    }

    /// Returns the number of active streams.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.streams.len()
    }

    /// Returns true if there are no active streams.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Clears all active streams.
    pub fn clear(&mut self) {
        self.streams.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_stream_new() {
        let corr_id = CorrelationId::new();
        let stream = ActiveStream::new(corr_id.clone());

        assert_eq!(stream.correlation_id, corr_id);
        assert!(stream.content.is_empty());
        assert!(stream.tool_calls.is_empty());
        assert!(!stream.started);
        assert!(stream.stop_reason.is_none());
    }

    #[test]
    fn active_stream_append_token() {
        let corr_id = CorrelationId::new();
        let mut stream = ActiveStream::new(corr_id);

        stream.append_token("Hello");
        stream.append_token(" ");
        stream.append_token("World");

        assert_eq!(stream.content, "Hello World");
        assert_eq!(stream.content_length(), 11);
    }

    #[test]
    fn active_stream_add_tool_call() {
        let corr_id = CorrelationId::new();
        let mut stream = ActiveStream::new(corr_id);

        let tool_call = ToolCall {
            id: "tc_123".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "rust"}),
        };

        stream.add_tool_call(tool_call);

        assert!(stream.has_tool_calls());
        assert_eq!(stream.tool_calls.len(), 1);
    }

    #[test]
    fn active_stream_lifecycle() {
        let corr_id = CorrelationId::new();
        let mut stream = ActiveStream::new(corr_id);

        assert!(!stream.started);
        assert!(!stream.is_ended());

        stream.mark_started();
        assert!(stream.started);
        assert!(!stream.is_ended());

        stream.mark_ended(StopReason::EndTurn);
        assert!(stream.is_ended());
        assert_eq!(stream.stop_reason, Some(StopReason::EndTurn));
    }

    #[test]
    fn stream_accumulator_start_stream() {
        let mut accumulator = StreamAccumulator::new();
        let corr_id = CorrelationId::new();

        assert!(accumulator.is_empty());

        accumulator.start_stream(&corr_id);

        assert_eq!(accumulator.active_count(), 1);
        assert!(!accumulator.is_empty());

        let stream = accumulator.get_stream(&corr_id).unwrap();
        assert!(stream.started);
    }

    #[test]
    fn stream_accumulator_append_token() {
        let mut accumulator = StreamAccumulator::new();
        let corr_id = CorrelationId::new();

        accumulator.start_stream(&corr_id);
        accumulator.append_token(&corr_id, "Hello");
        accumulator.append_token(&corr_id, " World");

        let stream = accumulator.get_stream(&corr_id).unwrap();
        assert_eq!(stream.content, "Hello World");
    }

    #[test]
    fn stream_accumulator_end_stream() {
        let mut accumulator = StreamAccumulator::new();
        let corr_id = CorrelationId::new();

        accumulator.start_stream(&corr_id);
        accumulator.append_token(&corr_id, "Test content");

        let stream = accumulator
            .end_stream(&corr_id, StopReason::EndTurn)
            .unwrap();

        assert_eq!(stream.content, "Test content");
        assert!(stream.is_ended());
        assert!(accumulator.is_empty());
    }

    #[test]
    fn stream_accumulator_remove_stream() {
        let mut accumulator = StreamAccumulator::new();
        let corr_id = CorrelationId::new();

        accumulator.start_stream(&corr_id);

        let stream = accumulator.remove_stream(&corr_id);

        assert!(stream.is_some());
        assert!(!stream.unwrap().is_ended());
        assert!(accumulator.is_empty());
    }

    #[test]
    fn stream_accumulator_multiple_streams() {
        let mut accumulator = StreamAccumulator::new();
        let corr_id1 = CorrelationId::new();
        let corr_id2 = CorrelationId::new();

        accumulator.start_stream(&corr_id1);
        accumulator.start_stream(&corr_id2);

        accumulator.append_token(&corr_id1, "Stream 1");
        accumulator.append_token(&corr_id2, "Stream 2");

        assert_eq!(accumulator.active_count(), 2);

        let stream1 = accumulator.get_stream(&corr_id1).unwrap();
        let stream2 = accumulator.get_stream(&corr_id2).unwrap();

        assert_eq!(stream1.content, "Stream 1");
        assert_eq!(stream2.content, "Stream 2");
    }

    #[test]
    fn stream_accumulator_clear() {
        let mut accumulator = StreamAccumulator::new();

        accumulator.start_stream(&CorrelationId::new());
        accumulator.start_stream(&CorrelationId::new());

        assert_eq!(accumulator.active_count(), 2);

        accumulator.clear();

        assert!(accumulator.is_empty());
    }

    #[test]
    fn stream_accumulator_add_tool_call() {
        let mut accumulator = StreamAccumulator::new();
        let corr_id = CorrelationId::new();

        accumulator.start_stream(&corr_id);

        let tool_call = ToolCall {
            id: "tc_456".to_string(),
            name: "calculator".to_string(),
            arguments: serde_json::json!({"expression": "2+2"}),
        };

        accumulator.add_tool_call(&corr_id, tool_call);

        let stream = accumulator.get_stream(&corr_id).unwrap();
        assert!(stream.has_tool_calls());
        assert_eq!(stream.tool_calls[0].name, "calculator");
    }

    // Import ToolCall for tests
    use crate::messages::ToolCall;
}
