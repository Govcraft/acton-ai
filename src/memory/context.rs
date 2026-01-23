//! Context window management for LLM interactions.
//!
//! This module provides utilities for managing the limited context window
//! of LLM models, including:
//! - Token estimation
//! - Message truncation strategies
//! - Memory injection into context
//!
//! ## Example
//!
//! ```rust,ignore
//! use acton_ai::memory::{ContextWindow, ContextWindowConfig, TruncationStrategy};
//! use acton_ai::messages::Message;
//!
//! let config = ContextWindowConfig {
//!     max_tokens: 4096,
//!     truncation_strategy: TruncationStrategy::KeepSystemAndRecent,
//!     ..Default::default()
//! };
//!
//! let window = ContextWindow::new(config);
//! let messages = vec![
//!     Message::system("You are helpful."),
//!     Message::user("Hello!"),
//!     Message::assistant("Hi there!"),
//! ];
//!
//! let fitted = window.fit_messages(&messages);
//! ```

use crate::memory::Memory;
use crate::messages::{Message, MessageRole};

// =============================================================================
// Truncation Strategy
// =============================================================================

/// Strategies for truncating context when it exceeds the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationStrategy {
    /// Keep the most recent messages, drop oldest.
    ///
    /// Best for conversations where recent context is most important.
    #[default]
    KeepRecent,

    /// Keep first (system) message + recent messages.
    ///
    /// Best for maintaining agent identity while preserving recent context.
    KeepSystemAndRecent,

    /// Keep first and last messages, drop middle.
    ///
    /// Best when both initial context and final state matter.
    KeepEnds,
}

// =============================================================================
// Context Window Config
// =============================================================================

/// Configuration for context window management.
#[derive(Debug, Clone)]
pub struct ContextWindowConfig {
    /// Maximum tokens in the context window.
    pub max_tokens: usize,

    /// Strategy for truncating when over limit.
    pub truncation_strategy: TruncationStrategy,

    /// Reserved tokens for response generation.
    pub reserved_for_response: usize,

    /// Approximate tokens per character (for estimation).
    ///
    /// Common values:
    /// - 0.25 for English text (~4 chars per token)
    /// - 0.33 for code (~3 chars per token)
    /// - 0.5 for non-Latin scripts (~2 chars per token)
    pub tokens_per_char: f32,
}

impl Default for ContextWindowConfig {
    fn default() -> Self {
        Self {
            max_tokens: 8192,
            truncation_strategy: TruncationStrategy::KeepSystemAndRecent,
            reserved_for_response: 1024,
            tokens_per_char: 0.25, // ~4 chars per token average for English
        }
    }
}

impl ContextWindowConfig {
    /// Creates a new config with the specified max tokens.
    ///
    /// Uses default values for other settings.
    #[must_use]
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Self::default()
        }
    }

    /// Sets the truncation strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation_strategy = strategy;
        self
    }

    /// Sets the reserved tokens for response.
    #[must_use]
    pub fn with_reserved_for_response(mut self, reserved: usize) -> Self {
        self.reserved_for_response = reserved;
        self
    }

    /// Sets the tokens per character ratio.
    #[must_use]
    pub fn with_tokens_per_char(mut self, ratio: f32) -> Self {
        self.tokens_per_char = ratio;
        self
    }
}

// =============================================================================
// Context Window
// =============================================================================

/// Manages context window size and message selection.
///
/// The context window is the limited "memory" available to the LLM during
/// a single request. This struct helps manage that constraint by:
///
/// 1. Estimating token counts for messages
/// 2. Truncating conversations to fit within limits
/// 3. Injecting relevant memories into context
#[derive(Debug, Clone)]
pub struct ContextWindow {
    config: ContextWindowConfig,
}

impl ContextWindow {
    /// Creates a new context window manager.
    #[must_use]
    pub fn new(config: ContextWindowConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &ContextWindowConfig {
        &self.config
    }

    /// Estimates the token count for a message.
    ///
    /// Uses character count as approximation. For accurate counts,
    /// use a tokenizer specific to the model (e.g., tiktoken for OpenAI).
    ///
    /// # Arguments
    ///
    /// * `message` - The message to estimate
    ///
    /// # Returns
    ///
    /// Estimated token count.
    #[must_use]
    pub fn estimate_tokens(&self, message: &Message) -> usize {
        let char_count = message.content.len();

        // Add overhead for role and formatting
        let role_overhead = 4; // Approximate tokens for role marker

        let content_tokens = (char_count as f32 * self.config.tokens_per_char).ceil() as usize;

        content_tokens + role_overhead
    }

    /// Estimates the token count for a string.
    #[must_use]
    pub fn estimate_string_tokens(&self, text: &str) -> usize {
        (text.len() as f32 * self.config.tokens_per_char).ceil() as usize
    }

    /// Estimates total tokens for a list of messages.
    #[must_use]
    pub fn estimate_total_tokens(&self, messages: &[Message]) -> usize {
        messages.iter().map(|m| self.estimate_tokens(m)).sum()
    }

    /// Returns the available tokens for context (after reserving for response).
    #[must_use]
    pub fn available_tokens(&self) -> usize {
        self.config
            .max_tokens
            .saturating_sub(self.config.reserved_for_response)
    }

    /// Fits messages within the context window.
    ///
    /// Returns messages that fit within the available token budget,
    /// applying the configured truncation strategy.
    ///
    /// # Arguments
    ///
    /// * `messages` - The messages to fit
    ///
    /// # Returns
    ///
    /// A subset of messages that fit within the context window.
    #[must_use]
    pub fn fit_messages(&self, messages: &[Message]) -> Vec<Message> {
        let available = self.available_tokens();
        let total = self.estimate_total_tokens(messages);

        if total <= available {
            return messages.to_vec();
        }

        match self.config.truncation_strategy {
            TruncationStrategy::KeepRecent => self.truncate_keep_recent(messages, available),
            TruncationStrategy::KeepSystemAndRecent => {
                self.truncate_keep_system_and_recent(messages, available)
            }
            TruncationStrategy::KeepEnds => self.truncate_keep_ends(messages, available),
        }
    }

    /// Truncation strategy: keep most recent messages.
    fn truncate_keep_recent(&self, messages: &[Message], available: usize) -> Vec<Message> {
        let mut result = Vec::new();
        let mut total = 0;

        for message in messages.iter().rev() {
            let tokens = self.estimate_tokens(message);
            if total + tokens > available {
                break;
            }
            result.push(message.clone());
            total += tokens;
        }

        result.reverse();
        result
    }

    /// Truncation strategy: keep system message + recent messages.
    fn truncate_keep_system_and_recent(
        &self,
        messages: &[Message],
        available: usize,
    ) -> Vec<Message> {
        if messages.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut total = 0;

        // Always keep first message if it's system
        let first = &messages[0];
        if first.role == MessageRole::System {
            let tokens = self.estimate_tokens(first);
            if tokens <= available {
                result.push(first.clone());
                total = tokens;
            }
        }

        // Add recent messages that fit
        let remaining_start = if result.is_empty() { 0 } else { 1 };
        let remaining = &messages[remaining_start..];
        let mut recent = Vec::new();

        for message in remaining.iter().rev() {
            let tokens = self.estimate_tokens(message);
            if total + tokens > available {
                break;
            }
            recent.push(message.clone());
            total += tokens;
        }

        recent.reverse();
        result.extend(recent);
        result
    }

    /// Truncation strategy: keep first and last messages.
    fn truncate_keep_ends(&self, messages: &[Message], available: usize) -> Vec<Message> {
        if messages.is_empty() {
            return Vec::new();
        }

        if messages.len() == 1 {
            let tokens = self.estimate_tokens(&messages[0]);
            if tokens <= available {
                return vec![messages[0].clone()];
            }
            return Vec::new();
        }

        let first_tokens = self.estimate_tokens(&messages[0]);
        let last_tokens = self.estimate_tokens(&messages[messages.len() - 1]);

        if first_tokens + last_tokens > available {
            // Can't even fit first and last, fall back to recent
            return self.truncate_keep_recent(messages, available);
        }

        // Start with first and last
        let mut result = vec![messages[0].clone()];
        let mut current_tokens = first_tokens + last_tokens;

        // Try to add messages from the end (before the last message)
        let middle = &messages[1..messages.len() - 1];
        let mut middle_to_add = Vec::new();

        for message in middle.iter().rev() {
            let tokens = self.estimate_tokens(message);
            if current_tokens + tokens > available {
                break;
            }
            middle_to_add.push(message.clone());
            current_tokens += tokens;
        }

        middle_to_add.reverse();
        result.extend(middle_to_add);
        result.push(messages[messages.len() - 1].clone());

        result
    }

    /// Builds an optimized context including relevant memories.
    ///
    /// This method:
    /// 1. Constructs a system message with the prompt and relevant memories
    /// 2. Adds conversation messages
    /// 3. Fits everything within the context window
    ///
    /// # Arguments
    ///
    /// * `system_prompt` - The agent's system prompt
    /// * `memories` - Relevant memories to inject
    /// * `conversation` - Current conversation messages
    ///
    /// # Returns
    ///
    /// Optimized list of messages fitting within context window.
    #[must_use]
    pub fn build_context(
        &self,
        system_prompt: &str,
        memories: &[Memory],
        conversation: &[Message],
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        // Build system message with memories
        let system_content = build_system_with_memories(system_prompt, memories);
        messages.push(Message::system(system_content));

        // Add conversation messages (skip any system messages as we built our own)
        messages.extend(
            conversation
                .iter()
                .filter(|m| m.role != MessageRole::System)
                .cloned(),
        );

        // Fit within window
        self.fit_messages(&messages)
    }

    /// Returns statistics about how messages fit in the context.
    #[must_use]
    pub fn get_context_stats(&self, messages: &[Message]) -> ContextStats {
        let total_tokens = self.estimate_total_tokens(messages);
        let available = self.available_tokens();

        ContextStats {
            message_count: messages.len(),
            estimated_tokens: total_tokens,
            available_tokens: available,
            utilization_percent: if available > 0 {
                ((total_tokens as f64 / available as f64) * 100.0).min(100.0)
            } else {
                0.0
            },
            is_truncated: total_tokens > available,
        }
    }
}

impl Default for ContextWindow {
    fn default() -> Self {
        Self::new(ContextWindowConfig::default())
    }
}

// =============================================================================
// Context Stats
// =============================================================================

/// Statistics about context window utilization.
#[derive(Debug, Clone)]
pub struct ContextStats {
    /// Number of messages in context.
    pub message_count: usize,
    /// Estimated total tokens.
    pub estimated_tokens: usize,
    /// Available tokens in window.
    pub available_tokens: usize,
    /// Utilization as percentage (0-100).
    pub utilization_percent: f64,
    /// Whether the context was truncated.
    pub is_truncated: bool,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Builds a system prompt with injected memories.
fn build_system_with_memories(system_prompt: &str, memories: &[Memory]) -> String {
    if memories.is_empty() {
        return system_prompt.to_string();
    }

    let mut content = system_prompt.to_string();
    content.push_str("\n\n## Relevant Context from Memory\n\n");

    for memory in memories {
        content.push_str(&format!("- {}\n", memory.content));
    }

    content
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentId;

    // Helper to create test messages
    fn msg(role: MessageRole, content: &str) -> Message {
        Message {
            role,
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    // -------------------------------------------------------------------------
    // Config Tests
    // -------------------------------------------------------------------------

    #[test]
    fn config_default() {
        let config = ContextWindowConfig::default();
        assert_eq!(config.max_tokens, 8192);
        assert_eq!(
            config.truncation_strategy,
            TruncationStrategy::KeepSystemAndRecent
        );
        assert_eq!(config.reserved_for_response, 1024);
    }

    #[test]
    fn config_with_max_tokens() {
        let config = ContextWindowConfig::with_max_tokens(4096);
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn config_builder_chain() {
        let config = ContextWindowConfig::with_max_tokens(4096)
            .with_strategy(TruncationStrategy::KeepRecent)
            .with_reserved_for_response(512)
            .with_tokens_per_char(0.33);

        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.truncation_strategy, TruncationStrategy::KeepRecent);
        assert_eq!(config.reserved_for_response, 512);
        assert!((config.tokens_per_char - 0.33).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Token Estimation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn estimate_tokens_empty_message() {
        let window = ContextWindow::default();
        let message = msg(MessageRole::User, "");
        let tokens = window.estimate_tokens(&message);
        assert_eq!(tokens, 4); // Just role overhead
    }

    #[test]
    fn estimate_tokens_with_content() {
        let window = ContextWindow::default();
        // 100 chars at 0.25 tokens/char = 25 tokens + 4 overhead = 29
        let message = msg(MessageRole::User, &"a".repeat(100));
        let tokens = window.estimate_tokens(&message);
        assert_eq!(tokens, 29);
    }

    #[test]
    fn estimate_total_tokens() {
        let window = ContextWindow::default();
        let messages = vec![
            msg(MessageRole::System, &"a".repeat(100)),
            msg(MessageRole::User, &"b".repeat(100)),
        ];
        let total = window.estimate_total_tokens(&messages);
        assert_eq!(total, 58); // 29 + 29
    }

    #[test]
    fn available_tokens() {
        let config = ContextWindowConfig {
            max_tokens: 4096,
            reserved_for_response: 1024,
            ..Default::default()
        };
        let window = ContextWindow::new(config);
        assert_eq!(window.available_tokens(), 3072);
    }

    // -------------------------------------------------------------------------
    // Fit Messages Tests
    // -------------------------------------------------------------------------

    #[test]
    fn fit_messages_under_limit() {
        let window = ContextWindow::default();
        let messages = vec![
            msg(MessageRole::System, "Hello"),
            msg(MessageRole::User, "Hi"),
        ];

        let fitted = window.fit_messages(&messages);
        assert_eq!(fitted.len(), 2);
    }

    #[test]
    fn fit_messages_keep_recent() {
        let config = ContextWindowConfig {
            max_tokens: 100,
            truncation_strategy: TruncationStrategy::KeepRecent,
            reserved_for_response: 20,
            tokens_per_char: 0.25,
        };
        let window = ContextWindow::new(config);

        // Create messages that exceed the limit
        let messages = vec![
            msg(MessageRole::System, &"a".repeat(200)), // Will be dropped
            msg(MessageRole::User, &"b".repeat(50)),    // ~17 tokens
            msg(MessageRole::Assistant, &"c".repeat(50)), // ~17 tokens
        ];

        let fitted = window.fit_messages(&messages);
        // Should keep only the last 2 messages
        assert!(fitted.len() <= 2);
        // Last message should be the assistant's
        if !fitted.is_empty() {
            assert_eq!(fitted.last().unwrap().content, "c".repeat(50));
        }
    }

    #[test]
    fn fit_messages_keep_system_and_recent() {
        let config = ContextWindowConfig {
            max_tokens: 100,
            truncation_strategy: TruncationStrategy::KeepSystemAndRecent,
            reserved_for_response: 20,
            tokens_per_char: 0.25,
        };
        let window = ContextWindow::new(config);

        let messages = vec![
            msg(MessageRole::System, "sys"),          // Small, should be kept
            msg(MessageRole::User, &"a".repeat(200)), // Will be dropped
            msg(MessageRole::Assistant, &"b".repeat(200)), // Will be dropped
            msg(MessageRole::User, &"c".repeat(50)),  // Should fit
        ];

        let fitted = window.fit_messages(&messages);

        // Should have system + recent
        assert!(!fitted.is_empty());
        assert_eq!(fitted[0].role, MessageRole::System);
    }

    #[test]
    fn fit_messages_keep_ends() {
        let config = ContextWindowConfig {
            max_tokens: 50, // Small window
            truncation_strategy: TruncationStrategy::KeepEnds,
            reserved_for_response: 10,
            tokens_per_char: 0.25,
        };
        let window = ContextWindow::new(config);

        // Available: 40 tokens
        // "start" = 5*0.25 + 4 = 6 tokens
        // "end" = 3*0.25 + 4 = 5 tokens
        // Total for first+last = ~11 tokens
        // Middle = 200*0.25 + 4 = 54 tokens (won't fit)
        let messages = vec![
            msg(MessageRole::System, "start"),
            msg(MessageRole::User, &"a".repeat(200)), // Too big for remaining space
            msg(MessageRole::Assistant, "end"),
        ];

        let fitted = window.fit_messages(&messages);

        // Should have first and last only
        assert_eq!(fitted.len(), 2);
        assert_eq!(fitted[0].content, "start");
        assert_eq!(fitted[1].content, "end");
    }

    #[test]
    fn fit_messages_empty() {
        let window = ContextWindow::default();
        let fitted = window.fit_messages(&[]);
        assert!(fitted.is_empty());
    }

    // -------------------------------------------------------------------------
    // Build Context Tests
    // -------------------------------------------------------------------------

    #[test]
    fn build_context_no_memories() {
        let window = ContextWindow::default();
        let conversation = vec![
            msg(MessageRole::User, "Hello"),
            msg(MessageRole::Assistant, "Hi there"),
        ];

        let context = window.build_context("You are helpful.", &[], &conversation);

        assert_eq!(context.len(), 3); // System + 2 conversation
        assert_eq!(context[0].role, MessageRole::System);
        assert_eq!(context[0].content, "You are helpful.");
    }

    #[test]
    fn build_context_with_memories() {
        let window = ContextWindow::default();
        let agent_id = AgentId::new();
        let memories = vec![
            Memory::new(agent_id.clone(), "User likes blue"),
            Memory::new(agent_id, "User is from Seattle"),
        ];

        let context = window.build_context("You are helpful.", &memories, &[]);

        assert_eq!(context.len(), 1); // Just system
        assert!(context[0].content.contains("Relevant Context"));
        assert!(context[0].content.contains("User likes blue"));
        assert!(context[0].content.contains("User is from Seattle"));
    }

    #[test]
    fn build_context_skips_conversation_system_messages() {
        let window = ContextWindow::default();
        let conversation = vec![
            msg(MessageRole::System, "Old system prompt"),
            msg(MessageRole::User, "Hello"),
        ];

        let context = window.build_context("New system prompt", &[], &conversation);

        // Should have our new system + user, not the old system
        assert_eq!(context.len(), 2);
        assert_eq!(context[0].content, "New system prompt");
        assert_eq!(context[1].content, "Hello");
    }

    // -------------------------------------------------------------------------
    // Context Stats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn context_stats_under_limit() {
        let window = ContextWindow::default();
        let messages = vec![msg(MessageRole::User, "Hello")];

        let stats = window.get_context_stats(&messages);

        assert_eq!(stats.message_count, 1);
        assert!(!stats.is_truncated);
        assert!(stats.utilization_percent < 100.0);
    }

    #[test]
    fn context_stats_over_limit() {
        let config = ContextWindowConfig {
            max_tokens: 50,
            reserved_for_response: 10,
            ..Default::default()
        };
        let window = ContextWindow::new(config);

        let messages = vec![msg(MessageRole::User, &"a".repeat(500))];

        let stats = window.get_context_stats(&messages);

        assert!(stats.is_truncated);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn fit_single_message_too_large() {
        let config = ContextWindowConfig {
            max_tokens: 10,
            truncation_strategy: TruncationStrategy::KeepRecent,
            reserved_for_response: 5,
            tokens_per_char: 0.25,
        };
        let window = ContextWindow::new(config);

        let messages = vec![msg(MessageRole::User, &"a".repeat(100))];
        let fitted = window.fit_messages(&messages);

        // Even the one message doesn't fit
        assert!(fitted.is_empty());
    }

    #[test]
    fn truncation_strategy_equality() {
        assert_eq!(
            TruncationStrategy::KeepRecent,
            TruncationStrategy::KeepRecent
        );
        assert_ne!(
            TruncationStrategy::KeepRecent,
            TruncationStrategy::KeepSystemAndRecent
        );
    }

    #[test]
    fn truncation_strategy_default() {
        let strategy = TruncationStrategy::default();
        assert_eq!(strategy, TruncationStrategy::KeepRecent);
    }
}
