//! Auto-compaction: summarizing the elided head of a history with the model.
//!
//! A turn that works through tools appends an assistant turn and every tool
//! result to the history on each round. Nothing bounds that growth, so a long
//! tool loop eventually exceeds the provider's context window — a hard error
//! that arrives mid-turn, after the money for the earlier rounds is spent.
//! A long-running conversation replayed through
//! [`continue_with`](crate::facade::ActonAI::continue_with) gets there more
//! slowly, but gets there.
//!
//! Truncation is the wrong instrument here. Dropping the oldest exchanges
//! silently erases the user's original request, and the model has no way to
//! know it happened; it simply answers a question it can no longer see.
//! Compaction removes the same messages but leaves a **summary** in their
//! place: the elided span is sent back to the *same provider* with
//! [`COMPACTION_PROMPT`], and the model's own condensed account of it is
//! spliced in where the span used to be. The model is told what it forgot,
//! rather than just forgetting.
//!
//! # Purity
//!
//! Everything in this module is a pure function of its inputs. The one side
//! effect — the summarization request — belongs to the prompt loop, which is
//! where every other provider round already lives. This module decides *what*
//! to elide ([`plan_compaction`]), renders *the request that summarizes it*
//! ([`summarization_messages`]), and assembles *the history that results*
//! ([`finish_compaction`]); each step is testable without a provider.
//!
//! # Shape of the result
//!
//! ```text
//! [system]  kept as-is, never elided
//! [user]    compaction notice + the model's summary of everything removed
//! [ ... ]   the last N exchanges, verbatim
//! ```
//!
//! Compaction never splits an exchange — an assistant turn carrying tool
//! calls travels with the tool results answering it — because a `tool_use`
//! with no `tool_result` is rejected by every provider for the rest of the
//! conversation, not just for the request that introduced it.
//!
//! # Example
//!
//! ```rust,ignore
//! use acton_ai::memory::{
//!     finish_compaction, plan_compaction, summarization_messages,
//!     CompactionConfig, ContextWindow,
//! };
//!
//! let window = ContextWindow::default();
//! let config = CompactionConfig::default();
//!
//! if let Some(plan) = plan_compaction(&window, &config, &history) {
//!     let request = summarization_messages(&plan);
//!     let summary = send_to_provider(request).await?; // the loop's job
//!     if let Some((compacted, outcome)) = finish_compaction(&window, &plan, &summary) {
//!         println!("elided {} messages", outcome.messages_elided);
//!         history = compacted;
//!     }
//! }
//! ```

use crate::memory::context::{exchanges, ContextWindow};
use crate::messages::{Message, MessageRole};
use std::fmt;
use std::str::FromStr;

// =============================================================================
// Constants
// =============================================================================

/// Opening line of every compacted history's summary message.
///
/// Stable and searchable on purpose: it is how a reader of a transcript, a
/// stored session, or a test recognizes that a history has been compacted.
pub const COMPACTION_NOTICE: &str =
    "[conversation compacted] Earlier messages were summarized to stay within the context \
     window. What follows is the summary; treat it as fact, and ask rather than guess if \
     you need a detail it does not carry.";

/// The instruction sent to the provider alongside the elided transcript.
///
/// This is the system message of every summarization request, verbatim. It is
/// a `const` rather than configuration because the invariants it encodes —
/// preserve intent, preserve results, answer with the summary alone — are what
/// [`apply_compaction`] depends on to build a history the model can continue
/// from.
pub const COMPACTION_PROMPT: &str =
    "You are compacting an ongoing conversation so it can continue within a limited \
     context window. Summarize the transcript below into a concise briefing that \
     preserves everything the conversation still depends on: the user's original \
     request and intent, decisions made and their reasons, facts and figures \
     discovered, tools invoked with their key results, the current state of the work, \
     and any open questions or next steps. Do not comment on this task or on the \
     transcript's format; reply with the briefing alone.";

/// Default fraction of the available budget at which compaction triggers.
const DEFAULT_THRESHOLD: f64 = 0.8;

/// Default number of trailing exchanges kept verbatim.
const DEFAULT_KEEP_RECENT_TURNS: usize = 3;

/// Character ceiling on any one message's rendering in the transcript.
///
/// The transcript rides inside a request that must itself fit the provider's
/// window; one enormous tool result must not be able to make the
/// summarization request bigger than the history it is trying to shrink.
const TRANSCRIPT_MESSAGE_CAP: usize = 6_000;

// =============================================================================
// Errors
// =============================================================================

/// A compaction setting that cannot be honored.
///
/// Returned by the validating constructors of [`CompactionThreshold`] and
/// [`KeepRecentTurns`], and surfaced at launch when the offending value came
/// from `[context]` in TOML.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionConfigError {
    kind: CompactionConfigErrorKind,
}

/// The specific setting that was rejected.
///
/// Marked `#[non_exhaustive]`: new settings bring new failure modes, and a
/// downstream `match` should not break each time one is added.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CompactionConfigErrorKind {
    /// A threshold outside `0.0 < value <= 1.0`, or not a number at all.
    ThresholdOutOfRange {
        /// The value that was rejected.
        value: f64,
    },
    /// A threshold string that does not parse as a number.
    ThresholdNotANumber {
        /// The text that was rejected.
        value: String,
    },
    /// A keep-recent-turns count of zero.
    ///
    /// Keeping nothing would replace the entire history with a summary,
    /// leaving the model no live context to answer from.
    KeepRecentTurnsZero,
}

impl CompactionConfigError {
    /// Creates an error with the given kind.
    #[must_use]
    pub fn new(kind: CompactionConfigErrorKind) -> Self {
        Self { kind }
    }

    /// Returns the specific setting that was rejected.
    #[must_use]
    pub fn kind(&self) -> &CompactionConfigErrorKind {
        &self.kind
    }
}

impl fmt::Display for CompactionConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            CompactionConfigErrorKind::ThresholdOutOfRange { value } => write!(
                f,
                "compact_threshold {value} is out of range; expected a fraction \
                 greater than 0.0 and at most 1.0 (for example 0.8)"
            ),
            CompactionConfigErrorKind::ThresholdNotANumber { value } => write!(
                f,
                "compact_threshold '{value}' is not a number; expected a fraction \
                 such as 0.8"
            ),
            CompactionConfigErrorKind::KeepRecentTurnsZero => write!(
                f,
                "keep_recent_turns must be at least 1; keeping none would replace \
                 the whole history with a summary"
            ),
        }
    }
}

impl std::error::Error for CompactionConfigError {}

// =============================================================================
// Newtypes
// =============================================================================

/// The fraction of the available token budget at which compaction triggers.
///
/// Valid range is `0.0 < value <= 1.0`. A threshold of `1.0` compacts only
/// once the history has actually filled the budget, which leaves no headroom
/// for the round that follows; the default of `0.8` compacts while there is
/// still room to work.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct CompactionThreshold(f64);

impl CompactionThreshold {
    /// Creates a threshold, rejecting anything outside `0.0 < value <= 1.0`.
    ///
    /// # Errors
    ///
    /// Returns [`CompactionConfigErrorKind::ThresholdOutOfRange`] for zero,
    /// negative, infinite, `NaN`, and values above `1.0`.
    pub fn new(value: f64) -> Result<Self, CompactionConfigError> {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(CompactionConfigError::new(
                CompactionConfigErrorKind::ThresholdOutOfRange { value },
            ));
        }
        Ok(Self(value))
    }

    /// Returns the fraction.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Default for CompactionThreshold {
    fn default() -> Self {
        Self(DEFAULT_THRESHOLD)
    }
}

impl fmt::Display for CompactionThreshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<f32> for CompactionThreshold {
    type Error = CompactionConfigError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(f64::from(value))
    }
}

impl TryFrom<f64> for CompactionThreshold {
    type Error = CompactionConfigError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl FromStr for CompactionThreshold {
    type Err = CompactionConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parsed: f64 = s.trim().parse().map_err(|_| {
            CompactionConfigError::new(CompactionConfigErrorKind::ThresholdNotANumber {
                value: s.to_string(),
            })
        })?;
        Self::new(parsed)
    }
}

/// How many trailing turns survive compaction verbatim.
///
/// Counted in **exchanges**, not messages: an assistant turn and the tool
/// results answering it are one unit, so a count of 3 always leaves three
/// whole units of work intact rather than three arbitrary messages. Splitting
/// one would orphan a `tool_use` or a `tool_result`, which every provider
/// rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeepRecentTurns(usize);

impl KeepRecentTurns {
    /// Creates a count, rejecting zero.
    ///
    /// # Errors
    ///
    /// Returns [`CompactionConfigErrorKind::KeepRecentTurnsZero`] for `0`.
    pub fn new(value: usize) -> Result<Self, CompactionConfigError> {
        if value == 0 {
            return Err(CompactionConfigError::new(
                CompactionConfigErrorKind::KeepRecentTurnsZero,
            ));
        }
        Ok(Self(value))
    }

    /// Returns the count.
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

impl Default for KeepRecentTurns {
    fn default() -> Self {
        Self(DEFAULT_KEEP_RECENT_TURNS)
    }
}

impl fmt::Display for KeepRecentTurns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<usize> for KeepRecentTurns {
    type Error = CompactionConfigError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl FromStr for KeepRecentTurns {
    type Err = CompactionConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parsed: usize = s.trim().parse().map_err(|_| {
            CompactionConfigError::new(CompactionConfigErrorKind::KeepRecentTurnsZero)
        })?;
        Self::new(parsed)
    }
}

// =============================================================================
// Config
// =============================================================================

/// The compaction policy in force.
///
/// A runtime holds `Option<CompactionConfig>`; `None` means compaction is off,
/// which is the default. Configuring one *is* the request to use it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CompactionConfig {
    /// Fraction of the available budget at which compaction triggers.
    pub threshold: CompactionThreshold,
    /// Trailing turns kept verbatim.
    pub keep_recent_turns: KeepRecentTurns,
}

impl CompactionConfig {
    /// Creates a policy with default settings: compact at 80% of the
    /// available budget, keep the last three turns.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the trigger threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: CompactionThreshold) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets how many trailing turns survive verbatim.
    #[must_use]
    pub fn with_keep_recent_turns(mut self, keep: KeepRecentTurns) -> Self {
        self.keep_recent_turns = keep;
        self
    }
}

// =============================================================================
// Plan, outcome, record
// =============================================================================

/// What a compaction would do, decided but not yet applied.
///
/// Produced only by [`plan_compaction`], so a plan always describes a split
/// that respects exchange boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPlan {
    system: Option<Message>,
    elided: Vec<Message>,
    preserved: Vec<Message>,
    messages_before: usize,
    tokens_before: usize,
}

impl CompactionPlan {
    /// The leading system message, held aside and never elided.
    #[must_use]
    pub fn system(&self) -> Option<&Message> {
        self.system.as_ref()
    }

    /// The messages the summary will replace, in order.
    #[must_use]
    pub fn elided(&self) -> &[Message] {
        &self.elided
    }

    /// How many leading non-system messages the summary will replace.
    ///
    /// [`plan_compaction`] always elides from the front: the elided span is
    /// exactly the first `elided_prefix_len()` messages of the history with
    /// its leading system message held aside, and [`preserved`](Self::preserved)
    /// is everything after them. This is the number a caller that keeps its
    /// own copy of the history needs in order to replay the compaction on it.
    #[must_use]
    pub fn elided_prefix_len(&self) -> usize {
        self.elided.len()
    }

    /// The trailing turns that survive verbatim, in order.
    #[must_use]
    pub fn preserved(&self) -> &[Message] {
        &self.preserved
    }

    /// Estimated tokens of the history this plan was made from.
    #[must_use]
    pub fn tokens_before(&self) -> usize {
        self.tokens_before
    }
}

/// What a compaction actually did.
///
/// Carried by the log line, by
/// [`TurnLifecycle::ContextCompacted`](crate::messages::TurnLifecycle::ContextCompacted),
/// and by every [`CompactionRecord`], so an operator can see compaction
/// happening rather than infer it from a history that quietly changed shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionOutcome {
    /// Messages in the history before compaction.
    pub messages_before: usize,
    /// Messages in the history after compaction.
    pub messages_after: usize,
    /// Estimated tokens before compaction.
    pub tokens_before: usize,
    /// Estimated tokens after compaction.
    pub tokens_after: usize,
    /// Messages the summary replaced.
    pub messages_elided: usize,
}

impl CompactionOutcome {
    /// Tokens the compaction reclaimed.
    #[must_use]
    pub fn tokens_reclaimed(&self) -> usize {
        self.tokens_before.saturating_sub(self.tokens_after)
    }
}

/// One compaction that happened during a turn, carried on
/// [`CollectedResponse`](crate::stream::CollectedResponse).
///
/// This is how compaction stays transparent in persistence: a caller that
/// stores a session can write [`Self::as_message`] alongside the turn's own
/// messages, and the stored conversation then records that — and what — the
/// model was told it forgot. The CLI's session store does exactly that.
///
/// An embedder that owns the history itself — one that hands the prompt loop
/// `history + prompt` on every turn and keeps its own copy — needs more than
/// the summary: it needs to know *which* of its messages the summary stands
/// for, or the next turn resends the elided span and pays for the same
/// summary again. [`Self::elided_prefix_len`] is that promise, stated as a
/// guarantee rather than left to be inferred from
/// [`CompactionOutcome::messages_elided`], and [`Self::adopt`] applies it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionRecord {
    /// The provider-written summary that replaced the elided span.
    pub summary: String,
    /// The measured effect of the pass.
    pub outcome: CompactionOutcome,
    /// How many leading messages the summary replaced, counted from the first
    /// **non-system** message of the history as the loop held it when the
    /// pass ran.
    ///
    /// The loop always elides a strict prefix: a leading system message is
    /// held aside and never counted, and the summary takes the place of
    /// exactly the next `elided_prefix_len` messages, in order, with nothing
    /// removed from the middle. Two records of one turn apply in sequence,
    /// each against the history the previous one produced (which then opens
    /// with that record's summary message).
    pub elided_prefix_len: usize,
}

impl CompactionRecord {
    /// The record as a message: byte-identical to the summary message the
    /// model saw on the wire, built by [`summary_message`].
    #[must_use]
    pub fn as_message(&self) -> Message {
        summary_message(&self.summary)
    }

    /// Replays this compaction onto a caller's own copy of the history.
    ///
    /// Holds a leading system message aside, drops the first
    /// [`elided_prefix_len`](Self::elided_prefix_len) messages after it —
    /// or all of them, when the caller's copy is shorter than the span the
    /// loop elided, because the loop's list also carried the turn's prompt
    /// and rounds the caller never keeps — and puts [`Self::as_message`] in
    /// their place. The caller's messages are otherwise left as they are:
    /// the structural repair the loop applies on the wire (which can merge a
    /// preserved user turn into the summary message) is not applied here,
    /// because the loop applies it again to whatever the caller sends next,
    /// and a stored session should keep the participants' turns separate.
    ///
    /// Pure and total: it never fails, and the summary is present in the
    /// result exactly once.
    #[must_use]
    pub fn adopt(&self, history: &[Message]) -> Vec<Message> {
        let (system, rest) = split_leading_system(history);
        let dropped = self.elided_prefix_len.min(rest.len());
        let mut adopted = Vec::with_capacity(rest.len() - dropped + 2);
        adopted.extend(system);
        adopted.push(self.as_message());
        adopted.extend(rest[dropped..].iter().cloned());
        adopted
    }
}

// =============================================================================
// Planning
// =============================================================================

/// Decides whether to compact, and what to elide if so.
///
/// Returns `None` — meaning "leave the history alone" — when:
///
/// - the history is below `config.threshold` of the window's available budget;
/// - there is nothing left to elide once `config.keep_recent_turns` exchanges
///   are reserved;
/// - the window has no available budget at all, which is a misconfiguration
///   compaction cannot repair.
#[must_use]
pub fn plan_compaction(
    window: &ContextWindow,
    config: &CompactionConfig,
    messages: &[Message],
) -> Option<CompactionPlan> {
    let available = window.available_tokens();
    if available == 0 {
        return None;
    }

    let tokens_before = window.estimate_total_tokens(messages);
    if tokens_before < trigger_tokens(available, config.threshold) {
        return None;
    }

    let (system, rest) = split_leading_system(messages);
    let runs = exchanges(rest);
    if runs.len() <= config.keep_recent_turns.get() {
        return None;
    }

    let split_at = runs.len() - config.keep_recent_turns.get();
    let elided = flatten(&runs[..split_at]);
    if elided.is_empty() {
        return None;
    }

    Some(CompactionPlan {
        system,
        elided,
        preserved: flatten(&runs[split_at..]),
        messages_before: messages.len(),
        tokens_before,
    })
}

/// The token count at which compaction triggers.
///
/// Rounded up, so a threshold of `1.0` means "at the budget" rather than
/// "just under it".
fn trigger_tokens(available: usize, threshold: CompactionThreshold) -> usize {
    let budget = available as f64;
    let scaled = (budget * threshold.get()).ceil().max(1.0);
    if scaled >= budget {
        // A threshold of 1.0 (or a budget so large the product rounds up to
        // it) means "at the budget"; clamping here is what makes the cast
        // below exact rather than truncating.
        return available.max(1);
    }
    // `scaled` is a whole, non-negative f64 strictly below `available`, so
    // the cast is exact by construction.
    scaled as usize
}

/// Splits a leading system message off the front of a history.
fn split_leading_system(messages: &[Message]) -> (Option<Message>, &[Message]) {
    match messages.first() {
        Some(first) if first.role == MessageRole::System => (Some(first.clone()), &messages[1..]),
        _ => (None, messages),
    }
}

/// Concatenates a slice of exchanges back into a flat message list.
fn flatten(runs: &[&[Message]]) -> Vec<Message> {
    runs.iter().flat_map(|run| run.iter().cloned()).collect()
}

// =============================================================================
// The summarization request
// =============================================================================

/// Builds the request that asks the provider to summarize a plan's elided span.
///
/// Two messages — [`COMPACTION_PROMPT`] as the system turn, the rendered
/// transcript as the user turn — which is the minimal shape that satisfies
/// every wire invariant: it opens on a user turn, alternates trivially, and
/// carries no tool traffic.
#[must_use]
pub fn summarization_messages(plan: &CompactionPlan) -> Vec<Message> {
    vec![
        Message::system(COMPACTION_PROMPT),
        Message::user(transcript(plan.elided())),
    ]
}

/// Renders messages as a plain-text transcript for the summarization request.
///
/// Deterministic, and bounded per message by `TRANSCRIPT_MESSAGE_CAP` so one
/// enormous tool result cannot make the summarization request bigger than the
/// history it is trying to shrink. Truncation counts characters, not bytes, so
/// it can never split a multi-byte scalar and produce text a provider rejects.
#[must_use]
pub fn transcript(messages: &[Message]) -> String {
    messages
        .iter()
        .map(transcript_entry)
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Renders one message as a transcript entry.
fn transcript_entry(message: &Message) -> String {
    let content = capped(&message.content, TRANSCRIPT_MESSAGE_CAP);
    match message.role {
        MessageRole::System => format!("system: {content}"),
        MessageRole::User => format!("user: {content}"),
        MessageRole::Assistant => match message.tool_calls.as_deref() {
            Some(calls) if !calls.is_empty() => {
                let names: Vec<&str> = calls.iter().map(|call| call.name.as_str()).collect();
                if content.is_empty() {
                    format!("assistant called: {}", names.join(", "))
                } else {
                    format!("assistant (calling {}): {content}", names.join(", "))
                }
            }
            _ => format!("assistant: {content}"),
        },
        MessageRole::Tool => format!(
            "tool result{}: {content}",
            message
                .tool_call_id
                .as_deref()
                .map(|id| format!(" [{id}]"))
                .unwrap_or_default(),
        ),
    }
}

/// Truncates text to `max` characters, marking the cut with an ellipsis.
fn capped(text: &str, max: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    let kept: String = trimmed.chars().take(max.saturating_sub(1)).collect();
    format!("{kept}…")
}

// =============================================================================
// Application
// =============================================================================

/// The summary as the message the model will see.
///
/// A **user** message: a history has to open on a user turn once the system
/// message is carried out of band, and the preserved tail rarely starts with
/// one. The content opens with [`COMPACTION_NOTICE`], which is what marks the
/// message — on the wire, in a stored session, and to the model — as the
/// framework's summary rather than something a participant said.
#[must_use]
pub fn summary_message(summary: &str) -> Message {
    let summary = summary.trim();
    if summary.is_empty() {
        Message::user(COMPACTION_NOTICE)
    } else {
        Message::user(format!("{COMPACTION_NOTICE}\n\n{summary}"))
    }
}

/// Assembles the compacted history from a plan and the provider's summary.
///
/// The result is passed through the same structural repair the clients apply,
/// which is what guarantees no orphaned `tool_result` survives the split and
/// no two same-role turns end up adjacent.
#[must_use]
pub fn apply_compaction(plan: &CompactionPlan, summary: &str) -> Vec<Message> {
    let mut assembled = Vec::with_capacity(plan.preserved.len() + 2);
    if let Some(system) = &plan.system {
        assembled.push(system.clone());
    }
    assembled.push(summary_message(summary));
    assembled.extend(plan.preserved.iter().cloned());

    crate::llm::sanitize::sanitize_history(&assembled)
}

/// Applies a plan's summary and measures it, declining a pass that reclaims
/// nothing.
///
/// Returns `None` when the compacted history would not actually be smaller —
/// a summary can outweigh a short elided span — which is what stops a history
/// that has already been compacted down to its floor from being rewritten,
/// and paid for, on every round.
#[must_use]
pub fn finish_compaction(
    window: &ContextWindow,
    plan: &CompactionPlan,
    summary: &str,
) -> Option<(Vec<Message>, CompactionOutcome)> {
    let compacted = apply_compaction(plan, summary);

    let tokens_after = window.estimate_total_tokens(&compacted);
    if tokens_after >= plan.tokens_before() {
        return None;
    }

    let outcome = CompactionOutcome {
        messages_before: plan.messages_before,
        messages_after: compacted.len(),
        tokens_before: plan.tokens_before(),
        tokens_after,
        messages_elided: plan.elided().len(),
    };
    Some((compacted, outcome))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::sanitize::sanitize_history;
    use crate::memory::{ContextWindowConfig, TruncationStrategy};
    use crate::messages::ToolCall;

    /// A window whose available budget is exactly `available` tokens under the
    /// cheap char-ratio estimator, so the tests can reason in round numbers.
    fn window(available: usize) -> ContextWindow {
        ContextWindow::new(ContextWindowConfig {
            max_tokens: available,
            truncation_strategy: TruncationStrategy::KeepRecent,
            reserved_for_response: 0,
            tokens_per_char: 0.25,
        })
    }

    fn tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: serde_json::json!({}),
        }
    }

    /// A plausible provider summary, sized like one.
    const SUMMARY: &str = "The user asked for the project's config values; the assistant \
                           located them via glob and read_file and reported the findings.";

    /// A history of `pairs` user/assistant exchanges, each message padded to
    /// `chars` characters so the token estimate is predictable.
    ///
    /// Messages are deliberately far longer than a summary: compaction only
    /// reclaims anything when the text it elides is bigger than the summary
    /// that replaces it, and a fixture of short messages would be testing the
    /// decline path while claiming to test the compaction path.
    fn chatty_history(pairs: usize, chars: usize) -> Vec<Message> {
        let mut messages = vec![Message::system("You are helpful.")];
        for i in 0..pairs {
            messages.push(Message::user(format!("q{i} {}", "x".repeat(chars))));
            messages.push(Message::assistant(format!("a{i} {}", "y".repeat(chars))));
        }
        messages
    }

    // -- newtypes ---------------------------------------------------------

    #[test]
    fn a_threshold_accepts_a_fraction_in_range() {
        assert_eq!(CompactionThreshold::new(0.5).unwrap().get(), 0.5);
        assert_eq!(CompactionThreshold::new(1.0).unwrap().get(), 1.0);
    }

    #[test]
    fn a_threshold_rejects_values_outside_its_range() {
        for bad in [0.0_f64, -0.1, 1.01, f64::NAN, f64::INFINITY] {
            let error = CompactionThreshold::new(bad).unwrap_err();
            assert!(
                matches!(
                    error.kind(),
                    CompactionConfigErrorKind::ThresholdOutOfRange { .. }
                ),
                "{bad} should be out of range",
            );
        }
    }

    #[test]
    fn a_threshold_parses_from_a_string() {
        assert_eq!("0.75".parse::<CompactionThreshold>().unwrap().get(), 0.75);
        let error = "eighty percent".parse::<CompactionThreshold>().unwrap_err();
        assert!(matches!(
            error.kind(),
            CompactionConfigErrorKind::ThresholdNotANumber { .. }
        ));
    }

    #[test]
    fn keep_recent_turns_rejects_zero() {
        assert_eq!(KeepRecentTurns::new(1).unwrap().get(), 1);
        let error = KeepRecentTurns::new(0).unwrap_err();
        assert_eq!(
            error.kind(),
            &CompactionConfigErrorKind::KeepRecentTurnsZero
        );
    }

    #[test]
    fn config_errors_say_what_would_have_been_accepted() {
        let rendered = CompactionConfigError::new(CompactionConfigErrorKind::ThresholdOutOfRange {
            value: 2.0,
        })
        .to_string();
        assert!(rendered.contains("0.8"), "{rendered}");
    }

    // -- planning ---------------------------------------------------------

    #[test]
    fn a_history_below_the_threshold_is_left_alone() {
        let window = window(10_000);
        let config = CompactionConfig::default();
        let messages = chatty_history(2, 40);

        assert!(plan_compaction(&window, &config, &messages).is_none());
    }

    #[test]
    fn a_history_over_the_threshold_elides_all_but_the_kept_tail() {
        let window = window(200);
        let config = CompactionConfig::default();
        let messages = chatty_history(6, 100);

        let plan = plan_compaction(&window, &config, &messages).expect("should compact");

        // System held aside, last three exchanges preserved (three messages,
        // since a plain user or assistant turn is one exchange each).
        assert_eq!(plan.system().map(|m| m.role), Some(MessageRole::System));
        assert_eq!(plan.preserved().len(), 3);
        assert_eq!(
            plan.preserved(),
            &messages[messages.len() - 3..],
            "the tail must survive verbatim",
        );
        assert_eq!(plan.elided().len(), messages.len() - 1 - 3);
    }

    #[test]
    fn a_history_with_only_the_kept_tail_is_left_alone() {
        let window = window(20);
        let config = CompactionConfig::default();
        // Three exchanges after the system message is exactly what is kept,
        // so there is nothing left to elide however large it is.
        let messages = vec![
            Message::system("s"),
            Message::user("u".repeat(400)),
            Message::assistant("a".repeat(400)),
            Message::user("u2".repeat(400)),
        ];

        assert!(plan_compaction(&window, &config, &messages).is_none());
    }

    #[test]
    fn a_window_with_no_budget_is_left_alone() {
        let window = ContextWindow::new(ContextWindowConfig {
            max_tokens: 100,
            truncation_strategy: TruncationStrategy::KeepRecent,
            reserved_for_response: 100,
            tokens_per_char: 0.25,
        });

        assert!(plan_compaction(
            &window,
            &CompactionConfig::default(),
            &chatty_history(10, 100)
        )
        .is_none());
    }

    #[test]
    fn a_tool_exchange_is_never_split_across_the_boundary() {
        let window = window(120);
        let config =
            CompactionConfig::default().with_keep_recent_turns(KeepRecentTurns::new(1).unwrap());

        let mut messages = vec![Message::system("s")];
        for i in 0..4 {
            messages.push(Message::user(format!("ask {i} {}", "x".repeat(200))));
            messages.push(Message::assistant_with_tools(
                "working",
                vec![tool_call(&format!("call-{i}"), "read_file")],
            ));
            messages.push(Message::tool(
                format!("call-{i}"),
                format!("result {i} {}", "y".repeat(200)),
            ));
        }

        let plan = plan_compaction(&window, &config, &messages).expect("should compact");

        // The preserved run is the final assistant-with-tools turn plus its
        // result — never the assistant turn on its own.
        assert_eq!(plan.preserved().len(), 2);
        assert_eq!(plan.preserved()[0].role, MessageRole::Assistant);
        assert_eq!(plan.preserved()[1].role, MessageRole::Tool);

        // And every elided tool result still has its assistant turn beside it.
        let elided_calls = plan
            .elided()
            .iter()
            .filter(|m| m.tool_calls.is_some())
            .count();
        let elided_results = plan
            .elided()
            .iter()
            .filter(|m| m.role == MessageRole::Tool)
            .count();
        assert_eq!(elided_calls, elided_results);
    }

    // -- the summarization request ----------------------------------------

    #[test]
    fn a_summarization_request_carries_the_prompt_and_the_elided_transcript() {
        let window = window(200);
        let messages = chatty_history(6, 100);
        let plan =
            plan_compaction(&window, &CompactionConfig::default(), &messages).expect("compact");

        let request = summarization_messages(&plan);

        assert_eq!(request.len(), 2);
        assert_eq!(request[0].role, MessageRole::System);
        assert_eq!(request[0].content, COMPACTION_PROMPT);
        assert_eq!(request[1].role, MessageRole::User);
        assert!(request[1].content.contains("q0"), "{}", request[1].content);
    }

    #[test]
    fn a_summarization_request_is_already_wire_shaped() {
        let window = window(200);
        let messages = chatty_history(6, 100);
        let plan =
            plan_compaction(&window, &CompactionConfig::default(), &messages).expect("compact");

        let request = summarization_messages(&plan);

        assert_eq!(
            sanitize_history(&request),
            request,
            "structural repair must find nothing to fix",
        );
    }

    #[test]
    fn a_transcript_names_the_tools_that_ran() {
        let messages = vec![
            Message::user("find the config"),
            Message::assistant_with_tools("looking", vec![tool_call("c1", "glob")]),
            Message::tool("c1", "acton-ai.toml"),
        ];

        let rendered = transcript(&messages);

        assert!(rendered.contains("find the config"), "{rendered}");
        assert!(rendered.contains("glob"), "{rendered}");
        assert!(rendered.contains("acton-ai.toml"), "{rendered}");
    }

    #[test]
    fn a_transcript_is_deterministic() {
        let messages = chatty_history(5, 50);

        assert_eq!(transcript(&messages), transcript(&messages));
    }

    #[test]
    fn a_transcript_of_nothing_is_empty() {
        assert!(transcript(&[]).is_empty());
    }

    #[test]
    fn a_transcript_caps_an_enormous_message_without_splitting_a_character() {
        let messages = vec![Message::user("日本語のテキスト".repeat(2_000))];

        let rendered = transcript(&messages);

        assert!(
            rendered.chars().count() < TRANSCRIPT_MESSAGE_CAP + 32,
            "one message must not exceed its cap by more than its framing",
        );
        // Reaching here at all means no panic on a char boundary; the
        // ellipsis proves the truncation actually happened.
        assert!(rendered.contains('…'), "{rendered}");
    }

    // -- application ------------------------------------------------------

    #[test]
    fn a_compacted_history_opens_with_the_system_turn_and_the_marked_summary() {
        let window = window(200);
        let messages = chatty_history(6, 2_000);
        let plan =
            plan_compaction(&window, &CompactionConfig::default(), &messages).expect("compact");

        let (compacted, _) =
            finish_compaction(&window, &plan, SUMMARY).expect("should reclaim tokens");

        assert_eq!(compacted[0].role, MessageRole::System);
        assert_eq!(compacted[1].role, MessageRole::User);
        assert!(compacted[1].content.starts_with(COMPACTION_NOTICE));
        assert!(compacted[1].content.contains(SUMMARY));
    }

    #[test]
    fn a_compacted_history_satisfies_the_wire_invariants() {
        let window = window(120);
        let config =
            CompactionConfig::default().with_keep_recent_turns(KeepRecentTurns::new(1).unwrap());

        let mut messages = vec![Message::system("s")];
        for i in 0..5 {
            messages.push(Message::assistant_with_tools(
                "working",
                vec![tool_call(&format!("call-{i}"), "bash")],
            ));
            messages.push(Message::tool(
                format!("call-{i}"),
                format!("out {i} {}", "y".repeat(3_000)),
            ));
        }
        let plan = plan_compaction(&window, &config, &messages).expect("compact");

        let (compacted, _) = finish_compaction(&window, &plan, SUMMARY).expect("should reclaim");

        // Idempotent under the same repair the clients apply: nothing left
        // to fix means every invariant already holds.
        assert_eq!(sanitize_history(&compacted), compacted);

        // And specifically: no orphaned tool result survived the split.
        let answered: Vec<&str> = compacted
            .iter()
            .filter_map(|m| m.tool_calls.as_deref())
            .flatten()
            .map(|c| c.id.as_str())
            .collect();
        for message in compacted.iter().filter(|m| m.role == MessageRole::Tool) {
            let id = message.tool_call_id.as_deref().expect("result needs an id");
            assert!(answered.contains(&id), "orphaned tool result {id}");
        }
    }

    #[test]
    fn compaction_reclaims_tokens() {
        let window = window(200);
        let messages = chatty_history(8, 2_000);
        let plan =
            plan_compaction(&window, &CompactionConfig::default(), &messages).expect("compact");

        let (compacted, outcome) =
            finish_compaction(&window, &plan, SUMMARY).expect("should reclaim");

        assert!(outcome.tokens_after < outcome.tokens_before);
        assert!(outcome.tokens_reclaimed() > 0);
        assert_eq!(outcome.messages_before, messages.len());
        assert_eq!(outcome.messages_after, compacted.len());
        assert!(outcome.messages_elided > 0);
    }

    #[test]
    fn repeated_compaction_reaches_a_fixpoint_instead_of_churning() {
        // A tiny window keeps the history over threshold even after the first
        // pass, so a second pass is attempted every round. Without the
        // no-progress guard this would pay for a summarization round forever,
        // mutating what the model sees on every single round.
        let window = window(24);
        let config =
            CompactionConfig::default().with_keep_recent_turns(KeepRecentTurns::new(1).unwrap());

        let mut current = chatty_history(6, 3_000);
        let mut passes = 0;
        while let Some(plan) = plan_compaction(&window, &config, &current) {
            let Some((next, outcome)) = finish_compaction(&window, &plan, SUMMARY) else {
                break;
            };
            assert!(
                outcome.tokens_after < outcome.tokens_before,
                "an accepted pass must reclaim tokens",
            );
            current = next;
            passes += 1;
            assert!(passes < 8, "compaction did not reach a fixpoint");
        }

        assert!(
            passes >= 1,
            "the fixture should have compacted at least once"
        );
        assert!(
            current
                .iter()
                .any(|m| m.content.contains(COMPACTION_NOTICE)),
            "the fixpoint still declares that it was compacted",
        );
    }

    #[test]
    fn a_summary_larger_than_what_it_replaces_is_declined() {
        // Short messages summarize to more than themselves, so the notice
        // alone makes the history bigger. Compacting anyway would spend
        // context to lose information — strictly worse than doing nothing.
        let window = window(8);
        let config =
            CompactionConfig::default().with_keep_recent_turns(KeepRecentTurns::new(1).unwrap());
        let messages = chatty_history(6, 4);

        let plan = plan_compaction(&window, &config, &messages)
            .expect("the fixture is over threshold, so the planner should be willing");
        assert!(
            finish_compaction(&window, &plan, SUMMARY).is_none(),
            "but applying it would grow the history, so it must decline",
        );
    }

    #[test]
    fn a_history_with_no_system_turn_compacts_to_the_summary_alone() {
        let window = window(120);
        let messages: Vec<Message> = (0..10)
            .map(|i| {
                if i % 2 == 0 {
                    Message::user(format!("u{i} {}", "x".repeat(2_000)))
                } else {
                    Message::assistant(format!("a{i} {}", "y".repeat(2_000)))
                }
            })
            .collect();
        let plan =
            plan_compaction(&window, &CompactionConfig::default(), &messages).expect("compact");

        let (compacted, _) = finish_compaction(&window, &plan, SUMMARY).expect("should reclaim");

        assert_eq!(compacted[0].role, MessageRole::User);
        assert!(compacted[0].content.starts_with(COMPACTION_NOTICE));
    }

    #[test]
    fn an_empty_summary_still_leaves_the_notice() {
        let window = window(200);
        let messages = chatty_history(6, 2_000);
        let plan =
            plan_compaction(&window, &CompactionConfig::default(), &messages).expect("compact");

        let compacted = apply_compaction(&plan, "   ");

        assert!(compacted.iter().any(|m| m.content == COMPACTION_NOTICE));
    }

    #[test]
    fn a_record_persists_the_exact_message_the_model_saw() {
        let record = CompactionRecord {
            summary: SUMMARY.to_string(),
            outcome: CompactionOutcome {
                messages_before: 10,
                messages_after: 4,
                tokens_before: 1_000,
                tokens_after: 200,
                messages_elided: 7,
            },
            elided_prefix_len: 7,
        };

        assert_eq!(record.as_message(), summary_message(SUMMARY));
    }

    // -- prefix elision and adoption --------------------------------------

    /// A record whose summary stands for the first `n` non-system messages.
    fn record(n: usize) -> CompactionRecord {
        CompactionRecord {
            summary: SUMMARY.to_string(),
            outcome: CompactionOutcome {
                messages_before: 0,
                messages_after: 0,
                tokens_before: 0,
                tokens_after: 0,
                messages_elided: n,
            },
            elided_prefix_len: n,
        }
    }

    #[test]
    fn a_plan_elides_a_strict_prefix_of_the_non_system_history() {
        let history = chatty_history(6, 400);
        let plan = plan_compaction(&window(1_000), &CompactionConfig::default(), &history)
            .expect("over threshold");

        let n = plan.elided_prefix_len();
        assert!(n > 0);
        assert_eq!(n, plan.elided().len());
        assert_eq!(plan.system(), Some(&history[0]));
        assert_eq!(plan.elided(), &history[1..=n]);
        assert_eq!(plan.preserved(), &history[n + 1..]);
    }

    #[test]
    fn a_plan_over_a_history_with_no_system_message_elides_from_index_zero() {
        let history: Vec<Message> = chatty_history(6, 400).into_iter().skip(1).collect();
        let plan = plan_compaction(&window(1_000), &CompactionConfig::default(), &history)
            .expect("over threshold");

        let n = plan.elided_prefix_len();
        assert!(n > 0);
        assert_eq!(plan.system(), None);
        assert_eq!(plan.elided(), &history[..n]);
        assert_eq!(plan.preserved(), &history[n..]);
    }

    #[test]
    fn the_prefix_length_is_what_the_outcome_counts_as_elided() {
        let history = chatty_history(6, 400);
        let plan = plan_compaction(&window(1_000), &CompactionConfig::default(), &history)
            .expect("over threshold");
        let (_, outcome) =
            finish_compaction(&window(1_000), &plan, SUMMARY).expect("summary is smaller");

        assert_eq!(plan.elided_prefix_len(), outcome.messages_elided);
    }

    #[test]
    fn adopting_replaces_the_prefix_and_keeps_the_system_message() {
        let history = vec![
            Message::system("sys"),
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant("a2"),
        ];

        let adopted = record(2).adopt(&history);

        assert_eq!(
            adopted,
            vec![
                Message::system("sys"),
                summary_message(SUMMARY),
                Message::user("u2"),
                Message::assistant("a2"),
            ]
        );
    }

    #[test]
    fn adopting_without_a_system_message_starts_with_the_summary() {
        let history = vec![
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
        ];

        let adopted = record(2).adopt(&history);

        assert_eq!(adopted, vec![summary_message(SUMMARY), Message::user("u2")]);
    }

    #[test]
    fn adopting_a_span_longer_than_the_owned_history_leaves_only_the_summary() {
        let history = vec![Message::user("u1"), Message::assistant("a1")];

        let adopted = record(5).adopt(&history);

        assert_eq!(adopted, vec![summary_message(SUMMARY)]);
    }

    #[test]
    fn adopting_matches_what_the_loop_produced_for_the_owned_prefix() {
        // The embedder owns the exchanges; the loop's list is those plus the
        // turn's own prompt, which the embedder does not keep.
        let owned = chatty_history(6, 400);
        let mut loop_view = owned.clone();
        loop_view.push(Message::user("this turn's prompt"));

        let plan = plan_compaction(&window(1_000), &CompactionConfig::default(), &loop_view)
            .expect("over threshold");
        let (compacted, outcome) =
            finish_compaction(&window(1_000), &plan, SUMMARY).expect("summary is smaller");
        let record = CompactionRecord {
            summary: SUMMARY.to_string(),
            outcome,
            elided_prefix_len: plan.elided_prefix_len(),
        };

        // `adopt` leaves the embedder's messages as they are; the loop's copy
        // has additionally been through the wire repair, which may merge a
        // preserved user turn into the summary message. The next turn sends
        // the adopted history through the same repair, so the two agree on
        // the wire.
        let adopted = sanitize_history(&record.adopt(&owned));
        assert!(adopted.len() <= compacted.len());
        assert_eq!(adopted.as_slice(), &compacted[..adopted.len()]);
    }

    #[test]
    fn two_records_adopt_in_sequence_against_the_previous_result() {
        let history = vec![
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant("a2"),
            Message::user("u3"),
        ];

        // The second pass saw a list opening with the first summary, so its
        // prefix counts that summary as message zero.
        let first = record(2);
        let mut second = record(3);
        second.summary = "later".to_string();
        let adopted = second.adopt(&first.adopt(&history));

        assert_eq!(adopted, vec![summary_message("later"), Message::user("u3")]);
    }
}
