//! Structural repair of a message history before it goes on the wire.
//!
//! Every provider enforces invariants that a model response can violate:
//! a `tool_use` block needs a matching `tool_result`, a history must open
//! with a user turn, roles must alternate, and no message may be empty. A
//! history that breaks one of these is rejected for the rest of the
//! conversation, not just for the request that introduced it — the bad
//! message is replayed on every subsequent turn. That makes a single
//! malformed response unrecoverable without dropping the whole history.
//!
//! This module is the last gate before serialization. It is provider
//! agnostic: it repairs the *structure* of a `Message` list. Wire-format
//! concerns (how a tool result is spelled, whether adjacent turns coalesce
//! into one request message) belong to the individual clients.
//!
//! Every pass is a pure `Vec<Message> -> Vec<Message>` function, so each is
//! testable in isolation and the pipeline order is explicit.

use crate::messages::{Message, MessageRole};
use std::collections::HashSet;

/// Stand-in for a tool result that came back with no output.
///
/// A tool that legitimately returns nothing still needs its result block:
/// dropping the message would orphan the `tool_use` that asked for it, which
/// is the very failure this module exists to prevent. Providers reject an
/// empty result block, so the emptiness is made explicit instead.
const EMPTY_TOOL_RESULT: &str = "(no output)";

/// Repairs a history into a shape every supported provider accepts.
///
/// The passes run in a deliberate order — see the comments inline. Each
/// pass assumes its predecessors have already run.
///
/// This never invents content. It removes what cannot be sent and merges
/// what must not be adjacent; a history that is already well formed comes
/// back unchanged.
pub(crate) fn sanitize_history(messages: &[Message]) -> Vec<Message> {
    // Unidentified results first: every later pass keys off `tool_call_id`,
    // and a result with no id can never be matched to anything.
    let stage = drop_unidentified_tool_results(messages.to_vec());
    // Before trimming, so an empty leading user turn cannot be mistaken for
    // the user turn the history is supposed to open with.
    let stage = normalize_empty(stage);
    let stage = trim_to_first_user(stage);
    // After trimming, which may have removed the assistant turn that owned
    // a surviving result.
    let stage = drop_orphan_tool_results(stage);
    let stage = strip_unanswered_tool_calls(stage);
    // Again: stripping every call off an assistant turn can leave it empty.
    let stage = normalize_empty(stage);
    merge_adjacent_same_role(stage)
}

/// Drops tool results carrying no `tool_call_id`.
///
/// Such a message cannot be addressed to any `tool_use` block, so a client
/// has nothing to serialize it as.
fn drop_unidentified_tool_results(mut messages: Vec<Message>) -> Vec<Message> {
    messages.retain(|msg| msg.role != MessageRole::Tool || msg.tool_call_id.is_some());
    messages
}

/// Drops messages with nothing to say, and fills in empty tool results.
///
/// An assistant turn that carries tool calls is kept whatever its text: the
/// calls are the payload, and the clients omit the empty text block.
fn normalize_empty(mut messages: Vec<Message>) -> Vec<Message> {
    for msg in &mut messages {
        if msg.role == MessageRole::Tool && msg.content.trim().is_empty() {
            msg.content = EMPTY_TOOL_RESULT.to_string();
        }
    }
    messages.retain(|msg| {
        msg.role == MessageRole::Tool
            || msg.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
            || !msg.content.trim().is_empty()
    });
    messages
}

/// Drops leading turns so the first non-system message is from the user.
///
/// Providers require a history to open with a user turn. Truncation is the
/// usual way to lose one: a token-greedy window keeps the tail, and the tail
/// often starts mid-exchange.
///
/// A history with no user turn at all is returned untouched — that is a
/// caller error the provider reports far more clearly than a silent repair.
fn trim_to_first_user(messages: Vec<Message>) -> Vec<Message> {
    let Some(first_user) = messages
        .iter()
        .position(|msg| msg.role == MessageRole::User)
    else {
        return messages;
    };

    messages
        .into_iter()
        .enumerate()
        .filter(|(index, msg)| *index >= first_user || msg.role == MessageRole::System)
        .map(|(_, msg)| msg)
        .collect()
}

/// Drops tool results that answer no preceding tool call.
///
/// An id is consumed the first time it is answered, so a duplicated result
/// is dropped along with a genuinely orphaned one. Both are rejected by the
/// providers for the same reason: the id does not name an outstanding call.
fn drop_orphan_tool_results(messages: Vec<Message>) -> Vec<Message> {
    let mut outstanding: HashSet<String> = HashSet::new();

    messages
        .into_iter()
        .filter(|msg| match msg.role {
            MessageRole::Assistant => {
                if let Some(calls) = &msg.tool_calls {
                    outstanding.extend(calls.iter().map(|call| call.id.clone()));
                }
                true
            }
            MessageRole::Tool => msg
                .tool_call_id
                .as_ref()
                .is_some_and(|id| outstanding.remove(id)),
            _ => true,
        })
        .collect()
}

/// Removes tool calls that no later message answers.
///
/// A `tool_use` with no `tool_result` is the single most common way a
/// history becomes permanently unsendable: the turn aborts between issuing
/// the call and recording its result, and the dangling call is replayed
/// forever after.
fn strip_unanswered_tool_calls(mut messages: Vec<Message>) -> Vec<Message> {
    for index in 0..messages.len() {
        if messages[index].role != MessageRole::Assistant || messages[index].tool_calls.is_none() {
            continue;
        }

        // Only results *after* this turn can answer it. Matching against the
        // whole history would let a recycled id from an earlier exchange
        // vouch for a call that never ran.
        let answered: HashSet<&str> = messages[index + 1..]
            .iter()
            .filter(|msg| msg.role == MessageRole::Tool)
            .filter_map(|msg| msg.tool_call_id.as_deref())
            .collect();

        let retained: Vec<_> = messages[index]
            .tool_calls
            .iter()
            .flatten()
            .filter(|call| answered.contains(call.id.as_str()))
            .cloned()
            .collect();

        messages[index].tool_calls = if retained.is_empty() {
            None
        } else {
            Some(retained)
        };
    }
    messages
}

/// Merges consecutive turns that share a role.
///
/// Providers that require strict alternation reject the run outright; the
/// rest quietly reinterpret it. Merging is lossless — the text is joined —
/// and keeps the two behaviours identical.
///
/// Tool results are never merged here. They are separate messages by
/// construction, and how they coalesce is a wire-format decision each client
/// makes for itself.
fn merge_adjacent_same_role(messages: Vec<Message>) -> Vec<Message> {
    messages.into_iter().fold(Vec::new(), |mut acc, msg| {
        let mergeable = acc.last().is_some_and(|last: &Message| {
            last.role == msg.role
                && msg.role != MessageRole::Tool
                // An assistant turn holding calls is followed by their
                // results, so it is never legitimately adjacent to another
                // assistant turn. Leave the shape alone rather than invent
                // an interleaving.
                && last.tool_calls.is_none()
        });

        if mergeable {
            let Some(last) = acc.last_mut() else {
                unreachable!("mergeable implies a last element")
            };
            if !last.content.is_empty() && !msg.content.is_empty() {
                last.content.push_str("\n\n");
            }
            last.content.push_str(&msg.content);
            last.tool_calls = msg.tool_calls;
        } else {
            acc.push(msg);
        }
        acc
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::ToolCall;

    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({}),
        }
    }

    #[test]
    fn well_formed_history_is_unchanged() {
        let messages = vec![
            Message::system("be helpful"),
            Message::user("search for rust"),
            Message::assistant_with_tools("on it", vec![call("tc_1")]),
            Message::tool("tc_1", "results"),
            Message::assistant("here you go"),
        ];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized.len(), 5);
        assert_eq!(sanitized[2].tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn dangling_tool_call_is_stripped() {
        let messages = vec![
            Message::user("search for rust"),
            Message::assistant_with_tools("on it", vec![call("tc_1")]),
        ];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized.len(), 2);
        assert!(sanitized[1].tool_calls.is_none());
    }

    #[test]
    fn assistant_left_empty_by_stripping_is_dropped() {
        let messages = vec![
            Message::user("search for rust"),
            Message::assistant_with_tools("", vec![call("tc_1")]),
        ];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized.len(), 1);
        assert_eq!(sanitized[0].role, MessageRole::User);
    }

    #[test]
    fn only_the_unanswered_call_is_stripped() {
        let messages = vec![
            Message::user("search"),
            Message::assistant_with_tools("on it", vec![call("tc_1"), call("tc_2")]),
            Message::tool("tc_2", "results"),
        ];

        let sanitized = sanitize_history(&messages);

        let retained = sanitized[1].tool_calls.as_ref().unwrap();
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].id, "tc_2");
    }

    #[test]
    fn orphan_tool_result_is_dropped() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant("hello"),
            Message::tool("tc_missing", "results"),
        ];

        let sanitized = sanitize_history(&messages);

        assert!(sanitized.iter().all(|msg| msg.role != MessageRole::Tool));
    }

    #[test]
    fn duplicate_tool_result_is_dropped() {
        let messages = vec![
            Message::user("search"),
            Message::assistant_with_tools("on it", vec![call("tc_1")]),
            Message::tool("tc_1", "first"),
            Message::tool("tc_1", "second"),
        ];

        let sanitized = sanitize_history(&messages);

        let results: Vec<_> = sanitized
            .iter()
            .filter(|msg| msg.role == MessageRole::Tool)
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "first");
    }

    #[test]
    fn tool_result_without_id_is_dropped() {
        let orphan = Message {
            role: MessageRole::Tool,
            content: "results".to_string(),
            tool_calls: None,
            tool_call_id: None,
        };
        let messages = vec![
            Message::user("search"),
            Message::assistant_with_tools("on it", vec![call("tc_1")]),
            orphan,
        ];

        let sanitized = sanitize_history(&messages);

        assert!(sanitized.iter().all(|msg| msg.role != MessageRole::Tool));
        assert!(sanitized[1].tool_calls.is_none());
    }

    #[test]
    fn empty_assistant_turn_is_dropped() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant(""),
            Message::user("still there?"),
        ];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized.len(), 1);
        assert_eq!(sanitized[0].content, "hi\n\nstill there?");
    }

    #[test]
    fn empty_tool_result_gets_a_placeholder() {
        let messages = vec![
            Message::user("search"),
            Message::assistant_with_tools("on it", vec![call("tc_1")]),
            Message::tool("tc_1", ""),
        ];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized[2].content, EMPTY_TOOL_RESULT);
        assert!(sanitized[1].tool_calls.is_some());
    }

    #[test]
    fn history_is_trimmed_to_open_on_a_user_turn() {
        let messages = vec![
            Message::system("be helpful"),
            Message::assistant("mid-exchange"),
            Message::user("a question"),
        ];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized.len(), 2);
        assert_eq!(sanitized[0].role, MessageRole::System);
        assert_eq!(sanitized[1].role, MessageRole::User);
    }

    #[test]
    fn trimming_a_lead_assistant_also_drops_its_results() {
        let messages = vec![
            Message::assistant_with_tools("on it", vec![call("tc_1")]),
            Message::tool("tc_1", "results"),
            Message::user("a question"),
        ];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized.len(), 1);
        assert_eq!(sanitized[0].role, MessageRole::User);
    }

    #[test]
    fn history_without_a_user_turn_is_left_for_the_provider_to_reject() {
        let messages = vec![Message::system("be helpful"), Message::assistant("hello")];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized.len(), 2);
    }

    #[test]
    fn consecutive_user_turns_are_merged() {
        let messages = vec![
            Message::user("extract the data"),
            Message::user("record your answer"),
        ];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized.len(), 1);
        assert_eq!(
            sanitized[0].content,
            "extract the data\n\nrecord your answer"
        );
    }

    #[test]
    fn tool_results_are_not_merged_with_each_other() {
        let messages = vec![
            Message::user("search"),
            Message::assistant_with_tools("on it", vec![call("tc_1"), call("tc_2")]),
            Message::tool("tc_1", "first"),
            Message::tool("tc_2", "second"),
        ];

        let sanitized = sanitize_history(&messages);

        let results: Vec<_> = sanitized
            .iter()
            .filter(|msg| msg.role == MessageRole::Tool)
            .collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn an_assistant_turn_with_calls_is_never_merged_away() {
        let messages = vec![
            Message::user("search"),
            Message::assistant_with_tools("on it", vec![call("tc_1")]),
            Message::tool("tc_1", "results"),
            Message::assistant("done"),
        ];

        let sanitized = sanitize_history(&messages);

        assert_eq!(sanitized.len(), 4);
    }

    #[test]
    fn sanitizing_is_idempotent() {
        let messages = vec![
            Message::user("search"),
            Message::assistant_with_tools("", vec![call("tc_1")]),
            Message::user("never mind"),
            Message::user("actually, do it"),
        ];

        let once = sanitize_history(&messages);
        let twice = sanitize_history(&once);

        assert_eq!(once, twice);
    }
}
