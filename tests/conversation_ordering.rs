//! Conversation mutators must order against later calls.
//!
//! `clear`, `set_system_prompt`, and `clear_system_prompt` used to hand their
//! `send` to a detached task, so the message could still be sitting in a
//! `tokio::spawn` queue when the caller's next `send` was already on its way to
//! the actor. Awaiting the send instead puts them in the mailbox before the
//! call returns, and FIFO delivery does the rest.
//!
//! No LLM is contacted here: every operation under test is local history and
//! system-prompt bookkeeping owned by the conversation actor.

use acton_ai::prelude::*;

/// Builds a conversation against a provider that is configured but never called.
async fn conversation_with_prompt(system_prompt: &str) -> (ActonAI, Conversation) {
    let runtime = ActonAI::builder()
        .app_name("conversation-ordering-test")
        .ollama("test-model")
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let conversation = runtime.conversation().system(system_prompt).build().await;

    (runtime, conversation)
}

#[tokio::test]
async fn set_system_prompt_is_visible_after_sync() {
    let (runtime, conv) = conversation_with_prompt("original").await;
    assert_eq!(conv.system_prompt(), Some("original".to_string()));

    conv.set_system_prompt("replacement").await;
    conv.sync().await.expect("sync must succeed");

    assert_eq!(conv.system_prompt(), Some("replacement".to_string()));

    runtime
        .runtime()
        .clone()
        .shutdown_all()
        .await
        .expect("shutdown failed");
}

#[tokio::test]
async fn clear_system_prompt_is_visible_after_sync() {
    let (runtime, conv) = conversation_with_prompt("original").await;

    conv.clear_system_prompt().await;
    conv.sync().await.expect("sync must succeed");

    assert_eq!(conv.system_prompt(), None);

    runtime
        .runtime()
        .clone()
        .shutdown_all()
        .await
        .expect("shutdown failed");
}

#[tokio::test]
async fn the_last_system_prompt_written_wins() {
    let (runtime, conv) = conversation_with_prompt("original").await;

    // Three mutations issued back to back. Detached sends could land in any
    // order, leaving any of the three in place; awaited sends cannot.
    conv.set_system_prompt("first").await;
    conv.set_system_prompt("second").await;
    conv.set_system_prompt("third").await;
    conv.sync().await.expect("sync must succeed");

    assert_eq!(conv.system_prompt(), Some("third".to_string()));

    runtime
        .runtime()
        .clone()
        .shutdown_all()
        .await
        .expect("shutdown failed");
}

#[tokio::test]
async fn clear_empties_history_seeded_at_build_time() {
    let runtime = ActonAI::builder()
        .app_name("conversation-ordering-test")
        .ollama("test-model")
        .launch()
        .await
        .expect("launching the runtime must succeed");

    let conv = runtime
        .conversation()
        .restore(vec![
            Message::user("earlier question"),
            Message::assistant("earlier answer"),
        ])
        .build()
        .await;

    assert_eq!(conv.len(), 2);

    conv.clear().await;
    conv.sync().await.expect("sync must succeed");

    assert!(
        conv.is_empty(),
        "history must be empty after a cleared sync"
    );
    assert!(conv.history().is_empty());

    runtime
        .runtime()
        .clone()
        .shutdown_all()
        .await
        .expect("shutdown failed");
}

#[tokio::test]
async fn sync_is_idempotent() {
    let (runtime, conv) = conversation_with_prompt("original").await;

    conv.sync().await.expect("first sync must succeed");
    conv.sync().await.expect("second sync must succeed");

    runtime
        .runtime()
        .clone()
        .shutdown_all()
        .await
        .expect("shutdown failed");
}
