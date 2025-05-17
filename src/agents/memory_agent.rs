use crate::messages::{MemoryRetrieveRequestMsg, MemoryStoreRequestMsg, MemoryRetrievedMsg};
use crate::persistence::impls::in_memory::InMemory;
use acton_reactive::prelude::*;
use crate::persistence::memory_store::MemoryStore;

#[acton_actor]
pub struct MemoryAgent {
    pub persistence: InMemory,
}

impl MemoryAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        Self::init_with_store(app, InMemory::default()).await
    }

    pub async fn init_with_store(app: &mut AgentRuntime, persistence: InMemory) -> AgentHandle {
        let mut agent = app.new_agent::<MemoryAgent>().await;
        agent.model.persistence = persistence;

        agent
            .after_start(|_agent| {
                AgentReply::immediate()
            })
            .act_on::<MemoryStoreRequestMsg>(|agent, context| {
                let conversation = context.message().conversation.clone();
                agent.model.persistence.store(conversation);
                AgentReply::immediate()
            })
            .act_on::<MemoryRetrieveRequestMsg>(|agent, context| {
                let conv = context.message().conversation.clone();
                let store = agent.model.persistence.clone();
                let broker = agent.broker().clone();
                AgentReply::from_async(async move {
                    let history = if store.get_by_id(conv.id()).is_some() {
                        vec![conv.id().to_string()]
                    } else {
                        Vec::new()
                    };
                    broker.broadcast(MemoryRetrievedMsg { history }).await;
                })
            });
        agent.handle().subscribe::<MemoryStoreRequestMsg>().await;
        agent.handle().subscribe::<MemoryRetrieveRequestMsg>().await;

        agent.start().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::common::Conversation;
    use std::sync::{Arc, Mutex};

    #[tokio::test(flavor = "multi_thread")]
    async fn stores_conversation_in_memory() {
        let mut app = ActonApp::launch();
        let store = InMemory::default();
        let shared = store.clone();
        let handle = MemoryAgent::init_with_store(&mut app, store).await;
        let convo = Conversation::default();
        handle
            .send(MemoryStoreRequestMsg {
                conversation: convo.clone(),
            })
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(shared.get_by_id(convo.id()).is_some());
        handle.stop().await.unwrap();
        app.shutdown_all().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retrieves_conversation_broadcast() {
        let mut app = ActonApp::launch();
        let store = InMemory::default();
        let handle = MemoryAgent::init_with_store(&mut app, store.clone()).await;

        let received: Arc<Mutex<Vec<MemoryRetrievedMsg>>> = Arc::new(Mutex::new(Vec::new()));
        let mut receiver = app.new_agent::<TestReceiver>().await;
        receiver.model.received = received.clone();
        receiver
            .act_on::<MemoryRetrievedMsg>(move |agent, context| {
                let msg = context.message().clone();
                let col = agent.model.received.clone();
                AgentReply::from_async(async move {
                    col.lock().unwrap().push(msg);
                })
            });
        receiver.handle().subscribe::<MemoryRetrievedMsg>().await;
        let receiver_handle = receiver.start().await;

        let convo = Conversation::default();
        handle
            .send(MemoryStoreRequestMsg {
                conversation: convo.clone(),
            })
            .await;
        handle
            .send(MemoryRetrieveRequestMsg { conversation: convo.clone() })
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let history = received.lock().unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].history, vec![convo.id().to_string()]);

        handle.stop().await.unwrap();
        receiver_handle.stop().await.unwrap();
        app.shutdown_all().await.unwrap();
    }

    #[acton_actor]
    struct TestReceiver {
        received: Arc<Mutex<Vec<MemoryRetrievedMsg>>>,
    }
}
