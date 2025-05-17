use crate::messages::{MemoryRetrieveRequestMsg, MemoryStoreRequestMsg};
use crate::persistence::impls::in_memory::InMemory;
use acton_reactive::prelude::*;
use crate::persistence::memory_store;
use crate::persistence::memory_store::MemoryStore;

#[acton_actor]
pub struct MemoryAgent {
    pub persistence: InMemory,
}

impl MemoryAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<MemoryAgent>().await;

        agent
            .after_start(|agent| {
                // println!("MemoryAgent started");
                // println!("Agent Id: {}", agent.id());
                AgentReply::immediate()
            })
            .act_on::<MemoryStoreRequestMsg>(|agent, context| {
                // This code runs when the agent receives a MemoryStoreRequestMsg.
                // We can safely mutate the agent's internal state (`model`).
                let conversation = context.message().conversation.clone();
                let mut memory = agent.model.persistence.clone();
                // add to the memory hashmap

                // Return AgentReply accordingly.
                AgentReply::from_async(async move {
                    // println!("MemoryAgent received a MemoryStoreRequestMsg");
                    memory.store(conversation);
                    
                })
            })
            .act_on::<MemoryRetrieveRequestMsg>(|agent, context| {
                // This code runs when the agent receives a MemoryRetrieveRequestMsg.
                // We can safely mutate the agent's internal state (`model`).

                // Return AgentReply accordingly.
                AgentReply::from_async(async move {})
            });
        agent.handle().subscribe::<MemoryStoreRequestMsg>().await;

        agent.start().await
    }
}
