use crate::agents::common::Conversation;
use crate::messages::{AiAgentTaskStartedEvent, MemoryStoreRequestMsg, ModelQueryRequestMsg};
use acton_reactive::prelude::*;
use std::error::Error;

#[derive(Debug)]
pub struct AIAgent {
    conversation: Conversation,
    purpose: String,
}

impl Default for AIAgent {
    fn default() -> Self {
        AIAgent {
            conversation: Conversation::default(),
            purpose: String::from("You are a helpful assistant"),
        }
    }
}

impl AIAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<AIAgent>().await;
        agent.after_start(|agent| {
            let broker = agent.broker().clone();
            // println!("AI Agent started");
            // println!("Agent Id: {}", agent.id());
            // println!("Conversation Id: {}", agent.model.conversation.id());
            let conversation = agent.model.conversation.clone();
            AgentReply::from_async(async move {
                broker
                    .broadcast(ModelQueryRequestMsg { prompt: "What's the meaning of life?".into(), history: conversation })
                    .await;
            })
        });

        agent.start().await
    }
}
