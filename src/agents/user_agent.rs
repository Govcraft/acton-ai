use acton_reactive::prelude::*;
use crate::messages::UserQueryReceivedEvent;
use mti::prelude::*;

#[acton_actor]
pub struct UserAgent {}

impl UserAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<UserAgent>().await;
        agent.after_start(|agent| {
            let broker = agent.broker().clone();
            AgentReply::from_async(async move {
                broker.broadcast(UserQueryReceivedEvent {
                    query: "Hello".into(),
                    user_id: "u1".into(),
                    conversation_id: "conversation".create_type_id::<V7>(),
                }).await;
            })
        });
        agent.start().await
    }
}
