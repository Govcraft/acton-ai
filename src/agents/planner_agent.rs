use acton_reactive::prelude::*;
use crate::messages::{UserQueryReceivedEvent, AiAgentTaskStartedEvent};
use mti::prelude::*;

#[acton_actor]
pub struct PlannerAgent {}

impl PlannerAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<PlannerAgent>().await;
        agent
            .act_on::<UserQueryReceivedEvent>(|_agent, ctx| {
                let convo_id = ctx.message().conversation_id.clone();
                let goal = ctx.message().query.clone();
                let event = AiAgentTaskStartedEvent {
                    agent_id: Ern::default(),
                    conversation_id: convo_id,
                    goal,
                };
                let broker = _agent.broker().clone();
                AgentReply::from_async(async move {
                    broker.broadcast(event).await;
                })
            });
        agent.handle().subscribe::<UserQueryReceivedEvent>().await;
        agent.start().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[tokio::test(flavor = "multi_thread")]
    async fn broadcasts_task_started() {
        let mut app = ActonApp::launch();
        let planner = PlannerAgent::init(&mut app).await;

        #[acton_actor]
        struct Receiver { events: Arc<Mutex<Vec<AiAgentTaskStartedEvent>>> }
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut receiver = app.new_agent::<Receiver>().await;
        receiver.model.events = events.clone();
        receiver.act_on::<AiAgentTaskStartedEvent>(|agent, ctx| {
            let col = agent.model.events.clone();
            let msg = ctx.message().clone();
            AgentReply::from_async(async move { col.lock().unwrap().push(msg); })
        });
        receiver.handle().subscribe::<AiAgentTaskStartedEvent>().await;
        let handle = receiver.start().await;

        planner
            .send(UserQueryReceivedEvent {
                query: "hi".into(),
                user_id: "u".into(),
                conversation_id: "c".create_type_id::<V7>(),
            })
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!events.lock().unwrap().is_empty());

        planner.stop().await.unwrap();
        handle.stop().await.unwrap();
        app.shutdown_all().await.unwrap();
    }
}
