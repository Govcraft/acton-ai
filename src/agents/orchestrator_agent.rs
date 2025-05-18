use acton_reactive::prelude::*;
use crate::messages::{ToolExecutionRequestMsg, ToolResultMsg};

#[acton_actor]
pub struct OrchestratorAgent {
    pub tool_agent: AgentHandle,
}

impl OrchestratorAgent {
    pub async fn init(app: &mut AgentRuntime, tool_agent: AgentHandle) -> AgentHandle {
        let mut agent = app.new_agent::<OrchestratorAgent>().await;
        agent.model.tool_agent = tool_agent.clone();

        agent
            .act_on::<ToolExecutionRequestMsg>(move |_agent, ctx| {
                let request = ctx.message().clone();
                let forward_envelope = ctx.new_envelope(&tool_agent.reply_address());
                let reply_envelope = ctx.reply_envelope();
                AgentReply::from_async(async move {
                    forward_envelope.send(request).await;
                    // In a real implementation we would wait for the ToolAgent to reply back.
                    reply_envelope
                        .send(ToolResultMsg { tool_name: "mock".into(), result: "ok".into() })
                        .await;
                })
            });

        agent.handle().subscribe::<ToolExecutionRequestMsg>().await;
        agent.start().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::tool_agent::ToolAgent;
    use std::sync::{Arc, Mutex};

    #[tokio::test(flavor = "multi_thread")]
    async fn forwards_and_replies() {
        let mut app = ActonApp::launch();
        let tool = ToolAgent::init(&mut app).await;
        let orchestrator = OrchestratorAgent::init(&mut app, tool.clone()).await;

        #[acton_actor]
        struct Receiver { result: Arc<Mutex<Vec<ToolResultMsg>>> }
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut receiver = app.new_agent::<Receiver>().await;
        receiver.model.result = results.clone();
        receiver.act_on::<ToolResultMsg>(|agent, ctx| {
            let col = agent.model.result.clone();
            let msg = ctx.message().clone();
            AgentReply::from_async(async move { col.lock().unwrap().push(msg); })
        });
        let handle = receiver.start().await;

        let mut env = handle.create_envelope(Some(orchestrator.reply_address()));
        env.send(ToolExecutionRequestMsg { tool_name: "t".into(), arguments: "".into() }).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!results.lock().unwrap().is_empty());

        orchestrator.stop().await.unwrap();
        tool.stop().await.unwrap();
        handle.stop().await.unwrap();
        app.shutdown_all().await.unwrap();
    }
}
