use acton_reactive::prelude::*;
use crate::messages::{ToolExecutionRequestMsg, ToolResultMsg};

#[acton_actor]
pub struct ExecutorAgent {}

impl ExecutorAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<ExecutorAgent>().await;
        agent
            .act_on::<ToolExecutionRequestMsg>(|_agent, ctx| {
                let reply = ctx.reply_envelope();
                let tool_name = ctx.message().tool_name.clone();
                AgentReply::from_async(async move {
                    reply
                        .send(ToolResultMsg { tool_name, result: "done".into() })
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
    use std::sync::{Arc, Mutex};

    #[tokio::test(flavor = "multi_thread")]
    async fn replies_to_tool_request() {
        let mut app = ActonApp::launch();
        let exec = ExecutorAgent::init(&mut app).await;
        // create a test receiver agent to collect results
        #[acton_actor]
        struct Receiver { result: Arc<Mutex<Vec<ToolResultMsg>>> }
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut builder = app.new_agent::<Receiver>().await;
        builder.model.result = results.clone();
        builder.act_on::<ToolResultMsg>(|agent, ctx| {
            let col = agent.model.result.clone();
            let msg = ctx.message().clone();
            AgentReply::from_async(async move { col.lock().unwrap().push(msg); })
        });
        builder.handle().subscribe::<ToolResultMsg>().await;
        let receiver = builder.start().await;

        let mut env = receiver.create_envelope(Some(exec.reply_address()));
        env
            .send(ToolExecutionRequestMsg {
                tool_name: "t".into(),
                arguments: "".into(),
            })
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!results.lock().unwrap().is_empty());
        exec.stop().await.unwrap();
        receiver.stop().await.unwrap();
        app.shutdown_all().await.unwrap();
    }
}
