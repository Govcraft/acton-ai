use acton_reactive::prelude::*;
use crate::messages::{AiAgentTaskStartedEvent, ToolExecutedEvent};
use tracing::info;

#[acton_actor]
pub struct InstrumentationAgent {}

impl InstrumentationAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<InstrumentationAgent>().await;
        agent
            .act_on::<AiAgentTaskStartedEvent>(|_agent, ctx| {
                let goal = ctx.message().goal.clone();
                AgentReply::from_async(async move {
                    info!(goal = %goal, "task_started");
                })
            })
            .act_on::<ToolExecutedEvent>(|_agent, ctx| {
                let name = ctx.message().tool_name.clone();
                AgentReply::from_async(async move {
                    info!(tool = %name, "tool_executed");
                })
            });
        agent.handle().subscribe::<AiAgentTaskStartedEvent>().await;
        agent.handle().subscribe::<ToolExecutedEvent>().await;
        agent.start().await
    }
}
