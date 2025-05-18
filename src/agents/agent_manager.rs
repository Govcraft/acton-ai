use acton_reactive::prelude::*;
use crate::agents::{
    ExecutorAgent, InstrumentationAgent, OrchestratorAgent, PlannerAgent, UserAgent,
};
use crate::agents::tool_agent::ToolAgent;

#[acton_actor]
pub struct AgentManager {}

impl AgentManager {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut builder = app.new_agent::<AgentManager>().await;
        builder.after_start(|manager| {
            let manager_handle = manager.handle().clone();
            let mut runtime = manager.runtime().clone();
            AgentReply::from_async(async move {
                let mut tool_builder = runtime.new_agent::<ToolAgent>().await;
                let tool_handle = manager_handle
                    .supervise(tool_builder)
                    .await
                    .unwrap();

                let mut orch_builder = runtime.new_agent::<OrchestratorAgent>().await;
                orch_builder.model.tool_agent = tool_handle.clone();
                manager_handle.supervise(orch_builder).await.unwrap();

                let planner_builder = runtime.new_agent::<PlannerAgent>().await;
                manager_handle.supervise(planner_builder).await.unwrap();

                let exec_builder = runtime.new_agent::<ExecutorAgent>().await;
                manager_handle.supervise(exec_builder).await.unwrap();

                let instr_builder = runtime.new_agent::<InstrumentationAgent>().await;
                manager_handle.supervise(instr_builder).await.unwrap();

                let user_builder = runtime.new_agent::<UserAgent>().await;
                manager_handle.supervise(user_builder).await.unwrap();
            })
        });
        builder.start().await
    }
}
