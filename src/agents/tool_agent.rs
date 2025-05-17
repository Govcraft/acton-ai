use acton_reactive::prelude::*;

#[acton_actor]
pub struct ToolAgent {}

impl ToolAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<ToolAgent>().await;
        agent.after_start(|agent| {
            println!("ToolAgent started");
            println!("Agent Id: {}", agent.id());
            AgentReply::immediate()
        });


        agent.start().await
    }
}