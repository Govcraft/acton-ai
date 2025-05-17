use acton_reactive::prelude::*;

#[acton_actor]
pub struct GuardrailAgent {}

impl GuardrailAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<GuardrailAgent>().await;
        agent.after_start(|agent| {
            println!("GuardrailAgent started");
            println!("Agent Id: {}", agent.id());
            AgentReply::immediate()
        });


        agent.start().await
    }
}