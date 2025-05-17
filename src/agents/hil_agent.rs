use acton_reactive::prelude::*;

#[acton_actor]
pub struct HilAgent {}

impl HilAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<HilAgent>().await;
        agent.after_start(|agent| {
            println!("HilAgent started");
            println!("Agent Id: {}", agent.id());
            AgentReply::immediate()
        });


        agent.start().await
    }
}