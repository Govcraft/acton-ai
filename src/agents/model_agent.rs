use crate::messages::ModelQueryRequestMsg;
use acton_reactive::prelude::*;
use rig::completion::AssistantContent::{Text, ToolCall};
use rig::completion::CompletionModel;
use rig::{completion::Prompt, providers::groq};

#[acton_actor]
pub struct ModelAgent {}

impl ModelAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let mut agent = app.new_agent::<ModelAgent>().await;
        agent
            .after_start(|agent| {
                println!("ModelAgent started");
                println!("Agent Id: {}", agent.id());
                AgentReply::immediate()
            })
            .act_on::<ModelQueryRequestMsg>(|agent, context| {
                // This code runs when the agent receives a CompletionRequest.
                // We can safely mutate the agent's internal state (`model`).
                let query = context.message().prompt.clone();
                // Return AgentReply accordingly.
                AgentReply::from_async(async move {
                    let client = groq::Client::from_env();
                    let model = client.completion_model("llama-3.1-8b-instant");
                    let request = model.completion_request(query).build();
                    let response = model
                        .completion(request)
                        .await
                        .expect("Failed to complete request")
                        .choice
                        .first();
                    match response {
                        Text(content) => {
                            println!("response: {}", content.text);
                        }
                        ToolCall(_tool) => ()
                    }
                })
            });

        agent.handle().subscribe::<ModelQueryRequestMsg>().await;

        agent.start().await
    }
}
