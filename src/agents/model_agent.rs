use crate::messages::ModelQueryRequestMsg;
use acton_reactive::prelude::*;
use rig::completion::AssistantContent::{Text, ToolCall};
use rig::completion::CompletionModel;
use rig::{completion::Prompt, providers::groq};
use std::sync::Arc;

#[acton_actor]
pub struct ModelAgent {}

impl ModelAgent {
    pub async fn init(app: &mut AgentRuntime) -> AgentHandle {
        let client = groq::Client::from_env();
        let model = client.completion_model("llama-3.1-8b-instant");
        Self::init_with_model(app, model).await
    }

    pub async fn init_with_model<M>(app: &mut AgentRuntime, model: M) -> AgentHandle
    where
        M: CompletionModel + Clone + Send + Sync + 'static,
    {
        let model = Arc::new(model);
        let mut agent = app.new_agent::<ModelAgent>().await;
        agent
            .after_start(|agent| {
                println!("ModelAgent started");
                println!("Agent Id: {}", agent.id());
                AgentReply::immediate()
            })
            .act_on::<ModelQueryRequestMsg>(move |_agent, context| {
                let query = context.message().prompt.clone();
                let model = model.clone();
                AgentReply::from_async(async move {
                    tokio::spawn(async move {
                        let request = model.completion_request(query).build();
                        if let Ok(response) = model.completion(request).await {
                            let choice = response.choice.first();
                            match choice {
                                Text(content) => println!("response: {}", content.text),
                                ToolCall(_tool) => (),
                            }
                        }
                    });
                })
            });

        agent.handle().subscribe::<ModelQueryRequestMsg>().await;

        agent.start().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::completion::{AssistantContent, CompletionError, CompletionRequest, CompletionResponse};
    use rig::completion::message::Text;
    use rig::OneOrMany;
    use crate::agents::common::Conversation;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

    #[derive(Clone)]
    struct MockCompletionModel {
        calls: Arc<AtomicUsize>,
    }

    impl MockCompletionModel {
        fn new() -> Self {
            Self { calls: Arc::new(AtomicUsize::new(0)) }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl CompletionModel for MockCompletionModel {
        type Response = ();

        fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse<Self::Response>, CompletionError>> + Send>> {
            let calls = self.calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(CompletionResponse {
                    choice: OneOrMany::one(AssistantContent::Text(Text { text: "mock".into() })),
                    raw_response: (),
                })
            })
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn model_agent_uses_completion_model() {
        let mut app = ActonApp::launch();
        let model = MockCompletionModel::new();
        let tracker = model.clone();
        let handle = ModelAgent::init_with_model(&mut app, model).await;
        handle
            .send(ModelQueryRequestMsg { prompt: "hi".into(), history: Conversation::default() })
            .await;
        // give the agent some time to process the message
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(tracker.call_count(), 1);
        handle.stop().await.unwrap();
        app.shutdown_all().await.unwrap();
    }
}

