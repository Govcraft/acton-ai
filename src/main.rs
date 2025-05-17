#![allow(warnings)]
//mod messages;
mod agents;
mod config;
mod llm_clients;
mod messages;
mod persistence;
mod prelude;
mod tools;
mod utils;
use acton_reactive::prelude::*;
use agents::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let mut app = ActonApp::launch();
    let memory_agent = MemoryAgent::init(&mut app).await;
    let model_agent = ModelAgent::init(&mut app).await;
    let agent = AIAgent::init(&mut app).await;

    model_agent.stop().await?;
    memory_agent.stop().await?;
    agent.stop().await?;
    app.shutdown_all().await?;

    Ok(())
}
