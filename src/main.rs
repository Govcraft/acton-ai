#![allow(dead_code, unused_imports, unused_variables, warnings)]
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
    let manager = AgentManager::init(&mut app).await;

    manager.stop().await?;
    app.shutdown_all().await?;

    Ok(())
}
