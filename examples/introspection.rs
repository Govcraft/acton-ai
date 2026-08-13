//! Example: Live Introspection
//!
//! A long-running agent process that can be asked what it is doing, and told
//! to stop taking new work, from another terminal.
//!
//! # Setup
//!
//! ```bash
//! ollama serve
//! ollama pull qwen2.5:7b
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example introspection
//! ```
//!
//! It prints its socket path and then loops, prompting once a second. From a
//! second terminal:
//!
//! ```bash
//! acton-ai status --socket /tmp/acton-ai-introspection-example.sock
//! acton-ai status --socket /tmp/acton-ai-introspection-example.sock --json | jq .active_turns
//! acton-ai pause  --socket /tmp/acton-ai-introspection-example.sock
//! acton-ai resume --socket /tmp/acton-ai-introspection-example.sock
//! acton-ai drain  --socket /tmp/acton-ai-introspection-example.sock --wait
//! ```
//!
//! While paused, the loop's prompts come back refused rather than hanging or
//! failing generically, and the process keeps answering `status` throughout —
//! which is the point: a wedged provider or an exhausted budget does not stop
//! you finding out what is wrong.
//!
//! Press Ctrl-C to stop, or send `SIGTERM` (`kill <pid>`) to watch
//! `drain_on_sigterm` close admission and let the turn in flight finish.
//!
//! # Under systemd
//!
//! A fixed `socket_path` is what makes an `ExecStop` possible, since nothing
//! outside the process can guess the PID in the default socket name.
//!
//! ```ini
//! [Unit]
//! Description=My acton-ai agent
//! After=network-online.target
//!
//! [Service]
//! Type=notify
//! ExecStart=/usr/local/bin/my-agent
//! # Ask the process to drain, and wait for the last turn to finish. The
//! # command returns as soon as the drain completes, so TimeoutStopSec is a
//! # backstop rather than a delay you pay every restart.
//! ExecStop=/usr/local/bin/acton-ai drain --socket /run/my-agent/control.sock --wait --timeout 0
//! # Longer than your longest turn. systemd sends SIGKILL when it expires,
//! # and a turn killed halfway may have already run a tool that changed the
//! # world, so err generously.
//! TimeoutStopSec=300
//! Restart=on-failure
//! RuntimeDirectory=my-agent
//! RuntimeDirectoryMode=0700
//!
//! [Install]
//! WantedBy=multi-user.target
//! ```
//!
//! `Type=notify` holds `systemctl start` open until the process reports
//! `READY=1`, which acton-ai sends once providers are resolved, tools are
//! registered, and MCP servers are connected. "Started" then means "actually
//! serving" rather than "forked", so anything ordered `After=` this unit can
//! rely on it.

use acton_ai::prelude::*;
use std::time::Duration;

/// A fixed path, so the commands in the docs above can be copied verbatim.
/// Real deployments put this under `RuntimeDirectory` (`/run/<unit>/`), which
/// systemd creates and removes with the service.
const SOCKET: &str = "/tmp/acton-ai-introspection-example.sock";

#[tokio::main]
async fn main() -> Result<(), ActonAIError> {
    tracing_subscriber::fmt()
        .with_env_filter("acton_ai=info")
        .init();

    let ai = ActonAI::builder()
        .app_name("introspection-example")
        .ollama("qwen2.5:7b")
        // Compiling the `ipc` feature in is not enough on its own: nothing
        // listens until this is called, or an `[introspection]` section
        // appears in the config file.
        .introspection_at(SOCKET)
        // SIGTERM stops new turns rather than killing the one in flight. Not
        // implied by the line above: an embedder with its own signal handling
        // should not have a library installing one behind its back.
        .drain_on_sigterm()
        .launch()
        .await?;

    // Tells systemd startup is complete, under `Type=notify`. A no-op when
    // nothing set `$NOTIFY_SOCKET`, which is the case when you run this by
    // hand.
    acton_ai::introspection::sd_notify::notify_ready();

    let socket = ai
        .introspection_socket()
        .expect("introspection was configured above");
    println!("listening on {}", socket.display());
    println!("try:  acton-ai status --socket {}", socket.display());

    let questions = [
        "Name one thing that is blue.",
        "Name one thing that is round.",
        "Name one thing that is loud.",
    ];

    for (turn, question) in questions.iter().cycle().enumerate() {
        // Draining is a state the loop must read, not an error it discovers:
        // once admission closes there is nothing left to do but stop.
        if !ai.admission_state().admits() {
            println!("no longer admitting turns; stopping the loop");
            break;
        }

        match ai.prompt(*question).collect().await {
            Ok(response) => println!("[{turn}] {}", response.text.trim()),
            // The distinguishable refusal: a caller can tell "you paused me"
            // from "the provider is down" and react differently. Here it means
            // a `pause` landed between the check above and the send, so the
            // loop simply waits for a `resume`.
            Err(error) if error.is_turns_not_admitted() => {
                println!("[{turn}] refused: {error}");
            }
            Err(error) => return Err(error),
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    // Removes the socket file as well as stopping the listener, so the next
    // run does not have to work out whether a leftover address is live.
    ai.shutdown().await?;
    Ok(())
}
