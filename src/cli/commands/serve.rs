//! The `serve` command — a resident process that runs the heartbeat on ticks.
//!
//! `acton-ai heartbeat` is a whole process per wake-up: launch, run what is
//! due, exit. `serve` is the long-lived alternative. It launches once, listens
//! on its introspection socket, and runs one heartbeat pass per tick, where a
//! tick is `acton-ai tick` — normally fired by a systemd timer — arriving over
//! that socket:
//!
//! ```text
//! acton-ai-tick.timer ──► acton-ai tick ──► socket ──► clock ──► serve: heartbeat pass at tick.at
//! ```
//!
//! # What a pass acts on
//!
//! Each pass runs against the *tick's* time, not the clock at the moment the
//! pass starts: what counts as due, and what each entry's `last_run` records,
//! are both the tick's `at`. A tick that waited behind a long pass therefore
//! does what it would have done on time.
//!
//! # Ticks that arrive during a pass
//!
//! Coalesced. When a pass finishes and several ticks are waiting, only the
//! latest runs: a pass at time *T* already runs everything due by *T*, so the
//! earlier ticks would find nothing the latest one does not. Every tick is
//! still in the journal; coalescing decides which ones start work.
//!
//! # Admission
//!
//! A tick is recorded whatever the admission state. While paused, a pass
//! defers every entry (the report says how many), and the next tick after
//! `acton-ai resume` picks them up. Once draining — `acton-ai drain` or
//! `SIGTERM` — the loop finishes the pass in progress, starts no more, and
//! exits.

use std::path::PathBuf;
use std::time::Duration;

use acton_reactive::prelude::tokio;

use crate::cli::commands::heartbeat;
use crate::cli::error::CliError;
use crate::cli::output::OutputWriter;
use crate::cli::runtime::CliRuntime;
use crate::clock::Ticked;
use crate::introspection::AdmissionState;

/// How often the idle loop looks for a drain.
///
/// The admission gate is an atomic with no way to wait on it, so an idle
/// `serve` checks it on this interval. A second is well inside any
/// `TimeoutStopSec`, and a relaxed load a second costs nothing.
const DRAIN_CHECK_INTERVAL: Duration = Duration::from_secs(1);

/// Options for `acton-ai serve`.
#[derive(Debug, clap::Args)]
pub struct ServeArgs {
    /// Append every tick to this journal before acting on it.
    ///
    /// Created with owner-only permissions if absent; if present, the
    /// timeline continues from its last entry. Without it, ticks are
    /// sequenced but not recorded.
    #[arg(long, value_name = "PATH")]
    pub tick_journal: Option<PathBuf>,

    /// Only process heartbeat entries for this session.
    #[arg(long)]
    pub session: Option<String>,
}

/// Executes `acton-ai serve`.
///
/// # Errors
///
/// Fails at launch when no introspection socket is configured — without one
/// no tick can ever arrive — or when the tick journal cannot be opened. Once
/// running, a failed pass is reported and the loop carries on: one bad
/// wake-up is not a reason for a resident process to exit.
pub async fn execute(
    args: &ServeArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
    provider_override: Option<&str>,
) -> Result<(), CliError> {
    let (sender, mut ticks) = tokio::sync::mpsc::unbounded_channel::<Ticked>();
    let journal = args.tick_journal.clone();

    let rt = CliRuntime::with_builder(config_path, provider_override, &[], move |builder| {
        let builder = builder.on_tick(move |tick| {
            // Only fails once the loop below has returned, when nobody is
            // waiting for ticks any more.
            let _ = sender.send(tick);
        });
        match journal {
            Some(path) => builder.tick_journal(path),
            None => builder,
        }
    })
    .await?;

    let Some(socket) = rt.ai.introspection_socket().map(ToOwned::to_owned) else {
        rt.shutdown().await?;
        return Err(CliError::configuration(
            "`acton-ai serve` waits for ticks on the introspection socket, but this \
             configuration does not open one; add an [introspection] section with a \
             socket_path, and point `acton-ai tick --socket` at the same path",
        ));
    };
    let _ = output.status(&format!(
        "serve: waiting for ticks on {} (send one with `acton-ai tick`)",
        socket.display()
    ));

    while let Some(tick) = next_tick(&rt, &mut ticks).await {
        let (tick, coalesced) = latest(tick, &mut ticks);
        if coalesced > 0 {
            let _ = output.status(&format!(
                "serve: {coalesced} earlier tick(s) folded into tick {}",
                tick.seq
            ));
        }

        match heartbeat::run_pass(
            &rt,
            args.session.as_deref(),
            tick.at,
            Some(tick.seq),
            output,
        )
        .await
        {
            Ok(report) => output.write_json(&report)?,
            Err(error) => {
                let _ = output.error(&format!("serve: tick {} failed: {error}", tick.seq));
            }
        }
    }

    let _ = output.status("serve: draining; no more ticks will start work");
    rt.shutdown().await?;
    Ok(())
}

/// Waits for the next tick, or `None` once the process is draining.
async fn next_tick(
    rt: &CliRuntime,
    ticks: &mut tokio::sync::mpsc::UnboundedReceiver<Ticked>,
) -> Option<Ticked> {
    loop {
        if rt.ai.admission_state() == AdmissionState::Draining {
            return None;
        }
        tokio::select! {
            tick = ticks.recv() => return tick,
            () = tokio::time::sleep(DRAIN_CHECK_INTERVAL) => {}
        }
    }
}

/// Folds every tick already waiting into the latest one.
///
/// Returns the tick to act on and how many earlier ones it replaced.
fn latest(
    mut tick: Ticked,
    ticks: &mut tokio::sync::mpsc::UnboundedReceiver<Ticked>,
) -> (Ticked, usize) {
    let mut coalesced = 0;
    while let Ok(newer) = ticks.try_recv() {
        tick = newer;
        coalesced += 1;
    }
    (tick, coalesced)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{EventTime, TickOrigin};

    fn tick(seq: u64) -> Ticked {
        Ticked {
            seq,
            at: EventTime::parse_rfc3339("2026-09-23T03:00:00Z").unwrap(),
            origin: TickOrigin::Socket,
            replayed: false,
        }
    }

    #[test]
    fn waiting_ticks_fold_into_the_latest() {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        sender.send(tick(2)).unwrap();
        sender.send(tick(3)).unwrap();

        let (latest, coalesced) = latest(tick(1), &mut receiver);
        assert_eq!((latest.seq, coalesced), (3, 2));
    }

    #[test]
    fn a_lone_tick_is_its_own_latest() {
        let (_sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<Ticked>();
        let (latest, coalesced) = latest(tick(7), &mut receiver);
        assert_eq!((latest.seq, coalesced), (7, 0));
    }
}
