//! Turn admission: whether the runtime is currently accepting new turns.
//!
//! Compiled unconditionally. This is ordinary in-process state, not IPC — an
//! embedder that builds without the `ipc` feature still gets
//! [`ActonAI::pause`](crate::facade::ActonAI::pause) and its siblings; the
//! feature only decides whether a *socket* can drive them.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Whether the runtime is accepting new turns, and if not, why.
///
/// A turn is one `collect()` / `extract()` call. Admission is checked once,
/// at the top of the turn: a turn that was admitted runs to completion even
/// if the runtime is paused underneath it. Interrupting work already in
/// flight would lose it, and losing it is strictly worse than finishing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AdmissionState {
    /// New turns are admitted. The state every runtime launches in.
    #[default]
    Running,
    /// New turns are refused, and the operator intends to resume.
    Paused,
    /// New turns are refused, and the process is on its way down.
    ///
    /// Identical to [`Paused`](Self::Paused) as far as a turn is concerned.
    /// The distinction is for the operator reading `acton-ai status`: a
    /// paused process is waiting for a human, a draining one is waiting for
    /// its own in-flight work.
    Draining,
}

impl AdmissionState {
    /// Whether a new turn may start.
    ///
    /// ```rust
    /// use acton_ai::introspection::AdmissionState;
    ///
    /// assert!(AdmissionState::Running.admits());
    /// assert!(!AdmissionState::Paused.admits());
    /// assert!(!AdmissionState::Draining.admits());
    /// ```
    #[must_use]
    pub const fn admits(self) -> bool {
        matches!(self, Self::Running)
    }

    /// Short, stable label. What `--json` carries and what an operator greps.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Draining => "draining",
        }
    }

    /// The wire encoding: one byte, so the whole state fits in an atomic.
    const fn as_u8(self) -> u8 {
        match self {
            Self::Running => 0,
            Self::Paused => 1,
            Self::Draining => 2,
        }
    }

    /// Decodes [`Self::as_u8`].
    ///
    /// Anything unrecognised decodes to [`Paused`](Self::Paused) rather than
    /// [`Running`](Self::Running): the byte can only be written by
    /// [`as_u8`](Self::as_u8), so an unknown value is impossible in practice,
    /// and the safe way to be wrong about admission is to refuse work rather
    /// than to spend money.
    const fn from_u8(raw: u8) -> Self {
        match raw {
            0 => Self::Running,
            2 => Self::Draining,
            _ => Self::Paused,
        }
    }
}

impl std::fmt::Display for AdmissionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The runtime's admission state, shared between the prompt loop and whoever
/// flips it.
///
/// # Why shared state rather than an actor ask
///
/// This crate's standing preference is a message, not a shared cell. The
/// exception is deliberate and narrow, and rests on the same reasoning as a
/// `watch` channel:
///
/// - **One writer.** Only the introspection surface writes — the
///   [`IntrospectionActor`](crate::introspection::IntrospectionActor)
///   serialising IPC commands, or the embedder calling
///   [`ActonAI::pause`](crate::facade::ActonAI::pause). Both funnel through
///   the same methods here, so there is no write-write race to lose.
/// - **One scalar.** A reader takes a consistent snapshot because there is
///   exactly one value to read. Nothing here can be observed half-updated,
///   which is the failure mode that makes shared mutable state dangerous.
/// - **The hot path.** The alternative is an `ask` to an IPC-facing actor at
///   the top of every turn — a round trip, and an actor that can be busy
///   serving a socket, placed on the critical path of every request the
///   process makes. A relaxed atomic load is free, which is what the 99% of
///   processes that never pause should pay.
///
/// `Relaxed` ordering is sufficient: the byte guards no other memory, so
/// there is nothing for an acquire/release pair to publish. The only
/// guarantee needed is that a pause eventually becomes visible, which
/// `Relaxed` gives.
#[derive(Debug, Clone, Default)]
pub struct AdmissionGate {
    state: Arc<AtomicU8>,
}

impl AdmissionGate {
    /// A gate in [`AdmissionState::Running`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The current state.
    ///
    /// ```rust
    /// use acton_ai::introspection::{AdmissionGate, AdmissionState};
    ///
    /// let gate = AdmissionGate::new();
    /// assert_eq!(gate.state(), AdmissionState::Running);
    /// ```
    #[must_use]
    pub fn state(&self) -> AdmissionState {
        AdmissionState::from_u8(self.state.load(Ordering::Relaxed))
    }

    /// Whether a new turn may start right now.
    #[must_use]
    pub fn admits(&self) -> bool {
        self.state().admits()
    }

    /// Moves to `state`, returning the state now in force.
    ///
    /// Returns the new state rather than the old one so a caller can echo
    /// what it achieved without a second load, which would be a second
    /// observation and could disagree.
    pub fn set(&self, state: AdmissionState) -> AdmissionState {
        self.state.store(state.as_u8(), Ordering::Relaxed);
        state
    }

    /// Refuses new turns, pending a [`resume`](Self::resume).
    pub fn pause(&self) -> AdmissionState {
        self.set(AdmissionState::Paused)
    }

    /// Refuses new turns because the process is going down.
    pub fn drain(&self) -> AdmissionState {
        self.set(AdmissionState::Draining)
    }

    /// Admits new turns again, from either refusing state.
    pub fn resume(&self) -> AdmissionState {
        self.set(AdmissionState::Running)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_gate_admits_turns() {
        let gate = AdmissionGate::new();
        assert_eq!(gate.state(), AdmissionState::Running);
        assert!(gate.admits());
    }

    #[test]
    fn pause_refuses_turns() {
        let gate = AdmissionGate::new();
        assert_eq!(gate.pause(), AdmissionState::Paused);
        assert_eq!(gate.state(), AdmissionState::Paused);
        assert!(!gate.admits());
    }

    #[test]
    fn drain_refuses_turns_and_is_distinguishable_from_pause() {
        let gate = AdmissionGate::new();
        assert_eq!(gate.drain(), AdmissionState::Draining);
        assert!(!gate.admits());
        // The operator-visible difference is the whole point of the third
        // state; collapsing it into Paused would be silently wrong.
        assert_ne!(gate.state(), AdmissionState::Paused);
    }

    #[test]
    fn resume_admits_turns_from_paused() {
        let gate = AdmissionGate::new();
        gate.pause();
        assert_eq!(gate.resume(), AdmissionState::Running);
        assert!(gate.admits());
    }

    #[test]
    fn resume_admits_turns_from_draining() {
        let gate = AdmissionGate::new();
        gate.drain();
        assert_eq!(gate.resume(), AdmissionState::Running);
        assert!(gate.admits());
    }

    #[test]
    fn a_clone_shares_one_state() {
        let gate = AdmissionGate::new();
        let other = gate.clone();
        other.pause();
        // A clone that did not share would still read Running here, and every
        // pause issued over IPC would be invisible to the prompt loop.
        assert_eq!(gate.state(), AdmissionState::Paused);
    }

    #[test]
    fn every_state_round_trips_through_its_byte() {
        for state in [
            AdmissionState::Running,
            AdmissionState::Paused,
            AdmissionState::Draining,
        ] {
            assert_eq!(AdmissionState::from_u8(state.as_u8()), state);
        }
    }

    #[test]
    fn an_unrecognised_byte_decodes_to_refusing() {
        // Unreachable in practice, but the direction of the failure matters:
        // refusing work is recoverable, spending money is not.
        assert_eq!(AdmissionState::from_u8(200), AdmissionState::Paused);
        assert!(!AdmissionState::from_u8(200).admits());
    }

    #[test]
    fn labels_are_distinct_and_stable() {
        assert_eq!(AdmissionState::Running.as_str(), "running");
        assert_eq!(AdmissionState::Paused.as_str(), "paused");
        assert_eq!(AdmissionState::Draining.as_str(), "draining");
        assert_eq!(AdmissionState::Draining.to_string(), "draining");
    }

    #[test]
    fn the_default_state_admits() {
        assert_eq!(AdmissionState::default(), AdmissionState::Running);
        assert!(AdmissionGate::default().admits());
    }
}
