//! What a restarted process does about the interrupted turns it finds.
//!
//! The policy is an operator decision, not a library one: whether an
//! interrupted turn should quietly pick itself back up, wait to be asked, or
//! be closed out as abandoned depends on what the turns do — a chat session
//! and a payment workflow want different answers. The default is
//! [`Abandon`](ResumePolicy::Abandon), the only choice that never dispatches a
//! request nobody just asked for.

use serde::{Deserialize, Serialize};

/// What launch does about interrupted turns left by a previous process.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResumePolicy {
    /// Resume every interrupted turn in the background as soon as the runtime
    /// is up. The strongest continuity, and the only policy that spends
    /// money without a fresh request.
    ResumeAuto,
    /// Leave interrupted turns where they are. The operator lists them with
    /// [`ActonAI::interrupted_turns`](crate::facade::ActonAI::interrupted_turns) and
    /// picks the ones worth finishing via
    /// [`ActonAI::resume_turn`](crate::facade::ActonAI::resume_turn).
    ResumeOnRequest,
    /// Mark every interrupted turn abandoned. The records are kept as
    /// evidence of the outcome; nothing runs, nothing is deleted. The
    /// default, because it is the only policy under which enabling
    /// checkpoints changes no behavior beyond the writes themselves.
    #[default]
    Abandon,
}

/// Checkpoint support, as the builder configures it.
///
/// Naming a database path is what turns checkpointing on: every prompt gets a
/// checkpoint written under a fresh ID, and launch applies `policy` to
/// whatever interrupted turns the previous process left behind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointConfig {
    /// Where the checkpoint database lives. `:memory:` works, but a resume
    /// only ever finds something to resume when the path outlives the
    /// process.
    pub db_path: String,
    /// What launch does about interrupted turns it finds there.
    pub policy: ResumePolicy,
    /// How many failed attempts [`ActonAI::resume_interrupted`](crate::facade::ActonAI::resume_interrupted)
    /// grants a turn before abandoning it instead of resuming it again.
    ///
    /// This is what bounds the unattended paths — the `resume_auto`
    /// background task and operator-driven `resume_interrupted()` sweeps. A
    /// turn that fails for the same reason on every process start would
    /// otherwise be re-dispatched, and re-paid for, on every restart forever.
    /// Once a `Failed` record's counted attempts reach this ceiling, the
    /// sweep marks it `Abandoned` — a recorded outcome an operator can list —
    /// rather than running it. A deliberate, per-turn
    /// [`resume_turn`](crate::facade::ActonAI::resume_turn) is never subject to the
    /// ceiling: an operator who picked one specific record out is asking for
    /// exactly one more attempt.
    ///
    /// Defaults to [`Self::DEFAULT_MAX_RESUME_ATTEMPTS`].
    pub max_resume_attempts: u32,
}

impl CheckpointConfig {
    /// Checkpoints stored at `db_path`, with the default [`Abandon`]
    /// policy.
    ///
    /// [`Abandon`]: ResumePolicy::Abandon
    #[must_use]
    pub fn new(db_path: impl Into<String>) -> Self {
        Self {
            db_path: db_path.into(),
            policy: ResumePolicy::default(),
            max_resume_attempts: Self::DEFAULT_MAX_RESUME_ATTEMPTS,
        }
    }

    /// How many failed attempts an unattended sweep grants a turn by default.
    pub const DEFAULT_MAX_RESUME_ATTEMPTS: u32 = 3;

    /// Sets what launch does about interrupted turns.
    #[must_use]
    pub fn policy(mut self, policy: ResumePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sets how many failed attempts an unattended sweep grants a turn
    /// before abandoning it. See [`Self::max_resume_attempts`].
    #[must_use]
    pub fn max_resume_attempts(mut self, attempts: u32) -> Self {
        self.max_resume_attempts = attempts;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal TOML shape carrying just the policy field.
    #[derive(Deserialize)]
    struct Probe {
        policy: ResumePolicy,
    }

    #[test]
    fn the_default_policy_is_abandon() {
        assert_eq!(ResumePolicy::default(), ResumePolicy::Abandon);
        assert_eq!(CheckpointConfig::new("x.db").policy, ResumePolicy::Abandon);
    }

    #[test]
    fn the_toml_spellings_parse() {
        let parse = |text: &str| -> ResumePolicy {
            toml::from_str::<Probe>(&format!("policy = \"{text}\""))
                .expect("the spelling must parse")
                .policy
        };
        assert_eq!(parse("resume_auto"), ResumePolicy::ResumeAuto);
        assert_eq!(parse("resume_on_request"), ResumePolicy::ResumeOnRequest);
        assert_eq!(parse("abandon"), ResumePolicy::Abandon);
    }

    #[test]
    fn an_unknown_spelling_is_refused() {
        assert!(toml::from_str::<Probe>("policy = \"always\"").is_err());
    }

    #[test]
    fn the_builder_sets_the_policy() {
        let config = CheckpointConfig::new("x.db").policy(ResumePolicy::ResumeAuto);
        assert_eq!(config.policy, ResumePolicy::ResumeAuto);
        assert_eq!(config.db_path, "x.db");
        assert_eq!(
            config.max_resume_attempts,
            CheckpointConfig::DEFAULT_MAX_RESUME_ATTEMPTS
        );
    }

    #[test]
    fn the_builder_sets_the_attempt_ceiling() {
        let config = CheckpointConfig::new("x.db").max_resume_attempts(1);
        assert_eq!(config.max_resume_attempts, 1);
    }
}
