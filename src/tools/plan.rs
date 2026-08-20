//! The plan a model publishes while it works, and the broadcast that
//! announces a new one.
//!
//! A long turn is opaque: the model calls tools for minutes and nothing
//! outside the process knows what it thinks it is doing. The `update_plan`
//! tool closes that gap — the model writes down its steps and which one it is
//! on, and this module is where that writing is checked and turned into a
//! value the rest of the process can react to.
//!
//! Everything here is **pure**. There is no I/O, no actor, and no clock:
//! [`Plan::parse`] takes what the model sent and returns either a validated
//! [`Plan`] or a [`PlanError`] whose `Display` is a sentence the model can act
//! on. The tool wrapper is
//! [`UpdatePlanTool`](crate::tools::builtins::UpdatePlanTool); the broadcast —
//! [`PlanUpdated`](crate::messages::PlanUpdated), which lives with the other
//! stream and lifecycle events in [`crate::messages`] — is issued by the
//! prompt round loop, which owns the turn's current plan and recovers each
//! update from the tool's result with [`plan_from_tool_result`].
//!
//! ## Invariants
//!
//! A [`Plan`] that exists is a plan that holds:
//!
//! - between [`MIN_PLAN_STEPS`] and [`MAX_PLAN_STEPS`] steps — a plan with a
//!   single step is refused outright, because it plans nothing;
//! - every step title trimmed, non-empty, and at most
//!   [`MAX_STEP_CHARS`] characters;
//! - no two steps with the same title;
//! - at most one step [`PlanStepStatus::InProgress`];
//! - a note, when present, trimmed, non-empty, and at most
//!   [`MAX_NOTE_CHARS`] characters.
//!
//! The invariants survive serde in both directions: [`Plan`]'s `Deserialize`
//! routes through [`Plan::parse`] rather than filling fields, so a plan that
//! arrives over the wire is checked exactly as hard as one built in-process.

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// The name the model calls to publish a plan.
///
/// Shared by the tool definition, the executor, and the round loop's
/// recognition of a plan-bearing result, so all three cannot drift apart.
pub const UPDATE_PLAN_TOOL: &str = "update_plan";

/// The fewest steps a plan may carry.
///
/// One step is not a plan, it is a restatement of the task. A model that
/// cannot find a second step should not be planning at all, so the minimum is
/// enforced rather than suggested.
pub const MIN_PLAN_STEPS: usize = 2;

/// The most steps a single plan may carry.
///
/// A plan is a working aid, not a work breakdown structure. Past a couple of
/// dozen steps the model is describing a different task than the one it was
/// asked to do.
pub const MAX_PLAN_STEPS: usize = 32;

/// The longest a single step title may be, in characters.
pub const MAX_STEP_CHARS: usize = 200;

/// The longest a note may be, in characters.
pub const MAX_NOTE_CHARS: usize = 1000;

// =============================================================================
// Errors
// =============================================================================

/// What is wrong with one piece of model-supplied text.
///
/// Deliberately a *fragment*: its `Display` completes a sentence started by
/// whoever holds it ("plan step 3 **is blank**", "the note **is 1240
/// characters long; the maximum is 1000**"), because the useful part of the
/// message — which step — is known one level up.
///
/// Marked `#[non_exhaustive]`: new text rules bring new refusals, and a
/// downstream `match` should not break each time one is added.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanTextError {
    /// The text was empty, or nothing but whitespace.
    Empty,
    /// The text was longer than the limit for its field.
    TooLong {
        /// How many characters the text actually had.
        chars: usize,
        /// The maximum allowed for this field.
        max: usize,
    },
}

impl fmt::Display for PlanTextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "is blank"),
            Self::TooLong { chars, max } => write!(
                f,
                "is {chars} characters long; the maximum is {max}, so shorten it"
            ),
        }
    }
}

impl std::error::Error for PlanTextError {}

/// A status string the model sent that is not one of the three allowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownPlanStepStatus {
    /// The status text that was not recognized.
    pub found: String,
}

impl fmt::Display for UnknownPlanStepStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown plan step status '{}'; use one of: pending, in_progress, completed",
            self.found
        )
    }
}

impl std::error::Error for UnknownPlanStepStatus {}

/// Why a proposed plan was refused.
///
/// Every variant's `Display` names the offending step by its 1-based position
/// — the way the model itself numbered them — and says what to do about it,
/// because this text is handed straight back to the model as the tool result.
///
/// Marked `#[non_exhaustive]`: new validation rules bring new refusals, and a
/// downstream `match` should not break each time one is added.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanError {
    /// The plan had no steps at all.
    NoSteps,
    /// The plan had exactly one step, which plans nothing.
    SingleStep,
    /// The plan had more steps than [`MAX_PLAN_STEPS`].
    TooManySteps {
        /// How many steps were sent.
        count: usize,
        /// The maximum allowed.
        max: usize,
    },
    /// One step's title was unusable.
    Step {
        /// 0-based position of the offending step.
        index: usize,
        /// What was wrong with its text.
        source: PlanTextError,
    },
    /// Two steps carried the same title.
    DuplicateStep {
        /// 0-based position of the repeat.
        index: usize,
        /// 0-based position of the step it repeats.
        first: usize,
        /// The title they share.
        title: String,
    },
    /// More than one step claimed to be in progress.
    MultipleInProgress {
        /// 0-based position of the first in-progress step.
        first: usize,
        /// 0-based position of the second one.
        second: usize,
    },
    /// The note was unusable.
    Note(PlanTextError),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSteps => write!(
                f,
                "the plan has no steps; send at least two, or skip planning \
                 entirely for a task too small to break down"
            ),
            Self::SingleStep => write!(
                f,
                "a single-step plan plans nothing; either break the work into \
                 at least two concrete steps, or skip planning entirely — \
                 trivial tasks do not need a plan"
            ),
            Self::TooManySteps { count, max } => write!(
                f,
                "the plan has {count} steps; the maximum is {max}, so group the small ones together"
            ),
            Self::Step { index, source } => write!(f, "plan step {} {source}", index + 1),
            Self::DuplicateStep {
                index,
                first,
                title,
            } => write!(
                f,
                "plan step {} repeats step {}, \"{title}\"; every step must be distinct",
                index + 1,
                first + 1
            ),
            Self::MultipleInProgress { first, second } => write!(
                f,
                "plan steps {} and {} are both in_progress; exactly one step may be in progress",
                first + 1,
                second + 1
            ),
            Self::Note(source) => write!(f, "the note {source}"),
        }
    }
}

impl std::error::Error for PlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Step { source, .. } | Self::Note(source) => Some(source),
            Self::NoSteps
            | Self::SingleStep
            | Self::TooManySteps { .. }
            | Self::DuplicateStep { .. }
            | Self::MultipleInProgress { .. } => None,
        }
    }
}

// =============================================================================
// Value types
// =============================================================================

/// A step title that is known to be non-empty, trimmed, and bounded.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanStepTitle(String);

impl PlanStepTitle {
    /// Validates raw model-supplied text into a title.
    ///
    /// Leading and trailing whitespace is removed first: models pad list
    /// items, and `" build the parser "` and `"build the parser"` are the same
    /// step, which matters because duplicates are refused.
    ///
    /// # Errors
    ///
    /// [`PlanTextError::Empty`] when nothing but whitespace was sent, and
    /// [`PlanTextError::TooLong`] past [`MAX_STEP_CHARS`] characters.
    pub fn parse(raw: &str) -> Result<Self, PlanTextError> {
        Ok(Self(validate_text(raw, MAX_STEP_CHARS)?))
    }

    /// The title as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the title, returning the owned text.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for PlanStepTitle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for PlanStepTitle {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl FromStr for PlanStepTitle {
    type Err = PlanTextError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for PlanStepTitle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for PlanStepTitle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// A free-text note about the plan as a whole, non-empty and bounded.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanNote(String);

impl PlanNote {
    /// Validates raw model-supplied text into an note.
    ///
    /// # Errors
    ///
    /// [`PlanTextError::Empty`] when nothing but whitespace was sent, and
    /// [`PlanTextError::TooLong`] past [`MAX_NOTE_CHARS`] characters.
    pub fn parse(raw: &str) -> Result<Self, PlanTextError> {
        Ok(Self(validate_text(raw, MAX_NOTE_CHARS)?))
    }

    /// Validates an optional note, treating blank text as absent.
    ///
    /// A model that has nothing to add sends `""` about as often as it omits
    /// the field, and refusing the whole plan over an empty note would be a
    /// silly place to fail.
    ///
    /// # Errors
    ///
    /// [`PlanTextError::TooLong`] past [`MAX_NOTE_CHARS`] characters.
    pub fn parse_optional(raw: Option<&str>) -> Result<Option<Self>, PlanTextError> {
        match raw.map(str::trim) {
            None => Ok(None),
            Some("") => Ok(None),
            Some(text) => Self::parse(text).map(Some),
        }
    }

    /// The note as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the note, returning the owned text.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for PlanNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for PlanNote {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl FromStr for PlanNote {
    type Err = PlanTextError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for PlanNote {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for PlanNote {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// Trims, rejects blank, and bounds the length of one piece of text.
///
/// The shared body of [`PlanStepTitle::parse`] and
/// [`PlanNote::parse`] — the two differ only in their limit, and a
/// second copy of this would be a second place for the trimming rule to change.
fn validate_text(raw: &str, max: usize) -> Result<String, PlanTextError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(PlanTextError::Empty);
    }

    // Characters, not bytes: the limit exists to keep a step readable, and a
    // model writing in Japanese should not get a third of the room.
    let chars = trimmed.chars().count();
    if chars > max {
        return Err(PlanTextError::TooLong { chars, max });
    }

    Ok(trimmed.to_string())
}

/// Where one step of a plan stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    /// Not started yet. The status a step gets when the model omits one.
    #[default]
    Pending,
    /// Being worked on right now. At most one step per plan may hold it.
    InProgress,
    /// Finished.
    Completed,
}

impl PlanStepStatus {
    /// The wire spelling of this status, as the model writes it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
        }
    }

    /// Every status, in the order a step travels through them.
    ///
    /// This is what the tool's JSON schema enumerates, so the model is offered
    /// exactly the set [`FromStr`] accepts.
    #[must_use]
    pub fn all() -> [Self; 3] {
        [Self::Pending, Self::InProgress, Self::Completed]
    }
}

impl fmt::Display for PlanStepStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PlanStepStatus {
    type Err = UnknownPlanStepStatus;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "in_progress" => Ok(Self::InProgress),
            "completed" => Ok(Self::Completed),
            other => Err(UnknownPlanStepStatus {
                found: other.to_string(),
            }),
        }
    }
}

/// One validated step of a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanStep {
    /// What the step is.
    title: PlanStepTitle,
    /// Where it stands.
    status: PlanStepStatus,
}

impl PlanStep {
    /// Builds a step from parts that are already validated.
    #[must_use]
    pub fn new(title: PlanStepTitle, status: PlanStepStatus) -> Self {
        Self { title, status }
    }

    /// Validates raw model-supplied text into a step.
    ///
    /// # Errors
    ///
    /// Whatever [`PlanStepTitle::parse`] rejects.
    pub fn parse(title: &str, status: PlanStepStatus) -> Result<Self, PlanTextError> {
        Ok(Self::new(PlanStepTitle::parse(title)?, status))
    }

    /// What this step is.
    #[must_use]
    pub fn title(&self) -> &PlanStepTitle {
        &self.title
    }

    /// Where this step stands.
    #[must_use]
    pub fn status(&self) -> PlanStepStatus {
        self.status
    }
}

impl Serialize for PlanStep {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PlanStep", 2)?;
        state.serialize_field("title", self.title.as_str())?;
        state.serialize_field("status", &self.status)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for PlanStep {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawPlanStep::deserialize(deserializer)?;
        Self::parse(&raw.title, raw.status).map_err(serde::de::Error::custom)
    }
}

/// One step exactly as the model sent it, before any validation.
///
/// This is the shape named in the tool's input schema and the shape a
/// serialized [`Plan`] uses, so the model reads back what it wrote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawPlanStep {
    /// The step's text.
    pub title: String,
    /// The step's status. Absent means [`PlanStepStatus::Pending`] — the
    /// schema asks for it, but refusing a whole plan over a field the model
    /// forgot on step 7 helps nobody.
    #[serde(default)]
    pub status: PlanStepStatus,
}

/// A validated plan: an ordered list of steps and an optional note.
///
/// Build one with [`Plan::parse`] (from model-supplied text) or [`Plan::new`]
/// (from parts already validated). There is no way to construct one that
/// breaks the invariants listed in the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The steps, in the order the model listed them.
    steps: Vec<PlanStep>,
    /// The model's note about the plan as a whole, if it wrote one.
    note: Option<PlanNote>,
}

impl Plan {
    /// Validates raw, model-supplied parts into a plan.
    ///
    /// This is the door every plan comes through, whether it arrived as tool
    /// arguments or over serde.
    ///
    /// # Errors
    ///
    /// A [`PlanError`] naming the first problem found, in the order a reader
    /// would find them: the note, then each step's text, then the
    /// whole-plan rules.
    pub fn parse(steps: &[RawPlanStep], note: Option<&str>) -> Result<Self, PlanError> {
        let note = PlanNote::parse_optional(note).map_err(PlanError::Note)?;

        let mut validated = Vec::with_capacity(steps.len());
        for (index, raw) in steps.iter().enumerate() {
            let step = PlanStep::parse(&raw.title, raw.status)
                .map_err(|source| PlanError::Step { index, source })?;
            validated.push(step);
        }

        Self::new(validated, note)
    }

    /// Assembles a plan from already-validated parts, checking the rules that
    /// span more than one step.
    ///
    /// # Errors
    ///
    /// [`PlanError::NoSteps`], [`PlanError::SingleStep`],
    /// [`PlanError::TooManySteps`], [`PlanError::DuplicateStep`], or
    /// [`PlanError::MultipleInProgress`].
    pub fn new(steps: Vec<PlanStep>, note: Option<PlanNote>) -> Result<Self, PlanError> {
        if steps.is_empty() {
            return Err(PlanError::NoSteps);
        }
        if steps.len() < MIN_PLAN_STEPS {
            return Err(PlanError::SingleStep);
        }
        if steps.len() > MAX_PLAN_STEPS {
            return Err(PlanError::TooManySteps {
                count: steps.len(),
                max: MAX_PLAN_STEPS,
            });
        }

        check_distinct(&steps)?;
        check_one_in_progress(&steps)?;

        Ok(Self { steps, note })
    }

    /// The steps, in order.
    #[must_use]
    pub fn steps(&self) -> &[PlanStep] {
        &self.steps
    }

    /// The model's note about the plan, if it wrote one.
    #[must_use]
    pub fn note(&self) -> Option<&PlanNote> {
        self.note.as_ref()
    }

    /// The step being worked on, if any.
    #[must_use]
    pub fn in_progress(&self) -> Option<&PlanStep> {
        self.steps
            .iter()
            .find(|step| step.status() == PlanStepStatus::InProgress)
    }

    /// How many steps are finished.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| step.status() == PlanStepStatus::Completed)
            .count()
    }

    /// How many steps the plan has. Never below [`MIN_PLAN_STEPS`].
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Whether every step is finished.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completed_count() == self.steps.len()
    }
}

/// Refuses a plan that lists the same step twice.
///
/// A repeat is how a model loses track of where it is — it re-adds a step it
/// already finished — so it is worth a refusal rather than a silent merge.
fn check_distinct(steps: &[PlanStep]) -> Result<(), PlanError> {
    let mut seen: HashMap<&str, usize> = HashMap::with_capacity(steps.len());
    for (index, step) in steps.iter().enumerate() {
        let title = step.title().as_str();
        if let Some(&first) = seen.get(title) {
            return Err(PlanError::DuplicateStep {
                index,
                first,
                title: title.to_string(),
            });
        }
        seen.insert(title, index);
    }
    Ok(())
}

/// Refuses a plan with two steps in flight at once.
fn check_one_in_progress(steps: &[PlanStep]) -> Result<(), PlanError> {
    let mut first_seen: Option<usize> = None;
    for (index, step) in steps.iter().enumerate() {
        if step.status() != PlanStepStatus::InProgress {
            continue;
        }
        match first_seen {
            Some(first) => {
                return Err(PlanError::MultipleInProgress {
                    first,
                    second: index,
                })
            }
            None => first_seen = Some(index),
        }
    }
    Ok(())
}

impl Serialize for Plan {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Plan", 2)?;
        state.serialize_field("steps", &self.steps)?;
        state.serialize_field("note", &self.note)?;
        state.end()
    }
}

/// The serialized shape of a [`Plan`], used only to get back off the wire.
#[derive(Deserialize)]
struct PlanWire {
    /// The steps, unvalidated.
    #[serde(default)]
    steps: Vec<RawPlanStep>,
    /// The note, unvalidated.
    #[serde(default)]
    note: Option<String>,
}

impl<'de> Deserialize<'de> for Plan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PlanWire::deserialize(deserializer)?;
        Self::parse(&wire.steps, wire.note.as_deref()).map_err(serde::de::Error::custom)
    }
}

// =============================================================================
// Round-loop integration
// =============================================================================

/// Recovers the plan a successful tool result carries, if it carries one.
///
/// Pure, and total: any other tool's name, a result of the wrong shape, or a
/// plan that no longer validates all yield `None`. The prompt round loop calls
/// this after every tool call, so the decision to broadcast is a value
/// computed from data rather than a branch buried in a handler.
#[must_use]
pub fn plan_from_tool_result(tool_name: &str, result: &Value) -> Option<Plan> {
    if tool_name != UPDATE_PLAN_TOOL {
        return None;
    }

    serde_json::from_value(result.get("plan")?.clone()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The plan most tests start from: three steps, the middle one running.
    fn raw_steps() -> Vec<RawPlanStep> {
        vec![
            RawPlanStep {
                title: "read the config loader".to_string(),
                status: PlanStepStatus::Completed,
            },
            RawPlanStep {
                title: "add the new key".to_string(),
                status: PlanStepStatus::InProgress,
            },
            RawPlanStep {
                title: "cover it with a test".to_string(),
                status: PlanStepStatus::Pending,
            },
        ]
    }

    fn raw(title: &str, status: PlanStepStatus) -> RawPlanStep {
        RawPlanStep {
            title: title.to_string(),
            status,
        }
    }

    // -- happy path -------------------------------------------------------

    #[test]
    fn a_well_formed_plan_keeps_its_steps_in_order() {
        let plan = Plan::parse(&raw_steps(), Some("working through the loader")).expect("valid");

        let titles: Vec<&str> = plan
            .steps()
            .iter()
            .map(|step| step.title().as_str())
            .collect();
        assert_eq!(
            titles,
            vec![
                "read the config loader",
                "add the new key",
                "cover it with a test"
            ]
        );
        assert_eq!(
            plan.note().map(PlanNote::as_str),
            Some("working through the loader")
        );
    }

    #[test]
    fn accessors_report_progress() {
        let plan = Plan::parse(&raw_steps(), None).expect("valid");

        assert_eq!(plan.step_count(), 3);
        assert_eq!(plan.completed_count(), 1);
        assert!(!plan.is_complete());
        assert_eq!(
            plan.in_progress().map(|step| step.title().as_str()),
            Some("add the new key")
        );
    }

    #[test]
    fn a_plan_with_everything_done_is_complete_and_has_nothing_running() {
        let plan = Plan::parse(
            &[
                raw("one", PlanStepStatus::Completed),
                raw("two", PlanStepStatus::Completed),
            ],
            None,
        )
        .expect("valid");

        assert!(plan.is_complete());
        assert!(plan.in_progress().is_none());
    }

    #[test]
    fn a_missing_status_means_pending() {
        let steps: Vec<RawPlanStep> = serde_json::from_value(json!([
            {"title": "do the thing"},
            {"title": "check the thing"},
        ]))
        .expect("a step may omit its status");

        let plan = Plan::parse(&steps, None).expect("valid");
        assert_eq!(plan.steps()[0].status(), PlanStepStatus::Pending);
    }

    // -- text validation --------------------------------------------------

    #[test]
    fn step_titles_are_trimmed() {
        let plan = Plan::parse(
            &[
                raw("  build the parser \n", PlanStepStatus::Pending),
                raw("run the suite", PlanStepStatus::Pending),
            ],
            None,
        )
        .expect("valid");

        assert_eq!(plan.steps()[0].title().as_str(), "build the parser");
    }

    #[test]
    fn a_blank_step_names_its_position() {
        let error = Plan::parse(
            &[
                raw("first", PlanStepStatus::Pending),
                raw("   \t ", PlanStepStatus::Pending),
            ],
            None,
        )
        .expect_err("a blank step is not a step");

        assert_eq!(
            error,
            PlanError::Step {
                index: 1,
                source: PlanTextError::Empty
            }
        );
        assert_eq!(error.to_string(), "plan step 2 is blank");
    }

    #[test]
    fn an_overlong_step_reports_the_limit() {
        let long = "x".repeat(MAX_STEP_CHARS + 1);
        let error = Plan::parse(
            &[
                raw(&long, PlanStepStatus::Pending),
                raw("short", PlanStepStatus::Pending),
            ],
            None,
        )
        .expect_err("the step is over the limit");

        assert_eq!(
            error,
            PlanError::Step {
                index: 0,
                source: PlanTextError::TooLong {
                    chars: MAX_STEP_CHARS + 1,
                    max: MAX_STEP_CHARS
                }
            }
        );
    }

    #[test]
    fn a_step_exactly_at_the_limit_is_accepted() {
        let at_limit = "x".repeat(MAX_STEP_CHARS);
        let plan = Plan::parse(
            &[
                raw(&at_limit, PlanStepStatus::Pending),
                raw("wrap up", PlanStepStatus::Pending),
            ],
            None,
        )
        .expect("the limit is inclusive");

        assert_eq!(
            plan.steps()[0].title().as_str().chars().count(),
            MAX_STEP_CHARS
        );
    }

    #[test]
    fn the_length_limit_counts_characters_not_bytes() {
        // Every one of these is three bytes, so a byte-based limit would
        // reject a plan a third of the allowed length.
        let japanese = "あ".repeat(MAX_STEP_CHARS);
        assert!(Plan::parse(
            &[
                raw(&japanese, PlanStepStatus::Pending),
                raw("done", PlanStepStatus::Pending),
            ],
            None
        )
        .is_ok());
    }

    // -- whole-plan rules --------------------------------------------------

    #[test]
    fn an_empty_plan_is_refused() {
        let error = Plan::parse(&[], None).expect_err("a plan needs steps");

        assert_eq!(error, PlanError::NoSteps);
        let message = error.to_string();
        assert!(message.contains("at least two"), "{message}");
        assert!(
            message.contains("skip planning"),
            "the refusal must offer the model a way out: {message}"
        );
    }

    #[test]
    fn a_single_step_plan_is_refused_with_a_way_forward() {
        let error = Plan::parse(&[raw("do the task", PlanStepStatus::InProgress)], None)
            .expect_err("one step is a restatement, not a plan");

        assert_eq!(error, PlanError::SingleStep);
        let message = error.to_string();
        assert!(
            message.contains("at least two"),
            "the model must be told to split the work: {message}"
        );
        assert!(
            message.contains("skip planning"),
            "or to not plan at all: {message}"
        );
    }

    #[test]
    fn too_many_steps_reports_the_count() {
        let many: Vec<RawPlanStep> = (0..=MAX_PLAN_STEPS)
            .map(|n| raw(&format!("step {n}"), PlanStepStatus::Pending))
            .collect();

        let error = Plan::parse(&many, None).expect_err("over the step limit");

        assert_eq!(
            error,
            PlanError::TooManySteps {
                count: MAX_PLAN_STEPS + 1,
                max: MAX_PLAN_STEPS
            }
        );
    }

    #[test]
    fn a_repeated_step_names_both_positions() {
        let error = Plan::parse(
            &[
                raw("write the test", PlanStepStatus::Completed),
                raw("run it", PlanStepStatus::Pending),
                raw("write the test", PlanStepStatus::Pending),
            ],
            None,
        )
        .expect_err("a repeat is a bookkeeping mistake");

        assert_eq!(
            error,
            PlanError::DuplicateStep {
                index: 2,
                first: 0,
                title: "write the test".to_string()
            }
        );
        assert!(
            error.to_string().contains("plan step 3 repeats step 1"),
            "{error}"
        );
    }

    #[test]
    fn a_repeat_is_caught_after_trimming() {
        let error = Plan::parse(
            &[
                raw("same step", PlanStepStatus::Pending),
                raw("  same step  ", PlanStepStatus::Pending),
            ],
            None,
        )
        .expect_err("padding does not make a step distinct");

        assert!(matches!(error, PlanError::DuplicateStep { index: 1, .. }));
    }

    #[test]
    fn two_steps_in_progress_are_refused() {
        let error = Plan::parse(
            &[
                raw("first", PlanStepStatus::InProgress),
                raw("second", PlanStepStatus::Pending),
                raw("third", PlanStepStatus::InProgress),
            ],
            None,
        )
        .expect_err("only one step may be in flight");

        assert_eq!(
            error,
            PlanError::MultipleInProgress {
                first: 0,
                second: 2
            }
        );
        assert!(
            error
                .to_string()
                .contains("plan steps 1 and 3 are both in_progress"),
            "{error}"
        );
    }

    // -- note -------------------------------------------------------

    #[test]
    fn a_blank_note_is_absent_rather_than_an_error() {
        let plan = Plan::parse(&raw_steps(), Some("   ")).expect("a blank note is not a failure");

        assert!(plan.note().is_none());
    }

    #[test]
    fn an_overlong_note_is_refused() {
        let long = "x".repeat(MAX_NOTE_CHARS + 1);
        let error = Plan::parse(&raw_steps(), Some(&long)).expect_err("over the note limit");

        assert_eq!(
            error,
            PlanError::Note(PlanTextError::TooLong {
                chars: MAX_NOTE_CHARS + 1,
                max: MAX_NOTE_CHARS
            })
        );
        assert!(error.to_string().starts_with("the note is"), "{error}");
    }

    #[test]
    fn errors_expose_their_text_cause() {
        use std::error::Error;

        let error = Plan::parse(&[raw(" ", PlanStepStatus::Pending)], None).expect_err("blank");
        let source = error.source().expect("a text failure has a cause");
        assert_eq!(source.to_string(), "is blank");

        let no_source = Plan::parse(&[], None).expect_err("empty");
        assert!(no_source.source().is_none());
    }

    // -- status ------------------------------------------------------------

    #[test]
    fn statuses_round_trip_through_their_wire_spelling() {
        for status in PlanStepStatus::all() {
            assert_eq!(
                PlanStepStatus::from_str(status.as_str()).expect("its own spelling parses"),
                status
            );
            assert_eq!(status.to_string(), status.as_str());
        }
    }

    #[test]
    fn an_unknown_status_lists_the_accepted_ones() {
        let error = PlanStepStatus::from_str("done").expect_err("'done' is not a status");

        assert_eq!(error.found, "done");
        let message = error.to_string();
        for status in PlanStepStatus::all() {
            assert!(message.contains(status.as_str()), "{message}");
        }
    }

    #[test]
    fn statuses_deserialize_from_snake_case() {
        let steps: Vec<RawPlanStep> = serde_json::from_value(json!([
            {"title": "a", "status": "in_progress"},
            {"title": "b", "status": "completed"},
        ]))
        .expect("snake_case is the wire spelling");

        assert_eq!(steps[0].status, PlanStepStatus::InProgress);
        assert_eq!(steps[1].status, PlanStepStatus::Completed);
    }

    // -- serde -------------------------------------------------------------

    #[test]
    fn a_plan_round_trips_through_json() {
        let plan = Plan::parse(&raw_steps(), Some("halfway")).expect("valid");

        let encoded = serde_json::to_value(&plan).expect("a plan serializes");
        let decoded: Plan = serde_json::from_value(encoded.clone()).expect("and comes back");

        assert_eq!(decoded, plan);
        assert_eq!(
            encoded,
            json!({
                "steps": [
                    {"title": "read the config loader", "status": "completed"},
                    {"title": "add the new key", "status": "in_progress"},
                    {"title": "cover it with a test", "status": "pending"},
                ],
                "note": "halfway",
            })
        );
    }

    #[test]
    fn deserializing_cannot_smuggle_a_broken_plan_past_the_rules() {
        let two_running = json!({
            "steps": [
                {"title": "a", "status": "in_progress"},
                {"title": "b", "status": "in_progress"},
            ],
        });

        let error = serde_json::from_value::<Plan>(two_running)
            .expect_err("the invariant must survive the wire");
        assert!(error.to_string().contains("in_progress"), "{error}");

        let empty = json!({"steps": []});
        assert!(serde_json::from_value::<Plan>(empty).is_err());
    }

    #[test]
    fn a_single_step_round_trips_on_its_own() {
        let step = PlanStep::parse("ship it", PlanStepStatus::Completed).expect("valid");

        let encoded = serde_json::to_value(&step).expect("serializes");
        assert_eq!(encoded, json!({"title": "ship it", "status": "completed"}));
        assert_eq!(
            serde_json::from_value::<PlanStep>(encoded).expect("deserializes"),
            step
        );
    }

    #[test]
    fn a_blank_step_is_refused_on_the_way_in() {
        let error = serde_json::from_value::<PlanStep>(json!({"title": "  ", "status": "pending"}))
            .expect_err("a blank step never becomes a value");
        assert!(error.to_string().contains("blank"), "{error}");
    }

    #[test]
    fn the_value_types_round_trip_on_their_own() {
        let title = PlanStepTitle::parse("write the docs").expect("valid");
        let encoded = serde_json::to_value(&title).expect("serializes");
        assert_eq!(encoded, json!("write the docs"));
        assert_eq!(
            serde_json::from_value::<PlanStepTitle>(encoded).expect("deserializes"),
            title
        );

        let note = PlanNote::parse("because").expect("valid");
        let encoded = serde_json::to_value(&note).expect("serializes");
        assert_eq!(encoded, json!("because"));
        assert_eq!(
            serde_json::from_value::<PlanNote>(encoded).expect("deserializes"),
            note
        );
        assert!(serde_json::from_value::<PlanNote>(json!("  ")).is_err());
    }

    #[test]
    fn value_types_parse_from_str_and_display_round_trip() {
        let title: PlanStepTitle = " tidy up ".parse().expect("trims and parses");
        assert_eq!(title.to_string(), "tidy up");
        assert_eq!(title.as_ref(), "tidy up");
        assert_eq!(title.clone().into_string(), "tidy up");

        let note: PlanNote = "why".parse().expect("parses");
        assert_eq!(note.to_string(), "why");
        assert_eq!(note.as_ref(), "why");
        assert_eq!(note.into_string(), "why");
    }

    // -- round-loop recognition -------------------------------------------

    #[test]
    fn a_plan_is_recovered_from_an_update_plan_result() {
        let plan = Plan::parse(&raw_steps(), Some("halfway")).expect("valid");
        let result = json!({
            "status": "recorded",
            "plan": serde_json::to_value(&plan).expect("serializes"),
        });

        assert_eq!(
            plan_from_tool_result(UPDATE_PLAN_TOOL, &result),
            Some(plan),
            "the round loop must recover exactly what was recorded"
        );
    }

    #[test]
    fn another_tools_result_carries_no_plan() {
        // Even one shaped exactly like a plan: the name is what decides.
        let plan = Plan::parse(&raw_steps(), None).expect("valid");
        let result = json!({"plan": serde_json::to_value(&plan).expect("serializes")});

        assert_eq!(plan_from_tool_result("read_file", &result), None);
    }

    #[test]
    fn a_malformed_result_carries_no_plan() {
        for result in [
            json!({}),
            json!({"plan": null}),
            json!({"plan": "not a plan"}),
            json!({"plan": {"steps": []}}),
            json!("a bare string"),
        ] {
            assert_eq!(
                plan_from_tool_result(UPDATE_PLAN_TOOL, &result),
                None,
                "{result} must not yield a plan"
            );
        }
    }
}
