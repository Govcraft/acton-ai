//! Typed structured output: schema generation and answer validation.
//!
//! This module holds the pure, IO-free half of [`PromptBuilder::extract`]
//! (see [`crate::prompt`]): turning a Rust type into a JSON Schema the model
//! can be pointed at, and turning the model's answer back into a checked
//! [`serde_json::Value`].
//!
//! # How extraction works
//!
//! Rather than asking the model to emit JSON as prose and hoping it parses,
//! acton-ai appends a synthetic tool named [`STRUCTURED_OUTPUT_TOOL`] whose
//! input schema *is* the schema of your type, then constrains the request
//! with [`ToolChoice`](crate::messages::ToolChoice) so the model has to call
//! it. The arguments of that call are the answer. If they don't deserialize
//! into your type, the loop feeds the serde error back to the model as a tool
//! result and asks it to try again, up to [`MAX_VALIDATION_REPAIRS`] times.
//!
//! # schemars API notes
//!
//! Pinned against **schemars 1.2.2** (verified in
//! `~/.cargo/registry/src/*/schemars-1.2.2/src/generate.rs`):
//!
//! - `SchemaSettings` lives in `schemars::generate`, not at the crate root.
//! - `SchemaSettings::inline_subschemas: bool` is the field that stops
//!   `$ref`/`$defs` from being emitted; `SchemaSettings::with` takes an
//!   `FnOnce(&mut Self)` to set it.
//! - `SchemaSettings::into_generator()` yields a `SchemaGenerator`, and
//!   `SchemaGenerator::into_root_schema_for::<T>()` yields a
//!   `schemars::Schema`, which is a newtype over `serde_json::Value` with a
//!   consuming `to_value()`.
//!
//! Subschemas are inlined because provider support for `$ref`/`$defs` inside
//! a tool's `input_schema` is inconsistent; a self-contained schema works
//! everywhere. Recursive types are the one case schemars still has to emit a
//! reference for, since inlining them would not terminate.

use crate::error::ActonAIError;
use crate::messages::ToolDefinition;
use schemars::generate::SchemaSettings;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::sync::Arc;

/// Name of the synthetic tool the model calls to record its final answer.
///
/// A prompt may not register a tool of its own under this name — see
/// [`ensure_name_is_available`].
pub const STRUCTURED_OUTPUT_TOOL: &str = "structured_output";

/// Description attached to the synthetic tool.
const STRUCTURED_OUTPUT_DESCRIPTION: &str =
    "Record your final answer. Call this tool exactly once, with the complete answer matching the schema.";

/// How many times the loop will hand a validation error back to the model
/// and ask for corrected arguments before giving up.
///
/// The budget is per extraction, not per round: with the default of 2, a run
/// that never produces valid arguments issues three requests in total (the
/// first attempt plus two repairs) and then fails.
pub const MAX_VALIDATION_REPAIRS: usize = 2;

/// Longest raw-argument dump included in an exhausted-repair error.
const RAW_ARGUMENTS_DUMP_LIMIT: usize = 500;

/// Type-erased check that a candidate answer deserializes into the caller's
/// target type.
type Validator = Arc<dyn Fn(&serde_json::Value) -> Result<(), String> + Send + Sync>;

/// A monomorphic description of "the caller wants a `T` back".
///
/// The prompt loop is deliberately non-generic: it holds one of these instead
/// of a type parameter, so a single `collect`-shaped code path serves both
/// plain and typed prompts. The type parameter survives only inside
/// [`Self::validate`], which closes over `T` without naming it.
#[derive(Clone)]
pub(crate) struct StructuredSpec {
    /// JSON Schema of the target type, with subschemas inlined.
    schema: serde_json::Value,
    /// Checks a candidate answer without the loop knowing the target type.
    validate: Validator,
}

impl std::fmt::Debug for StructuredSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructuredSpec")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl StructuredSpec {
    /// Builds the spec for a target type.
    pub(crate) fn for_type<T>() -> Self
    where
        T: DeserializeOwned + JsonSchema,
    {
        Self {
            schema: inlined_schema_for::<T>(),
            // `T` is not `Clone`, so the parsed value is simply dropped; all
            // this closure reports is whether parsing would succeed.
            validate: Arc::new(|value: &serde_json::Value| {
                deserialization_error::<T>(value).map_or(Ok(()), Err)
            }),
        }
    }

    /// The synthetic tool definition to append to the request.
    pub(crate) fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: STRUCTURED_OUTPUT_TOOL.to_string(),
            description: STRUCTURED_OUTPUT_DESCRIPTION.to_string(),
            input_schema: self.schema.clone(),
        }
    }

    /// Reports whether `value` deserializes into the target type, returning
    /// the serde error message when it does not.
    pub(crate) fn validate(&self, value: &serde_json::Value) -> Result<(), String> {
        (self.validate)(value)
    }
}

/// Attempts to deserialize `value` into `T`, returning a path-qualified
/// error message when it fails.
///
/// Plain [`serde_json::from_value`] reports type mismatches without saying
/// *which* field mismatched — `"invalid type: string, expected u64"` is
/// nearly useless feedback for a type with three `u64` fields. Routing the
/// same deserialization through [`serde_path_to_error`] prefixes the element
/// path (`line_items[0].cents: invalid type: …`), which is what makes the
/// repair round worth running: the model is told exactly what to fix.
fn deserialization_error<T>(value: &serde_json::Value) -> Option<String>
where
    T: DeserializeOwned,
{
    // The parsed value is discarded — this only reports parseability.
    serde_path_to_error::deserialize::<_, T>(value)
        .err()
        .map(|error| error.to_string())
}

/// Generates a self-contained JSON Schema for `T`.
///
/// Subschemas are inlined so the result carries no `$ref`/`$defs`, and the
/// `$schema` meta key is dropped: it describes the schema document itself,
/// not the tool's parameters, and some providers reject unexpected top-level
/// keys inside a tool's `input_schema`.
#[must_use]
pub fn inlined_schema_for<T>() -> serde_json::Value
where
    T: JsonSchema,
{
    let mut value = SchemaSettings::default()
        .with(|settings| settings.inline_subschemas = true)
        .into_generator()
        .into_root_schema_for::<T>()
        .to_value();

    if let Some(object) = value.as_object_mut() {
        object.remove("$schema");
    }

    value
}

/// Rejects a prompt whose own tools already claim
/// [`STRUCTURED_OUTPUT_TOOL`].
///
/// Silently shadowing the caller's tool would make one of the two
/// unreachable, and which one won would depend on ordering — so this is an
/// error the caller has to resolve by renaming their tool.
///
/// # Errors
///
/// Returns a configuration error naming the conflicting tool.
pub(crate) fn ensure_name_is_available<'a>(
    tool_names: impl IntoIterator<Item = &'a str>,
) -> Result<(), ActonAIError> {
    if tool_names
        .into_iter()
        .any(|name| name == STRUCTURED_OUTPUT_TOOL)
    {
        return Err(ActonAIError::configuration(
            "tool",
            format!(
                "a tool named '{STRUCTURED_OUTPUT_TOOL}' is already registered on this prompt, \
                 which extract() needs for the answer; rename that tool"
            ),
        ));
    }
    Ok(())
}

/// Builds the tool result that asks the model to correct invalid arguments.
///
/// Kept next to the validator so the wording the model reacts to lives with
/// the code that decides to send it.
#[must_use]
pub(crate) fn validation_feedback(serde_error: &str) -> String {
    format!(
        "Validation failed: {serde_error}. \
         Call {STRUCTURED_OUTPUT_TOOL} again with corrected arguments."
    )
}

/// Builds the error returned when the repair budget is exhausted.
///
/// Includes both the last serde error and a truncated dump of the arguments
/// the model actually produced, so the caller can see the mismatch without
/// re-running with logging turned up.
#[must_use]
pub(crate) fn repairs_exhausted_error(
    serde_error: &str,
    last_arguments: &serde_json::Value,
) -> ActonAIError {
    let rendered =
        serde_json::to_string(last_arguments).unwrap_or_else(|_| "<unrenderable>".to_string());
    ActonAIError::extraction(format!(
        "validation exhausted after {attempts} attempts: {serde_error}; \
         last arguments were {dump}",
        attempts = MAX_VALIDATION_REPAIRS + 1,
        dump = truncate(&rendered, RAW_ARGUMENTS_DUMP_LIMIT),
    ))
}

/// Truncates on a character boundary, marking the cut with an ellipsis.
fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct LineItem {
        description: String,
        cents: u64,
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct Invoice {
        vendor: String,
        total_cents: u64,
        line_items: Vec<LineItem>,
    }

    /// Walks a JSON value looking for any `$ref` key at any depth.
    fn contains_ref(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(map) => {
                map.contains_key("$ref") || map.values().any(contains_ref)
            }
            serde_json::Value::Array(items) => items.iter().any(contains_ref),
            _ => false,
        }
    }

    #[test]
    fn nested_schema_is_fully_inlined() {
        let schema = inlined_schema_for::<Invoice>();

        assert!(
            !contains_ref(&schema),
            "schema must not contain any $ref: {schema:#}"
        );
        assert!(
            schema.get("$defs").is_none(),
            "schema must not carry a $defs section: {schema:#}"
        );
        // The nested type's own fields must have been pulled in, which is
        // what makes the schema self-contained.
        let items = &schema["properties"]["line_items"]["items"];
        assert_eq!(items["type"], "object", "nested item schema: {schema:#}");
        assert!(items["properties"].get("description").is_some());
        assert!(items["properties"].get("cents").is_some());
    }

    #[test]
    fn schema_describes_the_top_level_object() {
        let schema = inlined_schema_for::<Invoice>();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"].get("vendor").is_some());
        assert!(schema["properties"].get("total_cents").is_some());
    }

    #[test]
    fn schema_drops_the_meta_schema_key() {
        let schema = inlined_schema_for::<Invoice>();

        assert!(
            schema.get("$schema").is_none(),
            "the $schema meta key belongs to the document, not the tool \
             parameters: {schema:#}"
        );
    }

    #[test]
    fn tool_definition_carries_the_schema_and_the_reserved_name() {
        let spec = StructuredSpec::for_type::<Invoice>();
        let definition = spec.tool_definition();

        assert_eq!(definition.name, STRUCTURED_OUTPUT_TOOL);
        assert!(!definition.description.is_empty());
        assert_eq!(definition.input_schema, inlined_schema_for::<Invoice>());
    }

    #[test]
    fn a_validated_answer_deserializes_into_the_target_type() {
        let answer = serde_json::json!({
            "vendor": "Acme",
            "total_cents": 1250,
            "line_items": [{"description": "widget", "cents": 1250}],
        });

        StructuredSpec::for_type::<Invoice>()
            .validate(&answer)
            .expect("fixture must validate");
        let invoice: Invoice = serde_json::from_value(answer).expect("validated value must parse");

        assert_eq!(invoice.vendor, "Acme");
        assert_eq!(invoice.total_cents, 1250);
        assert_eq!(invoice.line_items.len(), 1);
        assert_eq!(invoice.line_items[0].description, "widget");
        assert_eq!(invoice.line_items[0].cents, 1250);
    }

    #[test]
    fn validate_accepts_a_well_formed_answer() {
        let spec = StructuredSpec::for_type::<Invoice>();

        let result = spec.validate(&serde_json::json!({
            "vendor": "Acme",
            "total_cents": 1250,
            "line_items": [{"description": "widget", "cents": 1250}],
        }));

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn validate_reports_the_serde_error_for_a_type_mismatch() {
        let spec = StructuredSpec::for_type::<Invoice>();

        let error = spec
            .validate(&serde_json::json!({
                "vendor": "Acme",
                "total_cents": "twelve fifty",
                "line_items": [],
            }))
            .expect_err("a string in a u64 field must not validate");

        assert_eq!(
            error, "total_cents: invalid type: string \"twelve fifty\", expected u64",
            "the error must name the offending field so the repair round is actionable"
        );
    }

    #[test]
    fn validate_names_the_path_of_a_nested_field() {
        let spec = StructuredSpec::for_type::<Invoice>();

        let error = spec
            .validate(&serde_json::json!({
                "vendor": "Acme",
                "total_cents": 1250,
                "line_items": [{"description": "widget", "cents": "free"}],
            }))
            .expect_err("a string in a nested u64 field must not validate");

        assert!(
            error.starts_with("line_items[0].cents:"),
            "the error must carry the full element path: {error}"
        );
    }

    #[test]
    fn validate_reports_a_missing_field() {
        let spec = StructuredSpec::for_type::<Invoice>();

        let error = spec
            .validate(&serde_json::json!({"vendor": "Acme"}))
            .expect_err("a missing required field must not validate");

        assert!(
            error.contains("total_cents"),
            "the error should name the missing field: {error}"
        );
    }

    #[test]
    fn name_is_available_when_no_tool_claims_it() {
        assert_eq!(
            ensure_name_is_available(["calculator", "read_file"]),
            Ok(())
        );
    }

    #[test]
    fn name_collision_is_rejected_rather_than_shadowed() {
        let error = ensure_name_is_available(["calculator", STRUCTURED_OUTPUT_TOOL])
            .expect_err("a user tool claiming the reserved name must be rejected");

        assert!(error.is_configuration(), "unexpected kind: {error:?}");
        assert!(
            error.to_string().contains(STRUCTURED_OUTPUT_TOOL),
            "the error should name the conflict: {error}"
        );
    }

    #[test]
    fn validation_feedback_quotes_the_error_and_asks_for_a_retry() {
        let feedback = validation_feedback("invalid type: string, expected u64");

        assert!(feedback.contains("Validation failed"));
        assert!(feedback.contains("invalid type: string, expected u64"));
        assert!(feedback.contains(STRUCTURED_OUTPUT_TOOL));
    }

    #[test]
    fn repairs_exhausted_error_carries_the_serde_error_and_the_arguments() {
        let error = repairs_exhausted_error(
            "invalid type: string, expected u64",
            &serde_json::json!({"total_cents": "twelve fifty"}),
        );

        let message = error.to_string();
        assert!(message.contains("validation exhausted"), "{message}");
        assert!(
            message.contains("invalid type: string, expected u64"),
            "{message}"
        );
        assert!(message.contains("twelve fifty"), "{message}");
    }

    #[test]
    fn repairs_exhausted_error_truncates_a_huge_argument_dump() {
        let bloated = serde_json::json!({"note": "x".repeat(5_000)});

        let message = repairs_exhausted_error("bad", &bloated).to_string();

        assert!(message.contains('…'), "dump should be elided: {message}");
        assert!(
            message.chars().count() < RAW_ARGUMENTS_DUMP_LIMIT + 200,
            "message should stay compact, got {} chars",
            message.chars().count()
        );
    }

    #[test]
    fn truncate_leaves_short_text_alone() {
        assert_eq!(truncate("short", 10), "short");
    }

    #[test]
    fn truncate_respects_character_boundaries() {
        assert_eq!(truncate("ααααα", 3), "ααα…");
    }
}
