//! Support code for macro-generated tool implementations. **Not a public API.**
//!
//! Everything here exists so that the code [`#[tool]`](macro@crate::tool)
//! expands to can stay small and free of logic. Generated code runs in the
//! *user's* crate, where the only thing guaranteed to resolve is
//! `::acton_ai::…` — so anything the expansion needs has to be reachable
//! through this module rather than inlined into the expansion, where a bug
//! would be frozen into every downstream build until they upgrade.
//!
//! # Stability
//!
//! These items are `#[doc(hidden)]` and exempt from semver. Call them from
//! macro expansions only. If you find yourself reaching for one by hand, the
//! thing you actually want is public: [`crate::extract::inlined_schema_for`]
//! for schemas, and [`crate::tools::Tool`] for the trait itself.

use crate::tools::ToolError;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

/// Generates the schema for one parameter of a generated tool.
///
/// A thin alias for [`crate::extract::inlined_schema_for`], which is the one
/// place in the crate that decides how a Rust type becomes a JSON Schema —
/// subschemas inlined, `$schema` dropped. Generated tools and
/// [`extract`](crate::prompt::PromptBuilder::extract) go through the same
/// function on purpose: a tool's schema and an extraction schema are read by
/// the same providers, so they must not drift apart.
#[doc(hidden)]
#[must_use]
pub fn inline_schema_for<T>() -> serde_json::Value
where
    T: JsonSchema,
{
    crate::extract::inlined_schema_for::<T>()
}

/// Assembles an object schema from a generated tool's parameter list.
///
/// `parameters` is `(name, schema, required)` per parameter, in declaration
/// order. `required` is decided syntactically by the macro: a parameter whose
/// type is spelled `Option<…>` is optional, everything else is required.
///
/// The `required` array comes out in signature order, because it is built
/// from a `Vec`. The `properties` object does **not**: `serde_json::Map` is a
/// `BTreeMap` unless the `preserve_order` feature is on, so properties are
/// serialized alphabetically. That is only cosmetic — JSON object members are
/// unordered by specification and every provider treats them as such — and
/// turning on `preserve_order` to fix it would change `serde_json`'s behavior
/// for the entire dependency graph, which is a steep price for the ordering
/// of a schema no human reads.
#[doc(hidden)]
#[must_use]
pub fn object_schema(
    parameters: Vec<(&'static str, serde_json::Value, bool)>,
) -> serde_json::Value {
    let mut properties = serde_json::Map::with_capacity(parameters.len());
    let mut required = Vec::new();

    for (name, schema, is_required) in parameters {
        if is_required {
            required.push(serde_json::Value::String(name.to_string()));
        }
        properties.insert(name.to_string(), schema);
    }

    serde_json::json!({
        "type": "object",
        "properties": serde_json::Value::Object(properties),
        "required": serde_json::Value::Array(required),
    })
}

/// Pulls one argument out of a tool call's arguments and deserializes it.
///
/// Called once per parameter by generated `Tool::call` bodies, in signature
/// order. Three things can go wrong, and all three produce a [`ToolError`]
/// that names the offending parameter — a model that gets an argument wrong
/// can only fix it if the error says which one:
///
/// 1. `args` is not a JSON object at all.
/// 2. A required parameter is absent. An *optional* parameter (`Option<T>`)
///    is absent legitimately, which is why `required` is passed in: absent
///    plus optional deserializes `null`, which is `None`.
/// 3. The value is present but the wrong shape. The error comes from
///    [`serde_path_to_error`], so a mismatch nested inside a struct parameter
///    reports the full path (`config.retries: invalid type: …`) rather than a
///    bare "invalid type" with no indication of where.
///
/// # Errors
///
/// Returns [`ToolError::validation_failed`] naming `tool` and `parameter`.
#[doc(hidden)]
pub fn argument<T>(
    tool: &'static str,
    parameter: &'static str,
    required: bool,
    args: &serde_json::Value,
) -> Result<T, ToolError>
where
    T: DeserializeOwned,
{
    let Some(object) = args.as_object() else {
        return Err(ToolError::validation_failed(
            tool,
            format!(
                "arguments must be a JSON object with a '{parameter}' field, got {}",
                kind_of(args)
            ),
        ));
    };

    let value = match object.get(parameter) {
        Some(value) => value,
        None if required => {
            return Err(ToolError::validation_failed(
                tool,
                format!("missing required parameter '{parameter}'"),
            ));
        }
        // An absent optional parameter is `null`, which is how `Option<T>`
        // spells `None`.
        None => &serde_json::Value::Null,
    };

    serde_path_to_error::deserialize::<_, T>(value).map_err(|error| {
        ToolError::validation_failed(tool, format!("parameter '{parameter}': {error}"))
    })
}

/// Serializes a generated tool's success value into the JSON the loop sends
/// back to the model.
///
/// The tool body returns `Result<T, ToolError>` for any `T: Serialize`, but
/// the prompt loop speaks [`serde_json::Value`]. A serialization failure here
/// is the tool author's bug, not the model's, so it surfaces as
/// [`ToolError::execution_failed`] rather than a validation error.
///
/// # Errors
///
/// Returns [`ToolError::execution_failed`] if `value` cannot be serialized.
#[doc(hidden)]
pub fn output<T>(tool: &'static str, value: T) -> Result<serde_json::Value, ToolError>
where
    T: serde::Serialize,
{
    serde_json::to_value(value).map_err(|error| {
        ToolError::execution_failed(
            tool,
            format!("could not serialize the tool's result: {error}"),
        )
    })
}

/// Names a JSON value's type for an error message.
fn kind_of(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema, PartialEq, Eq)]
    struct Config {
        retries: u8,
    }

    #[test]
    fn object_schema_lists_required_parameters_only() {
        let schema = object_schema(vec![
            ("expr", serde_json::json!({"type": "string"}), true),
            ("precision", serde_json::json!({"type": "integer"}), false),
        ]);

        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"], serde_json::json!(["expr"]));
        assert_eq!(schema["properties"]["expr"]["type"], "string");
        assert_eq!(schema["properties"]["precision"]["type"], "integer");
    }

    #[test]
    fn object_schema_with_no_parameters_is_an_empty_object_schema() {
        let schema = object_schema(Vec::new());

        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"], serde_json::json!([]));
        assert_eq!(schema["properties"], serde_json::json!({}));
    }

    #[test]
    fn object_schema_lists_required_in_signature_order() {
        // Not alphabetical: `zebra` is declared first, so it is listed first.
        // The model is told which fields are mandatory, and reading that list
        // in signature order costs nothing and matches the source.
        let schema = object_schema(vec![
            ("zebra", serde_json::json!({"type": "string"}), true),
            ("alpha", serde_json::json!({"type": "string"}), true),
        ]);

        assert_eq!(schema["required"], serde_json::json!(["zebra", "alpha"]));
    }

    #[test]
    fn argument_deserializes_a_present_value() {
        let args = serde_json::json!({"expr": "2 + 2"});

        let value: String = argument("calculator", "expr", true, &args).expect("must deserialize");

        assert_eq!(value, "2 + 2");
    }

    #[test]
    fn argument_reports_a_missing_required_parameter_by_name() {
        let args = serde_json::json!({"other": 1});

        let error = argument::<String>("calculator", "expr", true, &args)
            .expect_err("a missing required parameter must be rejected");

        let message = error.to_string();
        assert!(
            message.contains("expr"),
            "must name the parameter: {message}"
        );
        assert!(message.contains("missing"), "{message}");
    }

    #[test]
    fn argument_treats_a_missing_optional_parameter_as_none() {
        let args = serde_json::json!({});

        let value: Option<u8> = argument("calculator", "precision", false, &args)
            .expect("absent optional must be None");

        assert_eq!(value, None);
    }

    #[test]
    fn argument_reports_a_type_mismatch_by_name() {
        let args = serde_json::json!({"precision": "lots"});

        let error = argument::<u8>("calculator", "precision", true, &args)
            .expect_err("a string in a u8 parameter must be rejected");

        let message = error.to_string();
        assert!(
            message.contains("precision"),
            "must name the parameter: {message}"
        );
        assert!(message.contains("invalid type"), "{message}");
    }

    #[test]
    fn argument_reports_the_path_inside_a_struct_parameter() {
        let args = serde_json::json!({"config": {"retries": "many"}});

        let error = argument::<Config>("calculator", "config", true, &args)
            .expect_err("a bad nested field must be rejected");

        let message = error.to_string();
        assert!(
            message.contains("config") && message.contains("retries"),
            "must carry the full path: {message}"
        );
    }

    #[test]
    fn argument_rejects_non_object_arguments() {
        let args = serde_json::json!("not an object");

        let error = argument::<String>("calculator", "expr", true, &args)
            .expect_err("a non-object arguments value must be rejected");

        let message = error.to_string();
        assert!(message.contains("expr"), "{message}");
        assert!(message.contains("a string"), "{message}");
    }

    #[test]
    fn output_serializes_a_success_value() {
        let value =
            output("calculator", serde_json::json!({"result": 4})).expect("a Value must serialize");

        assert_eq!(value, serde_json::json!({"result": 4}));
    }

    #[test]
    fn output_serializes_a_plain_rust_value() {
        let value = output("calculator", 4_u8).expect("a u8 must serialize");

        assert_eq!(value, serde_json::json!(4));
    }

    #[test]
    fn inline_schema_for_matches_the_extraction_schema() {
        assert_eq!(
            inline_schema_for::<Config>(),
            crate::extract::inlined_schema_for::<Config>(),
            "tool schemas and extraction schemas must not drift apart"
        );
    }

    #[test]
    fn inline_schema_for_carries_no_refs() {
        let schema = inline_schema_for::<Config>();

        assert!(schema.get("$defs").is_none(), "{schema:#}");
        assert!(schema.get("$ref").is_none(), "{schema:#}");
    }

    #[test]
    fn kind_of_names_each_json_type() {
        assert_eq!(kind_of(&serde_json::Value::Null), "null");
        assert_eq!(kind_of(&serde_json::json!(true)), "a boolean");
        assert_eq!(kind_of(&serde_json::json!(1)), "a number");
        assert_eq!(kind_of(&serde_json::json!("x")), "a string");
        assert_eq!(kind_of(&serde_json::json!([])), "an array");
        assert_eq!(kind_of(&serde_json::json!({})), "an object");
    }
}
