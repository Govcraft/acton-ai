//! Procedural macros for [acton-ai](https://docs.rs/acton-ai).
//!
//! This crate exists only to back `acton_ai::tool`. Depend on `acton-ai`
//! (which re-exports the macro under its default `derive` feature) rather
//! than on this crate directly — the macro's expansion names `::acton_ai::…`
//! paths and will not compile without it.
//!
//! # Why the examples here are `ignore`d
//!
//! For the same reason. `acton-ai` depends on this crate, so this crate
//! cannot depend on `acton-ai` to compile a doc test against — the expansion
//! has nothing to resolve `::acton_ai::` to. The examples below are therefore
//! displayed but not run. They *are* compiled and executed as doc tests on
//! the re-exporting side, where they resolve normally: see
//! `PromptBuilder::add_tool` and the `Tool` trait in `acton-ai`, plus the
//! `trybuild` pass case in that crate's `tests/ui/`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod tool;

use proc_macro::TokenStream;

/// Turns an `async fn` into a callable, self-describing LLM tool.
///
/// Defining a tool by hand means writing a name, a description, a JSON Schema,
/// and a closure that plucks arguments out of a `serde_json::Value` — four
/// things that all restate the same function signature, and that drift apart
/// the first time the signature changes. `#[tool]` derives all four from the
/// function you already wrote.
///
/// # What it generates
///
/// The annotated function is emitted **unchanged**, so it stays an ordinary
/// function you can call and unit-test directly. Alongside it, the macro adds
/// a unit struct named after the function in `PascalCase` (`calculator` →
/// `Calculator`, `read_file` → `ReadFile`) implementing
/// [`Tool`](../acton_ai/tools/trait.Tool.html). The struct inherits the
/// function's visibility.
///
/// ```rust,ignore
/// use acton_ai::prelude::*;
///
/// /// Adds two numbers and returns the sum.
/// #[tool]
/// async fn add(a: i64, b: i64) -> Result<serde_json::Value, ToolError> {
///     Ok(serde_json::json!({ "sum": a + b }))
/// }
///
/// # async fn run(runtime: ActonAI) -> Result<(), ActonAIError> {
/// let response = runtime
///     .prompt("What is 42 + 17?")
///     .add_tool(Add)
///     .collect()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// # The contract
///
/// | Generated | Comes from |
/// |---|---|
/// | `name()` | the function name, verbatim — snake_case is what LLM APIs expect |
/// | `description()` | the `///` doc comment, lines joined and trimmed |
/// | `input_schema()` | one property per parameter, named after the parameter |
/// | `call()` | deserializes each argument, then awaits your function |
///
/// The function must:
///
/// - be `async`, and a free function — no `self` receiver, no generics;
/// - carry a doc comment. This is a hard error, not a warning: the
///   description is the only thing the model reads when deciding whether to
///   call your tool, and an undescribed tool silently never gets called;
/// - return `Result<T, ToolError>` where `T: serde::Serialize`. The
///   `Ok` value is serialized with `serde_json::to_value`;
/// - take parameters that are each `serde::de::DeserializeOwned +
///   schemars::JsonSchema`. Zero parameters is fine and yields an empty
///   object schema.
///
/// # Optional parameters
///
/// A parameter is optional — omitted from the schema's `required` list — when
/// its type is *spelled* `Option<…>`:
///
/// ```rust,ignore
/// use acton_ai::prelude::*;
///
/// /// Formats a number to a given precision.
/// #[tool]
/// async fn format_number(value: f64, precision: Option<u8>) -> Result<String, ToolError> {
///     Ok(format!("{value:.*}", usize::from(precision.unwrap_or(2))))
/// }
/// ```
///
/// The test is syntactic, because a macro sees tokens and not types. A type
/// alias hides it:
///
/// ```rust,ignore
/// type MaybePrecision = Option<u8>;
///
/// /// ...
/// #[tool]
/// async fn f(precision: MaybePrecision) -> Result<String, ToolError> { /* ... */ }
/// //         ^ treated as REQUIRED — the macro cannot see through the alias
/// ```
///
/// Write `Option<u8>` in the signature when you mean optional. `std::option::Option<u8>`
/// and `core::option::Option<u8>` are recognized too.
///
/// # Errors the model sees
///
/// Arguments arrive from a language model, so they are wrong sometimes. Each
/// one is deserialized individually and any failure becomes a `ToolError`
/// naming the parameter — including the path inside a struct parameter, so
/// `config.retries: invalid type: string "many", expected u8` comes back
/// rather than a bare "invalid type". That specificity is what lets the model
/// correct itself on the next round instead of retrying the same mistake.
///
/// # Errors
///
/// Rejects, with a compile error pointing at the offending code: a missing
/// doc comment, a non-`async` function, a `self` receiver, generic
/// parameters, a return type that is not spelled `Result<_, ToolError>`,
/// a destructuring parameter pattern, and any argument passed to the
/// attribute itself.
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    tool::expand(args.into(), input.into()).into()
}
