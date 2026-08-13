//! The [`Tool`] trait: a self-describing, callable tool.
//!
//! A tool the model can call is four things bundled together: a name, a
//! description, an input schema, and something to run. The rest of the crate
//! carries those four as separate arguments — see
//! [`PromptBuilder::tool`](crate::prompt::PromptBuilder::tool). This trait
//! bundles them into one value so a tool can be *defined* in one place and
//! *registered* in another, and so the [`#[tool]`](macro@crate::tool)
//! attribute macro has something to generate an implementation of.
//!
//! # Implementing by hand
//!
//! ```rust
//! use acton_ai::prelude::*;
//! use acton_ai::tools::{Tool, ToolFuture};
//!
//! struct Echo;
//!
//! impl Tool for Echo {
//!     fn name(&self) -> &'static str {
//!         "echo"
//!     }
//!
//!     fn description(&self) -> &'static str {
//!         "Repeats the text it is given."
//!     }
//!
//!     fn input_schema(&self) -> serde_json::Value {
//!         serde_json::json!({
//!             "type": "object",
//!             "properties": {"text": {"type": "string"}},
//!             "required": ["text"],
//!         })
//!     }
//!
//!     fn call(&self, args: serde_json::Value) -> ToolFuture {
//!         Box::pin(async move { Ok(args) })
//!     }
//! }
//! ```
//!
//! Writing that by hand is exactly what the [`#[tool]`](macro@crate::tool)
//! attribute exists to avoid.

use crate::tools::ToolError;
use std::future::Future;
use std::pin::Pin;

/// The boxed future a [`Tool::call`] returns.
///
/// Tools are held behind `dyn Tool`, so `call` cannot be an `async fn` — it
/// returns an erased future instead. Inside a [`Tool::call`] body, write the
/// work as an ordinary `async` block and wrap it in `Box::pin`.
pub type ToolFuture =
    Pin<Box<dyn Future<Output = Result<serde_json::Value, ToolError>> + Send + 'static>>;

/// A tool the model can be offered: name, description, input schema, and the
/// code that runs when the model calls it.
///
/// Register one on a prompt with
/// [`PromptBuilder::add_tool`](crate::prompt::PromptBuilder::add_tool).
///
/// Implementations are normally *generated* rather than written: annotate an
/// `async fn` with [`#[tool]`](macro@crate::tool) and the macro derives all
/// four methods from the signature and doc comment. See the module docs for
/// what a hand-written implementation looks like.
///
/// # Contract
///
/// - [`name`](Self::name) must be stable for the life of the value and unique
///   among the tools registered on a single prompt. It is what the model
///   sends back when it decides to call the tool.
/// - [`input_schema`](Self::input_schema) must be a JSON Schema *object*
///   schema. Providers differ on `$ref`/`$defs` support, so prefer a
///   self-contained schema — which is what
///   [`inlined_schema_for`](crate::extract::inlined_schema_for) produces.
/// - [`call`](Self::call) receives whatever arguments the model produced.
///   They are **not** pre-validated against the schema: models get this wrong,
///   so treat the value as untrusted and report bad input as a
///   [`ToolError`] rather than panicking.
pub trait Tool: Send + Sync + 'static {
    /// The tool's name, as the model sees it.
    fn name(&self) -> &'static str;

    /// What the tool does, as the model sees it.
    ///
    /// This is the single most important field for whether the model calls
    /// the tool correctly, or at all.
    fn description(&self) -> &'static str;

    /// The JSON Schema describing the tool's arguments.
    fn input_schema(&self) -> serde_json::Value;

    /// Runs the tool against the model-supplied `args`.
    fn call(&self, args: serde_json::Value) -> ToolFuture;
}
