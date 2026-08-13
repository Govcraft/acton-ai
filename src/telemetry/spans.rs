//! Span helpers for the prompt loop.
//!
//! # Shape
//!
//! Each type here is one public surface with a feature-gated inside, so
//! instrumentation sites read as ordinary Rust and carry no `#[cfg]`. With
//! the `otel` feature off every method is an empty body; with it on but no
//! tracer provider installed, the global tracer is a no-op and the cost is a
//! call.
//!
//! # Parenting
//!
//! Nesting is explicit. A [`TurnSpan`] owns a [`Context`] holding its span,
//! and children are started with `start_with_context` against that value —
//! never against an ambient thread-local current context, which does not
//! survive the `await` points and actor hops this loop is made of.
//!
//! ```text
//! acton_ai.turn
//! ├── acton_ai.llm_round
//! ├── acton_ai.tool
//! └── acton_ai.llm_round
//! ```
//!
//! # What is never recorded
//!
//! Tool arguments and tool results. They are user data, unbounded in size,
//! and frequently secret. A tool span carries the tool's name and whether it
//! succeeded, and that is the whole of it.

/// Span names, shared with the tests that assert on them.
///
/// Feature-gated with the spans they name.
#[cfg(feature = "otel")]
pub(crate) mod names {
    /// One whole prompt loop run.
    pub const TURN: &str = "acton_ai.turn";
    /// One provider dispatch within a turn.
    pub const ROUND: &str = "acton_ai.llm_round";
    /// One tool execution within a turn.
    pub const TOOL: &str = "acton_ai.tool";
}

#[cfg(feature = "otel")]
use opentelemetry::trace::{Span as _, Status, TraceContextExt as _, Tracer as _};
#[cfg(feature = "otel")]
use opentelemetry::{global, Context, KeyValue};

/// Instrumentation scope name reported alongside every span.
#[cfg(feature = "otel")]
const SCOPE: &str = "acton-ai";

/// The span covering one whole [`run_prompt_loop`](crate::prompt) run.
///
/// Started as a **root** span against an empty context rather than
/// `Context::current()`: a turn is the top of this crate's trace, and
/// inheriting whatever ambient span a caller happened to leave active would
/// attach agent traffic to unrelated work.
pub(crate) struct TurnSpan {
    #[cfg(feature = "otel")]
    cx: Context,
}

impl TurnSpan {
    /// Starts a turn span for a run against `provider`.
    pub(crate) fn start(provider: &str, structured: bool) -> Self {
        #[cfg(feature = "otel")]
        {
            let tracer = global::tracer(SCOPE);
            let span = tracer
                .span_builder(names::TURN)
                .with_attributes([
                    KeyValue::new("acton_ai.provider", provider.to_string()),
                    KeyValue::new("acton_ai.structured", structured),
                ])
                .start_with_context(&tracer, &Context::new());
            Self {
                cx: Context::new().with_span(span),
            }
        }
        #[cfg(not(feature = "otel"))]
        {
            let _ = (provider, structured);
            Self {}
        }
    }

    /// Starts a round span as a child of this turn.
    pub(crate) fn round(
        &self,
        provider: &str,
        model: &str,
        correlation_id: &str,
        agent_id: &str,
        index: usize,
    ) -> RoundSpan {
        #[cfg(feature = "otel")]
        {
            let tracer = global::tracer(SCOPE);
            let span = tracer
                .span_builder(names::ROUND)
                .with_attributes([
                    KeyValue::new("acton_ai.provider", provider.to_string()),
                    KeyValue::new("acton_ai.model", model.to_string()),
                    KeyValue::new("acton_ai.correlation_id", correlation_id.to_string()),
                    KeyValue::new("acton_ai.agent_id", agent_id.to_string()),
                    KeyValue::new("acton_ai.round", i64::try_from(index).unwrap_or(i64::MAX)),
                ])
                .start_with_context(&tracer, &self.cx);
            RoundSpan { span: Some(span) }
        }
        #[cfg(not(feature = "otel"))]
        {
            let _ = (provider, model, correlation_id, agent_id, index);
            RoundSpan {}
        }
    }

    /// Starts a tool span as a child of this turn.
    pub(crate) fn tool(&self, tool: &str) -> ToolSpan {
        #[cfg(feature = "otel")]
        {
            let tracer = global::tracer(SCOPE);
            let span = tracer
                .span_builder(names::TOOL)
                .with_attributes([KeyValue::new("acton_ai.tool", tool.to_string())])
                .start_with_context(&tracer, &self.cx);
            ToolSpan { span: Some(span) }
        }
        #[cfg(not(feature = "otel"))]
        {
            let _ = tool;
            ToolSpan {}
        }
    }

    /// Closes the turn, recording the totals it accumulated and how it ended.
    ///
    /// `outcome` is `"ok"` for a completed turn, otherwise the error kind —
    /// `"budget_exceeded"` among them, which is the label operators filter on
    /// when a cap starts biting.
    pub(crate) fn finish(
        self,
        rounds: usize,
        usage: &crate::messages::Usage,
        outcome: &'static str,
    ) {
        #[cfg(feature = "otel")]
        {
            let span = self.cx.span();
            span.set_attribute(KeyValue::new(
                "acton_ai.rounds",
                i64::try_from(rounds).unwrap_or(i64::MAX),
            ));
            span.set_attribute(KeyValue::new(
                "acton_ai.tokens.input",
                i64::try_from(usage.input_tokens).unwrap_or(i64::MAX),
            ));
            span.set_attribute(KeyValue::new(
                "acton_ai.tokens.output",
                i64::try_from(usage.output_tokens).unwrap_or(i64::MAX),
            ));
            span.set_attribute(KeyValue::new("acton_ai.outcome", outcome));
            if outcome == crate::telemetry::metrics::OUTCOME_OK {
                span.set_status(Status::Ok);
            } else {
                span.set_status(Status::error(outcome));
            }
            span.end();
        }
        #[cfg(not(feature = "otel"))]
        {
            let _ = (rounds, usage, outcome);
        }
    }
}

/// The span covering one provider dispatch.
pub(crate) struct RoundSpan {
    #[cfg(feature = "otel")]
    span: Option<opentelemetry::global::BoxedSpan>,
}

impl RoundSpan {
    /// Closes the round, recording what the provider reported.
    pub(crate) fn finish(
        mut self,
        usage: &crate::messages::Usage,
        stop_reason: crate::messages::StopReason,
        outcome: &'static str,
    ) {
        #[cfg(feature = "otel")]
        {
            let Some(mut span) = self.span.take() else {
                return;
            };
            span.set_attribute(KeyValue::new(
                "acton_ai.tokens.input",
                i64::try_from(usage.input_tokens).unwrap_or(i64::MAX),
            ));
            span.set_attribute(KeyValue::new(
                "acton_ai.tokens.output",
                i64::try_from(usage.output_tokens).unwrap_or(i64::MAX),
            ));
            span.set_attribute(KeyValue::new(
                "acton_ai.tokens.cache_read",
                i64::try_from(usage.cache_read_tokens).unwrap_or(i64::MAX),
            ));
            span.set_attribute(KeyValue::new(
                "acton_ai.tokens.cache_creation",
                i64::try_from(usage.cache_creation_tokens).unwrap_or(i64::MAX),
            ));
            span.set_attribute(KeyValue::new(
                "acton_ai.stop_reason",
                format!("{stop_reason:?}"),
            ));
            if outcome == crate::telemetry::metrics::OUTCOME_OK {
                span.set_status(Status::Ok);
            } else {
                span.set_status(Status::error(outcome));
            }
            span.end();
        }
        #[cfg(not(feature = "otel"))]
        {
            let _ = (&mut self, usage, stop_reason, outcome);
        }
    }
}

/// The span covering one tool execution.
///
/// Carries the tool's name and its outcome. Never its arguments or its
/// result: both are user data of unbounded size, and a trace backend is not
/// where either belongs.
pub(crate) struct ToolSpan {
    #[cfg(feature = "otel")]
    span: Option<opentelemetry::global::BoxedSpan>,
}

impl ToolSpan {
    /// Closes the tool span.
    pub(crate) fn finish(mut self, outcome: &'static str) {
        #[cfg(feature = "otel")]
        {
            let Some(mut span) = self.span.take() else {
                return;
            };
            span.set_attribute(KeyValue::new("acton_ai.outcome", outcome));
            if outcome == crate::telemetry::metrics::OUTCOME_OK {
                span.set_status(Status::Ok);
            } else {
                span.set_status(Status::error(outcome));
            }
            span.end();
        }
        #[cfg(not(feature = "otel"))]
        {
            let _ = (&mut self, outcome);
        }
    }
}
