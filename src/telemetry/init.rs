//! Building and installing the tracer and meter providers.
//!
//! # The injectable seam
//!
//! [`init_telemetry`] builds OTLP exporters and hands them to
//! [`install_with_exporters`], which is where the providers are actually
//! assembled and installed as globals. Splitting it that way is what lets the
//! integration tests drive the real installation path with in-memory
//! exporters instead of a socket — the same trick
//! [`crate::mcp::connector`] uses to make transports injectable.
//!
//! # Why the OTLP transport is HTTP/protobuf over a blocking client
//!
//! `BatchSpanProcessor` and `PeriodicReader` both export from a dedicated
//! `std::thread`, driving the exporter future with
//! `futures_executor::block_on`. That thread has no tokio reactor, so an
//! async HTTP client's IO would never be driven and every export would fail
//! at runtime — invisibly, since nothing on the hot path waits for the
//! result. reqwest's blocking client owns its own runtime and is therefore
//! the transport that actually works here. gRPC/tonic is not involved at all.

use crate::error::ActonAIError;
use crate::telemetry::TelemetryConfig;
use opentelemetry::global;
use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanExporter};
use opentelemetry_sdk::Resource;

/// Holds the installed providers so they can be flushed and shut down.
///
/// The facade keeps one for the lifetime of the runtime. Dropping it flushes
/// as a last resort, but the ordered path is
/// [`ActonAI::shutdown`](crate::facade::ActonAI::shutdown), which flushes
/// *after* the actors stop so their final broadcasts are recorded. The last
/// spans before exit are the ones an operator actually came for.
#[derive(Debug)]
pub struct TelemetryGuard {
    tracer_provider: SdkTracerProvider,
    meter_provider: SdkMeterProvider,
    /// Cleared by [`Self::shutdown`] so `Drop` does not shut down twice.
    live: bool,
}

impl TelemetryGuard {
    /// Flushes both providers without shutting them down.
    ///
    /// Used by tests to make a batch observable without waiting out an
    /// export interval, and available to callers who want a checkpoint.
    pub fn flush(&self) {
        // A failed flush is worth a log and nothing more: telemetry that
        // cannot be delivered must never take the application down with it.
        if let Err(e) = self.tracer_provider.force_flush() {
            tracing::debug!(error = %e, "flushing spans failed");
        }
        if let Err(e) = self.meter_provider.force_flush() {
            tracing::debug!(error = %e, "flushing metrics failed");
        }
    }

    /// Flushes and shuts down both providers.
    pub fn shutdown(&mut self) {
        if !self.live {
            return;
        }
        self.live = false;
        self.flush();
        if let Err(e) = self.tracer_provider.shutdown() {
            tracing::debug!(error = %e, "shutting down the tracer provider failed");
        }
        if let Err(e) = self.meter_provider.shutdown() {
            tracing::debug!(error = %e, "shutting down the meter provider failed");
        }
    }
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Builds OTLP exporters from `config` and installs the providers.
///
/// # Errors
///
/// Returns a configuration error when either exporter cannot be built, which
/// in practice means the endpoint or a header value was not acceptable to the
/// HTTP layer.
pub(crate) fn init_telemetry(config: &TelemetryConfig) -> Result<TelemetryGuard, ActonAIError> {
    use opentelemetry_otlp::{WithExportConfig as _, WithHttpConfig as _};

    let headers: std::collections::HashMap<String, String> = config
        .headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let span_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(format!(
            "{}/v1/traces",
            config.otlp_endpoint.trim_end_matches('/')
        ))
        .with_headers(headers.clone())
        .build()
        .map_err(|e| {
            ActonAIError::configuration(
                "telemetry.otlp_endpoint",
                format!("could not build the OTLP span exporter: {e}"),
            )
        })?;

    let metric_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .with_endpoint(format!(
            "{}/v1/metrics",
            config.otlp_endpoint.trim_end_matches('/')
        ))
        .with_headers(headers)
        .build()
        .map_err(|e| {
            ActonAIError::configuration(
                "telemetry.otlp_endpoint",
                format!("could not build the OTLP metric exporter: {e}"),
            )
        })?;

    Ok(install_with_exporters(
        config,
        span_exporter,
        metric_exporter,
    ))
}

/// Assembles both providers around the supplied exporters and installs them
/// as the process-wide globals.
///
/// This is the seam the integration tests use: hand it
/// `opentelemetry_sdk`'s in-memory exporters and the whole instrumentation
/// path runs unchanged with nothing on the network. It is equally the way to
/// export somewhere OTLP does not reach.
///
/// Installing globals is process-wide and last-writer-wins, which is the
/// OpenTelemetry SDK's own model. Tests that assert on exported data should
/// therefore run one runtime per process — `cargo nextest run` does exactly
/// that.
pub fn install_with_exporters<S, M>(
    config: &TelemetryConfig,
    span_exporter: S,
    metric_exporter: M,
) -> TelemetryGuard
where
    S: SpanExporter + 'static,
    M: PushMetricExporter + 'static,
{
    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
        .build();

    let tracer_provider = SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_batch_exporter(span_exporter)
        .build();

    let reader = PeriodicReader::builder(metric_exporter)
        .with_interval(config.metrics_interval)
        .build();

    let meter_provider = SdkMeterProvider::builder()
        .with_resource(resource)
        .with_reader(reader)
        .build();

    global::set_tracer_provider(tracer_provider.clone());
    global::set_meter_provider(meter_provider.clone());

    TelemetryGuard {
        tracer_provider,
        meter_provider,
        live: true,
    }
}
