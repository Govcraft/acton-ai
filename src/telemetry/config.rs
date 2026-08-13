//! Telemetry configuration: the validated [`TelemetryConfig`] every code
//! path downstream reads, and the fluent [`Telemetry`] builder that produces
//! one programmatically.
//!
//! # Why this module compiles without the `otel` feature
//!
//! [`TelemetryConfig`] and its validation are **unconditional**. A build
//! without `otel` still has to parse `[telemetry]` out of a config file and
//! then refuse to launch, naming the missing feature — see
//! [`crate::telemetry`]. Quietly dropping the section would send an operator
//! hunting for a broken collector when the real answer is a lean build.
//!
//! Only the [`Telemetry`] builder is feature-gated, because a caller writing
//! `.telemetry(..)` in Rust is asking for something the build cannot do, and
//! that is better said at compile time than at launch.

use crate::error::ActonAIError;
use std::collections::BTreeMap;
use std::time::Duration;

/// `service.name` used when none is configured.
pub const DEFAULT_SERVICE_NAME: &str = "acton-ai";

/// How often the periodic metrics reader exports when none is configured.
pub const DEFAULT_METRICS_INTERVAL_SECS: u64 = 60;

/// Validated telemetry settings.
///
/// Produced by [`Telemetry::to_config`] or
/// [`TelemetryFileConfig::to_telemetry`](crate::config::TelemetryFileConfig::to_telemetry).
/// Holding one means the endpoint parsed and the interval is non-zero, so
/// nothing downstream re-validates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelemetryConfig {
    /// Base OTLP HTTP endpoint. Signal paths are appended by the exporter.
    pub otlp_endpoint: String,
    /// `service.name` resource attribute.
    pub service_name: String,
    /// Periodic metrics reader interval.
    pub metrics_interval: Duration,
    /// Headers sent with every OTLP request.
    ///
    /// Values are secrets: render key names, never values.
    pub headers: BTreeMap<String, String>,
}

impl TelemetryConfig {
    /// Validates raw parts and fills in defaults.
    ///
    /// Shared by the TOML path and the builder path so both refuse the same
    /// inputs for the same reasons, rather than drifting into two dialects of
    /// "valid".
    ///
    /// # Errors
    ///
    /// Returns a configuration error when `endpoint` is not a parseable URL
    /// or does not use `http`/`https`, and when `interval_secs` is `Some(0)`.
    pub(crate) fn resolve(
        endpoint: &str,
        service_name: Option<String>,
        interval_secs: Option<u64>,
        headers: BTreeMap<String, String>,
    ) -> Result<Self, ActonAIError> {
        let parsed = url::Url::parse(endpoint).map_err(|e| {
            ActonAIError::configuration(
                "telemetry.otlp_endpoint",
                format!("`{endpoint}` is not a valid URL: {e}"),
            )
        })?;

        // An OTLP/HTTP exporter speaks HTTP and nothing else. Catching a
        // `grpc://` or bare-host typo here beats a collector that silently
        // receives nothing.
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(ActonAIError::configuration(
                "telemetry.otlp_endpoint",
                format!(
                    "`{endpoint}` uses scheme `{}`; the OTLP exporter is HTTP/protobuf, so the \
                     endpoint must be http or https (e.g. \"http://localhost:4318\")",
                    parsed.scheme()
                ),
            ));
        }

        // Zero is not "as often as possible"; it is a reader that either spins
        // or never fires, depending on the SDK's mood. Refuse it outright.
        let interval_secs = interval_secs.unwrap_or(DEFAULT_METRICS_INTERVAL_SECS);
        if interval_secs == 0 {
            return Err(ActonAIError::configuration(
                "telemetry.metrics_interval_secs",
                "the metrics interval must be greater than zero seconds",
            ));
        }

        Ok(Self {
            otlp_endpoint: endpoint.to_string(),
            service_name: service_name.unwrap_or_else(|| DEFAULT_SERVICE_NAME.to_string()),
            metrics_interval: Duration::from_secs(interval_secs),
            headers,
        })
    }

    /// The configured metrics interval in whole seconds.
    #[must_use]
    pub fn metrics_interval_secs(&self) -> u64 {
        self.metrics_interval.as_secs()
    }
}

/// Fluent builder for [`TelemetryConfig`].
///
/// The one-liner form is
/// [`ActonAIBuilder::telemetry_otlp`](crate::facade::ActonAIBuilder::telemetry_otlp);
/// reach for this when you need a service name, a custom interval, or headers.
///
/// ```rust
/// use acton_ai::telemetry::Telemetry;
///
/// let telemetry = Telemetry::otlp("http://localhost:4318")
///     .service_name("my-agent")
///     .metrics_interval_secs(15)
///     .header("authorization", "Bearer hunter2");
///
/// let config = telemetry.to_config().unwrap();
/// assert_eq!(config.service_name, "my-agent");
/// assert_eq!(config.metrics_interval_secs(), 15);
/// ```
#[cfg(feature = "otel")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Telemetry {
    endpoint: String,
    service_name: Option<String>,
    metrics_interval_secs: Option<u64>,
    headers: BTreeMap<String, String>,
}

#[cfg(feature = "otel")]
impl Telemetry {
    /// Exports to the OTLP HTTP endpoint at `endpoint`.
    ///
    /// The endpoint is validated at launch, not here, so a builder chain
    /// stays infallible and every configuration error surfaces from one place.
    #[must_use]
    pub fn otlp(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Self::default()
        }
    }

    /// Sets the `service.name` resource attribute.
    ///
    /// Defaults to [`DEFAULT_SERVICE_NAME`]. This is the name traces and
    /// metrics are grouped under in a backend, so it is worth setting to
    /// something an operator will recognise.
    #[must_use]
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = Some(name.into());
        self
    }

    /// Sets how often the periodic metrics reader exports.
    ///
    /// Defaults to [`DEFAULT_METRICS_INTERVAL_SECS`]. Must be greater than
    /// zero; that is checked at launch.
    #[must_use]
    pub fn metrics_interval_secs(mut self, secs: u64) -> Self {
        self.metrics_interval_secs = Some(secs);
        self
    }

    /// Adds a header sent with every OTLP request.
    ///
    /// Repeat for several headers. The usual use is bearer auth against a
    /// hosted collector.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Validates the builder into a [`TelemetryConfig`].
    ///
    /// # Errors
    ///
    /// Returns a configuration error when the endpoint is not a parseable
    /// `http`/`https` URL, or the metrics interval is zero.
    pub fn to_config(&self) -> Result<TelemetryConfig, ActonAIError> {
        TelemetryConfig::resolve(
            &self.endpoint,
            self.service_name.clone(),
            self.metrics_interval_secs,
            self.headers.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_fill_in_service_name_and_interval() {
        let config = TelemetryConfig::resolve("http://localhost:4318", None, None, BTreeMap::new())
            .expect("a plain http endpoint is valid");

        assert_eq!(config.service_name, DEFAULT_SERVICE_NAME);
        assert_eq!(
            config.metrics_interval_secs(),
            DEFAULT_METRICS_INTERVAL_SECS
        );
        assert!(config.headers.is_empty());
    }

    #[test]
    fn https_endpoints_are_accepted() {
        assert!(TelemetryConfig::resolve(
            "https://collector.example:4318",
            None,
            None,
            BTreeMap::new()
        )
        .is_ok());
    }

    #[test]
    fn a_non_http_scheme_is_refused_and_names_the_key() {
        // The exporter is HTTP/protobuf; a grpc:// endpoint would export
        // nothing at all while looking configured.
        let err = TelemetryConfig::resolve("grpc://localhost:4317", None, None, BTreeMap::new())
            .expect_err("grpc is not an HTTP scheme");

        let rendered = err.to_string();
        assert!(rendered.contains("telemetry.otlp_endpoint"), "{rendered}");
        assert!(rendered.contains("http"), "{rendered}");
    }

    #[test]
    fn an_unparseable_endpoint_is_refused() {
        let err = TelemetryConfig::resolve("localhost:4318", None, None, BTreeMap::new())
            .expect_err("a bare host:port is not a URL");

        assert!(err.to_string().contains("telemetry.otlp_endpoint"));
    }

    #[test]
    fn a_zero_metrics_interval_is_refused() {
        let err = TelemetryConfig::resolve("http://localhost:4318", None, Some(0), BTreeMap::new())
            .expect_err("a zero interval is not a schedule");

        assert!(err.to_string().contains("telemetry.metrics_interval_secs"));
    }

    #[cfg(feature = "otel")]
    #[test]
    fn the_builder_carries_every_field_through() {
        let config = Telemetry::otlp("http://localhost:4318")
            .service_name("my-agent")
            .metrics_interval_secs(15)
            .header("authorization", "Bearer hunter2")
            .to_config()
            .expect("a fully specified builder is valid");

        assert_eq!(config.otlp_endpoint, "http://localhost:4318");
        assert_eq!(config.service_name, "my-agent");
        assert_eq!(config.metrics_interval_secs(), 15);
        assert_eq!(
            config.headers.get("authorization").map(String::as_str),
            Some("Bearer hunter2")
        );
    }

    #[cfg(feature = "otel")]
    #[test]
    fn the_builder_validates_through_the_same_path_as_toml() {
        let err = Telemetry::otlp("nonsense")
            .to_config()
            .expect_err("a non-URL endpoint is refused wherever it came from");

        assert!(err.to_string().contains("telemetry.otlp_endpoint"));
    }
}
