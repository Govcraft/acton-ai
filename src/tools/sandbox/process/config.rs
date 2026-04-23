//! Configuration for the [`ProcessSandbox`](super) layer.
//!
//! Mirrors the shape of the retiring Hyperlight config but drops VM-specific
//! fields (pool warmup, guest binary source) and adds OS hardening knobs.
//!
//! Duration fields are serialized as whole seconds (`u64`) on disk. This
//! matches the pattern already used in [`crate::config::types`] and avoids
//! pulling in a new dependency solely for human-friendly durations.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::tools::ToolError;

/// Default execution timeout applied when none is specified. 30 seconds is
/// the same ceiling Hyperlight used and matches the LLM tool-call budget.
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Default address-space ceiling enforced via `setrlimit(RLIMIT_AS/DATA)`
/// inside the child. 256 MiB keeps builtin tools comfortable without
/// allowing runaway allocations.
pub const DEFAULT_MEMORY_LIMIT: u64 = 256 * 1024 * 1024;

/// Default CPU-time ceiling (wall time is enforced separately by the parent
/// via `tokio::time::timeout`).
pub const DEFAULT_CPU_LIMIT_SECS: u64 = 30;

/// Default maximum file-size ceiling. 128 MiB is enough for typical artifact
/// writes while preventing log/temp file blowouts.
pub const DEFAULT_FSIZE_LIMIT: u64 = 128 * 1024 * 1024;

/// Environment variables forwarded to the child process by default.
///
/// Anything not in this list is stripped before `execve`, reducing the
/// blast radius of leaked secrets.
pub const DEFAULT_ENV_ALLOWLIST: &[&str] = &["PATH", "LANG", "LC_ALL", "HOME", "TMPDIR"];

/// OS-hardening policy applied to the child process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HardeningMode {
    /// Do not apply landlock/seccomp restrictions. Useful for local
    /// debugging; unsafe in production.
    Off,
    /// Apply hardening, but tolerate failures (e.g. kernels without
    /// landlock support) by logging a warning and continuing.
    BestEffort,
    /// Apply hardening, and abort child startup if any step fails.
    Enforce,
}

impl Default for HardeningMode {
    fn default() -> Self {
        Self::BestEffort
    }
}

/// Configuration for a [`ProcessSandbox`](super) instance.
///
/// All fields have sensible defaults; construct with `Default::default()` or
/// `ProcessSandboxConfig::new()` and customize with the builder methods.
///
/// ```
/// use acton_ai::tools::sandbox::ProcessSandboxConfig;
/// use std::time::Duration;
///
/// let cfg = ProcessSandboxConfig::new()
///     .with_timeout(Duration::from_secs(60))
///     .with_memory_limit(Some(512 * 1024 * 1024));
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessSandboxConfig {
    /// Wall-clock timeout enforced by the parent.
    #[serde(with = "duration_secs")]
    pub timeout: Duration,

    /// Address-space ceiling in bytes; `None` disables the limit.
    pub memory_limit: Option<u64>,

    /// CPU-time ceiling in seconds; `None` disables the limit.
    pub cpu_limit_secs: Option<u64>,

    /// Maximum file size the child may create, in bytes; `None` disables the
    /// limit.
    pub fsize_limit: Option<u64>,

    /// Environment variable names forwarded to the child process.
    pub env_allowlist: Vec<String>,

    /// OS-hardening policy.
    pub hardening: HardeningMode,
}

impl Default for ProcessSandboxConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            memory_limit: Some(DEFAULT_MEMORY_LIMIT),
            cpu_limit_secs: Some(DEFAULT_CPU_LIMIT_SECS),
            fsize_limit: Some(DEFAULT_FSIZE_LIMIT),
            env_allowlist: DEFAULT_ENV_ALLOWLIST
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            hardening: HardeningMode::default(),
        }
    }
}

impl ProcessSandboxConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the wall-clock timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the address-space ceiling in bytes. Pass `None` to disable.
    #[must_use]
    pub fn with_memory_limit(mut self, limit: Option<u64>) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Sets the CPU-time ceiling in seconds. Pass `None` to disable.
    #[must_use]
    pub fn with_cpu_limit_secs(mut self, limit: Option<u64>) -> Self {
        self.cpu_limit_secs = limit;
        self
    }

    /// Sets the file-size ceiling in bytes. Pass `None` to disable.
    #[must_use]
    pub fn with_fsize_limit(mut self, limit: Option<u64>) -> Self {
        self.fsize_limit = limit;
        self
    }

    /// Sets the environment allowlist.
    #[must_use]
    pub fn with_env_allowlist<I, S>(mut self, allowlist: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.env_allowlist = allowlist.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the hardening mode.
    #[must_use]
    pub fn with_hardening(mut self, mode: HardeningMode) -> Self {
        self.hardening = mode;
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] if:
    /// - `timeout` is zero,
    /// - `memory_limit` is `Some(0)`,
    /// - `cpu_limit_secs` is `Some(0)`,
    /// - `fsize_limit` is `Some(0)`,
    /// - `env_allowlist` is empty.
    pub fn validate(&self) -> Result<(), ToolError> {
        if self.timeout.is_zero() {
            return Err(ToolError::sandbox_error(
                "process sandbox timeout must be greater than zero",
            ));
        }
        if let Some(0) = self.memory_limit {
            return Err(ToolError::sandbox_error(
                "process sandbox memory_limit must be greater than zero when set",
            ));
        }
        if let Some(0) = self.cpu_limit_secs {
            return Err(ToolError::sandbox_error(
                "process sandbox cpu_limit_secs must be greater than zero when set",
            ));
        }
        if let Some(0) = self.fsize_limit {
            return Err(ToolError::sandbox_error(
                "process sandbox fsize_limit must be greater than zero when set",
            ));
        }
        if self.env_allowlist.is_empty() {
            return Err(ToolError::sandbox_error(
                "process sandbox env_allowlist must not be empty",
            ));
        }
        Ok(())
    }
}

/// Serde adapter that represents a [`Duration`] as whole seconds (`u64`).
mod duration_secs {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(value.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_constants() {
        let cfg = ProcessSandboxConfig::default();
        assert_eq!(cfg.timeout, Duration::from_secs(DEFAULT_TIMEOUT_SECS));
        assert_eq!(cfg.memory_limit, Some(DEFAULT_MEMORY_LIMIT));
        assert_eq!(cfg.cpu_limit_secs, Some(DEFAULT_CPU_LIMIT_SECS));
        assert_eq!(cfg.fsize_limit, Some(DEFAULT_FSIZE_LIMIT));
        assert_eq!(
            cfg.env_allowlist,
            vec![
                "PATH".to_string(),
                "LANG".to_string(),
                "LC_ALL".to_string(),
                "HOME".to_string(),
                "TMPDIR".to_string(),
            ]
        );
        assert_eq!(cfg.hardening, HardeningMode::BestEffort);
    }

    #[test]
    fn hardening_mode_default_is_best_effort() {
        assert_eq!(HardeningMode::default(), HardeningMode::BestEffort);
    }

    #[test]
    fn hardening_mode_serde_uses_lowercase() {
        let on = serde_json::to_string(&HardeningMode::Off).unwrap();
        assert_eq!(on, "\"off\"");
        let be = serde_json::to_string(&HardeningMode::BestEffort).unwrap();
        assert_eq!(be, "\"besteffort\"");
        let enforce = serde_json::to_string(&HardeningMode::Enforce).unwrap();
        assert_eq!(enforce, "\"enforce\"");

        let parsed: HardeningMode = serde_json::from_str("\"enforce\"").unwrap();
        assert_eq!(parsed, HardeningMode::Enforce);
    }

    #[test]
    fn builders_mutate_expected_fields() {
        let cfg = ProcessSandboxConfig::new()
            .with_timeout(Duration::from_secs(90))
            .with_memory_limit(Some(128 * 1024 * 1024))
            .with_cpu_limit_secs(Some(60))
            .with_fsize_limit(Some(1024))
            .with_env_allowlist(["FOO", "BAR"])
            .with_hardening(HardeningMode::Enforce);
        assert_eq!(cfg.timeout, Duration::from_secs(90));
        assert_eq!(cfg.memory_limit, Some(128 * 1024 * 1024));
        assert_eq!(cfg.cpu_limit_secs, Some(60));
        assert_eq!(cfg.fsize_limit, Some(1024));
        assert_eq!(
            cfg.env_allowlist,
            vec!["FOO".to_string(), "BAR".to_string()]
        );
        assert_eq!(cfg.hardening, HardeningMode::Enforce);
    }

    #[test]
    fn builder_can_disable_limits() {
        let cfg = ProcessSandboxConfig::new()
            .with_memory_limit(None)
            .with_cpu_limit_secs(None)
            .with_fsize_limit(None);
        assert!(cfg.memory_limit.is_none());
        assert!(cfg.cpu_limit_secs.is_none());
        assert!(cfg.fsize_limit.is_none());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_accepts_default() {
        assert!(ProcessSandboxConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_timeout() {
        let cfg = ProcessSandboxConfig::new().with_timeout(Duration::ZERO);
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn validate_rejects_zero_memory_limit() {
        let cfg = ProcessSandboxConfig::new().with_memory_limit(Some(0));
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("memory_limit"));
    }

    #[test]
    fn validate_rejects_zero_cpu_limit() {
        let cfg = ProcessSandboxConfig::new().with_cpu_limit_secs(Some(0));
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("cpu_limit_secs"));
    }

    #[test]
    fn validate_rejects_zero_fsize_limit() {
        let cfg = ProcessSandboxConfig::new().with_fsize_limit(Some(0));
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("fsize_limit"));
    }

    #[test]
    fn validate_rejects_empty_env_allowlist() {
        let cfg = ProcessSandboxConfig::new().with_env_allowlist(Vec::<String>::new());
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("env_allowlist"));
    }

    #[test]
    fn config_serde_roundtrip_via_toml() {
        let cfg = ProcessSandboxConfig::new()
            .with_timeout(Duration::from_secs(45))
            .with_memory_limit(Some(64 * 1024 * 1024))
            .with_cpu_limit_secs(Some(20))
            .with_fsize_limit(Some(4096))
            .with_env_allowlist(["PATH", "HOME"])
            .with_hardening(HardeningMode::Enforce);
        let serialized = toml::to_string(&cfg).unwrap();
        let parsed: ProcessSandboxConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, cfg);
    }

    #[test]
    fn config_deserializes_with_missing_fields_using_defaults() {
        // Empty table should produce the default config thanks to
        // #[serde(default)].
        let cfg: ProcessSandboxConfig = toml::from_str("").unwrap();
        assert_eq!(cfg, ProcessSandboxConfig::default());
    }

    #[test]
    fn duration_serializes_as_seconds_in_toml() {
        let cfg = ProcessSandboxConfig::new().with_timeout(Duration::from_secs(120));
        let rendered = toml::to_string(&cfg).unwrap();
        assert!(rendered.contains("timeout = 120"));
    }
}
