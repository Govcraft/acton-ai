//! Shared runtime bootstrap for CLI commands.
//!
//! Initializes the ActonAI runtime and database from CLI options.

use crate::cli::error::CliError;
use crate::cli::output::OutputWriter;
use crate::config;
use crate::facade::ActonAI;
use crate::memory::persistence::{initialize_schema, open_database};
use crate::memory::PersistenceConfig;
use libsql::Connection;
use std::io::IsTerminal;
use std::path::PathBuf;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Shared runtime state for CLI commands.
pub struct CliRuntime {
    /// The ActonAI high-level API handle.
    pub ai: ActonAI,
    /// Database path for direct connection access.
    db_path: String,
}

impl CliRuntime {
    /// Bootstrap the full runtime from CLI options.
    ///
    /// Loads config, launches ActonAI. Database connections are opened
    /// on-demand via [`connection()`](Self::connection) to avoid lock
    /// contention with SQLite.
    ///
    /// `skill_paths` appends to whatever `[skills] paths` the loaded config
    /// supplied — CLI and config union. Passing `&[]` is equivalent to
    /// supplying no programmatic skill paths.
    pub async fn new(
        config_path: Option<&PathBuf>,
        provider_override: Option<&str>,
        skill_paths: &[PathBuf],
    ) -> Result<Self, CliError> {
        // Load config — try explicit path first, then default search paths.
        // Track the source path so we can log it at info level for operators
        // trying to understand which file wins.
        let (loaded_config, resolved_config_path) = if let Some(path) = config_path {
            let cfg = config::from_path(path).map_err(|e| {
                CliError::configuration(format!(
                    "failed to load config from {}: {e}",
                    path.display()
                ))
            })?;
            (Some(cfg), Some(path.clone()))
        } else {
            let found = config::search_paths().into_iter().find(|p| p.exists());
            match &found {
                Some(p) => (Some(config::from_path(p).map_err(|e| {
                    CliError::configuration(format!("failed to load config from {}: {e}", p.display()))
                })?), found.clone()),
                None => (None, None),
            }
        };

        match &resolved_config_path {
            Some(p) => tracing::info!(path = %p.display(), "loaded config"),
            None => tracing::info!("no config file found; using defaults"),
        }

        if let Some(cfg) = &loaded_config {
            for (name, p) in &cfg.providers {
                tracing::info!(
                    provider = %name,
                    provider_type = %p.provider_type,
                    model = %p.model,
                    base_url = %p.base_url.as_deref().unwrap_or(""),
                    "provider configured",
                );
            }
            if let Some(default) = &cfg.default_provider {
                let model = cfg
                    .providers
                    .get(default)
                    .map(|p| p.model.as_str())
                    .unwrap_or("?");
                tracing::info!(
                    default_provider = %default,
                    model = %model,
                    "config default provider",
                );
            }
        }

        if let Some(provider) = provider_override {
            tracing::info!(provider = %provider, "provider override from CLI");
        }
        if !skill_paths.is_empty() {
            tracing::info!(
                count = skill_paths.len(),
                "skill paths from CLI",
            );
            for p in skill_paths {
                tracing::debug!(path = %p.display(), "skill path");
            }
        }

        let mut builder = ActonAI::builder().app_name("acton-ai");

        // Stage CLI-supplied skill paths before apply_config so they appear
        // first in the resulting list; config-supplied paths append.
        if !skill_paths.is_empty() {
            builder = builder.with_skill_paths(skill_paths);
        }

        if let Some(cfg) = loaded_config {
            builder = builder.apply_config(cfg)?;
        }

        if let Some(provider) = provider_override {
            builder = builder.default_provider(provider);
        }

        builder = builder.with_builtins();

        let ai = builder.launch().await?;
        let db_path = resolve_db_path(config_path);
        tracing::info!(
            default_provider = %ai.default_provider_name(),
            provider_count = ai.provider_count(),
            max_tool_rounds = ai.default_max_tool_rounds(),
            skills_loaded = ai.skills().map(|r| r.len()).unwrap_or(0),
            context_max_tokens = ai.context_window().map(|c| c.config().max_tokens).unwrap_or(0),
            context_estimator = ai.context_window().map(|c| c.estimator_name()).unwrap_or("disabled"),
            db_path = %db_path,
            "runtime launched",
        );

        Ok(Self { ai, db_path })
    }

    /// Opens a database connection and ensures the schema is initialized.
    ///
    /// Each call opens a fresh connection — callers should hold the
    /// connection for the duration of their transaction, then drop it.
    pub async fn connection(&self) -> Result<Connection, CliError> {
        let config = PersistenceConfig::new(&self.db_path);
        let db = open_database(&config).await?;
        let conn = db
            .connect()
            .map_err(|e| CliError::configuration(format!("failed to connect to database: {e}")))?;
        initialize_schema(&conn).await?;
        Ok(conn)
    }

    /// Gracefully shut down the runtime.
    pub async fn shutdown(self) -> Result<(), CliError> {
        self.ai.shutdown().await?;
        Ok(())
    }
}

/// Resolve the database path from config or XDG data directory.
pub(crate) fn resolve_db_path(config_path: Option<&PathBuf>) -> String {
    // If a config path is given, put the DB next to it
    if let Some(path) = config_path {
        if let Some(parent) = path.parent() {
            return parent.join("acton-ai.db").to_string_lossy().to_string();
        }
    }

    // Use XDG data directory
    if let Some(data_dir) = dirs::data_dir() {
        let acton_dir = data_dir.join("acton-ai");
        // Ensure directory exists
        let _ = std::fs::create_dir_all(&acton_dir);
        return acton_dir.join("acton-ai.db").to_string_lossy().to_string();
    }

    // Fallback
    "acton-ai.db".to_string()
}

/// Initialize tracing with the given verbosity.
///
/// Composes two layers into a single subscriber:
/// - stderr (human-readable, colorized when attached to a TTY)
/// - journald (on Linux hosts with a running systemd-journald socket)
///
/// After installation, marks the kernel's logging sentinel so
/// `Kernel::spawn_with_config` does not race to install its own subscriber.
///
/// - quiet: suppress all output
/// - verbosity 0: warn only
/// - verbosity 1: info for acton_ai
/// - verbosity 2: debug for acton_ai
/// - verbosity 3+: trace for acton_ai
pub fn init_tracing(verbosity: u8, quiet: bool) {
    let filter = match (quiet, verbosity) {
        (true, _) => "off".to_string(),
        (_, 0) => "warn".to_string(),
        (_, 1) => "acton_ai=info".to_string(),
        (_, 2) => "acton_ai=debug".to_string(),
        (_, _) => "acton_ai=trace".to_string(),
    };

    let use_ansi = std::io::stderr().is_terminal() && OutputWriter::use_colors();

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| filter.into());

    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(use_ansi);

    let journald_cfg = crate::kernel::LoggingConfig::default();
    let journald = crate::kernel::journald_layer(&journald_cfg);

    let result = tracing_subscriber::registry()
        .with(env_filter)
        .with(stderr_layer)
        .with(journald)
        .try_init();

    if result.is_ok() {
        crate::kernel::mark_subscriber_installed();
    }
}
