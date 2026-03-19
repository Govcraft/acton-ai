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
    pub async fn new(
        config_path: Option<&PathBuf>,
        provider_override: Option<&str>,
    ) -> Result<Self, CliError> {
        // Load config — try explicit path first, then default search paths
        let loaded_config = if let Some(path) = config_path {
            Some(config::from_path(path).map_err(|e| {
                CliError::configuration(format!(
                    "failed to load config from {}: {e}",
                    path.display()
                ))
            })?)
        } else {
            config::load().ok()
        };

        let mut builder = ActonAI::builder().app_name("acton-ai");
        if let Some(cfg) = loaded_config {
            builder = builder.apply_config(cfg)?;
        }

        if let Some(provider) = provider_override {
            builder = builder.default_provider(provider);
        }

        builder = builder.with_builtins();

        let ai = builder.launch().await?;
        let db_path = resolve_db_path(config_path);

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

/// Initialize tracing to stderr with the given verbosity.
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

    // Use try_init to avoid panic if a subscriber is already set
    let _ = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(use_ansi),
        )
        .try_init();
}
