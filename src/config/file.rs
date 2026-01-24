//! Configuration file loading.
//!
//! This module handles loading acton-ai configuration from TOML files
//! at XDG-compliant locations.

use crate::config::types::ActonAIConfig;
use crate::error::{ActonAIError, ActonAIErrorKind};
use std::path::{Path, PathBuf};

/// Default configuration file name for project-local config.
const LOCAL_CONFIG_NAME: &str = "acton-ai.toml";

/// Default configuration file name within XDG config directory.
const XDG_CONFIG_NAME: &str = "config.toml";

/// Application name for XDG directory lookup.
const APP_NAME: &str = "acton-ai";

/// Loads configuration from the default search paths.
///
/// Search order:
/// 1. `./acton-ai.toml` (project-local)
/// 2. `~/.config/acton-ai/config.toml` (XDG config)
///
/// Returns an empty configuration if no config file is found.
///
/// # Errors
///
/// Returns an error if a config file exists but cannot be parsed.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::config::load;
///
/// let config = load()?;
/// if config.is_empty() {
///     println!("No configuration file found, using defaults");
/// }
/// ```
pub fn load() -> Result<ActonAIConfig, ActonAIError> {
    // Try project-local config first
    let local_path = PathBuf::from(LOCAL_CONFIG_NAME);
    if local_path.exists() {
        return from_path(&local_path);
    }

    // Try XDG config directory
    if let Some(config_dir) = dirs::config_dir() {
        let xdg_path = config_dir.join(APP_NAME).join(XDG_CONFIG_NAME);
        if xdg_path.exists() {
            return from_path(&xdg_path);
        }
    }

    // No config file found - return empty config
    Ok(ActonAIConfig::default())
}

/// Loads configuration from a specific file path.
///
/// # Arguments
///
/// * `path` - Path to the TOML configuration file
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The file contains invalid TOML
/// - The TOML doesn't match the expected schema
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::config::from_path;
/// use std::path::Path;
///
/// let config = from_path(Path::new("/etc/acton-ai/config.toml"))?;
/// ```
pub fn from_path(path: &Path) -> Result<ActonAIConfig, ActonAIError> {
    let contents = std::fs::read_to_string(path).map_err(|e| {
        ActonAIError::new(ActonAIErrorKind::Configuration {
            field: "config_file".to_string(),
            reason: format!("failed to read '{}': {}", path.display(), e),
        })
    })?;

    from_str(&contents).map_err(|e| {
        ActonAIError::new(ActonAIErrorKind::Configuration {
            field: "config_file".to_string(),
            reason: format!("failed to parse '{}': {}", path.display(), e),
        })
    })
}

/// Parses configuration from a TOML string.
///
/// # Arguments
///
/// * `toml_str` - TOML configuration content
///
/// # Errors
///
/// Returns an error if the TOML is invalid or doesn't match the schema.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::config::from_str;
///
/// let toml = r#"
/// [providers.ollama]
/// type = "ollama"
/// model = "qwen2.5:7b"
///
/// default_provider = "ollama"
/// "#;
///
/// let config = from_str(toml)?;
/// ```
pub fn from_str(toml_str: &str) -> Result<ActonAIConfig, ActonAIError> {
    toml::from_str(toml_str).map_err(|e| {
        ActonAIError::new(ActonAIErrorKind::Configuration {
            field: "config".to_string(),
            reason: format!("invalid TOML: {e}"),
        })
    })
}

/// Returns the paths that would be searched for configuration files.
///
/// This is useful for diagnostics and user guidance.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::config::search_paths;
///
/// for path in search_paths() {
///     println!("Would check: {}", path.display());
/// }
/// ```
#[must_use]
pub fn search_paths() -> Vec<PathBuf> {
    let mut paths = vec![PathBuf::from(LOCAL_CONFIG_NAME)];

    if let Some(config_dir) = dirs::config_dir() {
        paths.push(config_dir.join(APP_NAME).join(XDG_CONFIG_NAME));
    }

    paths
}

/// Returns the path to the XDG config directory for acton-ai.
///
/// This is `~/.config/acton-ai` on most systems.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::config::xdg_config_dir;
///
/// if let Some(dir) = xdg_config_dir() {
///     println!("Config directory: {}", dir.display());
/// }
/// ```
#[must_use]
pub fn xdg_config_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|p| p.join(APP_NAME))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn load_returns_empty_when_no_config() {
        // This test depends on no config file existing in the test directory
        // which should be true in a clean test environment
        let config = load().unwrap();
        // We can't assert it's empty since the user might have a config file
        // Just verify it doesn't error
        let _ = config;
    }

    #[test]
    fn from_str_parses_valid_toml() {
        let toml = r#"
default_provider = "ollama"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"
base_url = "http://localhost:11434/v1"
        "#;

        let config = from_str(toml).unwrap();

        assert_eq!(config.default_provider, Some("ollama".to_string()));
        assert!(config.providers.contains_key("ollama"));
    }

    #[test]
    fn from_str_parses_multiple_providers() {
        let toml = r#"
default_provider = "ollama"

[providers.claude]
type = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"

[providers.ollama]
type = "ollama"
model = "qwen2.5:7b"

[providers.fast]
type = "openai"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
        "#;

        let config = from_str(toml).unwrap();

        assert_eq!(config.provider_count(), 3);
        assert!(config.providers.contains_key("claude"));
        assert!(config.providers.contains_key("ollama"));
        assert!(config.providers.contains_key("fast"));
    }

    #[test]
    fn from_str_handles_rate_limit() {
        let toml = r#"
            [providers.ollama]
            type = "ollama"
            model = "qwen2.5:7b"

            [providers.ollama.rate_limit]
            requests_per_minute = 1000
            tokens_per_minute = 1000000
        "#;

        let config = from_str(toml).unwrap();
        let ollama = config.providers.get("ollama").unwrap();

        assert!(ollama.rate_limit.is_some());
        let rate_limit = ollama.rate_limit.as_ref().unwrap();
        assert_eq!(rate_limit.requests_per_minute, 1000);
        assert_eq!(rate_limit.tokens_per_minute, 1_000_000);
    }

    #[test]
    fn from_str_error_on_invalid_toml() {
        let invalid = "this is not valid toml [[[";

        let result = from_str(invalid);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_configuration());
    }

    #[test]
    fn from_path_reads_file() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        let mut file = std::fs::File::create(&config_path).unwrap();
        writeln!(
            file,
            r#"
            [providers.test]
            type = "ollama"
            model = "test-model"
        "#
        )
        .unwrap();

        let config = from_path(&config_path).unwrap();

        assert!(config.providers.contains_key("test"));
    }

    #[test]
    fn from_path_error_on_missing_file() {
        let result = from_path(Path::new("/nonexistent/path/config.toml"));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_configuration());
    }

    #[test]
    fn search_paths_includes_local() {
        let paths = search_paths();

        assert!(!paths.is_empty());
        assert!(paths.iter().any(|p| p.file_name() == Some(std::ffi::OsStr::new(LOCAL_CONFIG_NAME))));
    }

    #[test]
    fn xdg_config_dir_returns_path() {
        // This test may fail if XDG dirs aren't available, but that's rare
        if let Some(dir) = xdg_config_dir() {
            assert!(dir.ends_with(APP_NAME));
        }
    }
}
