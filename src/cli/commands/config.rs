//! The `config` command — show the resolved configuration.
//!
//! Prints which file the CLI loaded (or was told to load via `--config`),
//! the full search order with a marker on the matching path, the database
//! path, and every effective value after merging config and CLI overrides.
//! Use this to debug "which TOML is winning?" questions.

use crate::cli::error::CliError;
use crate::cli::output::{OutputMode, OutputWriter};
use crate::cli::runtime::resolve_db_path;
use crate::config::{self, ActonAIConfig, NamedProviderConfig};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Options for the config command.
#[derive(Debug, clap::Args)]
pub struct ConfigArgs {}

/// Source of the effective default provider, surfaced so operators can tell
/// at a glance whether a `--provider` override won or the TOML value did.
#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum DefaultSource {
    /// Explicit `default_provider` key in the loaded TOML.
    Config,
    /// `--provider` flag on the CLI.
    CliOverride,
    /// Only one provider was defined, so it's the de-facto default.
    InferredSingle,
    /// No default could be resolved.
    None,
}

/// One entry in the search-path view, with `used = true` on the path that
/// would actually load.
#[derive(Debug, Serialize)]
struct SearchPathEntry {
    path: String,
    exists: bool,
    used: bool,
}

#[derive(Debug, Serialize)]
struct DefaultProviderInfo {
    effective: Option<String>,
    source: DefaultSource,
    cli_override: Option<String>,
}

#[derive(Debug, Serialize)]
struct ConfigReport {
    schema_version: u32,
    config_path: Option<String>,
    search_paths: Vec<SearchPathEntry>,
    db_path: String,
    default_provider: DefaultProviderInfo,
    config: ActonAIConfig,
}

const SCHEMA_VERSION: u32 = 1;

/// Execute the config command.
pub fn execute(
    _args: &ConfigArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
    provider_override: Option<&str>,
) -> Result<(), CliError> {
    let report = build_report(config_path, provider_override)?;

    match output.mode() {
        OutputMode::Json => output.write_json(&report)?,
        OutputMode::Plain => write_plain(output, &report)?,
    }

    Ok(())
}

fn build_report(
    config_path: Option<&PathBuf>,
    provider_override: Option<&str>,
) -> Result<ConfigReport, CliError> {
    let search = config::search_paths();
    let resolved: Option<PathBuf> = if let Some(path) = config_path {
        Some(path.clone())
    } else {
        search.iter().find(|p| p.exists()).cloned()
    };

    let search_paths = search
        .into_iter()
        .map(|p| SearchPathEntry {
            used: resolved.as_ref().is_some_and(|r| paths_equal(r, &p)),
            exists: p.exists(),
            path: p.to_string_lossy().into_owned(),
        })
        .collect::<Vec<_>>();

    // Load the file (if any) and scrub embedded api_key values so the
    // report is safe to paste into a bug report.
    let cfg = match &resolved {
        Some(path) => scrub_secrets(config::from_path(path).map_err(|e| {
            CliError::configuration(format!(
                "failed to load config from {}: {e}",
                path.display()
            ))
        })?),
        None => ActonAIConfig::default(),
    };

    let (effective, source) = resolve_default_provider(&cfg, provider_override);

    Ok(ConfigReport {
        schema_version: SCHEMA_VERSION,
        config_path: resolved.as_ref().map(|p| p.to_string_lossy().into_owned()),
        search_paths,
        db_path: resolve_db_path(config_path),
        default_provider: DefaultProviderInfo {
            effective,
            source,
            cli_override: provider_override.map(str::to_string),
        },
        config: cfg,
    })
}

/// Canonicalize when possible so `./acton-ai.toml` matches an absolute path
/// to the same file; fall back to a literal compare for paths that don't yet
/// exist.
fn paths_equal(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(ac), Ok(bc)) => ac == bc,
        _ => a == b,
    }
}

fn scrub_secrets(mut cfg: ActonAIConfig) -> ActonAIConfig {
    for provider in cfg.providers.values_mut() {
        if provider.api_key.is_some() {
            provider.api_key = Some("[redacted]".to_string());
        }
    }
    cfg
}

fn resolve_default_provider(
    cfg: &ActonAIConfig,
    provider_override: Option<&str>,
) -> (Option<String>, DefaultSource) {
    if let Some(name) = provider_override {
        return (Some(name.to_string()), DefaultSource::CliOverride);
    }
    if let Some(name) = &cfg.default_provider {
        return (Some(name.clone()), DefaultSource::Config);
    }
    if cfg.providers.len() == 1 {
        return (
            cfg.providers.keys().next().cloned(),
            DefaultSource::InferredSingle,
        );
    }
    (None, DefaultSource::None)
}

fn write_plain(output: &OutputWriter, report: &ConfigReport) -> Result<(), CliError> {
    let cfg_path = report
        .config_path
        .clone()
        .unwrap_or_else(|| "<none — using built-in defaults>".to_string());
    output.write_line(&format!("Config file:      {cfg_path}"))?;
    output.write_line(&format!("Database path:    {}", report.db_path))?;

    output.write_line("")?;
    output.write_line("Search paths (first match wins):")?;
    for entry in &report.search_paths {
        let marker = match (entry.used, entry.exists) {
            (true, _) => "[*]",
            (false, true) => "[.]",
            (false, false) => "[ ]",
        };
        output.write_line(&format!("  {marker} {}", entry.path))?;
    }
    output.write_line("  ([*] loaded  [.] present but not loaded  [ ] absent)")?;

    output.write_line("")?;
    let def = &report.default_provider;
    let default_display = def.effective.as_deref().unwrap_or("<none>");
    let source_str = match def.source {
        DefaultSource::Config => "from config",
        DefaultSource::CliOverride => "from --provider",
        DefaultSource::InferredSingle => "inferred (single provider)",
        DefaultSource::None => "unresolved",
    };
    output.write_line(&format!(
        "Default provider: {default_display}  ({source_str})"
    ))?;
    if let Some(ovr) = &def.cli_override {
        output.write_line(&format!("  --provider override: {ovr}"))?;
    }

    output.write_line("")?;
    let providers = &report.config.providers;
    if providers.is_empty() {
        output.write_line("Providers: <none defined>")?;
    } else {
        output.write_line(&format!("Providers ({}):", providers.len()))?;
        let mut names: Vec<&String> = providers.keys().collect();
        names.sort();
        for name in names {
            if let Some(p) = providers.get(name) {
                write_provider_plain(output, name, p)?;
            }
        }
    }

    if let Some(defaults) = &report.config.defaults {
        output.write_line("")?;
        output.write_line("Defaults:")?;
        if let Some(r) = defaults.max_tool_rounds {
            output.write_line(&format!("  max_tool_rounds: {r}"))?;
        }
    }

    if let Some(ctx) = &report.config.context {
        output.write_line("")?;
        output.write_line("Context window:")?;
        if let Some(m) = ctx.max_tokens {
            output.write_line(&format!("  max_tokens:            {m}"))?;
        }
        if let Some(r) = ctx.reserved_for_response {
            output.write_line(&format!("  reserved_for_response: {r}"))?;
        }
        if let Some(s) = &ctx.strategy {
            output.write_line(&format!("  strategy:              {s}"))?;
        }
    }

    if let Some(budget) = &report.config.budget {
        output.write_line("")?;
        output.write_line("Budget:")?;
        if let Some(total) = budget.total_usd {
            output.write_line(&format!("  total_usd:        {total:.2}"))?;
        }
        if let Some(percent) = budget.warn_at_percent {
            output.write_line(&format!("  warn_at_percent:  {percent}"))?;
        }
        if let Some(allow) = budget.allow_unpriced {
            output.write_line(&format!("  allow_unpriced:   {allow}"))?;
        }
        for (name, cap) in &budget.providers {
            output.write_line(&format!("  provider {name}: {cap:.2}"))?;
        }
    }

    if let Some(sb) = &report.config.sandbox {
        output.write_line("")?;
        output.write_line("Sandbox:")?;
        if let Some(h) = sb.hardening {
            output.write_line(&format!("  hardening:         {h:?}"))?;
        }
        if let Some(limits) = &sb.limits {
            if let Some(ms) = limits.max_execution_ms {
                output.write_line(&format!("  max_execution_ms:  {ms}"))?;
            }
            if let Some(mb) = limits.max_memory_mb {
                output.write_line(&format!("  max_memory_mb:     {mb}"))?;
            }
        }
    }

    if let Some(p) = &report.config.persistence {
        if let Some(path) = &p.db_path {
            output.write_line("")?;
            output.write_line("Persistence:")?;
            output.write_line(&format!("  db_path:          {path}"))?;
        }
    }

    if let Some(cli) = &report.config.cli {
        if cli.default_session.is_some() || cli.default_system_prompt.is_some() {
            output.write_line("")?;
            output.write_line("CLI defaults:")?;
            if let Some(s) = &cli.default_session {
                output.write_line(&format!("  default_session:       {s}"))?;
            }
            if let Some(p) = &cli.default_system_prompt {
                output.write_line(&format!("  default_system_prompt: {p}"))?;
            }
        }
    }

    if let Some(skills) = &report.config.skills {
        if !skills.paths.is_empty() {
            output.write_line("")?;
            output.write_line("Skill paths:")?;
            for p in &skills.paths {
                output.write_line(&format!("  {}", p.display()))?;
            }
        }
    }

    if let Some(jobs) = &report.config.jobs {
        if !jobs.is_empty() {
            output.write_line("")?;
            output.write_line(&format!("Jobs ({}):", jobs.len()))?;
            let mut names: Vec<&String> = jobs.keys().collect();
            names.sort();
            for n in names {
                output.write_line(&format!("  {n}"))?;
            }
        }
    }

    Ok(())
}

fn write_provider_plain(
    output: &OutputWriter,
    name: &str,
    p: &NamedProviderConfig,
) -> Result<(), CliError> {
    output.write_line(&format!("  - {name}: {}/{}", p.provider_type, p.model))?;
    if let Some(url) = &p.base_url {
        output.write_line(&format!("      base_url:    {url}"))?;
    }
    if let Some(env) = &p.api_key_env {
        let status = if std::env::var(env).is_ok() {
            "set"
        } else {
            "unset"
        };
        output.write_line(&format!("      api_key_env: {env}  ({status})"))?;
    } else if p.api_key.is_some() {
        output.write_line("      api_key:     [redacted — set directly in TOML]")?;
    }
    if let Some(t) = p.timeout_secs {
        output.write_line(&format!("      timeout:     {t}s"))?;
    }
    if let Some(m) = p.max_tokens {
        output.write_line(&format!("      max_tokens:  {m}"))?;
    }
    if let Some(cw) = p.context_window_tokens {
        output.write_line(&format!("      context_window_tokens: {cw}"))?;
    }
    if let Some(rl) = &p.rate_limit {
        output.write_line(&format!(
            "      rate_limit:  {} req/min, {} tok/min",
            rl.requests_per_minute, rl.tokens_per_minute
        ))?;
    }
    let mut sampling = Vec::new();
    if let Some(t) = p.temperature {
        sampling.push(format!("temperature={t}"));
    }
    if let Some(k) = p.top_k {
        sampling.push(format!("top_k={k}"));
    }
    if let Some(tp) = p.top_p {
        sampling.push(format!("top_p={tp}"));
    }
    if let Some(fp) = p.frequency_penalty {
        sampling.push(format!("frequency_penalty={fp}"));
    }
    if let Some(pp) = p.presence_penalty {
        sampling.push(format!("presence_penalty={pp}"));
    }
    if let Some(s) = p.seed {
        sampling.push(format!("seed={s}"));
    }
    if let Some(seqs) = &p.stop_sequences {
        sampling.push(format!("stop={seqs:?}"));
    }
    if !sampling.is_empty() {
        output.write_line(&format!("      sampling:    {}", sampling.join(", ")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_secrets_redacts_direct_api_key() {
        let cfg = ActonAIConfig::new().with_provider(
            "claude",
            NamedProviderConfig::anthropic("claude-sonnet-4-20250514").with_api_key("sk-secret"),
        );

        let scrubbed = scrub_secrets(cfg);
        let claude = scrubbed.providers.get("claude").unwrap();
        assert_eq!(claude.api_key.as_deref(), Some("[redacted]"));
    }

    #[test]
    fn scrub_secrets_leaves_api_key_env_alone() {
        let cfg = ActonAIConfig::new().with_provider(
            "claude",
            NamedProviderConfig::anthropic("claude-sonnet-4-20250514"),
        );

        let scrubbed = scrub_secrets(cfg);
        let claude = scrubbed.providers.get("claude").unwrap();
        assert!(claude.api_key.is_none());
        assert_eq!(claude.api_key_env.as_deref(), Some("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn default_provider_cli_override_wins() {
        let cfg = ActonAIConfig::new()
            .with_provider("claude", NamedProviderConfig::anthropic("m"))
            .with_provider("ollama", NamedProviderConfig::ollama("q"))
            .with_default_provider("claude");

        let (effective, source) = resolve_default_provider(&cfg, Some("ollama"));
        assert_eq!(effective.as_deref(), Some("ollama"));
        assert!(matches!(source, DefaultSource::CliOverride));
    }

    #[test]
    fn default_provider_config_used_when_no_override() {
        let cfg = ActonAIConfig::new()
            .with_provider("claude", NamedProviderConfig::anthropic("m"))
            .with_provider("ollama", NamedProviderConfig::ollama("q"))
            .with_default_provider("claude");

        let (effective, source) = resolve_default_provider(&cfg, None);
        assert_eq!(effective.as_deref(), Some("claude"));
        assert!(matches!(source, DefaultSource::Config));
    }

    #[test]
    fn default_provider_inferred_for_single_provider() {
        let cfg = ActonAIConfig::new().with_provider("only", NamedProviderConfig::ollama("q"));

        let (effective, source) = resolve_default_provider(&cfg, None);
        assert_eq!(effective.as_deref(), Some("only"));
        assert!(matches!(source, DefaultSource::InferredSingle));
    }

    #[test]
    fn default_provider_none_when_ambiguous() {
        let cfg = ActonAIConfig::new()
            .with_provider("a", NamedProviderConfig::ollama("x"))
            .with_provider("b", NamedProviderConfig::ollama("y"));

        let (effective, source) = resolve_default_provider(&cfg, None);
        assert!(effective.is_none());
        assert!(matches!(source, DefaultSource::None));
    }
}
