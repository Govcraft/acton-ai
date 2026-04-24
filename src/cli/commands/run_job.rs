//! The `run-job` command -- execute named jobs with agentic loops.
//!
//! Jobs are defined in the config file and support autonomous tool use
//! until the agent decides the work is complete.

use crate::cli::error::CliError;
use crate::cli::output::{OutputMode, OutputWriter};
use crate::cli::runtime::CliRuntime;
use serde::Serialize;
use std::io::{self, Read as _};
use std::path::PathBuf;

/// Options for the run-job command.
#[derive(Debug, clap::Args)]
pub struct RunJobArgs {
    /// Name of the job to execute (defined in config).
    pub name: String,

    /// Session to persist the interaction to.
    #[arg(long, env = "ACTON_SESSION")]
    pub session: Option<String>,

    /// Template parameters as KEY=VALUE pairs.
    #[arg(long = "param", value_parser = parse_param)]
    pub params: Vec<(String, String)>,

    /// Maximum tool execution rounds.
    #[arg(long)]
    pub max_rounds: Option<usize>,
}

/// Parse a KEY=VALUE parameter.
fn parse_param(s: &str) -> Result<(String, String), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid param '{s}': expected KEY=VALUE format"))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}

/// JSON response envelope for `--json` mode.
#[derive(Serialize)]
struct JobResponse {
    job: String,
    text: String,
    token_count: usize,
}

/// Apply template substitution to a message template.
///
/// Replaces `{{input}}` with the stdin content and `{{key}}` for each
/// `--param key=value` pair.
fn apply_template(template: &str, input: Option<&str>, params: &[(String, String)]) -> String {
    let mut result = template.to_string();

    if let Some(input_text) = input {
        result = result.replace("{{input}}", input_text);
    }

    for (key, value) in params {
        let placeholder = format!("{{{{{key}}}}}");
        result = result.replace(&placeholder, value);
    }

    result
}

/// Collect available job names from the config for error messages.
fn available_job_names(
    jobs: &Option<std::collections::HashMap<String, crate::config::JobConfig>>,
) -> Vec<String> {
    jobs.as_ref()
        .map(|j| j.keys().cloned().collect())
        .unwrap_or_default()
}

/// Execute the run-job command.
pub async fn execute(
    args: &RunJobArgs,
    output: &OutputWriter,
    config_path: Option<&PathBuf>,
    provider_override: Option<&str>,
) -> Result<(), CliError> {
    // Step 1: Load config to find the job definition
    let loaded_config = if let Some(path) = config_path {
        crate::config::from_path(path).map_err(|e| {
            CliError::configuration(format!(
                "failed to load config from {}: {e}",
                path.display()
            ))
        })?
    } else {
        crate::config::load()
            .map_err(|e| CliError::configuration(format!("failed to load config: {e}")))?
    };

    // Step 2: Look up the job by name
    let available = available_job_names(&loaded_config.jobs);
    let job = loaded_config
        .jobs
        .as_ref()
        .and_then(|jobs| jobs.get(&args.name))
        .ok_or_else(|| CliError::job_not_found(&args.name, available))?
        .clone();

    // Step 3: Read stdin as {{input}} if not a TTY
    let stdin_input = if !OutputWriter::stdin_is_tty() {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        let trimmed = input.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    } else {
        None
    };

    // Step 4: Build the message from template substitution
    let message = if let Some(ref template) = job.message_template {
        apply_template(template, stdin_input.as_deref(), &args.params)
    } else if let Some(ref input) = stdin_input {
        input.clone()
    } else {
        return Err(CliError::no_input());
    };

    // Step 5: Determine the effective provider override
    // Job-level provider takes precedence, then CLI-level override
    let effective_provider = job.provider.as_deref().or(provider_override);

    // Step 6: Bootstrap runtime
    let rt = CliRuntime::new(config_path, effective_provider, &[]).await?;

    // Step 7: Build and execute the prompt
    let mut prompt = rt.ai.prompt(&message).system(&job.system_prompt);

    // Apply provider override if the job specifies one and it differs from default
    if let Some(provider_name) = &job.provider {
        prompt = prompt.provider(provider_name);
    }

    // Apply max_tool_rounds from args, then job config
    if let Some(max) = args.max_rounds.or(job.max_tool_rounds) {
        prompt = prompt.max_tool_rounds(max);
    }

    let response = prompt.collect().await?;

    // Step 8: Output the result
    match output.mode() {
        OutputMode::Json => {
            output.write_json(&JobResponse {
                job: args.name.clone(),
                text: response.text.clone(),
                token_count: response.token_count,
            })?;
        }
        OutputMode::Plain => {
            output.write_line(&response.text)?;
        }
    }

    // Step 9: Persist to session if requested
    if let Some(ref session_name) = args.session {
        let conn = rt.connection().await?;
        let session = crate::memory::persistence::resolve_session(&conn, session_name).await?;

        let conversation_id = if let Some(info) = session {
            info.conversation_id
        } else {
            let agent_id = crate::types::AgentId::new();
            crate::memory::persistence::create_session(
                &conn,
                session_name,
                &agent_id,
                Some(&job.system_prompt),
            )
            .await?
        };

        let user_msg = crate::messages::Message::user(&message);
        let assistant_msg = crate::messages::Message::assistant(&response.text);
        crate::memory::persistence::save_message(&conn, &conversation_id, &user_msg).await?;
        crate::memory::persistence::save_message(&conn, &conversation_id, &assistant_msg).await?;
        crate::memory::persistence::touch_session(&conn, session_name).await?;
    }

    // Step 10: Shutdown
    rt.shutdown().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_param_valid() {
        let result = parse_param("key=value").unwrap();
        assert_eq!(result, ("key".to_string(), "value".to_string()));
    }

    #[test]
    fn parse_param_value_with_equals() {
        let result = parse_param("key=val=ue").unwrap();
        assert_eq!(result, ("key".to_string(), "val=ue".to_string()));
    }

    #[test]
    fn parse_param_missing_equals() {
        let result = parse_param("noequals");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("KEY=VALUE"));
    }

    #[test]
    fn apply_template_replaces_input() {
        let result = apply_template("Summarize: {{input}}", Some("hello world"), &[]);
        assert_eq!(result, "Summarize: hello world");
    }

    #[test]
    fn apply_template_replaces_params() {
        let params = vec![
            ("lang".to_string(), "rust".to_string()),
            ("style".to_string(), "concise".to_string()),
        ];
        let result = apply_template("Write {{lang}} code, be {{style}}", None, &params);
        assert_eq!(result, "Write rust code, be concise");
    }

    #[test]
    fn apply_template_replaces_both() {
        let params = vec![("format".to_string(), "JSON".to_string())];
        let result = apply_template(
            "Convert {{input}} to {{format}}",
            Some("some data"),
            &params,
        );
        assert_eq!(result, "Convert some data to JSON");
    }

    #[test]
    fn apply_template_no_replacements() {
        let result = apply_template("plain text", None, &[]);
        assert_eq!(result, "plain text");
    }

    #[test]
    fn apply_template_input_not_replaced_when_none() {
        let result = apply_template("prefix {{input}} suffix", None, &[]);
        assert_eq!(result, "prefix {{input}} suffix");
    }

    #[test]
    fn available_job_names_none() {
        let names = available_job_names(&None);
        assert!(names.is_empty());
    }

    #[test]
    fn available_job_names_some() {
        use std::collections::HashMap;
        let mut jobs = HashMap::new();
        jobs.insert(
            "summarize".to_string(),
            crate::config::JobConfig {
                system_prompt: String::new(),
                message_template: None,
                provider: None,
                tools: None,
                max_tool_rounds: None,
            },
        );
        let names = available_job_names(&Some(jobs));
        assert_eq!(names, vec!["summarize".to_string()]);
    }
}
