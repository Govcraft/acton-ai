//! Hyperlight sandbox implementation.
//!
//! Provides hardware-isolated execution using Hyperlight micro-VMs.

use super::config::SandboxConfig;
use super::error::SandboxErrorKind;
use super::guest::{GuestType, GUEST_BINARIES};
use crate::tools::error::ToolError;
use crate::tools::sandbox::traits::{Sandbox, SandboxExecutionFuture};
use hyperlight_host::sandbox::uninitialized::{GuestBinary, UninitializedSandbox};
use hyperlight_host::sandbox::SandboxConfiguration;
use hyperlight_host::MultiUseSandbox;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Request sent to the shell executor guest.
///
/// Used when calling the guest's `execute_shell` function.
#[derive(Debug, Serialize, Deserialize)]
struct ShellRequest {
    /// Command to execute
    command: String,
    /// Additional arguments as JSON
    args: Value,
    /// Timeout in seconds
    timeout_secs: u64,
}

/// Response from the shell executor guest.
///
/// Parsed from the JSON response from the guest's `execute_shell` function.
#[derive(Debug, Deserialize)]
struct ShellResponse {
    /// Exit code from the command
    exit_code: i32,
    /// Standard output
    stdout: String,
    /// Standard error
    stderr: String,
    /// Whether the command succeeded
    success: bool,
    /// Whether output was truncated
    #[serde(default)]
    truncated: bool,
}

/// Hyperlight-based sandbox for hardware-isolated code execution.
///
/// This sandbox uses Hyperlight micro-VMs to provide strong isolation
/// for executing untrusted code. Each sandbox instance wraps a
/// `MultiUseSandbox` that can be reused for multiple executions.
///
/// # Thread Safety
///
/// The sandbox is `Send + Sync` and uses interior mutability for the
/// underlying VM. Only one execution can occur at a time within a
/// single sandbox instance.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::sandbox::hyperlight::{HyperlightSandbox, SandboxConfig};
///
/// let config = SandboxConfig::new()
///     .with_memory_limit(64 * 1024 * 1024)
///     .with_timeout(Duration::from_secs(30));
///
/// let sandbox = HyperlightSandbox::new(config)?;
/// let result = sandbox.execute("echo hello", serde_json::json!({})).await?;
/// ```
pub struct HyperlightSandbox {
    /// The underlying Hyperlight sandbox (protected by mutex for interior mutability)
    inner: Mutex<Option<MultiUseSandbox>>,
    /// Whether the sandbox has been destroyed
    destroyed: AtomicBool,
    /// Configuration used to create this sandbox
    config: SandboxConfig,
}

impl std::fmt::Debug for HyperlightSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HyperlightSandbox")
            .field("destroyed", &self.destroyed.load(Ordering::SeqCst))
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl HyperlightSandbox {
    /// Creates a new Hyperlight sandbox with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The sandbox configuration
    ///
    /// # Returns
    ///
    /// A new sandbox instance, or an error if creation fails.
    ///
    /// # Errors
    ///
    /// Returns `SandboxErrorKind::CreationFailed` if the sandbox cannot be created.
    /// Returns `SandboxErrorKind::HypervisorNotAvailable` if no hypervisor is present.
    pub fn new(config: SandboxConfig) -> Result<Self, SandboxErrorKind> {
        config.validate()?;

        // Check hypervisor availability
        if !hyperlight_host::is_hypervisor_present() {
            return Err(SandboxErrorKind::HypervisorNotAvailable);
        }

        // Create Hyperlight sandbox configuration
        // Use default and configure via setters since new() is private
        let mut hl_config = SandboxConfiguration::default();
        // Configure heap size based on our memory limit
        hl_config.set_heap_size(config.memory_limit as u64);

        // Get guest binary
        let guest_binary = Self::load_guest_binary(&config)?;

        // Create uninitialized sandbox
        let mut uninit = UninitializedSandbox::new(guest_binary, Some(hl_config)).map_err(|e| {
            SandboxErrorKind::CreationFailed {
                reason: e.to_string(),
            }
        })?;

        // Register host function for shell command execution
        // This is called by the guest's host_run_command function
        uninit
            .register("host_run_command", |request_json: String| -> String {
                Self::execute_shell_command(&request_json)
            })
            .map_err(|e| SandboxErrorKind::CreationFailed {
                reason: format!("failed to register host_run_command: {e}"),
            })?;

        // Evolve to multi-use sandbox
        let sandbox = uninit
            .evolve()
            .map_err(|e| SandboxErrorKind::CreationFailed {
                reason: format!("failed to initialize VM: {e}"),
            })?;

        Ok(Self {
            inner: Mutex::new(Some(sandbox)),
            destroyed: AtomicBool::new(false),
            config,
        })
    }

    /// Creates a new Hyperlight sandbox for the specified guest type.
    ///
    /// This constructor allows creating sandboxes for specific guest types
    /// (Shell or Http), with the appropriate host functions registered.
    ///
    /// # Arguments
    ///
    /// * `config` - The sandbox configuration
    /// * `guest_type` - The type of guest to run
    ///
    /// # Returns
    ///
    /// A new sandbox instance, or an error if creation fails.
    ///
    /// # Errors
    ///
    /// Returns `SandboxErrorKind::CreationFailed` if the sandbox cannot be created.
    /// Returns `SandboxErrorKind::HypervisorNotAvailable` if no hypervisor is present.
    pub fn new_with_guest_type(
        config: SandboxConfig,
        guest_type: GuestType,
    ) -> Result<Self, SandboxErrorKind> {
        config.validate()?;

        // Check hypervisor availability
        if !hyperlight_host::is_hypervisor_present() {
            return Err(SandboxErrorKind::HypervisorNotAvailable);
        }

        // Create Hyperlight sandbox configuration
        let mut hl_config = SandboxConfiguration::default();
        hl_config.set_heap_size(config.memory_limit as u64);

        // Get guest binary for the specified type
        let guest_binary = GuestBinary::Buffer(guest_type.binary());

        // Create uninitialized sandbox
        let mut uninit = UninitializedSandbox::new(guest_binary, Some(hl_config)).map_err(|e| {
            SandboxErrorKind::CreationFailed {
                reason: e.to_string(),
            }
        })?;

        // Register appropriate host functions based on guest type
        Self::register_host_functions(&mut uninit, guest_type)?;

        // Evolve to multi-use sandbox
        let sandbox = uninit
            .evolve()
            .map_err(|e| SandboxErrorKind::CreationFailed {
                reason: format!("failed to initialize VM for {}: {}", guest_type, e),
            })?;

        Ok(Self {
            inner: Mutex::new(Some(sandbox)),
            destroyed: AtomicBool::new(false),
            config,
        })
    }

    /// Registers host functions appropriate for the guest type.
    fn register_host_functions(
        uninit: &mut UninitializedSandbox,
        guest_type: GuestType,
    ) -> Result<(), SandboxErrorKind> {
        match guest_type {
            GuestType::Shell => {
                uninit
                    .register("host_run_command", |request_json: String| -> String {
                        Self::execute_shell_command(&request_json)
                    })
                    .map_err(|e| SandboxErrorKind::CreationFailed {
                        reason: format!("failed to register host_run_command: {e}"),
                    })?;
            }
            GuestType::Http => {
                // HTTP guest uses different host functions for network I/O
                uninit
                    .register("host_http_request", |request_json: String| -> String {
                        Self::execute_http_request(&request_json)
                    })
                    .map_err(|e| SandboxErrorKind::CreationFailed {
                        reason: format!("failed to register host_http_request: {e}"),
                    })?;
            }
        }
        Ok(())
    }

    /// Executes an HTTP request on the host.
    ///
    /// This is the host function called by the HTTP guest's `host_http_fetch`.
    /// Parses the JSON request and performs the HTTP operation.
    fn execute_http_request(request_json: &str) -> String {
        // Parse the request
        #[derive(serde::Deserialize)]
        struct HttpRequest {
            url: String,
            #[serde(default = "default_method")]
            method: String,
            #[serde(default)]
            headers: std::collections::HashMap<String, String>,
            body: Option<String>,
            #[serde(default = "default_timeout")]
            timeout_secs: u64,
        }

        fn default_method() -> String {
            "GET".to_string()
        }

        fn default_timeout() -> u64 {
            30
        }

        let request: HttpRequest = match serde_json::from_str(request_json) {
            Ok(req) => req,
            Err(e) => {
                return serde_json::json!({
                    "status": 0,
                    "headers": {},
                    "body": "",
                    "success": false,
                    "error": format!("Failed to parse request: {}", e)
                })
                .to_string();
            }
        };

        // Build and execute the HTTP request using a blocking client
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(request.timeout_secs))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                return serde_json::json!({
                    "status": 0,
                    "headers": {},
                    "body": "",
                    "success": false,
                    "error": format!("Failed to create HTTP client: {}", e)
                })
                .to_string();
            }
        };

        let mut req_builder = match request.method.to_uppercase().as_str() {
            "GET" => client.get(&request.url),
            "POST" => client.post(&request.url),
            "PUT" => client.put(&request.url),
            "DELETE" => client.delete(&request.url),
            "PATCH" => client.patch(&request.url),
            "HEAD" => client.head(&request.url),
            method => {
                return serde_json::json!({
                    "status": 0,
                    "headers": {},
                    "body": "",
                    "success": false,
                    "error": format!("Unsupported HTTP method: {}", method)
                })
                .to_string();
            }
        };

        // Add headers
        for (key, value) in &request.headers {
            req_builder = req_builder.header(key, value);
        }

        // Add body if present
        if let Some(body) = request.body {
            req_builder = req_builder.body(body);
        }

        // Execute request
        match req_builder.send() {
            Ok(response) => {
                let status = response.status().as_u16();
                let headers: std::collections::HashMap<String, String> = response
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();
                let body = response.text().unwrap_or_default();

                serde_json::json!({
                    "status": status,
                    "headers": headers,
                    "body": body,
                    "success": true,
                    "error": null
                })
                .to_string()
            }
            Err(e) => {
                serde_json::json!({
                    "status": 0,
                    "headers": {},
                    "body": "",
                    "success": false,
                    "error": format!("HTTP request failed: {}", e)
                })
                .to_string()
            }
        }
    }

    /// Loads the guest binary based on configuration.
    fn load_guest_binary(config: &SandboxConfig) -> Result<GuestBinary<'static>, SandboxErrorKind> {
        use super::config::GuestBinarySource;

        match &config.guest_binary {
            GuestBinarySource::Embedded => {
                // Use the embedded shell executor guest
                Ok(GuestBinary::Buffer(GUEST_BINARIES.shell))
            }
            GuestBinarySource::FromPath(path) => {
                let path_str = path.to_string_lossy().to_string();
                Ok(GuestBinary::FilePath(path_str))
            }
            GuestBinarySource::FromBytes(bytes) => {
                // We need to leak the bytes to get a 'static lifetime
                // This is intentional for sandbox lifetime management
                let leaked: &'static [u8] = Box::leak(bytes.clone().into_boxed_slice());
                Ok(GuestBinary::Buffer(leaked))
            }
        }
    }

    /// Executes a shell command on the host.
    ///
    /// This is the host function called by the guest's `host_run_command`.
    /// It parses the JSON request, executes the command, and returns a JSON response.
    fn execute_shell_command(request_json: &str) -> String {
        // Parse the request
        let request: ShellRequest = match serde_json::from_str(request_json) {
            Ok(req) => req,
            Err(e) => {
                return serde_json::json!({
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": format!("Failed to parse request: {}", e),
                    "success": false,
                    "truncated": false
                })
                .to_string();
            }
        };

        // Execute the command
        let output = match Command::new("sh")
            .arg("-c")
            .arg(&request.command)
            .output()
        {
            Ok(output) => output,
            Err(e) => {
                return serde_json::json!({
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": format!("Failed to execute command: {}", e),
                    "success": false,
                    "truncated": false
                })
                .to_string();
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(1);

        serde_json::json!({
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "success": output.status.success(),
            "truncated": false
        })
        .to_string()
    }

    /// Internal implementation of synchronous execution.
    ///
    /// This is called by the `Sandbox::execute_sync` trait method.
    fn execute_sync_internal(&self, code: &str, args: Value) -> Result<Value, ToolError> {
        if self.destroyed.load(Ordering::SeqCst) {
            return Err(SandboxErrorKind::AlreadyDestroyed.into());
        }

        let mut guard = self
            .inner
            .lock()
            .map_err(|_| ToolError::sandbox_error("sandbox lock poisoned"))?;

        let sandbox = guard
            .as_mut()
            .ok_or_else(|| SandboxErrorKind::AlreadyDestroyed.into_tool_error())?;

        // Prepare request for the guest
        let request = ShellRequest {
            command: code.to_string(),
            args,
            timeout_secs: self.config.timeout.as_secs(),
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| ToolError::sandbox_error(format!("failed to serialize request: {e}")))?;

        // Call the guest's execute_shell function
        let result: String = sandbox.call("execute_shell", request_json).map_err(|e| {
            SandboxErrorKind::GuestCallFailed {
                function: "execute_shell".to_string(),
                reason: e.to_string(),
            }
            .into_tool_error()
        })?;

        // Parse the response
        let response: ShellResponse = serde_json::from_str(&result).map_err(|e| {
            ToolError::sandbox_error(format!("failed to parse guest response: {e}"))
        })?;

        // Convert to JSON value
        Ok(serde_json::json!({
            "exit_code": response.exit_code,
            "stdout": response.stdout,
            "stderr": response.stderr,
            "success": response.success,
            "truncated": response.truncated
        }))
    }
}

// Safety: The sandbox uses a Mutex for interior mutability and atomic
// operations for the destroyed flag, making it safe to share across threads.
unsafe impl Send for HyperlightSandbox {}
unsafe impl Sync for HyperlightSandbox {}

impl Sandbox for HyperlightSandbox {
    fn execute(&self, code: &str, _args: Value) -> SandboxExecutionFuture {
        let _timeout = self.config.timeout;

        // Check destroyed state early
        if self.destroyed.load(Ordering::SeqCst) {
            return Box::pin(async move { Err(SandboxErrorKind::AlreadyDestroyed.into()) });
        }

        // Hyperlight's MultiUseSandbox::call is synchronous, but we can't easily
        // capture &self in an async block due to lifetime constraints.
        //
        // For production use, the SandboxPool actor pattern should be used instead,
        // which manages sandbox lifecycle properly within the actor system.
        let code = code.to_string();

        Box::pin(async move {
            Err(ToolError::sandbox_error(format!(
                "direct HyperlightSandbox::execute() is not supported; \
                 use SandboxPool for managed async execution (code: {}...)",
                if code.len() > 20 { &code[..20] } else { &code }
            )))
        })
    }

    fn destroy(&mut self) {
        self.destroyed.store(true, Ordering::SeqCst);

        // Clear the inner sandbox to release resources
        if let Ok(mut guard) = self.inner.lock() {
            *guard = None;
        }

        tracing::debug!("HyperlightSandbox destroyed");
    }

    fn is_alive(&self) -> bool {
        !self.destroyed.load(Ordering::SeqCst)
    }

    fn execute_sync(&self, code: &str, args: Value) -> Result<Value, ToolError> {
        self.execute_sync_internal(code, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a hypervisor, so they're marked as ignored
    // by default. Run with `cargo test -- --ignored` on a KVM-enabled system.

    #[test]
    fn sandbox_debug_impl() {
        // Just verify Debug is implemented (can't create without hypervisor)
        let config = SandboxConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("SandboxConfig"));
    }

    #[test]
    #[ignore = "requires hypervisor"]
    fn sandbox_creation_requires_hypervisor() {
        let config = SandboxConfig::default();
        let result = HyperlightSandbox::new(config);

        // On systems without a hypervisor, this should fail
        if !hyperlight_host::is_hypervisor_present() {
            assert!(matches!(
                result.unwrap_err(),
                SandboxErrorKind::HypervisorNotAvailable
            ));
        }
    }

    #[test]
    fn sandbox_config_validation_in_new() {
        // Invalid config should fail during new()
        let config = SandboxConfig::new().with_memory_limit(100); // Too small

        let result = HyperlightSandbox::new(config);
        assert!(result.is_err());
    }
}
