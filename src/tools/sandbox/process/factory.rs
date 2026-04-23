//! Factory that produces [`ProcessSandbox`] instances pointing at a fixed
//! executable path.
//!
//! Most callers build one of these via [`ProcessSandboxFactory::new`], which
//! resolves [`std::env::current_exe`] once and caches it. Tests and advanced
//! consumers can inject an arbitrary binary with [`ProcessSandboxFactory::with_exe`]
//! (useful with `env!("CARGO_BIN_EXE_<name>")`).

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use super::config::ProcessSandboxConfig;
use super::sandbox::ProcessSandbox;
use crate::tools::error::ToolError;
use crate::tools::sandbox::traits::{Sandbox, SandboxFactory, SandboxFactoryFuture};

/// Factory for constructing [`ProcessSandbox`] instances.
///
/// Cloning is cheap: a factory holds only the resolved executable path and
/// the sandbox configuration.
#[derive(Debug, Clone)]
pub struct ProcessSandboxFactory {
    exe: PathBuf,
    config: ProcessSandboxConfig,
}

impl ProcessSandboxFactory {
    /// Builds a factory that re-execs the currently running binary.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] if:
    /// - [`std::env::current_exe`] fails (rare, but can happen in sandboxed
    ///   environments that hide `/proc/self/exe`),
    /// - the resolved path cannot be canonicalized (e.g. the binary was
    ///   deleted after launch),
    /// - the supplied configuration is invalid per
    ///   [`ProcessSandboxConfig::validate`].
    pub fn new(config: ProcessSandboxConfig) -> Result<Self, ToolError> {
        let exe = std::env::current_exe().map_err(|err| {
            ToolError::sandbox_error(format!("failed to resolve current_exe: {err}"))
        })?;
        Self::with_exe(exe, config)
    }

    /// Builds a factory that re-execs an arbitrary binary.
    ///
    /// Useful in tests: `env!("CARGO_BIN_EXE_acton-ai")` gives a guaranteed
    /// path to the crate's binary without consulting `current_exe`.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] if the path cannot be canonicalized or the
    /// configuration fails validation.
    pub fn with_exe(exe: PathBuf, config: ProcessSandboxConfig) -> Result<Self, ToolError> {
        config.validate()?;
        let exe = canonicalize(&exe)?;
        Ok(Self { exe, config })
    }

    /// Returns the resolved executable path.
    #[must_use]
    pub fn exe(&self) -> &Path {
        &self.exe
    }

    /// Returns the sandbox configuration.
    #[must_use]
    pub fn config(&self) -> &ProcessSandboxConfig {
        &self.config
    }
}

impl SandboxFactory for ProcessSandboxFactory {
    fn create(&self) -> SandboxFactoryFuture {
        let exe = self.exe.clone();
        let config = self.config.clone();
        Box::pin(async move {
            let sandbox: Box<dyn Sandbox> = Box::new(ProcessSandbox {
                exe,
                config,
                destroyed: AtomicBool::new(false),
            });
            Ok(sandbox)
        })
    }

    fn is_available(&self) -> bool {
        self.exe.exists()
    }
}

fn canonicalize(path: &Path) -> Result<PathBuf, ToolError> {
    path.canonicalize().map_err(|err| {
        ToolError::sandbox_error(format!(
            "failed to canonicalize sandbox executable {}: {err}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_invalid_config() {
        let mut bad = ProcessSandboxConfig::new();
        bad.env_allowlist.clear();
        let err = ProcessSandboxFactory::with_exe(PathBuf::from("/bin/true"), bad).unwrap_err();
        assert!(err.to_string().contains("env_allowlist"));
    }

    #[test]
    fn with_exe_rejects_missing_binary() {
        let err = ProcessSandboxFactory::with_exe(
            PathBuf::from("/nonexistent/acton-ai-sandbox-factory-test"),
            ProcessSandboxConfig::new(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("canonicalize"));
    }

    #[test]
    fn with_exe_canonicalizes_symlinks() {
        // `/bin` is commonly a symlink to `/usr/bin` on modern Linux; pick a
        // binary that is virtually always present.
        let factory = ProcessSandboxFactory::with_exe(
            PathBuf::from("/bin/true"),
            ProcessSandboxConfig::new(),
        )
        .expect("/bin/true exists on supported Linux distributions");
        assert!(factory.exe().is_absolute());
        assert!(factory.is_available());
    }
}
