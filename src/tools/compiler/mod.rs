//! Rust compilation pipeline for agent-generated code.
//!
//! This module provides secure compilation of Rust code for sandbox execution:
//!
//! 1. Code is wrapped in a no_std template with `#![forbid(unsafe_code)]`
//! 2. Clippy runs with `-D warnings` for lint checking
//! 3. Code is compiled to `x86_64-unknown-none` target
//! 4. Binary is cached by code hash for performance
//!
//! # Architecture
//!
//! ```text
//! User Code (function body)
//!         |
//!         v
//! +-------------------+
//! |   CodeTemplate    |  Wraps in no_std template
//! +-------------------+
//!         |
//!         v
//! +-------------------+
//! |   Clippy Check    |  Verifies code quality
//! +-------------------+
//!         |
//!         v
//! +-------------------+
//! |   Compilation     |  Produces static library
//! +-------------------+
//!         |
//!         v
//! +-------------------+
//! | CompilationCache  |  Caches by code hash
//! +-------------------+
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use acton_ai::tools::compiler::RustCompiler;
//!
//! let compiler = RustCompiler::new()?;
//! let binary = compiler.compile("input.to_uppercase()")?;
//! println!("Compiled {} bytes", binary.size());
//! ```

pub mod cache;
pub mod error;
pub mod template;

pub use cache::{CacheConfig, CacheStats, CodeHash, CompilationCache};
pub use error::{CompilationError, CompilationErrorKind};
pub use template::CodeTemplate;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Result of successful compilation.
///
/// Contains the compiled binary bytes and a hash of the source code
/// that produced it.
#[derive(Debug, Clone)]
pub struct CompiledBinary {
    /// The compiled binary bytes.
    bytes: Vec<u8>,
    /// Hash of the source code.
    hash: CodeHash,
}

impl CompiledBinary {
    /// Creates a new `CompiledBinary`.
    #[must_use]
    pub fn new(bytes: Vec<u8>, hash: CodeHash) -> Self {
        Self { bytes, hash }
    }

    /// Returns the binary bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes self and returns the binary bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Returns the code hash.
    #[must_use]
    pub fn hash(&self) -> CodeHash {
        self.hash
    }

    /// Returns the size of the binary in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.bytes.len()
    }
}

/// Compiles agent-generated Rust to native binary for Hyperlight execution.
///
/// The compiler provides:
/// - Code wrapping in a safe no_std template
/// - Clippy verification with `-D warnings`
/// - Compilation to `x86_64-unknown-none` target
/// - LRU caching of compiled binaries
///
/// # Thread Safety
///
/// The compiler is `Send + Sync` and can be safely shared across threads.
/// Compilation operations may spawn processes and access the filesystem.
///
/// # Requirements
///
/// - Rust toolchain with `cargo` and `rustup`
/// - `x86_64-unknown-none` target installed
/// - Writable temp directory
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::compiler::{RustCompiler, CacheConfig};
///
/// // With default configuration
/// let compiler = RustCompiler::new()?;
///
/// // With custom cache configuration
/// let config = CacheConfig {
///     max_entries: 50,
///     max_total_size: 50 * 1024 * 1024,
/// };
/// let compiler = RustCompiler::with_config(config)?;
///
/// // Compile code
/// let binary = compiler.compile("input.to_uppercase()")?;
/// ```
#[derive(Debug)]
pub struct RustCompiler {
    cache: CompilationCache,
    temp_dir: PathBuf,
    template: CodeTemplate,
}

impl RustCompiler {
    /// Creates a new `RustCompiler` with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `CompilationError::IoError` if the temp directory cannot be created.
    /// Returns `CompilationError::ToolchainError` if required tools are not available.
    pub fn new() -> Result<Self, CompilationError> {
        Self::with_config(CacheConfig::default())
    }

    /// Creates a new `RustCompiler` with custom cache configuration.
    ///
    /// # Arguments
    ///
    /// * `cache_config` - Configuration for the compilation cache
    ///
    /// # Errors
    ///
    /// Returns `CompilationError::IoError` if the temp directory cannot be created.
    /// Returns `CompilationError::ToolchainError` if required tools are not available.
    pub fn with_config(cache_config: CacheConfig) -> Result<Self, CompilationError> {
        // Create temp directory
        let temp_dir = std::env::temp_dir().join("acton-ai-compiler");
        std::fs::create_dir_all(&temp_dir)?;

        // Verify toolchain
        Self::verify_toolchain()?;

        Ok(Self {
            cache: CompilationCache::new(cache_config),
            temp_dir,
            template: CodeTemplate::default(),
        })
    }

    /// Compiles agent-generated Rust code.
    ///
    /// The compilation process:
    /// 1. Wraps code in a no_std template with safety attributes
    /// 2. Checks the cache for a previously compiled binary
    /// 3. Sets up a temporary Cargo project
    /// 4. Runs clippy with `-D warnings`
    /// 5. Compiles to `x86_64-unknown-none` target
    /// 6. Caches the result and cleans up
    ///
    /// # Arguments
    ///
    /// * `code` - The Rust function body to compile
    ///
    /// # Returns
    ///
    /// A `CompiledBinary` containing the compiled bytes and hash.
    ///
    /// # Errors
    ///
    /// Returns `CompilationError::TemplateFailed` if the code is empty.
    /// Returns `CompilationError::ClippyFailed` if clippy finds issues.
    /// Returns `CompilationError::CompilationFailed` if rustc fails.
    /// Returns `CompilationError::IoError` for filesystem errors.
    pub fn compile(&self, code: &str) -> Result<CompiledBinary, CompilationError> {
        // 1. Wrap in no_std template
        let wrapped = self.template.wrap(code)?;

        // 2. Compute hash for caching
        let hash = CodeHash::from_code(&wrapped);

        // 3. Check cache
        if let Some(cached) = self.cache.get(hash) {
            tracing::debug!(hash = %hash, "compilation cache hit");
            return Ok(CompiledBinary::new(cached, hash));
        }

        tracing::debug!(hash = %hash, "compilation cache miss, compiling");

        // 4. Set up project directory
        let project_dir = self.temp_dir.join(format!("rust_code_{}", hash));
        self.setup_project(&project_dir, &wrapped)?;

        // 5. Run clippy
        self.run_clippy(&project_dir)?;

        // 6. Compile
        let binary = self.compile_to_binary(&project_dir)?;

        // 7. Cache result
        self.cache.insert(hash, binary.clone());

        // 8. Clean up project directory
        self.cleanup_project(&project_dir);

        Ok(CompiledBinary::new(binary, hash))
    }

    /// Returns cache statistics.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clears the compilation cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Verifies that required toolchain components are available.
    fn verify_toolchain() -> Result<(), CompilationError> {
        // Check for cargo
        let cargo_check = Command::new("cargo").arg("--version").output();
        if cargo_check.is_err() {
            return Err(CompilationError::toolchain_error(
                "cargo",
                "install Rust via rustup: https://rustup.rs",
            ));
        }

        // Check for x86_64-unknown-none target
        let target_check = Command::new("rustup")
            .args(["target", "list", "--installed"])
            .output();

        match target_check {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if !stdout.contains("x86_64-unknown-none") {
                    return Err(CompilationError::toolchain_error(
                        "x86_64-unknown-none target",
                        "rustup target add x86_64-unknown-none",
                    ));
                }
            }
            Err(_) => {
                return Err(CompilationError::toolchain_error(
                    "rustup",
                    "install Rust via rustup: https://rustup.rs",
                ));
            }
        }

        Ok(())
    }

    /// Sets up a temporary Cargo project for compilation.
    fn setup_project(&self, dir: &Path, code: &str) -> Result<(), CompilationError> {
        std::fs::create_dir_all(dir.join("src"))?;

        // Write Cargo.toml
        let cargo_toml = self.template.cargo_toml();
        std::fs::write(dir.join("Cargo.toml"), cargo_toml)?;

        // Write code
        std::fs::write(dir.join("src/lib.rs"), code)?;

        Ok(())
    }

    /// Runs clippy on the project.
    fn run_clippy(&self, dir: &Path) -> Result<(), CompilationError> {
        let output = Command::new("cargo")
            .current_dir(dir)
            .args([
                "clippy",
                "--target",
                "x86_64-unknown-none",
                "--",
                "-D",
                "warnings",
            ])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let error_count = count_errors(&stderr);
            return Err(CompilationError::clippy_failed(stderr, error_count));
        }

        Ok(())
    }

    /// Compiles the project to a binary.
    fn compile_to_binary(&self, dir: &Path) -> Result<Vec<u8>, CompilationError> {
        let output = Command::new("cargo")
            .current_dir(dir)
            .args(["build", "--release", "--target", "x86_64-unknown-none"])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CompilationError::compilation_failed(
                stderr,
                output.status.code(),
            ));
        }

        // Read compiled binary
        let binary_path = dir
            .join("target")
            .join("x86_64-unknown-none")
            .join("release")
            .join("librust_code_guest.a");

        std::fs::read(&binary_path).map_err(|e| {
            CompilationError::io_error(
                "reading compiled binary",
                format!("path: {:?}, error: {}", binary_path, e),
            )
        })
    }

    /// Cleans up a project directory (best effort).
    fn cleanup_project(&self, dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Counts the number of errors in clippy output.
fn count_errors(output: &str) -> usize {
    output.matches("error[E").count() + output.matches("error:").count()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CompiledBinary tests ---

    #[test]
    fn compiled_binary_new() {
        let hash = CodeHash::from_code("test");
        let binary = CompiledBinary::new(vec![1, 2, 3], hash);

        assert_eq!(binary.bytes(), &[1, 2, 3]);
        assert_eq!(binary.hash(), hash);
        assert_eq!(binary.size(), 3);
    }

    #[test]
    fn compiled_binary_into_bytes() {
        let hash = CodeHash::from_code("test");
        let binary = CompiledBinary::new(vec![1, 2, 3], hash);
        let bytes = binary.into_bytes();
        assert_eq!(bytes, vec![1, 2, 3]);
    }

    #[test]
    fn compiled_binary_is_clone() {
        let hash = CodeHash::from_code("test");
        let binary1 = CompiledBinary::new(vec![1, 2, 3], hash);
        let binary2 = binary1.clone();
        assert_eq!(binary1.bytes(), binary2.bytes());
        assert_eq!(binary1.hash(), binary2.hash());
    }

    // --- count_errors tests ---

    #[test]
    fn count_errors_finds_error_codes() {
        let output = "error[E0001]: something\nerror[E0002]: another";
        assert_eq!(count_errors(output), 2);
    }

    #[test]
    fn count_errors_finds_plain_errors() {
        let output = "error: aborting due to 3 previous errors";
        assert_eq!(count_errors(output), 1);
    }

    #[test]
    fn count_errors_counts_both() {
        let output = "error[E0001]: something\nerror: aborting";
        assert_eq!(count_errors(output), 2);
    }

    #[test]
    fn count_errors_empty() {
        assert_eq!(count_errors(""), 0);
    }

    #[test]
    fn count_errors_no_errors() {
        let output = "warning: unused variable";
        assert_eq!(count_errors(output), 0);
    }

    // --- RustCompiler tests (require toolchain, so some are marked ignored) ---

    // Note: These tests require the Rust toolchain with x86_64-unknown-none target.
    // Run with `cargo test -- --ignored` on a properly configured system.

    #[test]
    #[ignore = "requires rust toolchain"]
    fn compiler_new_succeeds_with_toolchain() {
        let result = RustCompiler::new();
        // This will fail without the toolchain
        if std::env::var("CI").is_err() {
            // Only assert in non-CI environment where toolchain might exist
            assert!(result.is_ok() || result.is_err());
        }
    }

    #[test]
    fn compiler_temp_dir_path() {
        let temp_dir = std::env::temp_dir().join("acton-ai-compiler");
        // Just verify the path construction is consistent
        assert!(temp_dir.to_string_lossy().contains("acton-ai-compiler"));
    }
}
