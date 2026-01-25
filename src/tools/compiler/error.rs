//! Compilation error types.
//!
//! Custom error types for Rust compilation operations including clippy
//! verification, compilation failures, template wrapping, and caching.

use std::fmt;
use std::io;

/// Errors that can occur during Rust code compilation.
///
/// This type uses `Box<CompilationErrorKind>` to keep the error size small,
/// enabling efficient use in Result types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilationError {
    /// The specific error that occurred (boxed for size efficiency)
    kind: Box<CompilationErrorKind>,
}

/// Specific compilation error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationErrorKind {
    /// Code failed clippy checks.
    ClippyFailed {
        /// The clippy error output
        errors: String,
        /// Number of errors found
        error_count: usize,
    },

    /// Compilation to native binary failed.
    CompilationFailed {
        /// The compiler error output
        errors: String,
        /// Exit code from cargo
        exit_code: Option<i32>,
    },

    /// Template wrapping failed.
    TemplateFailed {
        /// Reason for template failure
        reason: String,
    },

    /// I/O error during compilation.
    IoError {
        /// Operation that failed
        operation: String,
        /// Error message
        message: String,
    },

    /// Cache error.
    CacheError {
        /// Description of the cache error
        reason: String,
    },

    /// Required tooling not available.
    ToolchainError {
        /// What's missing
        missing: String,
        /// How to install
        install_hint: String,
    },
}

impl CompilationError {
    /// Creates a new `CompilationError` with the given kind.
    #[must_use]
    pub fn new(kind: CompilationErrorKind) -> Self {
        Self {
            kind: Box::new(kind),
        }
    }

    /// Returns a reference to the error kind.
    #[must_use]
    pub fn kind(&self) -> &CompilationErrorKind {
        &self.kind
    }

    /// Creates a clippy failed error.
    #[must_use]
    pub fn clippy_failed(errors: impl Into<String>, error_count: usize) -> Self {
        Self::new(CompilationErrorKind::ClippyFailed {
            errors: errors.into(),
            error_count,
        })
    }

    /// Creates a compilation failed error.
    #[must_use]
    pub fn compilation_failed(errors: impl Into<String>, exit_code: Option<i32>) -> Self {
        Self::new(CompilationErrorKind::CompilationFailed {
            errors: errors.into(),
            exit_code,
        })
    }

    /// Creates a template failed error.
    #[must_use]
    pub fn template_failed(reason: impl Into<String>) -> Self {
        Self::new(CompilationErrorKind::TemplateFailed {
            reason: reason.into(),
        })
    }

    /// Creates an I/O error.
    #[must_use]
    pub fn io_error(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(CompilationErrorKind::IoError {
            operation: operation.into(),
            message: message.into(),
        })
    }

    /// Creates a cache error.
    #[must_use]
    pub fn cache_error(reason: impl Into<String>) -> Self {
        Self::new(CompilationErrorKind::CacheError {
            reason: reason.into(),
        })
    }

    /// Creates a toolchain error.
    #[must_use]
    pub fn toolchain_error(missing: impl Into<String>, install_hint: impl Into<String>) -> Self {
        Self::new(CompilationErrorKind::ToolchainError {
            missing: missing.into(),
            install_hint: install_hint.into(),
        })
    }

    /// Returns true if this error is due to code issues (clippy/compilation).
    #[must_use]
    pub fn is_code_error(&self) -> bool {
        matches!(
            *self.kind,
            CompilationErrorKind::ClippyFailed { .. }
                | CompilationErrorKind::CompilationFailed { .. }
        )
    }

    /// Returns true if this error is due to infrastructure issues.
    #[must_use]
    pub fn is_infrastructure_error(&self) -> bool {
        matches!(
            *self.kind,
            CompilationErrorKind::IoError { .. }
                | CompilationErrorKind::CacheError { .. }
                | CompilationErrorKind::ToolchainError { .. }
        )
    }
}

impl fmt::Display for CompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind.as_ref() {
            CompilationErrorKind::ClippyFailed { errors, error_count } => {
                write!(
                    f,
                    "clippy found {} error(s); fix the issues and retry:\n{}",
                    error_count, errors
                )
            }
            CompilationErrorKind::CompilationFailed { errors, exit_code } => {
                if let Some(code) = exit_code {
                    write!(
                        f,
                        "compilation failed (exit code {}); check the code:\n{}",
                        code, errors
                    )
                } else {
                    write!(f, "compilation failed; check the code:\n{}", errors)
                }
            }
            CompilationErrorKind::TemplateFailed { reason } => {
                write!(f, "failed to wrap code in template: {}", reason)
            }
            CompilationErrorKind::IoError { operation, message } => {
                write!(f, "I/O error during {}: {}", operation, message)
            }
            CompilationErrorKind::CacheError { reason } => {
                write!(f, "compilation cache error: {}", reason)
            }
            CompilationErrorKind::ToolchainError { missing, install_hint } => {
                write!(
                    f,
                    "required tooling not available: {}; install with: {}",
                    missing, install_hint
                )
            }
        }
    }
}

impl std::error::Error for CompilationError {}

impl From<io::Error> for CompilationError {
    fn from(error: io::Error) -> Self {
        Self::io_error("file operation", error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clippy_failed_error_display() {
        let error = CompilationError::clippy_failed("error: unused variable", 1);
        let msg = error.to_string();
        assert!(msg.contains("clippy"));
        assert!(msg.contains("1 error"));
        assert!(msg.contains("unused variable"));
    }

    #[test]
    fn compilation_failed_with_exit_code() {
        let error = CompilationError::compilation_failed("syntax error", Some(1));
        let msg = error.to_string();
        assert!(msg.contains("compilation failed"));
        assert!(msg.contains("exit code 1"));
        assert!(msg.contains("syntax error"));
    }

    #[test]
    fn compilation_failed_without_exit_code() {
        let error = CompilationError::compilation_failed("unknown error", None);
        let msg = error.to_string();
        assert!(msg.contains("compilation failed"));
        assert!(!msg.contains("exit code"));
    }

    #[test]
    fn template_failed_error_display() {
        let error = CompilationError::template_failed("code cannot be empty");
        let msg = error.to_string();
        assert!(msg.contains("failed to wrap code"));
        assert!(msg.contains("code cannot be empty"));
    }

    #[test]
    fn io_error_display() {
        let error = CompilationError::io_error("writing file", "permission denied");
        let msg = error.to_string();
        assert!(msg.contains("I/O error"));
        assert!(msg.contains("writing file"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn cache_error_display() {
        let error = CompilationError::cache_error("cache full");
        let msg = error.to_string();
        assert!(msg.contains("cache error"));
        assert!(msg.contains("cache full"));
    }

    #[test]
    fn toolchain_error_display() {
        let error = CompilationError::toolchain_error("cargo", "install via rustup");
        let msg = error.to_string();
        assert!(msg.contains("tooling not available"));
        assert!(msg.contains("cargo"));
        assert!(msg.contains("install via rustup"));
    }

    #[test]
    fn is_code_error_for_clippy() {
        let error = CompilationError::clippy_failed("", 0);
        assert!(error.is_code_error());
    }

    #[test]
    fn is_code_error_for_compilation() {
        let error = CompilationError::compilation_failed("", None);
        assert!(error.is_code_error());
    }

    #[test]
    fn is_not_code_error_for_io() {
        let error = CompilationError::io_error("test", "test");
        assert!(!error.is_code_error());
    }

    #[test]
    fn is_infrastructure_error_for_io() {
        let error = CompilationError::io_error("test", "test");
        assert!(error.is_infrastructure_error());
    }

    #[test]
    fn is_infrastructure_error_for_cache() {
        let error = CompilationError::cache_error("test");
        assert!(error.is_infrastructure_error());
    }

    #[test]
    fn is_infrastructure_error_for_toolchain() {
        let error = CompilationError::toolchain_error("test", "test");
        assert!(error.is_infrastructure_error());
    }

    #[test]
    fn is_not_infrastructure_error_for_clippy() {
        let error = CompilationError::clippy_failed("", 0);
        assert!(!error.is_infrastructure_error());
    }

    #[test]
    fn error_kind_accessor() {
        let error = CompilationError::template_failed("reason");
        assert!(matches!(
            error.kind(),
            CompilationErrorKind::TemplateFailed { .. }
        ));
    }

    #[test]
    fn from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let compilation_err: CompilationError = io_err.into();
        assert!(matches!(
            compilation_err.kind(),
            CompilationErrorKind::IoError { .. }
        ));
    }

    #[test]
    fn errors_are_clone() {
        let error1 = CompilationError::clippy_failed("test", 1);
        let error2 = error1.clone();
        assert_eq!(error1, error2);
    }

    #[test]
    fn errors_are_eq() {
        let error1 = CompilationError::template_failed("reason");
        let error2 = CompilationError::template_failed("reason");
        assert_eq!(error1, error2);

        let error3 = CompilationError::template_failed("different");
        assert_ne!(error1, error3);
    }

    #[test]
    fn error_is_std_error() {
        let error = CompilationError::clippy_failed("test", 1);
        let _: &dyn std::error::Error = &error;
    }
}
