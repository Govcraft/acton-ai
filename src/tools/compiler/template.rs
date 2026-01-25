//! Code template for wrapping agent-generated code.
//!
//! Provides the template that wraps user code in a no_std guest binary
//! with safety guarantees like `#![forbid(unsafe_code)]`.

use super::error::CompilationError;

/// Template for wrapping agent code in a no_std guest binary.
///
/// The template provides:
/// - `#![no_std]` and `#![no_main]` for bare-metal execution
/// - `#![forbid(unsafe_code)]` to prevent unsafe operations (configurable)
/// - Access to `alloc` crate for heap allocations
/// - Hyperlight guest function bindings for host communication
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::compiler::CodeTemplate;
///
/// let template = CodeTemplate::default();
/// let wrapped = template.wrap("input.to_uppercase()")?;
/// // wrapped contains a complete no_std Rust crate
/// ```
#[derive(Debug, Clone)]
pub struct CodeTemplate {
    /// Whether to forbid unsafe code.
    forbid_unsafe: bool,
}

impl Default for CodeTemplate {
    fn default() -> Self {
        Self { forbid_unsafe: true }
    }
}

impl CodeTemplate {
    /// Creates a new `CodeTemplate` with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether to forbid unsafe code.
    ///
    /// When enabled (the default), the generated code will include
    /// `#![forbid(unsafe_code)]`, causing compilation to fail if
    /// the user code contains any unsafe blocks.
    ///
    /// # Arguments
    ///
    /// * `forbid` - Whether to forbid unsafe code
    #[must_use]
    pub fn with_forbid_unsafe(mut self, forbid: bool) -> Self {
        self.forbid_unsafe = forbid;
        self
    }

    /// Wraps code in the no_std template.
    ///
    /// # Arguments
    ///
    /// * `code` - The Rust function body to wrap
    ///
    /// # Returns
    ///
    /// The complete Rust source code ready for compilation.
    ///
    /// # Errors
    ///
    /// Returns `CompilationError::TemplateFailed` if the code is empty.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let template = CodeTemplate::default();
    ///
    /// // Simple expression
    /// let wrapped = template.wrap("input.to_uppercase()")?;
    ///
    /// // Multi-statement code
    /// let wrapped = template.wrap(r#"
    ///     let parts: Vec<&str> = input.split(',').collect();
    ///     parts.join("-")
    /// "#)?;
    /// ```
    pub fn wrap(&self, code: &str) -> Result<String, CompilationError> {
        if code.trim().is_empty() {
            return Err(CompilationError::template_failed("code cannot be empty"));
        }

        let unsafe_attr = if self.forbid_unsafe {
            "#![forbid(unsafe_code)]"
        } else {
            ""
        };

        Ok(format!(
            r#"#![no_std]
#![no_main]
{unsafe_attr}

extern crate alloc;

use alloc::string::String;
use alloc::format;
use alloc::vec::Vec;
use hyperlight_guest_bin::{{guest_function, host_function}};

/// Host function for logging (optional).
#[host_function]
fn host_log(msg: String) -> i32;

/// The code execution function called by the host.
///
/// Takes an input string and returns the result as a string.
#[guest_function("run_code")]
fn run_code(input: String) -> String {{
    // Agent-generated code begins here
    {code}
    // Agent-generated code ends here
}}
"#
        ))
    }

    /// Returns the Cargo.toml content for the guest project.
    ///
    /// The generated Cargo.toml configures:
    /// - Static library output for linking with the Hyperlight host
    /// - Dependency on `hyperlight-guest-bin` for guest function macros
    /// - Release profile optimized for size with LTO
    #[must_use]
    pub fn cargo_toml(&self) -> String {
        r#"[package]
name = "rust_code_guest"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["staticlib"]

[dependencies]
hyperlight-guest-bin = "0.12"

[profile.release]
panic = "abort"
lto = true
opt-level = "s"
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::compiler::CompilationErrorKind;

    // --- CodeTemplate construction tests ---

    #[test]
    fn template_default_forbids_unsafe() {
        let template = CodeTemplate::default();
        assert!(template.forbid_unsafe);
    }

    #[test]
    fn template_new_equals_default() {
        let t1 = CodeTemplate::new();
        let t2 = CodeTemplate::default();
        assert_eq!(t1.forbid_unsafe, t2.forbid_unsafe);
    }

    #[test]
    fn template_with_forbid_unsafe_false() {
        let template = CodeTemplate::new().with_forbid_unsafe(false);
        assert!(!template.forbid_unsafe);
    }

    // --- Template wrap tests ---

    #[test]
    fn template_wrap_includes_no_std() {
        let template = CodeTemplate::default();
        let wrapped = template.wrap("input.to_string()").unwrap();
        assert!(wrapped.contains("#![no_std]"));
    }

    #[test]
    fn template_wrap_includes_no_main() {
        let template = CodeTemplate::default();
        let wrapped = template.wrap("input.to_string()").unwrap();
        assert!(wrapped.contains("#![no_main]"));
    }

    #[test]
    fn template_wrap_includes_forbid_unsafe() {
        let template = CodeTemplate::default();
        let wrapped = template.wrap("input.to_string()").unwrap();
        assert!(wrapped.contains("#![forbid(unsafe_code)]"));
    }

    #[test]
    fn template_wrap_without_forbid_unsafe() {
        let template = CodeTemplate::new().with_forbid_unsafe(false);
        let wrapped = template.wrap("input.to_string()").unwrap();
        assert!(!wrapped.contains("#![forbid(unsafe_code)]"));
    }

    #[test]
    fn template_wrap_includes_alloc() {
        let template = CodeTemplate::default();
        let wrapped = template.wrap("input.to_string()").unwrap();
        assert!(wrapped.contains("extern crate alloc"));
    }

    #[test]
    fn template_wrap_includes_guest_function() {
        let template = CodeTemplate::default();
        let wrapped = template.wrap("input.to_string()").unwrap();
        assert!(wrapped.contains("#[guest_function(\"run_code\")]"));
    }

    #[test]
    fn template_wrap_includes_host_function() {
        let template = CodeTemplate::default();
        let wrapped = template.wrap("input.to_string()").unwrap();
        assert!(wrapped.contains("#[host_function]"));
        assert!(wrapped.contains("fn host_log"));
    }

    #[test]
    fn template_wrap_includes_user_code() {
        let template = CodeTemplate::default();
        let code = "input.chars().rev().collect()";
        let wrapped = template.wrap(code).unwrap();
        assert!(wrapped.contains(code));
    }

    #[test]
    fn template_wrap_empty_code_fails() {
        let template = CodeTemplate::default();
        let result = template.wrap("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().kind(),
            CompilationErrorKind::TemplateFailed { .. }
        ));
    }

    #[test]
    fn template_wrap_whitespace_only_fails() {
        let template = CodeTemplate::default();
        let result = template.wrap("   \n\t  ");
        assert!(result.is_err());
    }

    #[test]
    fn template_wrap_multiline_code() {
        let template = CodeTemplate::default();
        let code = r#"
            let parts: Vec<&str> = input.split(',').collect();
            parts.join("-")
        "#;
        let result = template.wrap(code);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("split"));
    }

    // --- Cargo.toml tests ---

    #[test]
    fn cargo_toml_has_package() {
        let template = CodeTemplate::default();
        let toml = template.cargo_toml();
        assert!(toml.contains("[package]"));
        assert!(toml.contains("name = \"rust_code_guest\""));
    }

    #[test]
    fn cargo_toml_has_lib_staticlib() {
        let template = CodeTemplate::default();
        let toml = template.cargo_toml();
        assert!(toml.contains("[lib]"));
        assert!(toml.contains("crate-type = [\"staticlib\"]"));
    }

    #[test]
    fn cargo_toml_has_hyperlight_dependency() {
        let template = CodeTemplate::default();
        let toml = template.cargo_toml();
        assert!(toml.contains("[dependencies]"));
        assert!(toml.contains("hyperlight-guest-bin"));
    }

    #[test]
    fn cargo_toml_has_release_profile() {
        let template = CodeTemplate::default();
        let toml = template.cargo_toml();
        assert!(toml.contains("[profile.release]"));
        assert!(toml.contains("panic = \"abort\""));
        assert!(toml.contains("lto = true"));
    }

    // --- Clone tests ---

    #[test]
    fn template_is_clone() {
        let template1 = CodeTemplate::new().with_forbid_unsafe(false);
        let template2 = template1.clone();
        assert_eq!(template1.forbid_unsafe, template2.forbid_unsafe);
    }
}
