//! Security utilities for tool execution.
//!
//! This module provides security controls for filesystem and other operations:
//!
//! - **Path Validation**: Restricts filesystem access to allowed directories
//!
//! ## Path Validation
//!
//! The [`PathValidator`] ensures filesystem tools only access permitted locations:
//!
//! ```rust,ignore
//! use std::path::{Path, PathBuf};
//! use acton_ai::tools::security::PathValidator;
//!
//! let validator = PathValidator::new()
//!     .with_allowed_root(PathBuf::from("/home/user/project"));
//!
//! // Validate paths before filesystem operations
//! match validator.validate(Path::new("/home/user/project/src/main.rs")) {
//!     Ok(canonical) => println!("Validated: {}", canonical.display()),
//!     Err(e) => eprintln!("Rejected: {}", e),
//! }
//! ```
//!
//! ## Default Security Settings
//!
//! By default, `PathValidator` blocks:
//! - Path traversal attempts (`..`)
//! - Git directories (`.git`)
//! - Environment files (`.env`)
//!
//! And restricts access to the current working directory.

mod path;

pub use path::{PathValidationError, PathValidator};
