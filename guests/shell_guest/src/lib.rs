//! Shell execution guest for Hyperlight sandboxes.
//!
//! This guest binary handles shell command execution requests
//! from the host and returns structured output.
//!
//! # Protocol
//!
//! Input: JSON string with format:
//! ```json
//! {
//!     "command": "echo hello",
//!     "args": {},
//!     "timeout_secs": 30
//! }
//! ```
//!
//! Output: JSON string with format:
//! ```json
//! {
//!     "exit_code": 0,
//!     "stdout": "hello\n",
//!     "stderr": "",
//!     "success": true,
//!     "truncated": false
//! }
//! ```

#![no_std]
#![no_main]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use hyperlight_guest_bin::guest_function;

/// Execute a shell command and return structured output.
///
/// # Arguments
///
/// * `input` - JSON-encoded shell request
///
/// # Returns
///
/// JSON-encoded shell response with exit code, stdout, stderr, and success flag.
///
/// # Implementation Note
///
/// Actual shell execution requires host functions for process spawning.
/// This placeholder returns a structured response indicating the limitation.
#[guest_function("Execute")]
fn execute(input: String) -> String {
    // Parse input to validate format (in a real implementation)
    // For now, return a placeholder response
    format!(
        r#"{{"exit_code":1,"stdout":"","stderr":"Shell execution requires host function support. Input length: {}","success":false,"truncated":false}}"#,
        input.len()
    )
}
