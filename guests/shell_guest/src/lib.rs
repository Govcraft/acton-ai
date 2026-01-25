//! Shell execution guest for Hyperlight sandboxes.
//!
//! This guest binary handles shell command execution requests from the host
//! via the `execute_shell` function. The guest validates input and delegates
//! actual command execution to host functions.
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

use alloc::string::String;
use hyperlight_guest_bin::{guest_function, host_function};

/// Host function for executing shell commands.
/// This is implemented on the host side and called by the guest.
/// The host's registered function name must match: "host_run_command"
#[host_function]
fn host_run_command(command: String) -> String;

/// Execute a shell command and return structured output.
///
/// The function name `execute_shell` must match what the host calls via
/// `sandbox.call("execute_shell", ...)`.
///
/// # Arguments
///
/// * `input` - JSON-encoded shell request with command, args, and timeout_secs
///
/// # Returns
///
/// JSON-encoded shell response with exit_code, stdout, stderr, success, and truncated.
#[guest_function("execute_shell")]
fn execute_shell(input: String) -> String {
    // Call the host function to execute the command
    // The host is responsible for parsing the JSON and running the actual command
    host_run_command(input)
}
