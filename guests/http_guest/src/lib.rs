//! HTTP client guest for Hyperlight sandboxes.
//!
//! This guest binary handles HTTP fetch requests from the host
//! and returns structured responses.
//!
//! # Protocol
//!
//! Input: JSON string with format:
//! ```json
//! {
//!     "url": "https://example.com",
//!     "method": "GET",
//!     "headers": {},
//!     "body": null,
//!     "timeout_secs": 30
//! }
//! ```
//!
//! Output: JSON string with format:
//! ```json
//! {
//!     "status": 200,
//!     "headers": {},
//!     "body": "...",
//!     "success": true,
//!     "error": null
//! }
//! ```

#![no_std]
#![no_main]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use hyperlight_guest_bin::guest_function;

/// Perform an HTTP fetch operation.
///
/// # Arguments
///
/// * `input` - JSON-encoded HTTP request
///
/// # Returns
///
/// JSON-encoded HTTP response with status, headers, body, and success flag.
///
/// # Implementation Note
///
/// Actual HTTP requests require host functions for network I/O.
/// This placeholder returns a structured response indicating the limitation.
#[guest_function("Fetch")]
fn fetch(input: String) -> String {
    format!(
        r#"{{"status":0,"headers":{{}},"body":"","success":false,"error":"HTTP fetch requires host function support. Input length: {}"}}"#,
        input.len()
    )
}
