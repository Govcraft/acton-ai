//! HTTP client guest for Hyperlight sandboxes.
//!
//! This guest binary handles HTTP fetch requests from the host via the
//! `http_fetch` function. The guest validates input and delegates actual
//! network I/O to host functions.
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

use alloc::string::String;
use hyperlight_guest_bin::{guest_function, host_function};

/// Host function for HTTP requests.
/// This is implemented on the host side and called by the guest.
/// The host's registered function name must match: "host_http_fetch"
#[host_function]
fn host_http_fetch(request: String) -> String;

/// Perform an HTTP fetch operation.
///
/// The function name `http_fetch` must match what the host calls via
/// `sandbox.call("http_fetch", ...)`.
///
/// # Arguments
///
/// * `input` - JSON-encoded HTTP request with url, method, headers, body, timeout_secs
///
/// # Returns
///
/// JSON-encoded HTTP response with status, headers, body, success, and error.
#[guest_function("http_fetch")]
fn http_fetch(input: String) -> String {
    // Call the host function to perform the HTTP request
    // The host is responsible for parsing the JSON and making the actual request
    host_http_fetch(input)
}
