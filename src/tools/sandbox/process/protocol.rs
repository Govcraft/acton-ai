//! Length-prefixed JSON wire protocol between parent and child sandbox processes.
//!
//! The parent serializes a [`Request`] to the child's stdin; the child replies
//! with a [`Response`] on its stdout. Both payloads are framed with a 4-byte
//! big-endian length prefix followed by the JSON body. The helpers in this
//! module are synchronous and operate on any [`std::io::Read`]/[`std::io::Write`],
//! so they are callable from both the blocking child runner and unit tests.

use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Upper bound on a single framed payload. Anything larger is rejected to
/// keep the wire protocol resistant to accidental or malicious resource
/// exhaustion. 16 MiB is generous for tool arguments/results while bounded.
pub const MAX_FRAME_SIZE: u32 = 16 * 1024 * 1024;

/// A tool invocation request sent from the parent process to the child.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    /// Name of the tool to execute inside the child process.
    pub tool_name: String,
    /// JSON arguments to pass to the tool.
    pub args: Value,
    /// Absolute deadline in milliseconds since the UNIX epoch. When the child
    /// observes the current time has exceeded this value it aborts execution.
    pub deadline_ms: u64,
}

/// Internal wire representation of a [`Response`].
///
/// Flattening the `Result` into a two-variant enum keeps the JSON shape
/// predictable (`{"Ok": ...}` / `{"Err": ...}`) and avoids any ambiguity with
/// serde's default handling of `Result`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum ResponseBody {
    Ok(Value),
    Err(String),
}

/// A tool invocation response returned from the child process to the parent.
///
/// The public API exposes the result as a [`Result<Value, String>`], but the
/// wire format uses an explicit two-variant enum (see [`ResponseBody`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// Either the tool output value or a stringified error.
    pub result: Result<Value, String>,
}

impl Response {
    /// Creates a successful response carrying the given JSON value.
    #[must_use]
    pub fn ok(value: Value) -> Self {
        Self { result: Ok(value) }
    }

    /// Creates an error response with the given message.
    #[must_use]
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            result: Err(message.into()),
        }
    }
}

impl Serialize for Response {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let body = match &self.result {
            Ok(value) => ResponseBody::Ok(value.clone()),
            Err(message) => ResponseBody::Err(message.clone()),
        };
        body.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Response {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let body = ResponseBody::deserialize(deserializer)?;
        let result = match body {
            ResponseBody::Ok(value) => Ok(value),
            ResponseBody::Err(message) => Err(message),
        };
        Ok(Self { result })
    }
}

/// Writes a length-prefixed JSON payload to `w`.
fn write_frame<W: Write, T: Serialize>(w: &mut W, value: &T) -> io::Result<()> {
    let bytes =
        serde_json::to_vec(value).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let len = u32::try_from(bytes.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("payload exceeds u32::MAX bytes ({} bytes)", bytes.len()),
        )
    })?;
    if len > MAX_FRAME_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "payload of {} bytes exceeds MAX_FRAME_SIZE ({} bytes)",
                len, MAX_FRAME_SIZE
            ),
        ));
    }
    w.write_all(&len.to_be_bytes())?;
    w.write_all(&bytes)?;
    Ok(())
}

/// Reads a length-prefixed JSON payload from `r` and deserializes it.
fn read_frame<R: Read, T: for<'de> Deserialize<'de>>(r: &mut R) -> io::Result<T> {
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes)?;
    let len = u32::from_be_bytes(len_bytes);
    if len > MAX_FRAME_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "incoming payload of {} bytes exceeds MAX_FRAME_SIZE ({} bytes)",
                len, MAX_FRAME_SIZE
            ),
        ));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    serde_json::from_slice(&buf).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

/// Writes a [`Request`] to `w` with a 4-byte big-endian length prefix.
///
/// # Errors
///
/// Returns an [`io::Error`] if serialization fails, the payload exceeds
/// [`MAX_FRAME_SIZE`], or the underlying writer fails.
pub fn write_request<W: Write>(w: &mut W, req: &Request) -> io::Result<()> {
    write_frame(w, req)
}

/// Reads a length-prefixed [`Request`] from `r`.
///
/// # Errors
///
/// Returns an [`io::Error`] with [`io::ErrorKind::UnexpectedEof`] if the
/// stream ends before the frame is complete, [`io::ErrorKind::InvalidInput`]
/// if the frame is too large, or [`io::ErrorKind::InvalidData`] if the JSON
/// payload is malformed.
pub fn read_request<R: Read>(r: &mut R) -> io::Result<Request> {
    read_frame(r)
}

/// Writes a [`Response`] to `w` with a 4-byte big-endian length prefix.
///
/// # Errors
///
/// See [`write_request`].
pub fn write_response<W: Write>(w: &mut W, resp: &Response) -> io::Result<()> {
    write_frame(w, resp)
}

/// Reads a length-prefixed [`Response`] from `r`.
///
/// # Errors
///
/// See [`read_request`].
pub fn read_response<R: Read>(r: &mut R) -> io::Result<Response> {
    read_frame(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_roundtrip_happy_path() {
        let req = Request {
            tool_name: "bash".to_string(),
            args: json!({"command": "echo hello"}),
            deadline_ms: 1_700_000_000_000,
        };
        let mut buf = Vec::new();
        write_request(&mut buf, &req).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let got = read_request(&mut cursor).unwrap();
        assert_eq!(got, req);
    }

    #[test]
    fn response_ok_roundtrip() {
        let resp = Response::ok(json!({"stdout": "hello", "exit_code": 0}));
        let mut buf = Vec::new();
        write_response(&mut buf, &resp).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let got = read_response(&mut cursor).unwrap();
        assert_eq!(got, resp);
    }

    #[test]
    fn response_err_roundtrip() {
        let resp = Response::err("permission denied");
        let mut buf = Vec::new();
        write_response(&mut buf, &resp).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let got = read_response(&mut cursor).unwrap();
        assert_eq!(got, resp);
        assert_eq!(got.result, Err("permission denied".to_string()));
    }

    #[test]
    fn empty_args_roundtrip() {
        let req = Request {
            tool_name: "noop".to_string(),
            args: json!({}),
            deadline_ms: 0,
        };
        let mut buf = Vec::new();
        write_request(&mut buf, &req).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let got = read_request(&mut cursor).unwrap();
        assert_eq!(got, req);
    }

    #[test]
    fn oversized_payload_errors_on_write() {
        // Construct a request whose serialized JSON exceeds MAX_FRAME_SIZE.
        let big = "x".repeat(MAX_FRAME_SIZE as usize + 1);
        let req = Request {
            tool_name: "bash".to_string(),
            args: json!({"payload": big}),
            deadline_ms: 0,
        };
        let mut buf = Vec::new();
        let err = write_request(&mut buf, &req).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn oversized_declared_length_errors_on_read() {
        // Declare a length just over MAX_FRAME_SIZE. read_request must reject
        // before attempting to allocate the buffer.
        let declared = MAX_FRAME_SIZE + 1;
        let mut bytes = declared.to_be_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let mut cursor = std::io::Cursor::new(bytes);
        let err = read_request(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn short_length_prefix_returns_unexpected_eof() {
        // Fewer than 4 bytes: read_exact on the prefix must fail with EOF.
        let bytes = [0u8, 0u8];
        let mut cursor = std::io::Cursor::new(&bytes[..]);
        let err = read_request(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn short_body_returns_unexpected_eof() {
        // Prefix claims 10 bytes, body only provides 3.
        let mut bytes = 10u32.to_be_bytes().to_vec();
        bytes.extend_from_slice(b"abc");
        let mut cursor = std::io::Cursor::new(bytes);
        let err = read_request(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn malformed_json_returns_invalid_data() {
        // Valid length prefix but garbage JSON body.
        let payload = b"not json!!!";
        let mut bytes = (payload.len() as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(payload);
        let mut cursor = std::io::Cursor::new(bytes);
        let err = read_request(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn response_wire_format_uses_ok_variant_tag() {
        // Lock in the wire shape so other language implementations have a
        // stable target. Serde's default for externally-tagged enums emits
        // {"Ok": ...} / {"Err": ...}.
        let resp = Response::ok(json!(42));
        let mut buf = Vec::new();
        write_response(&mut buf, &resp).unwrap();
        // Skip the 4-byte length prefix.
        let body = std::str::from_utf8(&buf[4..]).unwrap();
        assert_eq!(body, r#"{"Ok":42}"#);
    }

    #[test]
    fn response_wire_format_uses_err_variant_tag() {
        let resp = Response::err("boom");
        let mut buf = Vec::new();
        write_response(&mut buf, &resp).unwrap();
        let body = std::str::from_utf8(&buf[4..]).unwrap();
        assert_eq!(body, r#"{"Err":"boom"}"#);
    }
}
