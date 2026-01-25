//! Guest binary types for Hyperlight sandboxes.
//!
//! This module defines the types of guest binaries available for sandbox execution
//! and provides access to the embedded binaries.

use std::fmt;

/// Types of guest binaries available for sandbox execution.
///
/// Each guest type represents a specialized binary that runs inside
/// Hyperlight micro-VMs to handle specific operations.
///
/// # Examples
///
/// ```
/// use acton_ai::tools::sandbox::GuestType;
///
/// let guest = GuestType::Shell;
/// assert_eq!(guest.name(), "shell");
/// assert_eq!(format!("{}", guest), "shell");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuestType {
    /// Shell command execution guest.
    ///
    /// Handles bash tool operations by executing shell commands
    /// inside the sandbox and returning structured output.
    Shell,

    /// HTTP client operations guest.
    ///
    /// Handles web_fetch tool operations by performing HTTP requests
    /// inside the sandbox (requires host functions for actual I/O).
    Http,
}

impl GuestType {
    /// Returns the name of this guest type.
    ///
    /// Used for logging and debugging.
    ///
    /// # Examples
    ///
    /// ```
    /// use acton_ai::tools::sandbox::GuestType;
    ///
    /// assert_eq!(GuestType::Shell.name(), "shell");
    /// assert_eq!(GuestType::Http.name(), "http");
    /// ```
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Shell => "shell",
            Self::Http => "http",
        }
    }

    /// Returns the embedded binary for this guest type.
    ///
    /// The guest binaries are compiled by build.rs from the guests/ workspace
    /// and embedded into the final binary at compile time.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use acton_ai::tools::sandbox::GuestType;
    ///
    /// let binary = GuestType::Shell.binary();
    /// assert!(!binary.is_empty());
    /// ```
    #[must_use]
    pub fn binary(&self) -> &'static [u8] {
        match self {
            Self::Shell => GUEST_BINARIES.shell,
            Self::Http => GUEST_BINARIES.http,
        }
    }

    /// Returns an iterator over all guest types.
    ///
    /// # Examples
    ///
    /// ```
    /// use acton_ai::tools::sandbox::GuestType;
    ///
    /// let types: Vec<_> = GuestType::all().collect();
    /// assert_eq!(types.len(), 2);
    /// ```
    pub fn all() -> impl Iterator<Item = Self> {
        [Self::Shell, Self::Http].into_iter()
    }
}

impl fmt::Display for GuestType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Container for embedded guest binaries.
///
/// Holds references to the compiled guest binaries that are
/// included in the host binary at compile time.
pub struct GuestBinaries {
    /// Shell execution guest binary.
    pub shell: &'static [u8],
    /// HTTP client guest binary.
    pub http: &'static [u8],
}

/// Embedded guest binaries, included at compile time.
///
/// These binaries are compiled by build.rs from the guests/ workspace
/// and embedded into the final binary.
///
/// If the x86_64-unknown-none target isn't installed, stub binaries
/// are embedded that will fail at runtime with a clear error message.
pub static GUEST_BINARIES: GuestBinaries = GuestBinaries {
    shell: include_bytes!(concat!(env!("OUT_DIR"), "/guests/shell_guest")),
    http: include_bytes!(concat!(env!("OUT_DIR"), "/guests/http_guest")),
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn guest_type_name_shell() {
        assert_eq!(GuestType::Shell.name(), "shell");
    }

    #[test]
    fn guest_type_name_http() {
        assert_eq!(GuestType::Http.name(), "http");
    }

    #[test]
    fn guest_type_display_shell() {
        assert_eq!(format!("{}", GuestType::Shell), "shell");
    }

    #[test]
    fn guest_type_display_http() {
        assert_eq!(format!("{}", GuestType::Http), "http");
    }

    #[test]
    fn guest_type_all_returns_both() {
        let types: Vec<GuestType> = GuestType::all().collect();
        assert_eq!(types.len(), 2);
        assert!(types.contains(&GuestType::Shell));
        assert!(types.contains(&GuestType::Http));
    }

    #[test]
    fn guest_type_clone() {
        let original = GuestType::Shell;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn guest_type_copy() {
        let original = GuestType::Http;
        let copied = original; // Copy, not move
        assert_eq!(original, copied);
    }

    #[test]
    fn guest_type_equality() {
        assert_eq!(GuestType::Shell, GuestType::Shell);
        assert_eq!(GuestType::Http, GuestType::Http);
        assert_ne!(GuestType::Shell, GuestType::Http);
    }

    #[test]
    fn guest_type_hash() {
        let mut set = HashSet::new();
        set.insert(GuestType::Shell);
        set.insert(GuestType::Http);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&GuestType::Shell));
        assert!(set.contains(&GuestType::Http));
    }

    #[test]
    fn guest_type_debug() {
        assert!(format!("{:?}", GuestType::Shell).contains("Shell"));
        assert!(format!("{:?}", GuestType::Http).contains("Http"));
    }

    #[test]
    fn guest_binaries_not_empty() {
        // Placeholder bytes should exist
        assert!(!GUEST_BINARIES.shell.is_empty());
        assert!(!GUEST_BINARIES.http.is_empty());
    }

    #[test]
    fn guest_type_binary() {
        // Placeholder bytes should be returned
        let shell_bytes = GuestType::Shell.binary();
        let http_bytes = GuestType::Http.binary();
        assert!(!shell_bytes.is_empty());
        assert!(!http_bytes.is_empty());
    }
}
