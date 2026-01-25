//! Path validation for filesystem security.
//!
//! Provides `PathValidator` for restricting filesystem access to allowed
//! directories and blocking paths containing denied patterns.

use std::fmt;
use std::path::{Path, PathBuf};

/// Error returned when path validation fails.
///
/// This error type provides detailed information about why a path
/// was rejected, enabling users to understand and fix the issue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathValidationError {
    /// Path could not be canonicalized (doesn't exist or permission denied).
    CanonicalizeError {
        /// The path that failed to canonicalize.
        path: PathBuf,
        /// The underlying error reason.
        reason: String,
    },
    /// Path is outside all allowed root directories.
    OutsideAllowedRoots {
        /// The path that was rejected.
        path: PathBuf,
        /// The list of allowed root directories.
        allowed_roots: Vec<PathBuf>,
    },
    /// Path contains a denied pattern.
    DeniedPattern {
        /// The path that was rejected.
        path: PathBuf,
        /// The pattern that matched.
        pattern: String,
    },
}

impl fmt::Display for PathValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CanonicalizeError { path, reason } => {
                write!(
                    f,
                    "cannot resolve path '{}': {}; verify the path exists and is accessible",
                    path.display(),
                    reason
                )
            }
            Self::OutsideAllowedRoots {
                path,
                allowed_roots,
            } => {
                let roots: Vec<String> = allowed_roots
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect();
                write!(
                    f,
                    "path '{}' is outside allowed directories [{}]; \
                     operations are restricted to these locations",
                    path.display(),
                    roots.join(", ")
                )
            }
            Self::DeniedPattern { path, pattern } => {
                write!(
                    f,
                    "path '{}' contains denied pattern '{}'; \
                     access to paths with this pattern is blocked for security",
                    path.display(),
                    pattern
                )
            }
        }
    }
}

impl std::error::Error for PathValidationError {}

/// Validates paths against an allowlist for filesystem operations.
///
/// `PathValidator` provides security controls for filesystem access by:
/// - Restricting operations to allowed root directories
/// - Blocking paths containing denied patterns (like `..` or `.env`)
/// - Canonicalizing paths to prevent symlink attacks
///
/// # Example
///
/// ```rust,ignore
/// use std::path::{Path, PathBuf};
/// use acton_ai::tools::security::PathValidator;
///
/// let validator = PathValidator::new()
///     .with_allowed_root(PathBuf::from("/home/user/project"));
///
/// // This succeeds (within allowed root)
/// let canonical = validator.validate(Path::new("/home/user/project/src/main.rs"));
///
/// // This fails (path traversal attempt)
/// let result = validator.validate(Path::new("/home/user/project/../secrets.txt"));
/// assert!(result.is_err());
/// ```
#[derive(Debug, Clone)]
pub struct PathValidator {
    /// Directories where filesystem operations are permitted.
    allowed_roots: Vec<PathBuf>,
    /// Patterns that are blocked from appearing in paths.
    denied_patterns: Vec<String>,
}

impl PathValidator {
    /// Creates a new `PathValidator` with default settings.
    ///
    /// Default settings:
    /// - Allowed roots: current working directory and system temp directory
    /// - Denied patterns: `..`, `.git`, `.env`
    #[must_use]
    pub fn new() -> Self {
        let mut allowed_roots = Vec::new();

        // Add current working directory
        if let Ok(cwd) = std::env::current_dir() {
            allowed_roots.push(cwd);
        }

        // Add system temp directory (for legitimate temp file operations)
        allowed_roots.push(std::env::temp_dir());

        Self {
            allowed_roots,
            denied_patterns: vec!["..".to_string(), ".git".to_string(), ".env".to_string()],
        }
    }

    /// Adds an allowed root directory.
    ///
    /// Paths must be within at least one allowed root to pass validation.
    #[must_use]
    pub fn with_allowed_root(mut self, root: PathBuf) -> Self {
        self.allowed_roots.push(root);
        self
    }

    /// Adds a denied pattern.
    ///
    /// Paths containing this pattern (anywhere in the path string) will be rejected.
    #[must_use]
    pub fn with_denied_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.denied_patterns.push(pattern.into());
        self
    }

    /// Clears the default denied patterns.
    ///
    /// Use this followed by `with_denied_pattern` to customize the deny list.
    #[must_use]
    pub fn clear_denied_patterns(mut self) -> Self {
        self.denied_patterns.clear();
        self
    }

    /// Clears the default allowed roots.
    ///
    /// Use this followed by `with_allowed_root` to customize allowed directories.
    #[must_use]
    pub fn clear_allowed_roots(mut self) -> Self {
        self.allowed_roots.clear();
        self
    }

    /// Returns a reference to the allowed roots.
    #[must_use]
    pub fn allowed_roots(&self) -> &[PathBuf] {
        &self.allowed_roots
    }

    /// Returns a reference to the denied patterns.
    #[must_use]
    pub fn denied_patterns(&self) -> &[String] {
        &self.denied_patterns
    }

    /// Validates a path against the configured rules.
    ///
    /// Validation steps:
    /// 1. Check for denied patterns in the original path string
    /// 2. Canonicalize the path (resolves symlinks, normalizes)
    /// 3. Verify the canonical path is within an allowed root
    ///
    /// # Returns
    ///
    /// The canonicalized path on success, or an error describing why validation failed.
    ///
    /// # Errors
    ///
    /// Returns `PathValidationError` if:
    /// - The path contains a denied pattern
    /// - The path cannot be canonicalized (doesn't exist, permission denied)
    /// - The canonical path is outside all allowed roots
    pub fn validate(&self, path: &Path) -> Result<PathBuf, PathValidationError> {
        // 1. Check denied patterns in the original path string FIRST
        // This catches path traversal attempts like "../../../etc/passwd"
        // before we try to canonicalize
        let path_str = path.to_string_lossy();
        for pattern in &self.denied_patterns {
            if path_str.contains(pattern) {
                return Err(PathValidationError::DeniedPattern {
                    path: path.to_path_buf(),
                    pattern: pattern.clone(),
                });
            }
        }

        // 2. Canonicalize the path (resolves symlinks, normalizes)
        let canonical =
            path.canonicalize()
                .map_err(|e| PathValidationError::CanonicalizeError {
                    path: path.to_path_buf(),
                    reason: e.to_string(),
                })?;

        // 3. Check if canonical path is within an allowed root
        let allowed = self.allowed_roots.iter().any(|root| {
            // Canonicalize the root too for proper comparison
            root.canonicalize()
                .map(|canonical_root| canonical.starts_with(&canonical_root))
                .unwrap_or(false)
        });

        if !allowed {
            return Err(PathValidationError::OutsideAllowedRoots {
                path: path.to_path_buf(),
                allowed_roots: self.allowed_roots.clone(),
            });
        }

        Ok(canonical)
    }

    /// Validates a path for directory operations.
    ///
    /// Same as `validate`, but performs an additional check that the path
    /// is actually a directory after canonicalization.
    ///
    /// # Errors
    ///
    /// Returns error if validation fails or the path is not a directory.
    pub fn validate_directory(&self, path: &Path) -> Result<PathBuf, PathValidationError> {
        let canonical = self.validate(path)?;

        // The path exists (canonicalize succeeded), but is it a directory?
        if !canonical.is_dir() {
            return Err(PathValidationError::CanonicalizeError {
                path: path.to_path_buf(),
                reason: "path is not a directory".to_string(),
            });
        }

        Ok(canonical)
    }

    /// Validates a path for file operations.
    ///
    /// Same as `validate`, but performs an additional check that the path
    /// is actually a file after canonicalization.
    ///
    /// # Errors
    ///
    /// Returns error if validation fails or the path is not a file.
    pub fn validate_file(&self, path: &Path) -> Result<PathBuf, PathValidationError> {
        let canonical = self.validate(path)?;

        // The path exists (canonicalize succeeded), but is it a file?
        if !canonical.is_file() {
            return Err(PathValidationError::CanonicalizeError {
                path: path.to_path_buf(),
                reason: "path is not a file".to_string(),
            });
        }

        Ok(canonical)
    }

    /// Validates a parent directory for file creation.
    ///
    /// For write operations where the file may not exist yet, this validates
    /// that the parent directory exists and is within allowed roots.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - The path contains denied patterns
    /// - The parent directory cannot be resolved
    /// - The parent is outside allowed roots
    pub fn validate_parent(&self, path: &Path) -> Result<PathBuf, PathValidationError> {
        // Check denied patterns first
        let path_str = path.to_string_lossy();
        for pattern in &self.denied_patterns {
            if path_str.contains(pattern) {
                return Err(PathValidationError::DeniedPattern {
                    path: path.to_path_buf(),
                    pattern: pattern.clone(),
                });
            }
        }

        // Get parent directory
        let parent = path.parent().ok_or_else(|| PathValidationError::CanonicalizeError {
            path: path.to_path_buf(),
            reason: "path has no parent directory".to_string(),
        })?;

        // If parent exists, validate it
        if parent.exists() {
            let canonical_parent = self.validate_directory(parent)?;
            // Return the intended file path with canonical parent
            Ok(canonical_parent.join(path.file_name().unwrap_or_default()))
        } else {
            // Parent doesn't exist - check if any ancestor is in allowed roots
            let mut ancestor = parent;
            while let Some(next_parent) = ancestor.parent() {
                if next_parent.exists() {
                    let canonical_ancestor =
                        next_parent
                            .canonicalize()
                            .map_err(|e| PathValidationError::CanonicalizeError {
                                path: next_parent.to_path_buf(),
                                reason: e.to_string(),
                            })?;

                    let allowed = self.allowed_roots.iter().any(|root| {
                        root.canonicalize()
                            .map(|r| canonical_ancestor.starts_with(&r))
                            .unwrap_or(false)
                    });

                    if !allowed {
                        return Err(PathValidationError::OutsideAllowedRoots {
                            path: path.to_path_buf(),
                            allowed_roots: self.allowed_roots.clone(),
                        });
                    }

                    // Parent would be created under an allowed ancestor
                    return Ok(path.to_path_buf());
                }
                ancestor = next_parent;
            }

            Err(PathValidationError::CanonicalizeError {
                path: path.to_path_buf(),
                reason: "no existing ancestor directory found".to_string(),
            })
        }
    }
}

impl Default for PathValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn new_uses_current_dir_as_allowed_root() {
        let validator = PathValidator::new();
        let cwd = std::env::current_dir().unwrap();
        assert!(validator.allowed_roots().iter().any(|r| r == &cwd));
    }

    #[test]
    fn new_has_default_denied_patterns() {
        let validator = PathValidator::new();
        assert!(validator.denied_patterns().contains(&"..".to_string()));
        assert!(validator.denied_patterns().contains(&".git".to_string()));
        assert!(validator.denied_patterns().contains(&".env".to_string()));
    }

    #[test]
    fn with_allowed_root_adds_root() {
        let validator = PathValidator::new().with_allowed_root(PathBuf::from("/custom/path"));

        assert!(validator
            .allowed_roots()
            .iter()
            .any(|r| r == Path::new("/custom/path")));
    }

    #[test]
    fn with_denied_pattern_adds_pattern() {
        let validator = PathValidator::new().with_denied_pattern("secrets");

        assert!(validator
            .denied_patterns()
            .contains(&"secrets".to_string()));
    }

    #[test]
    fn validate_succeeds_within_allowed_root() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "content").unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate(&file_path);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_fails_outside_allowed_roots() {
        let allowed_dir = TempDir::new().unwrap();
        let outside_dir = TempDir::new().unwrap();
        let file_path = outside_dir.path().join("test.txt");
        fs::write(&file_path, "content").unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(allowed_dir.path().to_path_buf());

        let result = validator.validate(&file_path);
        assert!(matches!(
            result,
            Err(PathValidationError::OutsideAllowedRoots { .. })
        ));
    }

    #[test]
    fn validate_catches_path_traversal() {
        let validator = PathValidator::new();
        let result = validator.validate(Path::new("/some/path/../../../etc/passwd"));

        assert!(
            matches!(result, Err(PathValidationError::DeniedPattern { pattern, .. }) if pattern == "..")
        );
    }

    #[test]
    fn validate_catches_git_directory() {
        let dir = TempDir::new().unwrap();
        let git_dir = dir.path().join(".git");
        fs::create_dir(&git_dir).unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate(&git_dir);
        assert!(
            matches!(result, Err(PathValidationError::DeniedPattern { pattern, .. }) if pattern == ".git")
        );
    }

    #[test]
    fn validate_catches_env_file() {
        let dir = TempDir::new().unwrap();
        let env_file = dir.path().join(".env");
        fs::write(&env_file, "SECRET=value").unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate(&env_file);
        assert!(
            matches!(result, Err(PathValidationError::DeniedPattern { pattern, .. }) if pattern == ".env")
        );
    }

    #[test]
    fn validate_fails_for_nonexistent_path() {
        let dir = TempDir::new().unwrap();
        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate(&dir.path().join("nonexistent.txt"));
        assert!(matches!(
            result,
            Err(PathValidationError::CanonicalizeError { .. })
        ));
    }

    #[test]
    fn validate_file_succeeds_for_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "content").unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate_file(&file_path);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_file_fails_for_directory() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate_file(&subdir);
        assert!(
            matches!(result, Err(PathValidationError::CanonicalizeError { reason, .. }) if reason.contains("not a file"))
        );
    }

    #[test]
    fn validate_directory_succeeds_for_directory() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate_directory(&subdir);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_directory_fails_for_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "content").unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate_directory(&file_path);
        assert!(
            matches!(result, Err(PathValidationError::CanonicalizeError { reason, .. }) if reason.contains("not a directory"))
        );
    }

    #[test]
    fn validate_parent_succeeds_for_existing_parent() {
        let dir = TempDir::new().unwrap();
        let new_file = dir.path().join("new_file.txt");

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate_parent(&new_file);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_parent_succeeds_for_nested_new_path() {
        let dir = TempDir::new().unwrap();
        let nested_file = dir.path().join("new_dir").join("nested").join("file.txt");

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate_parent(&nested_file);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_parent_fails_outside_allowed_roots() {
        let allowed_dir = TempDir::new().unwrap();
        let outside_dir = TempDir::new().unwrap();
        let new_file = outside_dir.path().join("new_file.txt");

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(allowed_dir.path().to_path_buf());

        let result = validator.validate_parent(&new_file);
        assert!(matches!(
            result,
            Err(PathValidationError::OutsideAllowedRoots { .. })
        ));
    }

    #[test]
    fn clear_denied_patterns_removes_all() {
        let validator = PathValidator::new().clear_denied_patterns();

        assert!(validator.denied_patterns().is_empty());
    }

    #[test]
    fn clear_allowed_roots_removes_all() {
        let validator = PathValidator::new().clear_allowed_roots();

        assert!(validator.allowed_roots().is_empty());
    }

    #[test]
    fn error_display_canonicalize() {
        let err = PathValidationError::CanonicalizeError {
            path: PathBuf::from("/some/path"),
            reason: "No such file or directory".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/some/path"));
        assert!(msg.contains("No such file"));
        assert!(msg.contains("verify the path exists"));
    }

    #[test]
    fn error_display_outside_roots() {
        let err = PathValidationError::OutsideAllowedRoots {
            path: PathBuf::from("/outside/path"),
            allowed_roots: vec![PathBuf::from("/allowed/root")],
        };
        let msg = err.to_string();
        assert!(msg.contains("/outside/path"));
        assert!(msg.contains("/allowed/root"));
        assert!(msg.contains("outside allowed directories"));
    }

    #[test]
    fn error_display_denied_pattern() {
        let err = PathValidationError::DeniedPattern {
            path: PathBuf::from("/some/.git/config"),
            pattern: ".git".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains(".git"));
        assert!(msg.contains("denied pattern"));
        assert!(msg.contains("blocked for security"));
    }

    #[test]
    fn validator_is_clone() {
        let validator = PathValidator::new();
        let cloned = validator.clone();
        assert_eq!(validator.allowed_roots(), cloned.allowed_roots());
    }

    #[test]
    fn validator_default_matches_new() {
        let new_validator = PathValidator::new();
        let default_validator = PathValidator::default();
        assert_eq!(
            new_validator.allowed_roots(),
            default_validator.allowed_roots()
        );
        assert_eq!(
            new_validator.denied_patterns(),
            default_validator.denied_patterns()
        );
    }

    #[cfg(unix)]
    #[test]
    fn validate_resolves_symlinks_within_allowed() {
        let dir = TempDir::new().unwrap();
        let real_file = dir.path().join("real.txt");
        let link = dir.path().join("link.txt");
        fs::write(&real_file, "content").unwrap();
        std::os::unix::fs::symlink(&real_file, &link).unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(dir.path().to_path_buf());

        let result = validator.validate(&link);
        assert!(result.is_ok());
        // Canonical path should be the real file
        assert_eq!(result.unwrap(), real_file.canonicalize().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn validate_blocks_symlink_escaping_allowed_root() {
        let allowed_dir = TempDir::new().unwrap();
        let outside_dir = TempDir::new().unwrap();
        let outside_file = outside_dir.path().join("secret.txt");
        fs::write(&outside_file, "secret").unwrap();

        let escape_link = allowed_dir.path().join("escape.txt");
        std::os::unix::fs::symlink(&outside_file, &escape_link).unwrap();

        let validator = PathValidator::new()
            .clear_allowed_roots()
            .with_allowed_root(allowed_dir.path().to_path_buf());

        let result = validator.validate(&escape_link);
        assert!(matches!(
            result,
            Err(PathValidationError::OutsideAllowedRoots { .. })
        ));
    }
}
