//! Skill types for the skills module.
//!
//! Defines types for representing loaded skills and skill metadata.

use std::fmt;
use std::path::PathBuf;

/// Information about a skill (without full content).
///
/// Used for listing and querying skills without loading full content.
#[derive(Debug, Clone)]
pub struct SkillInfo {
    /// Unique name of the skill (from filename or frontmatter)
    pub name: String,
    /// Short description of the skill
    pub description: String,
    /// Path to the skill file
    pub path: PathBuf,
    /// Optional tags/categories for the skill
    pub tags: Vec<String>,
}

/// A fully loaded skill with content.
///
/// Contains all skill information including the full markdown instructions.
#[derive(Debug, Clone)]
pub struct LoadedSkill {
    /// Skill metadata
    pub info: SkillInfo,
    /// Full markdown content (instructions)
    pub content: String,
    /// Optional trigger patterns that activate this skill
    pub triggers: Vec<String>,
    /// Whether this skill is enabled by default
    pub enabled_by_default: bool,
}

impl LoadedSkill {
    /// Returns the skill name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Returns the skill description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.info.description
    }

    /// Returns the full instructions content.
    #[must_use]
    pub fn instructions(&self) -> &str {
        &self.content
    }

    /// Returns true if this skill has trigger patterns.
    #[must_use]
    pub fn has_triggers(&self) -> bool {
        !self.triggers.is_empty()
    }

    /// Checks if the given text matches any of the skill's triggers.
    #[must_use]
    pub fn matches_trigger(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        self.triggers
            .iter()
            .any(|t| text_lower.contains(&t.to_lowercase()))
    }
}

/// Errors that can occur when working with skills.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillsError {
    /// Failed to load skill from path
    LoadFailed {
        /// Path that failed to load
        path: PathBuf,
        /// Reason for failure
        reason: String,
    },
    /// Skill not found
    NotFound {
        /// Name of the skill that was not found
        name: String,
    },
    /// Invalid skill format
    InvalidFormat {
        /// Path to the invalid skill
        path: PathBuf,
        /// What was wrong with the format
        reason: String,
    },
    /// Path does not exist
    PathNotFound {
        /// Path that was not found
        path: PathBuf,
    },
}

impl fmt::Display for SkillsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadFailed { path, reason } => {
                write!(f, "failed to load skill from {}: {}", path.display(), reason)
            }
            Self::NotFound { name } => {
                write!(f, "skill '{}' not found", name)
            }
            Self::InvalidFormat { path, reason } => {
                write!(
                    f,
                    "invalid skill format in {}: {}",
                    path.display(),
                    reason
                )
            }
            Self::PathNotFound { path } => {
                write!(f, "skill path not found: {}", path.display())
            }
        }
    }
}

impl std::error::Error for SkillsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_info_creation() {
        let info = SkillInfo {
            name: "test-skill".to_string(),
            description: "A test skill".to_string(),
            path: PathBuf::from("/path/to/skill.md"),
            tags: vec!["test".to_string(), "example".to_string()],
        };

        assert_eq!(info.name, "test-skill");
        assert_eq!(info.description, "A test skill");
        assert_eq!(info.tags.len(), 2);
    }

    #[test]
    fn loaded_skill_accessors() {
        let skill = LoadedSkill {
            info: SkillInfo {
                name: "code-review".to_string(),
                description: "Review code changes".to_string(),
                path: PathBuf::from("/skills/code-review.md"),
                tags: vec!["code".to_string()],
            },
            content: "# Code Review Instructions\n\nReview the code...".to_string(),
            triggers: vec!["review".to_string(), "code review".to_string()],
            enabled_by_default: true,
        };

        assert_eq!(skill.name(), "code-review");
        assert_eq!(skill.description(), "Review code changes");
        assert!(skill.instructions().contains("Code Review"));
        assert!(skill.has_triggers());
        assert!(skill.matches_trigger("please review this code"));
        assert!(!skill.matches_trigger("unrelated text"));
    }

    #[test]
    fn skill_without_triggers() {
        let skill = LoadedSkill {
            info: SkillInfo {
                name: "basic".to_string(),
                description: "Basic skill".to_string(),
                path: PathBuf::from("/skills/basic.md"),
                tags: vec![],
            },
            content: "Instructions".to_string(),
            triggers: vec![],
            enabled_by_default: false,
        };

        assert!(!skill.has_triggers());
        assert!(!skill.matches_trigger("anything"));
    }

    #[test]
    fn skills_error_display() {
        let err = SkillsError::LoadFailed {
            path: PathBuf::from("/path/to/skill.md"),
            reason: "file not readable".to_string(),
        };
        assert!(err.to_string().contains("failed to load"));
        assert!(err.to_string().contains("/path/to/skill.md"));

        let err = SkillsError::NotFound {
            name: "missing-skill".to_string(),
        };
        assert!(err.to_string().contains("not found"));
        assert!(err.to_string().contains("missing-skill"));

        let err = SkillsError::InvalidFormat {
            path: PathBuf::from("/bad.md"),
            reason: "missing frontmatter".to_string(),
        };
        assert!(err.to_string().contains("invalid skill format"));

        let err = SkillsError::PathNotFound {
            path: PathBuf::from("/nonexistent"),
        };
        assert!(err.to_string().contains("path not found"));
    }
}
