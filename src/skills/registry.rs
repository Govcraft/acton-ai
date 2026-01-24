//! Skill registry for loading and managing skills.
//!
//! Wraps the `agent-skills` crate to provide skill loading and querying.

use super::types::{LoadedSkill, SkillInfo, SkillsError};
use acton_reactive::prelude::tokio;
use agent_skills::Skill;
use std::collections::HashMap;
use std::path::Path;

/// Registry of loaded skills.
///
/// Provides methods to load skills from paths, list available skills,
/// and retrieve skills by name or trigger patterns.
#[derive(Debug, Default)]
pub struct SkillRegistry {
    /// Skills indexed by name
    skills: HashMap<String, LoadedSkill>,
}

impl SkillRegistry {
    /// Creates a new empty skill registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
        }
    }

    /// Loads skills from the given paths.
    ///
    /// Paths can be directories (scanned recursively for .md files) or
    /// individual .md files.
    ///
    /// # Arguments
    ///
    /// * `paths` - Paths to skill files or directories
    ///
    /// # Returns
    ///
    /// A registry containing all successfully loaded skills.
    ///
    /// # Errors
    ///
    /// Returns an error if any path doesn't exist. Individual skill load
    /// failures are logged but don't prevent other skills from loading.
    pub async fn from_paths(paths: &[&Path]) -> Result<Self, SkillsError> {
        let mut registry = Self::new();

        for path in paths {
            if !path.exists() {
                return Err(SkillsError::PathNotFound {
                    path: path.to_path_buf(),
                });
            }

            if path.is_file() {
                if let Err(e) = registry.load_skill_file(path).await {
                    tracing::warn!(path = %path.display(), error = %e, "Failed to load skill");
                }
            } else if path.is_dir() {
                registry.load_skill_directory(path).await?;
            }
        }

        Ok(registry)
    }

    /// Loads a single skill file.
    async fn load_skill_file(&mut self, path: &Path) -> Result<(), SkillsError> {
        // Read the file content
        let content = tokio::fs::read_to_string(path).await.map_err(|e| {
            SkillsError::LoadFailed {
                path: path.to_path_buf(),
                reason: e.to_string(),
            }
        })?;

        // Parse using agent-skills crate
        let skill = Skill::parse(&content).map_err(|e| SkillsError::InvalidFormat {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })?;

        // Convert to our LoadedSkill type
        let loaded = convert_skill(skill, path);
        self.skills.insert(loaded.info.name.clone(), loaded);

        Ok(())
    }

    /// Recursively loads skills from a directory.
    async fn load_skill_directory(&mut self, dir: &Path) -> Result<(), SkillsError> {
        let mut entries = tokio::fs::read_dir(dir).await.map_err(|e| {
            SkillsError::LoadFailed {
                path: dir.to_path_buf(),
                reason: e.to_string(),
            }
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            SkillsError::LoadFailed {
                path: dir.to_path_buf(),
                reason: e.to_string(),
            }
        })? {
            let path = entry.path();

            if path.is_dir() {
                // Skip hidden directories
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('.'))
                {
                    continue;
                }
                // Use Box::pin for recursive async call
                Box::pin(self.load_skill_directory(&path)).await?;
            } else if path.extension().is_some_and(|ext| ext == "md") {
                if let Err(e) = self.load_skill_file(&path).await {
                    tracing::warn!(path = %path.display(), error = %e, "Failed to load skill");
                }
            }
        }

        Ok(())
    }

    /// Returns the number of loaded skills.
    #[must_use]
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Returns true if no skills are loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Lists all loaded skills (metadata only).
    #[must_use]
    pub fn list(&self) -> Vec<SkillInfo> {
        self.skills.values().map(|s| s.info.clone()).collect()
    }

    /// Gets a skill by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&LoadedSkill> {
        self.skills.get(name)
    }

    /// Gets skills that match the given trigger text.
    #[must_use]
    pub fn find_by_trigger(&self, text: &str) -> Vec<&LoadedSkill> {
        self.skills
            .values()
            .filter(|s| s.matches_trigger(text))
            .collect()
    }

    /// Gets all skills that are enabled by default.
    #[must_use]
    pub fn default_skills(&self) -> Vec<&LoadedSkill> {
        self.skills
            .values()
            .filter(|s| s.enabled_by_default)
            .collect()
    }

    /// Returns an iterator over all loaded skills.
    pub fn iter(&self) -> impl Iterator<Item = &LoadedSkill> {
        self.skills.values()
    }

    /// Adds a skill to the registry.
    pub fn add(&mut self, skill: LoadedSkill) {
        self.skills.insert(skill.info.name.clone(), skill);
    }

    /// Removes a skill from the registry by name.
    pub fn remove(&mut self, name: &str) -> Option<LoadedSkill> {
        self.skills.remove(name)
    }
}

/// Converts an agent_skills::Skill to our LoadedSkill type.
fn convert_skill(skill: Skill, path: &Path) -> LoadedSkill {
    let frontmatter = skill.frontmatter();

    // Get name from frontmatter
    let name = frontmatter.name().as_str().to_string();

    // Get description from frontmatter
    let description = frontmatter.description().as_str().to_string();

    // Extract triggers from metadata if present
    let triggers = frontmatter
        .metadata()
        .and_then(|m| m.get("triggers"))
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    // Check if enabled by default from metadata
    let enabled_by_default = frontmatter
        .metadata()
        .and_then(|m| m.get("enabled"))
        .is_some_and(|v| v == "true");

    // Extract tags from metadata
    let tags = frontmatter
        .metadata()
        .and_then(|m| m.get("tags"))
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    LoadedSkill {
        info: SkillInfo {
            name,
            description,
            path: path.to_path_buf(),
            tags,
        },
        content: skill.body().to_string(),
        triggers,
        enabled_by_default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    async fn create_test_skill(dir: &Path, name: &str, content: &str) {
        let path = dir.join(format!("{name}.md"));
        let mut file = std::fs::File::create(&path).unwrap();
        write!(file, "{content}").unwrap();
    }

    #[tokio::test]
    async fn load_skill_from_file() {
        let dir = TempDir::new().unwrap();
        let skill_content = r#"---
name: test-skill
description: A test skill
metadata:
  triggers: test, example
  enabled: "true"
  tags: testing, example
---

# Test Skill

This is the skill content.
"#;

        create_test_skill(dir.path(), "test-skill", skill_content).await;

        let registry = SkillRegistry::from_paths(&[dir.path()]).await.unwrap();

        assert_eq!(registry.len(), 1);

        let skill = registry.get("test-skill").unwrap();
        assert_eq!(skill.name(), "test-skill");
        assert_eq!(skill.description(), "A test skill");
        assert!(skill.instructions().contains("Test Skill"));
        assert!(skill.has_triggers());
        assert!(skill.matches_trigger("run the test"));
        assert!(skill.enabled_by_default);
    }

    #[tokio::test]
    async fn load_skills_from_directory() {
        let dir = TempDir::new().unwrap();

        let skill1 = r#"---
name: skill-one
description: First skill
---
Content one
"#;

        let skill2 = r#"---
name: skill-two
description: Second skill
---
Content two
"#;

        create_test_skill(dir.path(), "skill-one", skill1).await;
        create_test_skill(dir.path(), "skill-two", skill2).await;

        let registry = SkillRegistry::from_paths(&[dir.path()]).await.unwrap();

        assert_eq!(registry.len(), 2);
        assert!(registry.get("skill-one").is_some());
        assert!(registry.get("skill-two").is_some());
    }

    #[tokio::test]
    async fn find_skills_by_trigger() {
        let dir = TempDir::new().unwrap();

        let skill_with_trigger = r#"---
name: triggered-skill
description: Has triggers
metadata:
  triggers: activate me
---
Content
"#;

        let skill_without = r#"---
name: no-trigger
description: No triggers
---
Content
"#;

        create_test_skill(dir.path(), "triggered", skill_with_trigger).await;
        create_test_skill(dir.path(), "no-trigger", skill_without).await;

        let registry = SkillRegistry::from_paths(&[dir.path()]).await.unwrap();

        let found = registry.find_by_trigger("please activate me now");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name(), "triggered-skill");
    }

    #[tokio::test]
    async fn path_not_found_error() {
        let result = SkillRegistry::from_paths(&[Path::new("/nonexistent/path")]).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SkillsError::PathNotFound { .. }));
    }

    #[tokio::test]
    async fn empty_registry() {
        let registry = SkillRegistry::new();

        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.list().is_empty());
        assert!(registry.get("anything").is_none());
    }

    #[test]
    fn add_and_remove_skill() {
        let mut registry = SkillRegistry::new();

        let skill = LoadedSkill {
            info: SkillInfo {
                name: "manual-skill".to_string(),
                description: "Added manually".to_string(),
                path: std::path::PathBuf::from("/fake/path.md"),
                tags: vec![],
            },
            content: "Manual content".to_string(),
            triggers: vec![],
            enabled_by_default: false,
        };

        registry.add(skill);
        assert_eq!(registry.len(), 1);
        assert!(registry.get("manual-skill").is_some());

        let removed = registry.remove("manual-skill");
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }
}
