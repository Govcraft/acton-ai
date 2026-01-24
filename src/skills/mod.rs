//! Agent Skills module.
//!
//! This module provides integration with the `agent-skills` crate for loading
//! and managing reusable agent behaviors defined via YAML frontmatter + markdown.
//!
//! Skills are feature-gated behind the `agent-skills` feature flag.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use acton_ai::skills::SkillRegistry;
//! use std::path::Path;
//!
//! // Load skills from a directory
//! let registry = SkillRegistry::from_paths(&[Path::new("./skills")]).await?;
//!
//! // List available skills
//! for skill in registry.list() {
//!     println!("Skill: {} - {}", skill.name, skill.description);
//! }
//!
//! // Get a skill by name
//! if let Some(skill) = registry.get("code-review") {
//!     println!("Instructions: {}", skill.instructions());
//! }
//! ```

mod registry;
mod types;

pub use registry::SkillRegistry;
pub use types::{LoadedSkill, SkillInfo, SkillsError};
