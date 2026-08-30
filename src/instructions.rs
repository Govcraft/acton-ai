//! Discovery and layering of cross-vendor `AGENTS.md` instruction files.
//!
//! Project instructions are loaded from the workspace root down to the
//! session's working directory. A user-level `~/.agents/AGENTS.md`, when
//! present, is appended last and therefore has the highest precedence.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

const INSTRUCTIONS_FILE: &str = "AGENTS.md";

/// The authority that supplied an instruction layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstructionScope {
    /// Instructions stored in the active project checkout.
    Project,
    /// Instructions owned by the user and applied over project instructions.
    User,
}

/// One discovered instruction document and its place in the merged stack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionLayer {
    /// Whether the document came from the project or the user.
    pub scope: InstructionScope,
    /// Absolute path from which the document was loaded.
    pub path: PathBuf,
    /// The document's Markdown content.
    pub content: String,
    /// Zero-based precedence; larger values override smaller values.
    pub precedence: usize,
}

/// The structured result of instruction discovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentInstructions {
    workspace_root: PathBuf,
    working_directory: PathBuf,
    layers: Vec<InstructionLayer>,
}

impl AgentInstructions {
    /// Discovers instructions for `working_directory`.
    ///
    /// The nearest ancestor containing a `.git` file or directory is the
    /// workspace root. If no checkout marker exists, the working directory is
    /// its own root. The optional user layer is read from
    /// `~/.agents/AGENTS.md`.
    ///
    /// # Errors
    ///
    /// Returns [`InstructionsError`] when a path cannot be resolved or a
    /// discovered instruction file cannot be read.
    pub fn discover(working_directory: impl AsRef<Path>) -> Result<Self, InstructionsError> {
        let working_directory = canonical_directory(working_directory.as_ref())?;
        let workspace_root = find_workspace_root(&working_directory);
        let user_file = dirs::home_dir().map(|home| home.join(".agents").join(INSTRUCTIONS_FILE));
        Self::discover_with_root(&working_directory, &workspace_root, user_file.as_deref())
    }

    /// Discovers instructions within an explicit workspace boundary.
    ///
    /// This form is intended for hosts that already assign a workspace to a
    /// session. `user_file` may be omitted to disable user instructions.
    /// Only files named exactly `AGENTS.md` are considered.
    ///
    /// # Errors
    ///
    /// Returns [`InstructionsError`] if either directory cannot be resolved,
    /// the working directory is outside the workspace, or a discovered file
    /// cannot be read.
    pub fn discover_with_root(
        working_directory: impl AsRef<Path>,
        workspace_root: impl AsRef<Path>,
        user_file: Option<&Path>,
    ) -> Result<Self, InstructionsError> {
        let working_directory = canonical_directory(working_directory.as_ref())?;
        let workspace_root = canonical_directory(workspace_root.as_ref())?;
        if !working_directory.starts_with(&workspace_root) {
            return Err(InstructionsError::OutsideWorkspace {
                working_directory,
                workspace_root,
            });
        }

        let mut layers = Vec::new();
        for directory in directories_from_root(&workspace_root, &working_directory) {
            load_if_present(
                &directory.join(INSTRUCTIONS_FILE),
                InstructionScope::Project,
                &mut layers,
            )?;
        }
        if let Some(path) = user_file {
            load_if_present(path, InstructionScope::User, &mut layers)?;
        }

        Ok(Self {
            workspace_root,
            working_directory,
            layers,
        })
    }

    /// Returns the resolved workspace boundary.
    #[must_use]
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// Returns the resolved session working directory.
    #[must_use]
    pub fn working_directory(&self) -> &Path {
        &self.working_directory
    }

    /// Returns the instruction layers from lowest to highest precedence.
    #[must_use]
    pub fn layers(&self) -> &[InstructionLayer] {
        &self.layers
    }

    /// Renders a context fragment suitable for insertion at turn start.
    ///
    /// Later documents have higher precedence. An empty discovery result
    /// renders as an empty string.
    #[must_use]
    pub fn context_fragment(&self) -> String {
        self.layers
            .iter()
            .map(|layer| {
                format!(
                    "## AGENTS.md instructions from {}\n\n{}",
                    layer.path.display(),
                    layer.content.trim()
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Returns true when no instruction documents were discovered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

/// Errors produced while discovering project instructions.
#[derive(Debug)]
#[non_exhaustive]
pub enum InstructionsError {
    /// A directory could not be resolved to an absolute path.
    ResolvePath {
        /// The path that could not be resolved.
        path: PathBuf,
        /// The underlying filesystem error.
        source: std::io::Error,
    },
    /// The requested working directory is outside its workspace boundary.
    OutsideWorkspace {
        /// The resolved working directory.
        working_directory: PathBuf,
        /// The resolved workspace root.
        workspace_root: PathBuf,
    },
    /// A discovered instruction document could not be read.
    ReadFile {
        /// The document that could not be read.
        path: PathBuf,
        /// The underlying filesystem error.
        source: std::io::Error,
    },
}

impl fmt::Display for InstructionsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResolvePath { path, source } => {
                write!(formatter, "failed to resolve {}: {source}", path.display())
            }
            Self::OutsideWorkspace {
                working_directory,
                workspace_root,
            } => write!(
                formatter,
                "working directory {} is outside workspace {}",
                working_directory.display(),
                workspace_root.display()
            ),
            Self::ReadFile { path, source } => {
                write!(formatter, "failed to read {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for InstructionsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ResolvePath { source, .. } | Self::ReadFile { source, .. } => Some(source),
            Self::OutsideWorkspace { .. } => None,
        }
    }
}

fn canonical_directory(path: &Path) -> Result<PathBuf, InstructionsError> {
    path.canonicalize()
        .map_err(|source| InstructionsError::ResolvePath {
            path: path.to_path_buf(),
            source,
        })
}

fn find_workspace_root(working_directory: &Path) -> PathBuf {
    working_directory
        .ancestors()
        .find(|directory| directory.join(".git").exists())
        .unwrap_or(working_directory)
        .to_path_buf()
}

fn directories_from_root(root: &Path, working_directory: &Path) -> Vec<PathBuf> {
    let mut directories = working_directory
        .ancestors()
        .take_while(|directory| directory.starts_with(root))
        .map(Path::to_path_buf)
        .collect::<Vec<_>>();
    directories.reverse();
    directories
}

fn load_if_present(
    path: &Path,
    scope: InstructionScope,
    layers: &mut Vec<InstructionLayer>,
) -> Result<(), InstructionsError> {
    if !path.is_file() {
        return Ok(());
    }
    let content = std::fs::read_to_string(path).map_err(|source| InstructionsError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;
    layers.push(InstructionLayer {
        scope,
        path: path.to_path_buf(),
        content,
        precedence: layers.len(),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn nested_layers_are_ordered_nearest_last_and_user_over_project() {
        let fixture = TempDir::new().unwrap();
        let root = fixture.path().join("checkout");
        let package = root.join("packages").join("api");
        let source = package.join("src");
        let user_file = fixture.path().join("user").join(INSTRUCTIONS_FILE);
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(user_file.parent().unwrap()).unwrap();
        fs::write(root.join(INSTRUCTIONS_FILE), "test-command: cargo test\n").unwrap();
        fs::write(
            package.join(INSTRUCTIONS_FILE),
            "test-command: cargo nextest run -p api\n",
        )
        .unwrap();
        fs::write(&user_file, "test-command: cargo nextest run\n").unwrap();

        let instructions =
            AgentInstructions::discover_with_root(&source, &root, Some(&user_file)).unwrap();

        assert_eq!(instructions.layers().len(), 3);
        assert_eq!(instructions.layers()[0].scope, InstructionScope::Project);
        assert_eq!(
            instructions.layers()[1].path,
            package.join(INSTRUCTIONS_FILE)
        );
        assert_eq!(instructions.layers()[2].scope, InstructionScope::User);
        assert_eq!(instructions.layers()[2].precedence, 2);
        let rendered = instructions.context_fragment();
        assert!(rendered.find("cargo test").unwrap() < rendered.find("-p api").unwrap());
        assert!(rendered.find("-p api").unwrap() < rendered.rfind("cargo nextest run").unwrap());
    }

    #[test]
    fn nested_checkout_does_not_load_outer_checkout_instructions() {
        let fixture = TempDir::new().unwrap();
        let outer = fixture.path().join("outer");
        let inner = outer.join("worktrees").join("inner");
        let source = inner.join("src");
        fs::create_dir_all(outer.join(".git")).unwrap();
        fs::create_dir_all(&source).unwrap();
        fs::write(inner.join(".git"), "gitdir: elsewhere\n").unwrap();
        fs::write(outer.join(INSTRUCTIONS_FILE), "outer instructions").unwrap();
        fs::write(inner.join(INSTRUCTIONS_FILE), "inner instructions").unwrap();

        let instructions = AgentInstructions::discover(&source).unwrap();

        assert_eq!(instructions.workspace_root(), inner.canonicalize().unwrap());
        assert!(instructions
            .context_fragment()
            .contains("inner instructions"));
        assert!(!instructions
            .context_fragment()
            .contains("outer instructions"));
    }

    #[test]
    fn ignores_vendor_specific_and_similarly_named_files() {
        let fixture = TempDir::new().unwrap();
        fs::write(fixture.path().join("CLAUDE.md"), "vendor instructions").unwrap();
        fs::write(
            fixture.path().join("AGENTS.override.md"),
            "override instructions",
        )
        .unwrap();

        let instructions =
            AgentInstructions::discover_with_root(fixture.path(), fixture.path(), None).unwrap();

        assert!(instructions.is_empty());
        assert!(instructions.context_fragment().is_empty());
    }

    #[test]
    fn rejects_a_working_directory_outside_the_workspace() {
        let fixture = TempDir::new().unwrap();
        let workspace = fixture.path().join("workspace");
        let elsewhere = fixture.path().join("elsewhere");
        fs::create_dir_all(&workspace).unwrap();
        fs::create_dir_all(&elsewhere).unwrap();

        let error =
            AgentInstructions::discover_with_root(&elsewhere, &workspace, None).unwrap_err();

        assert!(matches!(error, InstructionsError::OutsideWorkspace { .. }));
    }
}
