//! Vector embeddings for semantic memory search.
//!
//! This module provides:
//! - [`Embedding`]: A vector embedding for semantic similarity
//! - [`EmbeddingProvider`]: Trait for embedding generation services
//! - [`StubEmbeddingProvider`]: Test implementation using deterministic hashing
//! - [`Memory`]: A memory entry with optional embedding
//! - [`ScoredMemory`]: A memory with its similarity score
//!
//! ## Example
//!
//! ```rust,ignore
//! use acton_ai::memory::{Embedding, EmbeddingProvider, StubEmbeddingProvider};
//!
//! #[tokio::main]
//! async fn main() {
//!     let provider = StubEmbeddingProvider::default();
//!     let embedding = provider.embed("Hello, world!").await.unwrap();
//!     println!("Embedding dimension: {}", embedding.dimension());
//! }
//! ```

use crate::types::{AgentId, MemoryId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Embedding Error
// =============================================================================

/// Errors that can occur in embedding operations.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingError {
    /// The embedding vector is empty.
    EmptyVector,
    /// Invalid byte length for conversion.
    InvalidByteLength {
        /// The actual byte length.
        length: usize,
    },
    /// Dimension mismatch between embeddings.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Provider failed to generate embedding.
    GenerationFailed {
        /// The provider name.
        provider: String,
        /// Error message.
        message: String,
    },
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVector => write!(f, "embedding vector cannot be empty"),
            Self::InvalidByteLength { length } => {
                write!(
                    f,
                    "invalid byte length {} for embedding; must be multiple of 4",
                    length
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "embedding dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::GenerationFailed { provider, message } => {
                write!(f, "embedding provider '{}' failed: {}", provider, message)
            }
        }
    }
}

impl std::error::Error for EmbeddingError {}

// =============================================================================
// Embedding
// =============================================================================

/// A vector embedding for semantic search.
///
/// Embeddings are fixed-dimensional vectors representing semantic content.
/// Common dimensions: 384 (MiniLM), 768 (BERT), 1536 (OpenAI).
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    /// The embedding vector values.
    values: Vec<f32>,
    /// The dimensionality of the embedding.
    dimension: usize,
}

impl Embedding {
    /// Creates a new embedding from a vector of values.
    ///
    /// # Arguments
    ///
    /// * `values` - The embedding vector values
    ///
    /// # Returns
    ///
    /// A new `Embedding` instance.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::EmptyVector` if values is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use acton_ai::memory::Embedding;
    ///
    /// let embedding = Embedding::new(vec![0.1, 0.2, 0.3]).unwrap();
    /// assert_eq!(embedding.dimension(), 3);
    /// ```
    pub fn new(values: Vec<f32>) -> Result<Self, EmbeddingError> {
        if values.is_empty() {
            return Err(EmbeddingError::EmptyVector);
        }
        let dimension = values.len();
        Ok(Self { values, dimension })
    }

    /// Returns the dimension of the embedding.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the embedding values as a slice.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Converts the embedding to bytes for storage.
    ///
    /// Uses little-endian byte order for cross-platform compatibility.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.values.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Creates an embedding from bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The byte representation of the embedding
    ///
    /// # Returns
    ///
    /// The reconstructed `Embedding`.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidByteLength` if bytes length is not a multiple of 4.
    /// Returns `EmbeddingError::EmptyVector` if bytes is empty.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EmbeddingError> {
        if bytes.is_empty() {
            return Err(EmbeddingError::EmptyVector);
        }
        if !bytes.len().is_multiple_of(4) {
            return Err(EmbeddingError::InvalidByteLength {
                length: bytes.len(),
            });
        }
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Self::new(values)
    }

    /// Computes cosine similarity between two embeddings.
    ///
    /// Cosine similarity measures the cosine of the angle between two vectors,
    /// ranging from -1 (opposite) to 1 (identical direction).
    ///
    /// # Arguments
    ///
    /// * `other` - The other embedding to compare against
    ///
    /// # Returns
    ///
    /// Similarity score between -1.0 and 1.0.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::DimensionMismatch` if embeddings have different dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use acton_ai::memory::Embedding;
    ///
    /// let a = Embedding::new(vec![1.0, 0.0]).unwrap();
    /// let b = Embedding::new(vec![1.0, 0.0]).unwrap();
    /// assert!((a.cosine_similarity(&b).unwrap() - 1.0).abs() < 0.0001);
    /// ```
    pub fn cosine_similarity(&self, other: &Self) -> Result<f32, EmbeddingError> {
        if self.dimension != other.dimension {
            return Err(EmbeddingError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let dot_product: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();

        let magnitude_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (magnitude_a * magnitude_b))
    }

    /// Normalizes the embedding to unit length.
    ///
    /// Returns a new embedding with the same direction but magnitude 1.
    /// If the embedding has zero magnitude, returns a copy unchanged.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let magnitude: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude == 0.0 {
            return self.clone();
        }
        let values: Vec<f32> = self.values.iter().map(|x| x / magnitude).collect();
        // Safe to unwrap since we know values is non-empty
        Self::new(values).expect("normalize should not produce empty vector")
    }
}

// =============================================================================
// Embedding Provider Trait
// =============================================================================

/// Trait for embedding generation services.
///
/// Implementations can wrap OpenAI, local models (sentence-transformers),
/// or other embedding APIs.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::memory::{EmbeddingProvider, StubEmbeddingProvider};
///
/// async fn example() {
///     let provider = StubEmbeddingProvider::default();
///     let embedding = provider.embed("Hello").await.unwrap();
///     assert_eq!(embedding.dimension(), 384);
/// }
/// ```
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generates an embedding for the given text.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// The embedding vector for the text.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::GenerationFailed` if embedding generation fails.
    async fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError>;

    /// Returns the dimension of embeddings produced by this provider.
    fn dimension(&self) -> usize;

    /// Returns the name/identifier of this provider.
    fn name(&self) -> &str;
}

// =============================================================================
// Stub Embedding Provider
// =============================================================================

/// A stub embedding provider for testing.
///
/// Generates deterministic pseudo-embeddings based on text hash.
/// NOT suitable for production semantic search - use a real embedding
/// model (OpenAI, sentence-transformers, etc.) for actual similarity.
///
/// # Determinism
///
/// The same text always produces the same embedding, making tests reproducible.
///
/// # Example
///
/// ```rust
/// use acton_ai::memory::{EmbeddingProvider, StubEmbeddingProvider};
///
/// # tokio_test::block_on(async {
/// let provider = StubEmbeddingProvider::default();
/// let e1 = provider.embed("hello").await.unwrap();
/// let e2 = provider.embed("hello").await.unwrap();
/// assert_eq!(e1, e2); // Same text = same embedding
/// # });
/// ```
#[derive(Debug, Clone)]
pub struct StubEmbeddingProvider {
    dimension: usize,
}

impl StubEmbeddingProvider {
    /// Creates a new stub provider with the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of generated embeddings
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Default for StubEmbeddingProvider {
    fn default() -> Self {
        Self::new(384) // MiniLM dimension
    }
}

#[async_trait]
impl EmbeddingProvider for StubEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Generate deterministic values based on text hash
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        let values: Vec<f32> = (0..self.dimension)
            .map(|i| {
                // Simple PRNG using seed + index
                let combined = seed
                    .wrapping_add(i as u64)
                    .wrapping_mul(0x5851_f42d_4c95_7f2d);
                // Map to [-1, 1] range using sin for smooth distribution
                ((combined as f64 / u64::MAX as f64) * std::f64::consts::PI * 2.0).sin() as f32
            })
            .collect();

        Embedding::new(values)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "stub"
    }
}

// =============================================================================
// Memory
// =============================================================================

/// A memory entry with optional embedding.
///
/// Memories are persisted facts or information that agents can recall
/// through semantic search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier for this memory.
    pub id: MemoryId,
    /// The agent this memory belongs to.
    pub agent_id: AgentId,
    /// The content of the memory.
    pub content: String,
    /// Optional embedding for semantic search.
    #[serde(skip)]
    pub embedding: Option<Embedding>,
    /// When this memory was created (ISO 8601 format).
    pub created_at: String,
}

impl Memory {
    /// Creates a new memory without embedding.
    ///
    /// # Arguments
    ///
    /// * `agent_id` - The agent this memory belongs to
    /// * `content` - The content of the memory
    #[must_use]
    pub fn new(agent_id: AgentId, content: impl Into<String>) -> Self {
        Self {
            id: MemoryId::new(),
            agent_id,
            content: content.into(),
            embedding: None,
            created_at: current_timestamp(),
        }
    }

    /// Creates a new memory with embedding.
    ///
    /// # Arguments
    ///
    /// * `agent_id` - The agent this memory belongs to
    /// * `content` - The content of the memory
    /// * `embedding` - The embedding for semantic search
    #[must_use]
    pub fn with_embedding(
        agent_id: AgentId,
        content: impl Into<String>,
        embedding: Embedding,
    ) -> Self {
        Self {
            id: MemoryId::new(),
            agent_id,
            content: content.into(),
            embedding: Some(embedding),
            created_at: current_timestamp(),
        }
    }
}

/// A memory with its similarity score from search.
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    /// The memory entry.
    pub memory: Memory,
    /// Similarity score (typically 0.0 to 1.0, can be negative for opposite vectors).
    pub score: f32,
}

/// Returns the current timestamp in ISO 8601 format.
fn current_timestamp() -> String {
    // Use a simple format since we don't want to add chrono dependency
    // In production, you might want chrono for proper timezone handling
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Convert to approximate ISO 8601 (without full date parsing)
    // This is a simplification - in production use chrono
    format!("{}", secs)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Embedding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn embedding_new_valid() {
        let embedding = Embedding::new(vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(embedding.dimension(), 3);
        assert_eq!(embedding.values(), &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn embedding_new_empty_fails() {
        let result = Embedding::new(vec![]);
        assert!(matches!(result, Err(EmbeddingError::EmptyVector)));
    }

    #[test]
    fn embedding_to_bytes_roundtrip() {
        let original = Embedding::new(vec![1.0, -0.5, 0.0, 0.25]).unwrap();
        let bytes = original.to_bytes();
        let restored = Embedding::from_bytes(&bytes).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn embedding_from_bytes_empty_fails() {
        let result = Embedding::from_bytes(&[]);
        assert!(matches!(result, Err(EmbeddingError::EmptyVector)));
    }

    #[test]
    fn embedding_from_bytes_invalid_length() {
        let result = Embedding::from_bytes(&[1, 2, 3]); // Not multiple of 4
        assert!(matches!(
            result,
            Err(EmbeddingError::InvalidByteLength { length: 3 })
        ));
    }

    #[test]
    fn embedding_cosine_similarity_same_vector() {
        let a = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        let b = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        let similarity = a.cosine_similarity(&b).unwrap();
        assert!((similarity - 1.0).abs() < 0.0001);
    }

    #[test]
    fn embedding_cosine_similarity_orthogonal() {
        let a = Embedding::new(vec![1.0, 0.0]).unwrap();
        let b = Embedding::new(vec![0.0, 1.0]).unwrap();
        let similarity = a.cosine_similarity(&b).unwrap();
        assert!(similarity.abs() < 0.0001);
    }

    #[test]
    fn embedding_cosine_similarity_opposite() {
        let a = Embedding::new(vec![1.0, 0.0]).unwrap();
        let b = Embedding::new(vec![-1.0, 0.0]).unwrap();
        let similarity = a.cosine_similarity(&b).unwrap();
        assert!((similarity + 1.0).abs() < 0.0001);
    }

    #[test]
    fn embedding_cosine_similarity_dimension_mismatch() {
        let a = Embedding::new(vec![1.0, 0.0]).unwrap();
        let b = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        let result = a.cosine_similarity(&b);
        assert!(matches!(
            result,
            Err(EmbeddingError::DimensionMismatch {
                expected: 2,
                actual: 3
            })
        ));
    }

    #[test]
    fn embedding_cosine_similarity_zero_vector() {
        let a = Embedding::new(vec![0.0, 0.0]).unwrap();
        let b = Embedding::new(vec![1.0, 0.0]).unwrap();
        let similarity = a.cosine_similarity(&b).unwrap();
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn embedding_normalize() {
        let a = Embedding::new(vec![3.0, 4.0]).unwrap();
        let normalized = a.normalize();

        // Magnitude should be 1
        let magnitude: f32 = normalized
            .values()
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((magnitude - 1.0).abs() < 0.0001);

        // Should be [0.6, 0.8]
        assert!((normalized.values()[0] - 0.6).abs() < 0.0001);
        assert!((normalized.values()[1] - 0.8).abs() < 0.0001);
    }

    #[test]
    fn embedding_normalize_zero_vector() {
        let a = Embedding::new(vec![0.0, 0.0]).unwrap();
        let normalized = a.normalize();
        assert_eq!(normalized.values(), &[0.0, 0.0]);
    }

    // -------------------------------------------------------------------------
    // EmbeddingError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn embedding_error_display_empty_vector() {
        let err = EmbeddingError::EmptyVector;
        assert_eq!(err.to_string(), "embedding vector cannot be empty");
    }

    #[test]
    fn embedding_error_display_invalid_byte_length() {
        let err = EmbeddingError::InvalidByteLength { length: 7 };
        assert!(err.to_string().contains("7"));
        assert!(err.to_string().contains("multiple of 4"));
    }

    #[test]
    fn embedding_error_display_dimension_mismatch() {
        let err = EmbeddingError::DimensionMismatch {
            expected: 384,
            actual: 768,
        };
        assert!(err.to_string().contains("384"));
        assert!(err.to_string().contains("768"));
    }

    #[test]
    fn embedding_error_display_generation_failed() {
        let err = EmbeddingError::GenerationFailed {
            provider: "openai".to_string(),
            message: "rate limited".to_string(),
        };
        assert!(err.to_string().contains("openai"));
        assert!(err.to_string().contains("rate limited"));
    }

    // -------------------------------------------------------------------------
    // StubEmbeddingProvider Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn stub_provider_embed_same_text_same_embedding() {
        let provider = StubEmbeddingProvider::default();
        let e1 = provider.embed("hello world").await.unwrap();
        let e2 = provider.embed("hello world").await.unwrap();
        assert_eq!(e1, e2);
    }

    #[tokio::test]
    async fn stub_provider_embed_different_text_different_embedding() {
        let provider = StubEmbeddingProvider::default();
        let e1 = provider.embed("hello").await.unwrap();
        let e2 = provider.embed("goodbye").await.unwrap();
        assert_ne!(e1, e2);
    }

    #[tokio::test]
    async fn stub_provider_dimension() {
        let provider = StubEmbeddingProvider::new(512);
        assert_eq!(provider.dimension(), 512);

        let embedding = provider.embed("test").await.unwrap();
        assert_eq!(embedding.dimension(), 512);
    }

    #[tokio::test]
    async fn stub_provider_name() {
        let provider = StubEmbeddingProvider::default();
        assert_eq!(provider.name(), "stub");
    }

    #[tokio::test]
    async fn stub_provider_default_dimension() {
        let provider = StubEmbeddingProvider::default();
        assert_eq!(provider.dimension(), 384); // MiniLM
    }

    // -------------------------------------------------------------------------
    // Memory Tests
    // -------------------------------------------------------------------------

    #[test]
    fn memory_new_creates_valid_memory() {
        let agent_id = AgentId::new();
        let memory = Memory::new(agent_id.clone(), "test content");

        assert_eq!(memory.agent_id, agent_id);
        assert_eq!(memory.content, "test content");
        assert!(memory.embedding.is_none());
        assert!(!memory.created_at.is_empty());
    }

    #[test]
    fn memory_with_embedding() {
        let agent_id = AgentId::new();
        let embedding = Embedding::new(vec![0.1, 0.2, 0.3]).unwrap();
        let memory = Memory::with_embedding(agent_id.clone(), "test", embedding.clone());

        assert_eq!(memory.agent_id, agent_id);
        assert_eq!(memory.content, "test");
        assert_eq!(memory.embedding, Some(embedding));
    }

    #[test]
    fn memory_unique_ids() {
        let agent_id = AgentId::new();
        let m1 = Memory::new(agent_id.clone(), "one");
        let m2 = Memory::new(agent_id, "two");
        assert_ne!(m1.id, m2.id);
    }

    #[test]
    fn memory_serialization_excludes_embedding() {
        let agent_id = AgentId::new();
        let embedding = Embedding::new(vec![0.1, 0.2, 0.3]).unwrap();
        let memory = Memory::with_embedding(agent_id, "test", embedding);

        let json = serde_json::to_string(&memory).unwrap();
        // Embedding should not be in JSON (serde skip)
        assert!(!json.contains("embedding"));
    }

    // -------------------------------------------------------------------------
    // ScoredMemory Tests
    // -------------------------------------------------------------------------

    #[test]
    fn scored_memory_creation() {
        let agent_id = AgentId::new();
        let memory = Memory::new(agent_id, "test");
        let scored = ScoredMemory {
            memory: memory.clone(),
            score: 0.85,
        };

        assert_eq!(scored.memory.content, "test");
        assert!((scored.score - 0.85).abs() < 0.0001);
    }
}
