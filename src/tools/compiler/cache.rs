//! Compilation cache types.
//!
//! Provides LRU caching for compiled binaries to avoid redundant compilation
//! of identical code.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

/// A hash of compiled code, used as cache key.
///
/// Uses `DefaultHasher` to compute a 64-bit hash of the source code.
/// Two identical code strings will always produce the same hash.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::compiler::CodeHash;
///
/// let hash1 = CodeHash::from_code("fn main() {}");
/// let hash2 = CodeHash::from_code("fn main() {}");
/// assert_eq!(hash1, hash2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CodeHash(u64);

impl CodeHash {
    /// Computes a hash from code string.
    #[must_use]
    pub fn from_code(code: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        Self(hasher.finish())
    }

    /// Returns the inner u64 value.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for CodeHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// Configuration for the compilation cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Maximum total size of cached binaries in bytes.
    pub max_total_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            max_total_size: 100 * 1024 * 1024, // 100 MB
        }
    }
}

/// Entry in the compilation cache.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Compiled binary bytes.
    binary: Vec<u8>,
    /// Last access time (monotonic counter for LRU).
    last_access: u64,
}

/// Inner state of the compilation cache, protected by a Mutex.
#[derive(Debug)]
struct CacheInner {
    /// Map from code hash to cached entry.
    entries: HashMap<CodeHash, CacheEntry>,
    /// Total size of all cached binaries in bytes.
    total_size: usize,
    /// Monotonically increasing counter for LRU tracking.
    access_counter: u64,
}

/// Thread-safe compilation cache with LRU eviction.
///
/// Caches compiled binaries by code hash to avoid redundant compilation.
/// When the cache exceeds its configured limits (entry count or total size),
/// the least recently used entries are evicted.
///
/// # Thread Safety
///
/// All operations on the cache are thread-safe. The cache uses a `Mutex`
/// internally and handles poisoned locks gracefully.
///
/// # Example
///
/// ```rust,ignore
/// use acton_ai::tools::compiler::{CompilationCache, CacheConfig, CodeHash};
///
/// let config = CacheConfig {
///     max_entries: 50,
///     max_total_size: 50 * 1024 * 1024,
/// };
///
/// let cache = CompilationCache::new(config);
/// let hash = CodeHash::from_code("fn main() {}");
///
/// cache.insert(hash, vec![0x7f, 0x45, 0x4c, 0x46]);
/// let binary = cache.get(hash);
/// assert!(binary.is_some());
/// ```
#[derive(Debug)]
pub struct CompilationCache {
    inner: Mutex<CacheInner>,
    config: CacheConfig,
}

impl CompilationCache {
    /// Creates a new compilation cache with the given configuration.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                total_size: 0,
                access_counter: 0,
            }),
            config,
        }
    }

    /// Gets a cached binary by hash, if present.
    ///
    /// Updates the LRU access time on successful retrieval.
    ///
    /// # Arguments
    ///
    /// * `hash` - The code hash to look up
    ///
    /// # Returns
    ///
    /// The cached binary bytes if found, or `None` if not in cache.
    pub fn get(&self, hash: CodeHash) -> Option<Vec<u8>> {
        let mut inner = self.inner.lock().ok()?;

        if inner.entries.contains_key(&hash) {
            inner.access_counter += 1;
            let access = inner.access_counter;
            if let Some(entry) = inner.entries.get_mut(&hash) {
                entry.last_access = access;
                return Some(entry.binary.clone());
            }
        }
        None
    }

    /// Inserts a compiled binary into the cache.
    ///
    /// If the cache would exceed its configured limits after insertion,
    /// the least recently used entries are evicted first.
    ///
    /// # Arguments
    ///
    /// * `hash` - The code hash as the cache key
    /// * `binary` - The compiled binary bytes to cache
    pub fn insert(&self, hash: CodeHash, binary: Vec<u8>) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };

        let binary_size = binary.len();

        // Evict entries if necessary
        while inner.entries.len() >= self.config.max_entries
            || inner.total_size + binary_size > self.config.max_total_size
        {
            if !self.evict_lru(&mut inner) {
                break;
            }
        }

        inner.access_counter += 1;
        let access = inner.access_counter;

        if let Some(old_entry) = inner.entries.insert(
            hash,
            CacheEntry {
                binary,
                last_access: access,
            },
        ) {
            inner.total_size -= old_entry.binary.len();
        }
        inner.total_size += binary_size;
    }

    /// Clears all entries from the cache.
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.entries.clear();
            inner.total_size = 0;
        }
    }

    /// Returns cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock().ok();
        match inner {
            Some(inner) => CacheStats {
                entry_count: inner.entries.len(),
                total_size: inner.total_size,
                max_entries: self.config.max_entries,
                max_total_size: self.config.max_total_size,
            },
            None => CacheStats::default(),
        }
    }

    /// Evicts the least recently used entry.
    ///
    /// Returns `true` if an entry was evicted, `false` if the cache was empty.
    fn evict_lru(&self, inner: &mut CacheInner) -> bool {
        let lru_key = inner
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| *key);

        if let Some(key) = lru_key {
            if let Some(entry) = inner.entries.remove(&key) {
                inner.total_size -= entry.binary.len();
                return true;
            }
        }
        false
    }
}

impl Default for CompilationCache {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Statistics about the compilation cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of entries in the cache.
    pub entry_count: usize,
    /// Total size of cached binaries in bytes.
    pub total_size: usize,
    /// Maximum number of entries configured.
    pub max_entries: usize,
    /// Maximum total size in bytes configured.
    pub max_total_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CodeHash tests ---

    #[test]
    fn code_hash_same_code_same_hash() {
        let hash1 = CodeHash::from_code("fn test() {}");
        let hash2 = CodeHash::from_code("fn test() {}");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn code_hash_different_code_different_hash() {
        let hash1 = CodeHash::from_code("fn test1() {}");
        let hash2 = CodeHash::from_code("fn test2() {}");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn code_hash_display() {
        let hash = CodeHash::from_code("test");
        let display = hash.to_string();
        assert_eq!(display.len(), 16); // 16 hex chars
    }

    #[test]
    fn code_hash_as_u64() {
        let hash = CodeHash::from_code("test");
        let _value: u64 = hash.as_u64();
    }

    #[test]
    fn code_hash_is_copy() {
        let hash1 = CodeHash::from_code("test");
        let hash2 = hash1; // Copy
        assert_eq!(hash1, hash2);
    }

    // --- CacheConfig tests ---

    #[test]
    fn cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_entries, 100);
        assert_eq!(config.max_total_size, 100 * 1024 * 1024);
    }

    #[test]
    fn cache_config_clone() {
        let config1 = CacheConfig {
            max_entries: 50,
            max_total_size: 50 * 1024 * 1024,
        };
        let config2 = config1.clone();
        assert_eq!(config1.max_entries, config2.max_entries);
    }

    // --- CompilationCache tests ---

    #[test]
    fn cache_get_missing_returns_none() {
        let cache = CompilationCache::default();
        let hash = CodeHash::from_code("test");
        assert!(cache.get(hash).is_none());
    }

    #[test]
    fn cache_insert_and_get() {
        let cache = CompilationCache::default();
        let hash = CodeHash::from_code("test");
        let binary = vec![1, 2, 3, 4];

        cache.insert(hash, binary.clone());
        let retrieved = cache.get(hash);

        assert_eq!(retrieved, Some(binary));
    }

    #[test]
    fn cache_respects_max_entries() {
        let config = CacheConfig {
            max_entries: 2,
            max_total_size: 1024 * 1024,
        };
        let cache = CompilationCache::new(config);

        cache.insert(CodeHash::from_code("a"), vec![1]);
        cache.insert(CodeHash::from_code("b"), vec![2]);
        cache.insert(CodeHash::from_code("c"), vec![3]);

        let stats = cache.stats();
        assert!(stats.entry_count <= 2);
    }

    #[test]
    fn cache_respects_max_size() {
        let config = CacheConfig {
            max_entries: 100,
            max_total_size: 10, // Very small
        };
        let cache = CompilationCache::new(config);

        cache.insert(CodeHash::from_code("a"), vec![1, 2, 3, 4, 5]);
        cache.insert(CodeHash::from_code("b"), vec![1, 2, 3, 4, 5]);
        cache.insert(CodeHash::from_code("c"), vec![1, 2, 3, 4, 5]);

        let stats = cache.stats();
        assert!(stats.total_size <= 10);
    }

    #[test]
    fn cache_clear() {
        let cache = CompilationCache::default();
        cache.insert(CodeHash::from_code("test"), vec![1, 2, 3]);
        cache.clear();
        assert_eq!(cache.stats().entry_count, 0);
        assert_eq!(cache.stats().total_size, 0);
    }

    #[test]
    fn cache_stats() {
        let cache = CompilationCache::default();
        cache.insert(CodeHash::from_code("test1"), vec![1, 2, 3]);
        cache.insert(CodeHash::from_code("test2"), vec![4, 5]);

        let stats = cache.stats();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.total_size, 5);
    }

    #[test]
    fn cache_lru_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            max_total_size: 1024 * 1024,
        };
        let cache = CompilationCache::new(config);

        let hash_a = CodeHash::from_code("a");
        let hash_b = CodeHash::from_code("b");
        let hash_c = CodeHash::from_code("c");

        cache.insert(hash_a, vec![1]);
        cache.insert(hash_b, vec![2]);

        // Access 'a' to make it more recently used
        cache.get(hash_a);

        // Insert 'c' - should evict 'b' (least recently used)
        cache.insert(hash_c, vec![3]);

        assert!(cache.get(hash_a).is_some());
        assert!(cache.get(hash_b).is_none()); // Evicted
        assert!(cache.get(hash_c).is_some());
    }

    #[test]
    fn cache_update_existing() {
        let cache = CompilationCache::default();
        let hash = CodeHash::from_code("test");

        cache.insert(hash, vec![1, 2, 3]);
        let stats_before = cache.stats();

        cache.insert(hash, vec![4, 5]); // Update with smaller binary
        let stats_after = cache.stats();

        assert_eq!(stats_before.entry_count, 1);
        assert_eq!(stats_after.entry_count, 1);
        assert_eq!(stats_before.total_size, 3);
        assert_eq!(stats_after.total_size, 2);
    }

    #[test]
    fn cache_default_is_new_with_default_config() {
        let cache1 = CompilationCache::default();
        let cache2 = CompilationCache::new(CacheConfig::default());

        let stats1 = cache1.stats();
        let stats2 = cache2.stats();

        assert_eq!(stats1.max_entries, stats2.max_entries);
        assert_eq!(stats1.max_total_size, stats2.max_total_size);
    }

    // --- CacheStats tests ---

    #[test]
    fn cache_stats_default() {
        let stats = CacheStats::default();
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.total_size, 0);
        assert_eq!(stats.max_entries, 0);
        assert_eq!(stats.max_total_size, 0);
    }
}
