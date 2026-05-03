//! Deterministic Batch Insertion for Multi-Threaded D-HNSW Construction
//!
//! This module provides a `batch_insert()` function that enables parallel
//! vector processing while maintaining **strict determinism**.
//!
//! ## Algorithm
//!
//! 1. **Sort**: All vectors are sorted by their content hash (SHA-256).
//!    This gives a canonical ordering independent of submission order.
//!
//! 2. **Insert**: Vectors are inserted sequentially in hash order.
//!    This ensures identical graph topology regardless of original ordering.
//!
//! ## Why Content Hash?
//!
//! Using SHA-256 of vector content (not insertion index) provides:
//! - **Consensus safety**: Same vectors → same hash order → same graph
//! - **Replay safety**: Re-ordering transactions doesn't change the result
//! - **Auditability**: The canonical order can be independently verified

use sha2::{Sha256, Digest};

use crate::{
    deterministic_rng::DeterministicRng,
    error::Result,
    fixed_point::FixedPointVector,
    graph::HnswGraph,
};

/// A vector paired with its content hash for deterministic ordering.
struct HashedVector<const D: usize> {
    /// SHA-256 hash of the vector's raw I64F32 bits
    hash: [u8; 32],
    /// The original vector
    vector: FixedPointVector<D>,
}

/// Compute the SHA-256 hash of a FixedPointVector's raw bit representation.
///
/// This is deterministic because:
/// 1. I64F32 values have a canonical bit representation
/// 2. SHA-256 is platform-independent
fn content_hash<const D: usize>(vector: &FixedPointVector<D>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for component in &vector.components {
        hasher.update(component.to_bits().to_le_bytes());
    }
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Insert a batch of vectors into the graph in deterministic order.
///
/// The vectors are sorted by their SHA-256 content hash before insertion,
/// ensuring that the resulting graph topology is identical regardless of
/// the order in which vectors are submitted.
///
/// # Arguments
/// * `graph` - The HNSW graph to insert into
/// * `vectors` - Vectors to insert (order-independent)
/// * `rng` - Deterministic RNG for level assignment
///
/// # Returns
/// A vector of (original_index, assigned_node_id) pairs.
///
/// # Determinism Guarantee
/// Given the same set of vectors (in any order) and the same RNG state,
/// this function always produces the same graph topology and returns
/// the same node IDs (though the original_index mapping may differ).
pub fn batch_insert<const D: usize>(
    graph: &mut HnswGraph<D>,
    vectors: Vec<FixedPointVector<D>>,
    rng: &mut DeterministicRng,
) -> Result<Vec<(usize, usize)>> {
    if vectors.is_empty() {
        return Ok(Vec::new());
    }

    // Step 1: Compute content hash for each vector
    let mut hashed: Vec<(usize, HashedVector<D>)> = vectors
        .into_iter()
        .enumerate()
        .map(|(idx, vector)| {
            let hash = content_hash(&vector);
            (idx, HashedVector { hash, vector })
        })
        .collect();

    // Step 2: Sort by content hash (deterministic canonical ordering)
    hashed.sort_by(|a, b| a.1.hash.cmp(&b.1.hash));

    // Step 3: Insert in canonical order
    let mut results = Vec::with_capacity(hashed.len());
    for (original_idx, hv) in hashed {
        let node_id = graph.insert(hv.vector, rng)?;
        results.push((original_idx, node_id));
    }

    // Step 4: Sort results by original index for caller convenience
    results.sort_by_key(|&(orig_idx, _)| orig_idx);

    Ok(results)
}

/// Compute the graph's structural hash (SHA-256 of serialized graph).
///
/// This is used to verify that two graphs built from the same vectors
/// (potentially in different order via batch_insert) produce identical
/// topologies.
pub fn graph_hash<const D: usize>(graph: &HnswGraph<D>) -> Result<[u8; 32]> {
    let serialized = graph.serialize()?;
    let mut hasher = Sha256::new();
    hasher.update(&serialized);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deterministic_rng::DeterministicRng;

    fn create_test_rng() -> DeterministicRng {
        DeterministicRng::from_seed([42u8; 32])
    }

    #[test]
    fn test_batch_insert_determinism() {
        // Insert same vectors in different orders → same graph
        let vectors: Vec<FixedPointVector<3>> = (0..20)
            .map(|i| {
                FixedPointVector::from_f32_slice(&[
                    i as f32 * 0.5,
                    i as f32 * 1.0,
                    i as f32 * 1.5,
                ])
                .unwrap()
            })
            .collect();

        // Order 1: original
        let mut graph1: HnswGraph<3> = HnswGraph::new();
        let mut rng1 = create_test_rng();
        let _r1 = batch_insert(&mut graph1, vectors.clone(), &mut rng1).unwrap();

        // Order 2: reversed
        let mut reversed = vectors.clone();
        reversed.reverse();
        let mut graph2: HnswGraph<3> = HnswGraph::new();
        let mut rng2 = create_test_rng();
        let _r2 = batch_insert(&mut graph2, reversed, &mut rng2).unwrap();

        // Both graphs should have identical topology
        let hash1 = graph_hash(&graph1).unwrap();
        let hash2 = graph_hash(&graph2).unwrap();
        assert_eq!(hash1, hash2, "Batch insert must produce identical graphs regardless of input order");
    }

    #[test]
    fn test_batch_insert_vs_sequential() {
        // Batch insert should produce same result as sequential insert in hash order
        let vectors: Vec<FixedPointVector<3>> = (0..10)
            .map(|i| {
                FixedPointVector::from_f32_slice(&[
                    i as f32 * 0.3,
                    i as f32 * 0.6,
                    i as f32 * 0.9,
                ])
                .unwrap()
            })
            .collect();

        // Batch insert
        let mut graph_batch: HnswGraph<3> = HnswGraph::new();
        let mut rng_batch = create_test_rng();
        batch_insert(&mut graph_batch, vectors.clone(), &mut rng_batch).unwrap();

        // Manual: sort by hash, then insert sequentially
        let mut hashed: Vec<(usize, [u8; 32], FixedPointVector<3>)> = vectors
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let h = content_hash(&v);
                (i, h, v)
            })
            .collect();
        hashed.sort_by(|a, b| a.1.cmp(&b.1));

        let mut graph_manual: HnswGraph<3> = HnswGraph::new();
        let mut rng_manual = create_test_rng();
        for (_, _, v) in hashed {
            graph_manual.insert(v, &mut rng_manual).unwrap();
        }

        let hash_batch = graph_hash(&graph_batch).unwrap();
        let hash_manual = graph_hash(&graph_manual).unwrap();
        assert_eq!(hash_batch, hash_manual, "Batch must equal manual hash-ordered sequential");
    }

    #[test]
    fn test_empty_batch() {
        let mut graph: HnswGraph<3> = HnswGraph::new();
        let mut rng = create_test_rng();
        let results = batch_insert(&mut graph, vec![], &mut rng).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_content_hash_determinism() {
        let v = FixedPointVector::<3>::from_f32_slice(&[1.0, 2.0, 3.0]).unwrap();
        let h1 = content_hash(&v);
        let h2 = content_hash(&v);
        assert_eq!(h1, h2, "Content hash must be deterministic");
    }
}
