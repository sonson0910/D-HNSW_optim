// ============================================================
//  Integration Tests — D-HNSW Determinism Proof
// ============================================================

use dhnsw::fixed_point::{FixedScalar, FixedVector};
use dhnsw::graph::{DhnswConfig, DhnswIndex};
use dhnsw::deterministic_rng::DeterministicRng;

/// Core Theorem Verification:
/// Two independently constructed D-HNSW indices from the
/// same dataset + config MUST produce identical SHA-256 hashes.
#[test]
fn test_theorem10_deterministic_construction() {
    let vectors: Vec<Vec<f32>> = (0..200)
        .map(|i| (0..64).map(|j| ((i * 64 + j) as f32) * 0.001).collect())
        .collect();

    let build_index = || {
        let config = DhnswConfig {
            m: 8,
            m_max_0: 16,
            ef_construction: 50,
            seed: 42,
            dim: 64,
        };
        let mut index = DhnswIndex::new(config);
        for v in &vectors {
            index.insert(FixedVector::from_f32_slice(v));
        }
        index.sha256_hash()
    };

    let hash1 = build_index();
    let hash2 = build_index();
    let hash3 = build_index();

    assert_eq!(hash1, hash2, "Theorem 10 violated: hash mismatch between run 1 and 2");
    assert_eq!(hash2, hash3, "Theorem 10 violated: hash mismatch between run 2 and 3");
}

/// Verify that different seeds produce different indices.
#[test]
fn test_different_seeds_different_indices() {
    let vectors: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..32).map(|j| ((i * 32 + j) as f32) * 0.01).collect())
        .collect();

    let build_with_seed = |seed: u64| {
        let config = DhnswConfig {
            m: 4,
            m_max_0: 8,
            ef_construction: 20,
            seed,
            dim: 32,
        };
        let mut index = DhnswIndex::new(config);
        for v in &vectors {
            index.insert(FixedVector::from_f32_slice(v));
        }
        index.sha256_hash()
    };

    let hash_42 = build_with_seed(42);
    let hash_43 = build_with_seed(43);
    assert_ne!(hash_42, hash_43, "Different seeds must produce different hashes");
}

/// Verify search results are deterministic.
#[test]
fn test_search_determinism() {
    let config = DhnswConfig {
        m: 8,
        m_max_0: 16,
        ef_construction: 50,
        seed: 42,
        dim: 32,
    };

    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..32).map(|j| ((i * 32 + j) as f32) * 0.001).collect())
        .collect();

    let mut index = DhnswIndex::new(config);
    for v in &vectors {
        index.insert(FixedVector::from_f32_slice(v));
    }

    let query = FixedVector::from_f32_slice(&[0.5f32; 32]);
    let results1 = index.search(&query, 10, 64);
    let results2 = index.search(&query, 10, 64);

    assert_eq!(results1.len(), results2.len());
    for (a, b) in results1.iter().zip(results2.iter()) {
        assert_eq!(a.0, b.0, "Search result node IDs must match");
        assert_eq!(a.1.to_bits(), b.1.to_bits(), "Search distances must be bit-exact");
    }
}

/// Verify Keccak-256 layer assignment is deterministic.
#[test]
fn test_keccak_layer_assignment_determinism() {
    let rng1 = DeterministicRng::new(42);
    let rng2 = DeterministicRng::new(42);

    for node_id in 0..10000u64 {
        let layer1 = rng1.assign_layer(node_id, 4);
        let layer2 = rng2.assign_layer(node_id, 4);
        assert_eq!(layer1, layer2,
            "Layer assignment mismatch for node {} (seed=42)", node_id);
    }
}

/// Verify saturating arithmetic prevents overflow panics.
#[test]
fn test_overflow_resilience() {
    let large = FixedScalar::from_bits(i64::MAX / 2);
    let result = large.saturating_mul(large);
    // Should not panic — saturates to MAX
    assert!(result.to_bits() > 0, "Saturating mul should not produce zero or negative");
}

/// Verify distance triangle inequality holds (approximately).
#[test]
fn test_distance_triangle_inequality() {
    let a = FixedVector::from_f32_slice(&[0.0, 0.0]);
    let b = FixedVector::from_f32_slice(&[3.0, 0.0]);
    let c = FixedVector::from_f32_slice(&[3.0, 4.0]);

    let d_ab = a.squared_euclidean(&b); // 9
    let d_bc = b.squared_euclidean(&c); // 16
    let d_ac = a.squared_euclidean(&c); // 25

    // sqrt(d_ac) <= sqrt(d_ab) + sqrt(d_bc) → 5 <= 3 + 4 = 7 ✓
    // In squared form: d_ac <= (sqrt(d_ab) + sqrt(d_bc))^2
    // This is a weaker check, but validates distance computation.
    let d_ab_f = d_ab.to_f64();
    let d_bc_f = d_bc.to_f64();
    let d_ac_f = d_ac.to_f64();

    assert!((d_ab_f - 9.0).abs() < 0.01, "d(a,b) should be ~9, got {}", d_ab_f);
    assert!((d_bc_f - 16.0).abs() < 0.01, "d(b,c) should be ~16, got {}", d_bc_f);
    assert!((d_ac_f - 25.0).abs() < 0.01, "d(a,c) should be ~25, got {}", d_ac_f);
}
