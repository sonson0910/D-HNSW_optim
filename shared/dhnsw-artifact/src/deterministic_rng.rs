// ============================================================
//  Keccak-256 Deterministic RNG
//  Used for layer assignment in D-HNSW (Algorithm 3).
//
//  Instead of relying on platform-specific PRNGs, we use
//  Keccak-256(node_id || global_seed) to deterministically
//  assign each node to a layer. This ensures:
//    1. Identical layer assignments across all architectures
//    2. Unpredictable distribution (adversarial resistance)
//    3. No dependency on system entropy sources
// ============================================================

use tiny_keccak::{Hasher, Keccak};

/// Deterministic layer assignment using Keccak-256.
///
/// Given a node index and a global seed, produces a layer number
/// in [0, max_layer] following a geometric distribution approximation.
///
/// This replaces the typical `uniform(0,1)` + `log()` pattern
/// from the original HNSW (Malkov & Yashunin, 2018) with a
/// purely deterministic, platform-independent alternative.
pub struct DeterministicRng {
    seed: [u8; 32],
}

impl DeterministicRng {
    /// Create from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let mut hasher = Keccak::v256();
        let mut hash = [0u8; 32];
        hasher.update(&seed.to_le_bytes());
        hasher.finalize(&mut hash);
        DeterministicRng { seed: hash }
    }

    /// Compute the layer for a given node.
    ///
    /// Algorithm (from paper, Algorithm 3):
    ///   h ← Keccak256(node_id ‖ seed)
    ///   u ← first 8 bytes of h, interpreted as u64 LE
    ///   level ← floor(-ln(u / 2^64) × m_L)
    ///
    /// We approximate this without floating-point:
    ///   level ← leading_zeros(u) / bits_per_level
    ///
    /// The leading-zeros approach gives a geometric distribution
    /// where P(level ≥ k) ≈ 1/2^k, matching HNSW requirements.
    pub fn assign_layer(&self, node_id: u64, m_l: u32) -> u32 {
        let hash = self.hash_node(node_id);

        // Extract first 8 bytes as u64
        let u = u64::from_le_bytes([
            hash[0], hash[1], hash[2], hash[3],
            hash[4], hash[5], hash[6], hash[7],
        ]);

        // Geometric distribution via leading zeros
        // Each leading zero bit doubles the probability
        // of being assigned to a higher layer
        let lz = u.leading_zeros(); // 0..64
        let level = lz / m_l;

        level
    }

    /// Raw Keccak-256 hash for a node.
    /// H(node_id ‖ seed)
    pub fn hash_node(&self, node_id: u64) -> [u8; 32] {
        let mut hasher = Keccak::v256();
        let mut hash = [0u8; 32];
        hasher.update(&node_id.to_le_bytes());
        hasher.update(&self.seed);
        hasher.finalize(&mut hash);
        hash
    }

    /// Deterministic neighbor selection order.
    /// Returns a permutation seed for breaking ties
    /// during greedy search (Algorithm 4).
    pub fn neighbor_order_seed(&self, node_a: u64, node_b: u64) -> u64 {
        let mut hasher = Keccak::v256();
        let mut hash = [0u8; 32];
        hasher.update(&node_a.to_le_bytes());
        hasher.update(&node_b.to_le_bytes());
        hasher.update(&self.seed);
        hasher.finalize(&mut hash);

        u64::from_le_bytes([
            hash[0], hash[1], hash[2], hash[3],
            hash[4], hash[5], hash[6], hash[7],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_layer_assignment() {
        let rng = DeterministicRng::new(42);

        // Same inputs must produce same outputs
        let layer1 = rng.assign_layer(100, 4);
        let layer2 = rng.assign_layer(100, 4);
        assert_eq!(layer1, layer2, "Layer assignment must be deterministic");
    }

    #[test]
    fn test_different_nodes_different_layers() {
        let rng = DeterministicRng::new(42);

        // With enough nodes, we expect variation
        let mut layers = Vec::new();
        for i in 0..1000 {
            layers.push(rng.assign_layer(i, 4));
        }

        let max_layer = *layers.iter().max().unwrap();
        assert!(max_layer > 0, "Expected at least some nodes on higher layers");

        // Most nodes should be on layer 0
        let layer0_count = layers.iter().filter(|&&l| l == 0).count();
        assert!(layer0_count > 400, "Expected majority of nodes on layer 0, got {}", layer0_count);
    }

    #[test]
    fn test_hash_consistency() {
        let rng = DeterministicRng::new(12345);
        let hash1 = rng.hash_node(999);
        let hash2 = rng.hash_node(999);
        assert_eq!(hash1, hash2, "Hash must be deterministic");
    }

    #[test]
    fn test_different_seeds() {
        let rng1 = DeterministicRng::new(42);
        let rng2 = DeterministicRng::new(43);
        let hash1 = rng1.hash_node(100);
        let hash2 = rng2.hash_node(100);
        assert_ne!(hash1, hash2, "Different seeds must produce different hashes");
    }
}
