// ============================================================
//  D-HNSW Graph — Core Index Structure
//  Implements the Deterministic HNSW graph with:
//    - Q32.32 fixed-point distance
//    - Keccak-256 layer assignment
//    - Deterministic greedy search (Algorithm 4)
//    - Deterministic insertion (Algorithm 5)
// ============================================================

use crate::deterministic_rng::DeterministicRng;
use crate::fixed_point::{FixedScalar, FixedVector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// Configuration for a D-HNSW index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DhnswConfig {
    /// Max bidirectional connections per node per layer (M)
    pub m: usize,
    /// Max connections for layer 0 (typically 2*M)
    pub m_max_0: usize,
    /// Construction search width
    pub ef_construction: usize,
    /// Global deterministic seed
    pub seed: u64,
    /// Vector dimensionality
    pub dim: usize,
}

impl Default for DhnswConfig {
    fn default() -> Self {
        DhnswConfig {
            m: 16,
            m_max_0: 32,
            ef_construction: 200,
            seed: 42,
            dim: 128,
        }
    }
}

/// A node in the D-HNSW graph.
#[derive(Clone, Debug)]
pub struct Node {
    /// Unique node identifier (insertion order)
    pub id: usize,
    /// Fixed-point vector
    pub vector: FixedVector,
    /// Neighbors per layer: neighbors[layer] = vec![(neighbor_id, distance)]
    pub neighbors: Vec<Vec<(usize, FixedScalar)>>,
    /// Maximum layer this node belongs to
    pub max_layer: u32,
}

/// The D-HNSW index.
pub struct DhnswIndex {
    pub config: DhnswConfig,
    pub nodes: Vec<Node>,
    pub entry_point: Option<usize>,
    pub max_level: u32,
    rng: DeterministicRng,
}

/// A candidate in the priority queue.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Candidate {
    distance: i64, // Using raw bits for Ord
    node_id: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl DhnswIndex {
    /// Create a new empty D-HNSW index.
    pub fn new(config: DhnswConfig) -> Self {
        let rng = DeterministicRng::new(config.seed);
        DhnswIndex {
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            rng,
        }
    }

    /// Insert a vector into the index.
    ///
    /// Implements Algorithm 5 from the paper.
    /// Key determinism guarantees:
    ///   1. Layer assignment via Keccak-256 (not PRNG)
    ///   2. Distance via Q32.32 saturating arithmetic
    ///   3. Tie-breaking by node_id (total ordering)
    pub fn insert(&mut self, vector: FixedVector) {
        let node_id = self.nodes.len();
        let m_l = (1.0 / (self.config.m as f64).ln()).ceil() as u32;
        let node_layer = self.rng.assign_layer(node_id as u64, m_l.max(1));

        let mut node = Node {
            id: node_id,
            vector,
            neighbors: (0..=node_layer as usize).map(|_| Vec::new()).collect(),
            max_layer: node_layer,
        };

        if let Some(ep) = self.entry_point {
            // Greedy descent from top layer to node_layer + 1
            let mut current_ep = ep;
            for layer in (node_layer as usize + 1..=self.max_level as usize).rev() {
                current_ep = self.greedy_closest(current_ep, &node.vector, layer);
            }

            // Search and connect at each layer from node_layer down to 0
            for layer in (0..=node_layer as usize).rev() {
                let m_max = if layer == 0 {
                    self.config.m_max_0
                } else {
                    self.config.m
                };

                let candidates = self.search_layer(
                    current_ep,
                    &node.vector,
                    self.config.ef_construction,
                    layer,
                );

                // Select M best neighbors (simple heuristic)
                let selected: Vec<(usize, FixedScalar)> = candidates
                    .into_iter()
                    .take(m_max)
                    .collect();

                // Add bidirectional connections
                node.neighbors[layer] = selected.clone();
                for &(neighbor_id, dist) in &selected {
                    // Skip if the neighbor doesn't participate in this layer
                    if layer >= self.nodes[neighbor_id].neighbors.len() {
                        continue;
                    }
                    self.nodes[neighbor_id].neighbors[layer].push((node_id, dist));
                    // Prune if exceeded M
                    if self.nodes[neighbor_id].neighbors[layer].len() > m_max {
                        self.nodes[neighbor_id].neighbors[layer]
                            .sort_by_key(|&(_, d)| d.to_bits());
                        self.nodes[neighbor_id].neighbors[layer].truncate(m_max);
                    }
                }

                if !selected.is_empty() {
                    current_ep = selected[0].0;
                }
            }

            if node_layer > self.max_level {
                self.max_level = node_layer;
                self.entry_point = Some(node_id);
            }
        } else {
            // First node insertion
            self.entry_point = Some(node_id);
            self.max_level = node_layer;
        }

        self.nodes.push(node);
    }

    /// Greedy search: find the single closest node at a given layer.
    fn greedy_closest(&self, start: usize, query: &FixedVector, layer: usize) -> usize {
        let mut current = start;
        let mut current_dist = self.nodes[current].vector.squared_euclidean(query);

        loop {
            let mut improved = false;
            let neighbors = if layer < self.nodes[current].neighbors.len() {
                &self.nodes[current].neighbors[layer]
            } else {
                return current;
            };

            for &(neighbor_id, _) in neighbors {
                let dist = self.nodes[neighbor_id].vector.squared_euclidean(query);
                if dist.to_bits() < current_dist.to_bits() {
                    current = neighbor_id;
                    current_dist = dist;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }
        current
    }

    /// ef-bounded search at a given layer.
    /// Returns sorted (closest first) list of (node_id, distance).
    fn search_layer(
        &self,
        start: usize,
        query: &FixedVector,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, FixedScalar)> {
        let start_dist = self.nodes[start].vector.squared_euclidean(query);

        // Min-heap for candidates
        let mut candidates = BinaryHeap::new();
        candidates.push(Reverse(Candidate {
            distance: start_dist.to_bits(),
            node_id: start,
        }));

        // Max-heap for results
        let mut results: Vec<(usize, FixedScalar)> = vec![(start, start_dist)];
        let mut visited = vec![false; self.nodes.len()];
        visited[start] = true;

        while let Some(Reverse(candidate)) = candidates.pop() {
            // Check if we can stop
            let worst_result_dist = results
                .iter()
                .map(|&(_, d)| d.to_bits())
                .max()
                .unwrap_or(i64::MAX);

            if candidate.distance > worst_result_dist && results.len() >= ef {
                break;
            }

            let node = &self.nodes[candidate.node_id];
            let neighbors = if layer < node.neighbors.len() {
                &node.neighbors[layer]
            } else {
                continue;
            };

            for &(neighbor_id, _) in neighbors {
                if visited[neighbor_id] {
                    continue;
                }
                visited[neighbor_id] = true;

                let dist = self.nodes[neighbor_id].vector.squared_euclidean(query);

                let worst_result_dist = results
                    .iter()
                    .map(|&(_, d)| d.to_bits())
                    .max()
                    .unwrap_or(i64::MAX);

                if dist.to_bits() < worst_result_dist || results.len() < ef {
                    candidates.push(Reverse(Candidate {
                        distance: dist.to_bits(),
                        node_id: neighbor_id,
                    }));
                    results.push((neighbor_id, dist));
                    // Keep only ef best
                    results.sort_by_key(|&(id, d)| (d.to_bits(), id));
                    if results.len() > ef {
                        results.truncate(ef);
                    }
                }
            }
        }

        results
    }

    /// K-NN search from the entry point.
    ///
    /// Implements Algorithm 4 from the paper.
    pub fn search(&self, query: &FixedVector, k: usize, ef: usize) -> Vec<(usize, FixedScalar)> {
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        // Phase 1: Greedy descent
        let mut current = ep;
        for layer in (1..=self.max_level as usize).rev() {
            current = self.greedy_closest(current, query, layer);
        }

        // Phase 2: ef-bounded search at layer 0
        let mut results = self.search_layer(current, query, ef.max(k), 0);

        // Return top-k
        results.sort_by_key(|&(id, d)| (d.to_bits(), id));
        results.truncate(k);
        results
    }

    /// Compute SHA-256 hash of the entire index structure.
    ///
    /// This is the core determinism verification mechanism:
    /// Two D-HNSW indices built from the same data with the same
    /// config MUST produce identical SHA-256 hashes, regardless
    /// of the CPU architecture (x86, ARM, RISC-V).
    pub fn sha256_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Hash config
        hasher.update(&(self.config.m as u64).to_le_bytes());
        hasher.update(&(self.config.m_max_0 as u64).to_le_bytes());
        hasher.update(&self.config.seed.to_le_bytes());
        hasher.update(&(self.config.dim as u64).to_le_bytes());

        // Hash all nodes in insertion order
        for node in &self.nodes {
            hasher.update(&(node.id as u64).to_le_bytes());
            hasher.update(&(node.max_layer as u64).to_le_bytes());

            // Hash vector components
            for comp in &node.vector.components {
                hasher.update(&comp.to_bits().to_le_bytes());
            }

            // Hash neighbor lists (sorted for canonical form)
            for layer_neighbors in &node.neighbors {
                hasher.update(&(layer_neighbors.len() as u64).to_le_bytes());
                let mut sorted_neighbors: Vec<_> = layer_neighbors.clone();
                sorted_neighbors.sort_by_key(|&(id, d)| (d.to_bits(), id));
                for &(nid, dist) in &sorted_neighbors {
                    hasher.update(&(nid as u64).to_le_bytes());
                    hasher.update(&dist.to_bits().to_le_bytes());
                }
            }
        }

        format!("{:x}", hasher.finalize())
    }

    /// Number of nodes in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dim: usize) -> DhnswConfig {
        DhnswConfig {
            m: 4,
            m_max_0: 8,
            ef_construction: 20,
            seed: 42,
            dim,
        }
    }

    #[test]
    fn test_single_insert() {
        let config = make_config(3);
        let mut index = DhnswIndex::new(config);
        let v = FixedVector::from_f32_slice(&[1.0, 2.0, 3.0]);
        index.insert(v);
        assert_eq!(index.len(), 1);
        assert!(index.entry_point.is_some());
    }

    #[test]
    fn test_search_returns_correct_k() {
        let config = make_config(2);
        let mut index = DhnswIndex::new(config);

        // Insert 20 points
        for i in 0..20 {
            let x = (i as f32) * 0.1;
            let v = FixedVector::from_f32_slice(&[x, x]);
            index.insert(v);
        }

        let query = FixedVector::from_f32_slice(&[0.5, 0.5]);
        let results = index.search(&query, 5, 20);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_determinism_sha256() {
        // Build two identical indices and verify SHA-256 match
        let build = |seed: u64| {
            let config = DhnswConfig {
                m: 4,
                m_max_0: 8,
                ef_construction: 20,
                seed,
                dim: 3,
            };
            let mut index = DhnswIndex::new(config);
            let vectors = vec![
                vec![0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6],
                vec![0.7, 0.8, 0.9],
                vec![1.0, 1.1, 1.2],
                vec![1.3, 1.4, 1.5],
            ];
            for v in &vectors {
                index.insert(FixedVector::from_f32_slice(v));
            }
            index.sha256_hash()
        };

        let hash1 = build(42);
        let hash2 = build(42);
        assert_eq!(hash1, hash2, "Same config/data must produce identical hashes");

        let hash3 = build(43);
        assert_ne!(hash1, hash3, "Different seeds must produce different hashes");
    }

    #[test]
    fn test_search_nearest_is_correct() {
        let config = make_config(2);
        let mut index = DhnswIndex::new(config);

        // Insert distinct points
        index.insert(FixedVector::from_f32_slice(&[0.0, 0.0]));
        index.insert(FixedVector::from_f32_slice(&[10.0, 10.0]));
        index.insert(FixedVector::from_f32_slice(&[0.1, 0.1]));

        // Query near origin
        let query = FixedVector::from_f32_slice(&[0.05, 0.05]);
        let results = index.search(&query, 1, 10);
        assert!(!results.is_empty());
        // Closest should be node 0 (0,0) or node 2 (0.1,0.1)
        let nearest_id = results[0].0;
        assert!(nearest_id == 0 || nearest_id == 2,
                "Nearest should be node 0 or 2, got {}", nearest_id);
    }
}
