//! # D-HNSW — Deterministic Hierarchical Navigable Small World Graphs
//!
//! A consensus-safe approximate nearest neighbor (ANN) search library designed
//! for blockchain and verifiable computation environments.
//!
//! ## Key Design Decisions
//!
//! 1. **Fixed-Point Arithmetic**: All distance calculations use Q32.32 (I64F32)
//!    instead of IEEE 754 floats, ensuring bit-identical results across x86_64,
//!    ARM64, and RISC-V.
//!
//! 2. **Deterministic RNG**: Level assignment uses a Keccak-256 seeded PRNG
//!    (no system entropy), producing identical graph topology from identical seeds.
//!
//! 3. **Canonical Ordering**: Insertions are strictly sequential, matching
//!    blockchain transaction ordering guarantees.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │  Public API: HnswGraph<D>                    │
//! │  insert(), search(), serialize()             │
//! ├──────────────────────────────────────────────┤
//! │  Graph Layer (graph.rs)                      │
//! │  - Beam search, greedy descent               │
//! │  - Bidirectional edges, pruning              │
//! │  - Two-phase search (O2)                     │
//! ├──────────────┬───────────────────────────────┤
//! │ Fixed-Point  │  SIMD Distance (simd.rs)      │
//! │ (I64F32)     │  AVX2 / scalar fallback       │
//! ├──────────────┼───────────────────────────────┤
//! │ Keccak RNG   │  Error types                  │
//! │ (det_rng.rs) │  (error.rs)                   │
//! └──────────────┴───────────────────────────────┘
//! ```

pub mod error;
pub mod fixed_point;
pub mod deterministic_rng;
pub mod graph;
pub mod simd;
pub mod quantization;
pub mod batch;

// Re-export primary types for convenience
pub use error::{HnswError, Result};
pub use fixed_point::FixedPointVector;
pub use graph::HnswGraph;
pub use deterministic_rng::DeterministicRng;
pub use quantization::{QuantizationParams, ScalarQuantizedVector};
pub use batch::batch_insert;

// ─── HNSW Configuration Constants ───────────────────────────────────────────
//
// These match the standard HNSW parameterization from Malkov & Yashunin (2018).
// They are `pub` so that `graph.rs` and `deterministic_rng.rs` can reference
// them via `crate::M`, `crate::M0`, etc.

/// Maximum number of bidirectional connections per node at layers ≥ 1.
///
/// M = 16 is the standard choice balancing search quality and memory.
/// Each node stores at most M neighbors per layer (except layer 0).
pub const M: usize = 16;

/// Maximum connections at layer 0 (double M per HNSW paper recommendation).
///
/// Layer 0 is the densest layer containing all nodes, so higher connectivity
/// improves recall with acceptable memory cost.
pub const M0: usize = 2 * M; // 32

/// Search expansion factor during construction.
///
/// Controls how many candidates are evaluated when connecting a new node.
/// Higher values improve graph quality at the cost of slower construction.
pub const EF_CONSTRUCTION: usize = 200;

/// Level multiplier for the geometric distribution of node levels.
///
/// `ml = 1 / ln(M)` ensures the expected number of nodes decreases by
/// factor M at each higher level, creating the hierarchical structure.
///
/// For M = 16: ml ≈ 1 / ln(16) ≈ 0.3607
#[inline]
pub fn ml() -> f64 {
    1.0 / (M as f64).ln()
}
