/// Multi-Dataset Cross-Platform Determinism Verification
///
/// Tests D-HNSW determinism across MULTIPLE datasets with different characteristics:
///   1. SIFT-128   (128-d, integer components [0,255])
///   2. GIST-960   (960-d, continuous float, high-dim)
///   3. Random-96  (96-d, uniform float [0,1])
///   4. Sparse-784 (784-d, sparse with many zeros, Fashion-MNIST-like)
///   5. LLM-1536   (1536-d, normalized embeddings [-1,1], OpenAI-like)
///
/// Each dataset produces INDEPENDENT SHA-256 hashes for graph/index/distances.
/// ALL hashes must be IDENTICAL across x86-64 and ARM64.
///
/// Usage:
///   cargo run --release --example multi_dataset_verify            # ARM64 native
///   arch -x86_64 cargo run --release --target x86_64-apple-darwin --example multi_dataset_verify

use sha2::{Sha256, Digest};
use dhnsw::{HnswGraph, DeterministicRng};
use dhnsw::fixed_point::FixedPointVector;
use std::time::Instant;

const K: usize = 10;
const EF_SEARCH: usize = 64;
const SEED: [u8; 32] = [
    0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Simple deterministic LCG-based vector generator
/// seed_offset differentiates datasets; value_transform controls data distribution
struct VectorGen {
    state: u64,
}

impl VectorGen {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Integer components in [0, 255] — mimics SIFT
    fn gen_sift(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| ((self.next_u64() >> 33) % 256) as f32).collect()
    }

    /// Continuous components in [0, 1] — mimics GIST/random uniform
    fn gen_uniform(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| {
            let v = (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32;
            v.min(1.0).max(0.0)
        }).collect()
    }

    /// Sparse components (80% zeros) — mimics Fashion-MNIST
    fn gen_sparse(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| {
            let r = self.next_u64();
            if (r >> 40) % 5 == 0 {
                // 20% non-zero
                ((r >> 33) % 256) as f32 / 255.0
            } else {
                0.0
            }
        }).collect()
    }

    /// Normalized components in [-1, 1] — mimics LLM embeddings
    fn gen_normalized(&mut self, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| {
            let raw = (self.next_u64() >> 33) as f64 / (1u64 << 31) as f64;
            (raw * 2.0 - 1.0) as f32
        }).collect();
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() { *x /= norm; }
        }
        v
    }
}

struct DatasetConfig {
    name: &'static str,
    n_vectors: usize,
    dim: usize,
    n_queries: usize,
    data_seed: u64,
    query_seed: u64,
    gen_type: GenType,
}

enum GenType {
    Sift,
    Uniform,
    Sparse,
    Normalized,
}

fn generate_dataset(config: &DatasetConfig) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut gen = VectorGen::new(config.data_seed);
    let base: Vec<Vec<f32>> = (0..config.n_vectors).map(|_| {
        match config.gen_type {
            GenType::Sift => gen.gen_sift(config.dim),
            GenType::Uniform => gen.gen_uniform(config.dim),
            GenType::Sparse => gen.gen_sparse(config.dim),
            GenType::Normalized => gen.gen_normalized(config.dim),
        }
    }).collect();

    let mut qgen = VectorGen::new(config.query_seed);
    let queries: Vec<Vec<f32>> = (0..config.n_queries).map(|_| {
        match config.gen_type {
            GenType::Sift => qgen.gen_sift(config.dim),
            GenType::Uniform => qgen.gen_uniform(config.dim),
            GenType::Sparse => qgen.gen_sparse(config.dim),
            GenType::Normalized => qgen.gen_normalized(config.dim),
        }
    }).collect();

    (base, queries)
}

/// Run verification for a given dimension (using const generics)
macro_rules! run_dataset {
    ($config:expr, $DIM:expr) => {{
        let config = $config;
        let start = Instant::now();
        print!("  [{:<12}] D={:<4} N={:<5} ", config.name, config.dim, config.n_vectors);

        let (base_vecs, queries) = generate_dataset(&config);

        // Build graph
        let mut graph = HnswGraph::<$DIM>::new();
        let mut rng = DeterministicRng::from_seed(SEED);

        for v in &base_vecs {
            let arr: [f32; $DIM] = v[..].try_into().unwrap();
            let fv = FixedPointVector::<$DIM>::from_f32_slice(&arr).unwrap();
            graph.insert(fv, &mut rng).unwrap();
        }

        // Hash graph structure
        let graph_bytes = graph.serialize().unwrap();
        let graph_hash = format!("{:x}", Sha256::digest(&graph_bytes));

        // Hash search results
        let mut index_hasher = Sha256::new();
        let mut dist_hasher = Sha256::new();

        for q in &queries {
            let arr: [f32; $DIM] = q[..].try_into().unwrap();
            let fq = FixedPointVector::<$DIM>::from_f32_slice(&arr).unwrap();
            let results = graph.search(&fq, K, EF_SEARCH).unwrap();

            for &(id, ref dist) in &results {
                index_hasher.update((id as u64).to_le_bytes());
                dist_hasher.update(dist.to_bits().to_le_bytes());
            }
        }

        let index_hash = format!("{:x}", index_hasher.finalize());
        let dist_hash = format!("{:x}", dist_hasher.finalize());
        let elapsed = start.elapsed();

        println!("graph={} idx={} dist={} ({:.1}s)",
            &graph_hash[..8], &index_hash[..8], &dist_hash[..8], elapsed.as_secs_f64());

        // Determinism self-check: build again, compare graph hash
        let mut g2 = HnswGraph::<$DIM>::new();
        let mut r2 = DeterministicRng::from_seed(SEED);
        for v in &base_vecs {
            let arr: [f32; $DIM] = v[..].try_into().unwrap();
            let fv = FixedPointVector::<$DIM>::from_f32_slice(&arr).unwrap();
            g2.insert(fv, &mut r2).unwrap();
        }
        let g2_bytes = g2.serialize().unwrap();
        let g2_hash = format!("{:x}", Sha256::digest(&g2_bytes));
        assert_eq!(graph_hash, g2_hash, "DETERMINISM FAILURE on {}", config.name);

        (config.name, graph_hash, index_hash, dist_hash)
    }};
}

fn main() {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  D-HNSW Multi-Dataset Cross-Platform Verification                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Platform: {:<63} ║", format!("{} / {}", arch, os));
    println!("║  Params:   M=16, ef={}, k={}, seed=0xDEADBEEF                             ║", EF_SEARCH, K);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();
    let mut results: Vec<(&str, String, String, String)> = Vec::new();

    // ── Dataset 1: SIFT-128 (integer components [0,255]) ──
    println!("── Running 5 datasets ──");
    let r = run_dataset!(DatasetConfig {
        name: "SIFT-128",
        n_vectors: 1000,
        dim: 128,
        n_queries: 100,
        data_seed: 0x123456789ABCDEF0,
        query_seed: 0xFEDCBA9876543210,
        gen_type: GenType::Sift,
    }, 128);
    results.push(r);

    // ── Dataset 2: Random-96 (uniform float [0,1], low-dim) ──
    let r = run_dataset!(DatasetConfig {
        name: "Random-96",
        n_vectors: 1000,
        dim: 96,
        n_queries: 100,
        data_seed: 0xAAAABBBBCCCCDDDD,
        query_seed: 0x1111222233334444,
        gen_type: GenType::Uniform,
    }, 96);
    results.push(r);

    // ── Dataset 3: Sparse-784 (Fashion-MNIST-like, 80% zeros) ──
    let r = run_dataset!(DatasetConfig {
        name: "Sparse-784",
        n_vectors: 500,
        dim: 784,
        n_queries: 50,
        data_seed: 0x5555666677778888,
        query_seed: 0x9999AAAABBBBCCCC,
        gen_type: GenType::Sparse,
    }, 784);
    results.push(r);

    // ── Dataset 4: GIST-960 (continuous float, high-dim) ──
    let r = run_dataset!(DatasetConfig {
        name: "GIST-960",
        n_vectors: 500,
        dim: 960,
        n_queries: 50,
        data_seed: 0xDDDDEEEEFFFF0000,
        query_seed: 0x0000111122223333,
        gen_type: GenType::Uniform,
    }, 960);
    results.push(r);

    // ── Dataset 5: LLM-1536 (normalized embeddings [-1,1]) ──
    let r = run_dataset!(DatasetConfig {
        name: "LLM-1536",
        n_vectors: 300,
        dim: 1536,
        n_queries: 30,
        data_seed: 0x4444555566667777,
        query_seed: 0x8888999AAAAABBBB,
        gen_type: GenType::Normalized,
    }, 1536);
    results.push(r);

    let total_elapsed = total_start.elapsed();

    // ── Summary Report ──
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                     MULTI-DATASET VERIFICATION SUMMARY                      ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Arch: {:<67}  ║", arch);
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");

    for (name, gh, ih, dh) in &results {
        println!("║  {:<12}: G={} I={} D={} ║", name, &gh[..10], &ih[..10], &dh[..10]);
    }

    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  All 5 datasets: determinism self-check PASSED                             ║");
    println!("║  Total time: {:.1}s                                                        ║", total_elapsed.as_secs_f64());
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    println!();
    println!("── Full hashes for cross-platform comparison ──");
    for (name, gh, ih, dh) in &results {
        println!("{}", name);
        println!("  Graph: {}", gh);
        println!("  Index: {}", ih);
        println!("  Dist:  {}", dh);
    }

    println!();
    println!("✅ {} datasets verified on {} ({:.1}s total)", results.len(), arch, total_elapsed.as_secs_f64());
    println!();
    println!("Compare these hashes with output from other architectures.");
    println!("They MUST be identical — that is the core D-HNSW guarantee.");
}
