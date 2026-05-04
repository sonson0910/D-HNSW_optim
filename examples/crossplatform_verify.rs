/// Cross-Platform Determinism Verification (Paper Table 4)
///
/// Reproduces Table 4 from the D-HNSW IEEE TKDE paper:
/// - Builds graph with SIFT-like 128-d vectors (N=1000, M=16, ef=64)
/// - Hashes (1) graph structure, (2) search result indices, (3) search distance values
/// - Reports SHA-256 hashes that should be IDENTICAL across x86-64, ARM64, RISC-V
///
/// Usage:
///   cargo run --release --example crossplatform_verify            # ARM64 native
///   arch -x86_64 cargo run --release --target x86_64-apple-darwin --example crossplatform_verify  # x86-64 via Rosetta 2

use sha2::{Sha256, Digest};
use dhnsw::{HnswGraph, DeterministicRng};
use dhnsw::fixed_point::FixedPointVector;

const N: usize = 1000;
const DIM: usize = 128;
const K: usize = 10;
const EF_SEARCH: usize = 64;
// Use paper's exact seed: 0xDEADBEEF
const SEED: [u8; 32] = [
    0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Generate deterministic SIFT-like vectors: 128-d, components in [0, 255]
/// Uses a simple LCG seeded deterministically so vectors are identical on all platforms.
fn generate_sift_like_vectors(n: usize) -> Vec<[f32; DIM]> {
    let mut vectors = Vec::with_capacity(n);
    let mut state: u64 = 0x123456789ABCDEF0;
    for _ in 0..n {
        let mut components = [0.0f32; DIM];
        for c in components.iter_mut() {
            // LCG: state = state * 6364136223846793005 + 1442695040888963407
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *c = ((state >> 33) % 256) as f32;
        }
        vectors.push(components);
    }
    vectors
}

/// Generate deterministic query vectors (different seed region)
fn generate_queries(n: usize) -> Vec<[f32; DIM]> {
    let mut vectors = Vec::with_capacity(n);
    let mut state: u64 = 0xFEDCBA9876543210;
    for _ in 0..n {
        let mut components = [0.0f32; DIM];
        for c in components.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *c = ((state >> 33) % 256) as f32;
        }
        vectors.push(components);
    }
    vectors
}

fn main() {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  D-HNSW Cross-Platform Determinism Verification (Table 4)      ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Platform: {:<52} ║", format!("{} / {}", arch, os));
    println!("║  Config:   N={}, D={}, M=16, ef={}, k={}, seed=0xDEADBEEF  ║", N, DIM, EF_SEARCH, K);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. Generate vectors ──
    println!("[1/4] Generating {} SIFT-like 128-d vectors...", N);
    let base_vecs = generate_sift_like_vectors(N);
    let queries = generate_queries(100);

    // Verify vector generation is deterministic
    let mut vec_hasher = Sha256::new();
    for v in &base_vecs {
        for &c in v.iter() {
            vec_hasher.update(c.to_le_bytes());
        }
    }
    let vec_hash = format!("{:x}", vec_hasher.finalize());
    println!("  Vector data hash: {}", &vec_hash[..16]);

    // ── 2. Build D-HNSW graph ──
    println!("\n[2/4] Building D-HNSW graph...");
    let mut graph = HnswGraph::<DIM>::new();
    let mut rng = DeterministicRng::from_seed(SEED);

    for v in &base_vecs {
        let fv = FixedPointVector::<DIM>::from_f32_slice(v).unwrap();
        graph.insert(fv, &mut rng).unwrap();
    }
    println!("  Graph: {} nodes, max_level={}", graph.len(), graph.max_level());

    // ── 3. Compute Graph Structure Hash ──
    println!("\n[3/4] Computing SHA-256 hashes...");
    
    // Hash 1: Serialized graph structure
    let graph_bytes = graph.serialize().unwrap();
    let graph_hash = format!("{:x}", Sha256::digest(&graph_bytes));
    
    // ── 4. Search and compute result hashes ──
    println!("[4/4] Running {} queries (k={}, ef={})...", queries.len(), K, EF_SEARCH);
    
    let mut index_hasher = Sha256::new();
    let mut dist_hasher = Sha256::new();
    
    for q in &queries {
        let fq = FixedPointVector::<DIM>::from_f32_slice(q).unwrap();
        let results = graph.search(&fq, K, EF_SEARCH).unwrap();
        
        for &(id, ref dist) in &results {
            // Hash 2: Result indices
            index_hasher.update((id as u64).to_le_bytes());
            // Hash 3: Distance values (raw I64F32 bits)
            dist_hasher.update(dist.to_bits().to_le_bytes());
        }
    }
    
    let index_hash = format!("{:x}", index_hasher.finalize());
    let dist_hash = format!("{:x}", dist_hasher.finalize());

    // ── 5. Report ──
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    VERIFICATION RESULTS                         ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Architecture: {:<49} ║", arch);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Graph Hash  : {}  ║", &graph_hash[..48]);
    println!("║  Index Hash  : {}  ║", &index_hash[..48]);
    println!("║  Dist Hash   : {}  ║", &dist_hash[..48]);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Full hashes (for Table 4):                                     ║");
    println!("║    Graph: {} ║", &graph_hash);
    println!("║    Index: {} ║", &index_hash);
    println!("║    Dist:  {} ║", &dist_hash);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    
    // ── 6. Determinism self-check (run 3 times) ──
    println!();
    println!("── Determinism self-check (3 independent builds) ──");
    let mut all_pass = true;
    for run in 2..=3 {
        let mut g2 = HnswGraph::<DIM>::new();
        let mut r2 = DeterministicRng::from_seed(SEED);
        for v in &base_vecs {
            let fv = FixedPointVector::<DIM>::from_f32_slice(v).unwrap();
            g2.insert(fv, &mut r2).unwrap();
        }
        let g2_bytes = g2.serialize().unwrap();
        let g2_hash = format!("{:x}", Sha256::digest(&g2_bytes));
        let matches = g2_hash == graph_hash;
        if !matches { all_pass = false; }
        println!("  Run {}: graph_hash={}... {}", run, &g2_hash[..16],
            if matches { "✅" } else { "❌" });
    }
    
    println!();
    if all_pass {
        println!("✅ ALL CHECKS PASSED on {}", arch);
        println!();
        println!("To verify cross-platform, run this SAME binary on different architectures");
        println!("and compare the three hashes above. They MUST be identical.");
    } else {
        println!("❌ DETERMINISM FAILURE on {}", arch);
        std::process::exit(1);
    }
}
