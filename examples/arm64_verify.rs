/// Cross-platform determinism verification on native ARM64 (M1 Pro).
///
/// Builds 3 identical graphs from the same seed and verifies
/// that the serialized bytes produce identical SHA-256 hashes.
use sha2::{Sha256, Digest};
use dhnsw::{HnswGraph, DeterministicRng};
use dhnsw::fixed_point::FixedPointVector;

fn main() {
    println!("=== D-HNSW ARM64 Native Determinism Verification ===");
    println!("Platform: {} / {}", std::env::consts::ARCH, std::env::consts::OS);
    println!();

    const NUM_RUNS: usize = 5;
    const NUM_VECTORS: usize = 200;
    const DIM: usize = 128;
    const SEED: [u8; 32] = [
        0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x42,
    ];

    let mut hashes: Vec<String> = Vec::new();
    let mut graph_sizes: Vec<usize> = Vec::new();

    for run in 1..=NUM_RUNS {
        let mut graph: HnswGraph<DIM> = HnswGraph::new();
        let mut rng = DeterministicRng::from_seed(SEED);

        // Generate deterministic vectors using the RNG
        for i in 0..NUM_VECTORS {
            let mut components = [0.0f32; DIM];
            for j in 0..DIM {
                // Deterministic pseudo-random values in [0, 255] (SIFT-like)
                let val = ((i * DIM + j) % 256) as f32;
                components[j] = val;
            }
            let vector = FixedPointVector::<DIM>::from_f32_slice(&components).unwrap();
            graph.insert(vector, &mut rng).unwrap();
        }

        // Serialize and hash
        let bytes = graph.serialize().unwrap();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());

        println!("Run {}: SHA-256 = {}...{} ({} bytes)",
            run,
            &hash[..16],
            &hash[hash.len()-8..],
            bytes.len()
        );

        graph_sizes.push(bytes.len());
        hashes.push(hash);
    }

    println!();

    // Verify all hashes are identical
    let all_match = hashes.windows(2).all(|w| w[0] == w[1]);
    let all_sizes_match = graph_sizes.windows(2).all(|w| w[0] == w[1]);

    if all_match && all_sizes_match {
        println!("✅ DETERMINISM VERIFIED: All {} runs produce identical SHA-256 hashes", NUM_RUNS);
        println!("   Hash: {}", &hashes[0]);
        println!("   Size: {} bytes", graph_sizes[0]);
    } else {
        println!("❌ DETERMINISM FAILURE: Hashes differ across runs!");
        for (i, h) in hashes.iter().enumerate() {
            println!("   Run {}: {}", i + 1, h);
        }
        std::process::exit(1);
    }

    // Also verify search determinism
    println!();
    println!("=== Search Determinism Verification ===");

    // Re-build graph once
    let mut graph: HnswGraph<DIM> = HnswGraph::new();
    let mut rng = DeterministicRng::from_seed(SEED);
    for i in 0..NUM_VECTORS {
        let mut components = [0.0f32; DIM];
        for j in 0..DIM {
            components[j] = ((i * DIM + j) % 256) as f32;
        }
        let vector = FixedPointVector::<DIM>::from_f32_slice(&components).unwrap();
        graph.insert(vector, &mut rng).unwrap();
    }

    // Search 10 times with same query
    let mut query_components = [0.0f32; DIM];
    for j in 0..DIM {
        query_components[j] = (j * 3 % 256) as f32;
    }
    let query = FixedPointVector::<DIM>::from_f32_slice(&query_components).unwrap();

    let mut search_hashes: Vec<String> = Vec::new();
    for run in 1..=10 {
        let results = graph.search(&query, 10, 64).unwrap();
        let mut hasher = Sha256::new();
        for (id, dist) in &results {
            hasher.update(id.to_le_bytes());
            hasher.update(dist.to_bits().to_le_bytes());
        }
        let hash = format!("{:x}", hasher.finalize());
        if run <= 3 {
            println!("Search run {}: {} results, hash = {}...{}",
                run, results.len(), &hash[..12], &hash[hash.len()-8..]);
        }
        search_hashes.push(hash);
    }

    let search_match = search_hashes.windows(2).all(|w| w[0] == w[1]);
    if search_match {
        println!("✅ SEARCH DETERMINISM VERIFIED: All 10 searches produce identical results");
    } else {
        println!("❌ SEARCH DETERMINISM FAILURE!");
        std::process::exit(1);
    }

    println!();
    println!("=== ALL VERIFICATIONS PASSED on {} ===", std::env::consts::ARCH);
}
