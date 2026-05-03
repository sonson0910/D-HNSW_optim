//! Cross-Platform Determinism Verification Benchmark
//!
//! This benchmark verifies the core D-HNSW determinism claim:
//! given identical inputs (vectors + RNG seed), the graph construction
//! and serialization produce **bit-identical** results.
//!
//! This serves both as:
//! 1. A CI test (pass/fail on hash mismatch)
//! 2. A Criterion benchmark (measuring hash computation overhead)
//!
//! # Running
//! ```shell
//! cargo bench --bench cross_platform_determinism --features benchmarks
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

use dhnsw::deterministic_rng::DeterministicRng;
use dhnsw::fixed_point::{FixedPointVector, I64F32};
use dhnsw::graph::HnswGraph;

/// Generate reproducible fixed-point vectors
fn generate_vectors<const D: usize>(count: usize, seed: u64) -> Vec<FixedPointVector<D>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut components = [I64F32::ZERO; D];
            for c in components.iter_mut() {
                *c = I64F32::from_num(rng.gen_range(-1.0f64..1.0));
            }
            FixedPointVector::new(components)
        })
        .collect()
}

/// Build graph and return serialized bytes + SHA-256 hash
fn build_and_hash<const D: usize>(
    vectors: &[FixedPointVector<D>],
    rng_seed: [u8; 32],
) -> (Vec<u8>, String) {
    let mut graph = HnswGraph::<D>::new();
    let mut rng = DeterministicRng::from_seed(rng_seed);

    for v in vectors {
        graph.insert(v.clone(), &mut rng).unwrap();
    }

    let bytes = graph.serialize().unwrap();
    let hash = Sha256::digest(&bytes);
    let hash_hex = format!("{:x}", hash);

    (bytes, hash_hex)
}

fn bench_determinism_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("determinism_verification");
    group.sample_size(10);

    let vectors: Vec<FixedPointVector<128>> = generate_vectors(500, 42);
    let rng_seed = [42u8; 32];

    // First: verify determinism by building twice and comparing
    let (bytes1, hash1) = build_and_hash(&vectors, rng_seed);
    let (bytes2, hash2) = build_and_hash(&vectors, rng_seed);

    assert_eq!(
        hash1, hash2,
        "DETERMINISM VIOLATION: Two identical builds produced different hashes!\n\
         Hash 1: {}\n\
         Hash 2: {}\n\
         Bytes differ at first position: {:?}",
        hash1,
        hash2,
        bytes1
            .iter()
            .zip(bytes2.iter())
            .position(|(a, b)| a != b)
    );

    eprintln!("\n✓ Determinism verified: SHA-256 = {}", hash1);
    eprintln!("  Graph size: {} bytes ({} nodes, D=128)", bytes1.len(), 500);

    // Benchmark: measure overhead of build + hash
    group.bench_function("build_and_hash_500_d128", |b| {
        b.iter(|| {
            let (_, hash) = build_and_hash(black_box(&vectors), rng_seed);
            black_box(hash);
        });
    });

    // Run 5 independent builds and verify all hashes match (as claimed in paper)
    group.bench_function("verify_5_runs", |b| {
        b.iter(|| {
            let mut hashes = Vec::new();
            for _ in 0..5 {
                let (_, hash) = build_and_hash(black_box(&vectors), rng_seed);
                hashes.push(hash);
            }
            // All hashes must be identical
            for h in &hashes[1..] {
                assert_eq!(&hashes[0], h, "Determinism violation across runs");
            }
            black_box(hashes);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_determinism_verification);
criterion_main!(benches);
