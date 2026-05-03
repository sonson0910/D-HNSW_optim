//! Batch Insertion Benchmark
//!
//! Verifies that batch_insert produces identical graph topology
//! regardless of input vector ordering, and measures throughput.

use criterion::{criterion_group, criterion_main, Criterion};
use dhnsw::{DeterministicRng, FixedPointVector, HnswGraph, batch_insert};
use dhnsw::batch::graph_hash;
use rand::SeedableRng;
use rand::Rng;
use sha2::{Sha256, Digest};

/// Generate random vectors for testing.
fn generate_vectors<const D: usize>(n: usize, seed: u64) -> Vec<FixedPointVector<D>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let vals: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();
            FixedPointVector::from_f32_slice(&vals).unwrap()
        })
        .collect()
}

fn batch_insertion_benchmark(c: &mut Criterion) {
    let sep = "=".repeat(70);
    println!("\n{}", sep);
    println!("  D-HNSW Deterministic Batch Insertion Benchmark");
    println!("{}\n", sep);

    let n = 200;
    let vectors = generate_vectors::<128>(n, 42);

    // Test 1: Order independence
    {
        // Forward order
        let mut graph1: HnswGraph<128> = HnswGraph::new();
        let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
        batch_insert(&mut graph1, vectors.clone(), &mut rng1).unwrap();
        let hash1 = graph_hash(&graph1).unwrap();

        // Reverse order
        let mut reversed = vectors.clone();
        reversed.reverse();
        let mut graph2: HnswGraph<128> = HnswGraph::new();
        let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
        batch_insert(&mut graph2, reversed, &mut rng2).unwrap();
        let hash2 = graph_hash(&graph2).unwrap();

        // Shuffled order
        let mut shuffled = vectors.clone();
        let mut shuffle_rng = rand::rngs::StdRng::seed_from_u64(99999);
        for i in (1..shuffled.len()).rev() {
            let j = shuffle_rng.gen_range(0..=i);
            shuffled.swap(i, j);
        }
        let mut graph3: HnswGraph<128> = HnswGraph::new();
        let mut rng3 = DeterministicRng::from_seed([42u8; 32]);
        batch_insert(&mut graph3, shuffled, &mut rng3).unwrap();
        let hash3 = graph_hash(&graph3).unwrap();

        println!("Order Independence Test (N={}, D=128):", n);
        println!("  Forward hash:  {}", hex::encode(&hash1[..8]));
        println!("  Reverse hash:  {}", hex::encode(&hash2[..8]));
        println!("  Shuffled hash: {}", hex::encode(&hash3[..8]));
        println!("  All match: {}", if hash1 == hash2 && hash2 == hash3 { "YES ✓" } else { "NO ✗" });
        println!();
    }

    // Test 2: Sequential vs Batch
    {
        let mut graph_seq: HnswGraph<128> = HnswGraph::new();
        let mut rng_seq = DeterministicRng::from_seed([42u8; 32]);
        // Sort vectors by hash manually for sequential
        let mut hashed: Vec<([u8; 32], FixedPointVector<128>)> = vectors
            .iter()
            .map(|v| {
                let mut hasher = Sha256::new();
                for c in &v.components {
                    hasher.update(c.to_bits().to_le_bytes());
                }
                let h = hasher.finalize();
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&h);
                (hash, v.clone())
            })
            .collect();
        hashed.sort_by(|a, b| a.0.cmp(&b.0));
        for (_, v) in &hashed {
            graph_seq.insert(v.clone(), &mut rng_seq).unwrap();
        }
        let hash_seq = graph_hash(&graph_seq).unwrap();

        let mut graph_batch: HnswGraph<128> = HnswGraph::new();
        let mut rng_batch = DeterministicRng::from_seed([42u8; 32]);
        batch_insert(&mut graph_batch, vectors.clone(), &mut rng_batch).unwrap();
        let hash_batch = graph_hash(&graph_batch).unwrap();

        println!("Sequential vs Batch Equivalence:");
        println!("  Sequential hash: {}", hex::encode(&hash_seq[..8]));
        println!("  Batch hash:      {}", hex::encode(&hash_batch[..8]));
        println!("  Match: {}", if hash_seq == hash_batch { "YES ✓" } else { "NO ✗" });
        println!();
    }

    // Criterion benchmarks
    let mut group = c.benchmark_group("batch_insertion");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    group.bench_function("sequential_128d_200", |b| {
        b.iter(|| {
            let mut graph: HnswGraph<128> = HnswGraph::new();
            let mut rng = DeterministicRng::from_seed([42u8; 32]);
            for v in &vectors {
                graph.insert(v.clone(), &mut rng).unwrap();
            }
        })
    });

    group.bench_function("batch_128d_200", |b| {
        b.iter(|| {
            let mut graph: HnswGraph<128> = HnswGraph::new();
            let mut rng = DeterministicRng::from_seed([42u8; 32]);
            batch_insert(&mut graph, vectors.clone(), &mut rng).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, batch_insertion_benchmark);
criterion_main!(benches);
