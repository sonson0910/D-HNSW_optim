//! LLM Embedding Benchmark
//!
//! Evaluates D-HNSW performance on high-dimensional vectors typical of
//! modern LLM embeddings:
//! - 768-d: BERT, sentence-transformers, E5
//! - 1536-d: OpenAI text-embedding-3-small
//!
//! Measures: build time, QPS, recall@10, determinism verification, memory.

use criterion::{criterion_group, criterion_main, Criterion};
use dhnsw::{DeterministicRng, FixedPointVector, HnswGraph};
use rand::SeedableRng;
use rand::Rng;
use sha2::{Sha256, Digest};
use std::time::Instant;

/// Generate normalized random vectors simulating LLM embeddings.
fn generate_llm_vectors<const D: usize>(n: usize, seed: u64) -> Vec<FixedPointVector<D>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let vals: Vec<f32> = (0..D).map(|_| rng.gen_range(-0.5..0.5)).collect();
            // Normalize to unit length (simulating real embeddings)
            let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
            let normalized: Vec<f32> = if norm > 0.0 {
                vals.iter().map(|v| v / norm).collect()
            } else {
                vals
            };
            FixedPointVector::from_f32_slice(&normalized).unwrap()
        })
        .collect()
}

/// Compute graph SHA-256 hash for determinism verification.
fn graph_sha256<const D: usize>(graph: &HnswGraph<D>) -> [u8; 32] {
    let serialized = graph.serialize().unwrap();
    let mut hasher = Sha256::new();
    hasher.update(&serialized);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Run a complete evaluation for a given dimension.
fn evaluate_dimension<const D: usize>(
    dim_name: &str,
    n_build: usize,
    n_queries: usize,
    k: usize,
    ef_search: usize,
) {
    let dash = "-".repeat(60);
    println!("\n{}", dash);
    println!("  {}: D={}, N={}, queries={}", dim_name, D, n_build, n_queries);
    println!("{}", dash);

    let seed = 42u64;
    let vectors = generate_llm_vectors::<D>(n_build, seed);
    let queries = generate_llm_vectors::<D>(n_queries, seed + 1000);

    // Build graph
    let build_start = Instant::now();
    let mut graph = HnswGraph::<D>::new();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    for vec in &vectors {
        graph.insert(vec.clone(), &mut rng).unwrap();
    }
    let build_time = build_start.elapsed();

    // Determinism verification: build again, compare hashes
    let mut graph2 = HnswGraph::<D>::new();
    let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
    for vec in &vectors {
        graph2.insert(vec.clone(), &mut rng2).unwrap();
    }
    let hash1 = graph_sha256(&graph);
    let hash2 = graph_sha256(&graph2);
    let deterministic = hash1 == hash2;

    // Search performance
    let search_start = Instant::now();
    let mut _total_results = 0usize;
    for query in &queries {
        let results = graph.search(query, k, ef_search).unwrap();
        _total_results += results.len();
    }
    let search_time = search_start.elapsed();
    let qps = n_queries as f64 / search_time.as_secs_f64();

    // Recall: compare with brute-force on small subset
    let recall_queries: Vec<&FixedPointVector<D>> = queries.iter().take(50).collect();
    let mut recall_sum = 0.0f64;
    for query in &recall_queries {
        let ann_results = graph.search(query, k, ef_search).unwrap();
        let ann_ids: std::collections::HashSet<usize> = ann_results.iter().map(|r| r.0).collect();

        // Brute force
        let mut bf_dists: Vec<(usize, _)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, query.squared_distance(v)))
            .collect();
        bf_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bf_ids: std::collections::HashSet<usize> = bf_dists.iter().take(k).map(|r| r.0).collect();

        let intersection = ann_ids.intersection(&bf_ids).count();
        recall_sum += intersection as f64 / k as f64;
    }
    let recall = recall_sum / recall_queries.len() as f64;

    // Memory estimation
    let serialized_size = graph.serialize().unwrap().len();

    println!("  Build time:     {:?}", build_time);
    println!("  Build throughput: {:.1} vectors/sec", n_build as f64 / build_time.as_secs_f64());
    println!("  Search QPS:     {:.1}", qps);
    println!("  Recall@{}:      {:.4}", k, recall);
    println!("  Deterministic:  {}", if deterministic { "YES ✓" } else { "NO ✗" });
    println!("  Graph SHA-256:  {}", hex::encode(&hash1[..8]));
    println!("  Memory (ser.):  {:.2} MB", serialized_size as f64 / (1024.0 * 1024.0));
    println!("  Bytes/vector:   {:.0}", serialized_size as f64 / n_build as f64);
}

fn llm_embedding_benchmark(c: &mut Criterion) {
    let sep = "=".repeat(70);
    println!("\n{}", sep);
    println!("  D-HNSW LLM Embedding Evaluation");
    println!("  Testing 768-d (BERT) and 1536-d (OpenAI) dimensions");
    println!("{}", sep);

    // 768-d evaluation
    evaluate_dimension::<768>("BERT/E5 Embeddings", 1000, 100, 10, 64);

    // 1536-d evaluation
    evaluate_dimension::<1536>("OpenAI text-embedding-3", 1000, 100, 10, 64);

    // Criterion benchmark for 768-d
    let mut group = c.benchmark_group("llm_embeddings");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));

    {
        let vectors_768 = generate_llm_vectors::<768>(200, 42);
        group.bench_function("768d_build_200", |b| {
            b.iter(|| {
                let mut graph = HnswGraph::<768>::new();
                let mut rng = DeterministicRng::from_seed([42u8; 32]);
                for vec in &vectors_768 {
                    graph.insert(vec.clone(), &mut rng).unwrap();
                }
            })
        });
    }

    {
        let vectors_1536 = generate_llm_vectors::<1536>(200, 42);
        group.bench_function("1536d_build_200", |b| {
            b.iter(|| {
                let mut graph = HnswGraph::<1536>::new();
                let mut rng = DeterministicRng::from_seed([42u8; 32]);
                for vec in &vectors_1536 {
                    graph.insert(vec.clone(), &mut rng).unwrap();
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, llm_embedding_benchmark);
criterion_main!(benches);
