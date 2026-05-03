//! Ablation Benchmark Suite for D-HNSW
//!
//! Measures the performance of each optimization layer:
//! - V0: Baseline f32 HNSW (simulated via f32 distance)
//! - V1: Naïve I64F32 (fixed-point, no optimizations)
//! - V2: I64F32 + SIMD distance computation (O1)
//! - V3: I64F32 + Two-Phase search (O2)
//! - V4: I64F32 + SIMD + Two-Phase (O1+O2)
//!
//! # Running
//! ```shell
//! cargo bench --bench ablation_benchmark --features benchmarks
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use dhnsw::deterministic_rng::DeterministicRng;
use dhnsw::fixed_point::{FixedPointVector, I64F32};
use dhnsw::graph::HnswGraph;
use dhnsw::simd;

// ─── Random Vector Generators ─────────────────────────────────────────────

/// Generate random f32 vectors for baseline comparison
fn generate_f32_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Generate random FixedPointVector for D-HNSW benchmarks
fn generate_fixed_vectors<const D: usize>(count: usize, seed: u64) -> Vec<FixedPointVector<D>> {
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

/// Build an HNSW graph from fixed-point vectors
fn build_graph<const D: usize>(vectors: &[FixedPointVector<D>]) -> HnswGraph<D> {
    let mut graph = HnswGraph::new();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);

    for v in vectors {
        graph.insert(v.clone(), &mut rng).unwrap();
    }

    graph
}

// ─── Baseline: f32 Squared Distance ───────────────────────────────────────

fn f32_squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ─── Benchmark Groups ─────────────────────────────────────────────────────

fn bench_distance_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_computation");
    group.sample_size(100);

    for &dim in &[96, 128, 784, 960] {
        // Generate test vectors
        let f32_vecs = generate_f32_vectors(2, dim, 42);

        // f32 vectors as I64F32
        let fixed_a: Vec<I64F32> = f32_vecs[0]
            .iter()
            .map(|&v| I64F32::from_num(v))
            .collect();
        let fixed_b: Vec<I64F32> = f32_vecs[1]
            .iter()
            .map(|&v| I64F32::from_num(v))
            .collect();

        // V0: Baseline f32 distance
        group.bench_with_input(
            BenchmarkId::new("V0_f32", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(f32_squared_distance(&f32_vecs[0], &f32_vecs[1]))
                });
            },
        );

        // V1: Naïve I64F32 scalar distance
        group.bench_with_input(
            BenchmarkId::new("V1_I64F32_scalar", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(simd::squared_distance_scalar(&fixed_a, &fixed_b))
                });
            },
        );

        // V2: I64F32 + SIMD distance
        group.bench_with_input(
            BenchmarkId::new("V2_I64F32_SIMD", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(simd::squared_distance_simd(&fixed_a, &fixed_b))
                });
            },
        );

        // V0b: f32 approximate distance (for Two-Phase)
        group.bench_with_input(
            BenchmarkId::new("V3_f32_approx", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(simd::approximate_distance_f32(&fixed_a, &fixed_b))
                });
            },
        );
    }

    group.finish();
}

fn bench_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction");
    group.sample_size(10);

    for &n in &[100, 500, 1000] {
        let vectors: Vec<FixedPointVector<128>> = generate_fixed_vectors(n, 42);

        group.bench_with_input(
            BenchmarkId::new("D-HNSW_build", n),
            &n,
            |b, _| {
                b.iter(|| {
                    black_box(build_graph(&vectors));
                });
            },
        );
    }

    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    group.sample_size(50);

    // Pre-build graph
    let n = 1000;
    let vectors: Vec<FixedPointVector<128>> = generate_fixed_vectors(n, 42);
    let graph = build_graph(&vectors);

    // Generate query vectors
    let queries: Vec<FixedPointVector<128>> = generate_fixed_vectors(10, 99);

    for &ef in &[10, 64, 128, 256] {
        // Standard exact search
        group.bench_with_input(
            BenchmarkId::new("exact_search", ef),
            &ef,
            |b, &ef| {
                b.iter(|| {
                    for q in &queries {
                        black_box(graph.search(q, 10, ef).unwrap());
                    }
                });
            },
        );

        // Two-Phase search (O2)
        group.bench_with_input(
            BenchmarkId::new("two_phase_search", ef),
            &ef,
            |b, &ef| {
                b.iter(|| {
                    for q in &queries {
                        black_box(graph.search_two_phase(q, 10, ef, 2).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_ablation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("ablation_overhead");
    group.sample_size(20);

    // Build a 1000-node graph and measure overhead ratios
    let n = 1000;
    let vectors: Vec<FixedPointVector<128>> = generate_fixed_vectors(n, 42);
    let graph = build_graph(&vectors);
    let queries: Vec<FixedPointVector<128>> = generate_fixed_vectors(50, 99);

    // V1: Naïve D-HNSW search (exact I64F32 only)
    group.bench_function("V1_naive_exact", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(graph.search(q, 10, 128).unwrap());
            }
        });
    });

    // V4: Full pipeline (Two-Phase with SIMD)
    group.bench_function("V4_two_phase_optimized", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(graph.search_two_phase(q, 10, 128, 2).unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_computation,
    bench_graph_construction,
    bench_search,
    bench_ablation_overhead,
);
criterion_main!(benches);
