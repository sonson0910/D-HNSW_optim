//! Paper Data Generator — Generates REAL experimental data for IEEE TKDE paper
//!
//! This benchmark produces actual measurements that can be directly used in the paper.
//! It measures:
//! 1. Throughput ratios (HNSW f32 vs D-HNSW) for multiple dimensions
//! 2. Graph Build + Search (exact vs two-phase) throughput
//! 3. Determinism verification (SHA-256 hashes, 5 runs)
//! 4. Fixed-point error analysis (I64F32 vs f32 vs f64 ground truth)
//!
//! # CRITICAL FIX (v2): Error analysis now uses SAME base vectors for all
//! representations. Previous version generated independent random vectors for
//! f32 and I64F32, causing meaningless 113% error.
//!
//! # Running
//! ```shell
//! cargo bench --bench paper_data_generator --features benchmarks 2>&1
//! ```

use std::time::Instant;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

use dhnsw::deterministic_rng::DeterministicRng;
use dhnsw::fixed_point::{FixedPointVector, I64F32};
use dhnsw::graph::HnswGraph;
use dhnsw::simd;

// ─── Config ───────────────────────────────────────────────────────────────
const N_VECTORS: usize = 5000;   // Number of vectors for graph benchmarks (scaled up from 1000)
const N_QUERIES: usize = 200;    // Number of queries
const N_DIST_PAIRS: usize = 200; // Number of vector pairs for distance benchmarks
const EF_SEARCH: usize = 64;     // Search expansion factor (realistic)
const K: usize = 10;             // Top-k neighbors
const N_RUNS: usize = 5;         // Number of repetitions for variance & determinism
const RNG_SEED: u64 = 42;
const HNSW_SEED: [u8; 32] = [42u8; 32];

// ─── Helper: Generate f64 base vectors (ground truth source) ────────────
fn generate_f64_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f64..1.0f64)).collect())
        .collect()
}

/// Convert f64 vectors to f32 (same base values, different precision)
fn to_f32_vecs(f64_vecs: &[Vec<f64>]) -> Vec<Vec<f32>> {
    f64_vecs.iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect()
}

/// Convert f64 vectors to I64F32 slices (same base values, fixed-point)
fn to_fixed_vecs(f64_vecs: &[Vec<f64>]) -> Vec<Vec<I64F32>> {
    f64_vecs.iter()
        .map(|v| v.iter().map(|&x| I64F32::from_num(x)).collect())
        .collect()
}

fn f32_squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn f64_squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ─── Benchmark: Distance Computation Throughput by Dimension ─────────────
fn bench_distance_throughput(dim: usize) -> (f64, f64, f64) {
    // Generate from SAME base f64 vectors
    let f64_vecs = generate_f64_vectors(N_DIST_PAIRS * 2, dim, RNG_SEED);
    let f32_vecs = to_f32_vecs(&f64_vecs);
    let fixed_vecs = to_fixed_vecs(&f64_vecs);

    // Warmup (3 iterations)
    for _ in 0..3 {
        for i in 0..N_DIST_PAIRS {
            let _ = std::hint::black_box(f32_squared_distance(&f32_vecs[i], &f32_vecs[i + N_DIST_PAIRS]));
            let _ = std::hint::black_box(simd::squared_distance_simd(&fixed_vecs[i], &fixed_vecs[i + N_DIST_PAIRS]));
            let _ = std::hint::black_box(simd::squared_distance_scalar(&fixed_vecs[i], &fixed_vecs[i + N_DIST_PAIRS]));
        }
    }

    // Measure f32
    let start = Instant::now();
    for _ in 0..N_RUNS {
        for i in 0..N_DIST_PAIRS {
            let _ = std::hint::black_box(
                f32_squared_distance(&f32_vecs[i], &f32_vecs[i + N_DIST_PAIRS]),
            );
        }
    }
    let f32_time = start.elapsed().as_nanos() as f64 / (N_RUNS * N_DIST_PAIRS) as f64;

    // Measure I64F32 scalar
    let start = Instant::now();
    for _ in 0..N_RUNS {
        for i in 0..N_DIST_PAIRS {
            let _ = std::hint::black_box(
                simd::squared_distance_scalar(&fixed_vecs[i], &fixed_vecs[i + N_DIST_PAIRS]),
            );
        }
    }
    let scalar_time = start.elapsed().as_nanos() as f64 / (N_RUNS * N_DIST_PAIRS) as f64;

    // Measure I64F32 SIMD (AVX2)
    let start = Instant::now();
    for _ in 0..N_RUNS {
        for i in 0..N_DIST_PAIRS {
            let _ = std::hint::black_box(
                simd::squared_distance_simd(&fixed_vecs[i], &fixed_vecs[i + N_DIST_PAIRS]),
            );
        }
    }
    let simd_time = start.elapsed().as_nanos() as f64 / (N_RUNS * N_DIST_PAIRS) as f64;

    (f32_time, scalar_time, simd_time)
}

// ─── Benchmark: Graph Build + Search Throughput ──────────────────────────
fn bench_graph_throughput<const D: usize>(n_vecs: usize) -> (f64, f64, f64, f64, f64) {
    let mut rng_data = StdRng::seed_from_u64(RNG_SEED);
    let vectors: Vec<FixedPointVector<D>> = (0..n_vecs)
        .map(|_| {
            let mut components = [I64F32::ZERO; D];
            for c in components.iter_mut() {
                *c = I64F32::from_num(rng_data.gen_range(-1.0f64..1.0));
            }
            FixedPointVector::new(components)
        })
        .collect();

    let queries: Vec<FixedPointVector<D>> = {
        let mut rng_q = StdRng::seed_from_u64(99);
        (0..N_QUERIES)
            .map(|_| {
                let mut components = [I64F32::ZERO; D];
                for c in components.iter_mut() {
                    *c = I64F32::from_num(rng_q.gen_range(-1.0f64..1.0));
                }
                FixedPointVector::new(components)
            })
            .collect()
    };

    // Build graph
    let build_start = Instant::now();
    let mut graph = HnswGraph::<D>::new();
    let mut rng = DeterministicRng::from_seed(HNSW_SEED);
    for v in &vectors {
        graph.insert(v.clone(), &mut rng).unwrap();
    }
    let build_time_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    // Search: exact (standard D-HNSW search)
    let search_start = Instant::now();
    for _ in 0..N_RUNS {
        for q in &queries {
            let _ = std::hint::black_box(graph.search(q, K, EF_SEARCH).unwrap());
        }
    }
    let exact_search_us = search_start.elapsed().as_nanos() as f64
        / (N_RUNS * N_QUERIES) as f64
        / 1000.0;

    // Search: two-phase (O2 optimization with f32 pre-filter)
    let tp_start = Instant::now();
    for _ in 0..N_RUNS {
        for q in &queries {
            let _ = std::hint::black_box(graph.search_two_phase(q, K, EF_SEARCH, 2).unwrap());
        }
    }
    let tp_search_us = tp_start.elapsed().as_nanos() as f64
        / (N_RUNS * N_QUERIES) as f64
        / 1000.0;

    let exact_qps = 1e6 / exact_search_us;
    let tp_qps = 1e6 / tp_search_us;

    (build_time_ms, exact_search_us, tp_search_us, exact_qps, tp_qps)
}

// ─── Benchmark: Determinism Verification (SHA-256 hash chain) ────────────
fn verify_determinism<const D: usize>(n_vecs: usize) -> (String, bool) {
    let mut rng_data = StdRng::seed_from_u64(RNG_SEED);
    let vectors: Vec<FixedPointVector<D>> = (0..n_vecs)
        .map(|_| {
            let mut components = [I64F32::ZERO; D];
            for c in components.iter_mut() {
                *c = I64F32::from_num(rng_data.gen_range(-1.0f64..1.0));
            }
            FixedPointVector::new(components)
        })
        .collect();

    let mut hashes = Vec::new();
    for _ in 0..N_RUNS {
        let mut graph = HnswGraph::<D>::new();
        let mut rng = DeterministicRng::from_seed(HNSW_SEED);
        for v in &vectors {
            graph.insert(v.clone(), &mut rng).unwrap();
        }
        let bytes = graph.serialize().unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        hashes.push(hash);
    }

    let all_match = hashes.iter().all(|h| h == &hashes[0]);
    (hashes[0].clone(), all_match)
}

// ─── Benchmark: Error Analysis (CORRECTED v2) ───────────────────────────
//
// CRITICAL FIX: Previous version generated INDEPENDENT random vectors for
// f32 and I64F32 representations, causing meaningless error measurements.
// Now we generate f64 ground truth first, then convert SAME values to both
// f32 and I64F32 — measuring only the representation error.
fn measure_error(dim: usize) -> (f64, f64, f64, f64) {
    let f64_vecs = generate_f64_vectors(100, dim, RNG_SEED);
    let f32_vecs = to_f32_vecs(&f64_vecs);
    let fixed_vecs = to_fixed_vecs(&f64_vecs);

    let mut max_i64_error: f64 = 0.0;
    let mut max_f32_error: f64 = 0.0;
    let mut sum_i64_error: f64 = 0.0;
    let mut sum_f32_error: f64 = 0.0;
    let mut total_comparisons: usize = 0;

    for i in 0..100 {
        for j in (i + 1)..100 {
            // Ground truth: f64 double-precision squared distance
            let gt = f64_squared_distance(&f64_vecs[i], &f64_vecs[j]);

            // f32 result
            let f32_dist = f32_squared_distance(&f32_vecs[i], &f32_vecs[j]) as f64;

            // I64F32 result (SIMD path — bit-identical to scalar)
            let i64_dist = simd::squared_distance_simd(&fixed_vecs[i], &fixed_vecs[j])
                .to_num::<f64>();

            if gt.abs() > 1e-12 {
                let f32_rel = ((f32_dist - gt) / gt).abs();
                let i64_rel = ((i64_dist - gt) / gt).abs();

                if f32_rel > max_f32_error { max_f32_error = f32_rel; }
                if i64_rel > max_i64_error { max_i64_error = i64_rel; }
                sum_f32_error += f32_rel;
                sum_i64_error += i64_rel;
            }
            total_comparisons += 1;
        }
    }

    let avg_i64 = if total_comparisons > 0 { sum_i64_error / total_comparisons as f64 } else { 0.0 };
    let avg_f32 = if total_comparisons > 0 { sum_f32_error / total_comparisons as f64 } else { 0.0 };

    eprintln!("  [d={dim:>4}] {total_comparisons} pairs | I64F32: max={max_i64_error:.2e} avg={avg_i64:.2e} | f32: max={max_f32_error:.2e} avg={avg_f32:.2e}");
    (max_i64_error, max_f32_error, avg_i64, avg_f32)
}

// ─── Main ────────────────────────────────────────────────────────────────
fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  D-HNSW Paper Data Generator v2 — REAL Benchmark Results    ║");
    eprintln!("║  N_VECTORS={N_VECTORS}, N_QUERIES={N_QUERIES}, EF={EF_SEARCH}, K={K}              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════
    // 1. Distance computation throughput by dimension
    // ═══════════════════════════════════════════════════════════════
    eprintln!("═══ [1/4] Distance Computation Throughput ═══");
    eprintln!("{:<8} {:>10} {:>12} {:>12} {:>10} {:>10} {:>10}",
             "Dim", "f32 (ns)", "Scalar (ns)", "SIMD (ns)", "Raw O/H", "SIMD O/H", "Speedup");
    eprintln!("{}", "-".repeat(82));

    for &dim in &[96, 100, 128, 256, 512, 784, 960] {
        let (f32_ns, scalar_ns, simd_ns) = bench_distance_throughput(dim);
        let raw_overhead = scalar_ns / f32_ns;
        let simd_overhead = simd_ns / f32_ns;
        let simd_speedup = scalar_ns / simd_ns;
        eprintln!("{:<8} {:>10.1} {:>12.1} {:>12.1} {:>10.2}x {:>10.2}x {:>10.1}x",
                 dim, f32_ns, scalar_ns, simd_ns, raw_overhead, simd_overhead, simd_speedup);
    }

    // ═══════════════════════════════════════════════════════════════
    // 2. Graph build + search throughput
    // ═══════════════════════════════════════════════════════════════
    eprintln!("\n═══ [2/4] Graph Build + Search Throughput (N={N_VECTORS}) ═══");
    eprintln!("{:<8} {:>12} {:>14} {:>14} {:>10} {:>10}",
             "Dim", "Build (ms)", "Exact (μs/q)", "TwoPhase(μs)", "ExactQPS", "TP_QPS");
    eprintln!("{}", "-".repeat(76));

    // D=96
    {
        let (build, exact, tp, eq, tq) = bench_graph_throughput::<96>(N_VECTORS);
        eprintln!("{:<8} {:>12.1} {:>14.2} {:>14.2} {:>10.0} {:>10.0}", "96", build, exact, tp, eq, tq);
    }
    // D=128
    {
        let (build, exact, tp, eq, tq) = bench_graph_throughput::<128>(N_VECTORS);
        eprintln!("{:<8} {:>12.1} {:>14.2} {:>14.2} {:>10.0} {:>10.0}", "128", build, exact, tp, eq, tq);
    }

    // ═══════════════════════════════════════════════════════════════
    // 3. Determinism verification (SHA-256, 5 runs each)
    // ═══════════════════════════════════════════════════════════════
    eprintln!("\n═══ [3/4] Determinism Verification (SHA-256, {N_RUNS} runs) ═══");
    {
        let (hash, pass) = verify_determinism::<96>(1000);
        eprintln!("  D=96,  N=1000: hash={} PASS={}", &hash[..16], pass);
    }
    {
        let (hash, pass) = verify_determinism::<128>(1000);
        eprintln!("  D=128, N=1000: hash={} PASS={}", &hash[..16], pass);
    }
    {
        let (hash, pass) = verify_determinism::<128>(N_VECTORS);
        eprintln!("  D=128, N={N_VECTORS}: hash={} PASS={}", &hash[..16], pass);
    }

    // ═══════════════════════════════════════════════════════════════
    // 4. Error analysis (CORRECTED — same base vectors for all representations)
    // ═══════════════════════════════════════════════════════════════
    eprintln!("\n═══ [4/4] Error Analysis (I64F32 vs f32 vs f64 ground truth) ═══");
    eprintln!("  NOTE: All representations derived from SAME f64 base vectors.");
    eprintln!("{:<8} {:>15} {:>15} {:>15} {:>15}",
             "Dim", "I64F32 MaxErr", "f32 MaxErr", "I64F32 AvgErr", "f32 AvgErr");
    eprintln!("{}", "-".repeat(72));
    for &dim in &[96, 100, 128, 256, 512, 784, 960] {
        let (i64_max, f32_max, i64_avg, f32_avg) = measure_error(dim);
        eprintln!("{:<8} {:>15.2e} {:>15.2e} {:>15.2e} {:>15.2e}",
                 dim, i64_max, f32_max, i64_avg, f32_avg);
    }

    eprintln!("\n✅ All measurements complete. Ready for paper tables.");
    eprintln!("   Benchmark config: N_VECTORS={N_VECTORS}, N_QUERIES={N_QUERIES}, EF={EF_SEARCH}, K={K}, N_RUNS={N_RUNS}");
}
