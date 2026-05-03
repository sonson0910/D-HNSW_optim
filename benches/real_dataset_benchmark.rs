//! Real Dataset Benchmark — D-HNSW on SIFT-1M
//!
//! Loads actual SIFT-1M vectors (.fvecs format), builds the D-HNSW graph,
//! and measures real performance metrics for the IEEE TKDE paper.
//!
//! Metrics measured:
//! 1. Build time (wall-clock)
//! 2. Search latency (μs/query) and QPS
//! 3. Recall@10 against ground truth
//! 4. Determinism verification (SHA-256 across 3 builds)
//! 5. Distance computation overhead (D-HNSW i64 vs f32 baseline)
//! 6. Numerical error analysis (I64F32 vs f32 vs f64)
//!
//! # Running
//! ```shell
//! cargo bench --bench real_dataset_benchmark --features benchmarks 2>&1
//! ```

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

use dhnsw::deterministic_rng::DeterministicRng;
use dhnsw::fixed_point::{FixedPointVector, I64F32};
use dhnsw::graph::HnswGraph;
use dhnsw::simd;

// ─── Config ───────────────────────────────────────────────────────────────
const EF_SEARCH: usize = 64;
const K: usize = 10;
const HNSW_SEED: [u8; 32] = [42u8; 32];

// ─── .fvecs / .ivecs Parsers ──────────────────────────────────────────────

/// Read .fvecs file: standard ANN benchmark format
/// Format per vector: [dim: u32_le] [dim × f32_le]
fn read_fvecs(path: &Path) -> Vec<Vec<f32>> {
    let file = File::open(path).expect(&format!("Cannot open {}", path.display()));
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        // Read dimension header (4 bytes, little-endian u32)
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break; // EOF
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        // Read dim × f32 values
        let mut data_buf = vec![0u8; dim * 4];
        reader.read_exact(&mut data_buf)
            .expect("Unexpected EOF while reading vector data");

        let vec: Vec<f32> = data_buf
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        vectors.push(vec);
    }

    vectors
}

/// Read .ivecs file (ground truth): [dim: u32_le] [dim × i32_le]
fn read_ivecs(path: &Path) -> Vec<Vec<i32>> {
    let file = File::open(path).expect(&format!("Cannot open {}", path.display()));
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break;
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut data_buf = vec![0u8; dim * 4];
        reader.read_exact(&mut data_buf)
            .expect("Unexpected EOF while reading ivecs data");

        let vec: Vec<i32> = data_buf
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        vectors.push(vec);
    }

    vectors
}

/// Convert f32 slice to FixedPointVector<128>
fn to_fixed_128(v: &[f32]) -> FixedPointVector<128> {
    assert_eq!(v.len(), 128, "Expected 128-dim vector");
    let mut components = [I64F32::ZERO; 128];
    for (i, &val) in v.iter().enumerate() {
        components[i] = I64F32::from_num(val);
    }
    FixedPointVector::new(components)
}

// ─── f32 baseline HNSW (simple brute-force for recall comparison) ────────

fn f32_squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| {
        let d = x - y;
        d * d
    }).sum()
}

/// Brute-force k-NN for ground truth verification
fn brute_force_knn(query: &[f32], base: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut dists: Vec<(usize, f32)> = base.iter().enumerate()
        .map(|(i, v)| (i, f32_squared_distance(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.into_iter().take(k).collect()
}

/// Compute Recall@k: fraction of true k-NN found by approximate search
fn compute_recall(approx_ids: &[usize], gt_ids: &[i32], k: usize) -> f64 {
    let gt_set: std::collections::HashSet<usize> = gt_ids.iter()
        .take(k)
        .map(|&id| id as usize)
        .collect();
    let found = approx_ids.iter().take(k).filter(|id| gt_set.contains(id)).count();
    found as f64 / k as f64
}

// ─── Main Benchmark ──────────────────────────────────────────────────────
fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  D-HNSW Real Dataset Benchmark — SIFT-1M                   ║");
    eprintln!("║  ACTUAL measurements on REAL vectors                       ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── Load SIFT-1M ──
    let data_dir = Path::new("benchmark_data/sift");
    let base_path = data_dir.join("sift_base.fvecs");
    let query_path = data_dir.join("sift_query.fvecs");
    let gt_path = data_dir.join("sift_groundtruth.ivecs");

    if !base_path.exists() {
        eprintln!("ERROR: SIFT-1M dataset not found at {}", base_path.display());
        eprintln!("Download: curl -L -o benchmark_data/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz");
        eprintln!("Extract:  tar -xzf benchmark_data/sift.tar.gz -C benchmark_data/");
        std::process::exit(1);
    }

    eprintln!("Loading SIFT-1M base vectors...");
    let base_vecs = read_fvecs(&base_path);
    eprintln!("  Loaded {} base vectors, dim={}", base_vecs.len(), base_vecs[0].len());

    eprintln!("Loading query vectors...");
    let query_vecs = read_fvecs(&query_path);
    eprintln!("  Loaded {} query vectors", query_vecs.len());

    eprintln!("Loading ground truth...");
    let ground_truth = read_ivecs(&gt_path);
    eprintln!("  Loaded {} ground truth entries, k={}", ground_truth.len(), ground_truth[0].len());

    // ── Run at multiple scales ──
    let scales: Vec<usize> = vec![10_000, 50_000, 100_000, 200_000];
    // Only attempt 1M if we have enough memory and time
    // scales.push(1_000_000);

    for &n in &scales {
        let actual_n = n.min(base_vecs.len());
        run_benchmark_at_scale(&base_vecs, &query_vecs, &ground_truth, actual_n);
    }

    // ── Distance overhead measurement (micro-benchmark on REAL vectors) ──
    eprintln!("\n═══ Distance Computation Overhead (SIFT-128, REAL vectors) ═══");
    measure_distance_overhead(&base_vecs);

    // ── Error analysis on REAL vectors ──
    eprintln!("\n═══ Error Analysis on REAL SIFT vectors ═══");
    measure_error_real(&base_vecs);

    eprintln!("\n✅ All REAL dataset measurements complete.");
}

fn run_benchmark_at_scale(
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<i32>],
    n: usize,
) {
    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  SIFT-1M benchmark @ N={}", n);
    eprintln!("{}", "=".repeat(70));

    let subset = &base_vecs[..n];
    let n_queries = query_vecs.len().min(1000); // Use up to 1000 queries
    let queries = &query_vecs[..n_queries];

    // ── 1. Convert to fixed-point ──
    eprintln!("\n[1/5] Converting {} vectors to I64F32...", n);
    let conv_start = Instant::now();
    let fixed_vectors: Vec<FixedPointVector<128>> = subset.iter()
        .map(|v| to_fixed_128(v))
        .collect();
    let conv_time = conv_start.elapsed();
    eprintln!("  Conversion time: {:.2}s ({:.1} vec/s)",
        conv_time.as_secs_f64(),
        n as f64 / conv_time.as_secs_f64());

    let fixed_queries: Vec<FixedPointVector<128>> = queries.iter()
        .map(|v| to_fixed_128(v))
        .collect();

    // ── 2. Build D-HNSW graph ──
    eprintln!("\n[2/5] Building D-HNSW graph (N={}, D=128, M=16, ef_c=200)...", n);
    let build_start = Instant::now();
    let mut graph = HnswGraph::<128>::new();
    let mut rng = DeterministicRng::from_seed(HNSW_SEED);

    for (i, v) in fixed_vectors.iter().enumerate() {
        graph.insert(v.clone(), &mut rng).unwrap();
        if (i + 1) % 10000 == 0 {
            let elapsed = build_start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            let eta = (n - i - 1) as f64 / rate;
            eprint!("\r  Inserted {}/{} ({:.0} vec/s, ETA {:.0}s)    ", i + 1, n, rate, eta);
        }
    }
    let build_time = build_start.elapsed();
    eprintln!("\n  Build time: {:.2}s ({:.0} vec/s)",
        build_time.as_secs_f64(),
        n as f64 / build_time.as_secs_f64());

    // ── 3. Search performance ──
    eprintln!("\n[3/5] Searching ({} queries, ef={}, k={})...", n_queries, EF_SEARCH, K);

    // Standard D-HNSW search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<usize>> = Vec::with_capacity(n_queries);
    for q in &fixed_queries {
        let results = graph.search(q, K, EF_SEARCH).unwrap();
        all_results.push(results.iter().map(|&(id, _)| id).collect());
    }
    let search_time = search_start.elapsed();
    let search_us = search_time.as_nanos() as f64 / n_queries as f64 / 1000.0;
    let search_qps = 1e6 / search_us;

    // Two-phase search
    let tp_start = Instant::now();
    let mut tp_results: Vec<Vec<usize>> = Vec::with_capacity(n_queries);
    for q in &fixed_queries {
        let results = graph.search_two_phase(q, K, EF_SEARCH, 2).unwrap();
        tp_results.push(results.iter().map(|&(id, _)| id).collect());
    }
    let tp_time = tp_start.elapsed();
    let tp_us = tp_time.as_nanos() as f64 / n_queries as f64 / 1000.0;
    let tp_qps = 1e6 / tp_us;

    eprintln!("  Standard search: {:.1} μs/q ({:.0} QPS)", search_us, search_qps);
    eprintln!("  Two-phase search: {:.1} μs/q ({:.0} QPS)", tp_us, tp_qps);

    // ── 4. Recall@10 ──
    eprintln!("\n[4/5] Computing Recall@{} against ground truth...", K);

    // Only compute recall for queries where GT neighbors are within our subset
    let mut total_recall = 0.0;
    let mut valid_queries = 0;

    for (i, result_ids) in all_results.iter().enumerate() {
        if i >= ground_truth.len() { break; }
        // For subsets smaller than 1M, we need to recompute GT via brute force
        if n < base_vecs.len() {
            // Brute-force GT on our subset
            let bf_knn = brute_force_knn(&query_vecs[i], &base_vecs[..n], K);
            let bf_ids: Vec<i32> = bf_knn.iter().map(|&(id, _)| id as i32).collect();
            total_recall += compute_recall(result_ids, &bf_ids, K);
        } else {
            // Use provided ground truth (only valid at full 1M scale)
            total_recall += compute_recall(result_ids, &ground_truth[i], K);
        }
        valid_queries += 1;
    }

    let avg_recall = if valid_queries > 0 { total_recall / valid_queries as f64 } else { 0.0 };
    eprintln!("  Recall@{}: {:.4} ({:.2}%) over {} queries",
        K, avg_recall, avg_recall * 100.0, valid_queries);

    // Two-phase recall
    let mut tp_total_recall = 0.0;
    for (i, result_ids) in tp_results.iter().enumerate() {
        if i >= ground_truth.len() { break; }
        if n < base_vecs.len() {
            let bf_knn = brute_force_knn(&query_vecs[i], &base_vecs[..n], K);
            let bf_ids: Vec<i32> = bf_knn.iter().map(|&(id, _)| id as i32).collect();
            tp_total_recall += compute_recall(result_ids, &bf_ids, K);
        } else {
            tp_total_recall += compute_recall(result_ids, &ground_truth[i], K);
        }
    }
    let tp_avg_recall = if valid_queries > 0 { tp_total_recall / valid_queries as f64 } else { 0.0 };
    eprintln!("  Two-phase Recall@{}: {:.4} ({:.2}%)",
        K, tp_avg_recall, tp_avg_recall * 100.0);

    // ── 5. Determinism verification ──
    eprintln!("\n[5/5] Determinism verification (3 builds, SHA-256)...");
    let det_n = n.min(10_000); // Use 10K for determinism check (faster)
    let mut hashes = Vec::new();
    for run in 0..3 {
        let mut g = HnswGraph::<128>::new();
        let mut r = DeterministicRng::from_seed(HNSW_SEED);
        for v in fixed_vectors.iter().take(det_n) {
            g.insert(v.clone(), &mut r).unwrap();
        }
        let bytes = g.serialize().unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        eprintln!("  Run {}: hash={}...", run + 1, &hash[..16]);
        hashes.push(hash);
    }
    let all_match = hashes.iter().all(|h| h == &hashes[0]);
    eprintln!("  Determinism: {} (all {} builds identical)", 
        if all_match { "✅ PASS" } else { "❌ FAIL" }, hashes.len());

    // ── Summary ──
    eprintln!("\n┌─── Summary (N={}) ───────────────────────────┐", n);
    eprintln!("│ Build time:       {:>10.2}s ({:.0} vec/s)", build_time.as_secs_f64(), n as f64 / build_time.as_secs_f64());
    eprintln!("│ Search latency:   {:>10.1} μs/q", search_us);
    eprintln!("│ Search QPS:       {:>10.0}", search_qps);
    eprintln!("│ Two-phase lat:    {:>10.1} μs/q", tp_us);
    eprintln!("│ Two-phase QPS:    {:>10.0}", tp_qps);
    eprintln!("│ Recall@{}:        {:>10.4} ({:.2}%)", K, avg_recall, avg_recall * 100.0);
    eprintln!("│ TP Recall@{}:     {:>10.4} ({:.2}%)", K, tp_avg_recall, tp_avg_recall * 100.0);
    eprintln!("│ Deterministic:    {:>10}", if all_match { "YES" } else { "NO" });
    eprintln!("└───────────────────────────────────────────────┘");
}

fn measure_distance_overhead(base_vecs: &[Vec<f32>]) {
    let n_pairs = 500;
    let pairs: Vec<(usize, usize)> = {
        let mut rng = StdRng::seed_from_u64(42);
        (0..n_pairs).map(|_| {
            let a = rng.gen_range(0..base_vecs.len().min(100_000));
            let b = rng.gen_range(0..base_vecs.len().min(100_000));
            (a, b)
        }).collect()
    };

    // Convert to fixed-point
    let fixed_vecs: Vec<Vec<I64F32>> = base_vecs.iter().take(100_000)
        .map(|v| v.iter().map(|&x| I64F32::from_num(x)).collect())
        .collect();

    let n_runs = 10;

    // Warmup
    for _ in 0..3 {
        for &(a, b) in &pairs {
            let _ = std::hint::black_box(f32_squared_distance(&base_vecs[a], &base_vecs[b]));
            let _ = std::hint::black_box(simd::squared_distance_simd(&fixed_vecs[a], &fixed_vecs[b]));
        }
    }

    // f32 baseline
    let start = Instant::now();
    for _ in 0..n_runs {
        for &(a, b) in &pairs {
            let _ = std::hint::black_box(f32_squared_distance(&base_vecs[a], &base_vecs[b]));
        }
    }
    let f32_ns = start.elapsed().as_nanos() as f64 / (n_runs * n_pairs) as f64;

    // I64F32 scalar
    let start = Instant::now();
    for _ in 0..n_runs {
        for &(a, b) in &pairs {
            let _ = std::hint::black_box(simd::squared_distance_scalar(&fixed_vecs[a], &fixed_vecs[b]));
        }
    }
    let scalar_ns = start.elapsed().as_nanos() as f64 / (n_runs * n_pairs) as f64;

    // I64F32 SIMD
    let start = Instant::now();
    for _ in 0..n_runs {
        for &(a, b) in &pairs {
            let _ = std::hint::black_box(simd::squared_distance_simd(&fixed_vecs[a], &fixed_vecs[b]));
        }
    }
    let simd_ns = start.elapsed().as_nanos() as f64 / (n_runs * n_pairs) as f64;

    eprintln!("  f32 baseline:    {:.1} ns/pair", f32_ns);
    eprintln!("  I64F32 scalar:   {:.1} ns/pair ({:.2}x overhead)", scalar_ns, scalar_ns / f32_ns);
    eprintln!("  I64F32 SIMD:     {:.1} ns/pair ({:.2}x overhead)", simd_ns, simd_ns / f32_ns);
    eprintln!("  SIMD speedup vs scalar: {:.2}x", scalar_ns / simd_ns);
}

fn measure_error_real(base_vecs: &[Vec<f32>]) {
    // Use first 200 real SIFT vectors for error analysis
    let n = 200.min(base_vecs.len());
    let f64_vecs: Vec<Vec<f64>> = base_vecs[..n].iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();
    let fixed_vecs: Vec<Vec<I64F32>> = base_vecs[..n].iter()
        .map(|v| v.iter().map(|&x| I64F32::from_num(x)).collect())
        .collect();

    let mut max_i64_err = 0.0f64;
    let mut max_f32_err = 0.0f64;
    let mut sum_i64_err = 0.0f64;
    let mut sum_f32_err = 0.0f64;
    let mut count = 0usize;

    for i in 0..n {
        for j in (i+1)..n {
            // f64 ground truth
            let gt: f64 = f64_vecs[i].iter().zip(&f64_vecs[j])
                .map(|(&a, &b)| { let d = a - b; d * d })
                .sum();

            // f32 result
            let f32_dist = f32_squared_distance(&base_vecs[i], &base_vecs[j]) as f64;

            // I64F32 SIMD result
            let i64_dist = simd::squared_distance_simd(&fixed_vecs[i], &fixed_vecs[j])
                .to_num::<f64>();

            if gt.abs() > 1e-6 {
                let f32_rel = ((f32_dist - gt) / gt).abs();
                let i64_rel = ((i64_dist - gt) / gt).abs();
                if f32_rel > max_f32_err { max_f32_err = f32_rel; }
                if i64_rel > max_i64_err { max_i64_err = i64_rel; }
                sum_f32_err += f32_rel;
                sum_i64_err += i64_rel;
                count += 1;
            }
        }
    }

    let avg_i64 = if count > 0 { sum_i64_err / count as f64 } else { 0.0 };
    let avg_f32 = if count > 0 { sum_f32_err / count as f64 } else { 0.0 };
    let ratio = if max_i64_err > 0.0 { max_f32_err / max_i64_err } else { f64::INFINITY };

    eprintln!("  {} vector pairs from REAL SIFT-1M data", count);
    eprintln!("  I64F32 max relative error: {:.2e}", max_i64_err);
    eprintln!("  I64F32 avg relative error: {:.2e}", avg_i64);
    eprintln!("  f32    max relative error: {:.2e}", max_f32_err);
    eprintln!("  f32    avg relative error: {:.2e}", avg_f32);
    eprintln!("  Accuracy ratio (f32_max / I64F32_max): {:.0}x", ratio);
}
