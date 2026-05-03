// ============================================================
//  D-HNSW Full Experiment Runner (Optimized)
//  Runs real benchmarks for paper data:
//    1. Distance computation overhead (I64F32 vs f32) 
//    2. Index build + SHA-256 determinism verification
//    3. Recall@10 at various ef values
//    4. Accuracy analysis (error vs f64 ground truth)
//    5. Scalability across dimensions (96, 100, 128, 784, 960, 1536)
// ============================================================

use dhnsw::fixed_point::{FixedVector};
use dhnsw::graph::{DhnswConfig, DhnswIndex};
use std::time::Instant;

// ---- Deterministic data generation ----

/// Simple LCG PRNG for reproducible experiments (no system entropy)
struct DetRng {
    state: u64,
}

impl DetRng {
    fn new(seed: u64) -> Self {
        DetRng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform float in [lo, hi)
    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + u * (hi - lo)
    }
}

/// Generate vectors with LLM-like distribution (normalized, small magnitudes)
fn generate_llm_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = DetRng::new(seed);
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32_range(-0.2, 0.2)).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v.iter_mut().for_each(|x| *x *= 0.1);
        vectors.push(v);
    }
    vectors
}

/// Generate SIFT-like vectors (integer values 0-255)
fn generate_sift_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = DetRng::new(seed);
    (0..n).map(|_| {
        (0..dim).map(|_| (rng.next_u64() % 256) as f32).collect()
    }).collect()
}

/// Generate uniform vectors in [0, 1)
fn generate_uniform_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = DetRng::new(seed);
    (0..n).map(|_| {
        (0..dim).map(|_| rng.next_f32_range(0.0, 1.0)).collect()
    }).collect()
}

// ---- Brute-force ground truth ----

fn brute_force_knn(base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f64)> {
    let mut dists: Vec<(usize, f64)> = base.iter().enumerate().map(|(i, b)| {
        let d: f64 = b.iter().zip(query.iter())
            .map(|(a, q)| { let diff = *a as f64 - *q as f64; diff * diff })
            .sum();
        (i, d)
    }).collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

// ---- Experiment: Distance Computation Overhead ----

fn benchmark_distance_overhead(n_pairs: usize, vectors: &[Vec<f32>]) -> (f64, f64, f64) {
    let n = vectors.len().min(n_pairs * 2);
    
    // Prepare fixed-point vectors
    let fp_vecs: Vec<FixedVector> = vectors[..n].iter()
        .map(|v| FixedVector::from_f32_slice(v))
        .collect();

    // --- Benchmark f32 distance ---
    let start = Instant::now();
    let mut f32_sum: f64 = 0.0;
    let pairs = n / 2;
    for i in 0..pairs {
        let a = &vectors[i * 2];
        let b = &vectors[i * 2 + 1];
        let d: f32 = a.iter().zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        f32_sum += d as f64;
    }
    let f32_time = start.elapsed().as_nanos() as f64 / pairs as f64;

    // --- Benchmark I64F32 distance ---
    let start = Instant::now();
    let mut fp_sum: f64 = 0.0;
    for i in 0..pairs {
        let d = fp_vecs[i * 2].squared_euclidean(&fp_vecs[i * 2 + 1]);
        fp_sum += d.to_f64();
    }
    let fp_time = start.elapsed().as_nanos() as f64 / pairs as f64;

    let overhead = fp_time / f32_time;
    
    // Prevent optimizer from removing computations
    if f32_sum < -1e30 || fp_sum < -1e30 {
        eprintln!("impossible");
    }

    (f32_time, fp_time, overhead)
}

// ---- Experiment: Accuracy Analysis ----

fn benchmark_accuracy(vectors: &[Vec<f32>]) -> (f64, f64, f64, f64, f64) {
    let n = vectors.len().min(2000);
    let pairs = n / 2;
    
    let mut max_rel_error_fp: f64 = 0.0;
    let mut sum_rel_error_fp: f64 = 0.0;
    let mut max_rel_error_f32: f64 = 0.0;
    let mut sum_rel_error_f32: f64 = 0.0;
    let mut count: usize = 0;

    for i in 0..pairs {
        let a = &vectors[i * 2];
        let b = &vectors[i * 2 + 1];

        // f64 ground truth
        let d_f64: f64 = a.iter().zip(b.iter())
            .map(|(x, y)| { let diff = *x as f64 - *y as f64; diff * diff })
            .sum();

        // f32 computation
        let d_f32: f64 = a.iter().zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>() as f64;

        // I64F32 computation  
        let fv_a = FixedVector::from_f32_slice(a);
        let fv_b = FixedVector::from_f32_slice(b);
        let d_fp = fv_a.squared_euclidean(&fv_b).to_f64();

        if d_f64.abs() > 1e-15 {
            let rel_err_fp = ((d_fp - d_f64) / d_f64).abs();
            let rel_err_f32 = ((d_f32 - d_f64) / d_f64).abs();
            
            max_rel_error_fp = max_rel_error_fp.max(rel_err_fp);
            sum_rel_error_fp += rel_err_fp;
            max_rel_error_f32 = max_rel_error_f32.max(rel_err_f32);
            sum_rel_error_f32 += rel_err_f32;
            count += 1;
        }
    }

    let avg_rel_error_fp = if count > 0 { sum_rel_error_fp / count as f64 } else { 0.0 };
    let avg_rel_error_f32 = if count > 0 { sum_rel_error_f32 / count as f64 } else { 0.0 };

    (max_rel_error_fp, avg_rel_error_fp, max_rel_error_f32, avg_rel_error_f32,
     if max_rel_error_fp > 0.0 { max_rel_error_f32 / max_rel_error_fp } else { f64::INFINITY })
}

// ---- Main Experiment ----

fn run_dataset_experiment(
    name: &str,
    dim: usize,
    base_vectors: &[Vec<f32>],
    query_vectors: &[Vec<f32>],
    n_base: usize,
    n_queries: usize,
    n_determinism_runs: usize,
) {
    let n_base = n_base.min(base_vectors.len());
    let n_queries = n_queries.min(query_vectors.len());
    
    println!("\n{}", "=".repeat(70));
    println!("  DATASET: {} (dim={}, n_base={}, n_queries={})", name, dim, n_base, n_queries);
    println!("{}", "=".repeat(70));

    // 1. Distance computation overhead
    println!("\n--- 1. Distance Computation Overhead ---");
    let (f32_ns, fp_ns, overhead) = benchmark_distance_overhead(5000, &base_vectors[..n_base]);
    println!("  f32 distance:    {:.1} ns/pair", f32_ns);
    println!("  I64F32 distance: {:.1} ns/pair", fp_ns);
    println!("  Overhead ratio:  {:.2}x", overhead);

    // 2. Accuracy analysis
    println!("\n--- 2. Accuracy Analysis ---");
    let (max_err_fp, avg_err_fp, max_err_f32, avg_err_f32, acc_ratio) = benchmark_accuracy(&base_vectors[..n_base]);
    println!("  I64F32 max relative error: {:.2e}", max_err_fp);
    println!("  I64F32 avg relative error: {:.2e}", avg_err_fp);
    println!("  f32   max relative error:  {:.2e}", max_err_f32);
    println!("  f32   avg relative error:  {:.2e}", avg_err_f32);
    println!("  f32/I64F32 max-error ratio: {:.1}x (I64F32 is this many times more accurate)", acc_ratio);

    // 3. Build index + determinism verification
    println!("\n--- 3. Build Index + Determinism Verification ({} runs) ---", n_determinism_runs);
    let config = DhnswConfig {
        m: 16,
        m_max_0: 32,
        ef_construction: 200,
        seed: 42,
        dim,
    };

    let mut hashes = Vec::new();
    let mut build_times = Vec::new();

    for run in 0..n_determinism_runs {
        let start = Instant::now();
        let mut index = DhnswIndex::new(config.clone());
        for v in &base_vectors[..n_base] {
            index.insert(FixedVector::from_f32_slice(v));
        }
        let build_time = start.elapsed();
        let hash = index.sha256_hash();
        
        println!("  Run {}: build={:.2}s, SHA-256={}", 
                 run + 1, build_time.as_secs_f64(), &hash[..16]);
        
        build_times.push(build_time.as_secs_f64());
        hashes.push(hash);

        // On last run, do recall measurement
        if run == n_determinism_runs - 1 {
            println!("\n--- 4. Recall@10 at various ef values ---");
            let ef_values = [16, 32, 64, 128, 256];
            
            for &ef in &ef_values {
                let mut total_recall = 0.0;
                let k = 10;
                
                let search_start = Instant::now();
                let mut total_queries = 0;

                for q in &query_vectors[..n_queries] {
                    let query = FixedVector::from_f32_slice(q);
                    let results = index.search(&query, k, ef);
                    
                    // Ground truth
                    let gt = brute_force_knn(&base_vectors[..n_base], q, k);
                    let gt_ids: std::collections::HashSet<usize> = gt.iter().map(|(id, _)| *id).collect();
                    
                    let hits = results.iter().filter(|(id, _)| gt_ids.contains(id)).count();
                    total_recall += hits as f64 / k as f64;
                    total_queries += 1;
                }
                
                let search_time = search_start.elapsed();
                let avg_recall = total_recall / total_queries as f64;
                let qps = total_queries as f64 / search_time.as_secs_f64();
                
                println!("  ef={:>3}: Recall@10={:.4} ({:.2}%), QPS={:.0}", 
                         ef, avg_recall, avg_recall * 100.0, qps);
            }
        }
    }

    // Determinism check
    println!("\n--- 5. Determinism Verification ---");
    let all_same = hashes.windows(2).all(|w| w[0] == w[1]);
    println!("  All {} hashes identical: {}", n_determinism_runs, if all_same { "YES (PASS)" } else { "NO (FAIL!)" });
    println!("  Hash: {}", hashes[0]);

    // Summary JSON
    let avg_build = build_times.iter().sum::<f64>() / build_times.len() as f64;
    println!("\n--- RESULTS JSON ---");
    println!("{{");
    println!("  \"dataset\": \"{}\",", name);
    println!("  \"dim\": {},", dim);
    println!("  \"n_base\": {},", n_base);
    println!("  \"n_queries\": {},", n_queries);
    println!("  \"f32_ns_per_pair\": {:.1},", f32_ns);
    println!("  \"i64f32_ns_per_pair\": {:.1},", fp_ns);
    println!("  \"overhead_ratio\": {:.4},", overhead);
    println!("  \"max_rel_error_i64f32\": {:.2e},", max_err_fp);
    println!("  \"avg_rel_error_i64f32\": {:.2e},", avg_err_fp);
    println!("  \"max_rel_error_f32\": {:.2e},", max_err_f32);
    println!("  \"avg_rel_error_f32\": {:.2e},", avg_err_f32);
    println!("  \"f32_vs_i64f32_accuracy_ratio\": {:.1},", acc_ratio);
    println!("  \"avg_build_time_s\": {:.2},", avg_build);
    println!("  \"deterministic\": {},", all_same);
    println!("  \"sha256\": \"{}\"", hashes[0]);
    println!("}}");
}

fn main() {
    println!("================================================================");
    println!("  D-HNSW Full Experiment Runner");
    println!("  Rust implementation - REAL measured data");
    println!("  Platform: {} {}", std::env::consts::ARCH, std::env::consts::OS);
    println!("================================================================");
    println!("  Timestamp: {:?}", std::time::SystemTime::now());
    println!("  Config: M=16, ef_construction=200, seed=42");
    println!();

    // ============================================================
    // Experiment 1: LLM Embeddings (1536-d) - PRIMARY TARGET
    // Using 2000 vectors (sufficient for paper validation)
    // ============================================================
    println!("========================================");
    println!("  EXPERIMENT 1: LLM Embeddings (1536-d)");
    println!("========================================");
    
    let llm_base = generate_llm_vectors(2000, 1536, 42);
    let llm_queries = generate_llm_vectors(100, 1536, 9999);
    run_dataset_experiment("DBPedia-OpenAI3-1536d-synthetic", 1536, &llm_base, &llm_queries, 2000, 100, 3);

    // ============================================================
    // Experiment 2: SIFT-like (128-d)
    // ============================================================
    println!("\n\n========================================");
    println!("  EXPERIMENT 2: SIFT-128");
    println!("========================================");
    
    let sift_base = generate_sift_vectors(5000, 128, 42);
    let sift_queries = generate_sift_vectors(100, 128, 9999);
    run_dataset_experiment("SIFT-128-synthetic", 128, &sift_base, &sift_queries, 5000, 100, 3);

    // ============================================================
    // Experiment 3: GloVe-like (100-d) 
    // ============================================================
    println!("\n\n========================================");
    println!("  EXPERIMENT 3: GloVe-100");
    println!("========================================");
    
    let glove_base = generate_uniform_vectors(5000, 100, 42);
    let glove_queries = generate_uniform_vectors(100, 100, 9999);
    run_dataset_experiment("GloVe-100-synthetic", 100, &glove_base, &glove_queries, 5000, 100, 3);

    // ============================================================
    // Experiment 4: GIST-like (960-d)
    // ============================================================
    println!("\n\n========================================");
    println!("  EXPERIMENT 4: GIST-960");
    println!("========================================");
    
    let gist_base = generate_uniform_vectors(2000, 960, 42);
    let gist_queries = generate_uniform_vectors(50, 960, 9999);
    run_dataset_experiment("GIST-960-synthetic", 960, &gist_base, &gist_queries, 2000, 50, 3);

    // ============================================================
    // Experiment 5: Deep-like (96-d)
    // ============================================================
    println!("\n\n========================================");
    println!("  EXPERIMENT 5: Deep-96");
    println!("========================================");
    
    let deep_base = generate_uniform_vectors(5000, 96, 42);
    let deep_queries = generate_uniform_vectors(100, 96, 9999);
    run_dataset_experiment("Deep-96-synthetic", 96, &deep_base, &deep_queries, 5000, 100, 3);

    // ============================================================
    // Experiment 6: Fashion-MNIST-like (784-d)
    // ============================================================
    println!("\n\n========================================");
    println!("  EXPERIMENT 6: Fashion-MNIST-784");
    println!("========================================");
    
    let fashion_base = generate_sift_vectors(2000, 784, 42);
    let fashion_queries = generate_sift_vectors(50, 784, 9999);
    run_dataset_experiment("Fashion-MNIST-784-synthetic", 784, &fashion_base, &fashion_queries, 2000, 50, 3);

    // ============================================================
    // Dimension Scaling Summary
    // ============================================================
    println!("\n\n========================================");
    println!("  DIMENSION SCALING: Distance Overhead");
    println!("========================================");
    
    for &dim in &[32, 64, 128, 256, 512, 768, 1024, 1536, 2048] {
        let vecs = generate_uniform_vectors(2000, dim, 42);
        let (f32_ns, fp_ns, ratio) = benchmark_distance_overhead(1000, &vecs);
        println!("  dim={:>5}: f32={:.0}ns, I64F32={:.0}ns, ratio={:.2}x", dim, f32_ns, fp_ns, ratio);
    }

    println!("\n\n================================================================");
    println!("  ALL EXPERIMENTS COMPLETED");
    println!("================================================================");
}
