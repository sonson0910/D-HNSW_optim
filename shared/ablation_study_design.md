# Ablation Study Framework Design for D-HNSW
## Ready-to-Implement Experimental Design

---

## 1. Overview

The ablation study isolates the overhead contribution of each deterministic component in D-HNSW. We define **7 configurations** that progressively add/remove components, measuring their individual and combined impact on latency, throughput, memory, and recall.

---

## 2. Configuration Matrix

### 2.1 Component Definitions

| Component ID | Name | Description | D-HNSW Setting | hnswlib Setting |
|---|---|---|---|---|
| **C1** | Distance Arithmetic | Type of arithmetic for distance computation | I64F32 fixed-point | f32 IEEE 754 |
| **C2** | Overflow Protection | Saturating vs wrapping arithmetic | Saturating (checked) | None (native f32) |
| **C3** | RNG Source | Random number generator for level assignment | ChaCha20 (deterministic) | System PRNG |
| **C4** | Insertion Order | Ordering constraint on vector insertion | Canonical (sorted by tx index) | Arbitrary |
| **C5** | Square Root | Method for computing Euclidean distance | Integer sqrt (isqrt) | Hardware FPU sqrt |

### 2.2 Seven Ablation Configurations

```
Config 0: hnswlib baseline
  C1=f32, C2=none, C3=system, C4=arbitrary, C5=fpu

Config 1: Full D-HNSW (all deterministic components)
  C1=I64F32, C2=saturating, C3=ChaCha20, C4=canonical, C5=isqrt

Config 2: Ablate distance only (f32 distance, rest deterministic)
  C1=f32, C2=none, C3=ChaCha20, C4=canonical, C5=fpu
  → Measures: overhead of I64F32 distance + isqrt + overflow protection

Config 3: Ablate RNG only (system RNG, rest deterministic)
  C1=I64F32, C2=saturating, C3=system, C4=canonical, C5=isqrt
  → Measures: overhead of ChaCha20 vs system PRNG

Config 4: Ablate ordering only (arbitrary order, rest deterministic)
  C1=I64F32, C2=saturating, C3=ChaCha20, C4=arbitrary, C5=isqrt
  → Measures: overhead of canonical ordering

Config 5: Ablate overflow protection (wrapping arithmetic)
  C1=I64F32, C2=wrapping, C3=ChaCha20, C4=canonical, C5=isqrt
  → Measures: overhead of saturating checks

Config 6: I64F32 distance but f32 sqrt (hybrid)
  C1=I64F32, C2=saturating, C3=ChaCha20, C4=canonical, C5=fpu_convert
  → Measures: overhead of isqrt vs convert-to-f64-sqrt-convert-back
```

### 2.3 Overhead Attribution Formula

Total overhead = $T_1 - T_0$ where $T_i$ is latency of Config $i$.

Individual contributions:
- **Distance arithmetic:** $\Delta_{dist} = T_1 - T_2$ (everything except distance)
- **RNG:** $\Delta_{rng} = T_1 - T_3$
- **Ordering:** $\Delta_{ord} = T_1 - T_4$
- **Overflow protection:** $\Delta_{ovf} = T_1 - T_5$
- **Square root:** $\Delta_{sqrt} = T_1 - T_6$

Verification: $\Delta_{dist} + \Delta_{rng} + \Delta_{ord} + \Delta_{ovf} + \Delta_{sqrt} \approx T_1 - T_0$ (may not be exact due to interactions)

---

## 3. Rust Implementation Architecture

### 3.1 Trait-Based Component Abstraction

```rust
// === Distance Trait ===
pub trait DistanceMetric: Send + Sync {
    type Scalar: Copy + PartialOrd;
    
    /// Compute squared distance between two vectors
    fn squared_distance(&self, a: &[Self::Scalar], b: &[Self::Scalar]) -> Self::Scalar;
    
    /// Compute Euclidean distance
    fn distance(&self, a: &[Self::Scalar], b: &[Self::Scalar]) -> Self::Scalar;
    
    /// Name for logging
    fn name(&self) -> &str;
}

// I64F32 implementation (deterministic)
pub struct I64F32Distance {
    pub use_saturating: bool,  // C2: overflow protection toggle
    pub use_isqrt: bool,       // C5: integer sqrt toggle
}

impl DistanceMetric for I64F32Distance {
    type Scalar = i64;  // I64F32 represented as i64
    
    fn squared_distance(&self, a: &[i64], b: &[i64]) -> i64 {
        let mut sum: i64 = 0;
        for i in 0..a.len() {
            let diff = if self.use_saturating {
                a[i].saturating_sub(b[i])
            } else {
                a[i].wrapping_sub(b[i])  // Ablation: no overflow protection
            };
            
            // 128-bit multiplication to avoid overflow
            let product = (diff as i128) * (diff as i128);
            let truncated = (product >> 32) as i64;  // Truncate to I64F32
            
            sum = if self.use_saturating {
                sum.saturating_add(truncated)
            } else {
                sum.wrapping_add(truncated)
            };
        }
        sum
    }
    
    fn distance(&self, a: &[i64], b: &[i64]) -> i64 {
        let sq_dist = self.squared_distance(a, b);
        if self.use_isqrt {
            isqrt_i64f32(sq_dist)  // Deterministic integer sqrt
        } else {
            // Convert to f64, sqrt, convert back (non-deterministic!)
            let f = (sq_dist as f64) / (1u64 << 32) as f64;
            (f.sqrt() * (1u64 << 32) as f64) as i64
        }
    }
    
    fn name(&self) -> &str { "I64F32" }
}

// f32 implementation (baseline, non-deterministic)
pub struct F32Distance;

impl DistanceMetric for F32Distance {
    type Scalar = f32;
    
    fn squared_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }
    
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.squared_distance(a, b).sqrt()
    }
    
    fn name(&self) -> &str { "f32" }
}
```

### 3.2 RNG Abstraction

```rust
// === RNG Trait ===
pub trait LevelGenerator: Send + Sync {
    /// Generate the maximum layer for a new node
    fn generate_level(&mut self, ml: f64) -> usize;
    
    /// Name for logging
    fn name(&self) -> &str;
}

// ChaCha20-based deterministic RNG (D-HNSW)
pub struct DeterministicRng {
    state: [u32; 16],  // ChaCha20 state
    counter: u64,
}

impl DeterministicRng {
    pub fn from_consensus_seed(block_hash: &[u8; 32]) -> Self {
        // Derive ChaCha20 key from block hash
        // ...
        Self { state: /* ... */, counter: 0 }
    }
}

impl LevelGenerator for DeterministicRng {
    fn generate_level(&mut self, ml: f64) -> usize {
        let random_value = self.next_u64();
        // Convert to uniform [0, 1) then to geometric distribution
        let uniform = (random_value as f64) / (u64::MAX as f64);
        (-uniform.ln() * ml) as usize  // Note: ln uses f64 but only for level selection
    }
    
    fn name(&self) -> &str { "ChaCha20" }
}

// System RNG (baseline)
pub struct SystemRng {
    rng: rand::rngs::ThreadRng,
}

impl LevelGenerator for SystemRng {
    fn generate_level(&mut self, ml: f64) -> usize {
        let uniform: f64 = self.rng.gen();
        (-uniform.ln() * ml) as usize
    }
    
    fn name(&self) -> &str { "SystemRNG" }
}
```

### 3.3 Ordering Abstraction

```rust
// === Insertion Order ===
pub enum InsertionOrder {
    Canonical,   // Sorted by (block_number, tx_index, vector_id)
    Arbitrary,   // Original order (as provided)
}

impl InsertionOrder {
    pub fn prepare_indices(&self, vectors: &[Vector], metadata: &[TxMetadata]) -> Vec<usize> {
        match self {
            InsertionOrder::Canonical => {
                let mut indices: Vec<usize> = (0..vectors.len()).collect();
                indices.sort_by_key(|&i| {
                    (metadata[i].block_number, metadata[i].tx_index, i)
                });
                indices
            }
            InsertionOrder::Arbitrary => {
                (0..vectors.len()).collect()
            }
        }
    }
}
```

### 3.4 Ablation Configuration

```rust
/// Complete ablation configuration
pub struct AblationConfig {
    pub name: String,
    pub distance_type: DistanceType,      // C1: I64F32 or f32
    pub use_saturating: bool,              // C2: overflow protection
    pub rng_type: RngType,                 // C3: ChaCha20 or System
    pub insertion_order: InsertionOrder,   // C4: Canonical or Arbitrary
    pub sqrt_type: SqrtType,              // C5: isqrt or fpu
}

pub enum DistanceType { I64F32, F32 }
pub enum RngType { ChaCha20, System }
pub enum SqrtType { IntegerSqrt, FpuSqrt }

impl AblationConfig {
    /// Generate all 7 configurations
    pub fn all_configs() -> Vec<Self> {
        vec![
            // Config 0: hnswlib baseline
            Self {
                name: "Config0_hnswlib_baseline".into(),
                distance_type: DistanceType::F32,
                use_saturating: false,
                rng_type: RngType::System,
                insertion_order: InsertionOrder::Arbitrary,
                sqrt_type: SqrtType::FpuSqrt,
            },
            // Config 1: Full D-HNSW
            Self {
                name: "Config1_full_dhnsw".into(),
                distance_type: DistanceType::I64F32,
                use_saturating: true,
                rng_type: RngType::ChaCha20,
                insertion_order: InsertionOrder::Canonical,
                sqrt_type: SqrtType::IntegerSqrt,
            },
            // Config 2: Ablate distance (use f32, keep rest)
            Self {
                name: "Config2_ablate_distance".into(),
                distance_type: DistanceType::F32,
                use_saturating: false,
                rng_type: RngType::ChaCha20,
                insertion_order: InsertionOrder::Canonical,
                sqrt_type: SqrtType::FpuSqrt,
            },
            // Config 3: Ablate RNG (use system, keep rest)
            Self {
                name: "Config3_ablate_rng".into(),
                distance_type: DistanceType::I64F32,
                use_saturating: true,
                rng_type: RngType::System,
                insertion_order: InsertionOrder::Canonical,
                sqrt_type: SqrtType::IntegerSqrt,
            },
            // Config 4: Ablate ordering (use arbitrary, keep rest)
            Self {
                name: "Config4_ablate_ordering".into(),
                distance_type: DistanceType::I64F32,
                use_saturating: true,
                rng_type: RngType::ChaCha20,
                insertion_order: InsertionOrder::Arbitrary,
                sqrt_type: SqrtType::IntegerSqrt,
            },
            // Config 5: Ablate overflow protection
            Self {
                name: "Config5_ablate_overflow".into(),
                distance_type: DistanceType::I64F32,
                use_saturating: false,
                rng_type: RngType::ChaCha20,
                insertion_order: InsertionOrder::Canonical,
                sqrt_type: SqrtType::IntegerSqrt,
            },
            // Config 6: Hybrid sqrt (I64F32 distance, fpu sqrt)
            Self {
                name: "Config6_hybrid_sqrt".into(),
                distance_type: DistanceType::I64F32,
                use_saturating: true,
                rng_type: RngType::ChaCha20,
                insertion_order: InsertionOrder::Canonical,
                sqrt_type: SqrtType::FpuSqrt,
            },
        ]
    }
}
```

---

## 4. Benchmark Harness

### 4.1 Metrics Collection

```rust
pub struct BenchmarkResult {
    pub config_name: String,
    pub dataset: String,
    
    // Build metrics
    pub build_time_ms: f64,
    pub memory_bytes: usize,
    pub index_size_bytes: usize,
    
    // Query metrics (at different ef_search values)
    pub query_results: Vec<QueryMetrics>,
    
    // Determinism verification
    pub is_bit_identical: bool,
    pub hash_of_graph: String,  // SHA-256 of serialized graph
    pub hash_of_results: String,  // SHA-256 of query results
}

pub struct QueryMetrics {
    pub ef_search: usize,
    pub recall_at_1: f64,
    pub recall_at_10: f64,
    pub recall_at_100: f64,
    pub avg_latency_us: f64,
    pub p99_latency_us: f64,
    pub qps: f64,
    pub avg_distance_computations: f64,
    pub avg_hops: f64,
}
```

### 4.2 Benchmark Runner

```rust
pub fn run_ablation_study(
    dataset: &Dataset,
    ground_truth: &GroundTruth,
    configs: &[AblationConfig],
    ef_search_values: &[usize],
    num_query_runs: usize,  // Number of repetitions for stable timing
) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    for config in configs {
        println!("=== Running config: {} ===", config.name);
        
        // 1. Build index
        let start = Instant::now();
        let index = build_index(dataset, config);
        let build_time = start.elapsed().as_millis() as f64;
        
        // 2. Measure memory
        let memory = index.memory_usage();
        
        // 3. Compute graph hash for determinism verification
        let graph_hash = sha256_of_serialized(&index);
        
        // 4. Run queries at different ef_search values
        let mut query_results = Vec::new();
        for &ef in ef_search_values {
            let metrics = benchmark_queries(
                &index, 
                &dataset.queries, 
                ground_truth, 
                ef, 
                num_query_runs
            );
            query_results.push(metrics);
        }
        
        // 5. Verify determinism (run build again, compare hash)
        let index2 = build_index(dataset, config);
        let graph_hash2 = sha256_of_serialized(&index2);
        let is_bit_identical = graph_hash == graph_hash2;
        
        results.push(BenchmarkResult {
            config_name: config.name.clone(),
            dataset: dataset.name.clone(),
            build_time_ms: build_time,
            memory_bytes: memory,
            index_size_bytes: index.serialized_size(),
            query_results,
            is_bit_identical,
            hash_of_graph: graph_hash,
            hash_of_results: sha256_of_results(&query_results),
        });
    }
    
    results
}
```

### 4.3 Output Format

```rust
pub fn print_ablation_table(results: &[BenchmarkResult]) {
    // Header
    println!("{:<30} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Configuration", "Build(ms)", "Mem(MB)", "QPS@0.95", "QPS@0.99", "Recall@10", "Deterministic");
    println!("{}", "-".repeat(92));
    
    let baseline_qps = results[0].query_results[/* ef for 0.95 recall */].qps;
    
    for result in results {
        let qps_95 = find_qps_at_recall(&result.query_results, 0.95);
        let qps_99 = find_qps_at_recall(&result.query_results, 0.99);
        let best_recall = result.query_results.last().unwrap().recall_at_10;
        let overhead = baseline_qps / qps_95;
        
        println!("{:<30} {:>10.1} {:>10.1} {:>10.0} {:>10.0} {:>10.4} {:>12}",
            result.config_name,
            result.build_time_ms,
            result.memory_bytes as f64 / 1e6,
            qps_95,
            qps_99,
            best_recall,
            if result.is_bit_identical { "✅ YES" } else { "❌ NO" }
        );
    }
    
    // Overhead attribution
    println!("\n=== Overhead Attribution ===");
    let t_baseline = results[0]./* latency */;
    let t_full = results[1]./* latency */;
    let t_ablate_dist = results[2]./* latency */;
    let t_ablate_rng = results[3]./* latency */;
    let t_ablate_ord = results[4]./* latency */;
    let t_ablate_ovf = results[5]./* latency */;
    let t_ablate_sqrt = results[6]./* latency */;
    
    let total_overhead = t_full - t_baseline;
    println!("Total overhead: {:.1}x ({:.1} us)", t_full / t_baseline, total_overhead);
    println!("  Distance arithmetic: {:.1}% ({:.1} us)", 
        (t_full - t_ablate_dist) / total_overhead * 100.0,
        t_full - t_ablate_dist);
    println!("  RNG (ChaCha20):      {:.1}% ({:.1} us)",
        (t_full - t_ablate_rng) / total_overhead * 100.0,
        t_full - t_ablate_rng);
    println!("  Canonical ordering:  {:.1}% ({:.1} us)",
        (t_full - t_ablate_ord) / total_overhead * 100.0,
        t_full - t_ablate_ord);
    println!("  Overflow protection: {:.1}% ({:.1} us)",
        (t_full - t_ablate_ovf) / total_overhead * 100.0,
        t_full - t_ablate_ovf);
    println!("  Integer sqrt:        {:.1}% ({:.1} us)",
        (t_full - t_ablate_sqrt) / total_overhead * 100.0,
        t_full - t_ablate_sqrt);
}
```

---

## 5. Dataset Loading

### 5.1 Standard Benchmark Datasets

```rust
pub struct Dataset {
    pub name: String,
    pub dimensions: usize,
    pub train_vectors: Vec<Vec<f32>>,   // Original f32 vectors
    pub query_vectors: Vec<Vec<f32>>,
    pub num_vectors: usize,
}

pub struct GroundTruth {
    pub neighbors: Vec<Vec<usize>>,  // ground truth k-NN for each query
    pub distances: Vec<Vec<f32>>,
}

/// Load standard ANN benchmark datasets
pub fn load_dataset(name: &str) -> (Dataset, GroundTruth) {
    match name {
        "sift-1m" => load_fvecs_dataset(
            "sift/sift_base.fvecs",
            "sift/sift_query.fvecs",
            "sift/sift_groundtruth.ivecs",
        ),
        "gist-1m" => load_fvecs_dataset(
            "gist/gist_base.fvecs",
            "gist/gist_query.fvecs",
            "gist/gist_groundtruth.ivecs",
        ),
        "glove-100" => load_hdf5_dataset("glove-100-angular.hdf5"),
        "deep-1m" => load_fvecs_dataset(
            "deep/deep_base.fvecs",
            "deep/deep_query.fvecs",
            "deep/deep_groundtruth.ivecs",
        ),
        _ => panic!("Unknown dataset: {}", name),
    }
}

/// Convert f32 vectors to I64F32 representation
pub fn f32_to_i64f32(vectors: &[Vec<f32>]) -> Vec<Vec<i64>> {
    vectors.iter().map(|v| {
        v.iter().map(|&x| {
            // Round to nearest I64F32 value
            (x as f64 * (1u64 << 32) as f64).round() as i64
        }).collect()
    }).collect()
}
```

---

## 6. Expected Output Table

Based on our theoretical analysis, here's what we expect the ablation to show:

```
=== Ablation Study Results (SIFT-1M, ef_search=128) ===

Configuration                    Build(ms)   Mem(MB)   QPS@0.95   QPS@0.99  Recall@10  Deterministic
------------------------------------------------------------------------------------------------
Config0_hnswlib_baseline            2500.0     256.0      5000       2000     0.9990      ❌ NO
Config1_full_dhnsw                  5200.0     512.0      2380       1000     0.9988      ✅ YES
Config2_ablate_distance             2600.0     256.0      4800       1900     0.9990      ❌ NO
Config3_ablate_rng                  5100.0     512.0      2450       1050     0.9988      ❌ NO
Config4_ablate_ordering             5150.0     512.0      2400       1020     0.9988      ❌ NO
Config5_ablate_overflow             5000.0     512.0      2500       1100     0.9988      ⚠️ RISK
Config6_hybrid_sqrt                 5050.0     512.0      2450       1050     0.9988      ❌ NO

=== Overhead Attribution ===
Total overhead: 2.10x (220 us per query)
  Distance arithmetic: 55-65% (~130 us)    ← DOMINANT FACTOR
  Integer sqrt:        10-15% (~25 us)
  Overflow protection: 8-12%  (~22 us)
  RNG (ChaCha20):      5-8%   (~15 us)
  Canonical ordering:  3-5%   (~8 us)
```

**Key prediction:** Distance arithmetic (I64F32 multiplication) will be the dominant overhead, confirming that SIMD optimization (O1) and Two-Phase search (O2) should be the top priorities.

---

## 7. Statistical Rigor

### 7.1 Measurement Protocol

1. **Warm-up:** Run 1000 queries before timing to warm CPU caches
2. **Repetitions:** Run each query set 10 times, report median
3. **Variance:** Report standard deviation and p99 latency
4. **Isolation:** Pin to single CPU core, disable turbo boost, disable hyperthreading
5. **Memory:** Measure RSS (Resident Set Size) after index construction

### 7.2 Hardware Specification

Report in paper:
- CPU model, frequency, cache sizes (L1/L2/L3)
- RAM size and speed
- OS and kernel version
- Compiler and optimization flags
- Rust version and target triple

### 7.3 Reproducibility

- All code open-source
- Docker container with exact environment
- Random seeds fixed and documented
- Dataset download scripts included


---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
