# Multi-Dataset Benchmark & Cross-Platform Testing Design
## Comprehensive Experimental Plan for IEEE TKDE

---

## 1. Multi-Dataset Benchmark Suite

### 1.1 Dataset Selection Rationale

| Dataset | Dims | Vectors | Queries | Metric | Why Include |
|---------|------|---------|---------|--------|-------------|
| **SIFT-1M** | 128 | 1,000,000 | 10,000 | L2 | Standard benchmark, existing results |
| **GIST-1M** | 960 | 1,000,000 | 1,000 | L2 | High-dimensional stress test |
| **GloVe-100** | 100 | 1,183,514 | 10,000 | Cosine | NLP embeddings, clustered |
| **Deep-1M** | 96 | 1,000,000 | 10,000 | L2 | Deep learning features |
| **Fashion-MNIST** | 784 | 60,000 | 10,000 | L2 | High-dim, small scale |
| **Random-1M** | 128 | 1,000,000 | 10,000 | L2 | Worst-case (uniform random) |

### 1.2 Dataset Sources

```bash
# SIFT-1M (from TEXMEX corpus)
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

# GIST-1M
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz

# GloVe-100 (from ANN Benchmarks)
wget http://ann-benchmarks.com/glove-100-angular.hdf5

# Deep-1M (from ANN Benchmarks)  
wget http://ann-benchmarks.com/deep-image-96-angular.hdf5

# Fashion-MNIST
wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5

# Random-1M: Generate programmatically with deterministic seed
```

### 1.3 Metrics to Report Per Dataset

For each dataset, report a comprehensive set of metrics:

#### Build Phase
- **Index construction time** (seconds)
- **Peak memory during construction** (MB)
- **Final index size** (MB)
- **Graph statistics:** avg degree, max degree, num layers

#### Query Phase (vary ef_search from 10 to 500)
- **Recall@1, Recall@10, Recall@100**
- **QPS** (queries per second) at each recall level
- **Average query latency** (μs) and p50/p95/p99
- **Average distance computations per query**
- **Average graph hops per query**

#### Determinism Verification
- **Bit-identical test:** Build twice, compare SHA-256 of serialized graph
- **Cross-run consistency:** Run 10 queries 100 times, verify all results identical
- **Distance error analysis:** Compare I64F32 distances vs f64 ground truth

### 1.4 Key Plots to Generate

#### Plot 1: QPS vs Recall Pareto Curves (One per dataset)
```
X-axis: Recall@10
Y-axis: QPS (log scale)
Lines: D-HNSW, hnswlib, HNSW-LVQ, (other baselines)
```

#### Plot 2: Memory vs Recall Trade-off
```
X-axis: Recall@10
Y-axis: Memory per vector (bytes)
Lines: D-HNSW, hnswlib, HNSW-PQ, RaBitQ
```

#### Plot 3: Overhead Breakdown Across Datasets
```
X-axis: Dataset name
Y-axis: Overhead multiplier (vs hnswlib)
Stacked bars: Distance, RNG, Ordering, Overflow, Sqrt
```

#### Plot 4: Distance Error Distribution
```
X-axis: Relative error |D̃ - D| / D
Y-axis: Frequency (histogram)
Separate subplot per dataset
Overlay: Theoretical bound from Theorem 1
```

#### Plot 5: Scalability (Index Size Sweep)
```
X-axis: Number of vectors (100K to 10M)
Y-axis: Build time / Query latency / Memory
Lines: D-HNSW, hnswlib
```

---

## 2. Baseline Comparison Framework

### 2.1 Baselines to Compare

| Baseline | Implementation | Deterministic? | Notes |
|----------|---------------|----------------|-------|
| **hnswlib** | C++ (Malkov) | ❌ | Gold standard HNSW |
| **Faiss HNSW** | C++ (Meta) | ❌ | Production-grade |
| **HNSW-LVQ (INT8)** | Custom or Intel QHNSW | ⚠️ Partial | Quantized HNSW |
| **RaBitQ + IVF** | C++ (NTU) | ⚠️ Partial | SOTA quantization |
| **Valori** | Rust (Q16.16) | ✅ | Direct competitor |
| **D-HNSW (ours)** | Rust (I64F32) | ✅ | Our method |
| **D-HNSW + Opts** | Rust (I64F32) | ✅ | Our method + optimizations |

### 2.2 Fair Comparison Protocol

To ensure fair comparison:

1. **Same hardware:** All methods run on the same machine
2. **Same dataset:** Identical vectors and queries
3. **Same ground truth:** Pre-computed exact k-NN
4. **Same recall targets:** Compare at recall@10 = {0.90, 0.95, 0.99}
5. **Parameter tuning:** Each method gets optimal parameters for each recall target
6. **Warm-up:** 1000 queries before timing
7. **Repetitions:** 10 runs, report median

### 2.3 Determinism Comparison Table

This is the **unique selling point** table for the paper:

```
┌──────────────────────────────────────────────────────────────────────┐
│              Determinism Verification Results                        │
├──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│ Method       │ x86→x86  │ x86→ARM  │ GCC→Clang│ Run1→Run2│ Overall  │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ hnswlib      │ ✅ Same  │ ❌ Diff  │ ❌ Diff  │ ❌ Diff  │ ❌       │
│ Faiss HNSW   │ ✅ Same  │ ❌ Diff  │ ❌ Diff  │ ❌ Diff  │ ❌       │
│ HNSW-LVQ     │ ✅ Same  │ ⚠️ ~Same │ ⚠️ ~Same │ ❌ Diff  │ ⚠️       │
│ RaBitQ       │ ✅ Same  │ ⚠️ ~Same │ ⚠️ ~Same │ ❌ Diff  │ ⚠️       │
│ Valori       │ ✅ Same  │ ✅ Same  │ ✅ Same  │ ✅ Same  │ ✅       │
│ D-HNSW       │ ✅ Same  │ ✅ Same  │ ✅ Same  │ ✅ Same  │ ✅       │
│ D-HNSW+Opts  │ ✅ Same  │ ✅ Same  │ ✅ Same  │ ✅ Same  │ ✅       │
└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

### 2.4 Key Comparison Dimensions

#### Dimension 1: Performance at Same Determinism Level
Compare only deterministic methods:
- D-HNSW vs Valori (Q16.16)
- Show I64F32 has better recall due to higher precision
- Show D-HNSW+Opts has competitive or better QPS

#### Dimension 2: Determinism at Same Performance Level
Compare D-HNSW+Opts with non-deterministic methods at similar QPS:
- D-HNSW+Opts achieves ~X QPS with 100% determinism
- hnswlib achieves ~X QPS but is non-deterministic
- Quantify the "cost of determinism" as a percentage

#### Dimension 3: Precision Analysis
Compare error bounds:
- I64F32: formal bound from Theorem 1 (Section 2 of error bounds)
- Q16.16: formal bound from Corollary 2
- INT8: empirical error measurement
- RaBitQ: their theoretical bound (compare directly)

---

## 3. Cross-Platform Testing Matrix

### 3.1 Platform Configurations

| ID | Architecture | OS | Compiler | Opt Level | SIMD |
|----|-------------|-----|----------|-----------|------|
| P1 | x86_64 | Ubuntu 22.04 | GCC 13.2 | -O2 | AVX2 |
| P2 | x86_64 | Ubuntu 22.04 | Clang 17 | -O2 | AVX2 |
| P3 | x86_64 | Windows 11 | MSVC 2022 | /O2 | AVX2 |
| P4 | ARM64 | macOS 14 (M2) | Clang 17 | -O2 | NEON |
| P5 | ARM64 | Ubuntu 22.04 | GCC 13.2 | -O2 | NEON |
| P6 | WASM32 | Browser (V8) | wasm32-unknown | -O2 | None |

### 3.2 Verification Protocol

```rust
/// Cross-platform determinism verification
pub struct DeterminismTest {
    pub dataset: String,
    pub num_vectors: usize,
    pub num_queries: usize,
    pub hnsw_params: HnswParams,
}

impl DeterminismTest {
    pub fn run(&self) -> DeterminismReport {
        // 1. Build index
        let index = DhsnwIndex::build(&self.vectors, &self.params);
        
        // 2. Compute hash of entire graph structure
        let graph_hash = {
            let mut hasher = Sha256::new();
            // Hash all edges in canonical order
            for node_id in 0..index.num_nodes() {
                for layer in 0..index.num_layers() {
                    for &neighbor in index.neighbors(node_id, layer) {
                        hasher.update(&node_id.to_le_bytes());
                        hasher.update(&layer.to_le_bytes());
                        hasher.update(&neighbor.to_le_bytes());
                    }
                }
            }
            hex::encode(hasher.finalize())
        };
        
        // 3. Run all queries and hash results
        let results_hash = {
            let mut hasher = Sha256::new();
            for query in &self.queries {
                let results = index.search(query, 10, 128);
                for (id, dist) in &results {
                    hasher.update(&id.to_le_bytes());
                    hasher.update(&dist.to_le_bytes());
                }
            }
            hex::encode(hasher.finalize())
        };
        
        // 4. Hash all stored vectors (verify conversion consistency)
        let vectors_hash = {
            let mut hasher = Sha256::new();
            for vec in index.vectors() {
                for &val in vec {
                    hasher.update(&val.to_le_bytes());
                }
            }
            hex::encode(hasher.finalize())
        };
        
        DeterminismReport {
            platform: get_platform_info(),
            graph_hash,
            results_hash,
            vectors_hash,
        }
    }
}

pub struct DeterminismReport {
    pub platform: PlatformInfo,
    pub graph_hash: String,     // SHA-256 of graph structure
    pub results_hash: String,   // SHA-256 of all query results
    pub vectors_hash: String,   // SHA-256 of stored vectors
}
```

### 3.3 Expected Results

All 6 platforms should produce **identical hashes**:

```
Platform P1 (x86_64/GCC):    graph=a1b2c3... results=d4e5f6... vectors=g7h8i9...
Platform P2 (x86_64/Clang):  graph=a1b2c3... results=d4e5f6... vectors=g7h8i9...
Platform P3 (x86_64/MSVC):   graph=a1b2c3... results=d4e5f6... vectors=g7h8i9...
Platform P4 (ARM64/macOS):   graph=a1b2c3... results=d4e5f6... vectors=g7h8i9...
Platform P5 (ARM64/Linux):   graph=a1b2c3... results=d4e5f6... vectors=g7h8i9...
Platform P6 (WASM32):        graph=a1b2c3... results=d4e5f6... vectors=g7h8i9...
                                    ✅ ALL IDENTICAL
```

### 3.4 Edge Cases to Test

1. **Subnormal numbers:** Vectors with very small values near I64F32 precision limit
2. **Large values:** Vectors near I64F32 overflow boundary
3. **Zero vectors:** All-zero vectors (degenerate case)
4. **Identical vectors:** Multiple copies of same vector
5. **Adversarial vectors:** Vectors designed to maximize distance computation error
6. **Different optimization levels:** -O0 vs -O2 vs -O3 (should be identical for integer ops)

---

## 4. Scalability Experiment Design

### 4.1 Dataset Size Sweep

| Size | Vectors | Expected Build Time | Expected Memory |
|------|---------|-------------------|-----------------|
| 100K | 100,000 | ~5s | ~50 MB |
| 500K | 500,000 | ~30s | ~256 MB |
| 1M | 1,000,000 | ~60s | ~512 MB |
| 5M | 5,000,000 | ~400s | ~2.5 GB |
| 10M | 10,000,000 | ~900s | ~5 GB |

### 4.2 Metrics to Track

- Build time (linear? superlinear?)
- Query latency (should be O(log n))
- Memory per vector (should be constant)
- Recall stability (should be constant)

### 4.3 Scalability Plot

```
X-axis: Number of vectors (log scale)
Y-axis (left): Build time (seconds)
Y-axis (right): Query latency (μs)
Lines: D-HNSW, hnswlib
Expected: Both scale similarly, D-HNSW with constant overhead factor
```

---

## 5. Blockchain Integration Experiment (Bonus)

### 5.1 On-Chain Verification Cost

Simulate the cost of verifying D-HNSW results on Ethereum:

```rust
pub struct OnChainCostAnalysis {
    /// Gas cost for storing one I64F32 vector (d dimensions)
    pub storage_gas_per_vector: u64,  // SSTORE: 20,000 gas per 32 bytes
    
    /// Gas cost for verifying one query result
    pub verification_gas: u64,  // Depends on proof structure
    
    /// Dollar cost at current gas prices
    pub usd_cost_per_query: f64,
}

impl OnChainCostAnalysis {
    pub fn compute(d: usize, gas_price_gwei: f64, eth_price_usd: f64) -> Self {
        // I64F32: 8 bytes per dimension
        let bytes_per_vector = d * 8;
        let storage_slots = (bytes_per_vector + 31) / 32;  // 32 bytes per slot
        let storage_gas = storage_slots as u64 * 20_000;
        
        // Verification: re-compute distance for top-K results
        // Each distance: ~d multiplications + d additions
        // Approximate gas: d * 8 (MUL=5, ADD=3) per distance
        let verification_gas = 10 * (d as u64) * 8;  // 10 distances for top-10
        
        let gas_cost_eth = (verification_gas as f64) * gas_price_gwei * 1e-9;
        let usd_cost = gas_cost_eth * eth_price_usd;
        
        Self {
            storage_gas_per_vector: storage_gas,
            verification_gas,
            usd_cost_per_query: usd_cost,
        }
    }
}
```

### 5.2 Comparison: D-HNSW vs ANNProof

| Metric | D-HNSW | ANNProof |
|--------|--------|----------|
| Verification method | Re-execute (byte-equal) | Merkle proof |
| Verification gas | ~10,240 (d=128) | ~50,000 (Merkle path) |
| Storage overhead | 0% (index IS the proof) | ~2% (Merkle tree) |
| Trust model | Deterministic re-execution | Cryptographic proof |
| Hardware requirement | Any (integer ALU) | Same as prover |

---

## 6. Paper Figure Plan

### Required Figures (Minimum for TKDE)

| Fig # | Content | Type | Section |
|-------|---------|------|---------|
| 1 | D-HNSW architecture overview | Diagram | §4 |
| 2 | Determinism Trilemma illustration | Diagram | §3 |
| 3 | QPS vs Recall (SIFT-1M) | Line plot | §7.1 |
| 4 | QPS vs Recall (GIST-1M) | Line plot | §7.1 |
| 5 | QPS vs Recall (GloVe-100) | Line plot | §7.1 |
| 6 | Ablation overhead breakdown | Stacked bar | §7.2 |
| 7 | Distance error distribution | Histogram | §7.3 |
| 8 | Cross-platform hash comparison | Table | §7.4 |
| 9 | Memory comparison | Bar chart | §7.5 |
| 10 | Scalability (build time + query latency) | Dual-axis line | §7.6 |
| 11 | Optimization impact progression | Bar chart | §7.7 |

### Optional Figures (Strengthen the Paper)

| Fig # | Content | Type |
|-------|---------|------|
| 12 | Edge reversal probability vs precision | Log-scale plot |
| 13 | I64F32 vs Q16.16 error comparison | Dual histogram |
| 14 | Blockchain verification cost analysis | Bar chart |
| 15 | Latency distribution (boxplot) | Boxplot |


---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
