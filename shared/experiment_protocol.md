## V. Experimental Evaluation Protocol

This section outlines the rigorous experimental protocol designed to empirically validate the mathematical claims of D-HNSW. The evaluation is split into two primary phases: Topological Integrity (measuring routing divergence) and Performance Ablation (isolating computational overhead).

### A. Experimental Setup & Dataset Configuration

To align with modern decentralized AI use cases (e.g., On-chain RAG), the experiments abandon low-dimensional datasets like SIFT-1M in favor of high-dimensional LLM embeddings.

**1. Datasets:**
*   **Dataset A (Dense - 768D):** 100,000 document embeddings generated via `sentence-transformers/all-mpnet-base-v2`. Normalized to unit length (Cosine Similarity).
*   **Dataset B (Sparse/High-Dim - 1536D):** 100,000 vectors from OpenAI `text-embedding-3-small` dataset.

**2. Hardware Environment:**
*   Platform A: AMD Ryzen 9 5900X (x86_64), 64 GB RAM (L3 Cache: 64MB)
*   Platform B: Apple M2 (ARM64), 16 GB RAM (L2 Cache: 16MB)

**3. HNSW Hyperparameters:**
*   $M = 16, M_0 = 32$
*   $ef_{construction} = 200, ef_{search} = 100$

### B. Phase 1: Topological Integrity (Routing Divergence)

**Objective:** Prove that the $10^{-7}$ maximum quantization error established in Theorem 2 does not cause the HNSW greedy search to traverse a different graph branch compared to standard `f32` floating-point execution.

**Measurement Metrics:**
We define the *Path Overlap Intersection-over-Union (IoU)* for a given query $q$:
$$ IoU_{path}(q) = \frac{| \text{Visited}_{float}(q) \cap \text{Visited}_{fixed}(q) |}{| \text{Visited}_{float}(q) \cup \text{Visited}_{fixed}(q) |} $$
where $\text{Visited}(q)$ is the set of Node IDs whose distances were computed during the search phase.

**Implementation Guide (Rust Tracking):**
In the `search_layer` function of both implementations, inject a tracking mechanism:
```rust
// Inject into HNSW Search Loop
let mut visited_nodes: HashSet<usize> = HashSet::new();

while let Some(current) = candidates.pop() {
    for neighbor_id in graph.get_neighbors(current.id) {
        if visited_nodes.insert(neighbor_id) { // Returns true if new
            let dist = squared_distance(&query, &graph.get_vector(neighbor_id));
            // ... continue HNSW logic
        }
    }
}
// Output visited_nodes for IoU calculation
```

**Table 3: Topological Integrity and Routing Overlap (N=100K)**

| Dataset | Dimension | Float Visited Nodes (Avg) | D-HNSW Visited Nodes (Avg) | Path Overlap (IoU) | Recall@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MPNet-Base | 768 | [To be filled] | [To be filled] | > 99.9% | [To be filled] |
| OpenAI-Ada | 1536 | [To be filled] | [To be filled] | > 99.9% | [To be filled] |

*Expected finding:* The IoU should be near 100%, proving that fixed-point arithmetic preserves the exact search trajectory.

### C. Phase 2: Performance Ablation Study

**Objective:** Isolate and quantify the exact sources of latency in D-HNSW by benchmarking against intermediate proxy configurations.

**Ablation Variants:**
*   **V0 (Baseline):** `f32` arithmetic + `rand::thread_rng()`
*   **V1 (Memory Proxy):** `f64` arithmetic + `rand::thread_rng()`. This isolates the cost of doubling the memory bandwidth (from 4 bytes to 8 bytes per dimension).
*   **V2 (ALU Isolate):** `I64F32` arithmetic + `rand::thread_rng()`. Compared against V1, this isolates the cost of saturating integer arithmetic vs. hardware FMA.
*   **V3 (D-HNSW):** `I64F32` + `Keccak256`. Compared against V2, this isolates the cryptographic RNG overhead.

**Profiling Toolkit:**
*   Latency: `criterion.rs` for statistically significant micro-benchmarking.
*   Hardware Counters: `perf stat -e L1-dcache-load-misses,cycles` (Linux) to measure cache efficiency.

**Table 4: Ablation Study of D-HNSW Overhead Components (N=100K, D=768)**

| Variant | Data Type | RNG | Mem/Vec | Insert Latency ($\mu s$) | Search Latency ($\mu s$) | L1 Miss Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| V0 (Baseline) | `f32` | Thread | ~3.1 KB | [Data] | [Data] | [Data]% |
| V1 (Memory Proxy) | `f64` | Thread | ~6.1 KB | [Data] (+ $\Delta$) | [Data] (+ $\Delta$) | [Data]% |
| V2 (ALU Isolate) | `I64F32` | Thread | ~6.1 KB | [Data] (+ $\Delta$) | [Data] (+ $\Delta$) | [Data]% |
| V3 (D-HNSW) | `I64F32` | Keccak | ~6.1 KB | [Data] (+ $\Delta$) | [Data] (+ $\Delta$) | [Data]% |

*Guidance for writing the analysis:* Use $\Delta(V1 - V0)$ to argue memory/cache bottleneck. Use $\Delta(V2 - V1)$ to argue the absence of hardware integer SIMD/FMA acceleration. Use $\Delta(V3 - V2)$ to calculate consensus-seeding cost (which only affects insertion, not search).

---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
