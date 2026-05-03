# D-HNSW: Deterministic HNSW for Blockchain Consensus and Verifiable AI

## Artifact Evaluation Package

This artifact reproduces all experiments from the paper **"D-HNSW: A Deterministic HNSW Construction for Blockchain-Native Approximate Nearest Neighbor Search"**.

---

## Quick Start

### Prerequisites
- **Docker** ≥ 24.0 (recommended)
- OR: **Rust** ≥ 1.78 + **Python** ≥ 3.10

### Option 1: Docker (Recommended for Reviewers)

```bash
# Build the container (~15 minutes, includes cross-compilation)
docker build -t dhnsw-artifact .

# Full evaluation (~2-4 hours)
docker run --rm -v $(pwd)/output:/artifact/output dhnsw-artifact --full

# Quick smoke test (~10 minutes)
docker run --rm -v $(pwd)/output:/artifact/output dhnsw-artifact --quick

# Determinism verification only (~5 minutes)
docker run --rm -v $(pwd)/output:/artifact/output dhnsw-artifact --verify

# Cross-architecture test only (~15 minutes)
docker run --rm -v $(pwd)/output:/artifact/output dhnsw-artifact --cross

# Performance benchmarks only (~1 hour)
docker run --rm -v $(pwd)/output:/artifact/output dhnsw-artifact --bench

# Interactive shell for manual exploration
docker run --rm -it dhnsw-artifact bash
```

### Option 2: Docker Compose

```bash
# Full evaluation
docker compose up dhnsw-full

# Quick test
docker compose --profile quick run dhnsw-quick

# Determinism verification
docker compose --profile verify run dhnsw-verify
```

### Option 3: Native Build

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# CLI usage
./target/release/dhnsw build --input data.hdf5 --output index.bin -M 16
./target/release/dhnsw search --index index.bin --queries queries.hdf5 -k 10
./target/release/dhnsw verify --index index.bin
```

---

## Repository Structure

```
dhnsw-artifact/
├── Cargo.toml              # Rust project manifest
├── Dockerfile              # Multi-stage container (x86 + ARM64 + RISC-V)
├── docker-compose.yml      # Service profiles for different eval modes
├── README.md               # This file
│
├── src/
│   ├── main.rs             # CLI entry point (build/search/verify/bench)
│   ├── lib.rs              # Library root
│   ├── fixed_point.rs      # Q32.32 fixed-point arithmetic (Theorem 9)
│   ├── deterministic_rng.rs# Keccak-256 layer assignment (Algorithm 3)
│   └── graph.rs            # D-HNSW core: insert (Alg 5), search (Alg 4)
│
├── benches/
│   ├── ablation_benchmark.rs       # Q32.32 vs f32 distance computation
│   ├── cross_arch_verification.rs  # Index build + SHA-256 verification
│   └── recall_benchmark.rs         # Recall@10 at various ef values
│
├── tests/
│   └── determinism_tests.rs  # Theorem 10 verification + integration tests
│
├── experiments/
│   ├── run_recall_benchmark.py       # Recall@K sweep
│   ├── run_performance_benchmark.py  # Throughput measurement
│   └── generate_plots.py            # Publication figure generation
│
├── scripts/
│   └── run_all.sh          # Master evaluation runner
│
└── datasets/               # Downloaded automatically
    ├── sift-128-euclidean.hdf5      (~500MB)
    ├── glove-100-angular.hdf5       (~400MB)
    ├── fashion-mnist-784-euclidean.hdf5 (~30MB)
    └── gist-960-euclidean.hdf5      (~4GB, --full only)
```

---

## Claims Verified by This Artifact

| # | Claim (Paper Section) | Verification Method | Mode |
|---|---|---|---|
| C1 | **Bit-exact determinism** (Theorem 10, §V) | SHA-256 hash matching across 5 independent runs | `--verify` |
| C2 | **Cross-architecture reproducibility** (§VII.B) | x86-64 vs ARM64 vs RISC-V hash comparison via QEMU | `--cross` |
| C3 | **Recall@10 parity** with f32 HNSW (Table IV) | Recall benchmark on SIFT-1M, GloVe-100 | `--bench` |
| C4 | **≤1.75× overhead** vs f32 (Table V, Fig 8) | Construction/query latency comparison | `--bench` |
| C5 | **Gas cost advantage** over ZK-SNARK (Table VIII) | Analytical comparison + figure generation | `--full` |

---

## Key Design Decisions

### Q32.32 Fixed-Point Arithmetic
All distance computations use `I64F32` (32-bit integer + 32-bit fractional part).
This guarantees identical results on every CPU architecture because:
- Only integer add, subtract, multiply are used
- Two's complement arithmetic is standardized across all modern ISAs
- No IEEE-754 rounding modes, FMA fusion, or SIMD lane ordering affects results

### Keccak-256 Layer Assignment
Layer numbers are determined by `Keccak256(node_id || seed)` instead of `log(uniform())`.
This ensures:
- Deterministic layer structure across all platforms
- Resistance to adversarial layer manipulation
- No dependency on system PRNG state

### Saturating Arithmetic
All operations use `saturating_add`, `saturating_sub`, `saturating_mul` to prevent
integer overflow panics. Values clamp to `i64::MAX` / `i64::MIN` instead of wrapping
or panicking.

---

## Expected Output

After running `--full`, the `output/` directory contains:

```
output/
├── ARTIFACT_REPORT.md          # Auto-generated summary report
├── determinism_result.txt      # PASS/FAIL
├── cross_arch_result.txt       # Per-architecture SHA-256 hashes
├── figures/
│   ├── recall_vs_latency.pdf   # Figure 4
│   ├── overhead_analysis.pdf   # Figure 8
│   └── verification_comparison.pdf  # New: ZK vs D-HNSW comparison
├── tables/
│   ├── recall_results.json     # Raw recall data
│   └── performance_results.json # Raw performance data
├── hashes/
│   ├── hashes.txt              # 5-run SHA-256 consistency check
│   ├── index_x86_64.bin        # Native index
│   ├── index_arm64.bin         # ARM64 cross-compiled index
│   └── index_riscv64.bin       # RISC-V cross-compiled index
└── logs/
    └── experiment_*.log        # Full execution log
```

---

## Hardware Requirements

| Mode | RAM | CPU Time | Disk |
|---|---|---|---|
| `--quick` | 2 GB | ~10 min | 1 GB |
| `--verify` | 4 GB | ~15 min | 2 GB |
| `--bench` | 8 GB | ~1 hour | 5 GB |
| `--full` | 16 GB | ~2-4 hours | 10 GB |

---

## License

MIT OR Apache-2.0

## Citation

```bibtex
@inproceedings{dhnsw2026,
  title={D-HNSW: A Deterministic HNSW Construction for Blockchain-Native Approximate Nearest Neighbor Search},
  author={[Authors]},
  booktitle={Proceedings of [Conference]},
  year={2026}
}
```
