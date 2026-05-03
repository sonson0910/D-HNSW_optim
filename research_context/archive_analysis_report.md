# Archive Analysis Report: LuxTensor D-HNSW Project

**Status: completed**

## 1. Overview

The archive `upload/project-ec00e669-workspace.tar.gz` contains a complete research project for **D-HNSW (Deterministic Hierarchical Navigable Small World)** — a novel approach to integrating approximate nearest neighbor (ANN) search directly into an EVM-compatible blockchain called **LuxTensor**. The project includes:

- **8 Rust source files** (6,091 lines total) implementing the blockchain-native D-HNSW system
- **A complete IEEE TKDE-format LaTeX paper** (73 KB main.tex + 24 figures + references)
- **Experiment data, plots, and design documents** from prior research phases

## 2. Project Structure

### 2.1 Rust Source Code (`shared/rust_source/`)

| File | Lines | Purpose |
|------|-------|---------|
| `ai_precompiles.rs` | ~1,800 | EVM precompiled contracts (addresses 0x10–0x14) for vector operations |
| `graph.rs` | ~750 | Core `HnswGraph<D>` data structure with layered navigation |
| `vector_store.rs` | ~600 | Persistent vector storage with serde/bincode serialization |
| `unified_state.rs` | ~650 | `UnifiedStateDB` integrating D-HNSW into revm's EVM state |
| `revm_integration.rs` | ~450 | Bridge between revm (Rust EVM) and D-HNSW operations |
| `semantic_registry.rs` | ~380 | Name-to-vector semantic registry for on-chain AI |
| `fixed_point.rs` | ~330 | `FixedI64<U32>` (I64F32) arithmetic for consensus-safe computation |
| `deterministic_rng.rs` | ~230 | Keccak-256 seeded PRNG for deterministic layer assignment |

### 2.2 Key Architecture Decisions

1. **Fixed-point arithmetic (I64F32)**: All vector math uses `FixedI64<U32>` from the `fixed` crate instead of floating-point, ensuring bit-exact reproducibility across all validator nodes — critical for blockchain consensus.

2. **Deterministic RNG**: Layer assignment in HNSW uses `Keccak256(TxHash XOR BlockHash)` as seed, making the graph structure fully deterministic and verifiable.

3. **EVM Integration via Precompiles**: Five precompiled contracts at addresses `0x10`–`0x14` expose vector operations (insert, search, cosine similarity, etc.) to smart contracts.

4. **revm + CANCUN spec**: Built on the Rust EVM implementation (`revm`) targeting the CANCUN hard fork specification.

5. **Soft deletion with tombstones**: Nodes are marked deleted rather than removed, preserving graph connectivity.

6. **Capacity**: `MAX_CAPACITY = 5,000,000` nodes per graph instance.

7. **Standard embedding dimension**: `dim = 768` (BERT/LLM compatible).

### 2.3 LaTeX Paper (`agent_archive_analysis_ccfb7c55_workdir/latex_build/`)

The paper is formatted for **IEEE Transactions on Knowledge and Data Engineering (TKDE)** and covers:

- **Problem**: Traditional ANN indices (HNSW, FAISS) use floating-point math incompatible with blockchain determinism requirements
- **Solution**: D-HNSW with fixed-point arithmetic, deterministic RNG, and EVM precompile integration
- **Experiments**: Benchmarks on SIFT-1M, GloVe-200, MNIST-784, and Deep-96 datasets
- **Results**: 97.2% recall@10 with only 8–12% overhead vs. standard HNSW; bit-exact determinism across nodes
- **13 figures** including Pareto curves, ablation studies, scalability analysis, and latency comparisons

### 2.4 Compiled PDF

The paper has been compiled to PDF: `shared/D_HNSW_IEEE_TKDE_v2.pdf` (14 pages, ~1 MB).

Five fixes were applied during compilation:
1. Cosine similarity formula split into multi-line to avoid column overflow
2. Fig. 9 x-axis labels rotated 30° to prevent overlap
3. "Our contribution" paragraph rewritten for academic clarity
4. "Remainder of paper" paragraph rewritten
5. "Surprising finding" sentence rewritten

## 3. Reusable Resources

### 3.1 For Other Agents

| Resource | Location | Description |
|----------|----------|-------------|
| Rust D-HNSW implementation | `shared/rust_source/` | 8 production-quality Rust files |
| Compiled paper PDF | `shared/D_HNSW_IEEE_TKDE_v2.pdf` | 14-page IEEE TKDE paper |
| LaTeX source | `agent_archive_analysis_ccfb7c55_workdir/latex_build/` | Full compilable LaTeX project |
| Experiment figures | `agent_archive_analysis_ccfb7c55_workdir/latex_build/figures/` | 24 publication-quality figures (PDF + PNG) |
| Modal compilation script | `agent_archive_analysis_ccfb7c55_workdir/compile_latex_modal.py` | Reusable LaTeX→PDF compiler via Modal |
| Academic plotting skills | `skills/academic-plotting/` | Matplotlib templates for research figures |
| ML paper writing skills | `skills/ml-paper-writing/` | LaTeX templates and writing guidelines |

### 3.2 Key Dependencies (Rust)

- `fixed` crate: `FixedI64<U32>` for consensus-safe arithmetic
- `serde` + `bincode`: Serialization/deserialization
- `parking_lot::RwLock`: Concurrent access to graph structures
- `revm`: Rust EVM implementation (CANCUN spec)
- `sha3` (Keccak-256): Deterministic seeding

### 3.3 Key Dependencies (LaTeX)

- `IEEEtran` document class (from `texlive-publishers`)
- `amsmath`, `amssymb`, `algorithmic`, `graphicx`, `booktabs`, `multirow`
- BibTeX with `IEEEtran.bst` style

## 4. Methodology Notes

- **Datasets**: SIFT-1M (128d), GloVe-200 (200d), MNIST-784 (784d), Deep-96 (96d)
- **Metrics**: Recall@K, QPS (queries per second), determinism verification, overhead breakdown
- **Baselines**: Standard HNSW (floating-point), FAISS, ScaNN, Annoy
- **Key finding**: Fixed-point I64F32 achieves 97.2% recall@10 vs. 98.1% for float32 HNSW — only 0.9% gap with full determinism guarantee

## 5. Compilation Instructions

To recompile the LaTeX paper:

```bash
cd /home/orchestra/projects/e7e8b3bb-52f8-43e4-b21f-bc193659e332
modal run agent_archive_analysis_ccfb7c55_workdir/compile_latex_modal.py
```

The script uploads all LaTeX sources to Modal, compiles with `pdflatex` + `bibtex` (3 passes), and downloads the resulting PDF.

## 6. Summary

This project represents a complete research contribution combining:
- **Systems engineering** (Rust blockchain integration)
- **Algorithm design** (deterministic HNSW variant)
- **Academic writing** (IEEE TKDE submission-ready paper)

The codebase is well-structured, production-quality Rust with comprehensive error handling, and the paper includes thorough experimental evaluation across multiple datasets and metrics.


---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
