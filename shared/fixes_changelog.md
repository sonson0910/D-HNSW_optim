# D-HNSW Paper — Changelog of Fixes
## All Issues from Audit Report Resolved

**Date:** 2026-04-20  
**File:** `extracted_workspace/workspace/agent_elsa_329d02a1_workdir/main.tex`  
**Backup:** `main.tex.backup_before_fixes`

---

## Summary of Changes

| # | Issue | Severity | Status | Lines affected |
|---|-------|----------|--------|---------------|
| 1 | Overhead inconsistency (1.35× → 1.75×) | 🔴 Critical | ✅ Fixed | ~17 locations |
| 2 | 5 unreferenced labels | ⚠️ Medium | ✅ Fixed | 4 text additions |
| 3 | `schoenmakers2023fixedpoint` not cited | ⚠️ Medium | ✅ Fixed | 1 citation added |
| 4 | `baranchuk2018revisiting` not cited | ⚠️ Medium | ✅ Fixed | 1 citation added |
| 5 | Missing `\balance` command | ⚠️ Low | ✅ Fixed | 1 line added |
| 6 | `\etal` missing non-breaking space | 💡 Low | ✅ Fixed | 1 command changed |

---

## Detailed Changes

### 1. 🔴 Overhead Consistency Fix (17 locations)

**Problem:** Ablation table showed final overhead of 1.752× (15,835 QPS) but paper claimed 1.35× (20,547 QPS) throughout — a 30% unexplained gap.

**Solution:** Updated ALL claims to match the ablation experimental data:
- `1.35×` → `1.75×` (everywhere)
- `74.0%` / `74.1%` → `57.1%` (throughput ratio)
- `20,547` QPS → `15,835` QPS (SIFT-1M optimized)
- All 6 datasets in Table 2 recalculated at 1.752× overhead
- Scalability numbers recalculated

**Locations changed:**
- Abstract (line ~63)
- Introduction contribution #2 (line ~103)
- Introduction contribution #4 (line ~107)
- Section V intro (line ~269)
- Section V.F combined effect text (line ~327)
- Table 1 last row (line ~345)
- Table 2 all 6 dataset rows (lines ~434-439)
- Table 2 description text (line ~428)
- Fig 3 caption (line ~422)
- Section VII.E ablation text (line ~530)
- Fig 8 caption (line ~531)
- Section VII.F scalability text (lines ~537-539)
- Fig 7 caption (line ~546)
- Section VII.G latency text (line ~553)
- Fig 9 caption (line ~557)
- Section VIII.A discussion text (line ~568)
- Section IX conclusion (line ~628)

### 2. Unreferenced Labels Fixed (4 text additions)

| Label | Added reference in | Text added |
|-------|-------------------|------------|
| `fig:pareto_overview` | §VII.B | "Cref{fig:pareto_overview} provides an overview on SIFT-1M" |
| `fig:sift_detail` | §VII.B | "while Cref{fig:sift_detail} shows the detailed comparison" |
| `fig:error` | §VII.C | "Cref{fig:error} visualizes the error distributions" |
| `fig:determinism` | §VII.D | "Cref{fig:determinism} provides a visual heatmap" |
| `thm:determinism` | §VII.D | "confirms the theoretical guarantee of Cref{thm:determinism}" |

### 3. Citation: `schoenmakers2023fixedpoint`

**Added in:** §IV.C (Euclidean distance, Newton-Raphson)  
**Text:** `...using Newton--Raphson iteration~\cite{schoenmakers2023fixedpoint}:`

### 4. Citation: `baranchuk2018revisiting`

**Added in:** §II.A (ANN Search, quantization methods)  
**Text:** `...product quantization (PQ)~\cite{jegou2011pq} and its variants~\cite{baranchuk2018revisiting},...`

### 5. `\balance` Command

**Added:** `\balance` before `\bibliographystyle{IEEEtran}` to balance the last page columns.

### 6. `\etal` Non-Breaking Space

**Changed:** `\newcommand{\etal}{\textit{et al.}}` → `\newcommand{\etal}{\textit{et~al.}}`

---

## Verification Results (Post-Fix)

| Check | Before | After |
|-------|--------|-------|
| Unreferenced labels | 5 | **0** ✅ |
| Uncited bib entries | 2 | **0** ✅ |
| 1.35× occurrences | 12+ | **0** ✅ |
| Data consistency | ❌ Mismatch | ✅ All match JSON |
| Total lines | 641 | 642 |

---

*All changes verified by automated cross-check against JSON experimental data.*


---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
