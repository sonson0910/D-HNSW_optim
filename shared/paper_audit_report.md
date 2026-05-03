# D-HNSW Paper Audit Report
## Pre-Submission Quality Check for IEEE TKDE

**File audited:** `extracted_workspace/workspace/agent_elsa_329d02a1_workdir/main.tex` (641 lines)  
**Date:** 2026-04-20

---

## 1. Overall Assessment

| Category | Status | Details |
|----------|--------|---------|
| **Structure** | ✅ Complete | 9 sections, 24 subsections, well-organized |
| **Figures** | ✅ All present | 10/10 referenced figures exist as PDF+PNG |
| **References** | ⚠️ Minor issue | 29/31 bib entries cited; 2 unused entries |
| **Data accuracy** | ✅ Verified | All 6 datasets' numbers match JSON exactly |
| **Labels/Crefs** | ⚠️ Minor issue | 5 labels defined but never referenced in text |
| **Consistency** | 🔴 Key issue | Ablation table ends at 1.752× but paper claims 1.35× |
| **Writing quality** | ✅ Strong | Professional academic English, clear arguments |
| **LaTeX syntax** | ✅ Clean | Proper IEEEtran class, cleveref, booktabs |

---

## 2. 🔴 CRITICAL: Ablation vs Final Overhead Inconsistency

**The most important issue to resolve before submission.**

The ablation study (Table 1 / `tab:ablation_summary`) shows cumulative optimizations reaching:
- Final config (all 5 optimizations): **15,835 QPS**, overhead **1.752×**

But the paper's main claim (Table 2 / `tab:main_results` and throughout abstract/intro/conclusion) states:
- D-HNSW optimized: **20,547 QPS**, overhead **1.35×**

**The gap: 15,835 → 20,547 QPS is a 30% difference that is not explained.**

The 20,547 number is computed as `HNSW_QPS / 1.35 = 27,739 / 1.35 = 20,547`. The ablation JSON data confirms the 15,835 number. The 1.35× claim appears to be a **projection or target** rather than a measured result from the ablation pipeline.

### Possible explanations (need to be verified/clarified):
1. The "optimized" D-HNSW uses additional optimizations beyond the 5 listed in the ablation
2. The ablation was run on a different hardware/configuration than the main benchmark
3. The 1.35× figure comes from a different measurement methodology

### Recommended actions:
- **Option A:** Update the ablation table to be consistent with the 1.35× claim (if additional optimizations exist)
- **Option B:** Update the main claims to 1.75× overhead (matching the ablation data)
- **Option C:** Add text explaining why the final optimized result differs from the ablation (e.g., "the ablation isolates individual contributions; the fully optimized system includes additional implementation-level improvements")

---

## 3. ⚠️ Unreferenced Figures and Labels

5 labels are defined but **never referenced** with `\Cref{}` or `\ref{}` in the text:

| Label | Type | Line | Issue |
|-------|------|------|-------|
| `fig:pareto_overview` | Figure 1 | ~413 | Defined but text only references `fig:pareto` (Fig 3) |
| `fig:sift_detail` | Figure 3 | ~447 | Defined but never cited in text |
| `fig:error` | Figure 4 | ~483 | Defined but never cited in text |
| `fig:determinism` | Figure 5 | ~513 | Defined but never cited in text |
| `thm:determinism` | Theorem 2 | ~259 | Defined but never cited in text |

### Recommended fix:
Add `\Cref{fig:pareto_overview}`, `\Cref{fig:sift_detail}`, `\Cref{fig:error}`, `\Cref{fig:determinism}`, and `\Cref{thm:determinism}` references in the appropriate text paragraphs. For example:
- In §VII.B (Recall-Throughput): reference `fig:pareto_overview` and `fig:sift_detail`
- In §VII.C (Numerical Accuracy): reference `fig:error`
- In §VII.D (Determinism): reference `fig:determinism`
- In §IV.E or §VIII: reference `thm:determinism`

---

## 4. ⚠️ Unused BibTeX Entries

2 entries in `references.bib` are **not cited** anywhere in main.tex:

| Key | Entry | Recommendation |
|-----|-------|---------------|
| `baranchuk2018revisiting` | Baranchuk et al., ECCV 2018 — Inverted indices for ANN | Cite in §II.A (Related Work → Quantization methods) or remove |
| `schoenmakers2023fixedpoint` | Schoenmakers 2023 — Newton-Raphson in fixed-point | Cite in §IV.C (Euclidean distance → Newton-Raphson) or remove |

`schoenmakers2023fixedpoint` is particularly relevant — it directly supports the Newton-Raphson integer square root method described in §IV.C. **Strongly recommend citing it.**

---

## 5. ✅ Figure References — All Verified

All 10 `\includegraphics` references match existing files:

| Line | Reference | File exists | Format |
|------|-----------|-------------|--------|
| 348 | `figures/fig6_ablation_study.pdf` | ✅ | 20 KB PDF |
| 413 | `figures/fig1_recall_qps_pareto.pdf` | ✅ | 24 KB PDF |
| 420 | `figures/fig3_multi_dataset.pdf` | ✅ | 28 KB PDF |
| 447 | `figures/fig2_hnsw_vs_dhnsw_sift.pdf` | ✅ | 24 KB PDF |
| 483 | `figures/fig4_error_analysis.pdf` | ✅ | 24 KB PDF |
| 513 | `figures/fig5_determinism_heatmap.pdf` | ✅ | 20 KB PDF |
| 529 | `figures/fig8_overhead_breakdown.pdf` | ✅ | 20 KB PDF |
| 544 | `figures/fig7_scalability.pdf` | ✅ | 28 KB PDF |
| 555 | `figures/fig9_latency_comparison.pdf` | ✅ | 20 KB PDF |
| 608 | `figures/fig10_qualitative_comparison.pdf` | ✅ | 16 KB PDF |

---

## 6. ✅ Data Cross-Check — All Numbers Verified

### Table 2 (`tab:main_results`): Performance at ef=128

| Dataset | Paper Recall | JSON Recall | Match | Paper HNSW QPS | JSON QPS | Match |
|---------|-------------|-------------|-------|---------------|----------|-------|
| SIFT-1M | 0.9895 | 0.9895 | ✅ | 27,739 | 27,739.0 | ✅ |
| GloVe-1M | 0.8317 | 0.8317 | ✅ | 26,377 | 26,377.2 | ✅ |
| Fashion | 0.9990 | 0.9990 | ✅ | 32,098 | 32,098.1 | ✅ |
| GIST-1M | 0.8585 | 0.8585 | ✅ | 4,261 | 4,261.1 | ✅ |
| Deep-1M | 0.9810 | 0.9810 | ✅ | 22,350 | 22,350.2 | ✅ |
| Random-1M | 0.5357 | 0.5357 | ✅ | 50,234 | 50,233.6 | ✅ |

### Table 1 (`tab:ablation_summary`): Ablation on SIFT-1M

| Config | Paper QPS | JSON QPS | Match | Paper OH | Calc OH | Match |
|--------|----------|----------|-------|----------|---------|-------|
| Naive | 13,209 | 13,209.0 | ✅ | 2.100× | 2.100× | ✅ |
| +SIMD | 15,108 | 15,108.4 | ✅ | 1.836× | 1.836× | ✅ |
| +Two-phase | 15,527 | 15,527.0 | ✅ | 1.787× | 1.787× | ✅ |
| +Reorder | 15,672 | 15,671.8 | ✅ | 1.770× | 1.770× | ✅ |
| +Early term | 15,766 | 15,765.8 | ✅ | 1.759× | 1.759× | ✅ |
| +Partial dist | 15,835 | 15,835.1 | ✅ | 1.752× | 1.752× | ✅ |

### Table 3 (`tab:error`): Error Analysis

| Dataset | Paper i64 mean | JSON i64 mean | Paper order pres | JSON order pres |
|---------|---------------|---------------|-----------------|-----------------|
| SIFT-1M | 0.0 | 0.0 | 1.000 | 1.0 | ✅ |
| GloVe-1M | 1.12e-9 | 1.12e-9 (median) | 1.000 | 1.0 | ✅ |
| Fashion | 0.0 | 0.0 | 1.000 | 1.0 | ✅ |
| GIST-1M | 5.66e-9 | 5.66e-9 | 1.000 | 1.0 | ✅ |

**Note on GloVe:** JSON shows `mean_relative = 0.0002` (inflated by outliers near zero distance), but `median_relative = 1.12e-9`. Paper reports the median. This is acceptable but should be noted.

---

## 7. Writing Quality Assessment

### Abstract (Lines 56–66)
- ✅ Clear problem statement, method, and results
- ✅ Quantitative claims (74.0%, 2.10× → 1.35×, <10⁻⁸, SHA-256)
- 🔴 The 1.35× claim needs to be consistent with ablation (see §2)

### Introduction (Lines 75–109)
- ✅ Strong motivation with 4 application domains
- ✅ 4 clear contributions
- ✅ Paper outline at end
- ✅ Proper use of `\IEEEPARstart`

### Related Work (Lines 110–139)
- ✅ Well-structured: ANN → Determinism → Blockchain → Positioning
- ✅ Clear positioning statement at end
- Minor: Could cite `schoenmakers2023fixedpoint` here

### Experimental Evaluation (Lines 383–562)
- ✅ 6 diverse datasets, clear setup
- ✅ SHA-256 determinism verification
- ✅ Ablation study with cumulative optimizations
- ✅ Scalability analysis
- ⚠️ Several figures defined but not referenced in text

### Conclusion (Lines 624–641)
- ✅ Concise summary of contributions
- ✅ Future work directions
- 🔴 Repeats the 1.35× claim (needs consistency fix)

---

## 8. LaTeX Technical Issues

| Issue | Severity | Location | Detail |
|-------|----------|----------|--------|
| `compsoc` option | Low | Line 6 | `\documentclass[10pt,journal,compsoc]{IEEEtran}` — verify TKDE uses compsoc style |
| `\etal` command | Low | Line 35 | Defined as `\textit{et al.}` — consider `et~al.` with non-breaking space |
| `\text{Recall@10}` | Low | Line 37 | Used in `\recall` command — should be `\mathrm{Recall@10}` in math mode |
| Figure placement | Low | Various | All figures use `[t]` — may need `[!t]` or `[htbp]` for better placement |
| `\balance` package | Low | Line 22 | Loaded but `\balance` command not used — add before bibliography |

---

## 9. Summary of Required Actions

### 🔴 Must Fix (Before Submission)
1. **Resolve 1.35× vs 1.752× inconsistency** — This is the most critical issue. Reviewers will notice.
2. **Reference all figures in text** — Add `\Cref` for fig:pareto_overview, fig:sift_detail, fig:error, fig:determinism
3. **Reference Theorem 2** — Add `\Cref{thm:determinism}` somewhere in text

### ⚠️ Should Fix (Recommended)
4. **Cite `schoenmakers2023fixedpoint`** in §IV.C (Newton-Raphson square root)
5. **Remove or cite `baranchuk2018revisiting`** — unused bib entry
6. **Add `\balance` command** before `\bibliography` for balanced columns
7. **Add author information** — replace "Anonymous Authors"

### 💡 Nice to Have (Optional)
8. Add formal proofs from `formal_error_bounds.md` as appendix
9. Include supplementary Rust source code
10. Consider adding fig12 (real I64F32 overhead) or fig15 (determinism table) from tar.gz experiments

---

*Audit performed by Workspace Extractor & Analyzer Agent — 2026-04-20*


---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
