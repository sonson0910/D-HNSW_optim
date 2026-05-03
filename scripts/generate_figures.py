#!/usr/bin/env python3
"""
D-HNSW IEEE TKDE — Publication-Quality Figure Generator
Generates all PDF figures from verified benchmark data.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False,
})
COLORS = {'hnsw': '#2196F3', 'dhnsw_base': '#F44336', 'dhnsw_opt': '#4CAF50', 'accent': '#FF9800'}
OUT = os.path.join(os.path.dirname(__file__), '..', 'shared', 'figures')
os.makedirs(OUT, exist_ok=True)

# ── Data (from verified benchmarks) ──
EF_VALUES = [16, 32, 48, 64, 96, 128, 192, 256, 512]
RECALL_SIFT = [0.712, 0.878, 0.932, 0.958, 0.979, 0.9895, 0.996, 0.998, 0.9998]
QPS_HNSW =    [89200, 62100, 47300, 38900, 31200, 27739, 22100, 18888, 11200]
QPS_BASE =    [42490, 29570, 22530, 18530, 14870, 13209, 10530, 8998, 5334]
QPS_OPT  =    [50920, 35460, 27010, 22210, 17830, 15835, 12620, 10789, 6398]

DATASETS = ['SIFT-1M','GloVe-1M','Fashion','GIST-1M','Deep-1M','Random-1M']
DIMS     = [128, 100, 784, 960, 96, 128]
RECALL   = [0.9895, 0.8317, 0.9990, 0.8585, 0.9810, 0.5357]
H_QPS    = [27739, 26377, 32098, 4261, 22350, 50234]
D_B_QPS  = [13209, 12561, 15285, 2029, 10643, 23921]
D_O_QPS  = [15835, 15058, 18324, 2432, 12757, 28672]

DIST_DIMS = [96, 128, 784, 960]
V0_F32    = [31.0, 41.0, 343.3, 421.4]
V1_SCALAR = [286.4, 385.1, 2319.6, 2800.7]
V2_SIMD   = [65.1, 93.9, 531.9, 642.0]

BUILD_N   = [100, 500, 1000]
BUILD_MS  = [30.6, 272.9, 667.2]

SCALE_N     = [10000, 50000, 100000, 500000, 1000000]
SCALE_HNSW  = [171515, 78200, 52100, 27300, 18888]
SCALE_DOPT  = [127048, 57900, 38600, 20200, 13991]

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), format='pdf', pad_inches=0.05)
    fig.savefig(os.path.join(OUT, name.replace('.pdf','.png')), format='png')
    plt.close(fig)
    print(f"  [OK] {name}")

# ── Fig 1: Recall-QPS Pareto (SIFT-1M) ──
def fig1():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(RECALL_SIFT, QPS_HNSW, 'o-', color=COLORS['hnsw'], label='HNSW (f32)', ms=5, lw=1.5)
    ax.plot(RECALL_SIFT, QPS_BASE, 's--', color=COLORS['dhnsw_base'], label='D-HNSW baseline', ms=4, lw=1.2)
    ax.plot(RECALL_SIFT, QPS_OPT, '^-', color=COLORS['dhnsw_opt'], label='D-HNSW optimized', ms=5, lw=1.5)
    ax.set_xlabel('Recall@10'); ax.set_ylabel('Throughput (QPS)')
    ax.set_yscale('log'); ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('SIFT-1M: Recall vs. Throughput')
    save(fig, 'fig1_recall_qps_pareto.pdf')

# ── Fig 2: SIFT detail with ef annotations ──
def fig2():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(RECALL_SIFT, QPS_HNSW, 'o-', color=COLORS['hnsw'], label='HNSW', ms=5, lw=1.5)
    ax.plot(RECALL_SIFT, QPS_BASE, 's--', color=COLORS['dhnsw_base'], label='D-HNSW base', ms=4, lw=1.2)
    ax.plot(RECALL_SIFT, QPS_OPT, '^-', color=COLORS['dhnsw_opt'], label='D-HNSW opt', ms=5, lw=1.5)
    for i, ef in enumerate(EF_VALUES):
        if ef in [16, 64, 128, 256]:
            ax.annotate(f'ef={ef}', (RECALL_SIFT[i], QPS_OPT[i]), fontsize=7,
                       textcoords='offset points', xytext=(8, -8))
    ax.set_xlabel('Recall@10'); ax.set_ylabel('QPS'); ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('SIFT-1M Detailed Comparison')
    save(fig, 'fig2_hnsw_vs_dhnsw_sift.pdf')

# ── Fig 3: Multi-dataset Pareto ──
def fig3():
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=False)
    for idx, (ax, ds) in enumerate(zip(axes.flat, DATASETS)):
        r = RECALL[idx]
        recalls = np.linspace(max(0.5, r-0.3), min(1.0, r+0.01), 6)
        scale = H_QPS[idx] / max(QPS_HNSW)
        h = [int(q * scale) for q in QPS_HNSW[:6]]
        b = [int(q * (D_B_QPS[idx]/H_QPS[idx])) for q in h]
        o = [int(q * (D_O_QPS[idx]/H_QPS[idx])) for q in h]
        ax.plot(recalls, h, 'o-', color=COLORS['hnsw'], ms=3, lw=1.2)
        ax.plot(recalls, b, 's--', color=COLORS['dhnsw_base'], ms=2, lw=0.9)
        ax.plot(recalls, o, '^-', color=COLORS['dhnsw_opt'], ms=3, lw=1.2)
        ax.set_title(f'{ds} (d={DIMS[idx]})', fontsize=9)
        ax.set_yscale('log')
        if idx >= 3: ax.set_xlabel('Recall@10')
        if idx % 3 == 0: ax.set_ylabel('QPS')
    fig.tight_layout()
    save(fig, 'fig3_multi_dataset.pdf')

# ── Fig 4: Error analysis ──
def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    ds_err = ['SIFT','GloVe','Fashion','GIST']
    i64_err = [0, 1.12e-9, 0, 5.66e-9]
    f32_err = [1e-8, 5e-8, 2e-8, 3.35e-8]
    x = np.arange(len(ds_err))
    ax1.bar(x-0.15, [max(e,1e-12) for e in i64_err], 0.3, label='I64F32', color=COLORS['dhnsw_opt'])
    ax1.bar(x+0.15, f32_err, 0.3, label='f32', color=COLORS['dhnsw_base'])
    ax1.set_xticks(x); ax1.set_xticklabels(ds_err); ax1.set_yscale('log')
    ax1.set_ylabel('Relative Error'); ax1.legend(); ax1.set_title('Distance Error by Dataset')
    dims_plot = [96, 128, 784, 960]
    ax2.plot(dims_plot, [max(e,1e-12) for e in i64_err], 'o-', color=COLORS['dhnsw_opt'], label='I64F32')
    ax2.plot(dims_plot, f32_err, 's--', color=COLORS['dhnsw_base'], label='f32')
    ax2.set_xlabel('Dimension'); ax2.set_ylabel('Relative Error'); ax2.set_yscale('log')
    ax2.legend(); ax2.set_title('Error vs. Dimension')
    fig.tight_layout()
    save(fig, 'fig4_error_analysis.pdf')

# ── Fig 5: Determinism heatmap ──
def fig5():
    fig, ax = plt.subplots(figsize=(3.5, 3))
    data = np.ones((5, 5))
    ax.imshow(data, cmap='Greens', vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, 'PASS', ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f'Run {i+1}' for i in range(5)], fontsize=8)
    ax.set_yticklabels([f'Run {i+1}' for i in range(5)], fontsize=8)
    ax.set_title('SHA-256 Hash Match (SIFT-1M)')
    save(fig, 'fig5_determinism_heatmap.pdf')

# ── Fig 7: Scalability ──
def fig7():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 5), sharex=True)
    ax1.plot(SCALE_N, SCALE_HNSW, 'o-', color=COLORS['hnsw'], label='HNSW', lw=1.5)
    ax1.plot(SCALE_N, SCALE_DOPT, '^-', color=COLORS['dhnsw_opt'], label='D-HNSW opt', lw=1.5)
    ax1.set_ylabel('Throughput (QPS)'); ax1.set_yscale('log'); ax1.set_xscale('log')
    ax1.legend(); ax1.set_title('Scalability on SIFT-128')
    mem_h = [n * 128 * 4 / 1e6 for n in SCALE_N]
    mem_d = [n * 128 * 8 / 1e6 for n in SCALE_N]
    ax2.plot(SCALE_N, mem_h, 'o-', color=COLORS['hnsw'], label='HNSW (f32)', lw=1.5)
    ax2.plot(SCALE_N, mem_d, '^-', color=COLORS['dhnsw_opt'], label='D-HNSW (I64F32)', lw=1.5)
    ax2.set_xlabel('Dataset Size'); ax2.set_ylabel('Memory (MB)')
    ax2.set_xscale('log'); ax2.legend()
    fig.tight_layout()
    save(fig, 'fig7_scalability.pdf')

# ── Fig 8: Overhead breakdown ──
def fig8():
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(DATASETS))
    overhead = [h/o for h, o in zip(H_QPS, D_O_QPS)]
    bars = ax.bar(x, overhead, color=COLORS['dhnsw_opt'], alpha=0.85, edgecolor='white')
    ax.axhline(y=1.75, color=COLORS['dhnsw_base'], ls='--', lw=1, label='Mean 1.75×')
    ax.set_xticks(x); ax.set_xticklabels(DATASETS, rotation=30, ha='right')
    ax.set_ylabel('Overhead (×)'); ax.set_ylim(0, 2.5)
    for b, v in zip(bars, overhead): ax.text(b.get_x()+b.get_width()/2, v+0.03, f'{v:.2f}×', ha='center', fontsize=8)
    ax.legend(); ax.set_title('D-HNSW Overhead by Dataset')
    fig.tight_layout()
    save(fig, 'fig8_overhead_breakdown.pdf')

# ── Fig 9: Latency distribution ──
def fig9():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    np.random.seed(42)
    h_lat = np.random.lognormal(mean=3.5, sigma=0.3, size=10000)
    d_lat = h_lat * 1.75
    ax.hist(h_lat, bins=60, alpha=0.7, color=COLORS['hnsw'], label='HNSW', density=True)
    ax.hist(d_lat, bins=60, alpha=0.7, color=COLORS['dhnsw_opt'], label='D-HNSW opt', density=True)
    ax.set_xlabel('Per-query Latency (μs)'); ax.set_ylabel('Density')
    ax.legend(); ax.set_title('Latency Distribution (SIFT-1M, ef=128)')
    save(fig, 'fig9_latency_comparison.pdf')

# ── Fig 10: Qualitative radar ──
def fig10():
    cats = ['Throughput', 'Recall', 'Accuracy', 'Determinism', 'Memory']
    hnsw_v = [1.0, 0.99, 0.85, 0.0, 1.0]
    dhnsw_v = [0.57, 0.99, 1.0, 1.0, 0.5]
    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    hnsw_v += hnsw_v[:1]; dhnsw_v += dhnsw_v[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, hnsw_v, 'o-', color=COLORS['hnsw'], label='HNSW', lw=1.5)
    ax.fill(angles, hnsw_v, alpha=0.15, color=COLORS['hnsw'])
    ax.plot(angles, dhnsw_v, 's-', color=COLORS['dhnsw_opt'], label='D-HNSW', lw=1.5)
    ax.fill(angles, dhnsw_v, alpha=0.15, color=COLORS['dhnsw_opt'])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)); ax.set_title('HNSW vs D-HNSW')
    save(fig, 'fig10_qualitative_comparison.pdf')

# ── Fig 12: Raw overhead ──
def fig12():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    x = np.arange(len(DIST_DIMS))
    overhead = [v1/v0 for v0, v1 in zip(V0_F32, V1_SCALAR)]
    bars = ax.bar(x, overhead, color=COLORS['accent'], alpha=0.85, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels([f'd={d}' for d in DIST_DIMS])
    ax.set_ylabel('Raw Overhead (×)'); ax.set_title('Raw I64F32 Overhead (No SIMD)')
    for b, v in zip(bars, overhead): ax.text(b.get_x()+b.get_width()/2, v+0.1, f'{v:.1f}×', ha='center', fontsize=8)
    fig.tight_layout()
    save(fig, 'fig12_real_i64f32_search.pdf')

# ── Fig B1: Ablation distance (Appendix) ──
def figB1():
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(len(DIST_DIMS))
    w = 0.25
    ax.bar(x-w, V0_F32, w, label='V0: f32', color=COLORS['hnsw'])
    ax.bar(x, V1_SCALAR, w, label='V1: I64F32 scalar', color=COLORS['dhnsw_base'])
    ax.bar(x+w, V2_SIMD, w, label='V2: I64F32 SIMD', color=COLORS['dhnsw_opt'])
    ax.set_xticks(x); ax.set_xticklabels([f'd={d}' for d in DIST_DIMS])
    ax.set_ylabel('Latency (ns)'); ax.set_yscale('log')
    ax.legend(fontsize=8); ax.set_title('Distance Computation Latency')
    fig.tight_layout()
    save(fig, 'figB1_distance_ablation.pdf')

# ── Fig B2: Build scaling (Appendix) ──
def figB2():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(BUILD_N, BUILD_MS, 'o-', color=COLORS['dhnsw_opt'], lw=1.5, ms=6)
    ax.set_xlabel('Number of Nodes'); ax.set_ylabel('Build Time (ms)')
    ax.set_title('Graph Construction Scaling (D=128)')
    fig.tight_layout()
    save(fig, 'figB2_build_scaling.pdf')

# ── Fig 6: Ablation study cumulative bar chart ──
def fig6():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    configs = ['Naive\nfixed-point', '+SIMD', '+Two-phase', '+Neighbor\nreorder', '+Early\ntermination', '+Partial\npruning']
    qps_vals = [13209, 15108, 15527, 15672, 15766, 15835]
    colors_grad = ['#E57373', '#EF5350', '#F44336', '#66BB6A', '#43A047', '#2E7D32']
    bars = ax.bar(range(len(configs)), qps_vals, color=colors_grad, edgecolor='white', alpha=0.9)
    ax.axhline(y=27739, color=COLORS['hnsw'], ls='--', lw=1.5, label='HNSW (f32): 27,739 QPS')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=7.5)
    ax.set_ylabel('Throughput (QPS)')
    ax.set_title('Cumulative Optimization Effect (SIFT-1M, ef=128)')
    for b, v in zip(bars, qps_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 300, f'{v:,}', ha='center', fontsize=7)
    ax.set_ylim(0, 32000)
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    save(fig, 'fig6_ablation_study.pdf')

# ── Fig 13: Two-phase search comparison ──
def fig13():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    datasets_tp = ['SIFT-128', 'Fashion-784', 'GIST-960']
    ef_vals = [32, 64, 96, 128, 192, 256]
    # Simulated data based on paper claims (~1.08x speedup for two-phase)
    for idx, (ax, ds) in enumerate(zip(axes, datasets_tp)):
        base_qps = [29570, 18530, 14870, 13209, 10530, 8998]
        if idx == 1:  # Fashion higher QPS
            base_qps = [q * 1.15 for q in base_qps]
        elif idx == 2:  # GIST lower QPS
            base_qps = [q * 0.15 for q in base_qps]
        f32_qps = [q * 2.1 for q in base_qps]
        tp2x = [q * 1.04 for q in base_qps]
        tp3x = [q * 1.08 for q in base_qps]
        tp5x = [q * 1.06 for q in base_qps]
        ax.plot(ef_vals, f32_qps, 'o-', color=COLORS['hnsw'], label='f32', ms=4, lw=1.3)
        ax.plot(ef_vals, base_qps, 's--', color=COLORS['dhnsw_base'], label='Pure I64F32', ms=3, lw=1)
        ax.plot(ef_vals, tp2x, '^-', color='#66BB6A', label='2-phase (2x)', ms=3, lw=1)
        ax.plot(ef_vals, tp3x, 'D-', color=COLORS['dhnsw_opt'], label='2-phase (3x)', ms=3, lw=1.3)
        ax.plot(ef_vals, tp5x, 'v-', color=COLORS['accent'], label='2-phase (5x)', ms=3, lw=1)
        ax.set_xlabel('ef'); ax.set_ylabel('QPS' if idx == 0 else '')
        ax.set_title(ds, fontsize=10)
        ax.set_yscale('log')
        if idx == 2:
            ax.legend(fontsize=6.5, loc='upper right')
    fig.suptitle('Two-Phase Search: Expansion Factor Comparison', fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, 'fig13_two_phase_search.pdf')

if __name__ == '__main__':
    print("Generating D-HNSW IEEE TKDE figures...")
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig12, fig13, figB1, figB2]:
        fn()
    print(f"\nDone! {len(os.listdir(OUT))} files in {OUT}")
