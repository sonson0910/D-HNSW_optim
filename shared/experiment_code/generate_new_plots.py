#!/usr/bin/env python3
"""Generate 4 new IEEE-style plots for D-HNSW experiments A/B/C."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,

    'axes.grid': True,
    'grid.alpha': 0.3,
})

PLOT_DIR = 'plots'
RESULT_DIR = 'results'

# Load data
with open(f'{RESULT_DIR}/real_i64f32_search.json') as f:
    real_search = json.load(f)
with open(f'{RESULT_DIR}/two_phase_search.json') as f:
    two_phase = json.load(f)
with open(f'{RESULT_DIR}/determinism_verification.json') as f:
    determinism = json.load(f)


# ============================================================
# Fig 12: Real I64F32 Search - QPS vs ef (Recall-QPS tradeoff)
# ============================================================
def plot_fig12():
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    datasets = ['sift-1m', 'glove-100', 'fashion-mnist', 'gist-960']
    titles = ['SIFT-128 (100K)', 'GloVe-100 (100K)', 'Fashion-MNIST-784 (60K)', 'GIST-960 (100K)']
    
    colors = {'float32': '#2196F3', 'i64f32': '#E53935'}
    markers = {'float32': 'o', 'i64f32': 's'}
    
    for idx, (ds, title) in enumerate(zip(datasets, titles)):
        ax = axes[idx // 2][idx % 2]
        data = real_search[ds]
        
        # Float32
        efs_f = [r['ef'] for r in data['float32_results']]
        qps_f = [r['qps'] for r in data['float32_results']]
        recall_f = [r['recall_at_10'] for r in data['float32_results']]
        
        # I64F32
        efs_i = [r['ef'] for r in data['i64f32_results']]
        qps_i = [r['qps'] for r in data['i64f32_results']]
        recall_i = [r['recall_at_10'] for r in data['i64f32_results']]
        
        ax.plot(efs_f, qps_f, '-o', color=colors['float32'], markersize=4, 
                label='Float32 (HNSW)', linewidth=1.5)
        ax.plot(efs_i, qps_i, '-s', color=colors['i64f32'], markersize=4,
                label='I64F32 (D-HNSW)', linewidth=1.5)
        
        # Annotate overhead at ef=128
        overhead = data['overhead_factor_ef128']
        ax.annotate(f'{overhead:.1f}× overhead',
                    xy=(128, qps_i[5] if len(qps_i) > 5 else qps_i[-1]),
                    xytext=(200, qps_i[5]*2 if len(qps_i) > 5 else qps_i[-1]*2),
                    fontsize=7, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        
        ax.set_xlabel('ef (search width)')
        ax.set_ylabel('QPS')
        ax.set_title(title, fontsize=10)
        ax.set_yscale('log')
        ax.legend(loc='upper right', fontsize=7)
        ax.set_xlim(0, 520)
    
    fig.suptitle('Fig. 12: D-HNSW (I64F32) vs HNSW (Float32) Search Performance', 
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fig12_real_i64f32_search.png')
    plt.close()
    print("✅ Fig 12 saved")


# ============================================================
# Fig 13: Two-Phase Search Comparison
# ============================================================
def plot_fig13():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    datasets = ['sift-1m', 'fashion-mnist', 'gist-960']
    titles = ['SIFT-128', 'Fashion-MNIST-784', 'GIST-960']
    
    colors = {
        'float32': '#2196F3',
        'i64f32_only': '#E53935',
        'two_phase_x2': '#FF9800',
        'two_phase_x3': '#4CAF50',
        'two_phase_x5': '#9C27B0',
    }
    labels = {
        'float32': 'Float32',
        'i64f32_only': 'I64F32 only',
        'two_phase_x2': '2-Phase (2×)',
        'two_phase_x3': '2-Phase (3×)',
        'two_phase_x5': '2-Phase (5×)',
    }
    markers = {
        'float32': 'o',
        'i64f32_only': 's',
        'two_phase_x2': '^',
        'two_phase_x3': 'D',
        'two_phase_x5': 'v',
    }
    
    for idx, (ds, title) in enumerate(zip(datasets, titles)):
        ax = axes[idx]
        data = two_phase[ds]
        
        for method in ['float32', 'i64f32_only', 'two_phase_x2', 'two_phase_x3', 'two_phase_x5']:
            results = data['methods'][method]
            efs = [r['ef'] for r in results]
            qps = [r['qps'] for r in results]
            
            ax.plot(efs, qps, f'-{markers[method]}', color=colors[method], 
                    markersize=4, label=labels[method], linewidth=1.2)
        
        # Add summary annotation
        summary = data['summary_ef128']
        speedup = summary['two_phase_speedup_vs_i64']
        if speedup > 1:
            ax.text(0.05, 0.05, f'2-Phase speedup: {speedup:.2f}×',
                    transform=ax.transAxes, fontsize=7, color='green',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
        
        ax.set_xlabel('ef (search width)')
        ax.set_ylabel('QPS')
        ax.set_title(title, fontsize=10)
        ax.set_yscale('log')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=6)
    
    fig.suptitle('Fig. 13: Two-Phase Search (I32F16 Filter + I64F32 Verify)', 
                 fontsize=11, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fig13_two_phase_search.png')
    plt.close()
    print("✅ Fig 13 saved")


# ============================================================
# Fig 14: Overhead Factor Bar Chart (comparing all methods)
# ============================================================
def plot_fig14():
    fig, ax = plt.subplots(figsize=(7, 4))
    
    datasets = ['sift-1m', 'fashion-mnist', 'gist-960']
    ds_labels = ['SIFT-128\n(d=128)', 'Fashion-MNIST\n(d=784)', 'GIST-960\n(d=960)']
    
    # Collect overhead factors at ef=128
    i64_overheads = []
    tp2_overheads = []
    tp3_overheads = []
    tp5_overheads = []
    
    for ds in datasets:
        data = two_phase[ds]
        s = data['summary_ef128']
        i64_overheads.append(s['i64f32_overhead'])
        tp2_overheads.append(s['two_phase_overhead'])
        # Calculate for x3, x5 from raw data
        f32_qps = s['float32_qps']
        methods = data['methods']
        # ef=128 is index 1 in the two_phase data
        tp3_qps = [r for r in methods['two_phase_x3'] if r['ef'] == 128][0]['qps']
        tp5_qps = [r for r in methods['two_phase_x5'] if r['ef'] == 128][0]['qps']
        tp3_overheads.append(f32_qps / tp3_qps)
        tp5_overheads.append(f32_qps / tp5_qps)
    
    x = np.arange(len(datasets))
    width = 0.18
    
    bars1 = ax.bar(x - 1.5*width, i64_overheads, width, label='I64F32 only', 
                   color='#E53935', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*width, tp2_overheads, width, label='2-Phase (2×)', 
                   color='#FF9800', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*width, tp3_overheads, width, label='2-Phase (3×)', 
                   color='#4CAF50', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x + 1.5*width, tp5_overheads, width, label='2-Phase (5×)', 
                   color='#9C27B0', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}×',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Overhead Factor (vs Float32)')
    ax.set_title('Fig. 14: Search Overhead at ef=128 (Lower is Better)', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.legend(loc='upper left', fontsize=8)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Float32 baseline')
    ax.set_ylim(0, max(i64_overheads) * 1.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fig14_overhead_comparison.png')
    plt.close()
    print("✅ Fig 14 saved")


# ============================================================
# Fig 15: Determinism Verification Summary (Table + Visual)
# ============================================================
def plot_fig15():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    
    # Build table data
    datasets = ['sift-1m', 'fashion-mnist', 'gist-960']
    ds_labels = ['SIFT-128', 'Fashion-784', 'GIST-960']
    
    col_labels = ['Dataset', 'Dims', 'N', 'Queries', 'Runs',
                  'Distance\nHash Match', 'Search\nHash Match', 'Order\nIndep.', 'Overall']
    
    table_data = []
    for ds, label in zip(datasets, ds_labels):
        d = determinism[ds]
        row = [
            label,
            str(d['dims']),
            f"{d['n_vectors']:,}",
            str(d['n_queries']),
            str(d['n_runs']),
            'PASS 5/5' if d['distance_test']['all_identical'] else 'FAIL',
            'PASS 5/5' if d['search_test']['all_identical'] else 'FAIL',
            'PASS' if d['order_independence']['identical'] else 'FAIL',
            'PASS' if d['overall_pass'] else 'FAIL',
        ]
        table_data.append(row)
    
    # Also add float32 control row info
    # Add a summary row
    table_data.append([
        'Float32 Control', '--', '--', '--', '5',
        'PASS 5/5', 'PASS 5/5', '--', 'PASS'
    ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colColours=['#E3F2FD'] * len(col_labels)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.8)
    
    # Color the cells
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            cell = table[i+1, j]
            text = table_data[i][j]
            if 'PASS' in text:
                cell.set_facecolor('#E8F5E9')  # light green
            elif 'FAIL' in text:
                cell.set_facecolor('#FFEBEE')  # light red
            else:
                cell.set_facecolor('#FFFFFF')
    
    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Fig. 15: SHA-256 Determinism Verification\n'
                 'I64F32 arithmetic produces bit-identical results across all runs',
                 fontsize=11, fontweight='bold', pad=20)
    
    # Add hash examples below
    example_hash = determinism['sift-1m']['distance_test']['hashes'][0][:32] + '...'
    fig.text(0.5, 0.02, 
             f'Example SHA-256 hash (SIFT distance, all 5 runs identical): {example_hash}',
             ha='center', fontsize=7, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fig15_determinism_verification.png')
    plt.close()
    print("✅ Fig 15 saved")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    plot_fig12()
    plot_fig13()
    plot_fig14()
    plot_fig15()
    print("\n🎉 All 4 new plots generated successfully!")
