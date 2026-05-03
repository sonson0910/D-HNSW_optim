"""
Generate corrected Figure 8: D-HNSW Overhead Breakdown by Dataset
Uses actual projected end-to-end overhead from Table III (tab:main_results)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from Table III (tab:main_results) - projected end-to-end overhead
datasets = ['SIFT-1M\n(128d)', 'GloVe-1M\n(100d)', 'Fashion\n(784d)', 'GIST-1M\n(960d)', 'Deep-1M\n(96d)', 'Random-1M\n(128d)']
overheads = [1.75, 1.61, 1.42, 1.39, 1.45, 1.75]

# Color gradient: higher overhead = more saturated
colors = []
for oh in overheads:
    # Map 1.39-1.75 to color intensity
    t = (oh - 1.35) / (1.80 - 1.35)  # normalize to 0-1
    r = 0.18 + 0.55 * t
    g = 0.55 + 0.20 * (1 - t)
    b = 0.34 + 0.20 * (1 - t)
    colors.append((r, g, b, 0.85))

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(datasets))
bars = ax.bar(x, overheads, width=0.6, color=colors, edgecolor='#333333', linewidth=0.8)

# Add value labels on top of bars
for bar, val in zip(bars, overheads):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Mean line
mean_oh = np.mean(overheads)
ax.axhline(y=mean_oh, color='#cc3333', linestyle='--', linewidth=1.5, 
           label=f'Mean {mean_oh:.2f}×', zorder=0)

# Reference line at 1.0 (no overhead)
ax.axhline(y=1.0, color='#999999', linestyle=':', linewidth=1.0, alpha=0.5,
           label='No overhead (1.0×)', zorder=0)

ax.set_ylabel('Projected End-to-End Overhead (×)', fontsize=12)
ax.set_title('D-HNSW Projected Overhead by Dataset', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10, rotation=0)
ax.set_ylim(0, 2.2)
ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='-')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotation explaining the trend
ax.annotate('Lower dim → higher $\\alpha$ → higher overhead',
            xy=(0, 1.75), xytext=(2.5, 2.05),
            fontsize=9, fontstyle='italic', color='#555555',
            arrowprops=dict(arrowstyle='->', color='#999999', lw=1.0))

plt.tight_layout()

# Save both PNG and PDF
outdir = r'd:\research_journal\project-e7e8b3bb-workspace\workspace\shared\figures'
fig.savefig(f'{outdir}/fig8_overhead_breakdown.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{outdir}/fig8_overhead_breakdown.pdf', bbox_inches='tight')
print("Figure 8 regenerated successfully!")
print(f"  Overheads: {dict(zip(['SIFT','GloVe','Fashion','GIST','Deep','Random'], overheads))}")
print(f"  Mean: {mean_oh:.2f}×, Range: {min(overheads):.2f}×–{max(overheads):.2f}×")
