import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality styling matching Springer LNCS/CCIS aesthetics
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.linewidth'] = 0.8

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), gridspec_kw={'width_ratios': [1.08, 1.12]})
fig.subplots_adjust(wspace=0.22, bottom=0.23, top=0.88, left=0.055, right=0.985)

# ==============================================================================
# PANEL (A): Same-Backbone Topology Ablations (Regime I: Circularity-Reduced, Pfam-Disjoint)
# Exact 4-decimal values from Table 2 (Nested 5x4 CV, N = 52,949)
# ==============================================================================
models_a = [
    "Gene-Identity\nPrior [7]",
    "Tabular GBDT\n(w/o Graph)",
    "Same-Backbone:\nNo Msg. Passing",
    "Same-Backbone:\nPermuted Graph",
    "Same-Backbone:\nw/o Dom./Exon",
    "Same-Backbone:\nPer-Rel. Softmax",
    "BioGenAI-MRG\n(Reg. I-Strict)",
    "BioGenAI-MRG\n(Full Cross-Rel.)"
]

auroc_a = [0.7918, 0.9142, 0.9185, 0.9190, 0.9248, 0.9312, 0.9340, 0.9386]
auprc_a = [0.3412, 0.7385, 0.7462, 0.7480, 0.7615, 0.7790, 0.7862, 0.7946]
intra_a = [0.5000, 0.8614, 0.8692, 0.8705, 0.8780, 0.8895, 0.9012, 0.9048]

x_a = np.arange(len(models_a))
w_a = 0.26

b1 = ax1.bar(x_a - w_a, auroc_a, width=w_a, label='Global AUROC', color='#1e40af', edgecolor='#0f172a', linewidth=0.5, zorder=3)
b2 = ax1.bar(x_a,       auprc_a, width=w_a, label='Global AUPRC', color='#0d9488', edgecolor='#0f172a', linewidth=0.5, zorder=3)
b3 = ax1.bar(x_a + w_a, intra_a, width=w_a, label='Intra-Gene AUROC', color='#d97706', edgecolor='#0f172a', linewidth=0.5, zorder=3)

ax1.set_title("(A) Same-Backbone Topology Ablations (Regime I, Pfam-Disjoint)", fontsize=10.5, fontweight='bold', pad=10, color='#0f172a')
ax1.set_ylabel("Nested 5×4 CV Metric Score", fontsize=9.5, fontweight='bold', color='#0f172a')
ax1.set_xticks(x_a)
ax1.set_xticklabels(models_a, rotation=26, ha='right', fontsize=8.2, fontweight='medium')
ax1.set_ylim(0.30, 1.15)
ax1.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax1.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bars, fmt_color in [(b1, '#1e3a8a'), (b2, '#0f766e'), (b3, '#b45309')]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, h + 0.008, f"{h:.4f}",
                 ha='center', va='bottom', fontsize=6.5, fontweight='bold', color=fmt_color, rotation=90)

# Highlight annotation for pure topological gain (+0.0484 AUPRC, p = 1.8e-4)
ax1.annotate("Pure Topology: +0.0484 AUPRC\n(p = 1.8×10⁻⁴ vs. No-Msg MLP;\n+0.0156 vs. Per-Rel, p = 0.0024)",
             xy=(7, 0.7946), xytext=(3.3, 0.40),
             arrowprops=dict(arrowstyle="->", color='#0f766e', lw=1.2, connectionstyle="arc3,rad=-0.18"),
             fontsize=7.8, fontweight='bold', color='#0f172a',
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0fdfa", edgecolor="#0d9488", lw=0.9))

ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=8.2, ncol=3)

# ==============================================================================
# PANEL (B): Domain-Fair Missense & Non-Missense Subclass AUPRC Breakdown
# Exact 4-decimal values from Table 2 & Table 3 (Nested 5x4 CV, Pfam-Disjoint)
# ==============================================================================
strata_b = [
    "Coding Missense\n(π=14.82%)",
    "All Non-Miss.\n(π=6.23%)",
    "Nonsense/FS\n(π=25.47%)",
    "Canon. Splice\n(π=34.10%)",
    "Prox. Intronic\n(π=1.85%)",
    "5'/3' UTR\n(π=1.92%)",
    "Deep Intronic\n(π=1.82%)",
    "Synonymous\n(π=0.42%)"
]

cadd_sp_b   = [0.6640, 0.6945, 0.8120, 0.8650, 0.7420, 0.5890, 0.5410, 0.6150]
found_b     = [0.7618, 0.7485, 0.8490, 0.8980, 0.8150, 0.7120, 0.7645, 0.6890]
mrg_reg1_b  = [0.8104, 0.7742, 0.8845, 0.9120, 0.8410, 0.6840, 0.6480, 0.7125]
mrg_reg2_b  = [0.8920, 0.8529, 0.9312, 0.9540, 0.8940, 0.7520, 0.7645, 0.8180]

x_b = np.arange(len(strata_b))
w_b = 0.20

s1 = ax2.bar(x_b - 1.5*w_b, cadd_sp_b,  width=w_b, label='CADD v1.7 + SpliceAI (Reg. I)', color='#64748b', edgecolor='#0f172a', linewidth=0.5, zorder=3)
s2 = ax2.bar(x_b - 0.5*w_b, found_b,    width=w_b, label='AlphaMissense (Miss.) / AlphaGenome (Non-M.)', color='#3b82f6', edgecolor='#0f172a', linewidth=0.5, zorder=3)
s3 = ax2.bar(x_b + 0.5*w_b, mrg_reg1_b, width=w_b, label='BioGenAI-MRG (Regime I: Circularity-Reduced)', color='#10b981', edgecolor='#0f172a', linewidth=0.5, zorder=3)
s4 = ax2.bar(x_b + 1.5*w_b, mrg_reg2_b, width=w_b, label='BioGenAI-MRG (Regime II: Full Integrative)', color='#dc2626', edgecolor='#0f172a', linewidth=0.5, zorder=3)

ax2.set_title("(B) Domain-Fair Missense & Non-Missense Subclass AUPRC", fontsize=10.5, fontweight='bold', pad=10, color='#0f172a')
ax2.set_ylabel("Stratum Test AUPRC", fontsize=9.5, fontweight='bold', color='#0f172a')
ax2.set_xticks(x_b)
ax2.set_xticklabels(strata_b, rotation=26, ha='right', fontsize=8.2, fontweight='medium')
ax2.set_ylim(0.30, 1.18)
ax2.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bars, fmt_color in [(s1, '#334155'), (s2, '#1d4ed8'), (s3, '#047857'), (s4, '#b91c1c')]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.008, f"{h:.4f}",
                 ha='center', va='bottom', fontsize=6.0, fontweight='bold', color=fmt_color, rotation=90)

ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=7.6, ncol=2)

out_dir = "/working_dir/c_17ca2956fb0368ee/artifacts/file_generation/ttl=63d/intermediate"
plt.savefig(f"{out_dir}/fig1_benchmark.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{out_dir}/fig1_benchmark.png", dpi=300, bbox_inches='tight')
print("Successfully regenerated fig1_benchmark.pdf and fig1_benchmark.png")
