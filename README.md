# BioGenAI-MRG: Multiscale Relational Graph Learning, Sequential Markov Modeling, and Calibrated Explainable AI for Clinical Genetic Variant Prioritization and Evidence Calibration

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-EE4C2C.svg)](https://pytorch.org/)
[![PyG 2.5](https://img.shields.io/badge/PyG-2.5-3C2179.svg)](https://pyg.org/)

Official implementation and reproducibility repository for the **ICCIES 2027** paper:

> **Multiscale Relational Graph Learning, Sequential Markov Modeling, and Calibrated Explainable AI for Clinical Genetic Variant Prioritization and Evidence Calibration**  
> **Pham Nguyen Tuan Kiet**  
> *Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam*  
> Contact: `52400207@student.tdtu.edu.vn`, `kietpham30406@gmail.com`

---

## 1. Overview & Key Contributions

Variants of Uncertain Significance (VUS) represent over 40% of clinically reported variants in the NCBI ClinVar archive. Because computational (`in silico`) evidence is capped at **+4 points (`PP3_Strong`)** under the quantitative ACMG/ClinGen Bayesian framework (Tavtigian et al., 2020; Pejaver et al., 2022; Bergquist et al., 2025) whereas Likely Pathogenic requires $\ge 6$ points, **BioGenAI-MRG** is designed as a **clinical variant prioritization and calibrated `PP3`/`BP4` evidence-scoring framework**.

1. **Cross-Relation 7-Node Heterogeneous Graph & Degree-Aware Gated Fusion (`JK-HeteroGAT`)**: Decomposes coarse gene hubs into sub-gene `Domain` (Pfam v36.0 / InterPro) and `ExonRegion` structural anchors across **89,254 nodes** and **604,195 edges** (7 node types, 9 relation types). Replaces degenerate per-relation attention with unified **Cross-Relation Softmax Attention** over $\mathcal{N}_{\text{all}}(v)$, Jumping Knowledge concatenation ($L=3, K=4, d=256$), and a **Degree-Aware Sigmoid Gate** conditioned on $\mathbf{s}_{g_v}$ with 30% `DropEdge`.
2. **Same-Backbone Topology Ablations & Multi-Tier Anti-Circularity Audit**: Evaluates **52,949 high-confidence ($\ge 2$-star) ClinVar variants** across **1,942 Pfam/Paralog clusters** (MMseqs2 $\ge 30\%$ sequence identity or shared Pfam-A clan) using **Nested $5 \times 4$ Cross-Validation** under three feature regimes:
   - **Regime I (Circularity-Reduced Intrinsic)**: Excludes gnomAD allele frequencies and all supervised/ensemble meta-predictors (`AlphaMissense`, `REVEL`, `BayesDel`, `SIFT`).
   - **Regime I-Strict (w/o HPO Edges & Degree Gate)**: Strips all 108,257 `Gene-Phenotype` (HPO) edges, HPO SVD priors, and degree features $\mathbf{s}_{g_v}$ to eliminate indirect Type 2 disease-gene curation proxies.
   - **Regime II (Full Integrative)**: Integrates Regime I with log-transformed gnomAD v4.1 AF, `AlphaMissense` (coding missense), and `SpliceAI` $\Delta$-scores.
3. **Timestamp-Audited Retrospective Temporal Evaluation ($T_0 \to T_1$)**: Evaluates variants from $T_0$ (`2025-09-15`) to $T_1$ (`2025-11-23`) across **Cohort A** (strictly $T_0$-observable rule: $\ge 3$ submitters at $T_0$, $n=812$ resolved of $2,140$) and **Cohort B** (all $n=1,288$ spontaneously resolved VUS out of the unselected $N=18,450$ archive; $93.02\%$ frozen).
4. **Prevalence-Adjusted ClinGen `PP3`/`BP4` Calibration & Path-Integrated XAI**: Calibrates stratum-adjusted local likelihood ratios $\text{LR}^+_s(v)$ via nested Temperature Scaling + Isotonic Regression with 95% bootstrap confidence bounds (`LR+_LB95`, `LR+_UB95`) and prior sensitivity ($\pi_0 \in \{0.0441, 0.10, \pi_s\}$), paired with $M=50$-step Riemann **Path-Integrated Gradients** and out-of-fold **MaveDB** Deep Mutational Scanning validation ($n=1,840$ across *TP53*, *BRCA1*, *PTEN*).

---

## 2. Repository Structure

```text
BioGenAI-MRG/
├── README.md                          # Documentation, benchmark tables, and reproduction guide
├── LICENSE                            # MIT License
├── requirements.txt                   # Python package dependencies
├── configs/
│   └── default_config.yaml            # Hyperparameters, graph schema, and CV split settings
├── src/
│   ├── __init__.py
│   ├── model.py                       # BioGenAI-MRG (Cross-Relation JK-HeteroGAT + Markov + Degree Gate)
│   └── calibration_xai.py             # Stratum-Adjusted LR+(v), 50-step Path-IG, and Paired Cluster Bootstrap
├── scripts/
│   ├── curation_and_splits.py         # ClinVar T0/T1 filtering, 7-node heterograph, & 1,942 Pfam cluster builder
│   ├── run_nested_cv.py               # Nested 5x4 CV training, same-backbone ablations, and DMS audit
│   └── generate_fig1.py               # Reproduces Figure 1 (fig1_benchmark.pdf / .png)
├── results/
│   └── benchmark_summary_tables.json  # Complete nested 5x4 CV metrics (Tables 1-5)
└── paper/
    ├── fig1_benchmark.png             # High-resolution benchmark figure
    └── fig1_benchmark.pdf             # Vector PDF figure for LaTeX/Overleaf
```

---

## 3. Summary of Main Benchmark Results (Nested $5 \times 4$ CV, $N = 52,949$)

| Model / Same-Backbone Ablation | Regime (Split) | Global AUROC [95% CI] | Global AUPRC [95% CI] | Missense AUPRC | Non-Missense AUPRC | Intra-Gene AUROC | ECE $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Gene-Identity Prior (Harrizi et al., 2026) | Prior (Pfam) | 0.7918 [0.781, 0.802] | 0.3412 [0.329, 0.354] | 0.3845 | 0.2410 | 0.5000 | 0.0842 |
| AlphaMissense (Cheng et al., 2023) | Ext. (Miss.) | 0.9245 [0.919, 0.930] | 0.7618 [0.750, 0.773] | 0.7618 | N/A | 0.8690 | 0.0645 |
| AlphaGenome (Avsec et al., 2026, Post-$T_0$) | Ext. (Non-M.) | 0.9182 [0.912, 0.924] | 0.7485 [0.736, 0.761] | N/A | 0.7485 | 0.8562 | 0.0568 |
| CADD v1.7 + SpliceAI | Reg. I (Pfam) | 0.8964 [0.890, 0.903] | 0.6782 [0.665, 0.691] | 0.6640 | 0.6945 | 0.8195 | 0.0580 |
| Tabular GBDT (XGBoost, w/o Graph) | Reg. I (Pfam) | 0.9142 [0.908, 0.920] | 0.7385 [0.727, 0.749] | 0.7512 | 0.7218 | 0.8614 | 0.0412 |
| Same-Backbone: No Msg. Passing ($\mathbf{g}_v \equiv \mathbf{0}$) | Reg. I (Pfam) | 0.9185 [0.913, 0.924] | 0.7462 [0.735, 0.757] | 0.7590 | 0.7312 | 0.8692 | 0.0049 |
| Same-Backbone: Permuted Graph | Reg. I (Pfam) | 0.9190 [0.913, 0.925] | 0.7480 [0.737, 0.759] | 0.7605 | 0.7330 | 0.8705 | 0.0048 |
| Same-Backbone: Per-Relation Softmax | Reg. I (Pfam) | 0.9312 [0.926, 0.936] | 0.7790 [0.769, 0.789] | 0.7945 | 0.7610 | 0.8895 | 0.0044 |
| **BioGenAI-MRG (w/o HPO & Degree)** | **Reg. I-Strict** | **0.9340 [0.929, 0.939]** | **0.7862 [0.776, 0.796]** | **0.8025** | **0.7654** | **0.9012** | **0.0043** |
| **BioGenAI-MRG (Full Cross-Rel.)** | **Reg. I (Pfam)** | **0.9386 [0.934, 0.943]** | **0.7946 [0.785, 0.804]** | **0.8104** | **0.7742** | **0.9048** | **0.0042** |
| **BioGenAI-MRG (Full Cross-Rel.)** | **Reg. I (Gene)** | **0.9462 [0.942, 0.950]** | **0.8128 [0.804, 0.821]** | **0.8295** | **0.7912** | **0.9165** | **0.0038** |
| **BioGenAI-MRG (Calibrated)** | **Reg. II (Gene)** | **0.9806 [0.978, 0.983]** | **0.8755 [0.868, 0.882]** | **0.8920** | **0.8529** | **0.9655** | **0.0023** |

---

## 4. Quick Start & Reproducibility

### Installation

```bash
git clone https://github.com/kietpham30406/BioGenAI-MRG.git
cd BioGenAI-MRG
pip install -r requirements.txt
```

### Reproducing Paper Figures & Nested $5 \times 4$ CV Audits

```bash
# 1. Regenerate Figure 1 (Same-Backbone Topology Ablations & Subclass AUPRC)
python scripts/generate_fig1.py

# 2. Build 7-node heterograph and 1,942 Pfam/Paralog clusters from ClinVar T0 VCF
python scripts/curation_and_splits.py --config configs/default_config.yaml

# 3. Run Nested 5x4 CV across Regime I, Regime I-Strict, and Regime II with same-backbone ablations
python scripts/run_nested_cv.py --config configs/default_config.yaml --regime regime_1_pfam --run-ablations
```

---

## 5. Citation

```bibtex
@inproceedings{kiet2027biogenai,
  author    = {Pham Nguyen Tuan Kiet},
  title     = {Multiscale Relational Graph Learning, Sequential Markov Modeling, and Calibrated Explainable AI for Clinical Genetic Variant Prioritization and Evidence Calibration},
  booktitle = {Proceedings of the International Conference on Computational Intelligence in Engineering Science (ICCIES 2027)},
  series    = {Communications in Computer and Information Science (CCIS)},
  publisher = {Springer},
  year      = {2027}
}
```
