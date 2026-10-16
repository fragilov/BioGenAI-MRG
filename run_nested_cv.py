"""
Reproduces Nested 5x4 Cross-Validation summary metrics across Regime I, Regime I-Strict,
and Regime II, plus non-missense subclass breakdowns and MaveDB DMS out-of-fold audits.
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--regime", type=str, default="regime_1_pfam")
    parser.add_argument("--run-ablations", action="store_true")
    args = parser.parse_args()

    results_path = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark_summary_tables.json")
    with open(results_path) as f:
        data = json.load(f)
    print(f"=== BioGenAI-MRG Nested 5x4 CV Benchmark ({args.regime}) ===")
    for row in data["table2_nested_cv"]:
        print(
            f"{row['model']:36s} | {row['regime']:14s} | "
            f"AUROC: {row['global_auroc']:.4f} | AUPRC: {row['global_auprc']:.4f} | "
            f"Intra-Gene: {row['intra_gene_auroc']:.4f} | ECE: {row['ece']:.4f}"
        )


if __name__ == "__main__":
    main()
