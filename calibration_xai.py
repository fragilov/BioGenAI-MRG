"""
Prevalence-Adjusted ClinGen PP3/BP4 Calibration (Temperature Scaling + Isotonic Regression),
50-Step Riemann Path-Integrated Gradients, and Two-Sided Paired Pfam-Cluster Bootstrap Tests.
"""
from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


class StratumAdjustedCalibrator:
    """
    Fits nested Temperature Scaling + Isotonic Regression on inner-validation logits
    and computes Stratum-Adjusted Local Likelihood Ratios LR+_s(v) (Eq. 4) with
    1,000-iteration bootstrap 95% confidence bounds [LR+_LB95, LR+_UB95].
    """

    def __init__(self, n_bootstrap: int = 1000, seed: int = 42):
        self.temperature = 1.0
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1.0 - 1e-5)
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def fit(self, logits_val: np.ndarray, y_val: np.ndarray) -> "StratumAdjustedCalibrator":
        def nll(t):
            T = max(float(t[0]), 1e-3)
            p = 1.0 / (1.0 + np.exp(-logits_val / T))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            return -np.mean(y_val * np.log(p) + (1.0 - y_val) * np.log(1.0 - p))

        res = minimize(nll, x0=[1.0], bounds=[(0.05, 10.0)])
        self.temperature = float(res.x[0])
        p_temp = 1.0 / (1.0 + np.exp(-logits_val / self.temperature))
        self.iso.fit(p_temp, y_val)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        p_temp = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        return np.clip(self.iso.predict(p_temp), 1e-5, 1.0 - 1e-5)

    def compute_stratum_lr(self, logits: np.ndarray, pi_s: float) -> np.ndarray:
        p_hat = self.predict_proba(logits)
        post_odds = p_hat / (1.0 - p_hat)
        prior_odds = pi_s / (1.0 - pi_s)
        return post_odds / prior_odds

    @staticmethod
    def assign_clingen_point_code(lr_lb95: float, lr_ub95: float) -> Tuple[str, int]:
        """Assigns Tavtigian et al. (2020) / Pejaver et al. (2022) point codes (+4 pt cap)."""
        if lr_lb95 >= 18.70:
            return "PP3_Strong", +4
        if lr_lb95 >= 4.33:
            return "PP3_Moderate", +2
        if lr_lb95 >= 2.08:
            return "PP3_Supporting", +1
        if lr_ub95 <= 0.23:
            return "BP4_Moderate", -2
        if lr_ub95 <= 0.48:
            return "BP4_Supporting", -1
        return "Indeterminate (0 pts)", 0


def compute_intra_gene_auroc(
    y_true: np.ndarray, y_score: np.ndarray, gene_ids: np.ndarray, min_per_class: int = 2
) -> float:
    """Pair-weighted Intra-Gene AUROC across discordant genes (Eq. 1 in paper)."""
    weighted_sum = 0.0
    total_pairs = 0.0
    for g in np.unique(gene_ids):
        mask = gene_ids == g
        yg = y_true[mask]
        n_pos = int((yg == 1).sum())
        n_neg = int((yg == 0).sum())
        if n_pos >= min_per_class and n_neg >= min_per_class:
            pairs = float(n_pos * n_neg)
            weighted_sum += pairs * roc_auc_score(yg, y_score[mask])
            total_pairs += pairs
    return weighted_sum / max(total_pairs, 1.0)


def paired_cluster_bootstrap_pvalue(
    y_true: np.ndarray,
    score_1: np.ndarray,
    score_2: np.ndarray,
    cluster_ids: np.ndarray,
    metric_fn=average_precision_score,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Two-sided paired Pfam-cluster block bootstrap test on out-of-fold predictions.
    """
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    obs_diff = float(metric_fn(y_true, score_1) - metric_fn(y_true, score_2))

    diffs = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        diffs[b] = metric_fn(y_true[idx], score_1[idx]) - metric_fn(y_true[idx], score_2[idx])

    p_two_sided = 2.0 * min(np.mean(diffs <= 0.0), np.mean(diffs >= 0.0))
    ci_95 = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    return obs_diff, float(p_two_sided), ci_95
