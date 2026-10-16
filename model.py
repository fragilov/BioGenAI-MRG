"""
BioGenAI-MRG: Multiscale Relational Graph Attention, Positional Markov Chains,
and Degree-Aware Gated Residual Fusion.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalMarkovModule(nn.Module):
    """Out-of-fold position-specific 1st & 2nd order k-mer Markov transition encoder."""

    def __init__(self, window_bp: int = 15, embed_dim: int = 32, alpha: float = 1.0):
        super().__init__()
        self.window_bp = window_bp
        self.alpha = alpha
        self.proj = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, markov_stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            markov_stats: [N, 4] tensor of [LLR_order1, LLR_order2, donor_motif_delta, acceptor_motif_delta]
                          estimated strictly on inner-training folds.
        Returns:
            llr_score: [N, 1] log-likelihood ratio score
            h_markov: [N, embed_dim] sequential embedding
        """
        llr_score = markov_stats[:, :2].mean(dim=-1, keepdim=True)
        h_markov = self.proj(markov_stats)
        return llr_score, h_markov


class CrossRelationHeteroGATLayer(nn.Module):
    """
    Unified Cross-Relation Softmax Attention layer (Eq. 2 in paper).
    Normalizes attention across the union of all biological relation types N_all(v)
    to prevent singleton-relation degeneracy (alpha_vu = 1).
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 9, num_heads: int = 4, negative_slope: float = 0.2):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.num_relations = num_relations

        self.W_target = nn.Linear(in_dim, out_dim, bias=False)
        self.W_rel = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.rel_emb = nn.Embedding(num_relations, out_dim)
        self.att_vec = nn.Parameter(torch.empty(1, num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.att_vec)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        per_relation_softmax: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [N_nodes, in_dim] node embeddings
            edge_index: [2, E] directed edges (source u -> target v)
            edge_type: [E] relation type index in {0, ..., R-1}
            per_relation_softmax: If True, runs ablation normalizing within each (v, r) separately.
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = h.size(0)

        h_dst = self.W_target(h)[dst].view(-1, self.num_heads, self.head_dim)
        h_src = torch.zeros_like(h_dst)
        for r in range(self.num_relations):
            mask = edge_type == r
            if mask.any():
                h_src[mask] = self.W_rel[r](h[src[mask]]).view(-1, self.num_heads, self.head_dim)

        r_emb = self.rel_emb(edge_type).view(-1, self.num_heads, self.head_dim)
        cat_feat = torch.cat([h_dst, h_src, r_emb], dim=-1)
        logits = self.leaky_relu((cat_feat * self.att_vec).sum(dim=-1))  # [E, K]

        if not per_relation_softmax:
            # Cross-Relation Softmax over N_all(v)
            logits_max = torch.zeros((num_nodes, self.num_heads), device=h.device)
            logits_max.index_reduce_(0, dst, logits, reduce="amax", include_self=False)
            exp_logits = torch.exp(logits - logits_max[dst])
            denom = torch.zeros((num_nodes, self.num_heads), device=h.device)
            denom.index_add_(0, dst, exp_logits)
            alpha = exp_logits / (denom[dst] + 1e-12)
        else:
            # Ablation: separate per-relation softmax over N_r(v)
            group_id = dst * self.num_relations + edge_type
            num_groups = num_nodes * self.num_relations
            logits_max = torch.zeros((num_groups, self.num_heads), device=h.device)
            logits_max.index_reduce_(0, group_id, logits, reduce="amax", include_self=False)
            exp_logits = torch.exp(logits - logits_max[group_id])
            denom = torch.zeros((num_groups, self.num_heads), device=h.device)
            denom.index_add_(0, group_id, exp_logits)
            alpha = exp_logits / (denom[group_id] + 1e-12)

        msg = (alpha.unsqueeze(-1) * h_src).reshape(-1, self.num_heads * self.head_dim)
        out = torch.zeros((num_nodes, self.num_heads * self.head_dim), device=h.device)
        out.index_add_(0, dst, msg)
        return self.norm(F.elu(out) + h), alpha.mean(dim=-1)


class BioGenAIMRG(nn.Module):
    """
    Full BioGenAI-MRG Architecture with Same-Backbone Ablation switches:
      - disable_graph (g_v = 0): Same-Backbone Neural MLP
      - per_relation_softmax: Per-Relation Softmax Ablation
      - strict_no_hpo_degree: Regime I-Strict (zeroes degree gate input s_{g_v})
    """

    def __init__(
        self,
        in_tabular_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        num_relations: int = 9,
        dropedge_rate: float = 0.30,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropedge_rate = dropedge_rate

        self.node_init = nn.Linear(in_tabular_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            CrossRelationHeteroGATLayer(hidden_dim, hidden_dim, num_relations, num_heads)
            for _ in range(num_layers)
        ])
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.tabular_mlp = nn.Sequential(
            nn.Linear(in_tabular_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.markov_module = PositionalMarkovModule(embed_dim=32)
        self.degree_gate = nn.Linear(hidden_dim * 2 + 32 + 3, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_tab: torch.Tensor,
        markov_stats: torch.Tensor,
        s_gv: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        variant_mask: Optional[torch.Tensor] = None,
        disable_graph: bool = False,
        per_relation_softmax: bool = False,
        strict_no_hpo_degree: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h0 = self.node_init(x_tab)
        h_list = [h0]

        if self.training and self.dropedge_rate > 0.0 and not disable_graph:
            keep_mask = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropedge_rate
            edge_index_used = edge_index[:, keep_mask]
            edge_type_used = edge_type[keep_mask]
        else:
            edge_index_used = edge_index
            edge_type_used = edge_type

        last_alpha = None
        h_curr = h0
        for layer in self.gat_layers:
            h_curr, last_alpha = layer(
                h_curr, edge_index_used, edge_type_used, per_relation_softmax=per_relation_softmax
            )
            h_list.append(h_curr)

        h_jk = self.jk_proj(torch.cat(h_list, dim=-1))
        h_tab = self.tabular_mlp(x_tab)
        _, h_markov = self.markov_module(markov_stats)

        s_deg = torch.zeros_like(s_gv) if strict_no_hpo_degree else s_gv
        if disable_graph:
            g_v = torch.zeros_like(h_tab)
            z_v = h_tab
        else:
            gate_in = torch.cat([h_jk, h_tab, h_markov, s_deg], dim=-1)
            g_v = torch.sigmoid(self.degree_gate(gate_in))
            z_v = g_v * h_jk + (1.0 - g_v) * h_tab

        logits = self.classifier(z_v).squeeze(-1)
        if variant_mask is not None:
            logits = logits[variant_mask]
        return logits, {"g_v": g_v, "alpha": last_alpha, "z_v": z_v}
