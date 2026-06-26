"""
ArogyaDrishti — Cross-Modal Fusion (corrected v3.0)
===================================================

This is the piece that was MISSING from the uploaded code and that makes the
system actually "analyse a person".  Each encoder emits an embedding of a
different size (1024 / 512 / 2048 / 512 / 1920 / 1664 / 100).  Fusion:

  1. projects every modality to a common 512-dim space,
  2. treats each modality as one token and runs multi-head self-attention
     across the tokens (8 heads), with a key-padding mask so that MISSING
     modalities are ignored (graceful degradation),
  3. residual + LayerNorm -> FFN(512->1024->512) -> residual + LayerNorm,
  4. mean-pools the present tokens and produces 7 risk scores in [0, 1],
  5. produces a per-modality confidence score.

The set of modalities and their dimensions is passed in as a dict, so the
fusion module automatically adapts to whichever encoders/backbones loaded.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Canonical order of the 7 risk outputs (matches the architecture document)
RISK_CATEGORIES = [
    "hematological",
    "metabolic",
    "renal",
    "hepatic",
    "cardiovascular",
    "dermatological",
    "nutritional",
]


class CrossModalFusion(nn.Module):
    def __init__(self, modality_dims: dict[str, int], d_model: int = 512,
                 n_heads: int = 8, n_risks: int = 7, dropout: float = 0.2):
        """
        Args:
            modality_dims: {modality_name: encoder.out_dim} for every encoder
                           that may participate, e.g.
                           {'eye':1024,'face':512,...,'tabular':100}
        """
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.d_model = d_model
        self.n_risks = n_risks

        # 1) per-modality projection to the common space
        self.proj = nn.ModuleDict({
            m: nn.Linear(dim, d_model) for m, dim in modality_dims.items()
        })

        # 2) cross-modal self-attention
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # 3) feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 1024), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(1024, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # 4) seven independent risk heads
        self.risk_heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(n_risks)])

        # 5) per-modality confidence head (shared)
        self.conf_head = nn.Linear(d_model, 1)

    def forward(self, embeddings: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor] | None = None):
        """
        Args:
            embeddings: {modality_name: tensor(B, out_dim)}.  You may pass every
                        modality here and use `masks` to mark which samples
                        actually have it, OR pass only the present modalities.
            masks:      optional {modality_name: bool tensor(B,)} where True
                        means "this sample has this modality".  If None, every
                        modality in `embeddings` is treated as present for all
                        samples.  This enables PER-SAMPLE graceful degradation
                        during batched training.
        Returns:
            dict with:
              'risk_logits'     tensor (B, 7)  raw logits (use for training)
              'risks'           tensor (B, 7)  sigmoid probabilities in [0,1]
              'modality_conf'   {modality: tensor(B,)} confidence per modality
              'overall_conf'    tensor (B,)    mean confidence over present mods
              'used_modalities' list[str]
        """
        present = [m for m in self.modalities if m in embeddings]
        if not present:
            raise ValueError("At least one modality is required for analysis.")

        B = embeddings[present[0]].shape[0]
        device = embeddings[present[0]].device

        # presence matrix (B, M): True where the sample has the modality
        if masks is None:
            present_mat = torch.ones(B, len(present), dtype=torch.bool,
                                     device=device)
        else:
            present_mat = torch.stack(
                [masks[m].to(device).bool() for m in present], dim=1)

        # project + stack as tokens -> (B, M, d_model)
        tokens = torch.stack(
            [self.proj[m](embeddings[m]) for m in present], dim=1)

        # attention ignores absent (modality, sample) pairs
        key_padding = ~present_mat                       # True == ignore
        attn_out, _ = self.attn(tokens, tokens, tokens,
                                key_padding_mask=key_padding)
        x = self.norm1(tokens + attn_out)
        x = self.norm2(x + self.ffn(x))                  # (B, M, d_model)

        # per-modality confidence (zeroed where absent)
        conf = torch.sigmoid(self.conf_head(x)).squeeze(-1)   # (B, M)
        conf = conf * present_mat.float()
        modality_conf = {m: conf[:, i] for i, m in enumerate(present)}
        denom = present_mat.float().sum(1).clamp(min=1.0)     # (B,)
        overall_conf = conf.sum(1) / denom                    # (B,)

        # masked mean-pool over present tokens, then 7 risk logits
        mask_f = present_mat.float().unsqueeze(-1)            # (B, M, 1)
        pooled = (x * mask_f).sum(1) / denom.unsqueeze(-1)   # (B, d_model)
        risk_logits = torch.cat([h(pooled) for h in self.risk_heads], dim=1)

        return {
            "risk_logits": risk_logits,
            "risks": torch.sigmoid(risk_logits),
            "modality_conf": modality_conf,
            "overall_conf": overall_conf,
            "used_modalities": present,
        }
