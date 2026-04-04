"""
ArogyaDrishti — Tabular Encoder
Encodes structured health features (age, BMI, hemoglobin, BP, symptoms, etc.)
into a 100-dim embedding for fusion with image encoders.

Primary model: MLP with batch normalization (TabPFN fallback if installed)
Input features (21 total):
  Demographics : age, gender, bmi
  Vitals       : systolic_bp, diastolic_bp, heart_rate, spo2
  Blood        : hemoglobin, glucose_fasting, glucose_pp, hba1c
  Liver        : alt, ast, bilirubin_total
  Kidney       : creatinine, urea
  Nutrition    : ferritin, vitamin_b12, vitamin_d, albumin
  Symptoms     : fatigue_score  (0-3 ordinal: none/mild/mod/severe)

Labels (7 risk heads — binary 0/1):
  hematological, metabolic, renal, hepatic,
  cardiovascular, dermatological, nutritional
"""

import torch
import torch.nn as nn
import numpy as np

# ── Feature columns (must match prepare_tabular_csv.py) ───────────────────────
TABULAR_FEATURES = [
    'age', 'gender', 'bmi',
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'spo2',
    'hemoglobin', 'glucose_fasting', 'glucose_pp', 'hba1c',
    'alt', 'ast', 'bilirubin_total',
    'creatinine', 'urea',
    'ferritin', 'vitamin_b12', 'vitamin_d', 'albumin',
    'fatigue_score',
]
NUM_FEATURES = len(TABULAR_FEATURES)   # 21
FEATURES = TABULAR_FEATURES           # alias for train_tabular.py import

TARGETS = [
    'risk_hematological',
    'risk_metabolic',
    'risk_renal',
    'risk_hepatic',
    'risk_cardiovascular',
    'risk_dermatological',
    'risk_nutritional',
]

# ── 7 risk target columns ──────────────────────────────────────────────────────
RISK_TARGETS = [
    'risk_hematological',
    'risk_metabolic',
    'risk_renal',
    'risk_hepatic',
    'risk_cardiovascular',
    'risk_dermatological',
    'risk_nutritional',
]


class TabularEncoder(nn.Module):
    """
    Deep MLP tabular encoder.
    Architecture: Input(21) → BN → 256 → BN → GELU → Dropout
                           → 256 → BN → GELU → Dropout
                           → 128 → BN → GELU → Dropout
                           → 100 (embedding)
    """

    def __init__(self,
                 num_features: int = NUM_FEATURES,
                 embed_dim: int = 100,
                 hidden_dims: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        self.embed_dim = embed_dim

        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, embed_dim))
        self.backbone = nn.Sequential(*layers)

        # Separate 7-head classifier for standalone training
        self.heads = nn.ModuleDict({
            name.replace('risk_', ''): nn.Linear(embed_dim, 1)
            for name in RISK_TARGETS
        })
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        x: (batch, num_features) float tensor — already normalised
        Returns embedding (batch, 100) if return_embedding=True,
        else dict of risk logits {name: (batch,)}
        """
        emb = self.backbone(x)
        if return_embedding:
            return emb
        return {
            name: torch.sigmoid(head(emb)).squeeze(1)
            for name, head in self.heads.items()
        }

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Called by FusionModule — returns (batch, embed_dim)."""
        return self.forward(x, return_embedding=True)
