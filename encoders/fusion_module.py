"""
ArogyaDrishti — Cross-Modal Attention Fusion Module
Takes embeddings from all 5 image encoders + tabular encoder.
Combines them using PyTorch MultiheadAttention.
Outputs 7 risk scores (0-1) for: hematological, metabolic, renal,
hepatic, cardiovascular, dermatological, nutritional.
"""
import torch
import torch.nn as nn


class FusionModule(nn.Module):
    def __init__(self, encoder_dims: dict, fusion_dim=512, num_heads=8, dropout=0.1):
        """
        encoder_dims: dict of {encoder_name: embedding_dim}
          e.g. {'face': 768, 'eye': 1024, 'tongue': 1024, 'skin': 512, 'nail': 1792, 'tabular': 100}
        fusion_dim: internal projection dimension (all encoders projected to this)
        """
        super().__init__()
        self.encoder_names = list(encoder_dims.keys())
        self.num_encoders = len(self.encoder_names)
        self.fusion_dim = fusion_dim

        # Project each encoder's embedding to the common fusion_dim
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, fusion_dim)
            for name, dim in encoder_dims.items()
        })

        # Cross-modal attention — each encoder can attend to all others
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

        # Aggregate: mean pool across encoder tokens, then MLP
        self.aggregator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 7 output heads — one per organ system risk score
        self.heads = nn.ModuleDict({
            'hematological':    nn.Linear(fusion_dim, 1),  # blood health / anemia
            'metabolic':        nn.Linear(fusion_dim, 1),  # diabetes, BMI, thyroid
            'renal':            nn.Linear(fusion_dim, 1),  # kidney function
            'hepatic':          nn.Linear(fusion_dim, 1),  # liver health
            'cardiovascular':   nn.Linear(fusion_dim, 1),  # heart risk
            'dermatological':   nn.Linear(fusion_dim, 1),  # skin conditions
            'nutritional':      nn.Linear(fusion_dim, 1),  # nutritional deficiency
        })

    def forward(self, embeddings: dict):
        """
        embeddings: dict of {encoder_name: tensor(batch, embed_dim)}
        Returns: dict of {risk_name: tensor(batch,)} with values in [0, 1]
        """
        # Project all embeddings to fusion_dim and stack as sequence
        # Shape: (batch, num_encoders, fusion_dim)
        tokens = torch.stack([
            self.projections[name](embeddings[name])
            for name in self.encoder_names
        ], dim=1)

        # Cross-modal attention (self-attention across encoder tokens)
        # need_weights=True forces Python path instead of native C++ (required for ONNX export)
        attn_out, _ = self.attention(tokens, tokens, tokens, need_weights=True)
        fused = self.norm(tokens + self.dropout(attn_out))  # residual

        # Mean pool across encoder dimension → (batch, fusion_dim)
        fused = fused.mean(dim=1)
        fused = self.aggregator(fused)

        # Each head produces a scalar risk score via sigmoid
        risks = {
            name: torch.sigmoid(head(fused)).squeeze(1)
            for name, head in self.heads.items()
        }
        return risks


class ArogyaDrishtiModel(nn.Module):
    """
    Full end-to-end model wrapping all encoders + fusion module.
    Used for training the fusion stage.
    """
    def __init__(self, encoders: nn.ModuleDict, fusion: FusionModule):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion

    def forward(self, inputs: dict):
        """
        inputs: dict of {encoder_name: pixel_values_tensor}
                plus optionally 'tabular': tabular_features_tensor
        Returns: dict of risk scores from fusion module
        """
        embeddings = {}
        for name, encoder in self.encoders.items():
            if name in inputs:
                embeddings[name] = encoder.get_embedding(inputs[name])
        return self.fusion(embeddings)
