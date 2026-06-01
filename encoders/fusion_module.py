"""
ArogyaDrishti — Cross-Modal Attention Fusion Module
Takes embeddings from all 5 image encoders + tabular encoder.
Combines them using PyTorch MultiheadAttention.
Outputs 7 risk scores (0-1) for: hematological, metabolic, renal,
hepatic, cardiovascular, dermatological, nutritional.

PARTIAL INFERENCE SUPPORT:
  Missing encoders are replaced with zero vectors and masked out
  of the attention mechanism. Results include a confidence score
  based on how many encoders contributed.
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
                    Missing encoders are handled gracefully with zero vectors.

        Returns: dict with keys:
            - risk scores: {risk_name: tensor(batch,)} with values in [0, 1]
            - 'active_encoders': list of encoder names that contributed
            - 'confidence': float 0-1 based on fraction of encoders present
        """
        if not embeddings:
            raise ValueError("At least one encoder embedding must be provided")

        # Determine batch size and device from any available embedding
        ref = next(iter(embeddings.values()))
        batch_size = ref.shape[0]
        device = ref.device

        # Build token sequence — zero vector for missing encoders
        tokens = []
        mask = []  # True = ignore this token in attention
        active_encoders = []

        for name in self.encoder_names:
            if name in embeddings:
                projected = self.projections[name](embeddings[name])
                tokens.append(projected)
                mask.append(False)   # attend to this token
                active_encoders.append(name)
            else:
                # Zero vector for missing encoder
                zero = torch.zeros(batch_size, self.fusion_dim, device=device)
                tokens.append(zero)
                mask.append(True)    # mask out in attention

        # Stack tokens: (batch, num_encoders, fusion_dim)
        tokens = torch.stack(tokens, dim=1)

        # Build key_padding_mask: (batch, num_encoders)
        # True = ignore this position in attention
        key_padding_mask = torch.tensor(
            mask, dtype=torch.bool, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        # If ALL encoders are masked (shouldn't happen), fall back to no mask
        if all(mask):
            key_padding_mask = None

        # Cross-modal attention with masking
        attn_out, _ = self.attention(
            tokens, tokens, tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True   # required for ONNX export
        )
        fused = self.norm(tokens + self.dropout(attn_out))  # residual

        # Mean pool only over ACTIVE encoder positions
        if key_padding_mask is not None:
            # Zero out masked positions before pooling
            active_mask = (~key_padding_mask).float().unsqueeze(-1)  # (batch, N, 1)
            fused = fused * active_mask
            num_active = active_mask.sum(dim=1)  # (batch, 1)
            num_active = num_active.clamp(min=1)
            pooled = fused.sum(dim=1) / num_active  # (batch, fusion_dim)
        else:
            pooled = fused.mean(dim=1)

        pooled = self.aggregator(pooled)

        # Confidence: fraction of encoders that contributed
        confidence = len(active_encoders) / self.num_encoders

        # Each head produces a scalar risk score via sigmoid
        risks = {
            name: torch.sigmoid(head(pooled)).squeeze(1)
            for name, head in self.heads.items()
        }

        # Add metadata
        risks['active_encoders'] = active_encoders
        risks['confidence'] = confidence

        return risks


class ArogyaDrishtiModel(nn.Module):
    """
    Full end-to-end model wrapping all encoders + fusion module.
    Used for training the fusion stage.

    Supports partial inference — pass only the encoders you have.
    """
    def __init__(self, encoders: nn.ModuleDict, fusion: FusionModule):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion

    def forward(self, inputs: dict):
        """
        inputs: dict of {encoder_name: pixel_values_tensor}
                plus optionally 'tabular': tabular_features_tensor

        Only encoders present in inputs are run — others are zero-masked.
        Returns: dict of risk scores + 'active_encoders' + 'confidence'
        """
        embeddings = {}
        for name, encoder in self.encoders.items():
            if name in inputs:
                embeddings[name] = encoder.get_embedding(inputs[name])
        return self.fusion(embeddings)
