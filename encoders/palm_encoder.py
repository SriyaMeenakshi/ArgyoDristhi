"""
ArogyaDrishti — Palm Encoder
Pretrained model: EfficientNet-B3 via timm (tf_efficientnet_b3)

Detects palm-visible health conditions:
  0 = normal
  1 = palmar_erythema   (liver disease / cirrhosis — red flush on palms)
  2 = pallor            (anaemia — pale/whitish palms)
  3 = jaundice_tinge    (hepatic — yellowish discolouration)
  4 = dupuytren         (fibrosis — thickened cords on palm)

Embedding dim: 1536 (EfficientNet-B3 feature output)

Dataset: 11k Hands (Kaggle) + any folder-organised palm images
  kaggle datasets download -d shyambhu/hands-and-palm-images-dataset
"""

import torch
import torch.nn as nn
import timm


class PalmEncoder(nn.Module):
    def __init__(self, num_classes: int = 5,
                 pretrained_model: str = "tf_efficientnet_b3",
                 dropout: float = 0.3):
        super().__init__()
        # EfficientNet-B3 — lighter than B4 used for nails, still strong feature extractor
        self.backbone = timm.create_model(pretrained_model, pretrained=True, num_classes=0)
        self.embed_dim = self.backbone.num_features   # 1536
        self.dropout   = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, pixel_values, return_embedding: bool = False):
        features = self.backbone(pixel_values)
        features = self.dropout(features)
        if return_embedding:
            return features
        return self.classifier(features)

    def get_embedding(self, pixel_values) -> torch.Tensor:
        """Called by FusionModule — returns (batch, 1536)."""
        return self.forward(pixel_values, return_embedding=True)
