"""
ArogyaDrishti — Tongue Encoder
Pretrained model: microsoft/swin-base-patch4-window7-224-in22k
Labels: 0=normal, 1=tooth-marked, 2=pale, 3=coated
"""
import torch
import torch.nn as nn
from transformers import SwinModel


class TongueEncoder(nn.Module):
    def __init__(self, num_classes=4, pretrained_model="microsoft/swin-base-patch4-window7-224-in22k",
                 dropout=0.3):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(pretrained_model)
        self.embed_dim = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, pixel_values, return_embedding=False):
        outputs = self.backbone(pixel_values=pixel_values)
        # Pooled output — shape (batch, embed_dim)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        if return_embedding:
            return pooled
        return self.classifier(pooled)

    def get_embedding(self, pixel_values):
        return self.forward(pixel_values, return_embedding=True)
