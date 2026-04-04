"""
ArogyaDrishti — Skin Encoder
Pretrained model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
Labels (HAM10000): 0=mel, 1=nv, 2=bcc, 3=akiec, 4=bkl, 5=df, 6=vasc
"""
import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class SkinEncoder(nn.Module):
    def __init__(self, num_classes=7,
                 pretrained_model="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 dropout=0.3):
        super().__init__()
        try:
            self.backbone = CLIPVisionModel.from_pretrained(pretrained_model)
            self.embed_dim = self.backbone.config.hidden_size
        except Exception:
            print("BiomedCLIP not available, falling back to openai/clip-vit-base-patch32")
            self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.embed_dim = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, pixel_values, return_embedding=False):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        if return_embedding:
            return pooled
        return self.classifier(pooled)

    def get_embedding(self, pixel_values):
        return self.forward(pixel_values, return_embedding=True)
