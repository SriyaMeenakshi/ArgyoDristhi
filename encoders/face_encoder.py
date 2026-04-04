"""
ArogyaDrishti — Face Encoder
Pretrained model: microsoft/beit-base-patch16-224-pt22k-ft22k
Task: classify face images into pallor/anemia signs, dark circles, genetic syndrome features
Labels: 0=healthy, 1=pallor/anemia, 2=dark circles/sleep deprivation, 3=genetic syndrome features
"""
import torch
import torch.nn as nn
from transformers import BeitModel, BeitConfig


class FaceEncoder(nn.Module):
    def __init__(self, num_classes=4, pretrained_model="microsoft/beit-base-patch16-224-pt22k-ft22k",
                 dropout=0.3, embed_dim=768):
        super().__init__()
        # BEiT requires exactly 224x224 — ignore size mismatch from position embeddings
        self.backbone = BeitModel.from_pretrained(
            pretrained_model, ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        # Expose embedding size for fusion module
        self.embed_dim = embed_dim

    def forward(self, pixel_values, return_embedding=False):
        outputs = self.backbone(pixel_values=pixel_values)
        # CLS token embedding — shape (batch, embed_dim)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        if return_embedding:
            return cls_embedding
        return self.classifier(cls_embedding)

    def get_embedding(self, pixel_values):
        return self.forward(pixel_values, return_embedding=True)
