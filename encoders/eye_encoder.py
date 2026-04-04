"""
ArogyaDrishti — Eye Encoder
Pretrained model: YunchengJiang/RETFound_cfp (trained on 1.6M eye photos)
Labels: 0=healthy conjunctiva, 1=pale conjunctiva (anemia), 2=yellow sclera (jaundice)
"""
import torch
import torch.nn as nn
from transformers import ViTModel


class EyeEncoder(nn.Module):
    def __init__(self, num_classes=3,
                 pretrained_model="YunchengJiang/RETFound_cfp",
                 dropout=0.3, embed_dim=1024):
        super().__init__()
        # RETFound is a ViT-Large model
        try:
            self.backbone = ViTModel.from_pretrained(pretrained_model)
            self.embed_dim = self.backbone.config.hidden_size
        except Exception:
            # Fallback to standard ViT-Base if RETFound is unavailable
            print("RETFound not available, falling back to google/vit-base-patch16-224-in21k")
            self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.embed_dim = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, pixel_values, return_embedding=False):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        if return_embedding:
            return cls_embedding
        return self.classifier(cls_embedding)

    def get_embedding(self, pixel_values):
        return self.forward(pixel_values, return_embedding=True)
