"""
ArogyaDrishti — Nail/Palm Encoder
Pretrained model: EfficientNet-B4 via timm (tf_efficientnet_b4)
Labels: 0=healthy, 1=clubbing, 2=blue_finger(cyanosis), 3=onychogryphosis, 4=pitting, 5=acral_lentiginous_melanoma
"""
import torch
import torch.nn as nn
import timm


class NailEncoder(nn.Module):
    def __init__(self, num_classes=6,
                 pretrained_model="tf_efficientnet_b4",
                 dropout=0.3):
        super().__init__()
        # timm's EfficientNet-B4 with pretrained ImageNet weights
        self.backbone = timm.create_model(pretrained_model, pretrained=True, num_classes=0)
        self.embed_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, pixel_values, return_embedding=False):
        features = self.backbone(pixel_values)
        features = self.dropout(features)
        if return_embedding:
            return features
        return self.classifier(features)

    def get_embedding(self, pixel_values):
        return self.forward(pixel_values, return_embedding=True)
