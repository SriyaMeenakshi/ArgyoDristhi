"""
ArogyaDrishti — Preprocessing (corrected v3.0)
==============================================

Two jobs the uploaded code never did:

1. Build the CORRECT image transform per encoder.  Vision backbones do not all
   use the same normalisation: ImageNet-trained CNNs/ViTs use ImageNet
   mean/std, while CLIP / BiomedCLIP use CLIP mean/std.  InceptionV3 needs
   299x299 while everything else uses 224x224.  Each encoder advertises its
   `.input_size` and `.norm`, and we read them here.

2. Turn a dict of 21 named health features into a correctly-ordered,
   standardised tensor for the tabular encoder.
"""
from __future__ import annotations

import torch
from PIL import Image
import torchvision.transforms as T

from .encoders import IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD


def make_image_transform(encoder):
    """Return a torchvision transform matching this encoder's requirements."""
    if encoder.norm == "clip":
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    size = encoder.input_size
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_image(path_or_pil) -> Image.Image:
    if isinstance(path_or_pil, Image.Image):
        return path_or_pil.convert("RGB")
    return Image.open(path_or_pil).convert("RGB")


def preprocess_image(path_or_pil, encoder) -> torch.Tensor:
    """-> (1, 3, H, W) tensor ready for the given encoder."""
    img = load_image(path_or_pil)
    tfm = make_image_transform(encoder)
    return tfm(img).unsqueeze(0)


# --------------------------------------------------------------------------- #
#  Tabular features                                                            #
# --------------------------------------------------------------------------- #
# 21 features in the fixed order expected by the tabular encoder.
FEATURE_ORDER = [
    "age", "gender", "bmi",                                   # demographics
    "systolic_bp", "diastolic_bp", "heart_rate", "spo2",      # vitals
    "hemoglobin", "glucose_fasting", "glucose_pp", "hba1c",   # hematology
    "alt", "ast", "bilirubin_total",                          # hepatic
    "creatinine", "urea",                                     # renal
    "ferritin", "vitamin_b12", "vitamin_d", "albumin",        # nutrition
    "fatigue_score",                                          # symptoms
]

# Approximate (mean, std) for standardisation.  These are reasonable clinical
# reference values used so the untrained network receives sane inputs; replace
# with statistics computed from your training set before deployment.
FEATURE_STATS = {
    "age": (40, 18), "gender": (0.5, 0.5), "bmi": (24, 5),
    "systolic_bp": (120, 18), "diastolic_bp": (80, 12),
    "heart_rate": (75, 14), "spo2": (97, 3),
    "hemoglobin": (13, 2.5), "glucose_fasting": (95, 30),
    "glucose_pp": (130, 45), "hba1c": (5.6, 1.4),
    "alt": (30, 20), "ast": (28, 18), "bilirubin_total": (0.8, 0.5),
    "creatinine": (0.9, 0.3), "urea": (28, 12),
    "ferritin": (120, 90), "vitamin_b12": (450, 220),
    "vitamin_d": (28, 12), "albumin": (4.3, 0.5),
    "fatigue_score": (1.0, 1.0),
}


def encode_gender(value) -> float:
    """Map M/F (or 0/1) to a float. M->0.0, F->1.0."""
    if isinstance(value, (int, float)):
        return float(value)
    return 1.0 if str(value).strip().lower().startswith("f") else 0.0


def preprocess_tabular(features: dict) -> torch.Tensor:
    """
    Args:
        features: dict using the keys in FEATURE_ORDER. Missing numeric keys
                  default to the population mean (i.e. "no information").
    Returns:
        (1, 21) standardised float tensor.
    """
    vals = []
    for k in FEATURE_ORDER:
        mean, std = FEATURE_STATS[k]
        if k not in features or features[k] is None:
            raw = mean                       # neutral default
        elif k == "gender":
            raw = encode_gender(features[k])
        else:
            raw = float(features[k])
        vals.append((raw - mean) / (std if std else 1.0))
    return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
