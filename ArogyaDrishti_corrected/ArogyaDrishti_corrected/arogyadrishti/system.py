"""
ArogyaDrishti — Integrated System (corrected v3.0)
==================================================

The single entry point that the uploaded code lacked.  Build the system once,
then call `analyze_person(...)` with whatever modalities are available for a
given individual.  Missing modalities are handled gracefully.

    system = ArogyaDrishti(pretrained=True)          # loads real medical weights
    result = system.analyze_person(
        images={"eye": "eye.jpg", "palm": "palm.jpg"},   # any subset
        tabular={"age": 34, "gender": "F", "hemoglobin": 8.1, ...},
    )
    print(result["risk_scores"])        # 7 calibrated scores in [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import (
    EyeEncoder, FaceEncoder, TongueEncoder, SkinEncoder,
    NailEncoder, PalmEncoder, TabularEncoder,
)
from .fusion import CrossModalFusion, RISK_CATEGORIES
from .preprocessing import preprocess_image, preprocess_tabular

# Which encoder class handles each image modality
_IMAGE_ENCODERS = {
    "eye": EyeEncoder,
    "face": FaceEncoder,
    "tongue": TongueEncoder,
    "skin": SkinEncoder,
    "nail": NailEncoder,
    "palm": PalmEncoder,
}


def _risk_band(score: float) -> str:
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "moderate"
    return "high"


class ArogyaDrishti(nn.Module):
    def __init__(self, pretrained: bool = True,
                 allow_random_fallback: bool = True,
                 device: str | None = None):
        super().__init__()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 1) build all image encoders
        self.image_encoders = nn.ModuleDict()
        for name, cls in _IMAGE_ENCODERS.items():
            self.image_encoders[name] = cls(
                pretrained=pretrained,
                allow_random_fallback=allow_random_fallback)

        # 2) build tabular encoder
        self.tabular_encoder = TabularEncoder()

        # 3) build fusion using the dims that ACTUALLY loaded
        modality_dims = {n: e.out_dim for n, e in self.image_encoders.items()}
        modality_dims["tabular"] = self.tabular_encoder.out_dim
        self.fusion = CrossModalFusion(modality_dims)

        self.to(self.device)
        self.eval()
        print(f"\nArogyaDrishti ready on {self.device}. "
              f"Modality dims: {modality_dims}")

    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def analyze_person(self, images: dict | None = None,
                       tabular: dict | None = None) -> dict:
        """
        Args:
            images:  {modality: PIL.Image | path}.  Any subset of
                     eye/face/tongue/skin/nail/palm, or None.
            tabular: dict of named health features (see preprocessing.
                     FEATURE_ORDER) or None.
        Returns:
            structured result dict (see bottom of this method).
        """
        self.eval()
        images = images or {}
        embeddings: dict[str, torch.Tensor] = {}

        # image modalities
        for name, img in images.items():
            if name not in self.image_encoders:
                raise ValueError(f"Unknown image modality '{name}'. "
                                 f"Valid: {list(self.image_encoders)}")
            enc = self.image_encoders[name]
            x = preprocess_image(img, enc).to(self.device)
            embeddings[name] = enc.get_embedding(x)

        # tabular modality
        if tabular is not None:
            x = preprocess_tabular(tabular).to(self.device)
            embeddings["tabular"] = self.tabular_encoder.get_embedding(x)

        if not embeddings:
            raise ValueError(
                "No modalities provided. Supply at least one image or the "
                "tabular health features.")

        out = self.fusion(embeddings)

        risks = out["risks"].squeeze(0).cpu().tolist()
        modality_conf = {m: float(v.squeeze(0).cpu())
                         for m, v in out["modality_conf"].items()}
        overall_conf = float(out["overall_conf"].squeeze(0).cpu())

        risk_scores = {cat: round(score, 4)
                       for cat, score in zip(RISK_CATEGORIES, risks)}
        risk_bands = {cat: _risk_band(score)
                      for cat, score in risk_scores.items()}

        all_modalities = list(self.image_encoders) + ["tabular"]
        used = out["used_modalities"]
        missing = [m for m in all_modalities if m not in used]

        # clinical-safety flag: low confidence -> recommend manual review
        needs_review = overall_conf < 0.5 or len(used) < 2

        return {
            "risk_scores": risk_scores,             # {category: 0..1}
            "risk_bands": risk_bands,               # {category: low/mod/high}
            "modality_confidence": modality_conf,   # per-modality 0..1
            "overall_confidence": round(overall_conf, 4),
            "used_modalities": used,
            "missing_modalities": missing,
            "needs_manual_review": needs_review,
            "disclaimer": ("Screening aid only — not a diagnosis. "
                           "Confirm findings with clinical examination and "
                           "laboratory tests."),
        }
