"""
ArogyaDrishti — Encoders (corrected, integrated v3.0)
=====================================================

Seven encoders, one uniform contract.  Every encoder exposes:

    .name        : modality name (str)
    .input_size  : expected square image side, e.g. 224 or 299 (vision only)
    .norm        : 'imagenet' | 'clip'  -> tells the preprocessor which
                   mean/std to use (vision only)
    .out_dim     : the embedding dimension this encoder contributes to fusion.
                   *Read from the backbone that actually loaded* — so the
                   downstream fusion module never breaks, even when a fallback
                   backbone is used.
    .num_classes : size of this encoder's own (optional) diagnostic head

    forward(x, return_embedding=False) -> logits | embedding
    get_embedding(x)                   -> embedding   (B, out_dim)

KEY FIXES vs the uploaded code
------------------------------
* BiomedCLIP (face + skin) is loaded via **open_clip**, not
  `transformers.CLIPVisionModel` — the HF loader cannot read that repo and
  crashes at construction.  out_dim is read from the model (512), not assumed.
* RETFound (eye) load is wrapped defensively; on failure it falls back to a
  ViT-Large (1024-dim) so the dimension stays consistent.
* Every encoder reads its real feature dimension instead of hard-coding it,
  so a fallback backbone never desynchronises the rest of the system.
* `pretrained` / `allow_random_fallback` flags let the whole system run
  offline (random weights) for testing, and load real medical weights in
  production.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Standard normalisation constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------- #
#  Base class                                                                  #
# --------------------------------------------------------------------------- #
class _BaseEncoder(nn.Module):
    """Shared plumbing.  Subclasses set self.backbone, self.out_dim and
    implement _extract(pixel_values) -> (B, out_dim)."""

    name: str = "base"
    input_size: int = 224
    norm: str = "imagenet"

    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        # classifier is created lazily once out_dim is known (see _finish)
        self.classifier: nn.Module | None = None

    def _finish(self):
        """Call after self.out_dim is set to build the diagnostic head."""
        assert self.out_dim, "out_dim must be set before _finish()"
        self.classifier = nn.Linear(self.out_dim, self.num_classes)

    def _extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, pixel_values: torch.Tensor, return_embedding: bool = False):
        emb = self._extract(pixel_values)           # (B, out_dim)
        emb = self.dropout(emb)
        if return_embedding:
            return emb
        return self.classifier(emb)

    def get_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward(pixel_values, return_embedding=True)


# --------------------------------------------------------------------------- #
#  Helper: timm backbone with graceful offline fallback                        #
# --------------------------------------------------------------------------- #
def _load_timm(model_name: str, pretrained: bool, allow_random_fallback: bool):
    """Return (backbone, num_features).  Falls back to random weights if the
    pretrained download is unavailable (e.g. offline) and fallback allowed."""
    import timm

    try:
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    except Exception as e:                                   # network/offline
        if not allow_random_fallback:
            raise
        print(f"   [{model_name}] pretrained unavailable ({type(e).__name__}); "
              f"using random init for offline run.")
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    return backbone, backbone.num_features


# --------------------------------------------------------------------------- #
#  1. Eye — RETFound (ViT-Large, 1024)                                         #
# --------------------------------------------------------------------------- #
class EyeEncoder(_BaseEncoder):
    name = "eye"
    input_size = 224
    norm = "imagenet"
    RETFOUND_ID = "iszt/RETFound_mae_meh"

    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 allow_random_fallback: bool = True, dropout: float = 0.3):
        super().__init__(num_classes, dropout)
        self._kind = None      # 'hf' or 'timm'
        self.backbone = None

        if pretrained:
            try:
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained(
                    self.RETFOUND_ID, trust_remote_code=True)
                self.out_dim = self.backbone.config.hidden_size      # 1024
                self._kind = "hf"
                print(f"✅ RETFound loaded: {self.out_dim}-dim")
            except Exception as e:
                print(f"⚠️  RETFound unavailable ({type(e).__name__}); "
                      f"falling back to ViT-Large.")

        if self.backbone is None:
            # ViT-Large keeps the 1024-dim contract that RETFound provides
            self.backbone, self.out_dim = _load_timm(
                "vit_large_patch16_224", pretrained, allow_random_fallback)
            self._kind = "timm"
        self._finish()

    def _extract(self, x):
        if self._kind == "hf":
            out = self.backbone(pixel_values=x)
            # prefer pooler_output, else CLS token of last_hidden_state
            if getattr(out, "pooler_output", None) is not None:
                return out.pooler_output
            return out.last_hidden_state[:, 0, :]
        return self.backbone(x)            # timm pooled features


# --------------------------------------------------------------------------- #
#  Shared BiomedCLIP loader (face + skin)                                      #
# --------------------------------------------------------------------------- #
class _BiomedCLIPEncoder(_BaseEncoder):
    """Loads BiomedCLIP the CORRECT way (open_clip).  Falls back to HF CLIP,
    then to a random ViT-Base for offline runs."""
    input_size = 224
    norm = "clip"
    BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, num_classes: int, pretrained: bool,
                 allow_random_fallback: bool, dropout: float):
        super().__init__(num_classes, dropout)
        self._kind = None
        self.backbone = None

        # 1) Correct path: open_clip
        if pretrained:
            try:
                import open_clip
                model, _, _ = open_clip.create_model_and_transforms(self.BIOMEDCLIP)
                self.backbone = model.visual          # vision tower only
                self.out_dim = int(model.visual.output_dim)   # 512
                self._kind = "openclip"
                print(f"✅ BiomedCLIP ({self.name}) loaded via open_clip: "
                      f"{self.out_dim}-dim")
            except Exception as e:
                print(f"⚠️  BiomedCLIP unavailable ({type(e).__name__}); "
                      f"falling back.")

        # 2) HF CLIP fallback (note: pooled output is 768-dim, not 512)
        if self.backbone is None and pretrained:
            try:
                from transformers import CLIPVisionModel
                self.backbone = CLIPVisionModel.from_pretrained(
                    "openai/clip-vit-base-patch32")
                self.out_dim = self.backbone.config.hidden_size       # 768
                self._kind = "hf"
                print(f"✅ CLIP fallback ({self.name}): {self.out_dim}-dim")
            except Exception as e:
                print(f"⚠️  HF CLIP unavailable ({type(e).__name__}); "
                      f"using random ViT-Base.")

        # 3) Offline random fallback
        if self.backbone is None:
            if not allow_random_fallback:
                raise RuntimeError(f"No backbone available for {self.name}")
            self.backbone, self.out_dim = _load_timm(
                "vit_base_patch16_224", pretrained=False,
                allow_random_fallback=True)
            self._kind = "timm"
        self._finish()

    def _extract(self, x):
        if self._kind == "openclip":
            return self.backbone(x)                       # (B, 512) projected
        if self._kind == "hf":
            return self.backbone(pixel_values=x).pooler_output   # (B, 768)
        return self.backbone(x)                           # timm (B, 768)


class FaceEncoder(_BiomedCLIPEncoder):
    name = "face"

    def __init__(self, num_classes: int = 4, pretrained: bool = True,
                 allow_random_fallback: bool = True, dropout: float = 0.3):
        super().__init__(num_classes, pretrained, allow_random_fallback, dropout)


class SkinEncoder(_BiomedCLIPEncoder):
    name = "skin"

    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 allow_random_fallback: bool = True, dropout: float = 0.3):
        super().__init__(num_classes, pretrained, allow_random_fallback, dropout)


# --------------------------------------------------------------------------- #
#  3. Tongue — InceptionV3 (2048, 299x299)                                     #
# --------------------------------------------------------------------------- #
class TongueEncoder(_BaseEncoder):
    name = "tongue"
    input_size = 299                       # InceptionV3 requirement
    norm = "imagenet"

    def __init__(self, num_classes: int = 4, pretrained: bool = True,
                 allow_random_fallback: bool = True, dropout: float = 0.3):
        super().__init__(num_classes, dropout)
        self.backbone, self.out_dim = _load_timm(
            "inception_v3", pretrained, allow_random_fallback)   # 2048
        self._finish()
        print(f"✅ InceptionV3 (tongue): {self.out_dim}-dim, input 299x299")

    def _extract(self, x):
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"TongueEncoder expects {self.input_size}x{self.input_size}, "
                f"got {tuple(x.shape[-2:])}. Use the matching preprocessor.")
        return self.backbone(x)


# --------------------------------------------------------------------------- #
#  4. Nail — DenseNet201 (1920)                                                #
# --------------------------------------------------------------------------- #
class NailEncoder(_BaseEncoder):
    name = "nail"
    input_size = 224
    norm = "imagenet"

    def __init__(self, num_classes: int = 6, pretrained: bool = True,
                 allow_random_fallback: bool = True, dropout: float = 0.3):
        super().__init__(num_classes, dropout)
        self.backbone, self.out_dim = _load_timm(
            "densenet201", pretrained, allow_random_fallback)     # 1920
        self._finish()
        print(f"✅ DenseNet201 (nail): {self.out_dim}-dim")

    def _extract(self, x):
        return self.backbone(x)


# --------------------------------------------------------------------------- #
#  5. Palm — DenseNet169 (1664)                                                #
# --------------------------------------------------------------------------- #
class PalmEncoder(_BaseEncoder):
    name = "palm"
    input_size = 224
    norm = "imagenet"

    def __init__(self, num_classes: int = 5, pretrained: bool = True,
                 allow_random_fallback: bool = True, dropout: float = 0.3):
        super().__init__(num_classes, dropout)
        self.backbone, self.out_dim = _load_timm(
            "densenet169", pretrained, allow_random_fallback)     # 1664
        self._finish()
        print(f"✅ DenseNet169 (palm): {self.out_dim}-dim")

    def _extract(self, x):
        return self.backbone(x)


# --------------------------------------------------------------------------- #
#  7. Tabular — Custom MLP (21 -> 100-dim embedding)                           #
# --------------------------------------------------------------------------- #
class TabularEncoder(nn.Module):
    """3-layer MLP per the architecture spec.

    Input (21 features) -> BN -> 21->256 GELU Dropout -> BN -> 256->256 GELU
    Dropout -> BN -> 256->128 GELU Dropout -> 128->100 embedding.
    """
    name = "tabular"

    def __init__(self, in_features: int = 21, embed_dim: int = 100,
                 num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.out_dim = embed_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256), nn.GELU(), nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        print(f"✅ Tabular MLP: {in_features} -> {embed_dim}-dim")

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        emb = self.net(x)
        if return_embedding:
            return emb
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)
