"""ArogyaDrishti — multimodal health screening (corrected, integrated v3.0)."""
from .system import ArogyaDrishti
from .fusion import CrossModalFusion, RISK_CATEGORIES
from .encoders import (
    EyeEncoder, FaceEncoder, TongueEncoder, SkinEncoder,
    NailEncoder, PalmEncoder, TabularEncoder,
)
from .preprocessing import (
    preprocess_image, preprocess_tabular, make_image_transform, FEATURE_ORDER,
)

__all__ = [
    "ArogyaDrishti", "CrossModalFusion", "RISK_CATEGORIES",
    "EyeEncoder", "FaceEncoder", "TongueEncoder", "SkinEncoder",
    "NailEncoder", "PalmEncoder", "TabularEncoder",
    "preprocess_image", "preprocess_tabular", "make_image_transform",
    "FEATURE_ORDER",
]
__version__ = "3.0.0"
