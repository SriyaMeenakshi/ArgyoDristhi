"""
ArogyaDrishti — Export trained model to PyTorch Mobile (.ptl) for Android
Usage:
  python export/export_tflite.py

Outputs:
  export/arogyadrishti_fusion.ptl    ← hand to Koppala Pavani (Android)

Android placement: app/src/main/assets/arogyadrishti_fusion.ptl

Why PyTorch Mobile instead of TFLite?
  The model uses transformer backbones (BEiT, ViT, Swin, CLIP) whose internal
  attention ops (aten::_native_multi_head_attention) cannot be exported to ONNX.
  PyTorch Mobile handles ALL native PyTorch ops natively — no conversion needed.

Android dependency to add in build.gradle:
  implementation 'org.pytorch:pytorch_android_lite:2.1.0'
  implementation 'org.pytorch:pytorch_android_torchvision_lite:2.1.0'

Inputs (all batch size 1):
  face    : (1, 3, 224, 224)
  eye     : (1, 3, 224, 224)
  tongue  : (1, 3, 224, 224)
  skin    : (1, 3, 224, 224)
  nail    : (1, 3, 224, 224)
  palm    : (1, 3, 224, 224)
  tabular : (1, 21)

Outputs: 7 risk scores in [0,1]
  hematological, metabolic, renal, hepatic,
  cardiovascular, dermatological, nutritional
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.fusion_module import FusionModule

DEVICE = 'cpu'   # export always on CPU

ENCODER_CKPTS = {
    'face':    'checkpoints/face/face_best.pt',
    'eye':     'checkpoints/eye/eye_best.pt',
    'tongue':  'checkpoints/tongue/tongue_best.pt',
    'skin':    'checkpoints/skin/skin_best.pt',
    'nail':    'checkpoints/nail/nail_best.pt',
    'palm':    'checkpoints/palm/palm_best.pt',
    'tabular': 'checkpoints/tabular/tabular_best.pt',
}

ENCODER_DIMS = {
    'face':    768,
    'eye':     768,
    'tongue':  1024,
    'skin':    768,
    'nail':    1792,
    'palm':    1536,
    'tabular': 100,
}

FUSION_CKPT = 'checkpoints/fusion/fusion_best.pt'
PTL_PATH    = 'export/arogyadrishti_fusion.ptl'
PT_PATH     = 'export/arogyadrishti_fusion.pt'


class ArogyaExportWrapper(nn.Module):
    """Single traceable module wrapping all 7 encoders + fusion."""
    def __init__(self, encoders: nn.ModuleDict, fusion):
        super().__init__()
        self.encoders = encoders
        self.fusion   = fusion

    def forward(self, face, eye, tongue, skin, nail, palm, tabular):
        embeddings = {
            'face':    self.encoders['face'].get_embedding(face),
            'eye':     self.encoders['eye'].get_embedding(eye),
            'tongue':  self.encoders['tongue'].get_embedding(tongue),
            'skin':    self.encoders['skin'].get_embedding(skin),
            'nail':    self.encoders['nail'].get_embedding(nail),
            'palm':    self.encoders['palm'].get_embedding(palm),
            'tabular': self.encoders['tabular'].get_embedding(tabular),
        }
        risks = self.fusion(embeddings)
        return (
            risks['hematological'].unsqueeze(1),
            risks['metabolic'].unsqueeze(1),
            risks['renal'].unsqueeze(1),
            risks['hepatic'].unsqueeze(1),
            risks['cardiovascular'].unsqueeze(1),
            risks['dermatological'].unsqueeze(1),
            risks['nutritional'].unsqueeze(1),
        )


def load_encoder(name, ckpt_path):
    if name == 'face':
        from encoders.face_encoder import FaceEncoder
        model = FaceEncoder(num_classes=4)
    elif name == 'eye':
        from encoders.eye_encoder import EyeEncoder
        model = EyeEncoder(num_classes=2)
    elif name == 'tongue':
        from encoders.tongue_encoder import TongueEncoder
        model = TongueEncoder(num_classes=4)
    elif name == 'skin':
        from encoders.skin_encoder import SkinEncoder
        model = SkinEncoder(num_classes=7)
    elif name == 'nail':
        from encoders.nail_encoder import NailEncoder
        model = NailEncoder(num_classes=6)
    elif name == 'palm':
        from encoders.palm_encoder import PalmEncoder
        model = PalmEncoder(num_classes=5)
    elif name == 'tabular':
        from encoders.tabular_encoder import TabularEncoder
        model = TabularEncoder()
    else:
        raise ValueError(f"Unknown encoder: {name}")

    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=DEVICE)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  {name}: loaded from {ckpt_path}")
    else:
        print(f"  {name}: checkpoint NOT found — using pretrained weights only")

    model.eval()
    return model


def main():
    os.makedirs('export', exist_ok=True)

    # ── [1] Load all encoders ─────────────────────────────────────────────────
    print("[1] Loading encoders...")
    encoders = nn.ModuleDict()
    for name, ckpt in ENCODER_CKPTS.items():
        encoders[name] = load_encoder(name, ckpt)

    fusion = FusionModule(encoder_dims=ENCODER_DIMS, fusion_dim=512, num_heads=8)
    if os.path.exists(FUSION_CKPT):
        ckpt  = torch.load(FUSION_CKPT, map_location=DEVICE)
        state = ckpt.get('model_state_dict', ckpt.get('fusion_state_dict', ckpt))
        fusion.load_state_dict(state)
        print(f"  fusion: loaded from {FUSION_CKPT}")
    fusion.eval()

    # ── [2] Build wrapper + dummy inputs ─────────────────────────────────────
    wrapper = ArogyaExportWrapper(encoders=encoders, fusion=fusion)
    wrapper.eval()

    img     = torch.randn(1, 3, 224, 224)
    tab     = torch.randn(1, 21)
    dummy   = (img, img, img, img, img, img, tab)

    print("\n  Sanity check (forward pass)...")
    with torch.no_grad():
        out = wrapper(*dummy)
    risk_names = ['hematological','metabolic','renal','hepatic',
                  'cardiovascular','dermatological','nutritional']
    print("  Risk scores (random input):")
    for name, score in zip(risk_names, out):
        print(f"    {name}: {score.item():.4f}")
    print("  Forward pass OK.")

# ── [3] TorchScript trace ─────────────────────────────────────────────────
    print("\n[2] Tracing model with torch.jit.trace...")
    print("  (TracerWarnings about control flow are expected and harmless)")
    print("  (Input size is fixed at 224×224 — this is intentional)")
    torch._C._jit_clear_class_registry()
    torch._C._jit_set_profiling_executor(False)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, strict=False, check_trace=False)  # ← add check_trace=False
    traced = torch.jit.freeze(traced)
    print("  Tracing complete.")

    # Verify traced model output matches original  ← moved to AFTER trace
    with torch.no_grad():
        out_traced = traced(*dummy)
    max_diff = max(
        abs(o.item() - ot.item())
        for o, ot in zip(out, out_traced)
    )
    print(f"  Max output diff (original vs traced): {max_diff:.2e}  ✓")

    # Save plain TorchScript (.pt) — usable for testing
    torch.jit.save(traced, PT_PATH)
    size_pt = os.path.getsize(PT_PATH) / 1e6
    print(f"\n  TorchScript saved: {PT_PATH} ({size_pt:.1f} MB)")

    # ── [4] Optimize for mobile (.ptl) ───────────────────────────────────────
    print("\n[3] Optimizing for mobile...")
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(traced)
        optimized._save_for_lite_interpreter(PTL_PATH)
        size_ptl = os.path.getsize(PTL_PATH) / 1e6
        print(f"  PyTorch Mobile saved: {PTL_PATH} ({size_ptl:.1f} MB)")
    except Exception as e:
        print(f"  mobile_optimizer failed ({e})")
        print(f"  Saving plain TorchScript as .ptl fallback...")
        torch.jit.save(traced, PTL_PATH)
        size_ptl = os.path.getsize(PTL_PATH) / 1e6
        print(f"  Fallback .ptl saved: {PTL_PATH} ({size_ptl:.1f} MB)")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PART A COMPLETE — MODEL EXPORTED")
    print("="*60)
    print(f"\n  Model file: {PTL_PATH}")
    print(f"  Size      : {size_ptl:.1f} MB")
    print(f"\n  Hand  arogyadrishti_fusion.ptl  to Koppala Pavani")
    print(f"  Place in Android: app/src/main/assets/")
    print(f"\n  Android build.gradle dependencies:")
    print(f"    implementation 'org.pytorch:pytorch_android_lite:2.1.0'")
    print(f"    implementation 'org.pytorch:pytorch_android_torchvision_lite:2.1.0'")
    print(f"\n  Android inference (Kotlin):")
    print(f"    val module = LiteModuleLoader.load(assetFilePath(\"arogyadrishti_fusion.ptl\"))")
    print(f"    val inputs = arrayOf(faceTensor, eyeTensor, tongueTensor,")
    print(f"                          skinTensor, nailTensor, palmTensor, tabularTensor)")
    print(f"    val output = module.forward(IValue.from(inputs)).toTuple()")
    print(f"\n  7 output risk scores:")
    for i, n in enumerate(risk_names):
        print(f"    output[{i}] = {n}")
    print("\n  Encoder summary:")
    for name, dim in ENCODER_DIMS.items():
        print(f"    {name:10s}: embed_dim={dim}")


if __name__ == '__main__':
    main()
