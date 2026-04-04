"""
ArogyaDrishti — Setup verification script
Run this to confirm everything is working before starting training.
Usage: python train/verify_setup.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def check(label, fn):
    try:
        result = fn()
        print(f"  [PASS] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def main():
    print("=" * 55)
    print("  ArogyaDrishti — Environment Verification")
    print("=" * 55)

    results = []

    # Core
    print("\n[Core]")
    results.append(check("PyTorch version", lambda: torch.__version__))
    results.append(check("CUDA available", lambda: str(torch.cuda.is_available())))
    results.append(check("GPU name", lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"))
    results.append(check("GPU memory (GB)", lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}" if torch.cuda.is_available() else "N/A"))

    # Packages
    print("\n[Packages]")
    pkgs = ['transformers','timm','einops','accelerate','datasets',
            'albumentations','wandb','onnx','onnxruntime','h5py','tabpfn']
    for pkg in pkgs:
        results.append(check(pkg, lambda p=pkg: __import__(p).__version__))
    results.append(check("pytorch_grad_cam", lambda: __import__('pytorch_grad_cam') and "installed"))

    # Encoder instantiation (no download — just structure check)
    print("\n[Encoder structure]")
    device = torch.device('cpu')
    dummy = torch.randn(2, 3, 256, 256).to(device)

    def test_nail():
        from encoders.nail_encoder import NailEncoder
        m = NailEncoder(num_classes=6).to(device)
        out = m(dummy)
        emb = m.get_embedding(dummy)
        return f"logits {tuple(out.shape)}, embedding {tuple(emb.shape)}"
    results.append(check("NailEncoder (timm — no download)", test_nail))

    # Fusion module structure check
    def test_fusion():
        from encoders.fusion_module import FusionModule
        dims = {'face': 768, 'tongue': 1024, 'skin': 512, 'nail': 1792}
        fusion = FusionModule(encoder_dims=dims, fusion_dim=256, num_heads=4).to(device)
        embeddings = {k: torch.randn(2, d).to(device) for k, d in dims.items()}
        risks = fusion(embeddings)
        return {k: tuple(v.shape) for k, v in risks.items()}
    results.append(check("FusionModule (7 risk heads)", test_fusion))

    # Project structure
    print("\n[Project folders]")
    required_dirs = [
        'data/raw/face', 'data/raw/eye', 'data/raw/tongue',
        'data/raw/skin', 'data/raw/nail_palm',
        'data/splits', 'checkpoints/face', 'checkpoints/fusion',
    ]
    for d in required_dirs:
        results.append(check(f"dir: {d}", lambda d=d: "exists" if os.path.isdir(d) else (_ for _ in ()).throw(FileNotFoundError(d))))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*55}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  All good — ready to train!")
    else:
        print("  Fix the FAIL items above before training.")
    print("=" * 55)


if __name__ == '__main__':
    main()
