"""
ArogyaDrishti — Inference Script
=================================
Each modality only shows the risk scores it can actually detect.

Usage
-----
Quick scan (single modality):
    python inference.py --eye patient_eye.jpg
    python inference.py --palm patient_palm.jpg
    python inference.py --nail patient_nail.jpg
    python inference.py --skin patient_skin.jpg
    python inference.py --face patient_face.jpg
    python inference.py --tongue patient_tongue.jpg

Multi modality:
    python inference.py --eye eye.jpg --palm palm.jpg

Full scan:
    python inference.py --eye eye.jpg --face face.jpg --tongue tongue.jpg --skin skin.jpg --nail nail.jpg --palm palm.jpg
"""
from __future__ import annotations

import argparse
import os
import torch

from arogyadrishti.system import ArogyaDrishti
from arogyadrishti.preprocessing import load_image
from arogyadrishti.training.train_utils import build_eval_transform


def risk_level(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MODERATE"
    return "LOW"


def load_system(ckpt_dir: str, device: str):
    from arogyadrishti.encoders import (
        EyeEncoder, FaceEncoder, TongueEncoder,
        SkinEncoder, NailEncoder, PalmEncoder,
    )
    ENCODER_CLASSES = {
        "eye": EyeEncoder,
        "face": FaceEncoder,
        "tongue": TongueEncoder,
        "skin": SkinEncoder,
        "nail": NailEncoder,
        "palm": PalmEncoder,
    }
    # these need pretrained backbone for correct dimensions
    NEEDS_PRETRAINED = ["face", "skin", "eye"]

    system = ArogyaDrishti(pretrained=False, device=device)
    for modality, enc_cls in ENCODER_CLASSES.items():
        ckpt_path = os.path.join(ckpt_dir, f"{modality}_best.pt")
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        classes = ckpt.get("classes", [])
        num_classes = len(classes)
        pretrained = modality in NEEDS_PRETRAINED
        enc = enc_cls(num_classes=num_classes, pretrained=pretrained)
        state = ckpt["state_dict"]
        model_state = enc.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        model_state.update(compatible)
        enc.load_state_dict(model_state)
        enc.classes_ = classes
        enc.to(device)
        system.image_encoders[modality] = enc
        print(f"Loaded {modality} ({num_classes} classes, {len(compatible)}/{len(state)} weights)")
    return system


def predict_modality(enc, image_path: str, ckpt_path: str, device: str):
    """Returns (detected_class, confidence, top3_results)."""
    classes = getattr(enc, 'classes_', [])
    if not classes:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        classes = ckpt.get("classes", [])
    img = load_image(image_path)
    tfm = build_eval_transform(enc)
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = enc(x)
        probs = torch.softmax(logits, dim=1)
        top3 = torch.topk(probs, min(3, len(classes)), dim=1)
        top3_idx = top3.indices[0].tolist()
        top3_conf = top3.values[0].tolist()
    detected = classes[top3_idx[0]] if classes else str(top3_idx[0])
    conf = top3_conf[0]
    top3_results = [
        (classes[i] if classes else str(i), c)
        for i, c in zip(top3_idx, top3_conf)
    ]
    return detected, conf, top3_results


def prediction_to_risk(modality: str, detected: str, confidence: float) -> dict:
    risks = {}
    if modality == "eye":
        risks["hematological"] = confidence if detected == "anemia" else (1 - confidence) * 0.3
    elif modality == "palm":
        risks["hematological"] = confidence if detected == "anemic" else (1 - confidence) * 0.3
    elif modality == "face":
        risks["hepatic"] = confidence if detected == "diseased" else (1 - confidence) * 0.3
        risks["metabolic"] = confidence * 0.8 if detected == "diseased" else (1 - confidence) * 0.3
    elif modality == "tongue":
        risks["hepatic"] = confidence if detected == "abnormal" else (1 - confidence) * 0.3
        risks["nutritional"] = confidence * 0.8 if detected == "abnormal" else (1 - confidence) * 0.3
    elif modality == "nail":
        cardio_classes = ["clubbing", "bluish nail", "red lunula", "splinter hemmorrage", "cardiovascular"]
        hema_classes = ["pale nail", "koilonychia", "leukonychia", "white nail",
                        "half and half nailes (Lindsay_s nails)", "hematological"]
        if detected in cardio_classes:
            risks["cardiovascular"] = confidence
        elif detected in hema_classes:
            risks["hematological"] = confidence
        else:
            risks["cardiovascular"] = confidence * 0.5
            risks["hematological"] = confidence * 0.5
    elif modality == "skin":
        risks["dermatological"] = confidence
    return risks


def print_report(modality_results: dict):
    print("\n" + "=" * 50)
    print("      ArogyaDrishti Screening Report")
    print("=" * 50)

    print("\nPer Modality Findings:")
    print("-" * 50)
    for modality, info in modality_results.items():
        detected = info["detected"]
        conf = info["confidence"]
        print(f"\n[{modality.upper()}]: {detected} ({int(conf * 100)}% confidence)")
        if "top3" in info and len(info["top3"]) > 1:
            print("   Top predictions:")
            for cls, c in info["top3"]:
                print(f"   - {cls}: {int(c * 100)}%")

    print("\n" + "-" * 50)
    print("Risk Scores:")
    print("-" * 50)

    all_risks = {}
    for modality, info in modality_results.items():
        for risk, score in info["risks"].items():
            if risk not in all_risks:
                all_risks[risk] = []
            all_risks[risk].append(score)

    for risk, scores in sorted(all_risks.items()):
        score = max(scores)
        level = risk_level(score)
        bar = "#" * int(score * 10) + "." * (10 - int(score * 10))
        print(f"\n[{level}] {risk.upper():20s}: {int(score * 100)}%  [{bar}]")

    all_confs = [info["confidence"] for info in modality_results.values()]
    overall_conf = sum(all_confs) / len(all_confs)
    needs_review = overall_conf < 0.6 or any(
        max(info["risks"].values()) >= 0.7
        for info in modality_results.values()
        if info["risks"]
    )

    print("\n" + "-" * 50)
    print(f"Overall Confidence  : {int(overall_conf * 100)}%")
    print(f"Needs Manual Review : {'YES' if needs_review else 'NO'}")
    print("=" * 50 + "\n")


def main():
    ap = argparse.ArgumentParser(description="ArogyaDrishti Inference")
    ap.add_argument("--eye",      default=None)
    ap.add_argument("--face",     default=None)
    ap.add_argument("--tongue",   default=None)
    ap.add_argument("--skin",     default=None)
    ap.add_argument("--nail",     default=None)
    ap.add_argument("--palm",     default=None)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    images = {}
    for modality in ["eye", "face", "tongue", "skin", "nail", "palm"]:
        path = getattr(args, modality)
        if path:
            images[modality] = path

    if not images:
        print("No images provided.")
        print("Usage: python inference.py --eye patient_eye.jpg")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nScanning: {list(images.keys())}  |  Device: {device}")

    system = load_system(args.ckpt_dir, device)

    modality_results = {}
    for modality, image_path in images.items():
        ckpt_path = os.path.join(args.ckpt_dir, f"{modality}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint for {modality} - skipping")
            continue
        enc = system.image_encoders[modality]
        detected, confidence, top3 = predict_modality(enc, image_path, ckpt_path, device)
        risks = prediction_to_risk(modality, detected, confidence)
        modality_results[modality] = {
            "detected": detected,
            "confidence": confidence,
            "top3": top3,
            "risks": risks,
        }
        print(f"Done: {modality}: {detected} ({int(confidence * 100)}%)")

    if not modality_results:
        print("No valid predictions made.")
        return

    print_report(modality_results)


if __name__ == "__main__":
    main()
