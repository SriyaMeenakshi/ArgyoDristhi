"""
ArogyaDrishti — end-to-end demo / self-test
============================================

Runs the FULL pipeline offline with randomly-initialised backbones
(`pretrained=False`) so the architecture, fusion and inference path can be
verified without downloading weights.  In production call
`ArogyaDrishti(pretrained=True)` to load the real medical models.

    python -m arogyadrishti.demo
"""
import numpy as np
from PIL import Image

from .system import ArogyaDrishti


def _fake_image(size=512):
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


def main():
    # pretrained=False -> no downloads; pure architecture check
    system = ArogyaDrishti(pretrained=False, allow_random_fallback=True,
                           device="cpu")

    sample_tabular = {
        "age": 34, "gender": "F", "bmi": 19.5,
        "systolic_bp": 118, "diastolic_bp": 76, "heart_rate": 88, "spo2": 97,
        "hemoglobin": 8.1, "glucose_fasting": 92, "glucose_pp": 120,
        "hba1c": 5.3, "alt": 26, "ast": 24, "bilirubin_total": 0.7,
        "creatinine": 0.8, "urea": 24, "ferritin": 14, "vitamin_b12": 280,
        "vitamin_d": 18, "albumin": 4.1, "fatigue_score": 2,
    }

    print("\n" + "=" * 64)
    print("CASE 1 — all modalities present")
    print("=" * 64)
    res = system.analyze_person(
        images={m: _fake_image() for m in
                ["eye", "face", "tongue", "skin", "nail", "palm"]},
        tabular=sample_tabular,
    )
    _print(res)

    print("\n" + "=" * 64)
    print("CASE 2 — only eye + palm + tabular (graceful degradation)")
    print("=" * 64)
    res = system.analyze_person(
        images={"eye": _fake_image(), "palm": _fake_image()},
        tabular=sample_tabular,
    )
    _print(res)

    print("\n" + "=" * 64)
    print("CASE 3 — tabular only (no images at all)")
    print("=" * 64)
    res = system.analyze_person(tabular=sample_tabular)
    _print(res)

    print("\n✅ All cases ran end-to-end.")


def _print(res):
    print("  risk scores :")
    for cat, score in res["risk_scores"].items():
        print(f"      {cat:16s} {score:.3f}  ({res['risk_bands'][cat]})")
    print("  used        :", res["used_modalities"])
    print("  missing     :", res["missing_modalities"])
    print(f"  confidence  : {res['overall_confidence']:.3f}")
    print("  needs review:", res["needs_manual_review"])


if __name__ == "__main__":
    main()
