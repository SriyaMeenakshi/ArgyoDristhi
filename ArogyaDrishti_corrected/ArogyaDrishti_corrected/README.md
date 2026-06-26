# ArogyaDrishti — Corrected, Integrated System (v3.0)

A multimodal health-screening model that analyses a person from any subset of
six images (eye, face, tongue, skin, nail, palm) plus 21 structured health
features, and produces **7 risk scores** — hematological, metabolic, renal,
hepatic, cardiovascular, dermatological, nutritional — each in `[0, 1]`, with
per-modality confidence and graceful handling of missing inputs.

> Screening aid only. Not a diagnosis. Findings must be confirmed by clinical
> examination and laboratory tests.

---

## What was wrong with the original code

The uploaded files were **six isolated image feature-extractors** with several
bugs, and the pieces that actually let you *analyse a person* were missing.

1. **BiomedCLIP loaded with the wrong API (face & skin).**
   `CLIPVisionModel.from_pretrained("microsoft/BiomedCLIP-...")` cannot load
   BiomedCLIP — that repository ships an **OpenCLIP** checkpoint, not a
   HuggingFace-transformers CLIP config, so the call raises at construction.
   The advertised `512-dim` is also wrong for `CLIPVisionModel` (its pooled
   output is `768-dim`); only OpenCLIP's projected image features are `512-dim`.
   **Fix:** load via `open_clip.create_model_and_transforms('hf-hub:...')` and
   read the real output dimension.

2. **No way to analyse a person.** The architecture calls for a tabular
   encoder, a cross-modal attention fusion module and 7 risk heads — none of
   which existed. The six encoders also emitted *incompatible* per-modality
   logits (3/4/4/7/6/5 classes) that cannot be combined directly.
   **Fix:** added `TabularEncoder`, `CrossModalFusion` (8-head attention →
   FFN → 7 sigmoid risk heads), preprocessing, and an `analyze_person()`
   entry point.

3. **Hard-coded dimensions broke on fallback.** Each encoder assumed its
   primary backbone loaded; a fallback silently changed the embedding size and
   desynchronised everything downstream.
   **Fix:** every encoder exposes a real, read-from-the-backbone `out_dim`, and
   the fusion module is built from those actual dimensions.

4. **Wrong / missing preprocessing.** CLIP/BiomedCLIP need CLIP mean/std (not
   ImageNet's); InceptionV3 needs 299×299. Nothing enforced this.
   **Fix:** per-encoder transforms driven by each encoder's `.input_size` and
   `.norm`.

---

## Install

```bash
pip install -r requirements.txt
```

## Use

```python
from arogyadrishti import ArogyaDrishti

system = ArogyaDrishti(pretrained=True)   # downloads real medical weights

result = system.analyze_person(
    images={"eye": "eye.jpg", "palm": "palm.jpg"},   # any subset, or none
    tabular={
        "age": 34, "gender": "F", "bmi": 19.5,
        "systolic_bp": 118, "diastolic_bp": 76, "heart_rate": 88, "spo2": 97,
        "hemoglobin": 8.1, "glucose_fasting": 92, "glucose_pp": 120,
        "hba1c": 5.3, "alt": 26, "ast": 24, "bilirubin_total": 0.7,
        "creatinine": 0.8, "urea": 24, "ferritin": 14, "vitamin_b12": 280,
        "vitamin_d": 18, "albumin": 4.1, "fatigue_score": 2,
    },
)

print(result["risk_scores"])           # {'hematological': 0.81, ...}
print(result["overall_confidence"])
print(result["needs_manual_review"])
```

Run the offline self-test (random weights, no downloads):

```bash
python -m arogyadrishti.demo
```

---

## Important note on the scores

The fusion module and the diagnostic/risk heads are **defined but untrained**.
Out of the box (or with random weights offline) the risk numbers are not
meaningful — they hover around 0.5. To get clinically useful outputs you must
**train** the fusion + risk heads (and fine-tune the encoder heads) on labelled
multimodal data, as described in the architecture document (multi-task loss,
class-weighted, Adam lr=1e-4, cosine schedule, early stopping). The code here
gives you a correct, runnable architecture to train; it does not ship trained
medical weights.

## Training

Because the datasets are **separate per modality** (no single source has all
six images + tabular + the 7 risk labels for one person), training is **two
stages**:

### Stage 1 — train each encoder on its own dataset
Standard image classification on each modality's own database
(ImageFolder layout: one subfolder per class). Includes the augmentation,
class-weighted loss, Adam + cosine schedule, gradient clipping and early
stopping from the architecture document.

```bash
python -m arogyadrishti.training.train_image_encoder \
    --modality eye --data-dir /path/to/eyes_defy_anemia \
    --epochs 50 --batch-size 32 --out-dir checkpoints
# repeat for: face, tongue, skin, nail, palm
```

The split is done **before** augmentation (80/20, stratified) to prevent
leakage; only the train split is augmented. Best checkpoint (by macro-F1) is
saved to `checkpoints/<modality>_best.pt`.

### Stage 2 — train the fusion + 7 risk heads (multi-task)
Needs a **manifest CSV**, one row per person, listing whatever image paths and
tabular features exist plus the 7 risk labels (see the header of
`train_multimodal.py` for the exact columns). Missing modalities are handled
per person via the fusion mask. The loss is `BCEWithLogitsLoss` over the 7 risk
heads with `pos_weight` for class imbalance.

```bash
python -m arogyadrishti.training.train_multimodal \
    --manifest data/multimodal.csv \
    --ckpt-dir checkpoints \          # warm-start from Stage-1 weights
    --epochs 50 --batch-size 16 --out-dir checkpoints
# add --finetune-encoders to also update the encoders (default: frozen)
```

### Verify the pipeline without real data
```bash
python -m arogyadrishti.training.train_multimodal --smoke-test   # synthetic
python -m arogyadrishti.training.train_image_encoder \
    --modality palm --data-dir <tiny ImageFolder> --no-pretrained --smoke-test
```

## Files

```
arogyadrishti/
  encoders.py                  7 encoders, robust loading, stable out_dim
  fusion.py                    cross-modal attention + 7 risk heads (logits)
  preprocessing.py             per-encoder transforms + tabular handling
  system.py                    ArogyaDrishti orchestrator + analyze_person()
  demo.py                      inference self-test
  training/
    train_utils.py             augmentation, metrics, early stopping, etc.
    train_image_encoder.py     Stage 1: per-modality classification
    train_multimodal.py        Stage 2: fusion multi-task training
```
