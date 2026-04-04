# ArogyaDrishti — Multi-Modal AI Health Screening

> **Arogya** (Telugu/Sanskrit: ఆరోగ్య) = *Health* · **Drishti** (దృష్టి) = *Vision*

A smartphone-based preventive health screening system designed for **ASHA workers in Andhra Pradesh**. One workflow — capture 5 photos + symptom checklist — produces an AI risk assessment across 7 organ systems, instantly, fully offline. Red-risk cases auto-alert the nearest PHC doctor via WhatsApp and sync the patient record to the national ABDM health stack.

---

## Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [AI Model (Part A)](#part-a--ai-model)
- [Android App (Part B)](#part-b--android-app)
- [Backend (Part D)](#part-d--backend)
- [Risk Thresholds](#risk-thresholds)
- [Setup Guide](#setup-guide)
- [Team](#team)

---

## Overview

| Dimension | Detail |
|-----------|--------|
| Target users | ASHA workers conducting village health camps |
| Target geography | Andhra Pradesh (multilingual alerts: Telugu / Hindi / Tamil / English) |
| Connectivity model | **Offline-first** — full screening works without internet; syncs when online |
| Inference | **On-device** — PyTorch Mobile, no data leaves the phone |
| Alert channel | WhatsApp (PHC doctor) + SMS (patient) via Twilio |
| Health records | ABDM FHIR R4 bundle sync using ABHA ID |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASHA Worker's Android Phone                  │
│                                                                 │
│  CameraX ──► 5 photos  ──► ImageQualityChecker (ML Kit)        │
│  Symptoms checklist    ──► tabular vector [1×21]               │
│                                  │                             │
│              ArogyaInferenceEngine (PyTorch Mobile)            │
│    face ──► MobileViT-S ──┐                                    │
│    eye  ──► CNN       ──┤                                      │
│    tongue ► ResNet-34  ──┤                                      │
│    skin ──► ViT        ──┼──► Cross-Modal Attention Fusion     │
│    nail ──► EfficientNet┤     (512-dim, 8-head)                │
│    palm ──► EfficientNet┤                                      │
│    tabular ► MLP       ──┘                                     │
│                                  │                             │
│              7 Risk Scores (0–1)                               │
│              Room DB (offline)                                 │
│                                  │                             │
│         WorkManager (CONNECTED constraint)                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │ REST API (Retrofit2)
                       ▼
         ┌─────────────────────────┐
         │   Node.js / Express     │   ◄── Render.com (free tier)
         │   Firebase Firestore    │
         │   Twilio WhatsApp + SMS │
         │   ABDM FHIR R4 sync     │
         │   48-h follow-up cron   │
         └─────────────────────────┘
```

---

## Part A — AI Model

### Architecture

The model is a **cross-modal attention fusion** of 7 independent encoders:

| Encoder | Backbone | Embedding dim | Detects |
|---------|----------|---------------|---------|
| Face | MobileViT-S | 768 | Pallor, jaundice, facial oedema |
| Eye | CNN | 768 | Conjunctival pallor, scleral icterus |
| Tongue | ResNet-34 | 1024 | Glossitis, tongue pallor, coating |
| Skin | ViT | 768 | Rashes, pigmentation, dryness |
| Nail | EfficientNet-B6 | 1792 | Koilonychia, clubbing, pale nails |
| Palm | EfficientNet-B5 | 1536 | Palmar pallor, creases |
| Tabular | MLP | 100 | Age, BMI, pregnancy, 10 symptoms |

**Fusion:** All 7 embeddings are projected to a common 512-dim space, then passed through `nn.MultiheadAttention` (8 heads). A mean-pool aggregator feeds 7 independent linear heads — one per organ system.

### Outputs (7 risk scores in [0, 1])

```
hematological   cardiovascular   renal
metabolic       dermatological   nutritional
hepatic
```

### Why PyTorch Mobile (not TFLite)

The transformer backbones use `aten::_native_multi_head_attention` which **cannot be exported via ONNX**. PyTorch Mobile handles all native PyTorch ops directly — no conversion step needed.

### Export

```bash
# From project root
python export/export_tflite.py
# Outputs: export/arogyadrishti_fusion.ptl
```

Model inputs at runtime:
```python
face, eye, tongue, skin, nail, palm  →  each (1, 3, 224, 224)  float32
tabular                              →  (1, 21)                 float32
```

> **Note:** `nail` and `palm` encoders both receive the **same Hand photo**. One capture, two encoder passes with different learned weights.

---

## Part B — Android App

**Language:** Kotlin · **Min SDK:** 26 (Android 8) · **Target SDK:** 34

### Screen flow

```
HomeActivity  ──►  PatientRegistrationActivity  ──►  PhotoCaptureActivity
                                                             │
                                                    ProcessingActivity
                                                             │
                                                    ResultsActivity
                                                             │ (RED only)
                                                   AlertConfirmationActivity
```

### Key features

| Feature | Implementation |
|---------|---------------|
| On-device inference | PyTorch Mobile `LiteModuleLoader` |
| Camera | CameraX with MINIMIZE_LATENCY capture mode |
| Photo quality gate | Laplacian variance (blur) + brightness check |
| Face detection | ML Kit Face Detection (offline) |
| Hand detection | ML Kit Pose Detection — wrist landmark check |
| ABHA barcode scan | ZXing `ScanContract` |
| Offline storage | Room DB — `patients` + `screenings` tables |
| Background sync | WorkManager `CoroutineWorker` with `CONNECTED` constraint |
| Network status | `ConnectivityManager.NetworkCallback` → Flow |
| Backend API | Retrofit2 + OkHttp (30s timeout) |

### Offline-first data flow

```
1. Patient registered  ──► Room DB  (always, immediately)
                       ──► Backend (if online, saves backendId)

2. Screening saved     ──► Room DB  (always)
                       ──► SyncWorker queues if offline

3. SyncWorker wakes    ──► POST /api/screening/save
   (network available)     POST /api/abdm/sync (if ABHA ID present)
                           marks syncedToFirebase = true
```

### Project structure

```
android/
├── app/src/main/
│   ├── AndroidManifest.xml
│   ├── assets/
│   │   └── arogyadrishti_fusion.ptl   ← place model here (not in repo)
│   └── java/com/arogyadrishti/
│       ├── ArogyaApp.kt               ← Application class, Firebase init
│       ├── database/                  ← Room: Patient, Screening, DAOs
│       ├── model/                     ← InferenceEngine, QualityChecker, RiskResult
│       ├── network/                   ← ApiService, ApiClient, request/response DTOs
│       ├── sync/                      ← SyncWorker (WorkManager)
│       ├── ui/                        ← 6 Activities
│       └── utils/                     ← NetworkMonitor, BitmapUtils
└── app/src/main/res/
    ├── layout/                        ← 6 activity layouts + item_risk_card
    ├── drawable/                      ← tile_normal/done, badges, dots, icons
    ├── anim/pulse.xml                 ← Red card pulsing animation
    └── values/                        ← colors, strings, themes
```

### Risk card colours

| Score | Colour | Label |
|-------|--------|-------|
| < 0.35 | Green `#2E7D32` | Low Risk |
| 0.35 – 0.70 | Amber `#E65100` | Monitor |
| ≥ 0.70 | Red `#D32F2F` | High Risk |

---

## Part D — Backend

**Runtime:** Node.js ≥ 20 · **Framework:** Express · **Deploy:** Render (free)

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/patient/register` | Create or find patient by ABHA ID |
| `GET` | `/api/patient/:abha_id` | Fetch patient + screening history |
| `POST` | `/api/screening/save` | Save screening with 7 risk scores |
| `GET` | `/api/screening/:id` | Fetch single screening |
| `POST` | `/api/alert/send` | Send WhatsApp (PHC doctor) + SMS (patient) |
| `POST` | `/api/abdm/sync` | Build FHIR R4 bundle → ABDM HIE |
| `GET` | `/api/health` | Health check |

### Alert logic

1. Looks up PHC doctor contact from Firestore `doctors` collection (indexed by village)
2. Builds a key-signal summary per organ system
3. Sends WhatsApp message to doctor via Twilio
4. Sends SMS to patient in their language (Telugu default for AP)
5. Stores alert document with `phcAcknowledged: false`

### Follow-up cron

`node-cron` runs hourly. Any alert with `phcAcknowledged = false` **and** `sentAt > 48 hours ago` triggers an automatic follow-up WhatsApp to the PHC doctor.

### ABDM FHIR R4 sync

Builds a FHIR R4 `Bundle` (type: `document`) containing a `DiagnosticReport` with 7 `Observation` resources — one per risk score — and POSTs it to the ABDM HIE using the patient's ABHA ID.

### Firestore collections

| Collection | Purpose |
|------------|---------|
| `patients` | Patient demographics + ABHA ID |
| `screenings` | Risk scores per session |
| `alerts` | Alert history + acknowledgement status |
| `doctors` | PHC doctor contacts keyed by village |

---

## Risk Thresholds

```
Score ∈ [0.00, 0.35)  →  GREEN   Low Risk     — routine monitoring
Score ∈ [0.35, 0.70)  →  YELLOW  Monitor      — follow up at next camp
Score ∈ [0.70, 1.00]  →  RED     High Risk    — immediate PHC referral + alert
```

Overall risk = **worst** individual organ score.

---

## Setup Guide

### Prerequisites

- Android Studio Hedgehog or later
- Node.js ≥ 20
- Python ≥ 3.9 + PyTorch 2.1 (for re-training / re-export only)
- Firebase project (Firestore enabled)
- Twilio account (WhatsApp sandbox or approved number)
- ABDM developer credentials (optional — remove `abdm.js` route if not available)

### 1 — Model file

The `.ptl` model file is **not stored in this repo** (1.5 GB). Place it manually:

```
android/app/src/main/assets/arogyadrishti_fusion.ptl
```

To regenerate from checkpoints:
```bash
pip install torch torchvision timm
python export/export_tflite.py
```

### 2 — Android app

```bash
cd android
# 1. Add app/google-services.json from Firebase console
# 2. Open in Android Studio
# 3. Plug in device (USB debugging on) or start emulator
# 4. Run ▶
```

Update the backend URL in `app/.../network/ApiService.kt`:
```kotlin
private const val BASE_URL = "https://your-backend.onrender.com/"
```

### 3 — Backend

```bash
cd backend
npm install
cp .env.example .env        # fill in all values
node seedDoctors.js          # one-time: seed PHC contacts into Firestore
npm start
```

**`.env` variables:**

```env
PORT=3000

# Firebase Admin SDK
FIREBASE_PROJECT_ID=
FIREBASE_CLIENT_EMAIL=
FIREBASE_PRIVATE_KEY=

# Twilio
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
TWILIO_SMS_FROM=

# ABDM
ABDM_BASE_URL=https://dev.abdm.gov.in
ABDM_CLIENT_ID=
ABDM_CLIENT_SECRET=
```

### 4 — Deploy backend to Render

```bash
# render.yaml is already configured
# Push to GitHub → connect repo in Render dashboard → Deploy
```

---

## Encoder files

```
encoders/
├── face_encoder.py       MobileViT-S  — pallor / facial oedema
├── eye_encoder.py        CNN          — conjunctival pallor / icterus
├── tongue_encoder.py     ResNet-34    — glossitis / coating
├── skin_encoder.py       ViT          — rashes / pigmentation
├── nail_encoder.py       EfficientNet-B6  — koilonychia / clubbing
├── palm_encoder.py       EfficientNet-B5  — palmar pallor
├── tabular_encoder.py    MLP          — age / BMI / pregnancy / symptoms
└── fusion_module.py      MultiheadAttention (8 heads, 512-dim)
```

Training scripts:
```
train/
├── train_encoder.py      Per-encoder supervised training
├── train_fusion.py       End-to-end fusion fine-tuning
├── train_tabular.py      Tabular encoder pre-training
├── dataset.py            Multi-modal dataset loader
└── verify_setup.py       Environment + GPU sanity check
```

---

## Team

| Part | Component | Member |
|------|-----------|--------|
| A | AI model training + PyTorch Mobile export | *(model team)* |
| B | Android application (Kotlin) | Koppala Pavani |
| C | *(reserved)* | — |
| D | Node.js backend + Firebase + Twilio + ABDM | Sriya |

---

## Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Android](https://img.shields.io/badge/Android-3DDC84?style=flat&logo=android&logoColor=white)
![Kotlin](https://img.shields.io/badge/Kotlin-0095D5?style=flat&logo=kotlin&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat&logo=firebase&logoColor=black)

| Layer | Technology |
|-------|-----------|
| On-device AI | PyTorch Mobile 2.1 (`LiteModuleLoader`) |
| Android | Kotlin, CameraX, Room, WorkManager, ML Kit, ZXing |
| Networking | Retrofit2 + OkHttp |
| Backend | Node.js + Express |
| Database | Firebase Firestore |
| Messaging | Twilio WhatsApp + SMS |
| Health records | ABDM FHIR R4 |
| Deployment | Render (backend), Google Play (future) |
