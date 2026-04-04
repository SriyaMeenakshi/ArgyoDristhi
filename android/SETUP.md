# ArogyaDrishti Android App — Koppala Pavani Setup

## Step 1 — Open in Android Studio
1. Open Android Studio
2. File → Open → select this `android/` folder
3. Let Gradle sync complete (first time: ~5 minutes, downloads dependencies)

## Step 2 — Copy the Model File
The AI model is too large for Git. Copy it manually:

**Option A — Put in assets (development)**
```
Copy: E:\New folder\arogyadrishti\export\arogyadrishti_fusion.ptl
  To: android\app\src\main\assets\arogyadrishti_fusion.ptl
```
Note: The APK will be ~1.5 GB. Only use this for demo. For production, use Option B.

**Option B — Push to device storage (recommended for field deployment)**
```bash
adb push arogyadrishti_fusion.ptl /sdcard/Android/data/com.arogyadrishti/files/
```
Then update `assetFilePath()` in `ArogyaInferenceEngine.kt` to load from external storage.

## Step 3 — Connect Firebase
1. Go to firebase.google.com → your project → Project Settings → Add Android App
2. Package name: `com.arogyadrishti`
3. Download `google-services.json`
4. Place it at: `android/app/google-services.json`
5. Add to `android/app/build.gradle` plugins: `id 'com.google.gms.google-services'`
6. Add to `android/build.gradle` plugins: `id 'com.google.gms.google-services' version '4.4.0' apply false`

## Step 4 — Update Backend URL
In `AlertApiService.kt`, replace:
```kotlin
private const val BASE_URL = "https://arogyadrishti-backend.onrender.com/"
```
with the URL Sriya gives you after deploying the backend on Render.

## Step 5 — Run on Device
1. Enable Developer Options on your Android phone (tap Build Number 7 times)
2. Enable USB Debugging
3. Connect phone to laptop via USB
4. In Android Studio, select your device from the dropdown
5. Click Run (green triangle)

## App Flow
```
HomeActivity (Screen 1 — Dashboard)
    ↓ "New Screening"
PatientRegistrationActivity (Screen 2 — Register patient)
    ↓ "Next"
PhotoCaptureActivity (Screen 3 — Capture 6 photos)
    ↓ "Analyse"
ProcessingActivity (Screen 4 — On-device AI inference)
    ↓ auto-advances
ResultsActivity (Screen 5 — 7 risk cards)
    ↓ "Send Alert" (only if Red risk detected)
AlertConfirmationActivity (Screen 6 — Alert sent confirmation)
    ↓ "Screen Next Patient" → back to Screen 1
```

## Model Input/Output (from export_tflite.py)
```kotlin
// 7 inputs (batch=1)
faceTensor    [1, 3, 224, 224]   // ImageNet normalised (224×224 — not 256)
eyeTensor     [1, 3, 224, 224]
tongueTensor  [1, 3, 224, 224]
skinTensor    [1, 3, 224, 224]
nailTensor    [1, 3, 224, 224]   // same Hand photo
palmTensor    [1, 3, 224, 224]   // same Hand photo — one capture, two signals
tabularTensor [1, 21]            // age, BMI, pregnancy, 10 symptoms, 8 reserved

// 7 outputs (0.0 – 1.0 each)
output[0] = hematological
output[1] = metabolic
output[2] = renal
output[3] = hepatic
output[4] = cardiovascular
output[5] = dermatological
output[6] = nutritional

// Risk thresholds
< 0.35    → Green (Low Risk)
0.35–0.70 → Yellow (Monitor)
> 0.70    → Red (High Risk) → triggers WhatsApp + SMS alert
```

## Why PyTorch Mobile, not TFLite
The implementation guide references TFLite, but the model uses transformer
backbones (BEiT, ViT, Swin) whose attention ops cannot be exported via ONNX.
PyTorch Mobile handles all native PyTorch ops without conversion.
This is documented in export_tflite.py (which actually performs the .ptl export).
