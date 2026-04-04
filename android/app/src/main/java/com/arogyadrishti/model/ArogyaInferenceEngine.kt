package com.arogyadrishti.model

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

/**
 * Wraps the PyTorch Mobile fusion model (arogyadrishti_fusion.ptl).
 *
 * Why PyTorch Mobile and not TFLite:
 *   The model uses transformer backbones (BEiT, ViT, Swin) whose attention ops
 *   cannot be exported via ONNX to TFLite. PyTorch Mobile handles all native
 *   PyTorch ops natively without conversion.
 *
 * Model takes 7 float32 tensors as input (batch=1):
 *   face    [1, 3, 224, 224]
 *   eye     [1, 3, 224, 224]
 *   tongue  [1, 3, 224, 224]
 *   skin    [1, 3, 224, 224]
 *   nail    [1, 3, 224, 224]   ← same Hand photo as palm
 *   palm    [1, 3, 224, 224]   ← same Hand photo as nail
 *   tabular [1, 21]
 *
 * Returns a tuple of 7 float32 scalars [1,1]:
 *   [0] hematological  [1] metabolic   [2] renal   [3] hepatic
 *   [4] cardiovascular [5] dermatological [6] nutritional
 *
 * Copy arogyadrishti_fusion.ptl → app/src/main/assets/
 */
class ArogyaInferenceEngine(private val context: Context) {

    private var module: Module? = null

    // ImageNet normalisation (matches training preprocessing)
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    companion object {
        private const val TAG          = "ArogyaEngine"
        private const val MODEL_FILE   = "arogyadrishti_fusion.ptl"
        private const val IMAGE_SIZE   = 224   // Must match export_tflite.py
        private const val TABULAR_DIM  = 21    // Must match export_tflite.py
    }

    /** Load model once on a background thread. */
    fun load() {
        if (module != null) return
        try {
            val path = assetFilePath(context, MODEL_FILE)
            module = LiteModuleLoader.load(path)
            Log.i(TAG, "Model loaded: $path")
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed: ${e.message}")
            throw RuntimeException("Model load failed: ${e.message}", e)
        }
    }

    fun isLoaded(): Boolean = module != null

    /**
     * Run inference.
     *
     * @param face     Face photo
     * @param eye      Eye photo (lower lid pulled)
     * @param tongue   Tongue photo
     * @param skin     Skin photo
     * @param hand     Hand photo — fed to BOTH nail encoder and palm encoder
     * @param tabular  21-element float vector (age, BMI, pregnancy, symptoms)
     */
    fun infer(
        face: Bitmap,
        eye: Bitmap,
        tongue: Bitmap,
        skin: Bitmap,
        hand: Bitmap,
        tabular: FloatArray
    ): InferenceResult {
        val mod = module ?: throw IllegalStateException("Call load() first")

        val t0 = System.currentTimeMillis()

        val faceTensor    = bitmapToTensor(face)
        val eyeTensor     = bitmapToTensor(eye)
        val tongueTensor  = bitmapToTensor(tongue)
        val skinTensor    = bitmapToTensor(skin)
        val handTensor    = bitmapToTensor(hand)   // used for both nail and palm
        val tabularTensor = buildTabularTensor(tabular)

        // Model forward: (face, eye, tongue, skin, nail, palm, tabular)
        // nail and palm both receive the same hand photo
        val output = mod.forward(
            IValue.from(faceTensor),
            IValue.from(eyeTensor),
            IValue.from(tongueTensor),
            IValue.from(skinTensor),
            IValue.from(handTensor),   // nail
            IValue.from(handTensor),   // palm (same tensor, different encoder weights)
            IValue.from(tabularTensor)
        ).toTuple()

        val elapsed = System.currentTimeMillis() - t0

        fun score(i: Int): Float = output[i].toTensor().dataAsFloatArray[0].coerceIn(0f, 1f)

        val labels = listOf(
            "Hematological", "Metabolic", "Renal", "Hepatic",
            "Cardiovascular", "Dermatological", "Nutritional"
        )

        return InferenceResult(
            hematological  = RiskScore(labels[0], score(0), score(0).toRiskLevel()),
            metabolic      = RiskScore(labels[1], score(1), score(1).toRiskLevel()),
            renal          = RiskScore(labels[2], score(2), score(2).toRiskLevel()),
            hepatic        = RiskScore(labels[3], score(3), score(3).toRiskLevel()),
            cardiovascular = RiskScore(labels[4], score(4), score(4).toRiskLevel()),
            dermatological = RiskScore(labels[5], score(5), score(5).toRiskLevel()),
            nutritional    = RiskScore(labels[6], score(6), score(6).toRiskLevel()),
            inferenceTimeMs = elapsed
        )
    }

    /**
     * Build 21-element tabular input vector.
     *
     * Slots:
     *   0        — normalised age (0–1, max=100)
     *   1        — normalised BMI (range 10–50, clamped 0–1)
     *   2        — pregnancy flag (0/1)
     *   3–12     — 10 common symptoms (0/1 each)
     *   13–20    — reserved (zeros for future lab values / vitals)
     */
    fun buildTabularVector(
        age: Int,
        bmi: Float?,
        isPregnant: Boolean,
        symptoms: Set<String>
    ): FloatArray {
        val vec = FloatArray(TABULAR_DIM) { 0f }

        vec[0] = age.toFloat() / 100f
        vec[1] = bmi?.let { ((it - 10f) / 40f).coerceIn(0f, 1f) } ?: 0f
        vec[2] = if (isPregnant) 1f else 0f

        val knownSymptoms = listOf(
            "Fatigue", "Breathlessness", "Pale skin", "Headache",
            "Poor appetite", "Swollen feet", "Yellowing eyes", "Skin rash",
            "Chest pain", "Dizziness"
        )
        knownSymptoms.forEachIndexed { i, s ->
            vec[3 + i] = if (symptoms.contains(s)) 1f else 0f
        }
        // Slots 13–20 remain 0 (future: Hb, SpO2, BP, glucose, etc.)
        return vec
    }

    private fun bitmapToTensor(bitmap: Bitmap): Tensor {
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        return TensorImageUtils.bitmapToFloat32Tensor(resized, mean, std)
    }

    private fun buildTabularTensor(data: FloatArray): Tensor {
        val padded = FloatArray(TABULAR_DIM) { if (it < data.size) data[it] else 0f }
        return Tensor.fromBlob(padded, longArrayOf(1, TABULAR_DIM.toLong()))
    }

    /** Copy asset to internal storage — PyTorch Mobile requires a file path. */
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buf = ByteArray(4 * 1024 * 1024)
                var n: Int
                while (inputStream.read(buf).also { n = it } != -1) {
                    outputStream.write(buf, 0, n)
                }
            }
        }
        return file.absolutePath
    }

    fun release() {
        module?.destroy()
        module = null
    }
}
