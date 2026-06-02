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
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.arogyadrishti.utils.ModelDownloadManager
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class ArogyaInferenceEngine(private val context: Context) {

    private val downloadManager = ModelDownloadManager(context)
    private val modules = mutableMapOf<String, Module>()

    // ImageNet normalisation (matches training preprocessing)
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    companion object {
        private const val TAG          = "ArogyaEngine"
        private const val IMAGE_SIZE   = 224   
        private const val TABULAR_DIM  = 21    
    }

    /** Load models from internal storage. */
    fun load() {
        val modalities = listOf("face", "eye", "tongue", "skin", "nail", "palm")
        modalities.forEach { modality ->
            if (!modules.containsKey(modality)) {
                val path = downloadManager.getModelPath(modality)
                if (path != null) {
                    try {
                        modules[modality] = LiteModuleLoader.load(path)
                        Log.i(TAG, "Model loaded: $modality from $path")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to load model $modality: ${e.message}")
                    }
                }
            }
        }
    }

    fun isLoaded(): Boolean = modules.size == 6

    /**
     * Run inference using individual encoder models.
     * Each model is assumed to return risk scores.
     * We aggregate them for the final result.
     */
    fun infer(
        face: Bitmap,
        eye: Bitmap,
        tongue: Bitmap,
        skin: Bitmap,
        hand: Bitmap,
        tabular: FloatArray
    ): InferenceResult {
        val t0 = System.currentTimeMillis()

        val results = mutableListOf<FloatArray>()

        // Helper to run inference on a modality if loaded
        fun runModality(name: String, bitmap: Bitmap) {
            modules[name]?.let { mod ->
                val tensor = bitmapToTensor(bitmap)
                // Assuming output is a tensor of 7 risk scores
                // If it's (Embedding, RiskScores), we might need to handle the tuple.
                val output = mod.forward(IValue.from(tensor))
                val riskScores = if (output.isTuple) {
                    // Assuming risk scores are the second element if it's (Embedding, Risks)
                    output.toTuple()[1].toTensor().dataAsFloatArray
                } else {
                    output.toTensor().dataAsFloatArray
                }
                results.add(riskScores)
            }
        }

        runModality("face", face)
        runModality("eye", eye)
        runModality("tongue", tongue)
        runModality("skin", skin)
        
        // Hand photo is used for both nail and palm
        runModality("nail", hand)
        runModality("palm", hand)

        if (results.isEmpty()) throw IllegalStateException("No models loaded for inference")

        // Average the scores across modalities
        val finalScores = FloatArray(7) { 0f }
        for (i in 0 until 7) {
            var sum = 0f
            for (res in results) {
                if (i < res.size) sum += res[i]
            }
            finalScores[i] = (sum / results.size).coerceIn(0f, 1f)
        }

        val elapsed = System.currentTimeMillis() - t0

        val labels = listOf(
            "Hematological", "Metabolic", "Renal", "Hepatic",
            "Cardiovascular", "Dermatological", "Nutritional"
        )

        return InferenceResult(
            hematological  = RiskScore(labels[0], finalScores[0], finalScores[0].toRiskLevel()),
            metabolic      = RiskScore(labels[1], finalScores[1], finalScores[1].toRiskLevel()),
            renal          = RiskScore(labels[2], finalScores[2], finalScores[2].toRiskLevel()),
            hepatic        = RiskScore(labels[3], finalScores[3], finalScores[3].toRiskLevel()),
            cardiovascular = RiskScore(labels[4], finalScores[4], finalScores[4].toRiskLevel()),
            dermatological = RiskScore(labels[5], finalScores[5], finalScores[5].toRiskLevel()),
            nutritional    = RiskScore(labels[6], finalScores[6], finalScores[6].toRiskLevel()),
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
        modules.values.forEach { it.destroy() }
        modules.clear()
    }
}
