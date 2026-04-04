package com.arogyadrishti.model

import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions

/**
 * Checks photo quality before accepting it for inference.
 *
 * Guide — Section 5.4:
 *   Blur check: Laplacian variance < 100 → too blurry
 *   Brightness check: mean < 50 → too dark, mean > 220 → overexposed
 *   Face detection: ML Kit Face Detection on face photo
 *   Hand detection: ML Kit Pose Detection on hand photo
 */
object ImageQualityChecker {

    data class QualityResult(val passed: Boolean, val reason: String = "")

    private const val BLUR_THRESHOLD   = 100.0
    private const val DARK_THRESHOLD   = 50
    private const val BRIGHT_THRESHOLD = 220

    // ── Basic blur + brightness check ────────────────────────────────────────

    fun check(bitmap: Bitmap): QualityResult {
        val small = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
        val brightness = averageBrightness(small)
        if (brightness < DARK_THRESHOLD)   return QualityResult(false, "Too dark — move to better lighting")
        if (brightness > BRIGHT_THRESHOLD) return QualityResult(false, "Too bright — avoid direct sunlight")
        if (laplacianVariance(small) < BLUR_THRESHOLD)
            return QualityResult(false, "Image too blurry — hold the phone steadier")
        return QualityResult(true)
    }

    // ── ML Kit face detection (Face modality) ────────────────────────────────

    /**
     * Uses ML Kit Face Detection to confirm at least one face is visible.
     * Works fully offline — no internet required.
     * Callback: (passed, errorMessage)
     */
    fun checkFacePresent(bitmap: Bitmap, callback: (Boolean, String) -> Unit) {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .build()
        val detector = FaceDetection.getClient(options)
        val image    = InputImage.fromBitmap(bitmap, 0)

        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) callback(false, "No face detected — centre your face in the guide oval")
                else callback(true, "")
            }
            .addOnFailureListener {
                // If detection fails, allow the photo anyway
                callback(true, "")
            }
    }

    // ── ML Kit Pose Detection (Hand modality) ────────────────────────────────

    /**
     * Uses ML Kit Pose Detection to confirm a hand/wrist is present in frame.
     * Detects wrist landmarks — if found, the hand photo is acceptable.
     * Works fully offline.
     * Callback: (passed, errorMessage)
     */
    fun checkHandPresent(bitmap: Bitmap, callback: (Boolean, String) -> Unit) {
        val options = AccuratePoseDetectorOptions.Builder()
            .setDetectorMode(AccuratePoseDetectorOptions.SINGLE_IMAGE_MODE)
            .build()
        val detector = PoseDetection.getClient(options)
        val image    = InputImage.fromBitmap(bitmap, 0)

        detector.process(image)
            .addOnSuccessListener { pose ->
                // Check if any wrist landmark has high confidence
                val leftWrist  = pose.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.LEFT_WRIST)
                val rightWrist = pose.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.RIGHT_WRIST)
                val found = (leftWrist != null  && leftWrist.inFrameLikelihood  > 0.5f) ||
                            (rightWrist != null && rightWrist.inFrameLikelihood > 0.5f)
                if (!found) callback(false, "Hand not detected — show all nails, dorsal side up")
                else callback(true, "")
            }
            .addOnFailureListener {
                // Allow on failure
                callback(true, "")
            }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    private fun averageBrightness(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        var sum = 0L
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8)  and 0xFF
            val b =  p         and 0xFF
            sum += (0.299 * r + 0.587 * g + 0.114 * b).toLong()
        }
        return sum.toDouble() / pixels.size
    }

    private fun laplacianVariance(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val gray = DoubleArray(w * h) { i ->
            val p = pixels[i]
            0.299 * ((p shr 16) and 0xFF) + 0.587 * ((p shr 8) and 0xFF) + 0.114 * (p and 0xFF)
        }
        val lap = mutableListOf<Double>()
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            lap.add(gray[(y-1)*w+x] + gray[(y+1)*w+x] + gray[y*w+(x-1)] + gray[y*w+(x+1)] - 4.0*gray[y*w+x])
        }
        val mean = lap.average()
        return lap.sumOf { (it - mean) * (it - mean) } / lap.size
    }
}
