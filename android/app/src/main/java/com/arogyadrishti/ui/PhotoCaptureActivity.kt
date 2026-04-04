package com.arogyadrishti.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.arogyadrishti.databinding.ActivityPhotoCaptureBinding
import com.arogyadrishti.model.ImageQualityChecker
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Screen 3 — Guided Photo Capture
 *
 * 5 modalities: Face, Eye, Tongue, Skin, Hand (nail+palm in one photo).
 * The Hand photo is fed to both the nail encoder and the palm encoder
 * inside ArogyaInferenceEngine.
 *
 * Guide reference — Section 5.2, Screen 3:
 *   "6 tiles in a grid: Face, Eye, Tongue, Skin, Hand (for nails+palm).
 *    Each tile turns green with a checkmark when captured.
 *    'Analyse' button unlocks only when all 5 are done."
 */
class PhotoCaptureActivity : AppCompatActivity() {

    private lateinit var binding: ActivityPhotoCaptureBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null

    private val captured = mutableMapOf<Modality, File>()
    private var currentModality: Modality = Modality.FACE

    enum class Modality(val label: String, val guideHint: String) {
        FACE  ("Face",   "Centre face in oval guide"),
        EYE   ("Eye",    "Pull lower lid slightly"),
        TONGUE("Tongue", "Extend tongue flat"),
        SKIN  ("Skin",   "Hold 10 cm from skin"),
        HAND  ("Hand",   "Show all nails, dorsal side up")
    }

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else { Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show(); finish() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPhotoCaptureBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.title = "Step 2 of 4 — Capture Photos"
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }

        setupModalityTiles()
        binding.btnCapture.setOnClickListener { takePhoto() }
        binding.btnAnalyse.isEnabled = false
        binding.btnAnalyse.alpha = 0.4f
        binding.btnAnalyse.setOnClickListener {
            if (allCaptured()) launchProcessing()
            else Toast.makeText(this, "Capture all 5 photos first", Toast.LENGTH_SHORT).show()
        }

        selectModality(Modality.FACE)
    }

    private fun setupModalityTiles() {
        mapOf(
            Modality.FACE   to binding.tileFace,
            Modality.EYE    to binding.tileEye,
            Modality.TONGUE to binding.tileTongue,
            Modality.SKIN   to binding.tileSkin,
            Modality.HAND   to binding.tileHand
        ).forEach { (modality, tile) ->
            tile.setOnClickListener { selectModality(modality) }
        }
    }

    private fun selectModality(modality: Modality) {
        currentModality = modality
        binding.tvGuideHint.text = modality.guideHint
        binding.tvCurrentModality.text = modality.label
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return
        val file = File(cacheDir, "${currentModality.name.lowercase()}.jpg")

        capture.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val bitmap = BitmapFactory.decodeFile(file.absolutePath)

                    // Image quality check (blur + brightness)
                    val quality = ImageQualityChecker.check(bitmap)
                    if (!quality.passed) {
                        Toast.makeText(this@PhotoCaptureActivity, quality.reason, Toast.LENGTH_LONG).show()
                        return
                    }

                    // ML Kit modality-specific check
                    when (currentModality) {
                        Modality.FACE -> ImageQualityChecker.checkFacePresent(bitmap) { faceOk, msg ->
                            if (!faceOk) {
                                Toast.makeText(this@PhotoCaptureActivity, msg, Toast.LENGTH_LONG).show()
                            } else {
                                acceptPhoto(file)
                            }
                        }
                        Modality.HAND -> ImageQualityChecker.checkHandPresent(bitmap) { handOk, msg ->
                            if (!handOk) {
                                Toast.makeText(this@PhotoCaptureActivity, msg, Toast.LENGTH_LONG).show()
                            } else {
                                acceptPhoto(file)
                            }
                        }
                        else -> acceptPhoto(file)
                    }
                }

                override fun onError(exc: ImageCaptureException) {
                    Toast.makeText(this@PhotoCaptureActivity,
                        "Capture failed: ${exc.message}", Toast.LENGTH_SHORT).show()
                }
            }
        )
    }

    private fun acceptPhoto(file: File) {
        captured[currentModality] = file
        markTileDone(currentModality)
        advanceToNextModality()
        if (allCaptured()) {
            binding.btnAnalyse.isEnabled = true
            binding.btnAnalyse.alpha = 1f
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture
                )
            } catch (e: Exception) {
                Toast.makeText(this, "Camera error: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun markTileDone(modality: Modality) {
        val tile = when (modality) {
            Modality.FACE   -> binding.tileFace
            Modality.EYE    -> binding.tileEye
            Modality.TONGUE -> binding.tileTongue
            Modality.SKIN   -> binding.tileSkin
            Modality.HAND   -> binding.tileHand
        }
        tile.setBackgroundResource(com.arogyadrishti.R.drawable.tile_done)
    }

    private fun advanceToNextModality() {
        Modality.values().firstOrNull { it !in captured }?.let { selectModality(it) }
    }

    private fun allCaptured() = Modality.values().all { it in captured }

    private fun launchProcessing() {
        val intent = Intent(this, ProcessingActivity::class.java).apply {
            putExtra(EXTRA_FACE_PATH,   captured[Modality.FACE]!!.absolutePath)
            putExtra(EXTRA_EYE_PATH,    captured[Modality.EYE]!!.absolutePath)
            putExtra(EXTRA_TONGUE_PATH, captured[Modality.TONGUE]!!.absolutePath)
            putExtra(EXTRA_SKIN_PATH,   captured[Modality.SKIN]!!.absolutePath)
            putExtra(EXTRA_HAND_PATH,   captured[Modality.HAND]!!.absolutePath)
            putExtra(PatientRegistrationActivity.EXTRA_PATIENT_ID,
                this@PhotoCaptureActivity.intent.getLongExtra(PatientRegistrationActivity.EXTRA_PATIENT_ID, -1))
            putExtra(PatientRegistrationActivity.EXTRA_PATIENT_NAME,
                this@PhotoCaptureActivity.intent.getStringExtra(PatientRegistrationActivity.EXTRA_PATIENT_NAME))
            putExtra(PatientRegistrationActivity.EXTRA_AGE,
                this@PhotoCaptureActivity.intent.getIntExtra(PatientRegistrationActivity.EXTRA_AGE, 0))
            putExtra(PatientRegistrationActivity.EXTRA_IS_PREGNANT,
                this@PhotoCaptureActivity.intent.getBooleanExtra(PatientRegistrationActivity.EXTRA_IS_PREGNANT, false))
            putStringArrayListExtra(PatientRegistrationActivity.EXTRA_SYMPTOMS,
                this@PhotoCaptureActivity.intent.getStringArrayListExtra(PatientRegistrationActivity.EXTRA_SYMPTOMS))
        }
        startActivity(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        const val EXTRA_FACE_PATH   = "extra_face_path"
        const val EXTRA_EYE_PATH    = "extra_eye_path"
        const val EXTRA_TONGUE_PATH = "extra_tongue_path"
        const val EXTRA_SKIN_PATH   = "extra_skin_path"
        const val EXTRA_HAND_PATH   = "extra_hand_path"   // nail + palm combined
    }
}
