package com.arogyadrishti.ui

import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.arogyadrishti.database.AppDatabase
import com.arogyadrishti.database.Screening
import com.arogyadrishti.databinding.ActivityProcessingBinding
import com.arogyadrishti.model.ArogyaInferenceEngine
import com.arogyadrishti.model.InferenceResult
import com.arogyadrishti.model.RiskLevel
import com.arogyadrishti.network.ApiClient
import com.arogyadrishti.network.SendAlertRequest
import com.arogyadrishti.utils.NetworkMonitor
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Screen 4 — Processing
 * Runs on-device PyTorch inference, saves to Room, and if Red risk is found
 * and device is online, fires the alert API immediately.
 */
class ProcessingActivity : AppCompatActivity() {

    private lateinit var binding: ActivityProcessingBinding
    private val engine by lazy { ArogyaInferenceEngine(applicationContext) }

    private val steps = listOf(
        "Loading AI model…",
        "Analysing face…",
        "Analysing eye…",
        "Analysing tongue…",
        "Analysing skin…",
        "Analysing hand (nail + palm)…",
        "Cross-modal fusion…",
        "Generating risk scores…"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityProcessingBinding.inflate(layoutInflater)
        setContentView(binding.root)
        runInference()
    }

    private fun runInference() {
        lifecycleScope.launch {
            try {
                val animJob = launch {
                    steps.forEachIndexed { i, step ->
                        withContext(Dispatchers.Main) {
                            binding.tvStep.text = step
                            binding.progressBar.progress = ((i + 1) * 100 / steps.size)
                            updateEncoderBar(i)
                        }
                        delay(300)
                    }
                }

                // ── On-device inference ───────────────────────────────────────
                val result: InferenceResult = withContext(Dispatchers.IO) {
                    engine.load()
                    val face   = BitmapFactory.decodeFile(intent.getStringExtra(PhotoCaptureActivity.EXTRA_FACE_PATH))
                    val eye    = BitmapFactory.decodeFile(intent.getStringExtra(PhotoCaptureActivity.EXTRA_EYE_PATH))
                    val tongue = BitmapFactory.decodeFile(intent.getStringExtra(PhotoCaptureActivity.EXTRA_TONGUE_PATH))
                    val skin   = BitmapFactory.decodeFile(intent.getStringExtra(PhotoCaptureActivity.EXTRA_SKIN_PATH))
                    val hand   = BitmapFactory.decodeFile(intent.getStringExtra(PhotoCaptureActivity.EXTRA_HAND_PATH))

                    val age        = intent.getIntExtra(PatientRegistrationActivity.EXTRA_AGE, 30)
                    val isPregnant = intent.getBooleanExtra(PatientRegistrationActivity.EXTRA_IS_PREGNANT, false)
                    val symptoms   = intent.getStringArrayListExtra(PatientRegistrationActivity.EXTRA_SYMPTOMS)?.toSet() ?: emptySet()
                    val tabular    = engine.buildTabularVector(age, null, isPregnant, symptoms)

                    engine.infer(face, eye, tongue, skin, hand, tabular)
                }

                animJob.join()

                // ── Save to Room DB ───────────────────────────────────────────
                val patientId = intent.getLongExtra(PatientRegistrationActivity.EXTRA_PATIENT_ID, -1)
                val screeningId = withContext(Dispatchers.IO) {
                    AppDatabase.getInstance(applicationContext).screeningDao().insert(
                        Screening(
                            patientId          = patientId,
                            symptoms           = intent.getStringArrayListExtra(
                                PatientRegistrationActivity.EXTRA_SYMPTOMS)?.joinToString(",") ?: "",
                            hematologicalRisk  = result.hematological.score,
                            metabolicRisk      = result.metabolic.score,
                            renalRisk          = result.renal.score,
                            hepaticRisk        = result.hepatic.score,
                            cardiovascularRisk = result.cardiovascular.score,
                            dermatologicalRisk = result.dermatological.score,
                            nutritionalRisk    = result.nutritional.score,
                            overallRiskLevel   = result.overallLevelString
                        )
                    )
                }

                // ── Immediate alert if Red + online ───────────────────────────
                if (result.overallLevel == RiskLevel.RED &&
                    NetworkMonitor.isOnline(this@ProcessingActivity)) {
                    withContext(Dispatchers.IO) {
                        try {
                            ApiClient.api.sendAlert(
                                SendAlertRequest(
                                    patientId   = patientId.toString(),
                                    screeningId = screeningId.toString(),
                                    riskScores  = result.allScores.associate { it.label to it.score }
                                )
                            )
                        } catch (e: Exception) {
                            // Non-fatal — ResultsActivity has a manual Send Alert button as fallback
                            Log.w("Processing", "Auto-alert failed: ${e.message}")
                        }
                    }
                }

                // ── Navigate to results ───────────────────────────────────────
                withContext(Dispatchers.Main) {
                    startActivity(
                        Intent(this@ProcessingActivity, ResultsActivity::class.java).apply {
                            putExtra(ResultsActivity.EXTRA_RESULT_JSON, Gson().toJson(result))
                            putExtra(ResultsActivity.EXTRA_SCREENING_ID, screeningId)
                            putExtra(PatientRegistrationActivity.EXTRA_PATIENT_ID, patientId)
                            putExtra(PatientRegistrationActivity.EXTRA_PATIENT_NAME,
                                this@ProcessingActivity.intent.getStringExtra(PatientRegistrationActivity.EXTRA_PATIENT_NAME))
                        }
                    )
                    finish()
                }

            } catch (e: Exception) {
                Log.e("Processing", "Inference failed", e)
                withContext(Dispatchers.Main) {
                    binding.tvStep.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun updateEncoderBar(step: Int) {
        listOf(
            binding.pbFace, binding.pbEye, binding.pbTongue,
            binding.pbSkin, binding.pbNailPalm, binding.pbFusion
        ).getOrNull(step - 1)?.progress = 100
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.release()
    }
}
