package com.arogyadrishti.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.arogyadrishti.databinding.ActivityAlertConfirmationBinding
import com.arogyadrishti.model.InferenceResult
import com.arogyadrishti.model.RiskLevel
import com.arogyadrishti.network.ApiClient
import com.arogyadrishti.network.SendAlertRequest
import com.arogyadrishti.utils.NetworkMonitor
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Screen 6 — Alert Confirmation
 * Shows the WhatsApp message sent to the PHC doctor and SMS to the patient.
 * Displays ABDM sync status.
 */
class AlertConfirmationActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAlertConfirmationBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAlertConfirmationBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.title = "Alert Sent"

        val json      = intent.getStringExtra(ResultsActivity.EXTRA_RESULT_JSON) ?: run { finish(); return }
        val result    = Gson().fromJson(json, InferenceResult::class.java)
        val patientId = intent.getLongExtra(PatientRegistrationActivity.EXTRA_PATIENT_ID, -1)

        buildAlertMessages(result)
        sendAlerts(result, patientId)

        binding.btnScreenNext.setOnClickListener {
            val intent = Intent(this, HomeActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            startActivity(intent)
        }
    }

    private fun buildAlertMessages(result: InferenceResult) {
        val redScores = result.allScores.filter { it.level == RiskLevel.RED }

        val whatsappMsg = buildString {
            append("ArogyaDrishti Alert\n")
            append("Risk Detected: ")
            append(redScores.joinToString(", ") { "${it.label} (${it.percent}%)" })
            append("\nKey Signal: Elevated risk across ${redScores.size} organ system(s).")
            append("\nRecommended: Urgent clinical evaluation.")
            append("\nScreened by ASHA Worker at ${java.text.SimpleDateFormat("dd MMM yyyy HH:mm").format(java.util.Date())}")
        }

        val smsMsg = "ArogyaDrishti: Health risk detected. Please visit your nearest PHC. Show this message to the doctor."

        binding.tvWhatsappMessage.text = whatsappMsg
        binding.tvSmsMessage.text = smsMsg
    }

    private fun sendAlerts(result: InferenceResult, patientId: Long) {
        if (!NetworkMonitor.isOnline(this)) {
            binding.tvAbdmStatus.text = "Offline — will sync when connected"
            binding.ivAbdmIcon.setImageResource(android.R.drawable.ic_menu_info_details)
            return
        }

        lifecycleScope.launch {
            binding.progressAlerts.visibility = View.VISIBLE
            try {
                val screeningId = intent.getLongExtra(ResultsActivity.EXTRA_SCREENING_ID, -1)
                withContext(Dispatchers.IO) {
                    ApiClient.api.sendAlert(
                        SendAlertRequest(
                            patientId   = patientId.toString(),
                            screeningId = screeningId.toString(),
                            riskScores  = result.allScores.associate { it.label to it.score }
                        )
                    )
                }
                binding.tvWhatsappStatus.text = "WhatsApp sent to PHC doctor"
                binding.tvSmsStatus.text      = "SMS sent to patient"
                binding.tvAbdmStatus.text     = "Synced to ABDM"
            } catch (e: Exception) {
                binding.tvAbdmStatus.text = "Alert queued — will retry"
            } finally {
                binding.progressAlerts.visibility = View.GONE
            }
        }
    }
}
