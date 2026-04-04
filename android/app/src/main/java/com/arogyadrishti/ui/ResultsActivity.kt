package com.arogyadrishti.ui

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import com.arogyadrishti.R
import com.arogyadrishti.databinding.ActivityResultsBinding
import com.arogyadrishti.model.InferenceResult
import com.arogyadrishti.model.RiskLevel
import com.arogyadrishti.model.RiskScore
import com.google.gson.Gson

/**
 * Screen 5 — AI Risk Assessment Results
 * 7 colour-coded risk cards. Red cards pulse. Send Alert button if any Red.
 */
class ResultsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultsBinding
    private lateinit var result: InferenceResult

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.title = "Step 4 of 4 — AI Risk Assessment"

        val json = intent.getStringExtra(EXTRA_RESULT_JSON) ?: run { finish(); return }
        result = Gson().fromJson(json, InferenceResult::class.java)

        val patientId   = intent.getLongExtra(PatientRegistrationActivity.EXTRA_PATIENT_ID, -1)
        val patientName = intent.getStringExtra(PatientRegistrationActivity.EXTRA_PATIENT_NAME) ?: ""
        val screeningId = intent.getLongExtra(EXTRA_SCREENING_ID, -1)

        if (patientName.isNotEmpty()) binding.tvPatientName.text = patientName

        renderRiskCards()
        renderSummaryBadge()

        // Show Send Alert button only if there are Red findings
        if (result.overallLevel == RiskLevel.RED) {
            binding.btnSendAlert.visibility = View.VISIBLE
            binding.btnSendAlert.setOnClickListener {
                startActivity(
                    Intent(this, AlertConfirmationActivity::class.java).apply {
                        putExtra(EXTRA_RESULT_JSON, json)
                        putExtra(PatientRegistrationActivity.EXTRA_PATIENT_ID, patientId)
                        putExtra(PatientRegistrationActivity.EXTRA_PATIENT_NAME, patientName)
                        putExtra(EXTRA_SCREENING_ID, screeningId)
                    }
                )
            }
        } else {
            binding.btnSendAlert.visibility = View.GONE
        }

        binding.btnScreenNext.setOnClickListener {
            // Go back to home, clearing this stack
            val intent = Intent(this, HomeActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            startActivity(intent)
        }
    }

    private fun renderRiskCards() {
        binding.llRiskCards.removeAllViews()
        result.allScores.forEach { score -> addRiskCard(score) }
    }

    private fun addRiskCard(score: RiskScore) {
        val card = LayoutInflater.from(this)
            .inflate(R.layout.item_risk_card, binding.llRiskCards, false)

        card.findViewById<TextView>(R.id.tvOrganism).text  = score.label
        card.findViewById<TextView>(R.id.tvPercent).text   = "${score.percent}%"
        card.findViewById<TextView>(R.id.tvStatus).text    = score.statusText

        val colorRes = when (score.level) {
            RiskLevel.RED    -> R.color.risk_red
            RiskLevel.YELLOW -> R.color.risk_yellow
            RiskLevel.GREEN  -> R.color.risk_green
        }
        card.findViewById<View>(R.id.viewIndicator)
            .setBackgroundResource(colorRes)

        // Pulsing animation for red cards
        if (score.level == RiskLevel.RED) {
            card.startAnimation(
                android.view.animation.AnimationUtils.loadAnimation(this, R.anim.pulse)
            )
        }

        binding.llRiskCards.addView(card)
    }

    private fun renderSummaryBadge() {
        val redCount = result.allScores.count { it.level == RiskLevel.RED }
        binding.tvSummaryBadge.text = when {
            redCount > 0 -> "$redCount urgent finding${if (redCount > 1) "s" else ""}"
            result.allScores.any { it.level == RiskLevel.YELLOW } -> "Monitor advised"
            else -> "All clear"
        }
        val badgeColor = when (result.overallLevel) {
            RiskLevel.RED    -> R.color.risk_red
            RiskLevel.YELLOW -> R.color.risk_yellow
            RiskLevel.GREEN  -> R.color.risk_green
        }
        binding.tvSummaryBadge.setBackgroundResource(badgeColor)
    }

    companion object {
        const val EXTRA_RESULT_JSON  = "extra_result_json"
        const val EXTRA_SCREENING_ID = "extra_screening_id"
    }
}
