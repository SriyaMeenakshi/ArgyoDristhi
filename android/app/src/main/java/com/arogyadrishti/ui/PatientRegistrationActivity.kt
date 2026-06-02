package com.arogyadrishti.ui

import android.content.Intent
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.arogyadrishti.database.AppDatabase
import com.arogyadrishti.database.Patient
import com.arogyadrishti.databinding.ActivityPatientRegistrationBinding
import com.arogyadrishti.network.ApiClient
import com.arogyadrishti.network.RegisterPatientRequest
import com.arogyadrishti.utils.NetworkMonitor
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Screen 2 — Patient Registration
 * Saves to Room (offline-first). If online, also registers on backend and stores backendId.
 * Passes patientId, patientName, age, isPregnant, symptoms to PhotoCaptureActivity.
 */
class PatientRegistrationActivity : AppCompatActivity() {

    private lateinit var binding: ActivityPatientRegistrationBinding
    private val selectedSymptoms = mutableSetOf<String>()

    private val barcodeLauncher = registerForActivityResult(ScanContract()) { result ->
        if (result.contents != null) binding.etAbhaId.setText(result.contents)
    }

    private var screeningType: String? = null
    private var quickType: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPatientRegistrationBinding.inflate(layoutInflater)
        setContentView(binding.root)

        screeningType = intent.getStringExtra("SCREENING_TYPE")
        quickType = intent.getStringExtra("QUICK_TYPE")

        supportActionBar?.apply {
            title = if (screeningType == "QUICK") "Quick Screening - $quickType" else "Full Screening"
            subtitle = "Step 1 - Patient Registration"
            setDisplayHomeAsUpEnabled(true)
        }

        if (screeningType == "QUICK") {
            binding.llFullScreeningFields.visibility = android.view.View.GONE
        } else {
            setupSymptomChips()
        }

        binding.btnScanAbha.setOnClickListener {
            barcodeLauncher.launch(ScanOptions().apply {
                setPrompt("Scan ABHA QR code")
                setBeepEnabled(true)
                setOrientationLocked(true)
            })
        }

        binding.btnNext.setOnClickListener { validateAndProceed() }
    }

    private fun setupSymptomChips() {
        listOf(
            "Fatigue", "Breathlessness", "Pale skin", "Headache",
            "Poor appetite", "Swollen feet", "Yellowing eyes", "Skin rash",
            "Chest pain", "Dizziness"
        ).forEach { symptom ->
            val chip = com.google.android.material.chip.Chip(this).apply {
                text = symptom
                isCheckable = true
                setOnCheckedChangeListener { _, checked ->
                    if (checked) selectedSymptoms.add(symptom) else selectedSymptoms.remove(symptom)
                }
            }
            binding.chipGroupSymptoms.addView(chip)
        }
    }

    private fun validateAndProceed() {
        val name       = binding.etName.text.toString().trim()
        val ageStr     = binding.etAge.text.toString().trim()

        if (name.isEmpty()) { binding.etName.error = "Name is required"; return }
        val age = ageStr.toIntOrNull()
        if (age == null || age < 1 || age > 120) { binding.etAge.error = "Enter a valid age"; return }

        val sex: String
        val abhaId: String
        val isPregnant: Boolean

        if (screeningType == "QUICK") {
            sex = "Not Specified"
            abhaId = ""
            isPregnant = false
        } else {
            sex = when (binding.spinnerSex.selectedItemPosition) {
                0 -> "Female"; 1 -> "Male"; else -> "Other"
            }
            abhaId = binding.etAbhaId.text.toString().trim()
            isPregnant = binding.spinnerPregnant.selectedItem.toString() == "Yes"
        }

        binding.btnNext.isEnabled = false

        val patient = Patient(
            name         = name,
            age          = age,
            sex          = sex,
            abhaId       = abhaId,
            village      = "",
            isPregnant   = isPregnant,
            ashaWorkerId = "ASHA_WORKER_1"
        )

        lifecycleScope.launch {
            val db       = AppDatabase.getInstance(this@PatientRegistrationActivity)
            val localId  = withContext(Dispatchers.IO) { db.patientDao().insert(patient) }

            // If online — register on backend immediately and save the UUID it returns
            if (NetworkMonitor.isOnline(this@PatientRegistrationActivity)) {
                try {
                    val resp = withContext(Dispatchers.IO) {
                        ApiClient.api.registerPatient(
                            RegisterPatientRequest(
                                name         = name,
                                age          = age,
                                sex          = sex,
                                abhaId       = abhaId,
                                village      = "",
                                isPregnant   = isPregnant,
                                ashaWorkerId = "ASHA_WORKER_1"
                            )
                        )
                    }
                    if (resp.success) {
                        // Store the backend UUID so SyncWorker can link screenings correctly
                        withContext(Dispatchers.IO) {
                            db.patientDao().updateBackendId(localId, resp.patientId)
                        }
                    }
                } catch (_: Exception) {
                    // Non-fatal — SyncWorker retries later
                }
            }

            startActivity(
                Intent(this@PatientRegistrationActivity, PhotoCaptureActivity::class.java).apply {
                    putExtra(EXTRA_PATIENT_ID,   localId)
                    putExtra(EXTRA_PATIENT_NAME, name)
                    putExtra(EXTRA_AGE,          age)
                    putExtra(EXTRA_IS_PREGNANT,  isPregnant)
                    putStringArrayListExtra(EXTRA_SYMPTOMS, ArrayList(selectedSymptoms))
                    putExtra("SCREENING_TYPE", screeningType)
                    putExtra("QUICK_TYPE", quickType)
                }
            )
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressedDispatcher.onBackPressed(); return true
    }

    companion object {
        const val EXTRA_PATIENT_ID   = "extra_patient_id"
        const val EXTRA_PATIENT_NAME = "extra_patient_name"   // ← new
        const val EXTRA_AGE          = "extra_age"
        const val EXTRA_IS_PREGNANT  = "extra_is_pregnant"
        const val EXTRA_SYMPTOMS     = "extra_symptoms"
    }
}
