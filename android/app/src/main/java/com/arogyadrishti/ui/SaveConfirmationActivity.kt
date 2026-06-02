package com.arogyadrishti.ui

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.arogyadrishti.databinding.ActivitySaveConfirmationBinding
import java.text.SimpleDateFormat
import java.util.*

class SaveConfirmationActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySaveConfirmationBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySaveConfirmationBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val name = intent.getStringExtra(PatientRegistrationActivity.EXTRA_PATIENT_NAME) ?: "Patient"
        val type = intent.getStringExtra("SCREENING_TYPE") ?: "FULL"
        val date = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault()).format(Date())

        binding.tvSavedPatientName.text = "Patient: $name"
        binding.tvSavedType.text = "Type: ${type.capitalize()} Screening"
        binding.tvSavedDate.text = "Date: $date"

        binding.btnStartNew.setOnClickListener {
            val intent = Intent(this, HomeActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            startActivity(intent)
            finish()
        }
    }
}
