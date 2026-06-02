package com.arogyadrishti.ui

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.arogyadrishti.databinding.ActivityScreeningTypeSelectionBinding

class ScreeningTypeSelectionActivity : AppCompatActivity() {

    private lateinit var binding: ActivityScreeningTypeSelectionBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityScreeningTypeSelectionBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.apply {
            title = "Quick Screening"
            setDisplayHomeAsUpEnabled(true)
        }

        binding.cardPalm.setOnClickListener { proceedWith("Palm") }
        binding.cardEye.setOnClickListener { proceedWith("Eye") }
        binding.cardNail.setOnClickListener { proceedWith("Nail") }
        binding.cardSkin.setOnClickListener { proceedWith("Skin") }
        binding.cardTongue.setOnClickListener { proceedWith("Tongue") }
        binding.cardFace.setOnClickListener { proceedWith("Face") }
    }

    private fun proceedWith(type: String) {
        val intent = Intent(this, PatientRegistrationActivity::class.java)
        intent.putExtra("SCREENING_TYPE", "QUICK")
        intent.putExtra("QUICK_TYPE", type)
        startActivity(intent)
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressedDispatcher.onBackPressed()
        return true
    }
}
