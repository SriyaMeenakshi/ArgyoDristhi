package com.arogyadrishti.ui

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.arogyadrishti.database.AppDatabase
import com.arogyadrishti.databinding.ActivityHomeBinding
import com.arogyadrishti.sync.SyncWorker
import com.arogyadrishti.utils.NetworkMonitor
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch

/**
 * Screen 1 — Home Dashboard
 * Shows today's stats, online/offline status, New Screening button.
 * Triggers background sync whenever internet is available.
 */
class HomeActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHomeBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHomeBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val db = AppDatabase.getInstance(this)
        val screeningDao = db.screeningDao()

        // Observe today's stats
        lifecycleScope.launch {
            combine(
                screeningDao.getTodayCount(),
                screeningDao.getTodayRedAlerts(),
                screeningDao.getTodayYellowFlags()
            ) { total, red, yellow -> Triple(total, red, yellow) }
                .collectLatest { (total, red, yellow) ->
                    binding.tvScreenedCount.text = total.toString()
                    binding.tvRedAlerts.text     = red.toString()
                    binding.tvYellowFlags.text   = yellow.toString()
                }
        }

        // Observe network — update badge + trigger sync when online
        lifecycleScope.launch {
            NetworkMonitor.observe(this@HomeActivity).collectLatest { online ->
                if (online) {
                    binding.tvNetworkStatus.text = "Online"
                    binding.ivNetworkDot.setBackgroundResource(com.arogyadrishti.R.drawable.dot_green)
                    // Enqueue background sync of any offline-collected data
                    SyncWorker.enqueue(this@HomeActivity)
                } else {
                    binding.tvNetworkStatus.text = "Offline"
                    binding.ivNetworkDot.setBackgroundResource(com.arogyadrishti.R.drawable.dot_grey)
                }
            }
        }

        binding.btnFullScreening.setOnClickListener {
            val intent = Intent(this, PatientRegistrationActivity::class.java)
            intent.putExtra("SCREENING_TYPE", "FULL")
            startActivity(intent)
        }

        binding.btnQuickScreening.setOnClickListener {
            val intent = Intent(this, ScreeningTypeSelectionActivity::class.java)
            startActivity(intent)
        }
    }
}
