package com.arogyadrishti.sync

import android.content.Context
import android.util.Log
import androidx.work.*
import com.arogyadrishti.database.AppDatabase
import com.arogyadrishti.network.ApiClient
import com.arogyadrishti.network.AbdmSyncRequest
import com.arogyadrishti.network.SaveScreeningRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit

/**
 * WorkManager background worker.
 * Guide Section 5.5:
 *   "When internet becomes available, WorkManager automatically syncs
 *    patient data and screening results to Firebase in the background."
 *
 * Syncs all unsynced screenings from Room → backend → marks synced.
 * Also triggers ABDM sync for each synced screening.
 *
 * Enqueue on app start and whenever network comes back online.
 */
class SyncWorker(context: Context, params: WorkerParameters) :
    CoroutineWorker(context, params) {

    companion object {
        private const val TAG         = "SyncWorker"
        private const val WORK_NAME   = "arogya_sync"

        /**
         * Schedule a one-time sync with network constraint.
         * Call this from HomeActivity and from NetworkMonitor callbacks.
         */
        fun enqueue(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()

            val request = OneTimeWorkRequestBuilder<SyncWorker>()
                .setConstraints(constraints)
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 15, TimeUnit.MINUTES)
                .build()

            WorkManager.getInstance(context)
                .enqueueUniqueWork(WORK_NAME, ExistingWorkPolicy.KEEP, request)

            Log.i(TAG, "Sync enqueued")
        }
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val db = AppDatabase.getInstance(applicationContext)
        val screeningDao = db.screeningDao()
        val patientDao   = db.patientDao()

        val unsynced = screeningDao.getUnsynced()
        if (unsynced.isEmpty()) {
            Log.i(TAG, "Nothing to sync.")
            return@withContext Result.success()
        }

        Log.i(TAG, "Syncing ${unsynced.size} screening(s)...")

        for (screening in unsynced) {
            try {
                val patient = patientDao.getById(screening.patientId)
                    ?: continue

                // ── 1. Save screening to backend ──────────────────────────────
                // Use backendId (Firebase UUID) if already registered online; fallback to local id
                val resolvedPatientId = patient.backendId.ifBlank { patient.id.toString() }

                val resp = ApiClient.api.saveScreening(
                    SaveScreeningRequest(
                        patientId          = resolvedPatientId,
                        symptoms           = screening.symptoms,
                        hematologicalRisk  = screening.hematologicalRisk,
                        metabolicRisk      = screening.metabolicRisk,
                        renalRisk          = screening.renalRisk,
                        hepaticRisk        = screening.hepaticRisk,
                        cardiovascularRisk = screening.cardiovascularRisk,
                        dermatologicalRisk = screening.dermatologicalRisk,
                        nutritionalRisk    = screening.nutritionalRisk,
                        overallRiskLevel   = screening.overallRiskLevel,
                        ashaWorkerId       = patient.ashaWorkerId,
                        village            = patient.village
                    )
                )

                if (resp.success) {
                    // Mark synced in local Room DB
                    screeningDao.markSynced(screening.id)
                    Log.i(TAG, "Screening ${screening.id} synced → backend ID ${resp.screeningId}")

                    // ── 2. Trigger ABDM sync if patient has ABHA ID ───────────
                    if (patient.abhaId.isNotBlank()) {
                        try {
                            ApiClient.api.syncToAbdm(AbdmSyncRequest(resp.screeningId))
                            Log.i(TAG, "ABDM sync triggered for ${resp.screeningId}")
                        } catch (e: Exception) {
                            // ABDM failure is non-fatal — screening is already saved
                            Log.w(TAG, "ABDM sync failed for ${resp.screeningId}: ${e.message}")
                        }
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Failed to sync screening ${screening.id}: ${e.message}")
                // Return retry so WorkManager tries again with backoff
                return@withContext Result.retry()
            }
        }

        Log.i(TAG, "Sync complete.")
        Result.success()
    }
}
