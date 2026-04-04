package com.arogyadrishti.database

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface PatientDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(patient: Patient): Long

    @Query("SELECT * FROM patients ORDER BY createdAt DESC")
    fun getAllPatients(): Flow<List<Patient>>

    @Query("SELECT * FROM patients WHERE id = :id")
    suspend fun getById(id: Long): Patient?

    @Query("SELECT * FROM patients WHERE abhaId = :abhaId LIMIT 1")
    suspend fun getByAbhaId(abhaId: String): Patient?

    @Query("SELECT COUNT(*) FROM patients WHERE date(createdAt/1000, 'unixepoch') = date('now')")
    fun getTodayScreeningCount(): Flow<Int>

    /** Store the UUID returned by the backend after online registration. */
    @Query("UPDATE patients SET backendId = :backendId WHERE id = :localId")
    suspend fun updateBackendId(localId: Long, backendId: String)
}
