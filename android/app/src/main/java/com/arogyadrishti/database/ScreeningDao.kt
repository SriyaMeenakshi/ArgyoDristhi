package com.arogyadrishti.database

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface ScreeningDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(screening: Screening): Long

    @Update
    suspend fun update(screening: Screening)

    @Query("SELECT * FROM screenings WHERE patientId = :patientId ORDER BY date DESC")
    fun getByPatient(patientId: Long): Flow<List<Screening>>

    @Query("SELECT * FROM screenings WHERE id = :id")
    suspend fun getById(id: Long): Screening?

    @Query("SELECT * FROM screenings WHERE syncedToFirebase = 0")
    suspend fun getUnsynced(): List<Screening>

    @Query("SELECT COUNT(*) FROM screenings WHERE date(date/1000, 'unixepoch') = date('now')")
    fun getTodayCount(): Flow<Int>

    @Query("""
        SELECT COUNT(*) FROM screenings
        WHERE date(date/1000, 'unixepoch') = date('now')
        AND overallRiskLevel = 'red'
    """)
    fun getTodayRedAlerts(): Flow<Int>

    @Query("""
        SELECT COUNT(*) FROM screenings
        WHERE date(date/1000, 'unixepoch') = date('now')
        AND overallRiskLevel = 'yellow'
    """)
    fun getTodayYellowFlags(): Flow<Int>

    @Query("UPDATE screenings SET syncedToFirebase = 1 WHERE id = :id")
    suspend fun markSynced(id: Long)
}
