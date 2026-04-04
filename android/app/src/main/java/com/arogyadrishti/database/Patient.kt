package com.arogyadrishti.database

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "patients")
data class Patient(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val name: String,
    val age: Int,
    val sex: String,
    val abhaId: String,
    val village: String,
    val isPregnant: Boolean,
    val ashaWorkerId: String,
    // UUID returned by backend after online registration — used by SyncWorker
    val backendId: String = "",
    val createdAt: Long = System.currentTimeMillis()
)
