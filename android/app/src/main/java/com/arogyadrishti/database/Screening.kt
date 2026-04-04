package com.arogyadrishti.database

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "screenings",
    foreignKeys = [ForeignKey(
        entity = Patient::class,
        parentColumns = ["id"],
        childColumns = ["patientId"],
        onDelete = ForeignKey.CASCADE
    )],
    indices = [Index("patientId")]
)
data class Screening(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val patientId: Long,
    val date: Long = System.currentTimeMillis(),

    // Reported symptoms (comma-separated list)
    val symptoms: String = "",

    // Individual encoder confidence scores (0–1)
    val faceScore: Float = 0f,
    val eyeScore: Float = 0f,
    val tongueScore: Float = 0f,
    val skinScore: Float = 0f,
    val nailScore: Float = 0f,
    val palmScore: Float = 0f,

    // 7 output risk scores from fusion model (0–1)
    val hematologicalRisk: Float = 0f,
    val metabolicRisk: Float = 0f,
    val renalRisk: Float = 0f,
    val hepaticRisk: Float = 0f,
    val cardiovascularRisk: Float = 0f,
    val dermatologicalRisk: Float = 0f,
    val nutritionalRisk: Float = 0f,

    // Aggregate risk: "green" / "yellow" / "red"
    val overallRiskLevel: String = "green",

    // Sync state
    val syncedToFirebase: Boolean = false,
    val syncedToAbdm: Boolean = false
)
