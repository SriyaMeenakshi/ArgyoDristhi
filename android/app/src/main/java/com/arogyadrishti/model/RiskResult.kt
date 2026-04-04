package com.arogyadrishti.model

enum class RiskLevel { GREEN, YELLOW, RED }

data class RiskScore(
    val label: String,
    val score: Float,
    val level: RiskLevel
) {
    val percent: Int get() = (score * 100).toInt()
    val statusText: String get() = when (level) {
        RiskLevel.GREEN  -> "Low Risk"
        RiskLevel.YELLOW -> "Monitor"
        RiskLevel.RED    -> "High Risk"
    }
}

data class InferenceResult(
    val hematological: RiskScore,
    val metabolic: RiskScore,
    val renal: RiskScore,
    val hepatic: RiskScore,
    val cardiovascular: RiskScore,
    val dermatological: RiskScore,
    val nutritional: RiskScore,
    val inferenceTimeMs: Long
) {
    val allScores: List<RiskScore> get() = listOf(
        hematological, metabolic, renal, hepatic,
        cardiovascular, dermatological, nutritional
    )

    val overallLevel: RiskLevel get() = when {
        allScores.any { it.level == RiskLevel.RED }    -> RiskLevel.RED
        allScores.any { it.level == RiskLevel.YELLOW } -> RiskLevel.YELLOW
        else                                            -> RiskLevel.GREEN
    }

    val overallLevelString: String get() = overallLevel.name.lowercase()
}

fun Float.toRiskLevel(): RiskLevel = when {
    this >= 0.70f -> RiskLevel.RED
    this >= 0.35f -> RiskLevel.YELLOW
    else          -> RiskLevel.GREEN
}
