package com.arogyadrishti.network

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.util.concurrent.TimeUnit

// ── Request / Response data classes ──────────────────────────────────────────

data class RegisterPatientRequest(
    val name        : String,
    val age         : Int,
    val sex         : String,
    val abhaId      : String,
    val village     : String,
    val isPregnant  : Boolean,
    val ashaWorkerId: String
)

data class RegisterPatientResponse(
    val success  : Boolean,
    val patientId: String,
    val existing : Boolean = false
)

data class SaveScreeningRequest(
    val patientId          : String,
    val symptoms           : String,
    val hematologicalRisk  : Float,
    val metabolicRisk      : Float,
    val renalRisk          : Float,
    val hepaticRisk        : Float,
    val cardiovascularRisk : Float,
    val dermatologicalRisk : Float,
    val nutritionalRisk    : Float,
    val overallRiskLevel   : String,
    val ashaWorkerId       : String,
    val village            : String
)

data class SaveScreeningResponse(
    val success    : Boolean,
    val screeningId: String
)

data class SendAlertRequest(
    val patientId  : String,
    val screeningId: String,
    val riskScores : Map<String, Float>
)

data class SendAlertResponse(
    val success     : Boolean,
    val alertId     : String,
    val whatsappSent: Boolean,
    val smsSent     : Boolean
)

data class AbdmSyncRequest(val screeningId: String)
data class AbdmSyncResponse(val success: Boolean)
data class HealthResponse(val status: String)

// ── Retrofit interface ────────────────────────────────────────────────────────

interface ArogyaApi {

    // Patient
    @POST("api/patient/register")
    suspend fun registerPatient(@Body request: RegisterPatientRequest): RegisterPatientResponse

    @GET("api/patient/{abha_id}")
    suspend fun getPatient(@Path("abha_id") abhaId: String): retrofit2.Response<Any>

    // Screening
    @POST("api/screening/save")
    suspend fun saveScreening(@Body request: SaveScreeningRequest): SaveScreeningResponse

    @GET("api/screening/{id}")
    suspend fun getScreening(@Path("id") screeningId: String): retrofit2.Response<Any>

    // Alert
    @POST("api/alert/send")
    suspend fun sendAlert(@Body request: SendAlertRequest): SendAlertResponse

    // ABDM
    @POST("api/abdm/sync")
    suspend fun syncToAbdm(@Body request: AbdmSyncRequest): AbdmSyncResponse

    // Health check
    @GET("api/health")
    suspend fun health(): HealthResponse
}

// ── Singleton client ──────────────────────────────────────────────────────────

object ApiClient {

    // Replace with Render URL after Sriya deploys
    private const val BASE_URL = "https://arogyadrishti-backend.onrender.com/"

    val api: ArogyaApi by lazy {
        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        }
        val client = OkHttpClient.Builder()
            .addInterceptor(logging)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build()

        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ArogyaApi::class.java)
    }
}
