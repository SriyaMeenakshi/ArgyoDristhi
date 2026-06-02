package com.arogyadrishti.utils

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class ModelDownloadManager(val context: Context) {
    private val client = OkHttpClient()
    private val modelsDir = File(context.filesDir, "models")
    
    private val models = mapOf(
        "face" to "https://github.com/Yasaswini-ch/ArogyaDrishti/releases/download/v1.0/face.ptl",
        "eye" to "https://github.com/Yasaswini-ch/ArogyaDrishti/releases/download/v1.0/eye.ptl",
        "tongue" to "https://github.com/Yasaswini-ch/ArogyaDrishti/releases/download/v1.0/tongue.ptl",
        "skin" to "https://github.com/Yasaswini-ch/ArogyaDrishti/releases/download/v1.0/skin.ptl",
        "nail" to "https://github.com/Yasaswini-ch/ArogyaDrishti/releases/download/v1.0/nail.ptl",
        "palm" to "https://github.com/Yasaswini-ch/ArogyaDrishti/releases/download/v1.0/palm.ptl",
    )
    
    fun modelsExist(): Boolean {
        return models.keys.all { 
            File(modelsDir, "$it.ptl").exists() && File(modelsDir, "$it.ptl").length() > 0
        }
    }
    
    fun downloadAllModels(onProgress: (Int, String) -> Unit) {
        if (!modelsDir.exists()) modelsDir.mkdirs()
        
        models.entries.forEachIndexed { index, (name, url) ->
            onProgress(index * 100 / models.size, "Downloading $name model...")
            downloadFile(url, File(modelsDir, "$name.ptl"))
        }
        onProgress(100, "All models downloaded")
    }
    
    private fun downloadFile(url: String, outputFile: File) {
        val request = Request.Builder().url(url).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw Exception("Failed to download file: $url")
            
            response.body?.byteStream()?.use { input ->
                FileOutputStream(outputFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
    }
    
    fun getModelPath(modality: String): String? {
        val file = File(modelsDir, "$modality.ptl")
        return if (file.exists()) file.absolutePath else null
    }
}
