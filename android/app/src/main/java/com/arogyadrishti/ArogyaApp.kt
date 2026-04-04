package com.arogyadrishti

import android.app.Application
import com.google.firebase.FirebaseApp

class ArogyaApp : Application() {

    override fun onCreate() {
        super.onCreate()
        FirebaseApp.initializeApp(this)
    }
}
