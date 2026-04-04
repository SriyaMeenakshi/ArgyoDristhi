const admin = require('firebase-admin');

/**
 * Initialise Firebase Admin SDK.
 * Credentials come from environment variables (set in .env / Render dashboard).
 * Never commit serviceAccountKey.json to GitHub.
 */
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert({
      projectId    : process.env.FIREBASE_PROJECT_ID,
      privateKey   : process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
      clientEmail  : process.env.FIREBASE_CLIENT_EMAIL,
    })
  });
}

const db = admin.firestore();

/**
 * Firestore collections — guide Section 7.2
 *
 * patients   — one doc per patient
 * screenings — one doc per screening session
 * alerts     — one doc per alert sent
 * doctors    — PHC doctor contacts (looked up during alert dispatch)
 */
const collections = {
  patients  : db.collection('patients'),
  screenings: db.collection('screenings'),
  alerts    : db.collection('alerts'),
  doctors   : db.collection('doctors'),
};

module.exports = { db, collections };
