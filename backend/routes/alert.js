const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { collections } = require('../config/firebase');
const twilioService = require('../services/twilioService');

const router = express.Router();

/**
 * POST /api/alert/send
 * Triggered by Android app when overallRiskLevel = 'red'.
 *
 * Flow (guide Section 7.3):
 *  1. Look up patient details from Firestore
 *  2. Look up nearest PHC doctor's WhatsApp from doctors collection
 *  3. Send WhatsApp to doctor via Twilio
 *  4. Send SMS to patient via Twilio (in regional language)
 *  5. Save alert record to Firestore
 *
 * Body: { patientId, screeningId, riskScores: { Hematological: 0.82, ... } }
 */
router.post('/send', async (req, res, next) => {
  try {
    const { patientId, screeningId, riskScores = {} } = req.body;

    if (!patientId || !screeningId) {
      return res.status(400).json({ success: false, error: 'patientId and screeningId required' });
    }

    // ── 1. Fetch patient ──────────────────────────────────────────────────────
    const patientDoc = await collections.patients.doc(patientId).get();
    if (!patientDoc.exists) {
      return res.status(404).json({ success: false, error: 'Patient not found' });
    }
    const patient = patientDoc.data();

    // ── 2. Find the highest-risk findings ────────────────────────────────────
    const redFindings = Object.entries(riskScores)
      .filter(([, score]) => score >= 0.70)
      .sort(([, a], [, b]) => b - a);

    const topFinding = redFindings[0];
    const riskSummary = redFindings
      .map(([label, score]) => `${label} (${Math.round(score * 100)}%)`)
      .join(', ');

    // ── 3. Look up PHC doctor for patient's village ───────────────────────────
    let doctorWhatsApp = process.env.DEFAULT_PHC_WHATSAPP;
    let doctorName     = process.env.DEFAULT_PHC_NAME || 'PHC Doctor';

    if (patient.village) {
      const doctorSnap = await collections.doctors
        .where('village', '==', patient.village)
        .limit(1)
        .get();
      if (!doctorSnap.empty) {
        const doctor = doctorSnap.docs[0].data();
        doctorWhatsApp = `whatsapp:${doctor.whatsappNumber}`;
        doctorName     = doctor.name;
      }
    }

    const now = new Date();
    const dateStr = now.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' });

    // ── 4a. Send WhatsApp to doctor ───────────────────────────────────────────
    const whatsappBody = [
      `ArogyaDrishti Alert — ${patient.name}, ${patient.age}Y, ${patient.sex}`,
      `ABHA ID: ${patient.abhaId || 'Not registered'}`,
      `Risk Detected: ${riskSummary}`,
      `Key Signal: ${buildKeySignal(redFindings)}`,
      `Recommended: ${buildRecommendation(redFindings)}`,
      `Screened by ASHA Worker ${patient.ashaWorkerId} at ${patient.village || 'Unknown'} on ${dateStr}`
    ].join('\n');

    const whatsappSid = await twilioService.sendWhatsApp(doctorWhatsApp, whatsappBody);

    // ── 4b. Send SMS to patient ───────────────────────────────────────────────
    let patientPhone = patient.phone || null;
    let smsSid = null;
    if (patientPhone) {
      const smsBody = buildPatientSms(patient, topFinding, doctorName);
      smsSid = await twilioService.sendSms(patientPhone, smsBody);
    }

    // ── 5. Save alert to Firestore ────────────────────────────────────────────
    const alertId = uuidv4();
    await collections.alerts.doc(alertId).set({
      alertId,
      screeningId,
      patientId,
      whatsappTo     : doctorWhatsApp,
      whatsappBody,
      whatsappSid,
      smsSid,
      sentAt         : now.toISOString(),
      phcAcknowledged: false,
      acknowledgedAt : null
    });

    res.json({
      success    : true,
      alertId,
      whatsappSent: !!whatsappSid,
      smsSent    : !!smsSid
    });

  } catch (err) {
    next(err);
  }
});

// ── Helper: key signal sentence from top red findings ────────────────────────
function buildKeySignal(redFindings) {
  const signals = {
    Hematological  : 'Pale conjunctiva + pale nail beds — severe anemia suspected',
    Metabolic      : 'Elevated metabolic risk — diabetes / thyroid disorder suspected',
    Renal          : 'Renal stress markers elevated — kidney function review needed',
    Hepatic        : 'Jaundice signals detected — liver function review needed',
    Cardiovascular : 'Cardiovascular stress markers elevated — ECG recommended',
    Dermatological : 'Skin lesion requiring urgent dermatology review',
    Nutritional    : 'Severe nutritional deficiency detected — dietary assessment needed'
  };
  if (!redFindings.length) return 'Multiple risk factors detected';
  return signals[redFindings[0][0]] || `${redFindings[0][0]} risk elevated`;
}

// ── Helper: recommended tests from top findings ───────────────────────────────
function buildRecommendation(redFindings) {
  const recs = {
    Hematological  : 'Urgent CBC + iron studies + peripheral smear',
    Metabolic      : 'Fasting glucose + HbA1c + TSH',
    Renal          : 'Serum creatinine + urine routine + eGFR',
    Hepatic        : 'LFT + bilirubin + hepatitis B/C screen',
    Cardiovascular : 'ECG + lipid profile + BP monitoring',
    Dermatological : 'Dermatology referral + skin biopsy if needed',
    Nutritional    : 'Micronutrient panel + dietary counselling'
  };
  const topRecs = redFindings
    .slice(0, 2)
    .map(([label]) => recs[label] || 'Clinical evaluation')
    .join('; ');
  return topRecs || 'Full clinical evaluation';
}

// ── Helper: patient SMS in regional language ──────────────────────────────────
function buildPatientSms(patient, topFinding, doctorName) {
  // Telugu (default for Andhra Pradesh)
  const teluguMsg = `ArogyaDrishti: మీ ఆరోగ్య పరీక్షలో ముఖ్యమైన సమస్య గుర్తించబడింది. దయచేసి వెంటనే సమీప PHC ని సందర్శించండి. డాక్టర్ ${doctorName} తో సంప్రదించండి.`;

  // Fallback English
  const englishMsg = `ArogyaDrishti: Health risk detected. Please visit your nearest PHC immediately. Ask for ${doctorName}. Show this message.`;

  // Use Telugu for Andhra Pradesh (AP), English fallback
  return patient.village?.toLowerCase().includes('andhra') ? teluguMsg : englishMsg;
}

module.exports = router;
