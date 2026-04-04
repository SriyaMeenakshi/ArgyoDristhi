const express = require('express');
const { collections } = require('../config/firebase');
const abdmService = require('../services/abdmService');

const router = express.Router();

/**
 * POST /api/abdm/sync
 * Converts a completed screening to FHIR R4 format and sends it to ABDM HIE.
 * Called automatically after each screening when the device is online.
 * Guide Section 7.4.
 *
 * Body: { screeningId }
 */
router.post('/sync', async (req, res, next) => {
  try {
    const { screeningId } = req.body;
    if (!screeningId) {
      return res.status(400).json({ success: false, error: 'screeningId required' });
    }

    // Fetch screening
    const screeningDoc = await collections.screenings.doc(screeningId).get();
    if (!screeningDoc.exists) {
      return res.status(404).json({ success: false, error: 'Screening not found' });
    }
    const screening = screeningDoc.data();

    // Fetch patient
    const patientDoc = await collections.patients.doc(screening.patientId).get();
    if (!patientDoc.exists) {
      return res.status(404).json({ success: false, error: 'Patient not found' });
    }
    const patient = patientDoc.data();

    if (!patient.abhaId) {
      return res.status(422).json({
        success: false,
        error  : 'Patient has no ABHA ID — cannot sync to ABDM'
      });
    }

    // Build FHIR R4 bundle and send to ABDM
    const fhirBundle = abdmService.buildFhirBundle(patient, screening);
    const abdmResponse = await abdmService.syncToAbdm(fhirBundle, patient.abhaId);

    // Mark synced in Firestore
    await collections.screenings.doc(screeningId).update({ syncedToAbdm: true });

    res.json({ success: true, abdmResponse });

  } catch (err) {
    next(err);
  }
});

module.exports = router;
